from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from crypto_scalper.data import interval_to_milliseconds
from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_execution_backtest import _mtf_closed, _mtf_required_timeframes, _resample_to_timeframe
from crypto_scalper.live_portfolio_backtest import HistoricalClient, _load_symbol_data
from crypto_scalper.live_trader import BinanceAutoTrader, EntryCandidate
from crypto_scalper.models import Direction
from crypto_scalper.mtf_4h_rsi_regime import (
    Mtf4hRsiRegimePullbackStrategy,
    funding_at,
    load_auxiliary_features,
    mtf_min_rank_score_for_direction,
    oi_change_at,
)
from crypto_scalper.mtf_candidate_quality import (
    candidate_metrics,
    config_sha256,
    grouped_report,
    quantile_bucket_report,
)
from crypto_scalper.mtf_momentum_reset import (
    MTF_MOMENTUM_RESET_VERSION,
    STAGE16_EXPERIMENTS,
    MtfMomentumResetConfig,
    MtfSetupExperiment,
    build_release_signal,
    can_be_contraction_release,
    detect_momentum_reset_release,
    mtf_momentum_reset_event_id,
)
from crypto_scalper.mtf_shadow_trigger import classify_structure_overlap
from crypto_scalper.realistic_data import load_funding_rate_directory
from crypto_scalper.risk import execution_config_from_live_config
from scripts.mtf_candidate_quality_research import _shadow_trade, _single_symbol_frames, _timestamp, _write_csv


BUCKET_FIELDS = (
    "rank_score",
    "reset_component_count",
    "rsi_recovery",
    "directional_ema_slope_atr",
    "directional_ema_distance_atr",
    "directional_macd_hist_change_atr",
    "contraction_range_atr",
    "contraction_tr_ratio",
    "contraction_volume_ratio",
    "release_body_atr",
    "release_directional_close_position",
    "release_volume_ratio",
    "breakout_distance_atr",
    "release_extension_atr",
    "fill_stop_pct",
    "target_to_cost_ratio",
)


def _structure_times(path: str | None) -> dict[tuple[str, str], list[datetime]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output: dict[tuple[str, str], list[datetime]] = {}
    for row in payload.get("rows", []):
        symbol = str(row.get("symbol", ""))
        side = str(row.get("side", "")).upper()
        timestamp = row.get("trigger_time")
        if not symbol or not side or not timestamp:
            continue
        output.setdefault((symbol, side), []).append(_timestamp(timestamp))
    for values in output.values():
        values.sort()
    return output


def _dedupe_event_clusters(rows: Iterable[dict[str, Any]], hours: int = 4) -> list[dict[str, Any]]:
    minimum_gap = timedelta(hours=max(0, int(hours)))
    selected: list[dict[str, Any]] = []
    last_selected: dict[tuple[str, str], datetime] = {}
    for row in sorted(rows, key=lambda item: _timestamp(item["release_time"])):
        key = (str(row["symbol"]), str(row["side"]))
        timestamp = _timestamp(row["release_time"])
        previous = last_selected.get(key)
        if previous is not None and timestamp - previous < minimum_gap:
            continue
        selected.append(row)
        last_selected[key] = timestamp
    return selected


def _experiment_rows(
    rows: list[dict[str, Any]],
    experiment: MtfSetupExperiment,
    independent_only: bool,
    cluster_hours: int,
) -> list[dict[str, Any]]:
    accepted = [
        row
        for row in rows
        if not row.get("shadow_reject_reason")
        and bool(row.get("passes_frozen_rank_floor"))
        and (not independent_only or bool(row.get("independent_from_structure_window")))
        and experiment.accepts(row)
    ]
    return _dedupe_event_clusters(accepted, cluster_hours)


def _concentration_stress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: float(item.get("shadow_net_r", 0.0)), reverse=True)
    result = {"all": candidate_metrics(ordered)}
    for count in (1, 3, 5):
        result[f"exclude_top_{count}"] = candidate_metrics(ordered[count:])
    if rows:
        month_totals: dict[str, float] = {}
        symbol_totals: dict[str, float] = {}
        for row in rows:
            value = float(row.get("shadow_net_r", 0.0))
            month_totals[str(row.get("month"))] = month_totals.get(str(row.get("month")), 0.0) + value
            symbol_totals[str(row.get("symbol"))] = symbol_totals.get(str(row.get("symbol")), 0.0) + value
        best_month = max(month_totals, key=month_totals.get)
        best_symbol = max(symbol_totals, key=symbol_totals.get)
        result["exclude_best_month"] = {
            "excluded": best_month,
            **candidate_metrics(row for row in rows if row.get("month") != best_month),
        }
        result["exclude_best_symbol"] = {
            "excluded": best_symbol,
            **candidate_metrics(row for row in rows if row.get("symbol") != best_symbol),
        }
    return result


def _experiment_report(
    rows: list[dict[str, Any]],
    experiment: MtfSetupExperiment,
    cluster_hours: int,
) -> dict[str, Any]:
    all_rows = _experiment_rows(rows, experiment, False, cluster_hours)
    independent = _experiment_rows(rows, experiment, True, cluster_hours)
    return {
        "parameters": experiment.__dict__,
        "rank_eligible_clustered": candidate_metrics(all_rows),
        "independent_clustered": candidate_metrics(independent),
        "by_side_independent": grouped_report(independent, "side"),
        "by_month_independent": grouped_report(independent, "month"),
        "concentration_independent": _concentration_stress(independent),
        "event_ids_independent": [row["event_id"] for row in independent],
    }


def research_momentum_reset_setup(
    config_path: str,
    data_dir: str,
    start: datetime,
    end: datetime,
    structure_candidate_report: str | None = None,
    detector_config: MtfMomentumResetConfig | None = None,
    experiments: tuple[MtfSetupExperiment, ...] = STAGE16_EXPERIMENTS,
    structure_window_minutes: int = 120,
    cluster_hours: int = 4,
    embargo_hours: int = 12,
    regime_mode: str = "frozen_rsi_reversal",
    max_symbols: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    config = load_live_config(config_path)
    detector_config = detector_config or MtfMomentumResetConfig()
    symbols = tuple(config.trading.symbols[:max_symbols]) if max_symbols else tuple(config.trading.symbols)
    known_structure_times = _structure_times(structure_candidate_report)
    effective_end = end - timedelta(hours=max(0, embargo_hours))

    btc_loaded = _load_symbol_data(data_dir, ("BTCUSDT",), "1m")
    if not btc_loaded:
        raise RuntimeError(f"BTCUSDT 1m data missing from {data_dir}")
    btc_1m = btc_loaded["BTCUSDT"]
    timeframes = _mtf_required_timeframes(config)
    if "30m" not in timeframes or "1h" not in timeframes or "4h" not in timeframes:
        raise RuntimeError(f"MTF frames missing required setup timeframes: {timeframes}")
    btc_by_timeframe = {
        timeframe: _resample_to_timeframe(btc_1m, "1m", timeframe)
        for timeframe in timeframes
    }
    auxiliary = load_auxiliary_features(
        symbols,
        str(getattr(config.strategy, "mtf_oi_data_dir", "data/binance_oi_taker_5m")),
        str(getattr(config.strategy, "mtf_funding_data_dir", "data/binance_30m_365d")),
    )
    execution_config = execution_config_from_live_config(config, cost_experiment="full_cost", mode="conservative")
    if execution_config.funding_enabled and getattr(config.risk, "funding_data_dir", ""):
        execution_config = replace(
            execution_config,
            funding_rates_by_symbol=load_funding_rate_directory(config.risk.funding_data_dir, symbols),
        )

    rows: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for symbol_index, symbol in enumerate(symbols, start=1):
        loaded = btc_loaded if symbol == "BTCUSDT" else _load_symbol_data(data_dir, (symbol,), "1m")
        if symbol not in loaded:
            rejection_counts["missing_symbol_data"] = rejection_counts.get("missing_symbol_data", 0) + 1
            continue
        execution_candles = loaded[symbol]
        mtf_frames, mtf_timestamps = _single_symbol_frames(symbol, execution_candles, btc_by_timeframe, timeframes)
        signal_candles = _resample_to_timeframe(execution_candles, "1m", config.trading.timeframe)
        client = HistoricalClient({symbol: signal_candles}, config.trading.timeframe, ())
        trader = BinanceAutoTrader(config, client)
        if regime_mode == "trend_pullback":
            regime_config = replace(
                config,
                strategy=replace(config.strategy, mtf_regime_mode="trend_pullback"),
            )
        elif regime_mode == "frozen_rsi_reversal":
            regime_config = config
        else:
            raise ValueError(f"unsupported regime mode: {regime_mode}")
        strategy = Mtf4hRsiRegimePullbackStrategy(regime_config)
        release_rows = mtf_frames["30m"][symbol]

        for release_index, release_candle in enumerate(release_rows):
            available_time = release_candle.timestamp + timedelta(milliseconds=interval_to_milliseconds("30m"))
            if available_time < start or available_time >= effective_end:
                continue
            if not can_be_contraction_release(
                release_rows,
                release_index,
                detector_config.contraction_lookback_30m,
            ):
                continue
            try:
                candles_30m = release_rows[max(0, release_index - 179):release_index + 1]
                candles_1h = _mtf_closed(mtf_frames, mtf_timestamps, "1h", symbol, available_time, 180)
                candles_4h = _mtf_closed(mtf_frames, mtf_timestamps, "4h", symbol, available_time, 180)
                btc_1h = _mtf_closed(mtf_frames, mtf_timestamps, "1h", "BTCUSDT", available_time, 80)
                btc_4h = _mtf_closed(mtf_frames, mtf_timestamps, "4h", "BTCUSDT", available_time, 80)
            except KeyError:
                continue
            if not all((candles_30m, candles_1h, candles_4h, btc_1h, btc_4h)):
                continue

            regime = strategy.regime(candles_4h, "4h")
            if regime.regime == "LONG_BIAS":
                direction = Direction.LONG
            elif regime.regime == "SHORT_BIAS":
                direction = Direction.SHORT
            else:
                continue
            features = auxiliary.get(symbol, {})
            oi_value = oi_change_at(
                features,
                available_time,
                int(getattr(config.strategy, "mtf_oi_max_age_minutes", 15)),
            )
            funding_rate = funding_at(
                features,
                available_time,
                float(getattr(config.risk, "funding_default_rate", 0.0)),
            )
            btc_1h_return = btc_1h[-1].close / max(btc_1h[-2].close, 1e-12) - 1.0 if len(btc_1h) >= 2 else 0.0
            btc_4h_return = btc_4h[-1].close / max(btc_4h[-2].close, 1e-12) - 1.0 if len(btc_4h) >= 2 else 0.0
            futures_reject = strategy._futures_filter_reject_reason(
                direction,
                oi_value,
                funding_rate,
                btc_1h_return,
                btc_4h_return,
            )
            if futures_reject:
                rejection_counts[futures_reject] = rejection_counts.get(futures_reject, 0) + 1
                continue

            event = detect_momentum_reset_release(direction, candles_1h, candles_30m, detector_config)
            if event is None:
                continue
            built = build_release_signal(config, event)
            if built is None:
                rejection_counts["setup_stop_outside_frozen_limits"] = rejection_counts.get("setup_stop_outside_frozen_limits", 0) + 1
                continue
            signal, metadata = built
            metadata.update(
                {
                    "4h_regime": regime.regime,
                    "4h_rsi": regime.rsi,
                    "regime_reason": regime.reason,
                    "btc_1h_return": btc_1h_return,
                    "btc_4h_return": btc_4h_return,
                    "funding_rate": funding_rate,
                    "oi_chg_30m": oi_value,
                }
            )
            rank_score, momentum_pct, rank_volume_ratio = trader._entry_rank_metrics(signal, candles_30m[-80:])
            metadata.update(
                {
                    "rank_score": rank_score,
                    "directional_momentum_pct": momentum_pct,
                    "entry_rank_volume_ratio": rank_volume_ratio,
                }
            )
            candidate = EntryCandidate(
                symbol,
                signal,
                event.candle,
                rank_score,
                momentum_pct,
                rank_volume_ratio,
                "mtf_momentum_reset_contraction_release",
                metadata=metadata,
            )
            trade, shadow = _shadow_trade(
                config,
                candidate,
                execution_candles,
                signal_candles,
                mtf_frames,
                mtf_timestamps,
                available_time,
                execution_config,
                client,
            )
            overlap = classify_structure_overlap(
                event.candle.timestamp,
                known_structure_times.get((symbol, direction.name), ()),
                structure_window_minutes,
            )
            rank_floor = mtf_min_rank_score_for_direction(config.strategy, direction)
            row: dict[str, Any] = {
                "event_id": mtf_momentum_reset_event_id(symbol, direction, event.candle.timestamp),
                "symbol": symbol,
                "side": direction.name,
                "release_time": event.candle.timestamp.isoformat(),
                "signal_available_time": available_time.isoformat(),
                "month": available_time.strftime("%Y-%m"),
                "frozen_rank_floor": rank_floor,
                "passes_frozen_rank_floor": rank_score >= rank_floor,
                **metadata,
                **overlap,
                **shadow,
            }
            if trade is not None:
                row.update(
                    {
                        "shadow_fee": trade.get("fee"),
                        "shadow_slippage_cost": trade.get("slippage_cost"),
                        "shadow_funding": trade.get("funding"),
                    }
                )
            rows.append(row)
        if progress:
            print(f"[{symbol_index}/{len(symbols)}] {symbol}: cumulative setup events={len(rows)}", flush=True)

    executable = [row for row in rows if not row.get("shadow_reject_reason")]
    rank_eligible = [row for row in executable if row.get("passes_frozen_rank_floor")]
    independent = [row for row in rank_eligible if row.get("independent_from_structure_window")]
    experiment_reports = {
        experiment.name: _experiment_report(rows, experiment, cluster_hours)
        for experiment in experiments
    }
    return {
        "feature_version": MTF_MOMENTUM_RESET_VERSION,
        "config": config_path,
        "config_hash": config_sha256(json.loads(Path(config_path).read_text(encoding="utf-8"))),
        "data_dir": data_dir,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "effective_end_after_embargo": effective_end.isoformat(),
        "symbols": list(symbols),
        "detector_config": detector_config.__dict__,
        "experiment_budget": len(experiments),
        "method": {
            "event": "closed 1h momentum reset followed by closed 30m contraction release",
            "direction_gate": "frozen 4h regime and BTC/funding filters",
            "rank_policy": "frozen direction-specific floor; no relaxation",
            "execution": "next 1m open helper, full-cost, conservative same-bar ordering",
            "portfolio_effect": "isolated shadow only",
            "label_embargo_hours": embargo_hours,
            "event_cluster_hours": cluster_hours,
            "independence_window_minutes": structure_window_minutes,
            "old_15m_trigger_used": False,
            "mtpc_used": False,
            "research_regime_mode": regime_mode,
        },
        "counts": {
            "raw_events": len(rows),
            "executable_events": len(executable),
            "rank_eligible_events": len(rank_eligible),
            "independent_rank_eligible_events": len(independent),
            "structure_related_events": len(rank_eligible) - len(independent),
            "fill_revalidation_rejected": sum(bool(row.get("shadow_reject_reason")) for row in rows),
        },
        "rejection_counts": rejection_counts,
        "all_executable": candidate_metrics(executable),
        "rank_eligible": candidate_metrics(rank_eligible),
        "independent_rank_eligible": candidate_metrics(independent),
        "by_side_independent": grouped_report(independent, "side"),
        "by_month_independent": grouped_report(independent, "month"),
        "feature_buckets_independent": {
            field: quantile_bucket_report(independent, field)
            for field in BUCKET_FIELDS
        },
        "experiments": experiment_reports,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research 1h momentum reset plus 30m contraction release")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--structure-candidate-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--structure-window-minutes", type=int, default=120)
    parser.add_argument("--cluster-hours", type=int, default=4)
    parser.add_argument("--embargo-hours", type=int, default=12)
    parser.add_argument(
        "--regime-mode",
        choices=("frozen_rsi_reversal", "trend_pullback"),
        default="frozen_rsi_reversal",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = research_momentum_reset_setup(
        args.config,
        args.data_dir,
        _timestamp(args.start),
        _timestamp(args.end),
        structure_candidate_report=args.structure_candidate_report,
        structure_window_minutes=max(0, args.structure_window_minutes),
        cluster_hours=max(0, args.cluster_hours),
        embargo_hours=max(0, args.embargo_hours),
        regime_mode=args.regime_mode,
        max_symbols=args.max_symbols,
        progress=not args.no_progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.csv_output:
        csv_output = Path(args.csv_output)
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(csv_output, payload["rows"])
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "independent_rank_eligible": payload["independent_rank_eligible"],
                "experiments": {
                    name: report["independent_clustered"]
                    for name, report in payload["experiments"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
