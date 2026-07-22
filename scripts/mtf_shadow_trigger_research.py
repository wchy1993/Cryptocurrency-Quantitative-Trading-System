from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crypto_scalper.data import interval_to_milliseconds
from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_execution_backtest import (
    _mtf_closed,
    _mtf_required_timeframes,
    _resample_to_timeframe,
)
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
    mtf_candidate_quality,
    quantile_bucket_report,
)
from crypto_scalper.mtf_shadow_trigger import (
    MTF_SHADOW_TRIGGER_VERSION,
    MtfNativeShadowTriggerConfig,
    MtfNativeShadowTriggerDetector,
    can_be_native_shadow_trigger,
    classify_structure_overlap,
    mtf_shadow_trigger_event_id,
)
from crypto_scalper.realistic_data import load_funding_rate_directory
from crypto_scalper.risk import execution_config_from_live_config
from scripts.mtf_candidate_quality_research import (
    _shadow_trade,
    _single_symbol_frames,
    _timestamp,
    _write_csv,
)


DEFAULT_BUCKET_FIELDS = (
    "rank_score",
    "trigger_volume_ratio",
    "trigger_body_atr",
    "shadow_probe_wick_ratio",
    "shadow_probe_range_atr",
    "shadow_directional_close_position",
    "shadow_probe_distance_from_setup_atr",
    "fill_stop_pct",
    "target_to_cost_ratio",
)


def _structure_times(path: str | None) -> dict[tuple[str, str], list[datetime]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output: dict[tuple[str, str], list[datetime]] = {}
    for row in payload.get("rows", []):
        key = (str(row.get("symbol", "")), str(row.get("side", "")).upper())
        trigger_time = row.get("trigger_time")
        if not key[0] or not key[1] or not trigger_time:
            continue
        output.setdefault(key, []).append(_timestamp(trigger_time))
    for times in output.values():
        times.sort()
    return output


def research_shadow_triggers(
    config_path: str,
    data_dir: str,
    start: datetime,
    end: datetime,
    structure_candidate_report: str | None = None,
    trigger_config: MtfNativeShadowTriggerConfig | None = None,
    structure_window_minutes: int = 120,
    max_symbols: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    config = load_live_config(config_path)
    detector_config = trigger_config or MtfNativeShadowTriggerConfig()
    detector = MtfNativeShadowTriggerDetector(detector_config)
    symbols = tuple(config.trading.symbols[:max_symbols]) if max_symbols else tuple(config.trading.symbols)
    known_structure_times = _structure_times(structure_candidate_report)

    btc_loaded = _load_symbol_data(data_dir, ("BTCUSDT",), "1m")
    if not btc_loaded:
        raise RuntimeError(f"BTCUSDT 1m data missing from {data_dir}")
    btc_1m = btc_loaded["BTCUSDT"]
    timeframes = _mtf_required_timeframes(config)
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
    for symbol_index, symbol in enumerate(symbols, start=1):
        loaded = btc_loaded if symbol == "BTCUSDT" else _load_symbol_data(data_dir, (symbol,), "1m")
        if symbol not in loaded:
            continue
        execution_candles = loaded[symbol]
        mtf_frames, mtf_timestamps = _single_symbol_frames(symbol, execution_candles, btc_by_timeframe, timeframes)
        signal_candles = _resample_to_timeframe(execution_candles, "1m", config.trading.timeframe)
        client = HistoricalClient({symbol: signal_candles}, config.trading.timeframe, ())
        trader = BinanceAutoTrader(config, client)
        strategy = Mtf4hRsiRegimePullbackStrategy(config)
        trigger_timeframe = str(config.strategy.mtf_trigger_timeframe)
        regime_timeframe = str(config.strategy.mtf_regime_timeframe)
        trigger_rows = mtf_frames[trigger_timeframe][symbol]

        for trigger_index, trigger in enumerate(trigger_rows):
            available_time = trigger.timestamp + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
            if available_time < start or available_time >= end:
                continue
            if not can_be_native_shadow_trigger(
                trigger_rows,
                trigger_index,
                detector_config.local_lookback,
            ):
                continue
            try:
                candles_trigger = trigger_rows[max(0, trigger_index - 179):trigger_index + 1]
                candles_30m = _mtf_closed(mtf_frames, mtf_timestamps, "30m", symbol, available_time, 140)
                candles_1h = _mtf_closed(mtf_frames, mtf_timestamps, "1h", symbol, available_time, 140)
                candles_regime = _mtf_closed(
                    mtf_frames,
                    mtf_timestamps,
                    regime_timeframe,
                    symbol,
                    available_time,
                    160,
                )
                btc_1h = _mtf_closed(mtf_frames, mtf_timestamps, "1h", "BTCUSDT", available_time, 80)
                btc_4h = _mtf_closed(mtf_frames, mtf_timestamps, "4h", "BTCUSDT", available_time, 80)
            except KeyError:
                continue
            if not all((candles_trigger, candles_30m, candles_1h, candles_regime, btc_1h, btc_4h)):
                continue

            features = auxiliary.get(symbol, {})
            decision = strategy.build_signal(
                symbol,
                candles_trigger,
                candles_30m,
                candles_1h,
                candles_regime,
                btc_1h,
                btc_4h,
                oi_change_at(
                    features,
                    available_time,
                    int(getattr(config.strategy, "mtf_oi_max_age_minutes", 15)),
                ),
                funding_at(
                    features,
                    available_time,
                    float(getattr(config.risk, "funding_default_rate", 0.0)),
                ),
                candles_regime=candles_regime,
                candles_trigger=candles_trigger,
                regime_timeframe=regime_timeframe,
                trigger_timeframe=trigger_timeframe,
                trigger_detector=detector,
            )
            if decision.signal is None or decision.candle is None:
                continue

            direction = decision.signal.direction
            rank_score, momentum_pct, volume_ratio = trader._entry_rank_metrics(decision.signal, decision.rank_candles)
            metadata = {
                **decision.metadata,
                "rank_score": rank_score,
                "directional_momentum_pct": momentum_pct,
                "entry_rank_volume_ratio": volume_ratio,
            }
            candidate = EntryCandidate(
                symbol,
                decision.signal,
                decision.candle,
                rank_score,
                momentum_pct,
                volume_ratio,
                "mtf_native_shadow_trigger",
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
            quality = mtf_candidate_quality({**metadata, **shadow}, direction)
            floor = mtf_min_rank_score_for_direction(config.strategy, direction)
            overlap = classify_structure_overlap(
                decision.candle.timestamp,
                known_structure_times.get((symbol, direction.name), ()),
                structure_window_minutes,
            )
            mode = str(metadata.get("shadow_trigger_mode", metadata.get("trigger_mode", "missing")))
            row: dict[str, Any] = {
                "event_id": mtf_shadow_trigger_event_id(symbol, direction, mode, decision.candle.timestamp),
                "symbol": symbol,
                "side": direction.name,
                "trigger_time": decision.candle.timestamp.isoformat(),
                "signal_available_time": available_time.isoformat(),
                "frozen_rank_floor": floor,
                "passes_frozen_rank_floor": rank_score >= floor,
                **metadata,
                **quality.as_dict(),
                **overlap,
                **shadow,
            }
            row["month"] = decision.candle.timestamp.strftime("%Y-%m")
            if trade is not None:
                row["shadow_fee"] = trade.get("fee")
                row["shadow_slippage_cost"] = trade.get("slippage_cost")
                row["shadow_funding"] = trade.get("funding")
            rows.append(row)
        if progress:
            print(f"[{symbol_index}/{len(symbols)}] {symbol}: cumulative shadow events={len(rows)}", flush=True)

    executable = [row for row in rows if not row.get("shadow_reject_reason")]
    rank_eligible = [row for row in executable if row.get("passes_frozen_rank_floor")]
    independent = [
        row
        for row in rank_eligible
        if row.get("independent_from_structure_window")
    ]
    precursors = [
        row
        for row in rank_eligible
        if row.get("structure_followed_within_window")
    ]
    report = {
        "feature_version": MTF_SHADOW_TRIGGER_VERSION,
        "config": config_path,
        "config_hash": config_sha256(json.loads(Path(config_path).read_text(encoding="utf-8"))),
        "data_dir": data_dir,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": list(symbols),
        "trigger_config": detector_config.__dict__,
        "method": {
            "higher_timeframe_gates": "frozen 4h regime, 1h confirmation, 30m setup, BTC/funding filters",
            "shadow_triggers": ["two_bar_false_break", "higher_low_break", "lower_high_break"],
            "execution": "next 1m open, frozen full-cost model, conservative same-bar ordering",
            "portfolio_effect": "isolated shadow only",
            "rank_policy": "frozen direction-specific floor; no relaxation",
            "independence_window_minutes": structure_window_minutes,
            "quality_score_uses_future_outcome": False,
        },
        "counts": {
            "raw_events": len(rows),
            "executable_events": len(executable),
            "rank_eligible_events": len(rank_eligible),
            "independent_rank_eligible_events": len(independent),
            "structure_precursor_events": len(precursors),
            "same_bar_structure_events": sum(bool(row.get("same_bar_structure_candidate")) for row in rank_eligible),
            "fill_revalidation_rejected": sum(bool(row.get("shadow_reject_reason")) for row in rows),
        },
        "all_executable": candidate_metrics(executable),
        "rank_eligible": candidate_metrics(rank_eligible),
        "independent_rank_eligible": candidate_metrics(independent),
        "by_trigger_independent": grouped_report(independent, "shadow_trigger_mode"),
        "by_side_independent": grouped_report(independent, "side"),
        "by_month_independent": grouped_report(independent, "month"),
        "by_overlap_rank_eligible": grouped_report(
            rank_eligible,
            lambda row: "independent" if row.get("independent_from_structure_window") else "structure_related",
        ),
        "quality_buckets_independent": {
            field: quantile_bucket_report(independent, field)
            for field in DEFAULT_BUCKET_FIELDS
        },
        "rows": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Research independent native MTF shadow triggers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--structure-candidate-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--structure-window-minutes", type=int, default=120)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = research_shadow_triggers(
        args.config,
        args.data_dir,
        _timestamp(args.start),
        _timestamp(args.end),
        structure_candidate_report=args.structure_candidate_report,
        structure_window_minutes=max(0, args.structure_window_minutes),
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
                "rank_eligible": payload["rank_eligible"],
                "independent_rank_eligible": payload["independent_rank_eligible"],
                "by_trigger_independent": payload["by_trigger_independent"],
                "by_side_independent": payload["by_side_independent"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
