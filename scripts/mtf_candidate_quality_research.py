from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crypto_scalper.data import interval_to_milliseconds, parse_timestamp
from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_execution_backtest import (
    _cached_mtf_4h_rsi_candidate,
    _closed_signal_index,
    _initialize_mtf_position,
    _mtf_adjust_candidate_for_fill,
    _mtf_exit_reason_for_position,
    _mtf_required_timeframes,
    _resample_to_timeframe,
)
from crypto_scalper.live_portfolio_backtest import (
    HistoricalClient,
    _close_position,
    _load_symbol_data,
    _open_position,
    _update_position_excursion,
)
from crypto_scalper.live_trader import BinanceAutoTrader
from crypto_scalper.models import Candle, Direction
from crypto_scalper.mtf_4h_rsi_regime import load_auxiliary_features
from crypto_scalper.mtf_candidate_quality import (
    MTF_CANDIDATE_FEATURE_VERSION,
    can_be_structure_break,
    candidate_metrics,
    config_sha256,
    executable_mark_pnl_usdt,
    full_cost_stop_risk_usdt,
    grouped_report,
    mtf_candidate_event_id,
    mtf_candidate_quality,
    next_1m_execution_index,
    quantile_bucket_report,
)
from crypto_scalper.realistic_data import load_funding_rate_directory
from crypto_scalper.risk import (
    BacktestExecutionStats,
    conservative_quantity,
    execution_config_from_live_config,
)


DEFAULT_BUCKET_FIELDS = (
    "rank_score",
    "quality_score_v1",
    "directional_momentum_pct",
    "trigger_volume_ratio",
    "entry_rank_volume_ratio",
    "trigger_body_atr",
    "directional_close_position",
    "fill_stop_pct",
    "target_to_cost_ratio",
    "4h_rsi",
    "1h_rsi",
    "btc_directional_1h_return",
)


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    return parse_timestamp(str(value)).replace(tzinfo=None)


def _baseline_trade_keys(path: str | None) -> set[tuple[str, str, datetime]]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("experiments"), list) and payload["experiments"]:
        report = payload["experiments"][0].get("report", {})
    else:
        report = payload
    keys = set()
    for trade in report.get("trades", []):
        if "mtf_4h_rsi_regime_pullback" not in str(trade.get("entry_reason", "")):
            continue
        keys.add((str(trade["symbol"]), str(trade["side"]).upper(), _timestamp(trade["signal_time"])))
    return keys


def _single_symbol_frames(
    symbol: str,
    symbol_1m: list[Candle],
    btc_by_timeframe: dict[str, list[Candle]],
    timeframes: tuple[str, ...],
) -> tuple[dict[str, dict[str, list[Candle]]], dict[str, dict[str, list[datetime]]]]:
    frames: dict[str, dict[str, list[Candle]]] = {}
    for timeframe in timeframes:
        symbol_rows = _resample_to_timeframe(symbol_1m, "1m", timeframe)
        btc_rows = symbol_rows if symbol == "BTCUSDT" else btc_by_timeframe[timeframe]
        frames[timeframe] = {symbol: symbol_rows, "BTCUSDT": btc_rows}
    timestamps = {
        timeframe: {name: [candle.timestamp for candle in candles] for name, candles in by_symbol.items()}
        for timeframe, by_symbol in frames.items()
    }
    return frames, timestamps


def _shadow_trade(
    config: Any,
    candidate: Any,
    execution_candles: list[Candle],
    signal_candles: list[Candle],
    mtf_frames: dict[str, dict[str, list[Candle]]],
    mtf_timestamps: dict[str, dict[str, list[datetime]]],
    available_time: datetime,
    execution_config: Any,
    client: HistoricalClient,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    execution_times = [candle.timestamp for candle in execution_candles]
    entry_index = next_1m_execution_index(execution_times, available_time)
    if entry_index is None:
        return None, {"shadow_reject_reason": "missing_next_1m_open"}
    entry_candle = execution_candles[entry_index]
    fill_stats: dict[str, int] = {}
    adjusted = _mtf_adjust_candidate_for_fill(
        config,
        candidate,
        entry_candle.open,
        execution_config,
        fill_stats,
    )
    if adjusted is None:
        reason = next(iter(fill_stats), "fill_revalidation")
        return None, {"shadow_reject_reason": reason}

    fill_metadata = {
        "entry_chase_atr": adjusted.metadata.get("entry_chase_atr"),
        "fill_stop_pct": adjusted.metadata.get("fill_stop_pct"),
        "target_to_cost_ratio": adjusted.metadata.get("target_to_cost_ratio"),
    }

    rules = client.symbol_rules(candidate.symbol)
    quantity = conservative_quantity(rules, 1000.0 / max(entry_candle.open, 1e-12))
    positions: dict[str, Any] = {}
    trades: list[dict[str, Any]] = []
    signal_times = [candle.timestamp for candle in signal_candles]
    signal_ms = interval_to_milliseconds(config.trading.timeframe)
    initial_signal_index = max(0, _closed_signal_index(signal_times, available_time, signal_ms))
    execution_candle = replace(
        entry_candle,
        high=entry_candle.open,
        low=entry_candle.open,
        close=entry_candle.open,
    )
    cash = _open_position(
        config,
        100_000.0,
        positions,
        candidate.symbol,
        adjusted.signal,
        execution_candle,
        quantity,
        initial_signal_index,
        raw_entry_price=entry_candle.open,
        entry_time=entry_candle.timestamp,
        signal_time=candidate.candle.timestamp,
        signal_available_time=available_time,
        execution_config=execution_config,
        rules=rules,
    )
    if candidate.symbol not in positions:
        return None, {
            "shadow_reject_reason": "execution_capacity_or_exchange_rules",
            **fill_metadata,
        }
    position = positions[candidate.symbol]
    position.strategy_metadata = dict(adjusted.metadata)
    _initialize_mtf_position(config, position, adjusted)
    stop_risk = full_cost_stop_risk_usdt(position, execution_config, rules)
    max_executable_pnl = executable_mark_pnl_usdt(
        position,
        position.raw_entry_price or position.entry_price,
        execution_config,
        rules,
    )
    min_executable_pnl = max_executable_pnl
    threshold_hits = {0.5: False, 1.0: False, 2.0: False}
    stop_first = False
    execution_stats = BacktestExecutionStats()

    for index in range(entry_index + 1, len(execution_candles)):
        candle = execution_candles[index]
        signal_index = max(0, _closed_signal_index(signal_times, candle.timestamp, signal_ms))
        position.bars_held = max(0, signal_index - position.entry_index)
        stop_hit = (
            candle.low <= position.stop_price
            if position.direction is Direction.LONG
            else candle.high >= position.stop_price
        )
        take_profit_hit = (
            candle.high >= position.take_profit_price
            if position.direction is Direction.LONG
            else candle.low <= position.take_profit_price
        )
        if stop_hit and take_profit_hit:
            execution_stats.same_bar_tp_sl_conflict_count += 1

        # Keep the production MFE/MAE fields for comparability, but do not award
        # favorable same-bar excursion when the conservative path stops first.
        _update_position_excursion(position, candle)
        adverse_price = candle.low if position.direction is Direction.LONG else candle.high
        min_executable_pnl = min(
            min_executable_pnl,
            executable_mark_pnl_usdt(position, adverse_price, execution_config, rules),
        )
        if not stop_hit:
            favorable_price = candle.high if position.direction is Direction.LONG else candle.low
            favorable_pnl = executable_mark_pnl_usdt(position, favorable_price, execution_config, rules)
            max_executable_pnl = max(max_executable_pnl, favorable_pnl)
            for threshold in threshold_hits:
                threshold_hits[threshold] = threshold_hits[threshold] or favorable_pnl >= threshold * stop_risk

        exit_price: float | None = None
        exit_reason = ""
        if stop_hit and (not take_profit_hit or execution_config.mode != "optimistic"):
            exit_price = position.stop_price
            exit_reason = "stop_loss_1m"
            stop_first = True
        elif take_profit_hit:
            exit_price = position.take_profit_price
            exit_reason = "take_profit_1m"
        else:
            mtf_reason = _mtf_exit_reason_for_position(config, position, candle, mtf_frames, mtf_timestamps)
            if mtf_reason:
                exit_price = candle.close
                exit_reason = mtf_reason
        if exit_price is not None:
            cash = _close_position(
                config,
                cash,
                positions,
                trades,
                candidate.symbol,
                exit_price,
                exit_reason,
                signal_index,
                candle.timestamp,
                execution_config=execution_config,
                rules=rules,
            )
            break

    if positions:
        final = execution_candles[-1]
        signal_index = max(0, _closed_signal_index(signal_times, final.timestamp, signal_ms))
        _close_position(
            config,
            cash,
            positions,
            trades,
            candidate.symbol,
            final.close,
            "end_of_data",
            signal_index,
            final.timestamp,
            execution_config=execution_config,
            rules=rules,
        )
    trade = trades[-1]
    full_cost = float(trade.get("fee", 0.0)) + float(trade.get("slippage_cost", 0.0)) - float(trade.get("funding", 0.0))
    return trade, {
        "shadow_reject_reason": "",
        **fill_metadata,
        "shadow_stop_full_cost_risk_usdt": stop_risk,
        "shadow_net_r": float(trade["net_pnl"]) / stop_risk,
        "shadow_cost_r": full_cost / stop_risk,
        "shadow_executable_mfe_r": max_executable_pnl / stop_risk,
        "shadow_executable_mae_r": min_executable_pnl / stop_risk,
        "shadow_raw_mfe_r": float(trade.get("mfe", 0.0)) / stop_risk,
        "shadow_raw_mae_r": float(trade.get("mae", 0.0)) / stop_risk,
        "hit_0p5r": threshold_hits[0.5],
        "hit_1r": threshold_hits[1.0],
        "hit_2r": threshold_hits[2.0],
        "stop_first": stop_first,
        "shadow_exit_reason": trade.get("exit_reason"),
        "shadow_entry_time": trade.get("entry_time"),
        "shadow_exit_time": trade.get("exit_time"),
        "shadow_hold_minutes": trade.get("hold_minutes"),
        "shadow_net_pnl_per_1000_notional": trade.get("net_pnl"),
        "same_bar_conflicts": execution_stats.same_bar_tp_sl_conflict_count,
    }


def research_candidates(
    config_path: str,
    data_dir: str,
    start: datetime,
    end: datetime,
    baseline_report: str | None = None,
    max_symbols: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    frozen_config = load_live_config(config_path)
    research_strategy = replace(
        frozen_config.strategy,
        mtf_min_rank_score=-999.0,
        mtf_long_min_rank_score=-999.0,
        mtf_short_min_rank_score=-999.0,
        mtf_max_open_positions=999,
        mtf_max_daily_trades=999,
        mtf_symbol_cooldown_hours=0,
    )
    config = replace(frozen_config, strategy=research_strategy)
    symbols = tuple(config.trading.symbols[:max_symbols]) if max_symbols else tuple(config.trading.symbols)
    baseline_keys = _baseline_trade_keys(baseline_report)
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
        candidate_cache: dict[tuple[Any, ...], Any] = {}
        trigger_timeframe = str(config.strategy.mtf_trigger_timeframe)
        trigger_rows = mtf_frames[trigger_timeframe][symbol]
        trigger_mode = str(getattr(config.strategy, "mtf_15m_trigger_mode", "both")).lower()
        structure_lookback = max(2, int(getattr(config.strategy, "mtf_15m_structure_break_lookback", 3)))
        for trigger_index, trigger in enumerate(trigger_rows):
            available_time = trigger.timestamp + timedelta(milliseconds=interval_to_milliseconds(trigger_timeframe))
            if available_time < start or available_time >= end:
                continue
            if trigger_mode == "structure_break" and not can_be_structure_break(
                trigger_rows,
                trigger_index,
                structure_lookback,
            ):
                continue
            stats: dict[str, int] = {}
            candidate = _cached_mtf_4h_rsi_candidate(
                trader,
                config,
                symbol,
                available_time,
                mtf_frames,
                mtf_timestamps,
                {symbol: auxiliary.get(symbol, {})},
                stats,
                {},
                {},
                candidate_cache,
                {},
            )
            if candidate is None:
                continue
            side = candidate.signal.direction.name
            event_id = mtf_candidate_event_id(symbol, candidate.signal.direction, candidate.candle.timestamp)
            metadata = dict(candidate.metadata)
            frozen_rank_floor = (
                float(frozen_config.strategy.mtf_long_min_rank_score)
                if candidate.signal.direction is Direction.LONG
                else float(frozen_config.strategy.mtf_short_min_rank_score)
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
            quality = mtf_candidate_quality({**metadata, **shadow}, candidate.signal.direction)
            row: dict[str, Any] = {
                "event_id": event_id,
                "symbol": symbol,
                "side": side,
                "trigger_time": candidate.candle.timestamp.isoformat(),
                "signal_available_time": available_time.isoformat(),
                "frozen_filled": (symbol, side, candidate.candle.timestamp) in baseline_keys,
                "passes_frozen_rank_floor": float(candidate.rank_score) >= frozen_rank_floor,
                "frozen_rank_floor": frozen_rank_floor,
                **metadata,
                **quality.as_dict(),
                **shadow,
            }
            row["directional_close_position"] = (
                float(metadata.get("trigger_close_position", 0.5))
                if side == "LONG"
                else 1.0 - float(metadata.get("trigger_close_position", 0.5))
            )
            row["btc_directional_1h_return"] = float(metadata.get("btc_1h_return", 0.0)) * candidate.signal.direction.value
            if trade is not None:
                row["shadow_fee"] = trade.get("fee")
                row["shadow_slippage_cost"] = trade.get("slippage_cost")
                row["shadow_funding"] = trade.get("funding")
            rows.append(row)
        if progress:
            print(f"[{symbol_index}/{len(symbols)}] {symbol}: cumulative candidates={len(rows)}", flush=True)

    executable_rows = [row for row in rows if not row.get("shadow_reject_reason")]
    unfilled_rows = [row for row in executable_rows if not row.get("frozen_filled")]
    report = {
        "feature_version": MTF_CANDIDATE_FEATURE_VERSION,
        "config": config_path,
        "config_hash": config_sha256(json.loads(Path(config_path).read_text(encoding="utf-8"))),
        "data_dir": data_dir,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": list(symbols),
        "method": {
            "candidate_generation": "frozen MTF structural signal with scheduling and rank floors disabled",
            "execution": "next 1m open, frozen full-cost model, conservative same-bar ordering",
            "portfolio_effect": "isolated shadow; no position, cooldown, or daily-limit effects",
            "quality_score_uses_future_outcome": False,
        },
        "counts": {
            "raw_candidates": len(rows),
            "executable_candidates": len(executable_rows),
            "frozen_filled_matches": sum(bool(row.get("frozen_filled")) for row in rows),
            "unfilled_executable_candidates": len(unfilled_rows),
            "rank_floor_rejected": sum(not bool(row.get("passes_frozen_rank_floor")) for row in executable_rows),
            "fill_revalidation_rejected": sum(bool(row.get("shadow_reject_reason")) for row in rows),
        },
        "overall": candidate_metrics(executable_rows),
        "unfilled_only": candidate_metrics(unfilled_rows),
        "by_side": grouped_report(executable_rows, "side"),
        "unfilled_by_side": grouped_report(unfilled_rows, "side"),
        "by_frozen_status": grouped_report(executable_rows, lambda row: "filled" if row.get("frozen_filled") else "unfilled"),
        "by_rank_floor": grouped_report(executable_rows, lambda row: "pass" if row.get("passes_frozen_rank_floor") else "below"),
        "by_exit_reason": grouped_report(executable_rows, "shadow_exit_reason"),
        "quality_buckets_unfilled": {
            field: quantile_bucket_report(unfilled_rows, field)
            for field in DEFAULT_BUCKET_FIELDS
        },
        "rows": rows,
    }
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row if not isinstance(row.get(field), (dict, list))})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Research frozen MTF unfilled candidate quality")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--baseline-report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = research_candidates(
        args.config,
        args.data_dir,
        _timestamp(args.start),
        _timestamp(args.end),
        baseline_report=args.baseline_report,
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
    print(json.dumps({key: payload[key] for key in ("counts", "overall", "unfilled_only", "by_side", "unfilled_by_side")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
