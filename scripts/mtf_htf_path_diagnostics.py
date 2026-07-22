from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_scalper.data import parse_timestamp
from crypto_scalper.live_portfolio_backtest import _load_symbol_data


MTF_REASON_TOKEN = "mtf_4h_rsi_regime_pullback"


def _reason_group(reason: Any) -> str:
    return str(reason or "unknown").split()[0]


def analyze(report_path: str, data_dir: str, horizon_minutes: int, take_profit_r: float) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    trades = [
        trade
        for trade in report.get("trades", [])
        if MTF_REASON_TOKEN in str(trade.get("entry_reason", ""))
    ]
    symbols = tuple(sorted({str(trade["symbol"]) for trade in trades}))
    candles_by_symbol = _load_symbol_data(data_dir, symbols, "1m")
    timestamps = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in candles_by_symbol.items()
    }
    rows: list[dict[str, Any]] = []
    for trade in trades:
        symbol = str(trade["symbol"])
        if symbol not in candles_by_symbol:
            continue
        side = 1.0 if str(trade.get("side", "")).upper() == "LONG" else -1.0
        entry_time = parse_timestamp(str(trade["entry_time"]))
        exit_time = parse_timestamp(str(trade["exit_time"]))
        entry_price = float(trade["entry_price"])
        stop_price = float(trade.get("initial_stop_price", 0.0))
        risk_price = side * (entry_price - stop_price)
        if risk_price <= 0:
            continue
        target_price = entry_price + side * risk_price * take_profit_r
        candles = candles_by_symbol[symbol]
        symbol_times = timestamps[symbol]
        start = bisect.bisect_right(symbol_times, entry_time)
        end = bisect.bisect_right(symbol_times, entry_time + timedelta(minutes=horizon_minutes))
        first_touch = "none"
        first_touch_minutes = None
        max_favorable_r = 0.0
        max_adverse_r = 0.0
        post_exit_max_favorable_r = 0.0
        horizon_close_r = 0.0
        for candle in candles[start:end]:
            favorable = candle.high if side > 0 else candle.low
            adverse = candle.low if side > 0 else candle.high
            favorable_r = side * (favorable - entry_price) / risk_price
            adverse_r = side * (adverse - entry_price) / risk_price
            max_favorable_r = max(max_favorable_r, favorable_r)
            max_adverse_r = min(max_adverse_r, adverse_r)
            if candle.timestamp > exit_time:
                post_exit_max_favorable_r = max(post_exit_max_favorable_r, favorable_r)
            stop_hit = candle.low <= stop_price if side > 0 else candle.high >= stop_price
            target_hit = candle.high >= target_price if side > 0 else candle.low <= target_price
            if first_touch == "none" and (stop_hit or target_hit):
                first_touch = "stop" if stop_hit else "target"
                first_touch_minutes = (candle.timestamp - entry_time).total_seconds() / 60.0
            horizon_close_r = side * (candle.close - entry_price) / risk_price
        rows.append(
            {
                "symbol": symbol,
                "side": trade.get("side"),
                "entry_time": trade.get("entry_time"),
                "actual_exit_reason": trade.get("exit_reason"),
                "actual_exit_group": _reason_group(trade.get("exit_reason")),
                "actual_net_pnl": float(trade.get("net_pnl", 0.0)),
                "first_touch": first_touch,
                "first_touch_minutes": first_touch_minutes,
                "max_favorable_r": max_favorable_r,
                "max_adverse_r": max_adverse_r,
                "post_exit_max_favorable_r": post_exit_max_favorable_r,
                "horizon_close_r": horizon_close_r,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["actual_exit_group"])].append(row)
    by_exit_group = {}
    for reason, reason_rows in sorted(grouped.items()):
        by_exit_group[reason] = {
            "trades": len(reason_rows),
            "first_touch": dict(Counter(row["first_touch"] for row in reason_rows)),
            "later_reached_0_5r": sum(row["post_exit_max_favorable_r"] >= 0.5 for row in reason_rows),
            "later_reached_1r": sum(row["post_exit_max_favorable_r"] >= 1.0 for row in reason_rows),
            "later_reached_target": sum(row["post_exit_max_favorable_r"] >= take_profit_r for row in reason_rows),
            "average_horizon_close_r": sum(row["horizon_close_r"] for row in reason_rows) / len(reason_rows),
        }
    return {
        "report": report_path,
        "horizon_minutes": horizon_minutes,
        "take_profit_r": take_profit_r,
        "trade_count": len(rows),
        "first_touch_distribution": dict(Counter(row["first_touch"] for row in rows)),
        "by_exit_group": by_exit_group,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MTF high-timeframe post-entry path counterfactual")
    parser.add_argument("--report", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=720)
    parser.add_argument("--take-profit-r", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = analyze(args.report, args.data_dir, args.horizon_minutes, args.take_profit_r)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
