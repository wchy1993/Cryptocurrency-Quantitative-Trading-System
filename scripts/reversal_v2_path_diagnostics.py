from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_scalper.data import parse_timestamp
from crypto_scalper.live_portfolio_backtest import _load_symbol_data


def analyze(report_path: str, data_dir: str, horizon_minutes: int, take_profit_r: float) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    trades = [
        trade
        for trade in report.get("trades", [])
        if "indicator_reversal_v2" in str(trade.get("entry_reason", ""))
    ]
    symbols = tuple(sorted({str(trade["symbol"]) for trade in trades}))
    candles_by_symbol = _load_symbol_data(data_dir, symbols, "1m")
    timestamps = {
        symbol: [candle.timestamp for candle in candles]
        for symbol, candles in candles_by_symbol.items()
    }

    rows = []
    for trade in trades:
        symbol = str(trade["symbol"])
        direction = 1.0 if str(trade["side"]).upper() == "LONG" else -1.0
        entry_time = parse_timestamp(str(trade["entry_time"]))
        entry_price = float(trade["entry_price"])
        stop_price = float(trade["initial_stop_price"])
        risk_price = direction * (entry_price - stop_price)
        if risk_price <= 0 or symbol not in candles_by_symbol:
            continue
        target_price = entry_price + direction * risk_price * take_profit_r
        candles = candles_by_symbol[symbol]
        symbol_timestamps = timestamps[symbol]
        start = bisect.bisect_right(symbol_timestamps, entry_time)
        end_time = entry_time + timedelta(minutes=horizon_minutes)
        end = bisect.bisect_right(symbol_timestamps, end_time)
        max_favorable_r = 0.0
        max_adverse_r = 0.0
        first_touch = "none"
        first_touch_minutes = None
        close_r = 0.0
        for candle in candles[start:end]:
            favorable_price = candle.high if direction > 0 else candle.low
            adverse_price = candle.low if direction > 0 else candle.high
            max_favorable_r = max(max_favorable_r, direction * (favorable_price - entry_price) / risk_price)
            max_adverse_r = min(max_adverse_r, direction * (adverse_price - entry_price) / risk_price)
            stop_hit = candle.low <= stop_price if direction > 0 else candle.high >= stop_price
            target_hit = candle.high >= target_price if direction > 0 else candle.low <= target_price
            if first_touch == "none" and (stop_hit or target_hit):
                first_touch = "stop" if stop_hit else "target"
                first_touch_minutes = (candle.timestamp - entry_time).total_seconds() / 60.0
            close_r = direction * (candle.close - entry_price) / risk_price
        rows.append(
            {
                "symbol": symbol,
                "entry_time": trade["entry_time"],
                "actual_exit_reason": trade.get("exit_reason"),
                "actual_net_pnl": float(trade.get("net_pnl", 0.0)),
                "first_touch": first_touch,
                "first_touch_minutes": first_touch_minutes,
                "max_favorable_r": max_favorable_r,
                "max_adverse_r": max_adverse_r,
                "horizon_close_r": close_r,
            }
        )

    fail_fast = [row for row in rows if row["actual_exit_reason"] == "reversal_v2_fail_fast"]
    return {
        "report": report_path,
        "horizon_minutes": horizon_minutes,
        "take_profit_r": take_profit_r,
        "trade_count": len(rows),
        "first_touch_distribution": dict(Counter(row["first_touch"] for row in rows)),
        "fail_fast_count": len(fail_fast),
        "fail_fast_first_touch_distribution": dict(Counter(row["first_touch"] for row in fail_fast)),
        "fail_fast_reached_0_5r": sum(row["max_favorable_r"] >= 0.5 for row in fail_fast),
        "fail_fast_reached_1r": sum(row["max_favorable_r"] >= 1.0 for row in fail_fast),
        "fail_fast_reached_target": sum(row["max_favorable_r"] >= take_profit_r for row in fail_fast),
        "fail_fast_average_horizon_close_r": (
            sum(row["horizon_close_r"] for row in fail_fast) / len(fail_fast)
            if fail_fast
            else 0.0
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reversal V2 post-entry path counterfactual")
    parser.add_argument("--report", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=180)
    parser.add_argument("--take-profit-r", type=float, default=1.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = analyze(args.report, args.data_dir, args.horizon_minutes, args.take_profit_r)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
