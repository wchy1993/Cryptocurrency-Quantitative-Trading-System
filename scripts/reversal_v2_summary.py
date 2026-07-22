from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) > 0]
    losses = [float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "trades": len(trades),
        "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (None if not gross_profit else float("inf")),
        "expectancy": sum(float(trade.get("net_pnl", 0.0)) for trade in trades) / len(trades) if trades else 0.0,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_win_loss_ratio": avg_win / abs(avg_loss) if avg_loss else None,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    trades = [
        trade
        for trade in report.get("trades", [])
        if "indicator_reversal_v2" in str(trade.get("entry_reason", ""))
    ]
    ordered_wins = sorted(
        (trade for trade in trades if float(trade.get("net_pnl", 0.0)) > 0),
        key=lambda trade: float(trade.get("net_pnl", 0.0)),
        reverse=True,
    )
    top5 = ordered_wins[:5]
    top_symbol_pnl: dict[str, float] = defaultdict(float)
    for trade in trades:
        top_symbol_pnl[str(trade.get("symbol", ""))] += float(trade.get("net_pnl", 0.0))
    largest_symbol = max(top_symbol_pnl, key=top_symbol_pnl.get) if top_symbol_pnl else ""
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_month[str(trade.get("entry_time", ""))[:7]].append(trade)
    total_positive = sum(float(trade["net_pnl"]) for trade in ordered_wins)
    total_cost = sum(float(trade.get("fee", 0.0)) + float(trade.get("slippage_cost", 0.0)) for trade in trades)
    raw_positive = sum(max(0.0, float(trade.get("gross_pnl", 0.0))) for trade in trades)
    return {
        "portfolio": {
            key: report.get(key)
            for key in (
                "initial_equity",
                "final_equity",
                "net_pnl",
                "net_return_pct",
                "max_drawdown_pct",
                "gross_pnl",
                "fee",
                "slippage_cost",
                "funding",
            )
        },
        "trades": trade_metrics(trades),
        "cost_to_positive_raw_gross_ratio": total_cost / raw_positive if raw_positive else None,
        "by_month": {month: trade_metrics(rows) for month, rows in sorted(by_month.items())},
        "by_exit_reason": dict(Counter(str(trade.get("exit_reason", "")) for trade in trades)),
        "top5_profit_contribution_pct": (
            sum(float(trade["net_pnl"]) for trade in top5) / total_positive * 100.0
            if total_positive
            else None
        ),
        "exclude_top5": trade_metrics([trade for trade in trades if trade not in top5]),
        "top_symbol": largest_symbol,
        "exclude_top_symbol": trade_metrics([trade for trade in trades if trade.get("symbol") != largest_symbol]),
        "candidate_stats": report.get("reversal_v2_stats", {}),
        "historical_test_only": True,
        "final_acceptance_requires_post_freeze_shadow": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a Reversal V2 full-cost report")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    payload = summarize(report)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
