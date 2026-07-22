from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MTF_REASON_TOKEN = "mtf_4h_rsi_regime_pullback"
MTPC_REASON_TOKEN = "multi_timeframe_trend_pullback_continuation"


def _strategy_name(trade: dict[str, Any]) -> str:
    reason = str(trade.get("entry_reason", ""))
    if MTPC_REASON_TOKEN in reason:
        return "mtpc"
    if MTF_REASON_TOKEN in reason:
        return "mtf_reversal"
    return "other"


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) > 0]
    losses = [float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    average_win = gross_profit / len(wins) if wins else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "trades": len(trades),
        "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "expectancy": sum(float(trade.get("net_pnl", 0.0)) for trade in trades) / len(trades) if trades else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_win_loss_ratio": average_win / abs(average_loss) if average_loss else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def _reason_tags(reason: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for token in str(reason).split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        tags[key] = value
    return tags


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _path_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    mfe_r_values: list[float] = []
    mae_r_values: list[float] = []
    for trade in trades:
        entry_price = float(trade.get("entry_price", 0.0))
        stop_price = float(trade.get("initial_stop_price", 0.0))
        quantity = abs(float(trade.get("quantity", trade.get("qty", 0.0))))
        risk_cash = abs(entry_price - stop_price) * quantity
        if risk_cash <= 0:
            continue
        mfe_r_values.append(float(trade.get("mfe", 0.0)) / risk_cash)
        mae_r_values.append(abs(float(trade.get("mae", 0.0))) / risk_cash)
    return {
        "samples": len(mfe_r_values),
        "mfe_r_p25": _percentile(mfe_r_values, 0.25),
        "mfe_r_median": _percentile(mfe_r_values, 0.50),
        "mfe_r_p75": _percentile(mfe_r_values, 0.75),
        "mae_r_median": _percentile(mae_r_values, 0.50),
        "mae_r_p75": _percentile(mae_r_values, 0.75),
        "reached_0_25r_pct": sum(value >= 0.25 for value in mfe_r_values) / len(mfe_r_values) * 100.0 if mfe_r_values else 0.0,
        "reached_0_50r_pct": sum(value >= 0.50 for value in mfe_r_values) / len(mfe_r_values) * 100.0 if mfe_r_values else 0.0,
        "reached_1_00r_pct": sum(value >= 1.00 for value in mfe_r_values) / len(mfe_r_values) * 100.0 if mfe_r_values else 0.0,
        "reached_1_50r_pct": sum(value >= 1.50 for value in mfe_r_values) / len(mfe_r_values) * 100.0 if mfe_r_values else 0.0,
    }


def _group(
    trades: list[dict[str, Any]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[str(key_fn(trade))].append(trade)
    return {key: _metrics(rows) for key, rows in sorted(grouped.items())}


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    all_trades = list(report.get("trades", []))
    has_mtpc = bool(report.get("mtpc_report"))
    has_mtf = bool(report.get("mtf_report"))
    if has_mtpc:
        # MTPC-only and combined reports are intentional staged experiments.
        # Their portfolio summary must include every active sleeve.
        trades = all_trades
    else:
        # Preserve the historical MTF diagnostic behavior for reports that may
        # contain unrelated legacy strategies.
        trades = [
            trade
            for trade in all_trades
            if MTF_REASON_TOKEN in str(trade.get("entry_reason", ""))
        ]
    wins = sorted(
        (trade for trade in trades if float(trade.get("net_pnl", 0.0)) > 0),
        key=lambda trade: float(trade.get("net_pnl", 0.0)),
        reverse=True,
    )
    top_five = wins[:5]
    by_symbol_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_symbol_rows[str(trade.get("symbol", ""))].append(trade)
    top_symbol = max(
        by_symbol_rows,
        key=lambda symbol: sum(float(trade.get("net_pnl", 0.0)) for trade in by_symbol_rows[symbol]),
        default="",
    )
    positive_pnl = sum(float(trade.get("net_pnl", 0.0)) for trade in wins)
    costs = sum(
        float(trade.get("fee", 0.0)) + float(trade.get("slippage_cost", 0.0))
        for trade in trades
    )
    positive_raw_gross = sum(max(0.0, float(trade.get("gross_pnl", 0.0))) for trade in trades)
    mtf_report = report.get("mtf_report", {})
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
        "overall": _metrics(trades),
        "path": _path_metrics(trades),
        "cost_to_positive_raw_gross_ratio": costs / positive_raw_gross if positive_raw_gross else None,
        "by_month": _group(trades, lambda trade: str(trade.get("entry_time", ""))[:7]),
        "by_side": _group(trades, lambda trade: trade.get("side", trade.get("direction", "unknown"))),
        "by_strategy": _group(trades, _strategy_name),
        "by_symbol": _group(trades, lambda trade: trade.get("symbol", "unknown")),
        "by_trigger": _group(trades, lambda trade: _reason_tags(str(trade.get("entry_reason", ""))).get("trigger", "unknown")),
        "by_regime_timeframe": _group(trades, lambda trade: _reason_tags(str(trade.get("entry_reason", ""))).get("regime_tf", "unknown")),
        "by_trigger_timeframe": _group(trades, lambda trade: _reason_tags(str(trade.get("entry_reason", ""))).get("trigger_tf", "unknown")),
        "by_exit_reason": dict(Counter(str(trade.get("exit_reason", "")) for trade in trades)),
        "top5_profit_contribution_pct": (
            sum(float(trade.get("net_pnl", 0.0)) for trade in top_five) / positive_pnl * 100.0
            if positive_pnl
            else None
        ),
        "exclude_top5": _metrics([trade for trade in trades if trade not in top_five]),
        "top_symbol": top_symbol,
        "exclude_top_symbol": _metrics([trade for trade in trades if trade.get("symbol") != top_symbol]),
        "reject_stats": mtf_report.get("reject_stats", {}),
        "active_sleeves": {
            "mtf_reversal": has_mtf,
            "mtpc": has_mtpc,
        },
        "historical_test_only": True,
        "final_acceptance_requires_post_freeze_shadow": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an MTF high-timeframe full-cost report")
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
