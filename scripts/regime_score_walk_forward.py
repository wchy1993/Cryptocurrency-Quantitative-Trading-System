from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PERIODS = {
    "train": (datetime(2025, 7, 1), datetime(2025, 10, 1)),
    "validation": (datetime(2025, 10, 1), datetime(2026, 1, 1)),
    "test": (datetime(2026, 1, 1), datetime(2026, 3, 1)),
    "observed_audit": (datetime(2026, 3, 1), datetime(2026, 6, 12)),
}


def _timestamp(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(row.get("entry_time", row["timestamp"])))


def _period_rows(rows: list[dict[str, Any]], period: str) -> list[dict[str, Any]]:
    start, end = PERIODS[period]
    return [row for row in rows if start <= _timestamp(row) < end]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [float(row["full_cost_net_pnl"]) for row in rows]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    monthly: dict[str, float] = defaultdict(float)
    symbols: dict[str, float] = defaultdict(float)
    for row, value in zip(rows, pnl):
        monthly[_timestamp(row).strftime("%Y-%m")] += value
        symbols[str(row["symbol"])] += value
    positive_profit = sorted(wins, reverse=True)
    top5_contribution = sum(positive_profit[:5]) / gross_profit if gross_profit > 0 else None
    avg_win = gross_profit / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    return {
        "trade_count": len(rows),
        "win_rate_pct": 100.0 * len(wins) / len(rows) if rows else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "net_pnl": sum(pnl),
        "expectancy": sum(pnl) / len(rows) if rows else None,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "average_win_loss_ratio": avg_win / abs(avg_loss) if avg_win is not None and avg_loss else None,
        "positive_month_count": sum(value > 0 for value in monthly.values()),
        "month_count": len(monthly),
        "monthly_pnl": dict(sorted(monthly.items())),
        "top5_profit_contribution": top5_contribution,
        "profitable_symbol_count": sum(value > 0 for value in symbols.values()),
        "symbol_count": len(symbols),
    }


def _selection_score(metrics: dict[str, Any]) -> float:
    if metrics["trade_count"] < 20:
        return -math.inf
    pf = float(metrics["profit_factor"] or 0.0)
    expectancy = float(metrics["expectancy"] or 0.0)
    avg_loss = abs(float(metrics["average_loss"] or 0.0))
    expectancy_to_loss = expectancy / avg_loss if avg_loss else -1.0
    win_loss = float(metrics["average_win_loss_ratio"] or 0.0)
    positive_month_ratio = metrics["positive_month_count"] / max(1, metrics["month_count"])
    concentration = float(metrics["top5_profit_contribution"] or 1.0)
    return (
        min(pf, 3.0) * 0.35
        + expectancy_to_loss * 0.25
        + min(win_loss, 2.0) * 0.15
        + positive_month_ratio * 0.20
        - concentration * 0.05
    )


def _validation_passes(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["trade_count"] >= 20
        and float(metrics["profit_factor"] or 0.0) > 1.10
        and float(metrics["expectancy"] or 0.0) > 0.0
        and float(metrics["average_win_loss_ratio"] or 0.0) > 0.85
        and metrics["positive_month_count"] >= 2
        and float(metrics["top5_profit_contribution"] or 1.0) < 0.60
    )


def _test_passes(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["trade_count"] >= 15
        and float(metrics["profit_factor"] or 0.0) > 1.0
        and float(metrics["expectancy"] or 0.0) > 0.0
        and metrics["positive_month_count"] >= 1
    )


def _vbp_rules() -> list[dict[str, float]]:
    return [
        {"trend_score_min": trend, "score_gap_min": gap}
        for trend, gap in itertools.product((50.0, 60.0, 70.0, 80.0), (0.0, 10.0, 20.0, 30.0))
    ]


def _reversal_rules() -> list[dict[str, float]]:
    return [
        {"reversal_score_min": score, "reversal_gap_min": gap}
        for score, gap in itertools.product((40.0, 50.0, 60.0, 70.0, 80.0), (0.0, 10.0, 20.0, 30.0))
    ]


def _vbp_filter(rule: dict[str, float]) -> Callable[[dict[str, Any]], bool]:
    return lambda row: (
        float(row.get("trend_score") or 0.0) >= rule["trend_score_min"]
        and float(row.get("score_gap") or -100.0) >= rule["score_gap_min"]
    )


def _reversal_filter(rule: dict[str, float]) -> Callable[[dict[str, Any]], bool]:
    return lambda row: (
        float(row.get("reversal_score") or 0.0) >= rule["reversal_score_min"]
        and -float(row.get("score_gap") or 100.0) >= rule["reversal_gap_min"]
    )


def _evaluate_strategy(
    rows: list[dict[str, Any]],
    rules: list[dict[str, float]],
    predicate_factory: Callable[[dict[str, float]], Callable[[dict[str, Any]], bool]],
) -> dict[str, Any]:
    validation_rows = _period_rows(rows, "validation")
    experiments = []
    for rule in rules:
        predicate = predicate_factory(rule)
        metrics = _metrics([row for row in validation_rows if predicate(row)])
        experiments.append(
            {
                "rule": rule,
                "validation": metrics,
                "validation_passed": _validation_passes(metrics),
                "selection_score": _selection_score(metrics),
            }
        )
    eligible = [experiment for experiment in experiments if experiment["validation_passed"]]
    selected = max(eligible, key=lambda item: item["selection_score"]) if eligible else None
    result: dict[str, Any] = {
        "baseline": {period: _metrics(_period_rows(rows, period)) for period in PERIODS},
        "validation_experiments": experiments,
        "selected_rule": selected["rule"] if selected else None,
        "status": "validation_passed" if selected else "no_proven_edge",
    }
    if selected:
        predicate = predicate_factory(selected["rule"])
        selected_metrics = {
            period: _metrics([row for row in _period_rows(rows, period) if predicate(row)])
            for period in PERIODS
        }
        result["selected_rule_metrics"] = selected_metrics
        result["test_passed"] = _test_passes(selected_metrics["test"])
        if not result["test_passed"]:
            result["status"] = "failed_unseen_test"
    return result


def build(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        row
        for row in payload.get("alpha_candidate_diagnostics", [])
        if row.get("status") == "traded" and "full_cost_net_pnl" in row and row.get("trend_score") is not None
    ]
    vbp = [row for row in candidates if row.get("strategy") == "volume_breakout_pullback"]
    reversal = [row for row in candidates if row.get("strategy") == "indicator_reversal"]
    return {
        "method": {
            "selection_period": "validation",
            "test_used_for_selection": False,
            "periods": {name: {"start": start.isoformat(), "end_exclusive": end.isoformat()} for name, (start, end) in PERIODS.items()},
            "validation_requirements": {
                "minimum_trades": 20,
                "profit_factor": "> 1.10",
                "expectancy": "> 0",
                "average_win_loss_ratio": "> 0.85",
                "positive_months": ">= 2",
                "top5_profit_contribution": "< 0.60",
            },
        },
        "candidate_counts": dict(Counter(row["strategy"] for row in candidates)),
        "vbp": _evaluate_strategy(vbp, _vbp_rules(), _vbp_filter),
        "reversal": _evaluate_strategy(reversal, _reversal_rules(), _reversal_filter),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for path in args.inputs:
        candidates.extend(json.loads(path.read_text()).get("alpha_candidate_diagnostics", []))
    args.output.write_text(json.dumps(build({"alpha_candidate_diagnostics": candidates}), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
