from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _label(value: float, bounds: list[float]) -> str:
    for low, high in zip(bounds, bounds[1:]):
        if low <= value < high:
            low_text = "-inf" if math.isinf(low) else f"{low:g}"
            high_text = "inf" if math.isinf(high) else f"{high:g}"
            return f"[{low_text},{high_text})"
    return "missing"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traded = [row for row in rows if row.get("status") == "traded" and "full_cost_net_pnl" in row]
    wins = [float(row["full_cost_net_pnl"]) for row in traded if float(row["full_cost_net_pnl"]) > 0]
    losses = [float(row["full_cost_net_pnl"]) for row in traded if float(row["full_cost_net_pnl"]) <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    mfe = [float(row["mfe_60m_pct"]) for row in rows if row.get("mfe_60m_pct") is not None]
    mae = [float(row["mae_60m_pct"]) for row in rows if row.get("mae_60m_pct") is not None]
    return {
        "candidate_count": len(rows),
        "traded_count": len(traded),
        "win_rate_pct": 100.0 * len(wins) / len(traded) if traded else None,
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "net_pnl": sum(float(row["full_cost_net_pnl"]) for row in traded),
        "expectancy": sum(float(row["full_cost_net_pnl"]) for row in traded) / len(traded) if traded else None,
        "median_mfe_60m_pct": statistics.median(mfe) if mfe else None,
        "median_mae_60m_pct": statistics.median(mae) if mae else None,
    }


def _numeric_buckets(rows: list[dict[str, Any]], field: str, bounds: list[float]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        grouped.setdefault(_label(float(value), bounds), []).append(row)
    return [{"bucket": name, **_metrics(grouped[name])} for name in sorted(grouped)]


def _boolean_buckets(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(bool(row.get(field))), []).append(row)
    return [{"bucket": name, **_metrics(grouped[name])} for name in sorted(grouped)]


def build(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("alpha_candidate_diagnostics", []))
    vbp = [row for row in candidates if row.get("strategy") == "volume_breakout_pullback"]
    reversal = [row for row in candidates if row.get("strategy") == "indicator_reversal"]
    for row in reversal:
        row["kdj_min"] = min(float(row.get("kdj_k", 0.0)), float(row.get("kdj_d", 0.0)))
    return {
        "source_summary": {
            key: payload.get(key)
            for key in ("initial_equity", "final_equity", "net_pnl", "trade_count", "profit_factor", "max_drawdown_pct")
        },
        "candidate_count": len(candidates),
        "vbp": {
            "summary": _metrics(vbp),
            "status_distribution": dict(Counter(str(row.get("status")) for row in vbp)),
            "reject_distribution": dict(Counter(str(row.get("filter_reason")) for row in vbp if row.get("filter_reason"))),
            "by_breakout_distance_atr": _numeric_buckets(vbp, "breakout_distance_atr", [-math.inf, 0.25, 0.5, 1.0, 2.0, math.inf]),
            "by_breakout_body_atr": _numeric_buckets(vbp, "breakout_body_atr", [-math.inf, 0.25, 0.5, 1.0, 2.0, math.inf]),
            "by_breakout_range_atr": _numeric_buckets(vbp, "breakout_range_atr", [-math.inf, 0.5, 1.0, 1.5, 2.5, math.inf]),
            "by_breakout_volume_ratio": _numeric_buckets(vbp, "breakout_volume_ratio", [-math.inf, 3.0, 4.0, 6.0, 10.0, math.inf]),
            "by_breakout_close_position": _numeric_buckets(vbp, "breakout_close_position", [-math.inf, 0.60, 0.75, 0.90, math.inf]),
            "by_upper_wick_ratio": _numeric_buckets(vbp, "upper_wick_ratio", [-math.inf, 0.10, 0.25, 0.40, math.inf]),
            "by_atr_percentile": _numeric_buckets(vbp, "pre_breakout_atr_percentile", [-math.inf, 0.20, 0.40, 0.60, 0.80, math.inf]),
            "by_compression_atr": _numeric_buckets(vbp, "pre_breakout_range_compression_atr", [-math.inf, 2.0, 4.0, 6.0, 10.0, math.inf]),
            "by_volume_contraction": _numeric_buckets(vbp, "pre_breakout_volume_contraction", [-math.inf, 0.6, 0.8, 1.0, 1.2, math.inf]),
            "by_failed_breakouts": _numeric_buckets(vbp, "previous_failed_breakout_count", [-math.inf, 1.0, 2.0, 3.0, 5.0, math.inf]),
            "by_pullback_depth_atr": _numeric_buckets(vbp, "pullback_depth_atr", [-math.inf, 0.50, 1.0, 1.5, 2.5, math.inf]),
            "by_pullback_depth_ratio": _numeric_buckets(vbp, "pullback_depth_to_breakout_distance", [-math.inf, 0.5, 1.0, 2.0, 4.0, math.inf]),
            "by_pullback_volume_ratio": _numeric_buckets(vbp, "pullback_volume_to_breakout_volume", [-math.inf, 0.20, 0.30, 0.40, 0.60, math.inf]),
            "by_pullback_bars": _numeric_buckets(vbp, "pullback_bars", [-math.inf, 2.0, 5.0, 10.0, 15.0, math.inf]),
            "by_confirmation_body_atr": _numeric_buckets(vbp, "confirmation_body_atr", [-math.inf, 0.25, 0.5, 1.0, 2.0, math.inf]),
            "by_confirmation_close_position": _numeric_buckets(vbp, "confirmation_close_position", [-math.inf, 0.60, 0.75, 0.90, math.inf]),
            "by_confirmation_wick_ratio": _numeric_buckets(vbp, "confirmation_wick_ratio", [-math.inf, 0.10, 0.25, 0.40, math.inf]),
            "by_confirmation_volume_ratio": _numeric_buckets(vbp, "confirmation_volume_ratio", [-math.inf, 0.20, 0.30, 0.40, 0.60, math.inf]),
            "by_entry_chase_distance_atr": _numeric_buckets(vbp, "entry_chase_distance_atr", [-math.inf, 0.25, 0.50, 1.0, 2.0, math.inf]),
            "by_target_to_full_cost_ratio": _numeric_buckets(vbp, "target_to_full_cost_ratio", [-math.inf, 3.0, 4.0, 5.0, 6.0, math.inf]),
            "by_stop_to_full_cost_ratio": _numeric_buckets(vbp, "stop_to_full_cost_ratio", [-math.inf, 3.0, 5.0, 8.0, 12.0, math.inf]),
        },
        "reversal": {
            "summary": _metrics(reversal),
            "status_distribution": dict(Counter(str(row.get("status")) for row in reversal)),
            "by_rsi14": _numeric_buckets(reversal, "rsi14", [-math.inf, 30.0, 35.0, 40.0, 45.0, 50.0, math.inf]),
            "by_kdj_min": _numeric_buckets(reversal, "kdj_min", [-math.inf, 20.0, 25.0, 30.0, 35.0, 45.0, math.inf]),
            "by_extension_ema21_atr": _numeric_buckets(reversal, "price_extension_ema21_atr", [-math.inf, -2.0, -1.0, -0.5, 0.0, 0.5, math.inf]),
            "by_macd_histogram_change": _numeric_buckets(reversal, "macd_histogram_change", [-math.inf, 0.0, math.inf]),
            "by_target_to_full_cost_ratio": _numeric_buckets(reversal, "target_to_full_cost_ratio", [-math.inf, 3.0, 4.0, 5.0, 6.0, math.inf]),
            "by_stop_to_full_cost_ratio": _numeric_buckets(reversal, "stop_to_full_cost_ratio", [-math.inf, 3.0, 5.0, 8.0, 12.0, math.inf]),
            "by_reclaim_ema9": _boolean_buckets(reversal, "reclaim_ema9"),
            "by_reclaim_ema21": _boolean_buckets(reversal, "reclaim_ema21"),
            "by_no_new_low": _boolean_buckets(reversal, "no_new_low_3"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
