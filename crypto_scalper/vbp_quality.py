from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import Candle


@dataclass(frozen=True)
class VbpQualityInputs:
    rvol: float
    rvol_threshold: float
    relative_vs_btc: float
    relative_rank_pct: float | None
    consolidation_range_pct: float
    consolidation_threshold_pct: float
    breakout_close_position: float
    breakout_upper_wick_ratio: float
    pullback_volume_ratio: float
    pullback_depth_pct: float
    reclaim_strength_pct: float
    momentum_rank_score: float = 0.5
    capacity_fill_ratio: float = 1.0


@dataclass(frozen=True)
class VbpQualityDecision:
    score: float
    tier: str
    risk_multiplier: float
    tp1_close_ratio: float
    runner_ratio: float
    allowed: bool


def _score_vbp_quality_absolute(inputs: VbpQualityInputs, entry_config: Any) -> VbpQualityDecision:
    rvol = _clamp(inputs.rvol / max(inputs.rvol_threshold * 2.0, 1e-12))
    relative = _clamp((inputs.relative_vs_btc + 0.002) / 0.022)
    market_rank = 0.5 if inputs.relative_rank_pct is None else _clamp(1.0 - inputs.relative_rank_pct)
    compression = _clamp(1.0 - inputs.consolidation_range_pct / max(inputs.consolidation_threshold_pct, 1e-12))
    close_position = _clamp(inputs.breakout_close_position)
    wick = _clamp(1.0 - inputs.breakout_upper_wick_ratio / 0.50)
    pullback_volume = _clamp(1.0 - inputs.pullback_volume_ratio)
    pullback_depth = _clamp(1.0 - inputs.pullback_depth_pct / 0.03)
    reclaim = _clamp(inputs.reclaim_strength_pct / 0.012)
    momentum_rank = _clamp(inputs.momentum_rank_score)
    capacity = _clamp(inputs.capacity_fill_ratio)
    score = 100.0 * (
        0.18 * rvol
        + 0.12 * relative
        + 0.08 * market_rank
        + 0.12 * compression
        + 0.10 * close_position
        + 0.06 * wick
        + 0.12 * pullback_volume
        + 0.06 * pullback_depth
        + 0.08 * reclaim
        + 0.04 * momentum_rank
        + 0.04 * capacity
    )
    minimum = float(getattr(entry_config, "quality_min_score", 45.0))
    a_score = float(getattr(entry_config, "quality_a_score", 60.0))
    a_plus_score = float(getattr(entry_config, "quality_a_plus_score", 75.0))
    if score >= a_plus_score:
        tier = "A+"
        risk = float(getattr(entry_config, "quality_a_plus_risk_multiplier", 1.20))
        tp1 = float(getattr(entry_config, "quality_a_plus_tp1_close_ratio", 0.25))
    elif score >= a_score:
        tier = "A"
        risk = float(getattr(entry_config, "quality_a_risk_multiplier", 0.80))
        tp1 = float(getattr(entry_config, "quality_a_tp1_close_ratio", 0.40))
    elif score >= minimum:
        tier = "B"
        risk = float(getattr(entry_config, "quality_b_risk_multiplier", 0.40))
        tp1 = float(getattr(entry_config, "quality_b_tp1_close_ratio", 0.50))
    else:
        tier = "REJECT"
        risk = 0.0
        tp1 = 1.0
    tp1 = _clamp(tp1, 0.05, 0.95)
    return VbpQualityDecision(score, tier, max(0.0, risk), tp1, 1.0 - tp1, tier != "REJECT")


def vbp_runner_exit_reason(
    exit_config: Any,
    current_profit: float,
    peak_profit: float,
    tp1_done: bool,
    minimum_profit: float,
    atr_pct_5m: float,
    large_bear_reason: str | None,
) -> str | None:
    pullback = max(0.0, peak_profit - current_profit)
    min_retrace = max(0.0, float(getattr(exit_config, "runner_min_retrace_pct", 0.0045)))
    max_retrace = max(min_retrace, float(getattr(exit_config, "runner_max_retrace_pct", 0.03)))
    atr_multiple = max(0.0, float(getattr(exit_config, "runner_atr_multiplier", 1.20)))
    retrace = min(max_retrace, max(min_retrace, atr_pct_5m * atr_multiple))
    trigger = max(0.0, float(getattr(exit_config, "peak_giveback_trigger_pct", 0.012)))
    floor = float(getattr(exit_config, "peak_giveback_floor_pct", 0.0025))
    pre_tp1_trigger = max(0.0, float(getattr(exit_config, "peak_giveback_pre_tp1_trigger_pct", 0.020)))
    if bool(getattr(exit_config, "peak_giveback_enabled", True)) and current_profit >= minimum_profit:
        post_tp1 = tp1_done and peak_profit >= trigger and (current_profit <= floor or pullback >= retrace)
        pre_tp1 = not tp1_done and peak_profit >= pre_tp1_trigger and pullback >= retrace
        if post_tp1 or pre_tp1:
            tag = "vbp_peak_giveback" if tp1_done else "vbp_peak_giveback_pre_tp1"
            return f"{tag} peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}% pullback={pullback * 100:.3f}% trail={retrace * 100:.3f}%"
    if large_bear_reason:
        return f"vbp_high_volume_bear_exit {large_bear_reason} peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}%"
    return None


def vbp_atr_pct_5m(candles: list[Candle], period: int = 14) -> float:
    groups = [candles[index:index + 5] for index in range(0, len(candles), 5)]
    bars = [group for group in groups if len(group) == 5]
    if len(bars) < period + 1:
        return 0.0
    merged = [
        (group[0].open, max(item.high for item in group), min(item.low for item in group), group[-1].close)
        for group in bars[-period - 1:]
    ]
    ranges = []
    for index in range(1, len(merged)):
        _, high, low, _ = merged[index]
        previous_close = merged[index - 1][3]
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    close = merged[-1][3]
    return sum(ranges[-period:]) / max(len(ranges[-period:]), 1) / max(close, 1e-12)


def _vbp_quality_report_legacy(trades: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in trades if str(row.get("strategy", "")) == "volume_breakout_pullback"]
    final_rows = [row for row in rows if row.get("exit_reason") != "vbp_tp1_partial"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in final_rows:
        tier = _reason_token(str(row.get("entry_reason", "")), "quality_tier") or "unscored"
        groups.setdefault(tier, []).append(row)
    by_tier = []
    for tier in ("A+", "A", "B", "unscored"):
        items = groups.get(tier, [])
        if not items:
            continue
        wins = [item for item in items if float(item.get("net_pnl", 0.0)) > 0]
        losses = [item for item in items if float(item.get("net_pnl", 0.0)) <= 0]
        gross_win = sum(float(item.get("net_pnl", 0.0)) for item in wins)
        gross_loss = abs(sum(float(item.get("net_pnl", 0.0)) for item in losses))
        mfe = sum(max(0.0, float(item.get("mfe", 0.0))) for item in items)
        net = sum(float(item.get("net_pnl", 0.0)) for item in items)
        by_tier.append({
            "tier": tier,
            "positions": len(items),
            "net_pnl": net,
            "win_rate_pct": len(wins) / len(items) * 100.0,
            "profit_factor": None if gross_loss == 0 else gross_win / gross_loss,
            "mfe_capture_ratio": None if mfe <= 0 else net / mfe,
        })
    positive = sorted((float(item.get("net_pnl", 0.0)) for item in final_rows if float(item.get("net_pnl", 0.0)) > 0), reverse=True)
    top_count = max(1, math.ceil(len(positive) * 0.10)) if positive else 0
    total_positive = sum(positive)
    partial_pnl = sum(float(item.get("net_pnl", 0.0)) for item in rows if item.get("exit_reason") == "vbp_tp1_partial")
    runner_rows = [item for item in final_rows if "vbp_tp1_done=1" in str(item.get("entry_reason", ""))]
    return {
        "by_quality_tier": by_tier,
        "top_10pct_winner_contribution_pct": 0.0 if total_positive <= 0 else sum(positive[:top_count]) / total_positive * 100.0,
        "tp1_partial_net_pnl": partial_pnl,
        "runner_final_net_pnl": sum(float(item.get("net_pnl", 0.0)) for item in runner_rows),
        "runner_positions": len(runner_rows),
        "runner_turned_loss_count": sum(1 for item in runner_rows if float(item.get("net_pnl", 0.0)) < 0),
    }


def _reason_token(reason: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", reason)
    return match.group(1) if match else None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))

# Rolling classification deliberately lives in the shared strategy module so live
# trading and backtests use identical, look-ahead-free quality tiers.
_ROLLING_QUALITY_HISTORY: dict[int, list[float]] = {}
_ROLLING_QUALITY_WINDOW = 500
_ROLLING_QUALITY_WARMUP = 100
_ROLLING_A_QUANTILE = 0.35
_ROLLING_A_PLUS_QUANTILE = 0.60
_ROLLING_OVERHEATED_QUANTILE = 0.85
_WARMUP_A_SCORE = 60.0
_WARMUP_A_PLUS_SCORE = 68.0
_WARMUP_OVERHEATED_SCORE = 75.0


def _historical_quality_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _replace_quality_decision(decision: VbpQualityDecision, **updates: Any) -> VbpQualityDecision:
    from dataclasses import fields, replace

    available = {item.name for item in fields(decision)}
    return replace(decision, **{key: value for key, value in updates.items() if key in available})


def score_vbp_quality(inputs: VbpQualityInputs, entry_config: Any) -> VbpQualityDecision:
    decision = _score_vbp_quality_absolute(inputs, entry_config)
    if not bool(getattr(entry_config, "quality_score_enabled", False)):
        return decision

    key = id(entry_config)
    history = _ROLLING_QUALITY_HISTORY.setdefault(key, [])
    score = float(decision.score)

    # During warmup use fixed bands derived from the first calibration run.
    # Afterwards classify only against scores observed before the signal.
    if len(history) < _ROLLING_QUALITY_WARMUP:
        a_cutoff = _WARMUP_A_SCORE
        a_plus_cutoff = _WARMUP_A_PLUS_SCORE
        overheated_cutoff = _WARMUP_OVERHEATED_SCORE
    else:
        a_cutoff = _historical_quality_quantile(history, _ROLLING_A_QUANTILE)
        a_plus_cutoff = max(a_cutoff, _historical_quality_quantile(history, _ROLLING_A_PLUS_QUANTILE))
        overheated_cutoff = max(
            a_plus_cutoff,
            _historical_quality_quantile(history, _ROLLING_OVERHEATED_QUANTILE),
        )

    if score >= overheated_cutoff:
        tier = "overheated"
        accepted = False
        risk_multiplier = 0.0
        tp1_ratio = float(getattr(entry_config, "quality_a_plus_tp1_partial_ratio", 0.20))
    elif score >= a_plus_cutoff:
        tier = "A+"
        accepted = True
        risk_multiplier = float(getattr(entry_config, "quality_a_plus_risk_multiplier", 1.15))
        tp1_ratio = float(getattr(entry_config, "quality_a_plus_tp1_partial_ratio", 0.20))
    elif score >= a_cutoff:
        tier = "A"
        accepted = True
        risk_multiplier = float(getattr(entry_config, "quality_a_risk_multiplier", 0.80))
        tp1_ratio = float(getattr(entry_config, "quality_a_tp1_partial_ratio", 0.35))
    else:
        tier = "B"
        accepted = False
        risk_multiplier = 0.0
        tp1_ratio = float(getattr(entry_config, "quality_b_tp1_partial_ratio", 0.50))

    history.append(score)
    if len(history) > _ROLLING_QUALITY_WINDOW:
        del history[:-_ROLLING_QUALITY_WINDOW]

    return _replace_quality_decision(
        decision,
        tier=tier,
        accepted=accepted,
        allowed=accepted,
        eligible=accepted,
        risk_multiplier=risk_multiplier,
        tp1_partial_ratio=tp1_ratio,
        rejection_reason=(
            "" if accepted else "quality_percentile_overheated" if tier == "overheated" else "quality_percentile_b"
        ),
    )


def _mfe_segment_capture(rows: list[dict[str, Any]]) -> dict[str, Any]:
    maximum_achievable = sum(max(0.0, float(row.get("mfe_max_pnl", row.get("mfe", 0.0)))) for row in rows)
    realized = sum(
        max(0.0, float(row.get("mfe_realized_favorable_pnl", row.get("gross_pnl", 0.0))))
        for row in rows
    )
    realized_net = sum(float(row.get("net_pnl", 0.0)) for row in rows)
    realized_gross = sum(float(row.get("gross_pnl", 0.0)) for row in rows)
    ratio = None if maximum_achievable <= 0 else realized / maximum_achievable
    return {
        "segments": len(rows),
        "realized_favorable_pnl": realized,
        "realized_gross_pnl": realized_gross,
        "realized_net_pnl": realized_net,
        "cost_and_funding_drag": realized_gross - realized_net,
        "maximum_achievable_pnl": maximum_achievable,
        "capture_ratio": ratio,
        "capture_pct": None if ratio is None else ratio * 100.0,
        "valid_range": ratio is None or ratio <= 1.000000001,
    }


def vbp_quality_report(trades: list[dict[str, Any]]) -> dict[str, Any]:
    report = _vbp_quality_report_legacy(trades)
    rows = [row for row in trades if str(row.get("strategy", "")) == "volume_breakout_pullback"]
    tp1_rows = [row for row in rows if row.get("exit_reason") == "vbp_tp1_partial"]
    final_rows = [row for row in rows if row.get("exit_reason") != "vbp_tp1_partial"]
    runner_rows = [row for row in final_rows if "vbp_tp1_done=1" in str(row.get("entry_reason", ""))]

    by_tier_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tier = _reason_token(str(row.get("entry_reason", "")), "quality_tier") or "unscored"
        by_tier_rows.setdefault(tier, []).append(row)

    for item in report.get("by_quality_tier", []):
        tier = str(item.get("tier", "unscored"))
        tier_rows = by_tier_rows.get(tier, [])
        tier_tp1 = [row for row in tier_rows if row.get("exit_reason") == "vbp_tp1_partial"]
        tier_runner = [
            row for row in tier_rows
            if row.get("exit_reason") != "vbp_tp1_partial"
            and "vbp_tp1_done=1" in str(row.get("entry_reason", ""))
        ]
        item.pop("mfe_capture_ratio", None)
        item["all_segment_mfe_capture"] = _mfe_segment_capture(tier_rows)
        item["tp1_mfe_capture"] = _mfe_segment_capture(tier_tp1)
        item["runner_mfe_capture"] = _mfe_segment_capture(tier_runner)

    report["mfe_capture"] = {
        "definition": "sum(segment favorable raw realized pnl) / sum(segment raw notional * raw-price MFE return); costs reported separately",
        "all_segments": _mfe_segment_capture(rows),
        "tp1": _mfe_segment_capture(tp1_rows),
        "runner": _mfe_segment_capture(runner_rows),
    }
    return report
