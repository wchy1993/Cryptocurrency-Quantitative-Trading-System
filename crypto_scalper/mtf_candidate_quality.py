from __future__ import annotations

import hashlib
import json
import math
import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any, Callable, Iterable

from .models import Direction
from .models import Candle
from .risk import BacktestExecutionConfig, SymbolRules, market_exit_fill


MTF_CANDIDATE_FEATURE_VERSION = "mtf_candidate_quality_v1"


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def mtf_candidate_event_id(symbol: str, direction: Direction | str, trigger_time: datetime) -> str:
    side = direction.name if isinstance(direction, Direction) else str(direction).upper()
    raw = f"{MTF_CANDIDATE_FEATURE_VERSION}|{symbol.upper()}|{side}|{trigger_time.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def next_1m_execution_index(timestamps: list[datetime], signal_available_time: datetime) -> int | None:
    index = bisect.bisect_left(timestamps, signal_available_time + timedelta(minutes=1))
    return index if index < len(timestamps) else None


def can_be_structure_break(candles: list[Candle], index: int, lookback: int) -> bool:
    """Cheap necessary condition for either direction's frozen structure-break trigger."""
    if index < lookback or lookback < 2:
        return False
    latest = candles[index]
    previous = candles[index - lookback:index]
    long_break = latest.low >= min(item.low for item in previous) and latest.close > max(item.high for item in previous)
    short_break = latest.high <= max(item.high for item in previous) and latest.close < min(item.low for item in previous)
    return long_break or short_break


@dataclass(frozen=True)
class MtfCandidateQuality:
    score: float
    rank_component: float
    close_component: float
    body_component: float
    btc_component: float
    one_hour_rsi_component: float
    cost_space_component: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "quality_feature_version": MTF_CANDIDATE_FEATURE_VERSION,
            "quality_score_v1": self.score,
            "quality_rank_component": self.rank_component,
            "quality_close_component": self.close_component,
            "quality_body_component": self.body_component,
            "quality_btc_component": self.btc_component,
            "quality_one_hour_rsi_component": self.one_hour_rsi_component,
            "quality_cost_space_component": self.cost_space_component,
        }


def mtf_candidate_quality(metadata: dict[str, Any], direction: Direction | str) -> MtfCandidateQuality:
    """Point-in-time diagnostic score; shadow outcomes are intentionally excluded."""
    side = direction.value if isinstance(direction, Direction) else (1 if str(direction).upper() == "LONG" else -1)
    rank = _finite(metadata.get("rank_score"))
    close_position = _clip(_finite(metadata.get("trigger_close_position"), 0.5))
    directional_close = close_position if side > 0 else 1.0 - close_position
    body_atr = max(0.0, _finite(metadata.get("trigger_body_atr")))
    btc_1h = _finite(metadata.get("btc_1h_return")) * side
    btc_4h = _finite(metadata.get("btc_4h_return")) * side
    rsi_1h = _finite(metadata.get("1h_rsi"), 50.0)
    directional_rsi = 50.0 + side * (rsi_1h - 50.0)
    target_to_cost = max(0.0, _finite(metadata.get("target_to_cost_ratio")))

    rank_component = _clip((rank + 1.0) / 7.0)
    close_component = _clip((directional_close - 0.5) / 0.4)
    body_component = _clip(body_atr / 1.2)
    btc_component = _clip(0.5 + btc_1h / 0.012 * 0.3 + btc_4h / 0.04 * 0.2)
    one_hour_rsi_component = _clip((directional_rsi - 42.0) / 24.0)
    cost_space_component = _clip((target_to_cost - 3.0) / 9.0)

    score = 100.0 * (
        0.25 * rank_component
        + 0.20 * close_component
        + 0.15 * body_component
        + 0.10 * btc_component
        + 0.10 * one_hour_rsi_component
        + 0.20 * cost_space_component
    )
    return MtfCandidateQuality(
        score=score,
        rank_component=rank_component,
        close_component=close_component,
        body_component=body_component,
        btc_component=btc_component,
        one_hour_rsi_component=one_hour_rsi_component,
        cost_space_component=cost_space_component,
    )


def full_cost_stop_risk_usdt(
    position: Any,
    execution_config: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    stop = float(position.initial_stop_price or position.stop_price)
    raw_entry = float(position.raw_entry_price or position.entry_price)
    exit_fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        stop,
        "stop_market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_price_pnl = position.direction.value * position.quantity * (stop - raw_entry)
    net_stop_pnl = (
        raw_price_pnl
        - position.entry_fee
        - exit_fill.fee
        - position.entry_slippage_cost
        - exit_fill.slippage_cost
    )
    return max(1e-12, -net_stop_pnl)


def executable_mark_pnl_usdt(
    position: Any,
    raw_exit_price: float,
    execution_config: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    exit_fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        raw_exit_price,
        "market",
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = float(position.raw_entry_price or position.entry_price)
    raw_price_pnl = position.direction.value * position.quantity * (raw_exit_price - raw_entry)
    return (
        raw_price_pnl
        - position.entry_fee
        - exit_fill.fee
        - position.entry_slippage_cost
        - exit_fill.slippage_cost
    )


def candidate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    r_values = [_finite(row.get("shadow_net_r")) for row in items]
    wins = [value for value in r_values if value > 0.0]
    losses = [value for value in r_values if value <= 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "count": len(items),
        "expectancy_r": sum(r_values) / len(r_values) if r_values else 0.0,
        "median_r": _percentile(r_values, 0.5),
        "profit_factor_r": gross_profit / gross_loss if gross_loss else ("Infinity" if gross_profit else 0.0),
        "win_rate_pct": len(wins) / len(items) * 100.0 if items else 0.0,
        "hit_0p5r_pct": _rate(items, "hit_0p5r"),
        "hit_1r_pct": _rate(items, "hit_1r"),
        "hit_2r_pct": _rate(items, "hit_2r"),
        "stop_first_pct": _rate(items, "stop_first"),
        "average_cost_r": sum(_finite(row.get("shadow_cost_r")) for row in items) / len(items) if items else 0.0,
        "long_count": sum(1 for row in items if str(row.get("side")) == "LONG"),
        "short_count": sum(1 for row in items if str(row.get("side")) == "SHORT"),
    }


def quantile_bucket_report(
    rows: Iterable[dict[str, Any]],
    field: str,
    buckets: int = 5,
    min_sample: int = 20,
) -> list[dict[str, Any]]:
    usable = [row for row in rows if math.isfinite(_finite(row.get(field), float("nan")))]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda row: _finite(row.get(field)))
    output: list[dict[str, Any]] = []
    for bucket_index in range(buckets):
        start = len(ordered) * bucket_index // buckets
        end = len(ordered) * (bucket_index + 1) // buckets
        bucket_rows = ordered[start:end]
        if not bucket_rows:
            continue
        values = [_finite(row.get(field)) for row in bucket_rows]
        output.append(
            {
                "bucket": bucket_index + 1,
                "minimum": min(values),
                "maximum": max(values),
                "sufficient_sample": len(bucket_rows) >= min_sample,
                **candidate_metrics(bucket_rows),
            }
        )
    return output


def grouped_report(
    rows: Iterable[dict[str, Any]],
    key: str | Callable[[dict[str, Any]], str],
    min_sample: int = 20,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    getter = key if callable(key) else lambda row: str(row.get(key, "missing"))
    for row in rows:
        grouped[getter(row)].append(row)
    return {
        name: {"sufficient_sample": len(group) >= min_sample, **candidate_metrics(group)}
        for name, group in sorted(grouped.items())
    }


def config_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * _clip(percentile)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row.get(field)) for row in rows) / len(rows) * 100.0 if rows else 0.0
