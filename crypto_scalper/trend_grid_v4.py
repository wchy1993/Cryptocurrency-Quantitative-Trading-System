from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import Direction
from .trend_grid_optimize import GridCandidate
from .volatility_breakout_v4_research import V4MarketSnapshot


TREND_GRID_V4_NAME = "dynamic_trend_following_grid_v4_live_robust"


@dataclass(frozen=True)
class GridV4EntryGate:
    """Post-signal point-in-time quality gate for Grid campaigns."""

    min_quality_score: float = -999.0
    max_quality_score: float = 999.0
    min_alignment_atr: float = -999.0
    max_alignment_atr: float = 999.0
    min_extension_atr: float = -999.0
    max_extension_atr: float = 999.0
    min_volume_ratio: float = 0.0
    max_volume_ratio: float = 999.0
    min_directional_fast_slope_atr: float = -999.0
    max_directional_fast_slope_atr: float = 999.0
    min_directional_slow_slope_atr: float = -999.0
    max_directional_slow_slope_atr: float = 999.0
    min_market_efficiency_12h: float = 0.0
    min_directional_breadth: float = 0.0
    max_directional_breadth: float = 1.0
    min_regime_score: float = -999.0
    max_regime_score: float = 999.0
    require_market_context: bool = True

    def validate(self) -> None:
        pairs = (
            (self.min_quality_score, self.max_quality_score),
            (self.min_alignment_atr, self.max_alignment_atr),
            (self.min_extension_atr, self.max_extension_atr),
            (self.min_volume_ratio, self.max_volume_ratio),
            (
                self.min_directional_fast_slope_atr,
                self.max_directional_fast_slope_atr,
            ),
            (
                self.min_directional_slow_slope_atr,
                self.max_directional_slow_slope_atr,
            ),
            (self.min_directional_breadth, self.max_directional_breadth),
            (self.min_regime_score, self.max_regime_score),
        )
        if any(low > high for low, high in pairs):
            raise ValueError("Grid v4 entry-gate minimum exceeds maximum")
        if not 0.0 <= self.min_directional_breadth <= 1.0:
            raise ValueError("minimum directional breadth must be in [0, 1]")
        if not 0.0 <= self.max_directional_breadth <= 1.0:
            raise ValueError("maximum directional breadth must be in [0, 1]")
        if not 0.0 <= self.min_market_efficiency_12h <= 1.0:
            raise ValueError("minimum market efficiency must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _grid_regime_score(
    snapshot: V4MarketSnapshot, direction: Direction
) -> float:
    side = float(direction.value)

    def clipped(value: float, scale: float) -> float:
        return max(-2.0, min(2.0, value / scale))

    directional_breadth = (
        snapshot.breadth_above_ema21
        if direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    return (
        0.18 * clipped(side * snapshot.btc_return_4h, 0.01)
        + 0.14 * clipped(side * snapshot.eth_return_4h, 0.012)
        + 0.18 * clipped(directional_breadth - 0.5, 0.20)
        + 0.12 * clipped(side * snapshot.breadth_change_4h, 0.10)
        + 0.18 * clipped(side * snapshot.symbol_return_4h, 0.02)
        + 0.12 * clipped(side * snapshot.symbol_efficiency_12h, 0.25)
        + 0.08 * clipped(side * snapshot.symbol_ema55_atr, 2.0)
    )


def passes_grid_v4_entry(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    gate: GridV4EntryGate,
) -> bool:
    signal = candidate.signal
    if signal.direction not in {Direction.LONG, Direction.SHORT}:
        return False
    if snapshot is None:
        return not gate.require_market_context
    side = float(signal.direction.value)
    directional_breadth = (
        snapshot.breadth_above_ema21
        if signal.direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    directional_fast_slope = side * signal.fast_slope_atr
    directional_slow_slope = side * signal.slow_slope_atr
    regime_score = _grid_regime_score(snapshot, signal.direction)
    return (
        gate.min_quality_score <= signal.quality_score <= gate.max_quality_score
        and gate.min_alignment_atr
        <= signal.alignment_atr
        <= gate.max_alignment_atr
        and gate.min_extension_atr
        <= signal.extension_atr
        <= gate.max_extension_atr
        and gate.min_volume_ratio <= signal.volume_ratio <= gate.max_volume_ratio
        and gate.min_directional_fast_slope_atr
        <= directional_fast_slope
        <= gate.max_directional_fast_slope_atr
        and gate.min_directional_slow_slope_atr
        <= directional_slow_slope
        <= gate.max_directional_slow_slope_atr
        and snapshot.market_efficiency_12h >= gate.min_market_efficiency_12h
        and gate.min_directional_breadth
        <= directional_breadth
        <= gate.max_directional_breadth
        and gate.min_regime_score <= regime_score <= gate.max_regime_score
    )


def filter_grid_v4_candidates(
    candidates: dict[int, list[GridCandidate]],
    context: dict[int, dict[str, V4MarketSnapshot]],
    gate: GridV4EntryGate,
) -> dict[int, list[GridCandidate]]:
    gate.validate()
    output: dict[int, list[GridCandidate]] = {}
    for minute in sorted(candidates):
        snapshots = context.get(minute - minute % 60, {})
        selected = [
            candidate
            for candidate in candidates[minute]
            if passes_grid_v4_entry(
                candidate,
                snapshots.get(candidate.signal.symbol),
                gate,
            )
        ]
        if selected:
            output[minute] = selected
    return output
