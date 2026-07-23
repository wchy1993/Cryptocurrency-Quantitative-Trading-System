from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .models import Direction
from .volatility_breakout_optimize import Candidate
from .volatility_breakout_v4_research import V4MarketSnapshot
from .volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    breakout_v7_risk_multiplier,
)


VOLATILITY_BREAKOUT_V8_NAME = (
    "dual_thrust_volatility_breakout_v8_score_convex"
)


@dataclass(frozen=True)
class BreakoutV8ScoreAllocation:
    """Causal score-convex risk overlay on the frozen Breakout v7 signal.

    Every factor is applied to the v7 risk multiplier after the v7 confidence
    score is computed from information available at entry.  A policy with all
    factors set to one and ``minimum_score=0`` is exactly path-equivalent to
    Breakout v7, which keeps optimization and live rollout auditable.
    """

    minimum_score: int = 0
    score_0_long_factor: float = 1.0
    score_0_short_factor: float = 1.0
    score_1_long_factor: float = 1.0
    score_1_short_factor: float = 1.0
    score_2_long_factor: float = 1.0
    score_2_short_factor: float = 1.0
    score_3_long_factor: float = 1.0
    score_3_short_factor: float = 1.0
    score_4_long_factor: float = 1.0
    score_4_short_factor: float = 1.0
    score_5_long_factor: float = 1.0
    score_5_short_factor: float = 1.0
    maximum_adjusted_multiplier: float = 3.0

    def validate(self) -> None:
        if not 0 <= self.minimum_score <= 5:
            raise ValueError("Breakout v8 minimum score must be in [0, 5]")
        factors = (
            self.score_0_long_factor,
            self.score_0_short_factor,
            self.score_1_long_factor,
            self.score_1_short_factor,
            self.score_2_long_factor,
            self.score_2_short_factor,
            self.score_3_long_factor,
            self.score_3_short_factor,
            self.score_4_long_factor,
            self.score_4_short_factor,
            self.score_5_long_factor,
            self.score_5_short_factor,
        )
        if any(value < 0.0 for value in factors):
            raise ValueError("Breakout v8 score factors cannot be negative")
        if self.maximum_adjusted_multiplier <= 0.0:
            raise ValueError(
                "Breakout v8 maximum adjusted multiplier must be positive"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def factor(self, score: int, direction: Direction) -> float:
        bounded_score = min(max(int(score), 0), 5)
        side = "long" if direction == Direction.LONG else "short"
        return float(getattr(self, f"score_{bounded_score}_{side}_factor"))


def breakout_v8_risk_multiplier(
    candidate: Candidate,
    snapshot: V4MarketSnapshot | None,
    confidence: BreakoutV7ConfidencePolicy,
    allocation: BreakoutV8ScoreAllocation,
) -> tuple[float, int, str]:
    """Return adjusted multiplier, causal score, and an audit lane label."""

    allocation.validate()
    base_multiplier, score, _ = breakout_v7_risk_multiplier(
        candidate, snapshot, confidence
    )
    if base_multiplier <= 0.0 or score < allocation.minimum_score:
        return 0.0, score, "rejected"
    factor = allocation.factor(score, candidate.signal.direction)
    if factor <= 0.0:
        return 0.0, score, "rejected"
    adjusted = min(
        base_multiplier * factor,
        allocation.maximum_adjusted_multiplier,
    )
    side = "long" if candidate.signal.direction == Direction.LONG else "short"
    return adjusted, score, f"score_{score}_{side}"
