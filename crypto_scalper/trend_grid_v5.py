from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .trend_grid_optimize import GridCandidate
from .trend_grid_v4 import _grid_regime_score
from .volatility_breakout_v4_research import V4MarketSnapshot


TREND_GRID_V5_NAME = "dynamic_trend_following_grid_v5_confidence_managed"


@dataclass(frozen=True)
class GridV5ConfidencePolicy:
    """Point-in-time campaign confidence and tier-specific management."""

    min_quality_score: float = -999.0
    min_alignment_atr: float = -999.0
    min_extension_atr: float = -999.0
    max_extension_atr: float = 999.0
    max_volume_ratio: float = 999.0
    max_regime_score: float = 999.0
    strong_score: int = 7
    weak_score: int = -1
    strong_risk_multiplier: float = 1.0
    standard_risk_multiplier: float = 1.0
    weak_risk_multiplier: float = 1.0
    strong_target_spacing: float = 1.85
    standard_target_spacing: float = 1.85
    weak_target_spacing: float = 1.85
    strong_loss_limit_r: float = 0.70
    standard_loss_limit_r: float = 0.70
    weak_loss_limit_r: float = 0.70
    strong_max_campaign_minutes: int = 4_320
    standard_max_campaign_minutes: int = 4_320
    weak_max_campaign_minutes: int = 4_320
    maximum_campaign_risk_pct: float = 0.10
    reject_weak_tier: bool = False

    def validate(self) -> None:
        if self.min_extension_atr > self.max_extension_atr:
            raise ValueError("Grid v5 extension minimum exceeds maximum")
        if not 0 <= self.strong_score <= 7:
            raise ValueError("Grid v5 strong score must be in [0, 7]")
        if not -1 <= self.weak_score <= 6:
            raise ValueError("Grid v5 weak score must be in [-1, 6]")
        if self.weak_score >= self.strong_score:
            raise ValueError("Grid v5 weak score must be below strong score")
        multipliers = (
            self.strong_risk_multiplier,
            self.standard_risk_multiplier,
            self.weak_risk_multiplier,
        )
        if any(value <= 0.0 for value in multipliers):
            raise ValueError("Grid v5 risk multipliers must be positive")
        if self.maximum_campaign_risk_pct <= 0.0:
            raise ValueError("Grid v5 campaign risk ceiling must be positive")
        for value in (
            self.strong_target_spacing,
            self.standard_target_spacing,
            self.weak_target_spacing,
        ):
            if value <= 0.0:
                raise ValueError("Grid v5 target spacing must be positive")
        for value in (
            self.strong_loss_limit_r,
            self.standard_loss_limit_r,
            self.weak_loss_limit_r,
        ):
            if value < 0.0:
                raise ValueError("Grid v5 loss limit cannot be negative")
        for value in (
            self.strong_max_campaign_minutes,
            self.standard_max_campaign_minutes,
            self.weak_max_campaign_minutes,
        ):
            if value <= 0:
                raise ValueError("Grid v5 campaign duration must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def grid_v5_confidence_score(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    policy: GridV5ConfidencePolicy,
) -> int:
    policy.validate()
    signal = candidate.signal
    regime_score = (
        _grid_regime_score(snapshot, signal.direction)
        if snapshot is not None
        else 999.0
    )
    conditions = (
        signal.quality_score >= policy.min_quality_score,
        signal.alignment_atr >= policy.min_alignment_atr,
        signal.extension_atr >= policy.min_extension_atr,
        signal.extension_atr <= policy.max_extension_atr,
        signal.volume_ratio <= policy.max_volume_ratio,
        regime_score <= policy.max_regime_score,
    )
    return sum(int(condition) for condition in conditions)


def grid_v5_tier(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    policy: GridV5ConfidencePolicy,
) -> tuple[str, int]:
    score = grid_v5_confidence_score(candidate, snapshot, policy)
    if score >= policy.strong_score:
        return "strong", score
    if score <= policy.weak_score:
        return "weak", score
    return "standard", score
