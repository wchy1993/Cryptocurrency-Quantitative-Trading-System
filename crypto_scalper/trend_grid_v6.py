from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .trend_grid_optimize import GridCandidate
from .trend_grid_v5 import GridV5ConfidencePolicy, grid_v5_tier
from .volatility_breakout_v4_research import V4MarketSnapshot


TREND_GRID_V6_NAME = "dynamic_trend_following_grid_v6_profit_protected"


@dataclass(frozen=True)
class GridV6CampaignPolicy:
    """Causal entry guard, score allocation, and campaign protection for v5.

    Defaults are deliberately path-equivalent to Grid v5.  All entry guards
    use fields fixed when the campaign is opened, while profit protection uses
    only realized and mark-to-market campaign PnL observed up to the current
    minute.
    """

    minimum_score: int = 0
    minimum_actual_extension_atr: float = -999.0
    maximum_actual_extension_atr: float = 999.0
    maximum_actual_volume_ratio: float = 999.0
    score_0_risk_factor: float = 1.0
    score_1_risk_factor: float = 1.0
    score_2_risk_factor: float = 1.0
    score_3_risk_factor: float = 1.0
    score_4_risk_factor: float = 1.0
    score_5_risk_factor: float = 1.0
    score_6_risk_factor: float = 1.0
    maximum_campaign_risk_pct: float = 0.11
    maximum_notional_multiple: float = 0.0
    target_spacing: float = 1.85
    campaign_loss_limit_r: float = 0.70
    max_campaign_minutes: int = 4_320
    campaign_take_profit_r: float = 0.0
    profit_lock_activation_r: float = 0.0
    profit_giveback_r: float = 0.0
    cycle_profit_floor_min_take_profits: int = 0
    cycle_profit_floor_activation_r: float = 0.0
    cycle_profit_floor_r: float = 0.0

    def validate(self) -> None:
        if not 0 <= self.minimum_score <= 6:
            raise ValueError("Grid v6 minimum score must be in [0, 6]")
        if self.minimum_actual_extension_atr > self.maximum_actual_extension_atr:
            raise ValueError("Grid v6 extension minimum exceeds maximum")
        if self.maximum_actual_volume_ratio <= 0.0:
            raise ValueError("Grid v6 maximum volume ratio must be positive")
        factors = tuple(
            float(getattr(self, f"score_{score}_risk_factor"))
            for score in range(7)
        )
        if any(value <= 0.0 for value in factors):
            raise ValueError("Grid v6 score risk factors must be positive")
        if self.maximum_campaign_risk_pct <= 0.0:
            raise ValueError("Grid v6 campaign risk ceiling must be positive")
        if self.maximum_notional_multiple < 0.0:
            raise ValueError(
                "Grid v6 notional multiple override cannot be negative"
            )
        if self.target_spacing <= 0.0:
            raise ValueError("Grid v6 target spacing must be positive")
        if self.campaign_loss_limit_r < 0.0:
            raise ValueError("Grid v6 campaign loss limit cannot be negative")
        if self.max_campaign_minutes <= 0:
            raise ValueError("Grid v6 campaign duration must be positive")
        if self.campaign_take_profit_r < 0.0:
            raise ValueError("Grid v6 campaign take profit cannot be negative")
        if self.profit_lock_activation_r < 0.0 or self.profit_giveback_r < 0.0:
            raise ValueError("Grid v6 profit protection cannot be negative")
        enabled = (
            self.profit_lock_activation_r > 0.0
            or self.profit_giveback_r > 0.0
        )
        if enabled and (
            self.profit_lock_activation_r <= 0.0
            or self.profit_giveback_r <= 0.0
        ):
            raise ValueError(
                "Grid v6 profit lock activation and giveback must be enabled together"
            )
        if self.cycle_profit_floor_min_take_profits < 0:
            raise ValueError(
                "Grid v6 cycle profit-floor count cannot be negative"
            )
        cycle_enabled = (
            self.cycle_profit_floor_min_take_profits > 0
            or self.cycle_profit_floor_activation_r > 0.0
            or self.cycle_profit_floor_r > 0.0
        )
        if cycle_enabled and not (
            self.cycle_profit_floor_min_take_profits > 0
            and self.cycle_profit_floor_activation_r > 0.0
            and 0.0
            <= self.cycle_profit_floor_r
            < self.cycle_profit_floor_activation_r
        ):
            raise ValueError(
                "Grid v6 cycle profit floor requires a take-profit count "
                "and a floor below its positive activation R"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def risk_factor(self, score: int) -> float:
        bounded = min(max(int(score), 0), 6)
        return float(getattr(self, f"score_{bounded}_risk_factor"))


def grid_v6_entry_decision(
    candidate: GridCandidate,
    snapshot: V4MarketSnapshot | None,
    confidence: GridV5ConfidencePolicy,
    policy: GridV6CampaignPolicy,
) -> tuple[bool, str, int, float]:
    """Return eligibility, v5 tier, score, and v6 relative risk factor."""

    policy.validate()
    tier, score = grid_v5_tier(candidate, snapshot, confidence)
    signal = candidate.signal
    accepted = not (
        (tier == "weak" and confidence.reject_weak_tier)
        or score < policy.minimum_score
        or signal.extension_atr < policy.minimum_actual_extension_atr
        or signal.extension_atr > policy.maximum_actual_extension_atr
        or signal.volume_ratio > policy.maximum_actual_volume_ratio
    )
    return accepted, tier, score, policy.risk_factor(score)
