from __future__ import annotations

import random
from dataclasses import replace

from .trend_grid_v6 import GridV6CampaignPolicy


TREND_GRID_V7_NAME = (
    "dynamic_trend_following_grid_v7_cycle_profit_floor"
)


def cycle_profit_floor_variants(
    source: GridV6CampaignPolicy,
    seed: int,
    budget: int,
) -> list[GridV6CampaignPolicy]:
    """Protect realized grid cycles while leaving new campaigns unchanged."""

    if budget <= 0:
        return []

    activations = (0.20, 0.24, 0.25, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40)
    floors = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)
    rows = [source]
    rows.extend(
        replace(
            source,
            cycle_profit_floor_min_take_profits=2,
            cycle_profit_floor_activation_r=activation,
            cycle_profit_floor_r=floor,
        )
        for activation in activations
        for floor in floors
        if floor < activation
    )
    rows.extend(
        replace(
            source,
            cycle_profit_floor_min_take_profits=minimum_take_profits,
            cycle_profit_floor_activation_r=activation,
            cycle_profit_floor_r=floor,
        )
        for minimum_take_profits in (1, 3)
        for activation in (0.24, 0.28, 0.32, 0.40)
        for floor in (0.0, 0.03, 0.06, 0.10)
        if floor < activation
    )

    rng = random.Random(seed)
    tail = [
        replace(
            source,
            cycle_profit_floor_min_take_profits=rng.choice((1, 2, 2, 2, 3)),
            cycle_profit_floor_activation_r=rng.choice(activations),
            cycle_profit_floor_r=rng.choice(floors),
            target_spacing=rng.choice((1.82, 1.85, 1.88)),
            campaign_loss_limit_r=rng.choice((0.65, 0.70)),
        )
        for _ in range(max(100, budget * 4))
    ]
    rows.extend(
        row
        for row in tail
        if row.cycle_profit_floor_r
        < row.cycle_profit_floor_activation_r
    )
    return list(dict.fromkeys(rows))[:budget]


def cycle_profit_floor_notional_variants(
    source: GridV6CampaignPolicy,
    seed: int,
    budget: int,
) -> list[GridV6CampaignPolicy]:
    """Refine a protected campaign around the binding notional ceiling."""

    if budget <= 0:
        return []

    notionals = (
        4.70,
        4.75,
        4.80,
        4.85,
        4.875,
        4.90,
        4.925,
        4.95,
        4.975,
        5.00,
    )
    activations = (0.20, 0.24, 0.28, 0.30)
    floors = (0.02, 0.03, 0.04)
    score6_factors = (0.80, 0.82, 0.85, 0.88, 0.90)

    def variant(
        notional: float,
        activation: float,
        floor: float,
        score6: float,
        standard: float = 1.02,
    ) -> GridV6CampaignPolicy:
        return replace(
            source,
            score_3_risk_factor=standard,
            score_4_risk_factor=standard,
            score_5_risk_factor=standard,
            score_6_risk_factor=score6,
            maximum_campaign_risk_pct=0.112,
            maximum_notional_multiple=notional,
            cycle_profit_floor_min_take_profits=2,
            cycle_profit_floor_activation_r=activation,
            cycle_profit_floor_r=floor,
        )

    rows = [source]
    rows.extend(
        variant(notional, 0.20, 0.03, 0.85)
        for notional in notionals
    )
    rows.extend(
        variant(notional, activation, floor, score6)
        for notional in (4.80, 4.85, 4.90, 4.925, 4.95)
        for activation in activations
        for floor in floors
        for score6 in (0.82, 0.85, 0.88)
    )

    rng = random.Random(seed)
    tail = [
        variant(
            rng.choice(notionals),
            rng.choice(activations),
            rng.choice(floors),
            rng.choice(score6_factors),
            rng.choice((1.0, 1.02, 1.04)),
        )
        for _ in range(max(100, budget * 5))
    ]
    rows.extend(tail)
    return list(dict.fromkeys(rows))[:budget]


def cycle_profit_floor_drawdown_variants(
    source: GridV6CampaignPolicy,
    _seed: int,
    budget: int,
) -> list[GridV6CampaignPolicy]:
    """Narrow, attributed search around the frozen v6 risk ceiling.

    The v7 drawdown audit attributes the three-month peak-to-trough path to a
    score-3 WLD campaign and the six-month path to a score-6 B campaign.  Both
    were sized at the campaign ceiling.  Keeping all score factors at their
    frozen v6 values and moving only that common ceiling slightly lower avoids
    redistributing risk toward another historical winner.
    """

    if budget <= 0:
        return []

    rows = [source]
    rows.extend(
        replace(
            source,
            score_3_risk_factor=1.0,
            score_4_risk_factor=1.0,
            score_5_risk_factor=1.0,
            score_6_risk_factor=1.0,
            maximum_campaign_risk_pct=cap,
            maximum_notional_multiple=5.0,
            cycle_profit_floor_min_take_profits=2,
            cycle_profit_floor_activation_r=activation,
            cycle_profit_floor_r=floor,
        )
        for cap in (0.1080, 0.1085, 0.1090, 0.1095, 0.1100)
        for activation in (0.20, 0.24)
        for floor in (0.02, 0.03, 0.04)
        if floor < activation
    )
    return list(dict.fromkeys(rows))[:budget]


def identity_allocation_variants(
    source: GridV6CampaignPolicy,
    _seed: int,
    _budget: int,
) -> list[GridV6CampaignPolicy]:
    return [source]
