from __future__ import annotations

import random
from dataclasses import replace
from typing import Tuple

from .volatility_breakout_v6_core_runner_optimize import ManagedLaneProfile


VOLATILITY_BREAKOUT_V9_NAME = (
    "dual_thrust_volatility_breakout_v9_profit_ladder"
)


ProfitFloor = Tuple[float, float]
ProfitLadder = Tuple[ProfitFloor, ...]
CaptureLane = Tuple[int, int, bool, bool]
PartialCapture = Tuple[float, float]


def _profit_capture_profile(
    source: ManagedLaneProfile,
    ladder: ProfitLadder,
    partial: PartialCapture = (0.0, 0.0),
    lane: CaptureLane = (0, 5, True, True),
    risk_scale: float = 1.0,
) -> ManagedLaneProfile:
    floors = [*ladder, *((0.0, 0.0),) * (3 - len(ladder))]
    partial_r, partial_fraction = partial
    minimum_score, maximum_score, capture_long, capture_short = lane
    return replace(
        source,
        core_risk_pct=source.core_risk_pct * risk_scale,
        core_strong_risk_pct=source.core_strong_risk_pct * risk_scale,
        core_profit_floor_1_activation_r=floors[0][0],
        core_profit_floor_1_lock_r=floors[0][1],
        core_profit_floor_2_activation_r=floors[1][0],
        core_profit_floor_2_lock_r=floors[1][1],
        core_profit_floor_3_activation_r=floors[2][0],
        core_profit_floor_3_lock_r=floors[2][1],
        core_profit_capture_min_score=minimum_score,
        core_profit_capture_max_score=maximum_score,
        core_profit_capture_long=capture_long,
        core_profit_capture_short=capture_short,
        core_partial_r=partial_r,
        core_partial_fraction=partial_fraction,
        core_move_breakeven_after_partial=False,
    )


def profit_capture_profile_variants(
    source: ManagedLaneProfile,
    seed: int,
    budget: int,
) -> list[ManagedLaneProfile]:
    """Build causal profit ladders without changing the frozen v8 signal.

    A ladder raises the stop to a fixed full-cost R floor after a position's
    observed MFE crosses each checkpoint.  Unlike a tight continuous trailing
    stop, a fixed floor can bank a small win while leaving most of the position
    enough room to participate in a large trend.
    """

    if budget <= 0:
        return []

    single_ladders: tuple[ProfitLadder, ...] = (
        ((1.5, 0.0),),
        ((2.0, 0.0),),
        ((2.0, 0.15),),
        ((2.0, 0.25),),
        ((2.5, 0.15),),
        ((2.5, 0.25),),
        ((3.0, 0.25),),
        ((3.0, 0.50),),
        ((3.5, 0.50),),
        ((4.0, 0.25),),
        ((4.0, 0.50),),
        ((4.0, 1.00),),
        ((5.0, 0.50),),
        ((5.0, 1.00),),
        ((6.0, 1.00),),
        ((7.0, 1.00),),
        ((8.0, 2.00),),
    )
    staged_ladders: tuple[ProfitLadder, ...] = (
        ((2.0, 0.15), (4.0, 1.00)),
        ((2.0, 0.25), (5.0, 1.50)),
        ((2.5, 0.25), (5.0, 1.00)),
        ((3.0, 0.25), (6.0, 1.50)),
        ((3.0, 0.50), (6.0, 2.00)),
        ((3.0, 0.25), (7.0, 2.00)),
        ((4.0, 0.50), (8.0, 2.50)),
        ((4.0, 1.00), (8.0, 3.00)),
        ((5.0, 1.00), (10.0, 3.00)),
        ((2.0, 0.15), (5.0, 1.00), (10.0, 3.00)),
        ((2.0, 0.25), (5.0, 1.50), (10.0, 4.00)),
        ((2.5, 0.25), (6.0, 1.50), (12.0, 4.00)),
        ((3.0, 0.50), (7.0, 2.00), (12.0, 5.00)),
        ((4.0, 0.50), (8.0, 2.50), (15.0, 4.00)),
    )
    ladders = single_ladders + staged_ladders
    partials: tuple[PartialCapture, ...] = (
        (0.0, 0.0),
        (1.5, 0.05),
        (2.0, 0.05),
        (2.0, 0.08),
        (2.0, 0.10),
        (2.5, 0.08),
        (3.0, 0.10),
        (4.0, 0.10),
    )
    lanes: tuple[CaptureLane, ...] = (
        (0, 5, True, True),
        (0, 4, True, True),
        (2, 4, True, True),
        (3, 4, True, True),
        (4, 4, True, False),
        (3, 4, True, False),
        (2, 4, False, True),
        (3, 4, False, True),
        (4, 4, True, True),
    )

    rows = [source]
    # Deterministic anchors ensure the search always covers simple, explainable
    # fixed floors before it explores partial exits and lane specialization.
    rows.extend(
        _profit_capture_profile(source, ladder)
        for ladder in ladders
    )
    for ladder in (
        ((2.0, 0.15),),
        ((2.5, 0.25),),
        ((3.0, 0.50),),
        ((4.0, 0.50),),
        ((2.0, 0.15), (5.0, 1.00), (10.0, 3.00)),
    ):
        for lane in lanes[1:]:
            rows.append(_profit_capture_profile(source, ladder, lane=lane))
    for ladder in (
        ((2.0, 0.15),),
        ((2.5, 0.25),),
        ((3.0, 0.50),),
        ((2.0, 0.15), (5.0, 1.00), (10.0, 3.00)),
    ):
        for partial in partials[1:]:
            rows.append(
                _profit_capture_profile(
                    source, ladder, partial=partial
                )
            )

    rng = random.Random(seed)
    tail = [
        _profit_capture_profile(
            source,
            ladder,
            partial=partial,
            lane=lane,
            risk_scale=risk_scale,
        )
        for ladder in ladders
        for partial in partials
        for lane in lanes
        for risk_scale in (0.98, 1.0, 1.02, 1.04)
    ]
    rng.shuffle(tail)
    rows.extend(tail)
    return list(dict.fromkeys(rows))[:budget]


def profit_capture_refinement_variants(
    source: ManagedLaneProfile,
    seed: int,
    budget: int,
) -> list[ManagedLaneProfile]:
    """Search locally around a strict v9 candidate.

    This stage jointly tests slightly different realized-profit fractions,
    fixed profit floors, risk levels, and causal drawdown governors.  It keeps
    the v8 entry signal and score allocation frozen.
    """

    if budget <= 0:
        return []

    ladders: tuple[ProfitLadder, ...] = (
        ((3.0, 0.50),),
        ((3.5, 0.50),),
        ((4.0, 0.50),),
        ((4.0, 0.75),),
        ((4.0, 1.00),),
        ((4.5, 0.75),),
        ((4.5, 1.00),),
        ((5.0, 1.00),),
        ((5.0, 1.25),),
        ((6.0, 1.00),),
        ((4.0, 0.75), (8.0, 2.50)),
        ((4.0, 1.00), (8.0, 3.00)),
        ((5.0, 1.00), (10.0, 3.00)),
    )
    partials: tuple[PartialCapture, ...] = (
        (0.0, 0.0),
        (1.50, 0.05),
        (1.75, 0.05),
        (1.75, 0.08),
        (2.00, 0.05),
        (2.00, 0.075),
        (2.00, 0.08),
        (2.00, 0.10),
        (2.00, 0.12),
        (2.25, 0.08),
        (2.25, 0.10),
        (2.50, 0.08),
        (2.50, 0.10),
        (3.00, 0.10),
    )
    lanes: tuple[CaptureLane, ...] = (
        (3, 4, False, True),
        (2, 4, False, True),
        (3, 5, False, True),
        (4, 4, False, True),
        (3, 4, True, True),
    )
    governors = (
        (1.0, 1.0, 1.0, 1.0),
        (0.20, 0.80, 0.27, 0.50),
        (0.22, 0.80, 0.28, 0.50),
        (0.24, 0.80, 0.29, 0.50),
        (0.25, 0.85, 0.29, 0.55),
        (0.26, 0.80, 0.29, 0.50),
        (0.27, 0.80, 0.295, 0.50),
        (0.28, 0.75, 0.30, 0.45),
    )
    risk_scales = (0.98, 1.0, 1.01, 1.02, 1.03, 1.04, 1.06)

    def governed(
        profile: ManagedLaneProfile,
        governor: tuple[float, float, float, float],
    ) -> ManagedLaneProfile:
        reduce_start, reduce_multiplier, deep_start, deep_multiplier = governor
        return replace(
            profile,
            drawdown_reduce_start=reduce_start,
            drawdown_reduce_multiplier=reduce_multiplier,
            drawdown_deep_start=deep_start,
            drawdown_deep_multiplier=deep_multiplier,
            drawdown_scope="global",
        )

    source_ladder: ProfitLadder = tuple(
        (activation, lock)
        for activation, lock in (
            (
                source.core_profit_floor_1_activation_r,
                source.core_profit_floor_1_lock_r,
            ),
            (
                source.core_profit_floor_2_activation_r,
                source.core_profit_floor_2_lock_r,
            ),
            (
                source.core_profit_floor_3_activation_r,
                source.core_profit_floor_3_lock_r,
            ),
        )
        if activation > 0.0
    )
    source_partial = (
        source.core_partial_r,
        source.core_partial_fraction,
    )
    source_lane = (
        source.core_profit_capture_min_score,
        source.core_profit_capture_max_score,
        source.core_profit_capture_long,
        source.core_profit_capture_short,
    )

    rows = [source]
    rows.extend(
        _profit_capture_profile(
            source,
            source_ladder,
            partial=source_partial,
            lane=source_lane,
            risk_scale=risk_scale,
        )
        for risk_scale in risk_scales
    )
    rows.extend(
        _profit_capture_profile(
            source,
            source_ladder,
            partial=partial,
            lane=source_lane,
        )
        for partial in partials
    )
    rows.extend(
        governed(source, governor) for governor in governors[1:]
    )
    for risk_scale in (1.01, 1.02, 1.03, 1.04):
        for governor in governors[1:]:
            rows.append(
                governed(
                    _profit_capture_profile(
                        source,
                        source_ladder,
                        partial=source_partial,
                        lane=source_lane,
                        risk_scale=risk_scale,
                    ),
                    governor,
                )
            )

    rng = random.Random(seed)
    tail = [
        governed(
            _profit_capture_profile(
                source,
                ladder,
                partial=partial,
                lane=lane,
                risk_scale=risk_scale,
            ),
            governor,
        )
        for ladder in ladders
        for partial in partials
        for lane in lanes
        for risk_scale in risk_scales
        for governor in governors
    ]
    rng.shuffle(tail)
    rows.extend(tail)
    return list(dict.fromkeys(rows))[:budget]


def long_profit_floor_refinement_variants(
    source: ManagedLaneProfile,
    seed: int,
    budget: int,
) -> list[ManagedLaneProfile]:
    """Add a separate, loose score-aware floor to long positions."""

    if budget <= 0:
        return []

    activations = (0.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)
    locks = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
    long_score_ranges = ((4, 4), (3, 4), (4, 5))
    short_partials = (
        (2.25, 0.10),
        (2.50, 0.08),
        (2.50, 0.10),
        (2.50, 0.12),
        (2.75, 0.08),
        (2.75, 0.10),
        (3.00, 0.10),
    )
    risk_scales = (0.99, 1.0, 1.01, 1.02)

    def variant(
        activation: float,
        lock: float,
        score_range: tuple[int, int],
        partial: PartialCapture,
        risk_scale: float,
    ) -> ManagedLaneProfile:
        return replace(
            source,
            core_risk_pct=source.core_risk_pct * risk_scale,
            core_strong_risk_pct=(
                source.core_strong_risk_pct * risk_scale
            ),
            core_partial_r=partial[0],
            core_partial_fraction=partial[1],
            core_long_profit_floor_activation_r=activation,
            core_long_profit_floor_lock_r=lock,
            core_long_profit_floor_min_score=score_range[0],
            core_long_profit_floor_max_score=score_range[1],
        )

    source_partial = (
        source.core_partial_r,
        source.core_partial_fraction,
    )
    rows = [source]
    for activation in activations[1:]:
        for lock in locks:
            if lock <= activation:
                rows.append(
                    variant(
                        activation,
                        lock,
                        (4, 4),
                        source_partial,
                        1.0,
                    )
                )
    for partial in short_partials:
        rows.append(variant(0.0, 0.0, (4, 4), partial, 1.0))
    for risk_scale in risk_scales:
        rows.append(
            variant(
                0.0,
                0.0,
                (4, 4),
                source_partial,
                risk_scale,
            )
        )

    rng = random.Random(seed)
    tail = [
        variant(activation, lock, score_range, partial, risk_scale)
        for activation in activations[1:]
        for lock in locks
        if lock <= activation
        for score_range in long_score_ranges
        for partial in short_partials
        for risk_scale in risk_scales
    ]
    rng.shuffle(tail)
    rows.extend(tail)
    return list(dict.fromkeys(rows))[:budget]
