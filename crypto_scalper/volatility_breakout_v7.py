from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from .models import Direction
from .volatility_breakout_optimize import Candidate, CompactSeries
from .volatility_breakout_v4_research import V4MarketSnapshot


VOLATILITY_BREAKOUT_V7_NAME = (
    "dual_thrust_volatility_breakout_v7_confidence_allocated"
)


@dataclass(frozen=True)
class BreakoutV7ConfidencePolicy:
    """Point-in-time risk allocation layered on the frozen v6 signal.

    The policy does not look at a future bar or a trade outcome.  It counts
    direction-specific conditions that are known when the order is submitted,
    then scales the v6 risk budget.  A permissive policy is path-equivalent to
    v6, which makes every v7 improvement auditable against the frozen anchor.
    """

    long_max_quality_score: float = 999.0
    long_min_body_atr: float = 0.0
    long_min_volume_ratio: float = 0.0
    long_max_breakout_extension_atr: float = 999.0
    long_max_directional_breadth: float = 1.0
    short_min_quality_score: float = -999.0
    short_min_body_atr: float = 0.0
    short_min_volume_ratio: float = 0.0
    short_max_breakout_extension_atr: float = 999.0
    short_max_directional_breadth: float = 1.0
    strong_score: int = 6
    weak_score: int = -1
    strong_risk_multiplier: float = 1.0
    weak_risk_multiplier: float = 1.0
    long_risk_multiplier: float = 1.0
    short_risk_multiplier: float = 1.0
    reject_score_at_or_below: int = -1
    maximum_risk_multiplier: float = 2.0

    def validate(self) -> None:
        if not 0.0 <= self.long_max_directional_breadth <= 1.0:
            raise ValueError("long directional breadth ceiling must be in [0, 1]")
        if not 0.0 <= self.short_max_directional_breadth <= 1.0:
            raise ValueError("short directional breadth ceiling must be in [0, 1]")
        if not 0 <= self.strong_score <= 6:
            raise ValueError("strong score must be in [0, 6]")
        if not -1 <= self.weak_score <= 5:
            raise ValueError("weak score must be in [-1, 5]")
        if self.weak_score >= self.strong_score:
            raise ValueError("weak score must be below strong score")
        if not -1 <= self.reject_score_at_or_below <= 5:
            raise ValueError("reject score must be in [-1, 5]")
        multipliers = (
            self.strong_risk_multiplier,
            self.weak_risk_multiplier,
            self.long_risk_multiplier,
            self.short_risk_multiplier,
            self.maximum_risk_multiplier,
        )
        if any(value < 0.0 for value in multipliers):
            raise ValueError("v7 risk multipliers cannot be negative")
        if self.maximum_risk_multiplier <= 0.0:
            raise ValueError("maximum risk multiplier must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BreakoutV7EntryTiming:
    """Optional one-minute, close-confirmed entry revalidation.

    With ``confirmation_minutes=1`` the decision-minute 1m candle must finish
    before the candidate is moved to the following minute's open.  This keeps
    the timing causal and gives the stress harness an additional one-minute
    delay to apply after the normal v7 entry.
    """

    confirmation_minutes: int = 0
    min_directional_close_move_atr: float = -999.0
    max_directional_close_move_atr: float = 999.0
    max_adverse_excursion_atr: float = 999.0

    def validate(self) -> None:
        if self.confirmation_minutes not in {0, 1}:
            raise ValueError("v7 supports zero or one confirmation minute")
        if self.min_directional_close_move_atr > self.max_directional_close_move_atr:
            raise ValueError("confirmation move minimum exceeds maximum")
        if self.max_adverse_excursion_atr < 0.0:
            raise ValueError("maximum adverse excursion cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _directional_breadth(
    snapshot: V4MarketSnapshot, direction: Direction
) -> float:
    return (
        snapshot.breadth_above_ema21
        if direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )


def breakout_v7_confidence_score(
    candidate: Candidate,
    snapshot: V4MarketSnapshot | None,
    policy: BreakoutV7ConfidencePolicy,
) -> int:
    """Return a transparent 0..5 confidence score for one direction."""

    policy.validate()
    signal = candidate.signal
    directional_breadth = (
        _directional_breadth(snapshot, signal.direction)
        if snapshot is not None
        else 1.0
    )
    if signal.direction == Direction.LONG:
        conditions = (
            signal.quality_score <= policy.long_max_quality_score,
            signal.body_atr >= policy.long_min_body_atr,
            signal.volume_ratio >= policy.long_min_volume_ratio,
            signal.breakout_extension_atr
            <= policy.long_max_breakout_extension_atr,
            directional_breadth <= policy.long_max_directional_breadth,
        )
    elif signal.direction == Direction.SHORT:
        conditions = (
            signal.quality_score >= policy.short_min_quality_score,
            signal.body_atr >= policy.short_min_body_atr,
            signal.volume_ratio >= policy.short_min_volume_ratio,
            signal.breakout_extension_atr
            <= policy.short_max_breakout_extension_atr,
            directional_breadth <= policy.short_max_directional_breadth,
        )
    else:
        return 0
    return sum(int(condition) for condition in conditions)


def breakout_v7_risk_multiplier(
    candidate: Candidate,
    snapshot: V4MarketSnapshot | None,
    policy: BreakoutV7ConfidencePolicy,
) -> tuple[float, int, str]:
    """Return risk multiplier, score and an auditable allocation tier."""

    score = breakout_v7_confidence_score(candidate, snapshot, policy)
    if score <= policy.reject_score_at_or_below:
        return 0.0, score, "rejected"
    multiplier = (
        policy.long_risk_multiplier
        if candidate.signal.direction == Direction.LONG
        else policy.short_risk_multiplier
    )
    tier = "standard"
    if score >= policy.strong_score:
        multiplier *= policy.strong_risk_multiplier
        tier = "strong"
    elif score <= policy.weak_score:
        multiplier *= policy.weak_risk_multiplier
        tier = "weak"
    return min(multiplier, policy.maximum_risk_multiplier), score, tier


def apply_breakout_v7_timing(
    candidates: dict[int, list[Candidate]],
    execution_data: dict[str, CompactSeries],
    timing: BreakoutV7EntryTiming,
) -> dict[int, list[Candidate]]:
    """Apply causal 1m confirmation and return an execution-minute timeline."""

    timing.validate()
    if timing.confirmation_minutes == 0:
        return candidates
    output: dict[int, list[Candidate]] = {}
    for rows in candidates.values():
        for candidate in rows:
            signal = candidate.signal
            series = execution_data.get(signal.symbol)
            if series is None:
                continue
            decision_index = series.index_at(candidate.entry_minute)
            entry_minute = candidate.entry_minute + 1
            if decision_index is None or series.index_at(entry_minute) is None:
                continue
            atr = max(signal.atr_value, 1e-12)
            side = float(signal.direction.value)
            open_price = float(series.opens[decision_index])
            close_price = float(series.closes[decision_index])
            high_price = float(series.highs[decision_index])
            low_price = float(series.lows[decision_index])
            directional_move = side * (close_price - open_price) / atr
            adverse = (
                (open_price - low_price) / atr
                if signal.direction == Direction.LONG
                else (high_price - open_price) / atr
            )
            if not (
                timing.min_directional_close_move_atr
                <= directional_move
                <= timing.max_directional_close_move_atr
                and adverse <= timing.max_adverse_excursion_atr
            ):
                continue
            output.setdefault(entry_minute, []).append(
                replace(candidate, entry_minute=entry_minute)
            )
    for rows in output.values():
        rows.sort(key=lambda row: (-row.signal.quality_score, row.signal.symbol))
    return output
