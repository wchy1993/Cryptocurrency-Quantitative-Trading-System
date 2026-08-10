from __future__ import annotations

from typing import Any

import numpy as np

from BreakoutV15Fixed50Score5FailureExitResearchFreqtrade import (
    BreakoutV15Fixed50Score5FailureExitResearchFreqtrade,
)


class BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade(
    BreakoutV15Fixed50Score5FailureExitResearchFreqtrade,
):
    """Protect only score-five longs whose breakout body lacks conviction.

    V13 runners are left unchanged unless the completed signal candle is both
    a low-efficiency score-five long and has a breakout body no larger than a
    broad ATR multiple.  This retains the original stop ladder for strong
    impulses while applying the already tested failure ladder to weaker
    breakouts.  Entries, ranking, stake, leverage, DCA and Max2 are unchanged.
    """

    V15_SCORE5_MAX_BODY_ATR = 4.25
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure425_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure425"
    )

    def _v15_is_low_efficiency_score5(
        self,
        component: str,
        side: str,
        row: Any,
    ) -> bool:
        if not super()._v15_is_low_efficiency_score5(
            component,
            side,
            row,
        ):
            return False
        body_atr = float(row.get("bo_body_atr", np.inf))
        return bool(
            np.isfinite(body_atr)
            and body_atr <= self.V15_SCORE5_MAX_BODY_ATR
        )


class BreakoutV15Fixed50WeakImpulseScore5Failure450ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    V15_SCORE5_MAX_BODY_ATR = 4.50
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure450_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure450"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure460ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    V15_SCORE5_MAX_BODY_ATR = 4.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure460_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure460"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure425ResearchFreqtrade,
):
    """Narrow neighbor protecting only the weakest 2024 score-five bodies."""

    V15_SCORE5_MAX_BODY_ATR = 3.35
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure335_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure335"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure340ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
):
    V15_SCORE5_MAX_BODY_ATR = 3.40
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure340_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure340"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure350ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
):
    V15_SCORE5_MAX_BODY_ATR = 3.50
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure350_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure350"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure360ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
):
    V15_SCORE5_MAX_BODY_ATR = 3.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure360_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure360"
    )


class BreakoutV15Fixed50WeakImpulseScore5Failure365ResearchFreqtrade(
    BreakoutV15Fixed50WeakImpulseScore5Failure335ResearchFreqtrade,
):
    """Protect ONDO-like weak bodies while excluding the next broad group."""

    V15_SCORE5_MAX_BODY_ATR = 3.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_weak_impulse_score5_failure365_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_weak_impulse_score5_failure365"
    )
