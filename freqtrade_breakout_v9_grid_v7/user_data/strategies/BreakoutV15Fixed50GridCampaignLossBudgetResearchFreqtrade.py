from __future__ import annotations

from datetime import datetime
from typing import Any

from freqtrade.strategy import Trade

from BreakoutV15Fixed50RelativeSqueezeAlwaysResearchFreqtrade import (
    BreakoutV15Fixed50RelativeSqueezeAlwaysV13ResearchFreqtrade,
)
from BreakoutV15Fixed50ConfirmedRelativeSqueezeResearchFreqtrade import (
    BreakoutV15Fixed50ConfirmedSqueezeV13ResearchFreqtrade,
    BreakoutV15Fixed50ConfirmedSqueezeWeak350ResearchFreqtrade,
    BreakoutV15Fixed50ConfirmedSqueezeWeak365ResearchFreqtrade,
    BreakoutV15Fixed50ConfirmedSqueezeWeakImpulseResearchFreqtrade,
)


class _V15GridLongCampaignLossBudgetMixin:
    """Give Grid longs a side-specific, fill-based campaign loss budget."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.60

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        inherited = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if (
            inherited
            or self._component(getattr(trade, "enter_tag", None)) != "grid"
            or bool(getattr(trade, "is_short", False))
        ):
            return inherited
        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        profit = float(trade.calculate_profit(current_rate).total_profit)
        if profit <= -risk_budget * self.V15_GRID_LONG_LOSS_LIMIT_R:
            return "grid_v15_long_campaign_loss_budget"
        return inherited


class _V15GridShortCampaignLossBudgetMixin:
    """Give Grid shorts a separate fill-based campaign loss budget."""

    V15_GRID_SHORT_LOSS_LIMIT_R = 0.60

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        inherited = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if (
            inherited
            or self._component(getattr(trade, "enter_tag", None)) != "grid"
            or not bool(getattr(trade, "is_short", False))
        ):
            return inherited
        risk_budget = max(
            float(trade.get_custom_data("grid_risk_budget", 0.0)),
            1e-12,
        )
        profit = float(trade.calculate_profit(current_rate).total_profit)
        if profit <= -risk_budget * self.V15_GRID_SHORT_LOSS_LIMIT_R:
            return "grid_v15_short_campaign_loss_budget"
        return inherited


class BreakoutV15Fixed50GridLongLoss55ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50RelativeSqueezeAlwaysV13ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.55
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_long_loss55_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_long_loss55"


class BreakoutV15Fixed50GridLongLoss60ResearchFreqtrade(
    BreakoutV15Fixed50GridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_long_loss60_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_long_loss60"


class BreakoutV15Fixed50GridLongLoss65ResearchFreqtrade(
    BreakoutV15Fixed50GridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_long_loss65_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_long_loss65"


class BreakoutV15Fixed50GridShortLoss55ResearchFreqtrade(
    _V15GridShortCampaignLossBudgetMixin,
    BreakoutV15Fixed50RelativeSqueezeAlwaysV13ResearchFreqtrade,
):
    V15_GRID_SHORT_LOSS_LIMIT_R = 0.55
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_short_loss55_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_short_loss55"


class BreakoutV15Fixed50GridShortLoss60ResearchFreqtrade(
    BreakoutV15Fixed50GridShortLoss55ResearchFreqtrade,
):
    V15_GRID_SHORT_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_short_loss60_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_short_loss60"


class BreakoutV15Fixed50GridShortLoss65ResearchFreqtrade(
    BreakoutV15Fixed50GridShortLoss55ResearchFreqtrade,
):
    V15_GRID_SHORT_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_short_loss65_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_short_loss65"


class BreakoutV15Fixed50GridBothLoss60ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    _V15GridShortCampaignLossBudgetMixin,
    BreakoutV15Fixed50RelativeSqueezeAlwaysV13ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.60
    V15_GRID_SHORT_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = "breakout_v15_fixed50_grid_both_loss60_20260810"
    ADAPTIVE_STATE_BASENAME = "breakout_v15_fixed50_grid_both_loss60"


class BreakoutV15Fixed50ConfirmedGridLongLoss55ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50ConfirmedSqueezeV13ResearchFreqtrade,
):
    """Confirmed first-squeeze defense plus a 0.55R Grid-long budget."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.55
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_grid_long_loss55_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_grid_long_loss55"
    )


class BreakoutV15Fixed50ConfirmedGridLongLoss60ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedGridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_grid_long_loss60_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_grid_long_loss60"
    )


class BreakoutV15Fixed50ConfirmedGridLongLoss65ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedGridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_grid_long_loss65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_grid_long_loss65"
    )


class BreakoutV15Fixed50ConfirmedWeakGridLongLoss55ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50ConfirmedSqueezeWeakImpulseResearchFreqtrade,
):
    """Weak-impulse protection plus confirmed squeeze and Grid-long budget."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.55
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss55_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss55"
    )


class BreakoutV15Fixed50ConfirmedWeakGridLongLoss60ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedWeakGridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss60_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss60"
    )


class BreakoutV15Fixed50ConfirmedWeakGridLongLoss65ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedWeakGridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak_grid_long_loss65"
    )


class BreakoutV15Fixed50ConfirmedWeak350GridLongLoss55ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50ConfirmedSqueezeWeak350ResearchFreqtrade,
):
    """Narrow score-five protection with a 0.55R Grid-long budget."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.55
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss55_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss55"
    )


class BreakoutV15Fixed50ConfirmedWeak350GridLongLoss60ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedWeak350GridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.60
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss60_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss60"
    )


class BreakoutV15Fixed50ConfirmedWeak350GridLongLoss65ResearchFreqtrade(
    BreakoutV15Fixed50ConfirmedWeak350GridLongLoss55ResearchFreqtrade,
):
    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak350_grid_long_loss65"
    )


class BreakoutV15Fixed50ConfirmedWeak365GridLongLoss65ResearchFreqtrade(
    _V15GridLongCampaignLossBudgetMixin,
    BreakoutV15Fixed50ConfirmedSqueezeWeak365ResearchFreqtrade,
):
    """ONDO-only weak-body protection with the selected 0.65R Grid budget."""

    V15_GRID_LONG_LOSS_LIMIT_R = 0.65
    STRATEGY_VERSION = (
        "breakout_v15_fixed50_confirmed_weak365_grid_long_loss65_20260810"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v15_fixed50_confirmed_weak365_grid_long_loss65"
    )
