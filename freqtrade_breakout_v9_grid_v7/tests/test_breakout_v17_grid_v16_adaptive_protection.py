from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (  # noqa: E402
    _PrecisionConfirmedInitialStopMixin,
)
import BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade as research  # noqa: E402
from BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade import (  # noqa: E402
    BreakoutV16GridV16DeepWeakTapeRisk50ResearchFreqtrade,
    BreakoutV16GridV16LowEfficiencyScore4GuardResearchFreqtrade,
    BreakoutV16GridV16WeakTapeRisk65ResearchFreqtrade,
    BreakoutV17GridV15MatureMomentumReversalResearchFreqtrade,
    BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade,
    BreakoutV17GridV16CurrentRegimeResearchFreqtrade,
    BreakoutV17GridV16DefensiveResearchFreqtrade,
    BreakoutV17GridV16MatureReversalRisk50ResearchFreqtrade,
    BreakoutV17GridV16MatureReversalRisk65ResearchFreqtrade,
    BreakoutV17GridV16EarlyBridgeE80RejectedResearchFreqtrade,
    BreakoutV17GridV16ParetoFinalResearchFreqtrade,
    BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade,
    BreakoutV17GridV16PortfolioGuardT08D50R85ResearchFreqtrade,
    BreakoutV17GridV16PortfolioGuardT08D55R80ResearchFreqtrade,
    BreakoutV17GridV16PortfolioGuardNoReversalResearchFreqtrade,
    BreakoutV17GridV16PortfolioGuardT12D65R80ResearchFreqtrade,
    BreakoutV17GridV16ParetoCandidateResearchFreqtrade,
    BreakoutV17GridV16TightScore2ResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3GuardResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3LowEfficiencyGuardResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3LowEfficiencyRisk80ResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk35ResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk40ResearchFreqtrade,
    BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk50ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyGuard210ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyGuard230ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk35ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2LowEfficiencyRisk80ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2PortfolioGuardT15ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2PortfolioGuardT16GentleResearchFreqtrade,
    BreakoutV17GridV16WatchScore2PortfolioGuardT17ResearchFreqtrade,
    BreakoutV17GridV16WatchScore2FastHighFailureResearchFreqtrade,
    BreakoutV17GridV16WatchScore2GuardResearchFreqtrade,
    _BreakoutV17ConfirmedRetentionMixin,
    _BreakoutV17HighScoreWatchFailureExitMixin,
    _BreakoutV17MomentumReversalExitMixin,
    _BreakoutV17SelectiveWatchEntryMixin,
    _BreakoutV17UnderwaterOrdinaryScore3RiskMixin,
    _GridV16ConfirmedReversalMixin,
    _GridV16LowEfficiencyScore4EntryMixin,
    _GridV16LowEfficiencyScore4RiskMixin,
    _PrecisionClockCompatibilityMixin,
    _V17GridV16EntryQualityRiskMixin,
)


class _Trade:
    def __init__(self, tag: str, *, is_short: bool = True) -> None:
        self.enter_tag = tag
        self.is_short = is_short
        self.open_rate = 1.0
        self.leverage = 1.0
        self.min_rate = 0.95 if is_short else 1.0
        self.max_rate = 1.05 if not is_short else 1.0
        self.open_date_utc = datetime(2026, 8, 20, tzinfo=timezone.utc)
        self.custom: dict[str, Any] = {
            "bo_score": 2,
            "bo_capture": False,
            "initial_unit_risk": 0.10,
            "grid_risk_budget": 100.0,
            "grid_tp_count": 1,
        }

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        return self.custom.get(key, default)

    def set_custom_data(self, key: str, value: Any) -> None:
        self.custom[key] = value

    def calculate_profit(self, rate: float) -> SimpleNamespace:
        side = -1.0 if self.is_short else 1.0
        return SimpleNamespace(
            total_profit=side * (float(rate) - self.open_rate) * 1000.0
        )


class _Parent:
    SIDE_COST = 0.0

    @staticmethod
    def _component(tag: str | None) -> str:
        return "grid" if str(tag or "").startswith("grid") else "breakout"

    @staticmethod
    def _trade_r(trade: _Trade, rate: float) -> float:
        side = -1.0 if trade.is_short else 1.0
        return side * (float(rate) - trade.open_rate) / 0.10

    @staticmethod
    def _confirmed_peak_r(
        pair: str,
        trade: _Trade,
        current_time: datetime,
    ) -> float:
        return float(trade.get_custom_data("confirmed_peak_r", 0.0))

    @staticmethod
    def _latest_row(
        pair: str,
        current_time: datetime,
    ) -> dict[str, float]:
        return {
            "fast_ema": 1.01,
            "slow_ema": 1.00,
            "symbol_return_4h": 0.01,
        }

    def custom_stoploss(self, *args: Any, **kwargs: Any) -> float:
        return 0.50

    def custom_exit(self, *args: Any, **kwargs: Any) -> None:
        return None


class _BreakoutHarness(_BreakoutV17ConfirmedRetentionMixin, _Parent):
    pass


class _GridHarness(_GridV16ConfirmedReversalMixin, _Parent):
    pass


class _QualityRiskHarness(_V17GridV16EntryQualityRiskMixin, _Parent):
    pass


class _MomentumReversalHarness(
    _BreakoutV17MomentumReversalExitMixin,
    _Parent,
):
    pass


class _WatchStateParent:
    @staticmethod
    def _v16_mark_path_states(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.copy()


class _WatchEntryHarness(
    _BreakoutV17SelectiveWatchEntryMixin,
    _WatchStateParent,
):
    pass


class _WatchFailureParent(_Parent):
    @staticmethod
    def _trade_open_time(trade: _Trade) -> datetime:
        return trade.open_date_utc


class _WatchFailureHarness(
    _BreakoutV17HighScoreWatchFailureExitMixin,
    _WatchFailureParent,
):
    pass


class _GridIndicatorParent:
    @staticmethod
    def populate_indicators(
        dataframe: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> pd.DataFrame:
        return dataframe.copy()


class _GridEfficiencyHarness(
    _GridV16LowEfficiencyScore4EntryMixin,
    _GridIndicatorParent,
):
    pass


class _StakeParent(_Parent):
    signal_row: dict[str, float] = {
        "grid_score": 4,
        "market_efficiency": 0.20,
    }

    @staticmethod
    def custom_stake_amount(*args: Any, **kwargs: Any) -> float:
        return 100.0

    @staticmethod
    def order_filled(*args: Any, **kwargs: Any) -> None:
        return None

    def _latest_signal_row(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> dict[str, float]:
        return self.signal_row


class _GridEfficiencyRiskHarness(
    _GridV16LowEfficiencyScore4RiskMixin,
    _StakeParent,
):
    pass


class _Wallets:
    def __init__(self, equity: float) -> None:
        self.equity = equity

    def get_total_stake_amount(self) -> float:
        return self.equity

    @staticmethod
    def get_starting_balance() -> float:
        return 200.0


class _UnderwaterScore3RiskHarness(
    _BreakoutV17UnderwaterOrdinaryScore3RiskMixin,
    _StakeParent,
):
    signal_row = {"bo_score": 3, "bo_capture": 0}
    _peak_equity = 100.0
    wallets = _Wallets(84.0)
    drawdown = 0.16
    reserved = 0.0
    recent_profits: list[float] = []

    def _v17_realized_drawdown(self, current_time: datetime) -> float:
        return self.drawdown

    def _v17_reserved_savings(self, current_time: datetime) -> float:
        return self.reserved

    def _v17_recent_closed_profits(
        self,
        current_time: datetime,
        count: int,
    ) -> list[float]:
        return self.recent_profits[-count:]


class _RealizedLedgerHarness(
    _BreakoutV17UnderwaterOrdinaryScore3RiskMixin,
    _StakeParent,
):
    wallets = _Wallets(0.0)

    @staticmethod
    def _trade_close_time(trade: Any) -> datetime | None:
        return getattr(trade, "close_date_utc", None)


class _ClockHarness(
    _PrecisionClockCompatibilityMixin,
    _PrecisionConfirmedInitialStopMixin,
):
    pass


def test_combined_candidate_is_a_thin_overlay_on_frozen_v3() -> None:
    selected = BreakoutV17GridV16AdaptiveProtectionResearchFreqtrade
    tight = BreakoutV17GridV16TightScore2ResearchFreqtrade

    assert selected.V17_BO_MAX_SCORE == 3
    assert selected.GRID_V16_MAX_SCORE == 5
    assert tight.V17_BO_MAX_SCORE == 2
    assert tight.V17_BO_FLOOR_1_R == pytest.approx(0.05)
    assert tight.populate_entry_trend is selected.populate_entry_trend
    assert tight.custom_stake_amount is selected.custom_stake_amount
    assert (
        BreakoutV17GridV16CurrentRegimeResearchFreqtrade
        .GRID_V16_WEAK_TAPE_SCALE
        == pytest.approx(0.50)
    )
    assert (
        BreakoutV17GridV16DefensiveResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.08)
    )


def test_third_round_neighbours_change_only_declared_boundaries() -> None:
    assert (
        BreakoutV16GridV16WeakTapeRisk65ResearchFreqtrade
        .GRID_V16_WEAK_TAPE_SCALE
        == pytest.approx(0.65)
    )
    assert (
        BreakoutV16GridV16DeepWeakTapeRisk50ResearchFreqtrade
        .GRID_V16_WEAK_TAPE_MAX_MEDIAN_RETURN_24H
        == pytest.approx(-0.02)
    )
    assert (
        BreakoutV17GridV15MatureMomentumReversalResearchFreqtrade
        .V17_REVERSAL_MIN_CONFIRMED_PEAK_R
        == pytest.approx(0.75)
    )
    assert not BreakoutV17GridV16MatureReversalRisk65ResearchFreqtrade.V17_WEAK_SHORT_ENABLED
    assert (
        BreakoutV17GridV16MatureReversalRisk50ResearchFreqtrade
        .GRID_V16_WEAK_TAPE_SCALE
        == pytest.approx(0.50)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.08)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT08D50R75ResearchFreqtrade
        .PORTFOLIO_DEFENSIVE_SCALE
        == pytest.approx(0.50)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT12D65R80ResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.12)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT12D65R80ResearchFreqtrade
        .PORTFOLIO_RECOVERY_SCALE
        == pytest.approx(0.80)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT08D50R85ResearchFreqtrade
        .PORTFOLIO_RECOVERY_SCALE
        == pytest.approx(0.85)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardT08D55R80ResearchFreqtrade
        .PORTFOLIO_DEFENSIVE_SCALE
        == pytest.approx(0.55)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardNoReversalResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.08)
    )
    assert (
        BreakoutV17GridV16PortfolioGuardNoReversalResearchFreqtrade.custom_exit
        is not _BreakoutV17MomentumReversalExitMixin.custom_exit
    )


def test_breakout_v17_uses_completed_peak_and_respects_score_boundary() -> None:
    strategy = _BreakoutHarness()
    trade = _Trade("bo_v9_s2_r30294_c0_l0")
    trade.set_custom_data("confirmed_peak_r", 0.60)
    now = datetime.now(timezone.utc)

    assert strategy._v17_breakout_locked_r("TEST", trade, now) == pytest.approx(
        -0.10
    )
    assert strategy.custom_exit("TEST", trade, now, 1.02, -0.02) == (
        strategy.V17_BO_EXIT_REASON
    )

    trade.set_custom_data("bo_score", 4)
    assert strategy._v17_breakout_locked_r("TEST", trade, now) is None
    assert strategy.custom_exit("TEST", trade, now, 1.02, -0.02) is None


def test_breakout_v17_does_not_reprotect_capture_trade() -> None:
    strategy = _BreakoutHarness()
    trade = _Trade("bo_v9_s2_r30294_c1_l0")
    trade.set_custom_data("bo_capture", True)
    trade.set_custom_data("confirmed_peak_r", 3.0)

    assert strategy._v17_breakout_locked_r(
        "TEST", trade, datetime.now(timezone.utc)
    ) is None


def test_grid_v16_requires_partial_profit_and_first_confirmed_reversal() -> None:
    strategy = _GridHarness()
    trade = _Trade("grid_v8_short_s5")
    trade.min_rate = 0.95
    now = datetime.now(timezone.utc)

    assert strategy.custom_exit("TEST", trade, now, 1.01, -0.01) == (
        strategy.GRID_V16_EXIT_REASON
    )

    trade.set_custom_data("grid_tp_count", 0)
    assert strategy.custom_exit("TEST", trade, now, 1.01, -0.01) is None


def test_entry_quality_risk_only_scales_preidentified_short_clusters() -> None:
    strategy = _QualityRiskHarness()

    assert strategy._v17_grid_v16_entry_scale(
        "breakout",
        "short",
        {"bo_score": 3, "symbol_return_4h": -0.004},
    ) == pytest.approx(0.35)
    assert strategy._v17_grid_v16_entry_scale(
        "breakout",
        "short",
        {"bo_score": 4, "symbol_return_4h": -0.004},
    ) == pytest.approx(1.0)
    assert strategy._v17_grid_v16_entry_scale(
        "grid",
        "short",
        {"grid_score": 5, "market_median_return_24h": -0.02},
    ) == pytest.approx(0.50)
    assert strategy._v17_grid_v16_entry_scale(
        "grid",
        "long",
        {"grid_score": 5, "market_median_return_24h": -0.02},
    ) == pytest.approx(1.0)


def test_breakout_momentum_reversal_requires_profit_then_completed_flip() -> None:
    strategy = _MomentumReversalHarness()
    trade = _Trade("bo_v9_s2_r30294_c0_l0")
    trade.set_custom_data("confirmed_peak_r", 0.60)
    now = datetime.now(timezone.utc)

    assert strategy.custom_exit("TEST", trade, now, 1.02, -0.02) == (
        strategy.V17_REVERSAL_EXIT_REASON
    )

    trade.set_custom_data("confirmed_peak_r", 0.40)
    assert strategy.custom_exit("TEST", trade, now, 1.02, -0.02) is None

    trade.set_custom_data("grid_tp_count", 1)
    trade.enter_tag = "grid_v8_short_s6"
    assert strategy.custom_exit("TEST", trade, now, 1.01, -0.01) is None


def test_watch_entry_guard_rejects_only_declared_scores() -> None:
    frame = pd.DataFrame(
        {
            "bo_entry_long": [0, 0, 0],
            "bo_entry_short": [1, 1, 1],
            "bo_entry": [1, 1, 1],
            "bo_score": [2, 3, 4],
            "v16_no_follow_watch": [1, 1, 0],
        }
    )

    guarded = _WatchEntryHarness()._v16_mark_path_states(frame)

    assert guarded["v17_watch_rejected"].tolist() == [1, 0, 0]
    assert guarded["bo_entry_short"].tolist() == [0, 1, 1]
    assert guarded["bo_entry"].tolist() == [0, 1, 1]


def test_high_score_watch_failure_exit_is_bounded_and_score_specific() -> None:
    strategy = _WatchFailureHarness()
    trade = _Trade("bo_v9_s4_r43320_c1_l0")
    trade.set_custom_data("bo_score", 4)
    trade.set_custom_data("v16_no_follow_watch", True)
    trade.min_rate = 1.0
    now = trade.open_date_utc.replace(minute=15)

    assert strategy.custom_exit("TEST", trade, now, 1.03, -0.03) == (
        strategy.V17_WATCH_FAILURE_EXIT_REASON
    )

    trade.set_custom_data("bo_score", 3)
    assert strategy.custom_exit("TEST", trade, now, 1.03, -0.03) is None

    trade.set_custom_data("bo_score", 4)
    trade.min_rate = 0.98
    assert strategy.custom_exit("TEST", trade, now, 1.03, -0.03) is None


def test_watch_guard_candidate_boundaries_are_explicit() -> None:
    assert (
        BreakoutV17GridV16WatchScore2GuardResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2,)
    )
    assert (
        BreakoutV17GridV16WatchNonScore3GuardResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2, 4, 5)
    )
    assert (
        BreakoutV17GridV16WatchScore2FastHighFailureResearchFreqtrade
        .V17_WATCH_FAILURE_CURRENT_R
        == pytest.approx(-0.20)
    )


def test_grid_v16_efficiency_guard_is_score_side_and_threshold_specific() -> None:
    frame = pd.DataFrame(
        {
            "grid_entry_long": [0, 0, 1, 0],
            "grid_entry_short": [1, 1, 0, 1],
            "grid_entry": [1, 1, 1, 1],
            "grid_score": [4, 5, 4, 4],
            "market_efficiency": [0.20, 0.20, 0.20, 0.24],
        }
    )

    guarded = _GridEfficiencyHarness().populate_indicators(frame, {})

    assert guarded["grid_v16_low_efficiency_rejected"].tolist() == [1, 0, 0, 0]
    assert guarded["grid_entry_short"].tolist() == [0, 1, 0, 1]
    assert guarded["grid_entry"].tolist() == [0, 1, 1, 1]


def test_grid_v16_efficiency_neighborhood_is_explicit() -> None:
    assert (
        BreakoutV16GridV16LowEfficiencyScore4GuardResearchFreqtrade
        .GRID_V16_MAX_MARKET_EFFICIENCY
        == pytest.approx(0.22)
    )
    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyGuardResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2,)
    )
    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyGuard210ResearchFreqtrade
        .GRID_V16_MAX_MARKET_EFFICIENCY
        == pytest.approx(0.21)
    )
    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyGuard230ResearchFreqtrade
        .GRID_V16_MAX_MARKET_EFFICIENCY
        == pytest.approx(0.23)
    )
    assert (
        BreakoutV17GridV16WatchNonScore3LowEfficiencyGuardResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2, 4, 5)
    )
    assert (
        BreakoutV17GridV16WatchScore2PortfolioGuardT15ResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.15)
    )
    assert (
        BreakoutV17GridV16WatchScore2PortfolioGuardT17ResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.17)
    )
    assert (
        BreakoutV17GridV16WatchScore2PortfolioGuardT16GentleResearchFreqtrade
        .PORTFOLIO_DEFENSIVE_SCALE
        == pytest.approx(0.50)
    )
    assert (
        BreakoutV17GridV16ParetoCandidateResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2, 4, 5)
    )
    assert (
        BreakoutV17GridV16ParetoCandidateResearchFreqtrade
        .PORTFOLIO_GOVERNOR_TRIGGER
        == pytest.approx(0.16)
    )


def test_grid_v16_efficiency_risk_preserves_slot_and_scales_only_target() -> None:
    strategy = _GridEfficiencyRiskHarness()
    now = datetime.now(timezone.utc)
    args = (
        "TEST",
        now,
        1.0,
        100.0,
        1.0,
        1000.0,
        1.0,
    )

    assert strategy.custom_stake_amount(
        *args,
        "grid_v8_short_s4",
        "short",
    ) == pytest.approx(50.0)

    strategy.signal_row = {"grid_score": 5, "market_efficiency": 0.20}
    assert strategy.custom_stake_amount(
        *args,
        "grid_v8_short_s5",
        "short",
    ) == pytest.approx(100.0)

    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyRisk35ResearchFreqtrade
        .GRID_V16_LOW_EFFICIENCY_SCALE
        == pytest.approx(0.35)
    )
    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyRisk50ResearchFreqtrade
        .GRID_V16_LOW_EFFICIENCY_SCALE
        == pytest.approx(0.50)
    )
    assert (
        BreakoutV17GridV16WatchScore2LowEfficiencyRisk80ResearchFreqtrade
        .GRID_V16_LOW_EFFICIENCY_SCALE
        == pytest.approx(0.80)
    )
    assert (
        BreakoutV17GridV16WatchNonScore3LowEfficiencyRisk80ResearchFreqtrade
        .V17_WATCH_REJECT_SCORES
        == (2, 4, 5)
    )


def test_underwater_score3_risk_is_narrow_and_preserves_the_slot() -> None:
    strategy = _UnderwaterScore3RiskHarness()
    now = datetime.now(timezone.utc)
    args = (
        "TEST",
        now,
        1.0,
        100.0,
        1.0,
        1000.0,
        1.0,
    )

    assert strategy.custom_stake_amount(
        *args,
        "bo_v9_s3_r31809_c0_l0",
        "long",
    ) == pytest.approx(40.0)

    assert strategy.custom_stake_amount(
        *args,
        "bo_v9_s3_r25750_c1_l0",
        "short",
    ) == pytest.approx(100.0)

    assert strategy.custom_stake_amount(
        *args,
        "bo_v9_s3_r31809_c0_l0",
        "short",
    ) == pytest.approx(100.0)

    strategy.drawdown = 0.19
    assert strategy.custom_stake_amount(
        *args,
        "bo_v9_s3_r31809_c0_l0",
        "long",
    ) == pytest.approx(100.0)

    assert (
        BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk35ResearchFreqtrade
        .V17_UNDERWATER_SCALE
        == pytest.approx(0.35)
    )
    assert (
        BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk40ResearchFreqtrade
        .V17_UNDERWATER_SCALE
        == pytest.approx(0.40)
    )
    assert (
        BreakoutV17GridV16WatchNonScore3UnderwaterS3Risk50ResearchFreqtrade
        .V17_UNDERWATER_SCALE
        == pytest.approx(0.50)
    )


def test_underwater_savings_are_reserved_from_later_stakes() -> None:
    strategy = _UnderwaterScore3RiskHarness()
    strategy.wallets = _Wallets(110.0)
    strategy.reserved = 10.0
    strategy.drawdown = 0.19
    now = datetime.now(timezone.utc)

    stake = strategy.custom_stake_amount(
        "TEST",
        now,
        1.0,
        100.0,
        1.0,
        1000.0,
        1.0,
        "grid_v15_s4_r10000_c0_l0",
        "short",
    )

    assert stake == pytest.approx(100.0 * 100.0 / 110.0)


def test_pareto_r35_early_bridge_requires_two_recent_losses() -> None:
    strategy = _UnderwaterScore3RiskHarness()
    strategy.V17_UNDERWATER_EARLY_MIN_DRAWDOWN = 0.13
    strategy.V17_UNDERWATER_EARLY_LOSS_STREAK = 2
    strategy.drawdown = 0.14
    now = datetime.now(timezone.utc)
    args = (
        "TEST",
        now,
        1.0,
        100.0,
        1.0,
        1000.0,
        1.0,
        "bo_v9_s3_r31809_c0_l0",
        "long",
    )

    strategy.recent_profits = [-2.0, -1.0]
    assert strategy.custom_stake_amount(*args) == pytest.approx(40.0)

    strategy.recent_profits = [-2.0, 1.0]
    assert strategy.custom_stake_amount(*args) == pytest.approx(100.0)

    assert (
        BreakoutV17GridV16EarlyBridgeE80RejectedResearchFreqtrade
        .V17_UNDERWATER_EARLY_MIN_DRAWDOWN
        == pytest.approx(0.13)
    )
    assert (
        BreakoutV17GridV16EarlyBridgeE80RejectedResearchFreqtrade
        .V17_UNDERWATER_EARLY_LOSS_STREAK
        == 2
    )
    assert (
        BreakoutV17GridV16EarlyBridgeE80RejectedResearchFreqtrade
        .V17_UNDERWATER_EARLY_SCALE
        == pytest.approx(0.80)
    )
    assert (
        BreakoutV17GridV16ParetoFinalResearchFreqtrade
        .V17_UNDERWATER_SCALE
        == pytest.approx(0.35)
    )
    assert (
        BreakoutV17GridV16ParetoFinalResearchFreqtrade
        .V17_UNDERWATER_EARLY_LOSS_STREAK
        == 0
    )


def test_underwater_fill_persists_the_applied_scale() -> None:
    strategy = _UnderwaterScore3RiskHarness()
    strategy.drawdown = 0.16
    trade = _Trade("bo_v9_s3_r31809_c0_l0", is_short=False)
    trade.entry_side = "buy"
    trade.nr_of_successful_entries = 1

    strategy.order_filled(
        "TEST",
        trade,
        SimpleNamespace(ft_order_side="buy"),
        datetime.now(timezone.utc),
    )

    assert trade.get_custom_data(
        strategy.V17_UNDERWATER_SCALE_KEY
    ) == pytest.approx(0.40)


def test_realized_drawdown_rebuilds_missed_close_time_high_water(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 4, tzinfo=timezone.utc)
    closed = [
        SimpleNamespace(
            close_date_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
            close_profit_abs=30.0,
        ),
        SimpleNamespace(
            close_date_utc=datetime(2026, 1, 3, tzinfo=timezone.utc),
            close_profit_abs=-20.0,
        ),
        SimpleNamespace(
            close_date_utc=datetime(2026, 1, 5, tzinfo=timezone.utc),
            close_profit_abs=100.0,
        ),
    ]
    monkeypatch.setattr(
        research.Trade,
        "get_trades_proxy",
        staticmethod(lambda **kwargs: closed),
    )

    drawdown = _RealizedLedgerHarness()._v17_realized_drawdown(now)

    assert drawdown == pytest.approx(1.0 - 210.0 / 230.0)


def test_reserved_savings_rebuild_from_closed_scaled_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 4, tzinfo=timezone.utc)

    class _ClosedTrade:
        def __init__(
            self,
            close_time: datetime,
            profit: float,
            scale: float,
        ) -> None:
            self.close_date_utc = close_time
            self.close_profit_abs = profit
            self.scale = scale

        def get_custom_data(self, key: str, default: Any = None) -> Any:
            return self.scale

    closed = [
        _ClosedTrade(
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            -4.0,
            0.40,
        ),
        _ClosedTrade(
            datetime(2026, 1, 3, tzinfo=timezone.utc),
            5.0,
            1.0,
        ),
        _ClosedTrade(
            datetime(2026, 1, 5, tzinfo=timezone.utc),
            -100.0,
            0.40,
        ),
    ]
    monkeypatch.setattr(
        research.Trade,
        "get_trades_proxy",
        staticmethod(lambda **kwargs: closed),
    )

    reserve = _RealizedLedgerHarness()._v17_reserved_savings(now)

    assert reserve == pytest.approx(6.0)


def test_precision_clock_accepts_live_millisecond_index_and_microseconds() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex(
                pd.to_datetime(["2026-08-19T08:00:00Z"])
            ).as_unit("ms"),
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
        }
    )

    class _Provider:
        runmode = SimpleNamespace(value="live")

        @staticmethod
        def get_pair_dataframe(pair: str, timeframe: str) -> pd.DataFrame:
            return frame.copy()

    strategy = _ClockHarness()
    strategy.dp = _Provider()
    candle = strategy._latest_completed_precision_candle(
        "TEST/USDT:USDT",
        datetime(2026, 8, 19, 8, 1, 0, 654321, tzinfo=timezone.utc),
    )

    assert candle is not None
    assert float(candle["close"]) == pytest.approx(1.0)
