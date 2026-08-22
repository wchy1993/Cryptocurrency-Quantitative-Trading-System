from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV17GridV16DynamicPositionResearchFreqtrade import (  # noqa: E402
    BreakoutV17GridV16DynamicPositionResearchFreqtrade,
    BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade,
    BreakoutV17GridV16DynamicProfitReversalOnlyResearchFreqtrade,
    BreakoutV17GridV16DynamicPositionWideRunnerResearchFreqtrade,
    BreakoutV17GridV16DynamicNoFollowConservativeResearchFreqtrade,
    BreakoutV17GridV16DynamicNoFollowModerateResearchFreqtrade,
    BreakoutV17GridV16DynamicNoFollowLongOnlyResearchFreqtrade,
    BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade,
    BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade,
    BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade,
    _V18DynamicShadowWalletProxy,
    _V18DynamicPositionManagementMixin,
)


class _Trade:
    def __init__(self, score=4, is_short=False):
        self.is_short = is_short
        self._score = score
        self._custom = {}

    def get_custom_data(self, key, default=None):
        if key == "bo_score":
            return self._score
        return self._custom.get(key, default)

    def set_custom_data(self, key, value):
        self._custom[key] = value


class _ContextHarness(_V18DynamicPositionManagementMixin):
    row = None

    def _latest_row(self, _pair, _current_time):
        return self.row

    @staticmethod
    def _closed_trade_return(trade):
        return trade.close_profit


def test_floor_schedule_only_tightens_as_peak_advances():
    schedule = ((0.75, -0.15), (1.5, 0.15), (3.0, 0.75))
    assert _V18DynamicPositionManagementMixin._v18_floor_for_peak(0.5, schedule) is None
    assert _V18DynamicPositionManagementMixin._v18_floor_for_peak(0.8, schedule) == -0.15
    assert _V18DynamicPositionManagementMixin._v18_floor_for_peak(2.0, schedule) == 0.15
    assert _V18DynamicPositionManagementMixin._v18_floor_for_peak(4.0, schedule) == 0.75


def test_completed_context_separates_reversal_from_strong_runner():
    harness = _ContextHarness()
    harness.row = {
        "atr": 2.0,
        "symbol_return_4h": -0.02,
        "fast_ema": 98.0,
        "slow_ema": 100.0,
        "market_efficiency": 0.60,
    }
    reversal, strong = harness._v18_directional_context(
        "TEST/USDT:USDT",
        _Trade(is_short=False),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert reversal and not strong

    harness.row.update(
        symbol_return_4h=0.02,
        fast_ema=102.0,
        slow_ema=100.0,
    )
    reversal, strong = harness._v18_directional_context(
        "TEST/USDT:USDT",
        _Trade(is_short=False),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert strong and not reversal


def test_strong_high_score_trade_uses_wide_runner_schedule():
    harness = _ContextHarness()
    trade = _Trade(score=4)
    assert harness._v18_breakout_locked_r(
        trade,
        peak_r=4.0,
        reversal=False,
        strong=True,
    ) is None
    assert harness._v18_breakout_locked_r(
        trade,
        peak_r=4.0,
        reversal=True,
        strong=False,
    ) == 0.75


def test_wide_neighbour_is_stricter_about_early_exit_and_looser_on_runner():
    assert (
        BreakoutV17GridV16DynamicPositionWideRunnerResearchFreqtrade.V18_BO_EARLY_MAX_CURRENT_R
        < BreakoutV17GridV16DynamicPositionResearchFreqtrade.V18_BO_EARLY_MAX_CURRENT_R
    )
    wide_floor = BreakoutV17GridV16DynamicPositionWideRunnerResearchFreqtrade._v18_floor_for_peak(
        2.0,
        BreakoutV17GridV16DynamicPositionWideRunnerResearchFreqtrade.V18_BO_REVERSAL_FLOORS,
    )
    balanced_floor = BreakoutV17GridV16DynamicPositionResearchFreqtrade._v18_floor_for_peak(
        2.0,
        BreakoutV17GridV16DynamicPositionResearchFreqtrade.V18_BO_REVERSAL_FLOORS,
    )
    assert wide_floor < balanced_floor


def test_ablation_flags_separate_loss_and_profit_management():
    assert BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade.V18_ENABLE_BO_EARLY_FAILURE
    assert not BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade.V18_ENABLE_BO_REVERSAL_FLOOR
    assert not BreakoutV17GridV16DynamicProfitReversalOnlyResearchFreqtrade.V18_ENABLE_BO_EARLY_FAILURE
    assert BreakoutV17GridV16DynamicProfitReversalOnlyResearchFreqtrade.V18_ENABLE_BO_REVERSAL_FLOOR


class _NoFollowTrade(_Trade):
    def __init__(
        self,
        *,
        max_rate=100.0,
        min_rate=100.0,
        is_short=False,
        score=4,
        capture=False,
    ):
        super().__init__(score=score, is_short=is_short)
        self._custom["bo_capture"] = capture
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.open_rate = 100.0
        self.open_date_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _NoFollowHarness(_ContextHarness):
    V18_ENABLE_BO_NO_FOLLOW_STOP = True
    V18_BO_NO_FOLLOW_MINUTES = 15
    V18_BO_NO_FOLLOW_MAX_BEST_R = 0.10
    V18_BO_NO_FOLLOW_FLOOR_R = -0.10
    V18_BO_NO_FOLLOW_MAX_SCORE = 4
    V18_BO_NO_FOLLOW_INCLUDE_CAPTURE = False

    @staticmethod
    def _trade_open_time(trade):
        return trade.open_date_utc

    @staticmethod
    def _trade_r(_trade, rate):
        return (rate - 100.0) / 10.0

    @staticmethod
    def _latest_completed_precision_candle(_pair, _current_time):
        return None


def test_no_follow_floor_arms_only_after_time_without_progress():
    harness = _NoFollowHarness()
    trade = _NoFollowTrade(max_rate=100.5)
    assert harness._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 14, tzinfo=timezone.utc),
        99.0,
    ) is None
    assert harness._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        99.0,
    ) == -0.10

    runner = _NoFollowTrade(max_rate=101.1)
    assert harness._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        runner,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        99.0,
    ) is None


def test_no_follow_neighbours_bound_progress_and_loss_floor():
    assert (
        BreakoutV17GridV16DynamicNoFollowConservativeResearchFreqtrade.V18_BO_NO_FOLLOW_MAX_BEST_R
        < BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade.V18_BO_NO_FOLLOW_MAX_BEST_R
    )
    assert (
        BreakoutV17GridV16DynamicNoFollowModerateResearchFreqtrade.V18_BO_NO_FOLLOW_FLOOR_R
        < BreakoutV17GridV16DynamicEarlyLossOnlyResearchFreqtrade.V18_BO_NO_FOLLOW_FLOOR_R
    )


def test_completed_minute_high_persistently_disarms_no_follow_stop():
    harness = _NoFollowHarness()
    harness._latest_completed_precision_candle = lambda _pair, _time: {
        "date": datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc),
        "high": 101.5,
        "low": 99.0,
    }
    trade = _NoFollowTrade(max_rate=100.0)
    assert harness._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc),
        99.0,
    ) is None
    harness._latest_completed_precision_candle = lambda _pair, _time: {
        "date": datetime(2026, 1, 1, 0, 14, tzinfo=timezone.utc),
        "high": 100.0,
        "low": 98.0,
    }
    assert harness._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        98.0,
    ) is None
    assert trade.get_custom_data("v18_no_follow_best_r") == 0.15
    assert trade.get_custom_data("v18_no_follow_disarmed") is True


def test_no_follow_protection_exempts_capture_and_score_five_runners():
    harness = _NoFollowHarness()
    now = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)
    for trade in (
        _NoFollowTrade(score=5),
        _NoFollowTrade(score=3, capture=True),
    ):
        assert harness._v18_no_follow_locked_r(
            "TEST/USDT:USDT",
            trade,
            now,
            98.0,
        ) is None


def test_long_only_neighbour_never_arms_for_shorts():
    strategy = object.__new__(
        BreakoutV17GridV16DynamicNoFollowLongOnlyResearchFreqtrade
    )
    trade = _NoFollowTrade(is_short=True, score=3)
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        102.0,
    ) is None


def test_stress_band_neighbour_arms_only_inside_narrow_drawdown_window():
    strategy = object.__new__(
        BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade
    )
    strategy._latest_completed_precision_candle = lambda _pair, _time: None
    strategy._trade_open_time = lambda candidate: candidate.open_date_utc
    strategy._trade_r = lambda _candidate, rate: (rate - 100.0) / 10.0

    trade = _NoFollowTrade(is_short=False, score=3)
    strategy._v17_realized_drawdown = lambda _time: 0.079
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        99.0,
    ) is None

    assert strategy.V18_BO_NO_FOLLOW_RESERVE_SCALE == 1.0
    assert strategy.V18_BO_NO_FOLLOW_PARTIAL_FRACTION == 0.90
    strategy._v17_realized_drawdown = lambda _time: 0.085
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 16, tzinfo=timezone.utc),
        99.0,
    ) is None

    trade = _NoFollowTrade(is_short=False, score=3)
    strategy._v17_realized_drawdown = lambda _time: 0.085
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        99.0,
    ) == -0.10

    trade = _NoFollowTrade(is_short=False, score=3)
    strategy._v17_realized_drawdown = lambda _time: 0.10
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        trade,
        datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc),
        99.0,
    ) is None


def test_protected_reserve_reduces_only_reinvestable_equity():
    scale = _V18DynamicPositionManagementMixin._v18_shadow_stake_scale(
        equity=220.0,
        reserve=20.0,
    )
    assert scale == 200.0 / 220.0
    assert (
        _V18DynamicPositionManagementMixin._v18_shadow_stake_scale(
            equity=220.0,
            reserve=0.0,
        )
        == 1.0
    )


def test_dynamic_shadow_wallet_hides_reserve_from_parent_governor():
    wallets = SimpleNamespace(
        get_total_stake_amount=lambda: 230.0,
        get_starting_balance=lambda: 200.0,
    )
    shadow = _V18DynamicShadowWalletProxy(wallets, 18.0)
    assert shadow.get_total_stake_amount() == 212.0
    assert shadow.get_starting_balance() == 200.0


def test_counterfactual_ledger_overrides_dynamic_exit_profit(monkeypatch):
    harness = _ContextHarness()
    harness.PORTFOLIO_RECENT_WINDOW = 2
    harness.wallets = SimpleNamespace(get_starting_balance=lambda: 100.0)
    harness._trade_close_time = lambda candidate: candidate.close_time

    first = _Trade()
    first.close_time = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
    first.close_profit = 0.10
    first.close_profit_abs = 10.0

    dynamic = _Trade()
    dynamic.close_time = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    dynamic.close_profit = -0.02
    dynamic.close_profit_abs = -2.0
    dynamic.set_custom_data("v18_no_follow_shadow_return", -0.20)
    dynamic.set_custom_data("v18_no_follow_shadow_profit_abs", -20.0)

    monkeypatch.setattr(
        "BreakoutV17GridV16DynamicPositionResearchFreqtrade.Trade.get_trades_proxy",
        lambda **_kwargs: [first, dynamic],
    )
    now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
    assert harness._recent_portfolio_returns(now) == (0.10, -0.20)
    assert abs(harness._v17_realized_drawdown(now) - (1.0 - 90.0 / 110.0)) < 1e-12


def test_protected_reserve_reconciles_after_balance_multiple(monkeypatch):
    harness = _ContextHarness()
    harness.V18_BO_NO_FOLLOW_RESERVE_SCALE = 1.0005
    harness.V18_BO_NO_FOLLOW_RECONCILE_BALANCE_MULTIPLE = 2.0
    harness.V18_BO_NO_FOLLOW_RECONCILE_SCALE = 1.0
    total = {"value": 150.0}
    harness.wallets = SimpleNamespace(
        get_starting_balance=lambda: 100.0,
        get_total_stake_amount=lambda: total["value"],
    )
    harness._trade_close_time = lambda candidate: candidate.close_time

    dynamic = _Trade()
    dynamic.close_time = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    dynamic.close_profit_abs = -2.0
    dynamic.set_custom_data("v18_no_follow_protected_reserve", 10.0)
    dynamic.set_custom_data("v18_no_follow_shadow_profit_abs", -20.0)
    monkeypatch.setattr(
        "BreakoutV17GridV16DynamicPositionResearchFreqtrade.Trade.get_trades_proxy",
        lambda **_kwargs: [dynamic],
    )
    now = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
    assert abs(harness._v18_no_follow_protected_reserve(now) - 10.005) < 1e-12
    total["value"] = 250.0
    assert harness._v18_no_follow_protected_reserve(now) == 18.0


def test_flat_tail_exit_applies_only_to_first_score_three_rebuild(monkeypatch):
    parent = BreakoutV17GridV16DynamicNoFollowStressBandLongResearchFreqtrade
    monkeypatch.setattr(
        parent,
        "adjust_trade_position",
        lambda *_args, **_kwargs: (25.0, "grid_dca_0"),
    )
    strategy = object.__new__(
        BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade
    )
    strategy._latest_row = lambda _pair, _time: {"symbol_return_4h": 0.001}
    trade = SimpleNamespace(
        pair="TEST/USDT:USDT",
        enter_tag="grid_v8_short_s3",
        nr_of_successful_entries=1,
        nr_of_successful_exits=1,
        stake_amount=10.0,
    )
    args = (
        trade,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        100.0,
        -0.01,
        None,
        100.0,
        100.0,
        100.0,
        -0.01,
        -0.01,
    )
    assert strategy.adjust_trade_position(*args) == (
        -10.0,
        strategy.V18_GRID_POST_TP_FLAT_EXIT_TAG,
    )

    trade.nr_of_successful_entries = 2
    assert strategy.adjust_trade_position(*args) == (25.0, "grid_dca_0")
    trade.nr_of_successful_entries = 1
    strategy._latest_row = lambda _pair, _time: {"symbol_return_4h": 0.003}
    assert strategy.adjust_trade_position(*args) == (25.0, "grid_dca_0")


def test_saturated_grid_partial_requires_severe_completed_reversal(monkeypatch):
    parent = (
        BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade
    )
    monkeypatch.setattr(
        parent,
        "adjust_trade_position",
        lambda *_args, **_kwargs: None,
    )
    strategy = object.__new__(
        BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade
    )
    opened = datetime(2026, 1, 1, tzinfo=timezone.utc)
    strategy._trade_open_time = lambda _trade: opened
    strategy._latest_row = lambda _pair, _time: {"symbol_return_4h": 0.016}
    strategy._v18_grid_campaign_r = lambda _trade, _rate: (0.0, -0.34)
    custom = {}
    trade = SimpleNamespace(
        pair="TEST/USDT:USDT",
        enter_tag="grid_v8_short_s4",
        nr_of_successful_entries=3,
        nr_of_successful_exits=0,
        stake_amount=100.0,
        get_custom_data=lambda key, default=None: custom.get(key, default),
        set_custom_data=lambda key, value: custom.__setitem__(key, value),
    )
    now = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    args = ("TEST/USDT:USDT", trade, now, 101.0, -0.01)
    assert strategy._v18_saturated_grid_armed(*args)

    strategy._latest_row = lambda _pair, _time: {"symbol_return_4h": 0.014}
    assert not strategy._v18_saturated_grid_armed(*args)
    strategy._latest_row = lambda _pair, _time: {"symbol_return_4h": 0.016}
    strategy._v18_grid_campaign_r = lambda _trade, _rate: (0.0, -0.30)
    assert not strategy._v18_saturated_grid_armed(*args)
    strategy._v18_grid_campaign_r = lambda _trade, _rate: (0.0, -0.34)
    deep_loss_args = ("TEST/USDT:USDT", trade, now, 101.0, -0.07)
    assert not strategy._v18_saturated_grid_armed(*deep_loss_args)

    adjust_args = (
        trade,
        now,
        101.0,
        -0.01,
        None,
        100.0,
        100.0,
        101.0,
        -0.01,
        -0.01,
    )
    assert strategy.adjust_trade_position(*adjust_args) == (
        -90.0,
        strategy.V18_GRID_SATURATED_EXIT_TAG,
    )
    assert strategy.adjust_trade_position(*adjust_args) is None


def test_risk_balanced_no_follow_is_limited_to_score_four(monkeypatch):
    parent = (
        BreakoutV17GridV16DynamicNoFollowGridScore3FlatTailExitResearchFreqtrade
    )
    monkeypatch.setattr(
        parent,
        "_v18_no_follow_locked_r",
        lambda *_args, **_kwargs: -0.10,
    )
    strategy = object.__new__(
        BreakoutV17GridV16DynamicRiskBalancedResearchFreqtrade
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        _Trade(score=3),
        now,
        99.0,
    ) is None
    assert strategy._v18_no_follow_locked_r(
        "TEST/USDT:USDT",
        _Trade(score=4),
        now,
        99.0,
    ) == -0.10
