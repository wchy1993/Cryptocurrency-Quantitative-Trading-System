from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from freqtrade.strategy import stoploss_from_absolute  # noqa: E402
from BreakoutV16GridV15SplitTrailRunnerResearchFreqtrade import (  # noqa: E402
    BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade,
    BreakoutV16GridV15SplitTrailV3Floor30ResearchFreqtrade,
    BreakoutV16GridV15SplitTrailV3Floor50ResearchFreqtrade,
    BreakoutV16GridV15SplitTrailV3Trigger60ResearchFreqtrade,
    _SplitTrailRunnerMixin,
)
from BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade,
)


class _Trade:
    def __init__(self, component: str, is_short: bool) -> None:
        self.enter_tag = "grid_v8_short_s4" if component == "grid" else "bo_test"
        self.is_short = is_short
        self.open_rate = 1.0
        self.leverage = 1.0
        self.min_rate = 0.95 if is_short else 1.0
        self.max_rate = 1.05 if not is_short else 1.0
        self.open_date_utc = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.custom: dict[str, Any] = {
            "initial_unit_risk": 0.10,
            "grid_risk_budget": 100.0,
            "grid_hard_stop": 1.20,
            "bo_score": 4,
        }

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        return self.custom.get(key, default)

    def set_custom_data(self, key: str, value: Any) -> None:
        self.custom[key] = value

    def calculate_profit(self, rate: float) -> SimpleNamespace:
        # A fee-free short campaign with 1,000 quote units of price exposure.
        total = (self.open_rate - float(rate)) * 1000.0
        return SimpleNamespace(total_profit=total)


class _Parent:
    SIDE_COST = 0.0
    parent_stop = 0.50
    parent_exit: str | None = None

    @staticmethod
    def _component(tag: str | None) -> str:
        return "grid" if str(tag or "").startswith("grid") else "breakout"

    @staticmethod
    def _trade_open_time(trade: _Trade) -> datetime:
        return trade.open_date_utc

    @staticmethod
    def _trade_r(trade: _Trade, rate: float) -> float:
        side = -1.0 if trade.is_short else 1.0
        return side * (float(rate) - trade.open_rate) / 0.01

    def _confirmed_peak_r(
        self,
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
            "atr": 0.01,
            "symbol_ema55_atr": 2.0,
            "fast_ema": 1.02,
            "slow_ema": 1.01,
            "symbol_return_4h": 0.03,
            "split_close_8h_ago": 0.98,
            "close": 1.04,
            "breadth": 0.70,
            "market_efficiency": 0.50,
        }

    def custom_stoploss(
        self,
        pair: str,
        trade: _Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float:
        return float(self.parent_stop)

    def custom_exit(self, *args: Any, **kwargs: Any) -> str | None:
        return self.parent_exit


class _Harness(_SplitTrailRunnerMixin, _Parent):
    pass


def test_selected_v3_is_exact_conservative_path_without_runner() -> None:
    production = BreakoutV16GridV15SplitTrailV3LiveParityFreqtrade
    selected = BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade

    assert production.populate_entry_trend is selected.populate_entry_trend
    assert production.custom_stake_amount is selected.custom_stake_amount
    assert production.custom_stoploss is selected.custom_stoploss
    assert production.custom_exit is selected.custom_exit
    assert production.GRID_SHORT_SPLIT_MIN_TP_COUNT == 1
    assert production.GRID_SHORT_SPLIT_TRIGGER_R == pytest.approx(0.50)
    assert production.GRID_SHORT_SPLIT_FLOOR_R == pytest.approx(-0.40)
    assert production.SPLIT_RUNNER_ENABLED is False


def test_v3_stress_neighborhood_changes_only_grid_floor_constants() -> None:
    selected = BreakoutV16GridV15SplitTrailV3ConservativeResearchFreqtrade
    candidates = (
        BreakoutV16GridV15SplitTrailV3Floor30ResearchFreqtrade,
        BreakoutV16GridV15SplitTrailV3Floor50ResearchFreqtrade,
        BreakoutV16GridV15SplitTrailV3Trigger60ResearchFreqtrade,
    )

    assert [candidate.GRID_SHORT_SPLIT_TRIGGER_R for candidate in candidates] == [
        0.50,
        0.50,
        0.60,
    ]
    assert [candidate.GRID_SHORT_SPLIT_FLOOR_R for candidate in candidates] == [
        -0.30,
        -0.50,
        -0.40,
    ]
    assert all(
        candidate.GRID_SHORT_SPLIT_MIN_TP_COUNT == 1
        for candidate in candidates
    )
    assert all(
        candidate.populate_entry_trend is selected.populate_entry_trend
        for candidate in candidates
    )
    assert all(
        candidate.custom_stake_amount is selected.custom_stake_amount
        for candidate in candidates
    )
    assert all(
        candidate.custom_stoploss is selected.custom_stoploss
        for candidate in candidates
    )
    assert all(
        candidate.custom_exit is selected.custom_exit
        for candidate in candidates
    )


def test_breakout_long_retains_parent_stop_exactly() -> None:
    strategy = _Harness()
    trade = _Trade("breakout", is_short=False)
    value = strategy.custom_stoploss(
        "TEST/USDT:USDT",
        trade,
        datetime.now(timezone.utc),
        1.05,
        0.05,
        False,
    )
    assert value == strategy.parent_stop


def test_non_capture_breakout_short_gets_its_own_confirmed_floor() -> None:
    strategy = _Harness()
    trade = _Trade("breakout", is_short=True)
    trade.set_custom_data("confirmed_peak_r", 3.5)
    value = strategy.custom_stoploss(
        "TEST/USDT:USDT",
        trade,
        datetime.now(timezone.utc),
        0.80,
        0.20,
        False,
    )
    expected = abs(
        float(
            stoploss_from_absolute(
                1.025,
                current_rate=0.80,
                is_short=True,
                leverage=1.0,
            )
        )
    )
    assert value == pytest.approx(expected)

    trade.set_custom_data("bo_capture", True)
    assert strategy.custom_stoploss(
        "TEST/USDT:USDT",
        trade,
        datetime.now(timezone.utc),
        0.80,
        0.20,
        False,
    ) == pytest.approx(strategy.parent_stop)


def test_grid_short_floor_is_fee_aware_and_has_a_named_exit_fallback() -> None:
    strategy = _Harness()
    trade = _Trade("grid", is_short=True)
    now = datetime.now(timezone.utc)

    value = strategy.custom_stoploss(
        "TEST/USDT:USDT",
        trade,
        now,
        0.98,
        0.02,
        False,
    )
    expected = abs(
        float(
            stoploss_from_absolute(
                1.01,
                current_rate=0.98,
                is_short=True,
                leverage=1.0,
            )
        )
    )
    assert value == pytest.approx(expected, rel=1e-8)

    assert strategy.custom_exit(
        "TEST/USDT:USDT",
        trade,
        now,
        1.02,
        -0.02,
    ) == strategy.GRID_SHORT_SPLIT_EXIT_REASON


def test_conservative_grid_floor_requires_a_completed_partial_tp() -> None:
    strategy = _Harness()
    strategy.GRID_SHORT_SPLIT_MIN_TP_COUNT = 1
    trade = _Trade("grid", is_short=True)
    now = datetime.now(timezone.utc)

    assert strategy.custom_exit(
        "TEST/USDT:USDT",
        trade,
        now,
        1.02,
        -0.02,
    ) is None
    trade.set_custom_data("grid_tp_count", 1)
    assert strategy.custom_exit(
        "TEST/USDT:USDT",
        trade,
        now,
        1.02,
        -0.02,
    ) == strategy.GRID_SHORT_SPLIT_EXIT_REASON


def test_confirmed_runner_suppresses_20h_exit_but_remains_time_bounded() -> None:
    strategy = _Harness()
    strategy.SPLIT_RUNNER_ENABLED = True
    strategy.parent_exit = "bo_v9_time_stop"
    trade = _Trade("breakout", is_short=False)
    trade.set_custom_data("initial_unit_risk", 0.01)
    pair = "TEST/USDT:USDT"

    armed_at = trade.open_date_utc + timedelta(hours=20)
    assert strategy.custom_exit(
        pair,
        trade,
        armed_at,
        1.05,
        0.05,
    ) is None
    assert trade.get_custom_data("bo_v16_split_runner_extended") is True

    assert strategy.custom_exit(
        pair,
        trade,
        trade.open_date_utc + timedelta(hours=21),
        1.04,
        0.04,
    ) is None
    assert strategy.custom_exit(
        pair,
        trade,
        trade.open_date_utc + timedelta(hours=36),
        1.04,
        0.04,
    ) == "bo_v16_split_runner_time_stop"


def test_protected_runner_keeps_half_of_its_arming_r_profit() -> None:
    strategy = _Harness()
    strategy.SPLIT_RUNNER_ENABLED = True
    strategy.BO_RUNNER_PROTECT_ARMED_PROFIT = True
    strategy.parent_exit = "bo_v9_time_stop"
    trade = _Trade("breakout", is_short=False)
    trade.set_custom_data("initial_unit_risk", 0.01)
    pair = "TEST/USDT:USDT"
    armed_at = trade.open_date_utc + timedelta(hours=20)

    assert strategy.custom_exit(
        pair,
        trade,
        armed_at,
        1.05,
        0.05,
    ) is None
    value = strategy.custom_stoploss(
        pair,
        trade,
        armed_at + timedelta(minutes=1),
        1.04,
        0.04,
        False,
    )
    expected = abs(
        float(
            stoploss_from_absolute(
                1.025,
                current_rate=1.04,
                is_short=False,
                leverage=1.0,
            )
        )
    )
    assert value == pytest.approx(expected)
