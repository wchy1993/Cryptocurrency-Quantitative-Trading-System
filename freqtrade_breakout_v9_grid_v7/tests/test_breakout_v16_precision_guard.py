from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

from ccxt import TICK_SIZE
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from freqtrade.strategy import stoploss_from_absolute  # noqa: E402
from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade,
    BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade,
    BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade,
    _PrecisionConfirmedInitialStopMixin,
)
from BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade,
)


class _Trade:
    def __init__(self, *, is_short: bool = True) -> None:
        self.pair = "FET/USDT:USDT"
        self.enter_tag = "bo_v9_s4_r43320_c1_l0"
        self.open_rate = 0.1280
        self.is_short = is_short
        self.leverage = 10.0
        self.price_precision = 0.0001
        self.precision_mode_price = TICK_SIZE
        self.stop_loss = 0.0
        self.stop_adjustments: list[tuple[float, float, bool]] = []
        self.custom: dict[str, Any] = {
            "initial_atr": 0.0011442505108449068,
        }

    def get_custom_data(self, key: str, default: object = None) -> object:
        return self.custom.get(key, default)

    def set_custom_data(self, key: str, value: object) -> None:
        self.custom[key] = value

    def adjust_stop_loss(
        self,
        current_price: float,
        stoploss: float,
        *,
        allow_refresh: bool = False,
    ) -> None:
        self.stop_adjustments.append(
            (float(current_price), float(stoploss), bool(allow_refresh))
        )
        distance = abs(float(stoploss)) / self.leverage
        self.stop_loss = float(current_price) * (
            1.0 + distance if self.is_short else 1.0 - distance
        )


class _Parent:
    BO_STOP_ATR = 0.77
    SIDE_COST = 0.0008

    @staticmethod
    def _component(tag: str | None) -> str | None:
        return "breakout" if str(tag or "").startswith("bo_") else "grid"

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
        stop = float(getattr(self, "parent_stop", 0.0) or 0.0)
        if stop <= 0.0:
            atr = float(trade.get_custom_data("initial_atr", 0.0))
            stop = trade.open_rate + (1.0 if trade.is_short else -1.0) * (
                self.BO_STOP_ATR * atr
            )
        return float(
            stoploss_from_absolute(
                stop,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )
        )

    def custom_exit(self, *args: Any, **kwargs: Any) -> None:
        return None

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []


class _Harness(_PrecisionConfirmedInitialStopMixin, _Parent):
    pass


class _Provider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.runmode = SimpleNamespace(value="backtest")
        self._exchange = SimpleNamespace(precision_mode_price=TICK_SIZE)

    def current_whitelist(self) -> list[str]:
        return ["FET/USDT:USDT"]

    def market(self, pair: str) -> dict[str, object]:
        return {"precision": {"price": 0.0001}}

    def get_pair_dataframe(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        assert timeframe == "1m"
        return self.frame.copy()


class _HistoricProvider(_Provider):
    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__(frame)
        self.history_calls = 0

    def historic_ohlcv(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        assert timeframe == "1m"
        self.history_calls += 1
        return self.frame.copy()

    def get_pair_dataframe(
        self,
        pair: str,
        timeframe: str,
    ) -> pd.DataFrame:
        raise AssertionError("backtest path must use the causal history cache")


def _harness(frame: pd.DataFrame | None = None) -> _Harness:
    strategy = _Harness()
    strategy.dp = _Provider(
        frame
        if frame is not None
        else pd.DataFrame(
            columns=["date", "open", "high", "low", "close"]
        )
    )
    return strategy


def test_selected_overlay_leaves_frozen_signal_and_grid_methods_owned_by_parent() -> None:
    selected = BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade
    frozen = BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade

    assert selected.populate_indicators is frozen.populate_indicators
    assert selected.populate_entry_trend is frozen.populate_entry_trend
    assert selected.leverage is frozen.leverage
    assert selected.adjust_trade_position is frozen.adjust_trade_position
    assert selected.custom_stoploss is not frozen.custom_stoploss
    assert selected.custom_exit is not frozen.custom_exit
    assert selected.custom_stake_amount is not frozen.custom_stake_amount


def test_production_class_is_the_exact_global_research_selection() -> None:
    production = BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade
    selected = BreakoutV16GridV15PrecisionGuardGlobalResearchFreqtrade

    assert production.PRECISION_GUARD_MIN_TICK_ATR == 0.0
    assert production.populate_indicators is selected.populate_indicators
    assert production.populate_entry_trend is selected.populate_entry_trend
    assert production.custom_stake_amount is selected.custom_stake_amount
    assert production.custom_stoploss is selected.custom_stoploss
    assert production.custom_exit is selected.custom_exit
    assert production.order_types["stoploss_on_exchange"] is True
    assert production.order_types["stoploss_on_exchange_interval"] == 30
    assert production.order_types["stoploss_price_type"] == "last"


def test_fet_plan_rounds_outward_arms_hard_stop_and_preserves_u_risk() -> None:
    strategy = _harness()
    atr = 0.0011442505108449068
    plan = strategy._precision_stop_plan(
        0.1280,
        atr,
        True,
        0.0001,
        TICK_SIZE,
    )

    assert plan is not None
    assert plan.soft_stop == pytest.approx(0.12888107289335057)
    assert plan.inward_stop == pytest.approx(0.1288)
    assert plan.outward_stop == pytest.approx(0.1289)
    assert plan.hard_stop == pytest.approx(0.1290)
    assert plan.tick_atr == pytest.approx(0.0873934501686704)
    assert plan.confirmation_enabled is True

    original_risk = (
        abs(plan.soft_stop - 0.1280)
        + 2.0 * strategy.SIDE_COST * 0.1280
    )
    guarded_risk = (
        abs(plan.hard_stop - 0.1280)
        + 2.0 * strategy.SIDE_COST * 0.1280
    )
    assert guarded_risk * plan.stake_scale == pytest.approx(original_risk)
    assert plan.stake_scale == pytest.approx(0.9012889221)


def test_fine_tick_market_keeps_immediate_stop_without_atr_buffer() -> None:
    strategy = _harness()
    plan = strategy._precision_stop_plan(
        100.0,
        2.0,
        True,
        0.001,
        TICK_SIZE,
    )

    assert plan is not None
    assert plan.confirmation_enabled is False
    assert plan.hard_stop == plan.outward_stop
    assert plan.hard_stop >= plan.soft_stop
    assert plan.stake_scale > 0.999


def test_long_rounding_and_hard_buffer_are_directionally_symmetric() -> None:
    strategy = _harness()
    plan = strategy._precision_stop_plan(
        0.1280,
        0.0011442505108449068,
        False,
        0.0001,
        TICK_SIZE,
    )

    assert plan is not None
    assert plan.inward_stop == pytest.approx(0.1272)
    assert plan.outward_stop == pytest.approx(0.1271)
    assert plan.hard_stop == pytest.approx(0.1270)
    assert plan.confirmation_enabled is True


def test_initial_parent_stop_is_replaced_by_hard_stop_but_profit_lock_wins() -> None:
    strategy = _harness()
    trade = _Trade()
    now = datetime(2026, 8, 15, 22, 0, tzinfo=timezone.utc)

    ratio = strategy.custom_stoploss(
        trade.pair,
        trade,
        now,
        current_rate=0.1280,
        current_profit=0.0,
        after_fill=False,
    )
    hard = strategy._absolute_stop_from_ratio(ratio, 0.1280, trade)
    assert hard == pytest.approx(0.1290)

    strategy.parent_stop = 0.1270
    protected_ratio = strategy.custom_stoploss(
        trade.pair,
        trade,
        now,
        current_rate=0.1260,
        current_profit=0.1,
        after_fill=False,
    )
    protected = strategy._absolute_stop_from_ratio(
        protected_ratio,
        0.1260,
        trade,
    )
    assert protected == pytest.approx(0.1270)


def test_filled_trade_is_armed_at_hard_stop_before_next_engine_cycle() -> None:
    strategy = _harness()
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)

    assert plan is not None
    strategy._arm_hard_stop_on_filled_trade(trade, plan)

    assert trade.stop_loss == pytest.approx(plan.hard_stop)
    assert len(trade.stop_adjustments) == 1
    assert trade.stop_adjustments[0][2] is True


def test_fet_exact_top_does_not_confirm_soft_stop() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-15T22:24:00Z"]),
            "open": [0.1284],
            "high": [0.1288],
            "low": [0.1284],
            "close": [0.1287],
        }
    )
    strategy = _harness(frame)
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)
    assert plan is not None
    trade.stop_loss = plan.hard_stop

    reason = strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2026, 8, 15, 22, 25, tzinfo=timezone.utc),
        current_rate=0.1288,
        current_profit=-0.07,
    )
    assert reason is None


def test_one_minute_close_beyond_soft_stop_exits_before_hard_stop() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-15T22:24:00Z"]),
            "open": [0.1288],
            "high": [0.1289],
            "low": [0.1288],
            "close": [0.1289],
        }
    )
    strategy = _harness(frame)
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)
    assert plan is not None
    assert frame.iloc[0]["high"] < plan.hard_stop
    trade.stop_loss = plan.hard_stop

    reason = strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2026, 8, 15, 22, 25, tzinfo=timezone.utc),
        current_rate=0.1289,
        current_profit=-0.08,
    )
    assert reason == strategy.PRECISION_GUARD_EXIT_REASON


def test_live_microsecond_clock_handles_millisecond_candle_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.Series(
                pd.to_datetime(["2026-08-15T22:24:00Z"]),
                dtype="datetime64[ms, UTC]",
            ),
            "open": [0.1288],
            "high": [0.1289],
            "low": [0.1288],
            "close": [0.1289],
        }
    )
    strategy = _harness(frame)
    strategy.dp.runmode = SimpleNamespace(value="live")
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)
    assert plan is not None
    trade.stop_loss = plan.hard_stop

    reason = strategy.custom_exit(
        trade.pair,
        trade,
        datetime(
            2026,
            8,
            15,
            22,
            25,
            0,
            123456,
            tzinfo=timezone.utc,
        ),
        current_rate=0.1289,
        current_profit=-0.08,
    )

    assert reason == strategy.PRECISION_GUARD_EXIT_REASON


def test_stale_minute_data_cannot_generate_delayed_exit() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-15T22:24:00Z"]),
            "open": [0.1288],
            "high": [0.1289],
            "low": [0.1288],
            "close": [0.1289],
        }
    )
    strategy = _harness(frame)
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)
    assert plan is not None
    trade.stop_loss = plan.hard_stop

    reason = strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2026, 8, 15, 22, 27, tzinfo=timezone.utc),
        current_rate=0.1287,
        current_profit=-0.05,
    )
    assert reason is None


def test_backtest_history_cache_is_causal_and_loaded_once() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-08-15T22:24:00Z",
                    "2026-08-15T22:25:00Z",
                ]
            ),
            "open": [0.1288, 0.1287],
            "high": [0.1289, 0.1287],
            "low": [0.1288, 0.1286],
            "close": [0.1289, 0.1286],
        }
    )
    provider = _HistoricProvider(frame)
    strategy = _Harness()
    strategy.dp = provider
    trade = _Trade()
    plan = strategy._trade_stop_plan(trade)
    assert plan is not None
    trade.stop_loss = plan.hard_stop
    at_boundary = datetime(2026, 8, 15, 22, 25, tzinfo=timezone.utc)

    first = strategy.custom_exit(
        trade.pair,
        trade,
        at_boundary,
        current_rate=0.1289,
        current_profit=-0.08,
    )
    second = strategy.custom_exit(
        trade.pair,
        trade,
        at_boundary,
        current_rate=0.1289,
        current_profit=-0.08,
    )

    assert first == strategy.PRECISION_GUARD_EXIT_REASON
    assert second == strategy.PRECISION_GUARD_EXIT_REASON
    assert provider.history_calls == 1
