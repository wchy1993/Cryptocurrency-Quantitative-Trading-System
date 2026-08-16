from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
GRID_STRATEGY_DIR = (
    PROJECT_DIR.parent
    / "freqtrade_grid_v8_dual_side"
    / "user_data"
    / "strategies"
)
for path in (STRATEGY_DIR, GRID_STRATEGY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade,
)
from BreakoutV16GridV15QualityPfCombinedResearchFreqtrade import (  # noqa: E402
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade,
)
from BreakoutV12MultiRegimeGridV9Freqtrade import (  # noqa: E402
    V12_CONTEXT_COLUMNS,
)


class _RunModeProvider:
    def __init__(self, mode: str) -> None:
        self.runmode = SimpleNamespace(value=mode)

    @staticmethod
    def current_whitelist() -> list[str]:
        return ["BTC/USDT:USDT", "ETH/USDT:USDT"]


PAIR = "BTC/USDT:USDT"
BATCH_TARGET = pd.Timestamp("2026-08-10T00:00:00Z")
AVAILABLE_AT = datetime(2026, 8, 10, 1, 0, tzinfo=timezone.utc)


def _strategy(
    *,
    mode: str = "live",
    applied_target: pd.Timestamp | None = BATCH_TARGET,
) -> BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade:
    strategy = object.__new__(
        BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade
    )
    strategy.dp = _RunModeProvider(mode)
    strategy._live_context_applied_at = applied_target
    strategy._ranked_candidates = lambda component, current_time: [
        (0.12, PAIR)
    ]
    strategy._latest_signal_row = lambda pair, component, current_time: (
        pd.Series({"v16_mtf_ready": 1})
    )
    return strategy


def _confirm(
    strategy: BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade,
    delay: float,
    entry_tag: str,
) -> bool:
    return strategy.confirm_trade_entry(
        pair=PAIR,
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="GTC",
        current_time=AVAILABLE_AT + timedelta(seconds=delay),
        entry_tag=entry_tag,
        side="short",
    )


def test_production_class_is_exact_combined_research_path() -> None:
    selected = BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade

    assert issubclass(
        selected,
        BreakoutV16GridV15QualityPfCombinedResearchFreqtrade,
    )
    assert selected.MAX_OPEN_TRADES == 2
    assert selected.V16_CONFIRM_TIMEFRAME == "15m"
    assert selected.GRID_V15_SCORE4_SCALE == pytest.approx(0.60)
    assert selected.GRID_V15_SCORE5_SCALE == pytest.approx(0.25)
    assert selected.GRID_V15_LONG_SCALE == pytest.approx(0.35)
    assert (
        selected.ADAPTIVE_STATE_BASENAME
        != BreakoutV16GridV15QualityPfCombinedResearchFreqtrade.
        ADAPTIVE_STATE_BASENAME
    )


def test_live_adapter_builds_and_attaches_complete_inherited_context() -> None:
    pairs = (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
        "BNB/USDT:USDT",
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
        "LTC/USDT:USDT",
    )
    dates = pd.date_range("2026-01-01T00:00:00Z", periods=620, freq="1h")
    snapshots: dict[str, pd.DataFrame] = {}
    for index, pair in enumerate(pairs):
        trend = np.linspace(100.0 + index, 135.0 + index, len(dates))
        wave = np.sin(np.arange(len(dates)) / (4.0 + index / 10.0))
        snapshots[pair] = pd.DataFrame(
            {"date": dates, "close": trend + wave}
        )

    provider = SimpleNamespace(
        runmode=SimpleNamespace(value="live"),
        current_whitelist=lambda: list(pairs),
    )
    strategy = object.__new__(
        BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade
    )
    strategy.dp = provider
    strategy._live_pair_snapshots = snapshots
    strategy._live_context_wait_signature = None

    target = strategy._prepare_live_context()

    assert target == dates[-1]
    cached = strategy._market_context_cache
    assert set(V12_CONTEXT_COLUMNS) <= set(cached.columns)
    latest = cached.loc[pd.to_datetime(cached["date"], utc=True) == target]
    assert len(latest) == 1
    assert not latest[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None)

    attached = strategy._attach_market_context(
        pd.DataFrame({"date": [target], "close": [1.0]})
    )
    assert set(V12_CONTEXT_COLUMNS) <= set(attached.columns)
    assert not attached[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None)


def test_90_second_expiry_is_live_instance_only() -> None:
    live = _strategy(mode="live")
    backtest = _strategy(mode="backtest")
    parent = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade

    with patch.object(parent, "bot_start", return_value=None):
        live.bot_start()
        backtest.bot_start()

    assert live.ignore_buying_expired_candle_after == 90
    assert (
        backtest.ignore_buying_expired_candle_after
        == parent.ignore_buying_expired_candle_after
    )


@pytest.mark.parametrize(
    "entry_tag",
    ("bo_v9_s5_r41351_c0_l0", "grid_v8_short_s6"),
)
def test_live_current_batch_reaches_parent_for_both_sleeves(
    entry_tag: str,
) -> None:
    parent = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    for delay in (0.0, 31.0, 55.973, 90.0):
        strategy = _strategy()
        with patch.object(
            parent,
            "confirm_trade_entry",
            return_value=True,
        ) as parent_confirm:
            assert _confirm(strategy, delay, entry_tag)
        parent_confirm.assert_called_once()


@pytest.mark.parametrize(
    "entry_tag",
    ("bo_v9_s5_r41351_c0_l0", "grid_v8_short_s6"),
)
def test_live_stale_missing_or_expired_batch_fails_closed_for_both_sleeves(
    entry_tag: str,
) -> None:
    parent = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    for target, delay in (
        (None, 10.0),
        (pd.Timestamp("2026-08-09T23:00:00Z"), 10.0),
        (BATCH_TARGET, 90.001),
    ):
        strategy = _strategy(applied_target=target)
        with patch.object(
            parent,
            "confirm_trade_entry",
            return_value=True,
        ) as parent_confirm:
            assert not _confirm(strategy, delay, entry_tag)
        parent_confirm.assert_not_called()


@pytest.mark.parametrize("path_ready", (0, float("nan")))
def test_live_breakout_requires_complete_v16_path_but_grid_does_not(
    path_ready: float,
) -> None:
    parent = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    strategy = _strategy()
    strategy._latest_signal_row = lambda pair, component, current_time: (
        pd.Series({"v16_mtf_ready": path_ready})
    )
    with patch.object(
        parent,
        "confirm_trade_entry",
        return_value=True,
    ) as parent_confirm:
        assert not _confirm(strategy, 31.0, "bo_v9_s5_r41351_c0_l0")
        assert _confirm(strategy, 31.0, "grid_v8_short_s6")
    parent_confirm.assert_called_once()


@pytest.mark.parametrize(
    "entry_tag",
    ("bo_v9_s5_r41351_c0_l0", "grid_v8_short_s6"),
)
def test_backtest_bypasses_only_wall_clock_gate(entry_tag: str) -> None:
    strategy = _strategy(mode="backtest", applied_target=None)
    parent = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    with patch.object(
        parent,
        "confirm_trade_entry",
        return_value=True,
    ) as parent_confirm:
        assert _confirm(strategy, 3_600.0, entry_tag)
    parent_confirm.assert_called_once()
