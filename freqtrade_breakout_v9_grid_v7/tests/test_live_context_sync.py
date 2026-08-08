from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal


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

import BreakoutV9GridV7Freqtrade as breakout_v9  # noqa: E402
from BreakoutV10FGridV8DualSideFreqtrade import (  # noqa: E402
    LIVE_CONTEXT_COLUMNS,
    BreakoutV10FGridV8DualSideFreqtrade,
    build_synchronized_market_context,
)


def _pairs() -> tuple[str, ...]:
    return (
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


def _snapshots(periods: int = 40) -> dict[str, DataFrame]:
    dates = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=periods,
        freq="1h",
    )
    output: dict[str, DataFrame] = {}
    for position, pair in enumerate(_pairs()):
        trend = np.linspace(100.0 + position, 112.0 + position, periods)
        wave = np.sin(np.arange(periods) / (3.0 + position / 10.0))
        output[pair] = DataFrame(
            {
                "date": dates,
                "close": trend + wave,
            }
        )
    return output


class _RawDataProvider:
    def __init__(self, snapshots: dict[str, DataFrame]) -> None:
        self.snapshots = snapshots

    def current_whitelist(self) -> list[str]:
        return list(_pairs())

    def get_pair_dataframe(
        self,
        pair: str,
        timeframe: str,
    ) -> DataFrame:
        del timeframe
        return self.snapshots[pair].copy()


class _CandidateDataProvider:
    @staticmethod
    def current_whitelist() -> list[str]:
        return ["BTC/USDT:USDT"]


class LiveContextBuilderTests(unittest.TestCase):
    def test_latest_closed_row_is_complete_for_every_pair(self) -> None:
        context, target, unavailable = build_synchronized_market_context(
            _snapshots(),
            _pairs(),
        )
        self.assertEqual(unavailable, ())
        self.assertEqual(
            target,
            pd.Timestamp("2026-01-02T15:00:00Z"),
        )
        latest = context.loc[context["date"] == target]
        self.assertEqual(len(latest), 1)
        self.assertFalse(
            latest[list(LIVE_CONTEXT_COLUMNS)].isna().any(axis=None)
        )

    def test_stale_pair_fails_closed_instead_of_using_old_context(self) -> None:
        snapshots = _snapshots()
        stale_pair = _pairs()[-1]
        snapshots[stale_pair] = snapshots[stale_pair].iloc[:-1].copy()
        context, target, unavailable = build_synchronized_market_context(
            snapshots,
            _pairs(),
        )
        self.assertTrue(context.empty)
        self.assertEqual(
            target,
            pd.Timestamp("2026-01-02T15:00:00Z"),
        )
        self.assertEqual(unavailable, (stale_pair,))

    def test_future_candle_cannot_change_prior_context(self) -> None:
        snapshots = _snapshots()
        original, original_target, unavailable = (
            build_synchronized_market_context(snapshots, _pairs())
        )
        self.assertEqual(unavailable, ())
        extended: dict[str, DataFrame] = {}
        future_date = original_target + pd.Timedelta(hours=1)
        for position, pair in enumerate(_pairs()):
            extended[pair] = pd.concat(
                (
                    snapshots[pair],
                    DataFrame(
                        {
                            "date": [future_date],
                            "close": [1_000_000.0 + position],
                        }
                    ),
                ),
                ignore_index=True,
            )
        updated, _updated_target, unavailable = (
            build_synchronized_market_context(extended, _pairs())
        )
        self.assertEqual(unavailable, ())
        columns = ["date", *LIVE_CONTEXT_COLUMNS]
        assert_frame_equal(
            original.loc[
                original["date"] <= original_target, columns
            ].reset_index(drop=True),
            updated.loc[
                updated["date"] <= original_target, columns
            ].reset_index(drop=True),
        )

    def test_live_formula_matches_frozen_backtest_formula(self) -> None:
        snapshots = _snapshots()
        expected_strategy = object.__new__(
            breakout_v9.BreakoutV9GridV7Freqtrade
        )
        expected_strategy.dp = _RawDataProvider(snapshots)
        expected = expected_strategy._build_market_context()
        actual, _target, unavailable = build_synchronized_market_context(
            snapshots,
            _pairs(),
        )
        self.assertEqual(unavailable, ())
        assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
        )


class CandidateOrderFlowTests(unittest.TestCase):
    def test_synchronized_grid_candidate_reaches_entry_confirmation(self) -> None:
        strategy = object.__new__(
            BreakoutV10FGridV8DualSideFreqtrade
        )
        strategy._candidate_ranks = {}
        strategy.dp = _CandidateDataProvider()
        strategy._component_limits_allow = (
            lambda pair, component, current_time: True
        )
        candle = DataFrame(
            {
                "date": [pd.Timestamp("2026-07-30T00:00:00Z")],
                "bo_entry_long": [0],
                "bo_entry_short": [0],
                "grid_entry_long": [0],
                "grid_entry_short": [1],
                "grid_long_score": [0],
                "grid_score": [6],
                "grid_rank": [0.12],
            }
        )
        analyzed = strategy.populate_entry_trend(
            candle,
            {"pair": "BTC/USDT:USDT"},
        )
        self.assertEqual(int(analyzed.iloc[-1]["enter_short"]), 1)
        self.assertEqual(
            analyzed.iloc[-1]["enter_tag"],
            "grid_v8_short_s6",
        )
        with patch.object(
            breakout_v9.Trade,
            "get_trades_proxy",
            return_value=[],
        ):
            accepted = strategy.confirm_trade_entry(
                pair="BTC/USDT:USDT",
                order_type="market",
                amount=1.0,
                rate=100.0,
                time_in_force="GTC",
                current_time=datetime(
                    2026,
                    7,
                    30,
                    1,
                    0,
                    tzinfo=timezone.utc,
                ),
                entry_tag="grid_v8_short_s6",
                side="short",
            )
        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
