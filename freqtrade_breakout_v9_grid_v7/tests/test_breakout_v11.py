from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pandas import DataFrame
from freqtrade.enums import RunMode


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

import BreakoutV11GridV8DualSideFreqtrade as v11  # noqa: E402
import BreakoutV11BreakoutGovernorGridV8DualSideFreqtrade as v11_bo_only  # noqa: E402
import BreakoutV11LossStreakGridV8DualSideFreqtrade as v11_loss_streak  # noqa: E402
import BreakoutV11Trigger15Recovery100ExhaustionGridV8DualSideFreqtrade as v11_exhaustion  # noqa: E402
import BreakoutV11PortfolioSleeveRecoveryExhaustionQ112GridV8DualSideFreqtrade as v11_sleeve  # noqa: E402
import BreakoutV11ConvictionFloor35Q112GridV8DualSideFreqtrade as v11_conviction  # noqa: E402
import BreakoutV11Recovery60FullRisk102Q112GridV8DualSideFreqtrade as v11_full_risk  # noqa: E402
import BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade as v11_recovery60  # noqa: E402
import BreakoutV11NoGridShortS3Recovery60Q112GridV8DualSideFreqtrade as v11_no_grid_s3  # noqa: E402
import BreakoutV11GridS3Risk50Recovery60Q112GridV8DualSideFreqtrade as v11_grid_s3_risk  # noqa: E402
import BreakoutV11DefensiveGridS3Risk50Recovery60Q112GridV8DualSideFreqtrade as v11_defensive_grid_s3  # noqa: E402
import BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade as v11_defensive_bo_s3  # noqa: E402
import BreakoutV11AdaptiveGridV8DualSideFreqtrade as v11_selected  # noqa: E402


class _ClosedTrade:
    def __init__(self, close_profit: float, hour: int) -> None:
        self.close_profit = close_profit
        self.close_date = datetime(2026, 1, 1, hour, tzinfo=timezone.utc)


class _Wallets:
    @staticmethod
    def get_total_stake_amount() -> float:
        return 200.0


class BreakoutV11EntryGuardTests(unittest.TestCase):
    @staticmethod
    def _frame() -> DataFrame:
        dates = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
        close = pd.Series([100.0 - position for position in range(24)])
        return DataFrame(
            {
                "date": dates,
                "close": close,
                "fast_ema": close + 1.0,
                "session_open": 100.0,
                "bo_entry_short": 0,
                "bo_entry": 0,
                "bo_score": 2,
                "bo_volume_ratio": 0.96,
                "bo_short_quality": 1.025,
            }
        )

    def test_ondo_like_late_thin_score_two_short_is_rejected(self) -> None:
        strategy = object.__new__(v11.BreakoutV11GridV8DualSideFreqtrade)
        frame = self._frame()
        frame.loc[frame.index[-1], ["bo_entry_short", "bo_entry"]] = 1
        guarded = strategy._apply_breakout_quality_guard(frame)
        latest = guarded.iloc[-1]
        self.assertEqual(int(latest["bo_v11_late_short_rejected"]), 1)
        self.assertEqual(int(latest["bo_entry_short"]), 0)
        self.assertEqual(int(latest["bo_entry"]), 0)

    def test_volume_confirmation_preserves_mature_score_two_short(self) -> None:
        strategy = object.__new__(v11.BreakoutV11GridV8DualSideFreqtrade)
        frame = self._frame()
        frame.loc[frame.index[-1], ["bo_entry_short", "bo_entry"]] = 1
        frame.loc[frame.index[-1], "bo_volume_ratio"] = 1.20
        guarded = strategy._apply_breakout_quality_guard(frame)
        latest = guarded.iloc[-1]
        self.assertEqual(int(latest["bo_v11_late_short_rejected"]), 0)
        self.assertEqual(int(latest["bo_entry_short"]), 1)

    def test_extreme_session_move_rejects_even_high_volume_score_two(self) -> None:
        strategy = object.__new__(
            v11_exhaustion.BreakoutV11Trigger15Recovery100ExhaustionGridV8DualSideFreqtrade
        )
        frame = self._frame()
        frame.loc[frame.index[-1], ["bo_entry_short", "bo_entry"]] = 1
        frame.loc[frame.index[-1], "bo_volume_ratio"] = 1.30
        frame.loc[frame.index[-1], "bo_short_quality"] = 1.40
        guarded = strategy._apply_breakout_quality_guard(frame)
        latest = guarded.iloc[-1]
        self.assertEqual(int(latest["bo_v11_extreme_short_rejected"]), 1)
        self.assertEqual(int(latest["bo_entry_short"]), 0)

    def test_extreme_session_move_preserves_higher_score_short(self) -> None:
        strategy = object.__new__(
            v11_exhaustion.BreakoutV11Trigger15Recovery100ExhaustionGridV8DualSideFreqtrade
        )
        frame = self._frame()
        frame.loc[frame.index[-1], ["bo_entry_short", "bo_entry"]] = 1
        frame.loc[frame.index[-1], "bo_score"] = 3
        guarded = strategy._apply_breakout_quality_guard(frame)
        latest = guarded.iloc[-1]
        self.assertEqual(int(latest["bo_v11_extreme_short_rejected"]), 0)
        self.assertEqual(int(latest["bo_entry_short"]), 1)


class BreakoutV11GridScoreGuardTests(unittest.TestCase):
    def test_only_score_three_short_is_removed(self) -> None:
        strategy = object.__new__(
            v11_no_grid_s3.BreakoutV11NoGridShortS3Recovery60Q112GridV8DualSideFreqtrade
        )
        frame = DataFrame(
            {
                "grid_entry_short": [1, 1, 0],
                "grid_entry_long": [0, 0, 1],
                "grid_score": [3, 4, 3],
            }
        )
        with patch.object(
            v11_no_grid_s3.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
            "_populate_grid",
            return_value=frame,
        ):
            guarded = strategy._populate_grid(frame)
        self.assertEqual(guarded["grid_entry_short"].tolist(), [0, 1, 0])
        self.assertEqual(guarded["grid_entry_long"].tolist(), [0, 0, 1])
        self.assertEqual(guarded["grid_entry"].tolist(), [0, 1, 1])
        self.assertEqual(
            guarded["grid_v11_short_score3_rejected"].tolist(),
            [1, 0, 0],
        )


class BreakoutV11GridScoreRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_grid_s3_risk.BreakoutV11GridS3Risk50Recovery60Q112GridV8DualSideFreqtrade
        )
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def _stake(self, tag: str, min_stake: float | None = 1.0) -> float:
        with patch.object(
            v11_grid_s3_risk.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=min_stake,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_score_three_short_keeps_slot_at_half_risk(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s3"), 10.0)

    def test_other_grid_scores_remain_frozen(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s4"), 20.0)

    def test_exchange_minimum_is_respected(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s3", min_stake=12.0), 12.0)


class BreakoutV11DefensiveGridScoreRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_defensive_grid_s3.BreakoutV11DefensiveGridS3Risk50Recovery60Q112GridV8DualSideFreqtrade
        )
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def _stake(self, tag: str, portfolio_scale: float) -> float:
        with patch.object(
            v11_defensive_grid_s3.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(
            self.strategy,
            "_portfolio_risk_scale",
            return_value=portfolio_scale,
        ):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_score_three_is_reduced_only_during_defense(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s3", 0.15), 10.0)
        self.assertEqual(self._stake("grid_v8_short_s3", 1.0), 20.0)

    def test_other_scores_remain_frozen_during_defense(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s4", 0.15), 20.0)


class BreakoutV11DefensiveBreakoutScoreRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_defensive_bo_s3.BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade
        )
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def _stake(self, tag: str, portfolio_scale: float) -> float:
        with patch.object(
            v11_defensive_bo_s3.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(
            self.strategy,
            "_portfolio_risk_scale",
            return_value=portfolio_scale,
        ):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_score_three_is_reduced_only_during_defense(self) -> None:
        tag = "bo_v9_s3_r31809_c0_l0"
        self.assertEqual(self._stake(tag, 0.15), 10.0)
        self.assertEqual(self._stake(tag, 1.0), 20.0)

    def test_score_four_and_five_remain_frozen(self) -> None:
        self.assertEqual(
            self._stake("bo_v9_s4_r41351_c0_l1", 0.15),
            20.0,
        )
        self.assertEqual(
            self._stake("bo_v9_s5_r41351_c0_l0", 0.15),
            20.0,
        )


class BreakoutV11SelectedEntryPointTests(unittest.TestCase):
    def test_selected_entry_point_freezes_winning_candidate(self) -> None:
        strategy = v11_selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
        self.assertTrue(
            issubclass(
                strategy,
                v11_defensive_bo_s3.BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade,
            )
        )
        self.assertEqual(strategy.PORTFOLIO_GOVERNOR_TRIGGER, 0.18)
        self.assertEqual(strategy.PORTFOLIO_DEFENSIVE_SCALE, 0.15)
        self.assertEqual(strategy.PORTFOLIO_RECOVERY_SCALE, 0.60)
        self.assertEqual(strategy.DEFENSIVE_BO_SCORE3_SCALE, 0.50)

    def test_backtesting_never_uses_runtime_state(self) -> None:
        strategy = object.__new__(
            v11_selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
        )
        strategy.config = {"runmode": RunMode.BACKTEST}
        self.assertIsNone(strategy._adaptive_state_mode())

    def test_live_high_water_round_trip_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            strategy = object.__new__(
                v11_selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
            )
            strategy.config = {"runmode": RunMode.LIVE}
            strategy.ADAPTIVE_STATE_DIR = Path(directory)
            strategy._peak_equity = 312.5
            strategy._adaptive_last_persisted_peak = 0.0
            strategy._persist_adaptive_high_water(
                "live",
                datetime(2026, 8, 1, 10, 30, tzinfo=timezone.utc),
            )
            path = strategy._adaptive_state_path("live")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["peak_equity"], 312.5)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            restored = object.__new__(
                v11_selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
            )
            restored.config = {"runmode": RunMode.LIVE}
            restored.ADAPTIVE_STATE_DIR = Path(directory)
            restored._peak_equity = 0.0
            restored._load_adaptive_high_water("live")
            self.assertEqual(restored._peak_equity, 312.5)
            self.assertEqual(restored._adaptive_last_persisted_peak, 312.5)

    def test_corrupt_live_high_water_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            strategy = object.__new__(
                v11_selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
            )
            strategy.config = {"runmode": RunMode.LIVE}
            strategy.ADAPTIVE_STATE_DIR = Path(directory)
            path = strategy._adaptive_state_path("live")
            path.write_text('{"version": 1, "mode": "live", "peak_equity": -1}')
            with self.assertRaisesRegex(RuntimeError, "权益高水位状态无效"):
                strategy._load_adaptive_high_water("live")

class BreakoutV11PortfolioGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(v11.BreakoutV11GridV8DualSideFreqtrade)
        self.strategy._peak_equity = 200.0
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def test_full_risk_above_drawdown_boundary(self) -> None:
        self.assertEqual(self.strategy._portfolio_risk_scale(161.0, self.now), 1.0)

    def test_weak_recent_campaigns_activate_defensive_scale(self) -> None:
        trades = [_ClosedTrade(-0.10, hour) for hour in range(8)]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(159.0, self.now),
                self.strategy.PORTFOLIO_DEFENSIVE_SCALE,
            )

    def test_confirmed_recent_recovery_rearms_full_risk(self) -> None:
        returns = (0.30, 0.20, 0.15, 0.10, 0.05, -0.05, -0.05, -0.05)
        trades = [
            _ClosedTrade(value, hour) for hour, value in enumerate(returns)
        ]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(159.0, self.now),
                1.0,
            )

    def test_first_grid_stake_initializes_shared_high_water_mark(self) -> None:
        self.strategy._peak_equity = 0.0
        self.strategy.wallets = _Wallets()
        with patch.object(
            v11.BreakoutV10FGridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ):
            stake = self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag="grid_v8_short_s5",
                side="short",
            )
        self.assertEqual(stake, 20.0)
        self.assertEqual(self.strategy._peak_equity, 200.0)


class BreakoutV11SelectedGovernorBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_recovery60.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 200.0
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def test_trigger_is_above_recent_baseline_drawdown(self) -> None:
        self.assertEqual(self.strategy._portfolio_risk_scale(165.0, self.now), 1.0)

    def test_eighteen_percent_drawdown_enters_defensive_state(self) -> None:
        trades = [_ClosedTrade(-0.10, hour) for hour in range(8)]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(163.0, self.now),
                self.strategy.PORTFOLIO_DEFENSIVE_SCALE,
            )

    def test_four_wins_recover_only_to_intermediate_allocation(self) -> None:
        returns = (0.10, 0.10, 0.10, 0.10, -0.05, -0.05, -0.05, -0.05)
        trades = [
            _ClosedTrade(value, hour) for hour, value in enumerate(returns)
        ]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(150.0, self.now),
                0.60,
            )


class BreakoutV11LossStreakGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_loss_streak.BreakoutV11LossStreakGridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 200.0
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def test_four_closed_losses_activate_early_warning(self) -> None:
        trades = [_ClosedTrade(-0.05, hour) for hour in range(4)]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(190.0, self.now),
                self.strategy.PORTFOLIO_DEFENSIVE_SCALE,
            )

    def test_three_closed_losses_leave_full_risk_unchanged(self) -> None:
        trades = [_ClosedTrade(-0.05, hour) for hour in range(3)]
        with patch.object(v11.Trade, "get_trades_proxy", return_value=trades):
            self.assertEqual(
                self.strategy._portfolio_risk_scale(190.0, self.now),
                1.0,
            )


class BreakoutV11BreakoutOnlyGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_bo_only.BreakoutV11BreakoutGovernorGridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 300.0
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def _stake(self, tag: str) -> float:
        with patch.object(
            v11_bo_only.BreakoutV10FGridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(v11.Trade, "get_trades_proxy", return_value=[]):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_underwater_grid_stake_remains_frozen(self) -> None:
        self.assertEqual(self._stake("grid_v8_short_s5"), 20.0)

    def test_underwater_breakout_stake_is_defensive(self) -> None:
        self.assertEqual(
            self._stake("bo_v9_s2_r30294_c0_l0"),
            20.0 * self.strategy.PORTFOLIO_DEFENSIVE_SCALE,
        )


class BreakoutV11SleeveRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_sleeve.BreakoutV11PortfolioSleeveRecoveryExhaustionQ112GridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 300.0
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        returns = (0.10, 0.10, 0.10, 0.10, -0.10, -0.10, -0.10, -0.10)
        self.trades = [
            _ClosedTrade(value, hour)
            for hour, value in enumerate(returns)
        ]

    def _stake(self, tag: str) -> float:
        with patch.object(
            v11_sleeve.BreakoutV10FGridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(v11.Trade, "get_trades_proxy", return_value=self.trades):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_recovery_allocates_more_risk_to_breakout(self) -> None:
        self.assertEqual(self._stake("bo_v9_s2_r30294_c0_l0"), 16.0)
        self.assertEqual(self._stake("grid_v8_short_s5"), 10.0)


class BreakoutV11ConvictionFloorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_conviction.BreakoutV11ConvictionFloor35Q112GridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 300.0
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self.trades = [_ClosedTrade(-0.10, hour) for hour in range(8)]

    def _stake(self, tag: str) -> float:
        with patch.object(
            v11_conviction.BreakoutV10FGridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(v11.Trade, "get_trades_proxy", return_value=self.trades):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag=tag,
                side="short",
            )

    def test_only_score_five_breakout_gets_defensive_floor(self) -> None:
        self.assertEqual(self._stake("bo_v9_s5_r100000_c0_l0"), 7.0)
        self.assertEqual(self._stake("bo_v9_s4_r43320_c1_l0"), 3.0)
        self.assertEqual(self._stake("grid_v8_short_s5"), 3.0)


class BreakoutV11FullRiskReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            v11_full_risk.BreakoutV11Recovery60FullRisk102Q112GridV8DualSideFreqtrade
        )
        self.strategy._peak_equity = 200.0
        self.strategy.wallets = _Wallets()
        self.now = datetime(2026, 1, 2, tzinfo=timezone.utc)

    def _stake(self, portfolio_scale: float) -> float:
        with patch.object(
            v11_full_risk.BreakoutV11PortfolioRecovery60ExhaustionQ112GridV8DualSideFreqtrade,
            "custom_stake_amount",
            return_value=20.0,
        ), patch.object(
            self.strategy,
            "_portfolio_risk_scale",
            return_value=portfolio_scale,
        ):
            return self.strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=self.now,
                current_rate=100.0,
                proposed_stake=20.0,
                min_stake=1.0,
                max_stake=200.0,
                leverage=10.0,
                entry_tag="bo_v9_s5_r100000_c0_l0",
                side="long",
            )

    def test_release_applies_only_at_full_portfolio_risk(self) -> None:
        self.assertAlmostEqual(self._stake(1.0), 20.4)

    def test_release_does_not_increase_defensive_or_recovery_stake(self) -> None:
        self.assertEqual(self._stake(0.60), 20.0)
        self.assertEqual(self._stake(0.15), 20.0)


if __name__ == "__main__":
    unittest.main()
