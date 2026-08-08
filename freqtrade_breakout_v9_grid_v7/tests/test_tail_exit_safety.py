from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

import BreakoutV11AdaptiveGridV8DualSideFreqtrade as selected  # noqa: E402


class _Exchange:
    def __init__(self, minimum_exit: float = 0.75) -> None:
        self.minimum_exit = minimum_exit

    @staticmethod
    def amount_to_contract_precision(_pair: str, amount: float) -> float:
        # XLMUSDT uses whole-contract precision in the incident timeline.
        return float(math.floor(amount))

    def get_min_pair_stake_amount(
        self,
        _pair: str,
        _rate: float,
        _stoploss: float,
        _leverage: float,
    ) -> float:
        return self.minimum_exit


def _xlm_trade() -> SimpleNamespace:
    return SimpleNamespace(
        id=2,
        pair="XLM/USDT:USDT",
        amount=2678.0,
        stake_amount=45.588916278098786,
        leverage=10.0,
        enter_tag="grid_v8_short_s6",
    )


class TailExitSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = object.__new__(
            selected.BreakoutV11AdaptiveGridV8DualSideFreqtrade
        )
        self.strategy.dp = SimpleNamespace(_exchange=_Exchange())
        self.now = datetime(2026, 8, 5, tzinfo=timezone.utc)

    def _adjust(self, returned: object, trade: SimpleNamespace | None = None):
        parent = (
            selected
            .BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade
        )
        active_trade = trade or _xlm_trade()
        with patch.object(parent, "adjust_trade_position", return_value=returned):
            return self.strategy.adjust_trade_position(
                trade=active_trade,
                current_time=self.now,
                current_rate=0.16588,
                current_profit=0.246,
                min_stake=0.50,
                max_stake=200.0,
                current_entry_rate=0.16589,
                current_exit_rate=0.16588,
                current_entry_profit=0.246,
                current_exit_profit=0.246,
            )

    def test_xlm_incident_tail_is_promoted_to_full_exit(self) -> None:
        trade = _xlm_trade()
        original_requested_stake = 45.58417
        original_exit_amount = math.floor(
            original_requested_stake * trade.amount / trade.stake_amount
        )
        self.assertEqual(trade.amount - original_exit_amount, 1.0)
        self.assertLess(
            (trade.amount - original_exit_amount) * 0.16588,
            0.75,
        )

        result = self._adjust(
            (-original_requested_stake, "grid_tp_0"),
            trade,
        )
        self.assertEqual(result[1], "grid_tp_0")
        self.assertAlmostEqual(result[0], -45.588916278098786, places=12)
        promoted_exit_amount = math.floor(
            abs(result[0]) * trade.amount / trade.stake_amount
        )
        self.assertEqual(trade.amount - promoted_exit_amount, 0.0)

    def test_valid_partial_exit_is_unchanged(self) -> None:
        result = self._adjust((-20.0, "grid_tp_0"))
        self.assertEqual(result, (-20.0, "grid_tp_0"))

    def test_entry_adjustment_is_unchanged(self) -> None:
        result = self._adjust((15.0, "grid_dca_1"))
        self.assertEqual(result, (15.0, "grid_dca_1"))

    def test_none_is_unchanged(self) -> None:
        self.assertIsNone(self._adjust(None))

    def test_plain_float_return_shape_is_preserved(self) -> None:
        result = self._adjust(-45.58417)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, -45.588916278098786, places=12)


if __name__ == "__main__":
    unittest.main()
