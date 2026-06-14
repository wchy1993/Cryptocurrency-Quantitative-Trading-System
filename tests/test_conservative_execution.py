from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import unittest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_execution_backtest import (
    PendingEntry,
    _closed_signal_index,
    _fill_pending_entries_1m,
    _manage_positions_1m,
)
from crypto_scalper.live_portfolio_backtest import (
    HistoricalClient,
    PortfolioPosition,
    _close_position,
    _summary,
)
from crypto_scalper.live_trader import BinanceAutoTrader, EntryCandidate
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.risk import (
    BacktestExecutionConfig,
    BacktestExecutionStats,
    conservative_price,
    conservative_quantity,
    funding_cashflow,
    limit_fill,
    limit_order_filled,
    market_entry_fill,
    market_exit_fill,
    validate_order_size,
)


class ConservativeExecutionTests(unittest.TestCase):
    def test_30m_signal_not_available_before_close(self) -> None:
        timestamps = [datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 30)]
        signal_ms = 30 * 60_000

        self.assertEqual(_closed_signal_index(timestamps, datetime(2026, 1, 1, 10, 29), signal_ms), -1)
        self.assertEqual(_closed_signal_index(timestamps, datetime(2026, 1, 1, 10, 30), signal_ms), 0)
        self.assertEqual(_closed_signal_index(timestamps, datetime(2026, 1, 1, 10, 59), signal_ms), 0)
        self.assertEqual(_closed_signal_index(timestamps, datetime(2026, 1, 1, 11, 0), signal_ms), 1)

    def test_signal_available_time_is_next_30m_open(self) -> None:
        signal_time = datetime(2026, 1, 1, 10, 0)
        pending = PendingEntry(
            candidate=object(),
            signal_index=0,
            signal_time=signal_time,
            signal_available_time=signal_time + timedelta(minutes=30),
            decision_time=signal_time + timedelta(minutes=30),
            earliest_execution_index=1,
        )

        self.assertEqual(pending.signal_available_time, datetime(2026, 1, 1, 10, 30))

    def test_current_1m_bar_signal_cannot_fill_current_bar(self) -> None:
        config = replace(default_live_config(), trading=replace(default_live_config().trading, symbols=("BTCUSDT",), entry_symbols=("BTCUSDT",)))
        candles = {"BTCUSDT": [_candle(datetime(2026, 1, 1, 10, 30), 100.0)]}
        signal_candles = {"BTCUSDT": [_candle(datetime(2026, 1, 1, 10, 0), 100.0, minutes=30)]}
        client = HistoricalClient(signal_candles, config.trading.timeframe, config.filters.timeframes)
        trader = BinanceAutoTrader(config, client)
        candidate = EntryCandidate(
            "BTCUSDT",
            Signal(Direction.LONG, 1.0, "test_breakout", 0.01, 0.02),
            signal_candles["BTCUSDT"][0],
            1.0,
            0.01,
            2.0,
            "ok",
        )
        pending = [PendingEntry(candidate, 0, datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 30), datetime(2026, 1, 1, 10, 30), 1)]
        positions = {}

        _fill_pending_entries_1m(
            trader,
            config,
            160.0,
            positions,
            pending,
            {},
            candles,
            execution_index=0,
            signal_index=0,
            timestamp=datetime(2026, 1, 1, 10, 30),
            client=client,
            execution_config=BacktestExecutionConfig(),
        )

        self.assertFalse(positions)
        self.assertEqual(len(pending), 1)

    def test_same_bar_tp_sl_conflict_conservative_exits_stop_first(self) -> None:
        config = replace(default_live_config(), trading=replace(default_live_config().trading, symbols=("BTCUSDT",), entry_symbols=("BTCUSDT",)))
        timestamp = datetime(2026, 1, 1, 10, 31)
        execution_candles = {"BTCUSDT": [_candle(timestamp, 100.0, high=103.0, low=97.0)]}
        signal_candles = {"BTCUSDT": [_candle(datetime(2026, 1, 1, 10, 0), 100.0, minutes=30)]}
        client = HistoricalClient(signal_candles, config.trading.timeframe, config.filters.timeframes)
        trader = BinanceAutoTrader(config, client)
        positions = {
            "BTCUSDT": PortfolioPosition(
                "BTCUSDT",
                Direction.LONG,
                1.0,
                100.0,
                99.0,
                102.0,
                0.0,
                0,
                10,
                100.0,
                entry_time=datetime(2026, 1, 1, 10, 30),
                raw_entry_price=100.0,
            )
        }
        trades = []
        stats = BacktestExecutionStats()

        _manage_positions_1m(
            trader,
            config,
            160.0,
            positions,
            {},
            trades,
            execution_candles,
            signal_candles,
            0,
            0,
            BacktestExecutionConfig(mode="conservative", market_slippage_bps=0, stop_slippage_bps=0, take_profit_slippage_bps=0, taker_fee_rate=0, maker_fee_rate=0),
            stats,
        )

        self.assertEqual(stats.same_bar_tp_sl_conflict_count, 1)
        self.assertEqual(trades[0]["exit_reason"], "stop_loss_1m")

    def test_limit_touch_does_not_fill_without_penetration(self) -> None:
        rules = _rules()
        touched, filled = limit_order_filled(rules, "buy", _candle(datetime(2026, 1, 1), 100.0, low=99.0), 99.0)

        self.assertTrue(touched)
        self.assertFalse(filled)

    def test_market_order_adds_adverse_slippage(self) -> None:
        fill = market_entry_fill(BacktestExecutionConfig(market_slippage_bps=10, taker_fee_rate=0), _rules(), Direction.LONG, 1.0, 100.0)

        self.assertGreater(fill.price, 100.0)
        self.assertGreater(fill.slippage_cost, 0.0)
        self.assertEqual(fill.liquidity, "taker")

    def test_stop_order_uses_extra_adverse_slippage(self) -> None:
        fill = market_exit_fill(BacktestExecutionConfig(stop_slippage_bps=20, taker_fee_rate=0), _rules(), Direction.LONG, 1.0, 100.0, "stop_market")

        self.assertLess(fill.price, 100.0)
        self.assertAlmostEqual(fill.slippage_cost, 0.2, places=6)

    def test_maker_and_taker_fee_are_separate(self) -> None:
        config = BacktestExecutionConfig(maker_fee_rate=0.0002, taker_fee_rate=0.0005, market_slippage_bps=0)
        taker = market_entry_fill(config, _rules(), Direction.LONG, 1.0, 100.0)
        maker = limit_fill(config, _rules(), Direction.LONG, "buy", 1.0, 100.0)

        self.assertAlmostEqual(taker.fee, 0.05)
        self.assertAlmostEqual(maker.fee, 0.02)

    def test_funding_direction_for_positive_and_negative_rates(self) -> None:
        self.assertAlmostEqual(funding_cashflow(Direction.LONG, 1000.0, [0.001]), -1.0)
        self.assertAlmostEqual(funding_cashflow(Direction.SHORT, 1000.0, [0.001]), 1.0)
        self.assertAlmostEqual(funding_cashflow(Direction.LONG, 1000.0, [-0.001]), 1.0)
        self.assertAlmostEqual(funding_cashflow(Direction.SHORT, 1000.0, [-0.001]), -1.0)

    def test_tick_and_step_rounding_are_conservative(self) -> None:
        rules = _rules()

        self.assertEqual(conservative_price(rules, 100.004, "buy"), 100.01)
        self.assertEqual(conservative_price(rules, 100.009, "sell"), 100.0)
        self.assertEqual(conservative_quantity(rules, 1.23456), 1.234)

    def test_min_qty_and_min_notional_skip_reason(self) -> None:
        rules = _rules()

        self.assertEqual(validate_order_size(rules, 0.0001, 100.0), "below_min_quantity")
        self.assertEqual(validate_order_size(rules, 0.01, 100.0), "below_exchange_min_notional")

    def test_summary_contains_enhanced_fields(self) -> None:
        first = _candle(datetime(2026, 1, 1), 100.0)
        last = _candle(datetime(2026, 1, 1, 1), 101.0)
        trade = _trade()
        summary = _summary(160.0, 161.0, [160.0, 161.0], [trade], first, last, execution_stats=BacktestExecutionStats(same_bar_tp_sl_conflict_count=1))

        self.assertIn("enhanced_summary", summary)
        self.assertIn("drawdown_analysis", summary)
        overall = summary["enhanced_summary"]["overall"]
        for field in ("gross_pnl", "fee", "slippage_cost", "funding", "net_pnl", "avg_mfe", "avg_mae", "maker_fill_rate", "taker_ratio", "expectancy_per_trade"):
            self.assertIn(field, overall)
        self.assertEqual(summary["same_bar_tp_sl_conflict_count"], 1)

    def test_summary_records_drawdown_peak_and_trough(self) -> None:
        first = _candle(datetime(2026, 1, 1), 100.0)
        last = _candle(datetime(2026, 1, 1, 3), 101.0)
        trade = _trade()
        timeline = [
            (datetime(2026, 1, 1), 100.0),
            (datetime(2026, 1, 1, 1), 150.0),
            (datetime(2026, 1, 1, 2), 90.0),
            (datetime(2026, 1, 1, 3), 120.0),
        ]

        summary = _summary(100.0, 120.0, [point[1] for point in timeline], [trade], first, last, equity_timeline=timeline)

        drawdown = summary["drawdown_analysis"]
        self.assertAlmostEqual(drawdown["max_drawdown_pct"], 40.0)
        self.assertEqual(drawdown["peak_time"], "2026-01-01T01:00:00")
        self.assertEqual(drawdown["trough_time"], "2026-01-01T02:00:00")
        self.assertAlmostEqual(drawdown["drawdown_usdt"], 60.0)

    def test_trade_log_contains_enhanced_fields(self) -> None:
        config = default_live_config()
        positions = {
            "BTCUSDT": PortfolioPosition(
                "BTCUSDT",
                Direction.LONG,
                1.0,
                100.0,
                99.0,
                102.0,
                0.0,
                0,
                10,
                100.0,
                entry_reason="long_breakout",
                entry_time=datetime(2026, 1, 1, 10, 30),
                raw_entry_price=100.0,
                signal_time=datetime(2026, 1, 1, 10, 0),
                signal_available_time=datetime(2026, 1, 1, 10, 30),
            )
        }
        trades = []

        _close_position(
            config,
            160.0,
            positions,
            trades,
            "BTCUSDT",
            101.0,
            "take_profit_1m",
            1,
            datetime(2026, 1, 1, 10, 35),
            execution_config=BacktestExecutionConfig(market_slippage_bps=0, take_profit_slippage_bps=0, taker_fee_rate=0, maker_fee_rate=0),
            rules=_rules(),
        )

        trade = trades[0]
        for field in ("symbol", "strategy", "side", "entry_time", "entry_price", "exit_time", "exit_price", "qty", "notional", "gross_pnl", "fee", "slippage_cost", "funding", "net_pnl", "mfe", "mae", "hold_minutes", "exit_reason", "entry_order_type", "exit_order_type", "signal_time", "signal_available_time", "skip_reason"):
            self.assertIn(field, trade)


def _rules() -> SymbolRules:
    return SymbolRules("BTCUSDT", "0.001", "0.001", "0.01", "5")


def _candle(timestamp: datetime, price: float, high: float | None = None, low: float | None = None, minutes: int = 1) -> Candle:
    high = price + 1.0 if high is None else high
    low = price - 1.0 if low is None else low
    return Candle(timestamp, price, high, low, price, 1000.0)


def _trade() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "strategy": "breakout",
        "side": "LONG",
        "direction": "LONG",
        "entry_time": "2026-01-01T10:30:00",
        "exit_time": "2026-01-01T10:35:00",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "qty": 1.0,
        "quantity": 1.0,
        "notional": 100.0,
        "gross_pnl": 1.0,
        "fee": 0.1,
        "fees": 0.1,
        "slippage_cost": 0.05,
        "funding": 0.0,
        "net_pnl": 0.85,
        "return_pct": 0.0085,
        "mfe": 1.2,
        "mae": -0.3,
        "hold_minutes": 5.0,
        "entry_liquidity": "taker",
        "exit_liquidity": "taker",
        "entry_reason": "long_breakout",
        "strategy_bucket": "breakout",
        "reason": "take_profit_1m",
        "exit_reason": "take_profit_1m",
        "scale_ins": 0,
    }


if __name__ == "__main__":
    unittest.main()
