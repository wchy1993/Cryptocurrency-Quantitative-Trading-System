from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Direction
from crypto_scalper.order_flow_absorption_scalper import (
    AggTradeEvent,
    BookSnapshot,
    DepthUpdateEvent,
    LocalOrderBook,
    MarketContext,
    OfasConfig,
    OfasState,
    OrderFlowAbsorptionScalper,
    load_ofas_config,
)


SYMBOL = "TESTUSDT"
START = datetime(2026, 7, 1, 0, 0, 0)


def _rules() -> SymbolRules:
    return SymbolRules(SYMBOL, Decimal("0.001"), Decimal("0.001"), Decimal("0.01"), Decimal("5"))


def _snapshot(timestamp: datetime = START, update_id: int = 100) -> BookSnapshot:
    bids = tuple((99.99 - index * 0.01, 1_000.0) for index in range(10))
    asks = tuple((100.01 + index * 0.01, 1_000.0) for index in range(10))
    return BookSnapshot(SYMBOL, timestamp, update_id, bids, asks)


def _depth(timestamp: datetime, first: int, final: int, previous: int, bid_qty: float) -> DepthUpdateEvent:
    return DepthUpdateEvent(SYMBOL, timestamp, first, final, previous, ((99.99, bid_qty),), ())


def _ask_depth(timestamp: datetime, first: int, final: int, previous: int, ask_qty: float) -> DepthUpdateEvent:
    return DepthUpdateEvent(SYMBOL, timestamp, first, final, previous, (), ((100.01, ask_qty),))


def _context(timestamp: datetime, equity: float = 1_000.0) -> MarketContext:
    return MarketContext(timestamp, equity, equity, 0)


class OfasConfigTests(unittest.TestCase):
    def test_default_file_is_safe_and_has_sixty_symbols(self) -> None:
        config = load_ofas_config("config.ofas.json")
        self.assertTrue(config.enabled)
        self.assertEqual(len(config.symbols), 60)
        self.assertEqual(config.entry_mode, "maker-first")
        self.assertEqual(config.max_open_positions, 1)
        self.assertLessEqual(config.short_risk_multiplier, 1.0)
        self.assertLessEqual(config.base_risk_pct, config.max_trade_risk_pct)

    def test_rejects_risk_above_hard_cap(self) -> None:
        with self.assertRaises(ValueError):
            OfasConfig(symbols=(SYMBOL,), base_risk_pct=0.003, max_trade_risk_pct=0.002)


class LocalOrderBookTests(unittest.TestCase):
    def test_update_id_gap_invalidates_book(self) -> None:
        book = LocalOrderBook(SYMBOL)
        book.apply_snapshot(_snapshot())
        self.assertTrue(book.valid)
        applied = book.apply_update(_depth(START + timedelta(milliseconds=100), 102, 102, 100, 500.0))
        self.assertFalse(applied)
        self.assertFalse(book.valid)
        self.assertEqual(book.invalid_reason, "update_id_gap")

    def test_contiguous_update_changes_depth(self) -> None:
        book = LocalOrderBook(SYMBOL)
        book.apply_snapshot(_snapshot())
        before = book.quote_depth("bid", 1)
        self.assertTrue(book.apply_update(_depth(START + timedelta(milliseconds=100), 101, 101, 100, 500.0)))
        self.assertLess(book.quote_depth("bid", 1), before)

    def test_subsequent_futures_batch_uses_previous_final_id(self) -> None:
        book = LocalOrderBook(SYMBOL)
        book.apply_snapshot(_snapshot())
        self.assertTrue(book.apply_update(_depth(START + timedelta(milliseconds=100), 101, 105, 99, 500.0)))
        self.assertTrue(book.apply_update(_depth(START + timedelta(milliseconds=200), 110, 115, 105, 600.0)))
        self.assertFalse(book.apply_update(_depth(START + timedelta(milliseconds=300), 120, 125, 104, 700.0)))
        self.assertEqual(book.invalid_reason, "update_id_gap")


class OfasStateMachineTests(unittest.TestCase):
    def _engine(self) -> OrderFlowAbsorptionScalper:
        config = OfasConfig(
            enabled=True,
            symbols=(SYMBOL,),
            evaluation_interval_ms=0,
            baseline_min_samples=1,
            sell_shock_volume_zscore=-1.0,
            sell_shock_price_sigma=0.0,
            sell_shock_min_return_bps=0.0,
            sell_shock_imbalance_threshold=0.7,
            buy_shock_volume_zscore=-1.0,
            buy_shock_price_sigma=0.0,
            buy_shock_min_return_bps=0.0,
            buy_shock_imbalance_threshold=0.7,
            no_new_low_seconds=0.0,
            no_new_high_seconds=0.0,
            bid_replenishment_min=0.5,
            ask_replenishment_min=0.5,
            normalized_ofi_long_min=0.01,
            normalized_ofi_short_max=-0.01,
            microprice_edge_long_bps=0.0,
            microprice_edge_short_bps=0.0,
            sell_exhaustion_threshold=1.0,
            buy_exhaustion_threshold=1.0,
            max_spread_bps=5.0,
            max_spread_percentile=1.0,
            min_l10_depth_usdt=1_000.0,
            min_trades_per_second=0.0,
            target_move_bps=100.0,
            max_spread_to_target_ratio=1.0,
            min_expected_move_to_cost_multiple=1.0,
            min_stop_bps=0.5,
            max_stop_bps=100.0,
            quality_a_min_score=0.0,
            quality_b_min_score=0.0,
            global_min_entry_interval_seconds=0,
            symbol_cooldown_minutes=1,
        )
        return OrderFlowAbsorptionScalper(config, {SYMBOL: _rules()})

    def test_long_shock_confirmation_fill_and_protection(self) -> None:
        engine = self._engine()
        engine.on_book_snapshot(_snapshot())
        engine.on_agg_trade(AggTradeEvent(SYMBOL, START + timedelta(milliseconds=50), 99.99, 10.0, True, 1))

        actions = engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=100)))
        self.assertEqual(actions[0].action, "PENDING_CREATED")
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.SELL_SHOCK_PENDING)

        engine.on_depth_update(_depth(START + timedelta(milliseconds=200), 101, 101, 100, 100.0))
        self.assertEqual(engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=250))), [])

        engine.on_depth_update(_depth(START + timedelta(milliseconds=500), 102, 102, 101, 1_200.0))
        actions = engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=550)))
        self.assertEqual(actions[0].action, "PLACE_ENTRY")
        self.assertEqual(actions[0].order_type, "LIMIT_MAKER")
        self.assertTrue(actions[0].post_only)
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.LONG_ORDER_WORKING)

        protection = engine.on_entry_fill(SYMBOL, START + timedelta(milliseconds=600), actions[0].price, actions[0].quantity)
        self.assertEqual([item.action for item in protection], ["CANCEL_ENTRY_REMAINDER", "PLACE_PROTECTIVE_STOP", "PLACE_TAKE_PROFIT"])
        self.assertTrue(all(item.reduce_only for item in protection[1:]))
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.LONG_POSITION_OPEN)

        emergency = engine.on_protection_rejected(SYMBOL)
        self.assertEqual(emergency[0].action, "EXIT_MARKET")
        self.assertTrue(emergency[0].reduce_only)

    def test_short_shock_confirmation_and_fill(self) -> None:
        engine = self._engine()
        engine.on_book_snapshot(_snapshot())
        engine.on_agg_trade(AggTradeEvent(SYMBOL, START + timedelta(milliseconds=50), 100.01, 10.0, False, 1))

        pending = engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=100)))
        self.assertEqual(pending[0].side, Direction.SHORT)
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.BUY_SHOCK_PENDING)

        engine.on_depth_update(_ask_depth(START + timedelta(milliseconds=200), 101, 101, 100, 100.0))
        engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=250)))
        engine.on_depth_update(_ask_depth(START + timedelta(milliseconds=500), 102, 102, 101, 1_200.0))
        entry = engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=550)))[0]
        self.assertEqual(entry.action, "PLACE_ENTRY")
        self.assertEqual(entry.side, Direction.SHORT)
        protection = engine.on_entry_fill(SYMBOL, START + timedelta(milliseconds=600), entry.price, entry.quantity)
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.SHORT_POSITION_OPEN)
        self.assertTrue(all(item.reduce_only for item in protection[1:]))

    def test_gap_cancels_pending_and_requests_resync(self) -> None:
        engine = self._engine()
        engine.on_book_snapshot(_snapshot())
        engine.on_agg_trade(AggTradeEvent(SYMBOL, START + timedelta(milliseconds=50), 99.99, 10.0, True, 1))
        engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=100)))

        actions = engine.on_depth_update(_depth(START + timedelta(milliseconds=200), 102, 102, 100, 100.0))
        self.assertEqual([item.action for item in actions], ["PENDING_CANCELLED", "RESYNC_BOOK"])
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.IDLE)

    def test_stale_data_never_opens_position(self) -> None:
        engine = self._engine()
        engine.on_book_snapshot(_snapshot())
        engine.on_agg_trade(AggTradeEvent(SYMBOL, START, 99.99, 10.0, True, 1))
        actions = engine.evaluate(SYMBOL, _context(START + timedelta(seconds=3)))
        self.assertEqual(actions, [])
        self.assertEqual(engine.runtime[SYMBOL].state, OfasState.IDLE)
        self.assertGreater(engine.reject_reasons["stale_market_data"], 0)

    def test_maker_timeout_cancels_without_taker_fallback(self) -> None:
        engine = self._engine()
        engine.on_book_snapshot(_snapshot())
        engine.on_agg_trade(AggTradeEvent(SYMBOL, START + timedelta(milliseconds=50), 99.99, 10.0, True, 1))
        engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=100)))
        engine.on_depth_update(_depth(START + timedelta(milliseconds=200), 101, 101, 100, 100.0))
        engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=250)))
        engine.on_depth_update(_depth(START + timedelta(milliseconds=500), 102, 102, 101, 1_200.0))
        entry = engine.evaluate(SYMBOL, _context(START + timedelta(milliseconds=550)))[0]

        engine.on_agg_trade(AggTradeEvent(SYMBOL, START + timedelta(seconds=2), 100.0, 1.0, False, 2))
        engine.on_depth_update(_depth(START + timedelta(seconds=2), 103, 103, 102, 1_200.0))
        actions = engine.evaluate(SYMBOL, _context(START + timedelta(seconds=2)))
        self.assertEqual(actions[0].action, "CANCEL_ENTRY")
        self.assertEqual(actions[0].reason, "ofas_maker_timeout")
        self.assertFalse(any(item.order_type == "MARKET" and not item.reduce_only for item in actions))
        self.assertEqual(entry.event_id, actions[0].event_id)


if __name__ == "__main__":
    unittest.main()
