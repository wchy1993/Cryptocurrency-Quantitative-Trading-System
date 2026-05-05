from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from crypto_scalper.binance_client import BinanceApiError, BinanceFuturesClient, SymbolRules
from crypto_scalper.live_config import DEFAULT_SYMBOLS, default_live_config, load_live_config, write_live_config
from crypto_scalper.live_trader import AccountSnapshot, BinanceAutoTrader, LivePosition, MarketState, SimPosition
from crypto_scalper.models import Candle, Direction, Signal


class LiveConfigTests(unittest.TestCase):
    def test_live_config_round_trip_normalizes_symbols(self) -> None:
        config = default_live_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live.json"
            write_live_config(path, config)
            loaded = load_live_config(path)
        self.assertIn("BTCUSDT", loaded.trading.symbols)
        self.assertTrue(loaded.trading.dry_run)

    def test_default_live_config_uses_cost_aware_short_timeframe_universe(self) -> None:
        config = default_live_config()
        self.assertEqual(config.trading.symbols, DEFAULT_SYMBOLS)
        self.assertEqual(len(config.trading.symbols), 50)
        self.assertIn("TONUSDT", config.trading.symbols)
        self.assertIn("1000BONKUSDT", config.trading.symbols)
        self.assertEqual(config.trading.timeframe, "15m")
        self.assertEqual(config.trading.kline_limit, 240)
        self.assertEqual(config.trading.poll_seconds, 8)
        self.assertEqual(config.trading.leverage, 5)
        self.assertEqual(config.trading.max_open_positions, 3)
        self.assertEqual(config.trading.max_long_positions, 3)
        self.assertEqual(config.trading.max_short_positions, 3)
        self.assertEqual(config.trading.candidate_batch_size, 1)
        self.assertEqual(config.trading.entry_frequency_window_seconds, 3600)
        self.assertEqual(config.trading.max_entries_per_window, 10)
        self.assertEqual(config.trading.max_symbol_entries_per_window, 1)
        self.assertEqual(config.trading.min_symbol_reentry_seconds, 1800)
        self.assertEqual(config.filters.timeframes, ("5m", "15m", "1h"))
        self.assertEqual(config.filters.rsi_period, 7)
        self.assertEqual(config.filters.macd_fast, 6)
        self.assertEqual(config.filters.macd_slow, 13)
        self.assertEqual(config.filters.kdj_period, 5)
        self.assertEqual(config.trading.initial_entry_fraction, 0.80)
        self.assertEqual(config.trading.scale_in_entry_fraction, 0.20)
        self.assertEqual(config.trading.max_scale_ins_per_symbol, 1)
        self.assertFalse(config.trading.allow_loss_scale_in)
        self.assertEqual(config.trading.min_signal_confidence, 0.70)
        self.assertEqual(config.trading.min_take_profit_cost_ratio, 6.0)
        self.assertEqual(config.trading.min_reward_risk_ratio, 2.3)
        self.assertEqual(config.trading.quick_take_profit_pct, 0.0120)
        self.assertEqual(config.trading.strong_take_profit_pct, 0.0200)
        self.assertEqual(config.trading.profit_exit_rsi_long, 72.0)
        self.assertEqual(config.trading.profit_exit_rsi_short, 28.0)
        self.assertEqual(config.filters.min_score, 6)
        self.assertEqual(config.filters.trend_timeframe, "4h")
        self.assertEqual(config.filters.range_timeframe, "1h")
        self.assertEqual(config.filters.trend_adx_threshold, 24.0)
        self.assertEqual(config.filters.range_adx_threshold, 18.0)
        self.assertEqual(config.risk.starting_capital_usdt, 10000.0)
        self.assertEqual(config.risk.max_position_notional_usdt, 4000.0)
        self.assertEqual(config.risk.max_account_margin_usage_pct, 0.10)
        self.assertEqual(config.risk.max_symbol_margin_pct, 0.040)
        self.assertEqual(config.risk.risk_per_trade_pct, 0.004)
        self.assertEqual(config.risk.fee_bps, 5.0)
        self.assertEqual(config.risk.slippage_bps, 2.0)
        self.assertEqual(config.trading.breakeven_trigger_pct, 0.0050)
        self.assertEqual(config.trading.breakeven_lock_pct, 0.0042)
        self.assertEqual(config.trading.trailing_activation_pct, 0.0065)
        self.assertTrue(config.trading.condition_stats_enabled)
        self.assertEqual(config.trading.condition_stats_log_interval_seconds, 60)
        self.assertTrue(config.trading.use_btc_market_state_filter)
        self.assertTrue(config.trading.use_symbol_trend_filter)
        self.assertFalse(config.trading.use_symbol_range_filter)
        self.assertTrue(config.trading.use_btc_direction_filter)
        self.assertTrue(config.trading.use_confidence_filter)
        self.assertTrue(config.trading.use_cost_edge_filter)
        self.assertTrue(config.trading.use_reward_risk_filter)
        self.assertTrue(config.trading.use_trend_atr_filter)
        self.assertTrue(config.trading.use_trend_adx_filter)
        self.assertTrue(config.trading.use_trend_volume_filter)
        self.assertTrue(config.trading.use_trend_ema_filter)
        self.assertTrue(config.trading.use_trend_setup_filter)
        self.assertTrue(config.trading.use_trend_score_filter)
        self.assertFalse(config.trading.trend_continuation_entry_enabled)
        self.assertEqual(config.trading.trend_continuation_max_holding_bars, 8)
        self.assertTrue(config.trading.use_bollinger_reclaim_entry)
        self.assertTrue(config.trading.use_rsi_extreme_entry)
        self.assertTrue(config.trading.session_profit_guard_enabled)
        self.assertEqual(config.trading.session_profit_guard_trigger_usdt, 0.35)
        self.assertEqual(config.trading.session_profit_guard_pullback_usdt, 0.20)
        self.assertEqual(config.strategy.channel_period, 32)
        self.assertEqual(config.strategy.min_volume_ratio, 1.00)
        self.assertEqual(config.strategy.mean_reversion_stop_atr, 1.2)
        self.assertEqual(config.filters.trend_atr_percentile_min, 25.0)
        self.assertEqual(config.filters.trend_bb_width_percentile_min, 25.0)
        self.assertEqual(config.filters.rsi_oversold, 35.0)
        self.assertEqual(config.filters.rsi_overbought, 65.0)
        self.assertEqual(config.filters.rsi_long_floor, 26.0)
        self.assertEqual(config.filters.rsi_short_ceiling, 74.0)
        self.assertEqual(config.filters.vwap_period, 96)
        self.assertEqual(config.filters.trend_score_entry, 5)
        self.assertEqual(config.filters.trend_score_normal, 5)
        self.assertEqual(config.filters.trend_score_strong, 6)
        self.assertEqual(config.filters.btc_opportunity_adx_threshold, 24.0)
        self.assertGreaterEqual(config.strategy.min_atr_pct, 0.0012)
        self.assertGreaterEqual(config.strategy.take_profit_atr, 6.0)
        self.assertEqual(config.risk.cooldown_seconds_after_loss, 0)
        self.assertEqual(config.risk.max_consecutive_losses, 0)
        self.assertEqual(config.risk.symbol_loss_cooldown_seconds, 0)
        self.assertEqual(config.risk.max_portfolio_risk_pct, 0.015)

    def test_default_live_config_requires_confirmed_reversal_entries(self) -> None:
        config = default_live_config()
        self.assertTrue(config.filters.extreme_reversal_entry_enabled)
        self.assertFalse(config.filters.pre_cross_entry_enabled)
        self.assertEqual(config.filters.reversal_cross_lookback_bars, 3)
        self.assertLessEqual(config.filters.long_extreme_rsi, 24.0)
        self.assertGreaterEqual(config.filters.short_extreme_rsi, 76.0)
        self.assertLess(config.filters.pre_cross_risk_multiplier, config.filters.confirmed_cross_risk_multiplier)

    def test_symbol_rules_rounding(self) -> None:
        rules = SymbolRules(
            symbol="BTCUSDT",
            quantity_step="0.001",
            min_quantity="0.001",
            price_tick="0.10",
            min_notional="5",
        )
        self.assertEqual(rules.round_quantity(0.123456), "0.123")
        self.assertEqual(rules.round_price(123.456), "123.4")

    def test_conditional_orders_use_binance_algo_endpoint(self) -> None:
        client = RecordingBinanceClient()

        client.new_stop_market_order("ADAUSDT", "BUY", "0.251", "65", reduce_only=True, working_type="MARK_PRICE")
        client.new_take_profit_market_order("ADAUSDT", "BUY", "0.248", "65", reduce_only=True, working_type="MARK_PRICE")
        client.cancel_all_algo_open_orders("ADAUSDT")

        stop_call = client.calls[0]
        take_profit_call = client.calls[1]
        cancel_call = client.calls[2]
        self.assertEqual(stop_call[1], "/fapi/v1/algoOrder")
        self.assertEqual(stop_call[2]["algoType"], "CONDITIONAL")
        self.assertEqual(stop_call[2]["triggerPrice"], "0.251")
        self.assertNotIn("stopPrice", stop_call[2])
        self.assertEqual(take_profit_call[1], "/fapi/v1/algoOrder")
        self.assertEqual(take_profit_call[2]["type"], "TAKE_PROFIT_MARKET")
        self.assertEqual(cancel_call[0], "DELETE")
        self.assertEqual(cancel_call[1], "/fapi/v1/algoOpenOrders")

    def test_live_protective_order_failure_market_closes_without_reraising(self) -> None:
        config = default_live_config()
        client = ProtectiveFailureClient()
        trader = BinanceAutoTrader(config, client)
        signal = Signal(Direction.SHORT, 0.8, "bb_reclaim_v2", 0.01, 0.02)

        trader._place_protective_orders("ADAUSDT", signal, "65", 0.2497)

        self.assertIn(("cancel_open", "ADAUSDT"), client.calls)
        self.assertIn(("cancel_algo", "ADAUSDT"), client.calls)
        self.assertIn(("market", "ADAUSDT", "BUY", "65", True), client.calls)

    def test_dry_run_enter_position_is_reflected_in_snapshot(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        candle = Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
        signal = Signal(Direction.LONG, 1.0, "test", 0.01, 0.02)
        trader._enter_position("BTCUSDT", signal, candle, "0.1")
        snapshot = trader.snapshot_account()
        self.assertIn("BTCUSDT", snapshot.positions)
        self.assertEqual(snapshot.positions["BTCUSDT"].direction, Direction.LONG)
        self.assertGreater(snapshot.initial_margin, 0)

    def test_initial_live_order_uses_fractional_entry_size(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        account = AccountSnapshot(
            equity=120.0,
            wallet_balance=120.0,
            available_balance=120.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        signal = Signal(Direction.LONG, 1.0, "test", 0.005, 0.01)
        qty, reason = trader._size_order("BTCUSDT", 100.0, signal, account)
        self.assertTrue(reason.startswith("ok "))
        self.assertIn("expected_loss", reason)
        notional = float(qty) * 100.0
        self.assertLessEqual(notional, config.risk.max_position_notional_usdt * config.trading.initial_entry_fraction)

    def test_large_account_sizing_uses_meaningful_notional(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, initial_entry_fraction=1.0))
        trader = BinanceAutoTrader(config, FakeClient())
        account = AccountSnapshot(
            equity=10000.0,
            wallet_balance=10000.0,
            available_balance=10000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        signal = Signal(Direction.LONG, 1.0, "score=6", 0.02, 0.05)
        qty, reason = trader._size_order("BTCUSDT", 100.0, signal, account)

        self.assertTrue(reason.startswith("ok "))
        self.assertIn("risk_usdt=40.00U", reason)
        self.assertIn("stop_distance=2.000%", reason)
        self.assertGreaterEqual(float(qty) * 100.0, 1500.0)

    def test_same_direction_capacity_blocks_extra_positions(self) -> None:
        trader = BinanceAutoTrader(default_live_config(), FakeClient())

        allowed, reason = trader._direction_capacity_allows(Direction.LONG, [Direction.LONG, Direction.LONG, Direction.LONG])

        self.assertFalse(allowed)
        self.assertIn("long_count", reason)

    def test_entry_frequency_blocks_same_symbol_on_same_candle(self) -> None:
        trader = BinanceAutoTrader(default_live_config(), FakeClient())
        candle_time = datetime(2025, 1, 1, 0, 0)

        trader._record_entry("BTCUSDT", candle_time)
        allowed, reason = trader._symbol_entry_frequency_allows("BTCUSDT", candle_time)

        self.assertFalse(allowed)
        self.assertIn("same_candle", reason)

    def test_global_entry_frequency_limit_blocks_overtrading(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        now = datetime.now().timestamp()
        trader._entry_timestamps = [now] * config.trading.max_entries_per_window

        self.assertFalse(trader._global_entry_frequency_allows())

    def test_same_direction_risk_scales_down_next_order(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(base.trading, initial_entry_fraction=1.0),
            risk=replace(base.risk, max_position_notional_usdt=500.0, max_symbol_margin_pct=1.0),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        signal = Signal(Direction.LONG, 1.0, "score=6", 0.01, 0.03)
        flat_account = AccountSnapshot(120.0, 120.0, 120.0, 0.0, 0.0, 0.0, {})
        long_position = LivePosition("ETHUSDT", "BOTH", Direction.LONG, 0.1, 100.0, 100.0, 10.0, 0.0, 20, "CROSSED", None)
        one_long_account = AccountSnapshot(120.0, 120.0, 120.0, 0.0, 0.0, 0.0, {"ETHUSDT": long_position})

        first_qty, first_reason = trader._size_order("BTCUSDT", 100.0, signal, flat_account)
        second_qty, second_reason = trader._size_order("BTCUSDT", 100.0, signal, one_long_account)

        self.assertTrue(first_reason.startswith("ok "))
        self.assertTrue(second_reason.startswith("ok "))
        self.assertLess(float(second_qty), float(first_qty))

    def test_dry_run_scale_in_merges_position(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        signal = Signal(Direction.LONG, 1.0, "test", 0.01, 0.02)
        first = Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
        second = Candle(datetime(2025, 1, 1, 0, 1), 101.0, 102.0, 100.0, 101.0, 1000.0)
        trader._enter_position("BTCUSDT", signal, first, "0.1")
        trader._enter_position("BTCUSDT", signal, second, "0.05", scale_in=True)
        snapshot = trader.snapshot_account()
        position = snapshot.positions["BTCUSDT"]
        self.assertAlmostEqual(position.quantity, 0.15)
        self.assertAlmostEqual(position.entry_price, (100.02 * 0.1 + 101.0202 * 0.05) / 0.15)
        self.assertGreaterEqual(trader._sim_positions["BTCUSDT"].stop_price, 99.0)

    def test_dry_run_charges_fee_and_slippage(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        entry = Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
        signal = Signal(Direction.LONG, 1.0, "test", 0.05, 0.02)

        trader._enter_position("BTCUSDT", signal, entry, "0.1")
        self.assertAlmostEqual(trader._sim_positions["BTCUSDT"].entry_price, 100.02)
        self.assertAlmostEqual(trader.stats.realized_pnl, -0.005001)

        trader._close_sim_position("BTCUSDT", 101.0, "manual")
        expected_exit = 101.0 * (1.0 - 0.0002)
        expected_gross = 0.1 * (expected_exit - 100.02)
        expected_exit_fee = 0.1 * expected_exit * 0.0005
        expected_net = expected_gross - 0.005001 - expected_exit_fee
        self.assertAlmostEqual(trader.stats.realized_pnl, expected_net)

    def test_signal_edge_filter_rejects_trades_that_cannot_cover_costs(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        weak_signal = Signal(Direction.LONG, 0.7, "tiny", 0.001, 0.002)
        strong_signal = Signal(Direction.LONG, 0.7, "wide", 0.004, 0.010)

        self.assertFalse(trader._signal_has_enough_edge(weak_signal)[0])
        self.assertTrue(trader._signal_has_enough_edge(strong_signal)[0])

    def test_profit_exit_waits_until_cost_buffer_is_covered(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                breakeven_trigger_pct=0.001,
                breakeven_lock_pct=0.0002,
                trailing_activation_pct=0.001,
                quick_take_profit_pct=0.001,
                strong_take_profit_pct=0.002,
                momentum_exit_min_profit_pct=0.001,
            ),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        candle = Candle(datetime(2025, 1, 1), 100.0, 100.31, 99.9, 100.30, 1000.0)
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=0.1,
            entry_price=100.0,
            stop_price=99.0,
            take_profit_price=110.0,
            max_holding_bars=48,
            entry_time=candle.timestamp,
            last_checked_time=candle.timestamp,
            best_price=100.31,
        )

        self.assertLess(0.003, trader._minimum_profit_exit_pct())
        self.assertIsNone(trader._profit_exit_reason(position, [candle], current_candle=candle))

    def test_profit_protection_locks_cost_aware_profit(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                breakeven_trigger_pct=0.001,
                breakeven_lock_pct=0.0002,
                trailing_activation_pct=1.0,
            ),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        candle = Candle(datetime(2025, 1, 1), 100.0, 100.70, 99.9, 100.60, 1000.0)
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=0.1,
            entry_price=100.0,
            stop_price=99.0,
            take_profit_price=110.0,
            max_holding_bars=48,
            entry_time=candle.timestamp,
            last_checked_time=candle.timestamp,
            best_price=100.0,
        )

        trader._update_sim_profit_protection(position, candle)

        self.assertAlmostEqual(position.stop_price, 100.0 * (1.0 + trader._cost_aware_profit_lock_pct()))

    def test_market_state_classifier_selects_trend_mode(self) -> None:
        config = default_live_config()
        client = MultiFrameClient(
            {
                "4h": _trend_candles(240, start=100.0, step=0.45),
                "1h": _range_candles(240),
                "15m": _breakout_candles(),
            }
        )
        trader = BinanceAutoTrader(config, client)
        state = trader._classify_market_state("BTCUSDT")

        self.assertEqual(state.mode, "trend")
        self.assertEqual(state.direction, Direction.LONG)

    def test_market_state_classifier_selects_range_mode(self) -> None:
        config = default_live_config()
        client = MultiFrameClient(
            {
                "4h": _range_candles(240),
                "1h": _range_candles(240),
                "15m": _bollinger_reclaim_candles(),
            }
        )
        trader = BinanceAutoTrader(config, client)
        state = trader._classify_market_state("BTCUSDT")

        self.assertEqual(state.mode, "range")

    def test_btc_opportunity_state_does_not_global_lock_symbols(self) -> None:
        config = default_live_config()
        client = MultiFrameClient(
            {
                "4h": _range_candles(240),
                "1h": _trend_candles(240, start=100.0, step=0.20),
                "15m": _trend_pullback_candles(),
            }
        )
        trader = BinanceAutoTrader(config, client)
        global_state = trader._classify_global_market_state()
        symbol_state = trader._classify_market_state("DOGEUSDT", global_state=global_state)

        self.assertEqual(global_state.mode, "opportunity")
        self.assertNotEqual(symbol_state.mode, "no_trade")

    def test_trend_breakout_signal_uses_donchian_confirmation(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            filters=replace(
                base.filters,
                trend_atr_percentile_min=0.0,
                trend_bb_width_percentile_min=0.0,
                trend_adx_threshold=0.0,
                trend_score_entry=4,
            ),
            strategy=replace(base.strategy, max_atr_pct=0.02),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_breakout_candles(), MarketState("trend", Direction.LONG, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertIn("trend_breakout_v2", signal.reason)
        self.assertGreater(signal.stop_loss_pct, 0.0)

    def test_bollinger_reclaim_signal_waits_for_close_back_inside_band(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_bollinger_reclaim_candles(), MarketState("range", Direction.FLAT, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertIn("bb_reclaim_v2", signal.reason)

    def test_range_rsi_extreme_reversal_can_enter_without_bollinger_reclaim(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_rsi_extreme_reversal_candles(), MarketState("range", Direction.FLAT, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertIn("rsi_extreme_reversal_v1", signal.reason)

    def test_trend_pullback_signal_can_enter_without_donchian_breakout(self) -> None:
        base = default_live_config()
        config = replace(base, strategy=replace(base.strategy, max_atr_pct=0.02))
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_trend_pullback_candles(), MarketState("trend", Direction.LONG, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertIn("trend_ema_pullback_v1", signal.reason)

    def test_trend_momentum_signal_can_follow_confirmed_uptrend(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            filters=replace(base.filters, trend_adx_threshold=0.0, trend_atr_percentile_min=0.0, trend_bb_width_percentile_min=0.0),
            strategy=replace(base.strategy, max_atr_pct=0.03, min_volume_ratio=0.0),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_trend_momentum_candles(), MarketState("trend", Direction.LONG, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertIn("trend_momentum_v2", signal.reason)
        self.assertGreater(trader._condition_stats["trend_momentum"].passed, 0)

    def test_trend_continuation_signal_enters_without_breakout_or_pullback(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(base.trading, trend_continuation_entry_enabled=True),
            filters=replace(base.filters, trend_atr_percentile_min=0.0, trend_bb_width_percentile_min=0.0),
            strategy=replace(base.strategy, max_atr_pct=0.03, min_volume_ratio=0.0),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        signal = trader._dual_mode_signal(_trend_continuation_candles(), MarketState("trend", Direction.LONG, "test"))

        self.assertEqual(signal.direction, Direction.LONG)
        self.assertGreater(trader._condition_stats["trend_continuation"].passed, 0)
        self.assertEqual(signal.max_holding_bars, config.trading.trend_continuation_max_holding_bars)

    def test_trend_continuation_positions_use_trend_exit_logic(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        signal = Signal(Direction.LONG, 1.0, "trend_continuation_v1 score=5", 0.01, 0.03, max_holding_bars=3)
        self.assertEqual(trader._signal_mode(signal), "trend")

        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_price=99.0,
            take_profit_price=103.0,
            max_holding_bars=3,
            entry_time=datetime(2025, 1, 1),
            last_checked_time=datetime(2025, 1, 1),
            best_price=100.6,
            mode="trend",
            initial_stop_distance=1.0,
            bars_held=3,
        )
        profitable_candle = Candle(datetime(2025, 1, 1, 0, 45), 100.5, 100.7, 100.4, 100.6, 1000.0)

        self.assertIsNone(trader._sim_time_exit_reason(position, profitable_candle))

    def test_dry_run_loss_does_not_start_global_cooldown_by_default(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        entry = Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
        signal = Signal(Direction.LONG, 1.0, "test", 0.05, 0.02)

        trader._enter_position("BTCUSDT", signal, entry, "0.1")
        trader._close_sim_position("BTCUSDT", 99.0, "manual_loss")

        self.assertEqual(trader._cooldown_until, 0.0)
        self.assertEqual(trader.stats.losing_trades, 1)

    def test_dry_run_does_not_exit_on_entry_candle_history(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        candle = Candle(datetime(2025, 1, 1), 100.0, 101.0, 90.0, 100.0, 1000.0)
        signal = Signal(Direction.LONG, 1.0, "test", 0.05, 0.02)
        trader._enter_position("BTCUSDT", signal, candle, "0.1")
        trader._manage_sim_positions()
        snapshot = trader.snapshot_account()
        self.assertIn("BTCUSDT", snapshot.positions)
        self.assertEqual(trader.stats.closed_trades, 0)

    def test_dry_run_profit_protection_exits_winner(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                breakeven_trigger_pct=0.001,
                breakeven_lock_pct=0.0002,
                trailing_activation_pct=0.002,
                trailing_pullback_pct=0.002,
                momentum_exit_min_profit_pct=1.0,
            ),
        )
        entry = Candle(datetime(2025, 1, 1), 100.0, 100.1, 99.9, 100.0, 1000.0)
        profit = Candle(datetime(2025, 1, 1, 0, 1), 100.0, 100.5, 99.95, 100.4, 1200.0)
        pullback = Candle(datetime(2025, 1, 1, 0, 2), 100.4, 100.42, 100.20, 100.25, 1200.0)
        open_candle = Candle(datetime(2025, 1, 1, 0, 3), 100.25, 100.3, 100.1, 100.2, 900.0)
        client = SequenceClient([entry, profit, pullback, open_candle])
        trader = BinanceAutoTrader(config, client)
        signal = Signal(Direction.LONG, 1.0, "test", 0.05, 0.10)
        trader._enter_position("BTCUSDT", signal, entry, "0.1")
        trader._manage_sim_positions()
        snapshot = trader.snapshot_account()
        self.assertNotIn("BTCUSDT", snapshot.positions)
        self.assertEqual(trader.stats.closed_trades, 1)
        self.assertEqual(trader.stats.winning_trades, 1)
        self.assertGreater(trader.stats.realized_pnl, 0.0)

    def test_indicator_dead_cross_above_zero_creates_short_signal(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            filters=replace(
                base.filters,
                rsi_period=7,
                macd_fast=2,
                macd_slow=5,
                macd_signal=2,
                kdj_period=5,
                short_extreme_rsi=101.0,
                short_extreme_kdj=101.0,
            ),
        )
        candles = _up_then_rollover_candles()[:20]
        trader = BinanceAutoTrader(config, SequenceClient(candles))
        signal = trader._indicator_reversal_signal(candles)
        self.assertEqual(signal.direction, Direction.SHORT)
        self.assertIn("indicator_short", signal.reason)
        self.assertGreater(signal.stop_loss_pct, 0.0)

    def test_session_profit_guard_no_longer_pauses_new_entries_after_pullback(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                session_profit_guard_enabled=True,
                session_profit_guard_trigger_usdt=0.45,
                session_profit_guard_pullback_usdt=0.25,
                session_profit_guard_cooldown_seconds=600,
            ),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        high = AccountSnapshot(
            equity=config.risk.starting_capital_usdt + 0.60,
            wallet_balance=config.risk.starting_capital_usdt + 0.60,
            available_balance=config.risk.starting_capital_usdt,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        pulled_back = AccountSnapshot(
            equity=config.risk.starting_capital_usdt + 0.30,
            wallet_balance=config.risk.starting_capital_usdt + 0.30,
            available_balance=config.risk.starting_capital_usdt,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        self.assertTrue(trader._global_risk_allows_trading(high))
        self.assertTrue(trader._global_risk_allows_trading(pulled_back))

    def test_session_profit_guard_closes_open_positions_on_peak_pullback(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                session_profit_guard_enabled=True,
                session_profit_guard_trigger_usdt=0.35,
                session_profit_guard_pullback_usdt=0.20,
            ),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        trader._session_peak_pnl = 0.60
        position = LivePosition("BTCUSDT", "SIM", Direction.LONG, 0.1, 100.0, 103.0, 10.0, 0.0, 20, "CROSSED", None)
        account = AccountSnapshot(
            equity=config.risk.starting_capital_usdt + 0.30,
            wallet_balance=config.risk.starting_capital_usdt + 0.30,
            available_balance=config.risk.starting_capital_usdt,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={"BTCUSDT": position},
        )
        closed: list[tuple[str, str]] = []

        def fake_exit(symbol: str, live_position: LivePosition, reason: str = "strategy_exit") -> None:
            closed.append((symbol, reason))

        trader._exit_position = fake_exit  # type: ignore[method-assign]

        self.assertTrue(trader._session_profit_guard_closes_positions(account))
        self.assertEqual(closed, [("BTCUSDT", "session_profit_pullback_guard")])

    def test_session_profit_guard_does_not_pause_when_flat(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        trader._session_peak_pnl = 0.60
        flat = AccountSnapshot(
            equity=config.risk.starting_capital_usdt + 0.30,
            wallet_balance=config.risk.starting_capital_usdt + 0.30,
            available_balance=config.risk.starting_capital_usdt,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )

        self.assertFalse(trader._session_profit_guard_closes_positions(flat))


class FakeClient:
    api_key = "key"
    api_secret = "secret"

    def klines(self, symbol: str, interval: str, limit: int = 200):
        return [Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)]

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")


class RecordingBinanceClient(BinanceFuturesClient):
    def __init__(self) -> None:
        super().__init__("key", "secret")
        self.calls: list[tuple[str, str, dict]] = []

    def _signed_request(self, method: str, path: str, params=None):  # type: ignore[override]
        payload = dict(params or {})
        self.calls.append((method, path, payload))
        return payload


class ProtectiveFailureClient(FakeClient):
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def new_stop_market_order(self, symbol: str, side: str, stop_price: str, quantity: str, reduce_only: bool = True, working_type: str = "MARK_PRICE"):
        self.calls.append(("stop", symbol, side, stop_price, quantity, reduce_only, working_type))
        raise BinanceApiError(400, "stop failed", {"code": -4120})

    def new_take_profit_market_order(self, symbol: str, side: str, stop_price: str, quantity: str, reduce_only: bool = True, working_type: str = "MARK_PRICE"):
        self.calls.append(("take_profit", symbol, side, stop_price, quantity, reduce_only, working_type))
        return {}

    def cancel_all_open_orders(self, symbol: str):
        self.calls.append(("cancel_open", symbol))
        return {}

    def cancel_all_algo_open_orders(self, symbol: str):
        self.calls.append(("cancel_algo", symbol))
        return {}

    def new_market_order(self, symbol: str, side: str, quantity: str, reduce_only: bool = False, new_client_order_id: str | None = None):
        self.calls.append(("market", symbol, side, quantity, reduce_only))
        return {}


class SequenceClient(FakeClient):
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles

    def klines(self, symbol: str, interval: str, limit: int = 200):
        return self.candles[-limit:]


class MultiFrameClient(FakeClient):
    def __init__(self, candles_by_interval: dict[str, list[Candle]]) -> None:
        self.candles_by_interval = candles_by_interval

    def klines(self, symbol: str, interval: str, limit: int = 200):
        candles = self.candles_by_interval.get(interval) or self.candles_by_interval.get("15m") or []
        return candles[-limit:]


def _trend_candles(count: int, start: float, step: float) -> list[Candle]:
    candles: list[Candle] = []
    price = start
    for index in range(count):
        open_price = price
        close_price = price + step
        high = close_price + abs(step) * 0.8 + 0.2
        low = open_price - abs(step) * 0.6 - 0.2
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, high, low, close_price, 1200.0 + index))
        price = close_price
    return candles


def _range_candles(count: int) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(count):
        open_price = 100.0
        close_price = 100.0
        high = 100.25
        low = 99.75
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, high, low, close_price, 1000.0))
    return candles


def _breakout_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(69):
        drift = 0.08 if index % 4 in {0, 1} else -0.08
        open_price = price
        close_price = price + drift
        high = max(open_price, close_price) + 0.08
        low = min(open_price, close_price) - 0.08
        volume = 1000.0
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, high, low, close_price, volume))
        price = close_price
    open_price = price
    close_price = price + 0.36
    candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, close_price + 0.08, open_price - 0.08, close_price, 1800.0))
    return candles


def _bollinger_reclaim_candles() -> list[Candle]:
    candles = _range_candles(110)
    price = candles[-1].close
    for index in range(3):
        open_price = price
        close_price = price - 0.7
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, open_price + 0.1, close_price - 0.2, close_price, 1000.0))
        price = close_price
    reclaim_open = price
    reclaim_close = price + 0.9
    candles.append(Candle(datetime(2025, 1, 1, 0, 0), reclaim_open, reclaim_close + 0.2, reclaim_open - 1.0, reclaim_close, 1100.0))
    return candles


def _rsi_extreme_reversal_candles() -> list[Candle]:
    candles = _range_candles(110)
    price = candles[-1].close
    for _ in range(8):
        open_price = price
        close_price = price - 0.55
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, open_price + 0.1, close_price - 0.2, close_price, 1000.0))
        price = close_price
    reversal_open = price
    reversal_close = price + 0.2
    candles.append(Candle(datetime(2025, 1, 1, 0, 0), reversal_open, reversal_close + 0.1, reversal_open - 0.05, reversal_close, 1000.0))
    return candles


def _trend_pullback_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(70):
        open_price = price
        close_price = price + 0.25
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, close_price + 0.2, open_price - 0.1, close_price, 1000.0 + index))
        price = close_price
    for _ in range(3):
        open_price = price
        close_price = price - 0.5
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, open_price + 0.1, close_price - 0.2, close_price, 1000.0))
        price = close_price
    pullback_open = price
    pullback_close = price + 0.7
    candles.append(Candle(datetime(2025, 1, 1, 0, 0), pullback_open, pullback_close + 0.15, pullback_open - 0.1, pullback_close, 1300.0))
    return candles


def _trend_continuation_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(90):
        open_price = price
        drift = 0.18 if index % 5 in {0, 1, 2} else -0.11
        close_price = price + drift
        high = close_price + 0.8
        low = open_price - 0.8
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, high, low, close_price, 1000.0 + index))
        price = close_price
    open_price = price
    close_price = price + 0.10
    candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, close_price + 4.0, open_price - 0.2, close_price, 1050.0))
    return candles


def _trend_momentum_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(220):
        drift = 0.07 if index % 6 in {0, 1, 2} else -0.07
        open_price = price
        close_price = price + drift
        high = max(open_price, close_price) + 0.30
        low = min(open_price, close_price) - 0.30
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, high, low, close_price, 1000.0 + index))
        price = close_price
    for _ in range(3):
        open_price = price
        close_price = price + 0.055
        candles.append(Candle(datetime(2025, 1, 1, 0, 0), open_price, close_price + 0.80, open_price - 0.80, close_price, 1800.0))
        price = close_price
    return candles


def _up_then_rollover_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    for index in range(18):
        open_price = price
        close_price = price + 0.5
        candles.append(Candle(datetime(2025, 1, 1, 0, index), open_price, close_price + 0.2, open_price - 0.1, close_price, 1000.0))
        price = close_price
    for index in range(18, 26):
        open_price = price
        close_price = price - 0.6
        candles.append(Candle(datetime(2025, 1, 1, 0, index), open_price, open_price + 0.1, close_price - 0.2, close_price, 1000.0))
        price = close_price
    return candles


if __name__ == "__main__":
    unittest.main()
