from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time
import unittest

from crypto_scalper.binance_client import BinanceApiError, BinanceFuturesClient, SymbolRules
from crypto_scalper.gui import (
    STRATEGY_MODE_INDICATOR,
    STRATEGY_MODE_MTF,
    STRATEGY_MODE_SUPER_VOLUME,
    _config_with_strategy_selection,
    _config_with_strategy_mode,
    _position_symbols_first,
)
from crypto_scalper.live_config import default_live_config, load_live_config, write_live_config
from crypto_scalper.live_trader import (
    AccountSnapshot,
    BinanceAutoTrader,
    EntryCandidate,
    LivePosition,
    SimPosition,
    _candidate_requires_extra_slot,
    _entry_position_limit,
    _super_volume_extra_slot_candidate_allowed,
)
from crypto_scalper.models import Candle, Direction, Signal


class LiveConfigTests(unittest.TestCase):
    def test_gui_ofas_selection_forces_paper_only_and_disables_other_strategies(self) -> None:
        base = default_live_config()
        config = _config_with_strategy_selection(
            base,
            indicator_enabled=True,
            vbp_enabled=True,
            ofas_enabled=True,
        )

        self.assertTrue(config.ofas_strategy.enabled)
        self.assertTrue(config.trading.dry_run)
        self.assertEqual(config.trading.max_open_positions, 1)
        self.assertEqual(config.trading.max_scale_ins_per_symbol, 0)
        self.assertFalse(config.vbp_strategy.enabled)
        self.assertFalse(config.filters.enabled)
        self.assertFalse(config.filters.extreme_reversal_entry_enabled)
        self.assertFalse(config.strategy.super_volume_breakout_enabled)
        self.assertFalse(config.strategy.mtf_4h_rsi_regime_enabled)
        self.assertFalse(config.strategy.oi_flush_reversal_enabled)
        self.assertFalse(config.macro_events.enabled)
        self.assertEqual(config.portfolio_control.max_indicator_positions, 0)
        self.assertEqual(config.portfolio_control.max_vbp_positions, 0)

    def test_ofas_gui_default_config_is_mainnet_market_data_dry_run(self) -> None:
        config = load_live_config("config.ofas.json")
        self.assertEqual(config.exchange.environment, "mainnet")
        self.assertTrue(config.trading.dry_run)
        self.assertTrue(config.ofas_strategy.enabled)
        self.assertEqual(len(config.ofas_strategy.symbols), 60)
        self.assertFalse(config.vbp_strategy.enabled)

    def test_position_symbols_are_ordered_with_active_positions_first(self) -> None:
        symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
        rows_by_symbol = {"BNBUSDT": [object()], "BTCUSDT": [object()]}

        ordered = _position_symbols_first(symbols, rows_by_symbol)

        self.assertEqual(ordered, ["BTCUSDT", "BNBUSDT", "ETHUSDT", "SOLUSDT"])

    def test_gui_indicator_strategy_mode_enables_only_indicator_reversal_risk0065(self) -> None:
        config = _config_with_strategy_mode(default_live_config(), STRATEGY_MODE_INDICATOR)

        self.assertTrue(config.filters.extreme_reversal_entry_enabled)
        self.assertFalse(config.filters.enabled)
        self.assertFalse(config.filters.pre_cross_entry_enabled)
        self.assertTrue(config.strategy.indicator_confirmed_cross_extreme_required_enabled)
        self.assertEqual(config.strategy.indicator_reversal_size_multiplier, 0.30)
        self.assertEqual(config.strategy.indicator_long_size_multiplier, 0.282)
        self.assertEqual(config.strategy.indicator_short_size_multiplier, 0.34)
        self.assertEqual(config.strategy.indicator_long_stop_loss_atr, 2.60)
        self.assertEqual(config.strategy.indicator_short_stop_loss_atr, 2.50)
        self.assertEqual(config.strategy.indicator_long_take_profit_atr, 1.60)
        self.assertEqual(config.strategy.indicator_short_take_profit_atr, 1.60)
        self.assertEqual(config.strategy.indicator_short_max_close_position, 0.55)
        self.assertEqual(config.strategy.indicator_short_high_close_risk_multiplier, 0.65)
        self.assertEqual(config.strategy.indicator_long_confirmed_cross_risk_multiplier, 0.45)
        self.assertEqual(config.strategy.indicator_short_confirmed_cross_risk_multiplier, 0.45)
        self.assertFalse(config.strategy.indicator_reversal_loss_pause_enabled)
        self.assertFalse(config.strategy.indicator_long_reclaim_filter_enabled)
        self.assertEqual(config.strategy.indicator_long_reclaim_ema_period, 9)
        self.assertEqual(config.strategy.indicator_long_reclaim_min_close_position, 0.55)
        self.assertFalse(config.strategy.indicator_long_fail_fast_enabled)
        self.assertEqual(config.strategy.indicator_long_fail_fast_minutes, 120)
        self.assertEqual(config.strategy.indicator_long_fail_fast_min_r, 0.25)
        self.assertTrue(config.strategy.allow_short)
        self.assertEqual(config.strategy.short_risk_bias, 1.05)
        self.assertFalse(config.strategy.btc_market_filter_enabled)
        self.assertFalse(config.strategy.weak_market_long_filter_enabled)
        self.assertTrue(config.strategy.strong_market_short_filter_enabled)
        self.assertEqual(config.strategy.strong_market_short_risk_multiplier, 0.45)
        self.assertFalse(config.strategy.super_volume_breakout_enabled)
        self.assertEqual(config.trading.max_open_positions, 4)
        self.assertFalse(config.trading.super_volume_extra_slot_enabled)
        self.assertEqual(config.risk.risk_per_trade_pct, 0.065)
        self.assertFalse(config.strategy.startup_breakout_enabled)
        self.assertFalse(config.strategy.ordinary_breakout_enabled)
        self.assertFalse(config.strategy.pullback_reclaim_enabled)
        self.assertFalse(config.strategy.fast_breakout_enabled)
        self.assertFalse(config.strategy.rsi_reversal_enabled)
        self.assertFalse(config.strategy.mtf_4h_rsi_regime_enabled)
        self.assertFalse(config.strategy.oi_flush_reversal_enabled)

    def test_gui_strategy_modes_switch_hidden_strategy_flags(self) -> None:
        base = default_live_config()
        super_volume = _config_with_strategy_mode(base, STRATEGY_MODE_SUPER_VOLUME)
        mtf = _config_with_strategy_mode(base, STRATEGY_MODE_MTF)

        self.assertTrue(super_volume.strategy.super_volume_breakout_enabled)
        self.assertFalse(super_volume.strategy.mtf_4h_rsi_regime_enabled)
        self.assertTrue(super_volume.trading.super_volume_extra_slot_enabled)
        self.assertTrue(mtf.strategy.mtf_4h_rsi_regime_enabled)
        self.assertTrue(mtf.strategy.mtf_disable_legacy_strategies)
        self.assertFalse(mtf.strategy.super_volume_breakout_enabled)

    def test_live_config_round_trip_normalizes_symbols(self) -> None:
        config = default_live_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "live.json"
            write_live_config(path, config)
            loaded = load_live_config(path)
        self.assertIn("BTCUSDT", loaded.trading.symbols)
        self.assertTrue(loaded.trading.dry_run)

    def test_live_config_supports_extends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "base.json"
            child = Path(temp_dir) / "child.json"
            base.write_text(
                """
{
  "trading": {"max_open_positions": 4, "dry_run": true},
  "ofas_strategy": {"trade_stale_ms": 1500}
}
""".strip(),
                encoding="utf-8",
            )
            child.write_text(
                """
{
  "extends": "base.json",
  "trading": {"max_open_positions": 1},
  "ofas_strategy": {"enabled": true, "trade_stale_ms": 5000}
}
""".strip(),
                encoding="utf-8",
            )
            loaded = load_live_config(child)

        self.assertEqual(loaded.trading.max_open_positions, 1)
        self.assertTrue(loaded.trading.dry_run)
        self.assertTrue(loaded.ofas_strategy.enabled)
        self.assertEqual(loaded.ofas_strategy.trade_stale_ms, 5000)

    def test_default_live_config_uses_ultra_short_multi_timeframe_universe(self) -> None:
        config = default_live_config()
        self.assertEqual(len(config.trading.symbols), 60)
        self.assertEqual(len(config.trading.entry_symbols), 60)
        self.assertIn("BTCUSDT", config.trading.entry_symbols)
        self.assertIn("ARBUSDT", config.trading.entry_symbols)
        self.assertIn("BNBUSDT", config.trading.entry_symbols)
        self.assertIn("SEIUSDT", config.trading.entry_symbols)
        self.assertIn("POLUSDT", config.trading.entry_symbols)
        self.assertIn("WLDUSDT", config.trading.entry_symbols)
        self.assertNotIn("币安人生USDT", config.trading.entry_symbols)
        self.assertEqual(config.trading.timeframe, "30m")
        self.assertEqual(config.trading.poll_seconds, 60)
        self.assertEqual(config.trading.entry_scan_seconds, 300)
        self.assertEqual(config.trading.symbol_reentry_cooldown_seconds, 3600)
        self.assertEqual(config.trading.leverage, 30)
        self.assertEqual(config.trading.max_open_positions, 4)
        self.assertFalse(config.trading.super_volume_extra_slot_enabled)
        self.assertEqual(config.trading.super_volume_extra_max_open_positions, 5)
        self.assertEqual(config.trading.max_new_entries_per_cycle, 1)
        self.assertEqual(config.filters.timeframes, ("30m", "1h", "2h"))
        self.assertEqual(config.filters.min_score, 4)
        self.assertEqual(config.filters.rsi_period, 14)
        self.assertEqual(config.filters.macd_fast, 12)
        self.assertEqual(config.filters.macd_slow, 26)
        self.assertEqual(config.filters.kdj_period, 9)
        self.assertEqual(config.trading.initial_entry_fraction, 0.75)
        self.assertEqual(config.trading.scale_in_entry_fraction, 0.35)
        self.assertEqual(config.trading.scale_in_min_profit_pct, 0.004)
        self.assertEqual(config.trading.max_scale_ins_per_symbol, 2)
        self.assertFalse(config.trading.allow_loss_scale_in)
        self.assertEqual(config.trading.loss_scale_in_entry_fraction, 0.25)
        self.assertEqual(config.risk.max_position_notional_usdt, 10_000.0)
        self.assertEqual(config.risk.risk_per_trade_pct, 0.06)
        self.assertEqual(config.risk.max_account_margin_usage_pct, 0.08)
        self.assertEqual(config.risk.max_symbol_margin_pct, 0.04)
        self.assertEqual(config.risk.min_symbol_margin_pct, 0.01)
        self.assertEqual(config.risk.max_daily_loss_pct, 0.15)
        self.assertEqual(config.risk.max_drawdown_pct, 0.15)
        self.assertEqual(config.risk.starting_capital_drawdown_stop_pct, 0.20)
        self.assertEqual(config.risk.weekly_profit_drawdown_stop_pct, 0.15)
        self.assertEqual(config.risk.soft_drawdown_reduce_pct, 0.05)
        self.assertEqual(config.risk.soft_drawdown_stop_pct, 0.10)
        self.assertEqual(config.risk.soft_drawdown_min_size_multiplier, 0.35)
        self.assertEqual(config.risk.estimated_fee_bps, 5.0)
        self.assertEqual(config.risk.min_profit_after_cost_pct, 0.001)
        self.assertFalse(config.strategy.rsi_reversal_enabled)

    def test_default_live_config_enables_extreme_reversal_entries(self) -> None:
        config = default_live_config()
        self.assertTrue(config.filters.extreme_reversal_entry_enabled)
        self.assertTrue(config.filters.pre_cross_entry_enabled)
        self.assertEqual(config.filters.reversal_cross_lookback_bars, 3)
        self.assertLessEqual(config.filters.long_extreme_rsi, 30.0)
        self.assertGreaterEqual(config.filters.short_extreme_rsi, 70.0)
        self.assertLess(config.filters.pre_cross_risk_multiplier, config.filters.confirmed_cross_risk_multiplier)

    def test_super_volume_extra_slot_requires_strong_startup_candidate(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(
                base.trading,
                super_volume_extra_slot_enabled=True,
                super_volume_extra_max_open_positions=5,
                super_volume_extra_min_rank_score=6.2,
                super_volume_extra_min_momentum_pct=0.035,
                super_volume_extra_min_volume_ratio=2.8,
            ),
        )
        candle = Candle(datetime(2025, 1, 1), 100.0, 105.0, 99.0, 104.0, 4000.0)
        strong = EntryCandidate(
            "BTCUSDT",
            Signal(Direction.LONG, 1.0, "long_breakout_super_volume volume=3.20x", 0.02, 0.03),
            candle,
            6.5,
            0.04,
            3.2,
            "ok",
        )
        weak = EntryCandidate(
            "ETHUSDT",
            Signal(Direction.LONG, 1.0, "long_breakout_super_volume volume=3.20x", 0.02, 0.03),
            candle,
            6.1,
            0.04,
            3.2,
            "ok",
        )
        normal = EntryCandidate(
            "BNBUSDT",
            Signal(Direction.LONG, 1.0, "long_breakout", 0.02, 0.03),
            candle,
            8.0,
            0.06,
            5.0,
            "ok",
        )

        self.assertEqual(_entry_position_limit(config), 5)
        self.assertFalse(_candidate_requires_extra_slot(config, 3))
        self.assertTrue(_candidate_requires_extra_slot(config, 4))
        self.assertTrue(_super_volume_extra_slot_candidate_allowed(config, strong))
        self.assertFalse(_super_volume_extra_slot_candidate_allowed(config, weak))
        self.assertFalse(_super_volume_extra_slot_candidate_allowed(config, normal))

    def test_account_snapshot_separates_initial_and_maintenance_margin_ratios(self) -> None:
        snapshot = AccountSnapshot(
            equity=100.0,
            wallet_balance=100.0,
            available_balance=95.91,
            initial_margin=4.09,
            maintenance_margin=0.98,
            total_unrealized_pnl=0.0,
            positions={},
        )

        self.assertAlmostEqual(snapshot.initial_margin_usage_pct, 0.0409)
        self.assertAlmostEqual(snapshot.margin_usage_pct, 0.0409)
        self.assertAlmostEqual(snapshot.maintenance_margin_ratio_pct, 0.0098)

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

    def test_stop_loss_and_take_profit_orders_use_algo_endpoint(self) -> None:
        client = RecordingBinanceClient()
        client.new_stop_market_order("btcusdt", "sell", "99.5", "0.1", reduce_only=True)
        client.new_take_profit_market_order("btcusdt", "sell", "102.5", "0.1", reduce_only=True)

        stop_order = client.calls[0]
        self.assertEqual(stop_order[0], "POST")
        self.assertEqual(stop_order[1], "/fapi/v1/algoOrder")
        self.assertEqual(stop_order[2]["algoType"], "CONDITIONAL")
        self.assertEqual(stop_order[2]["type"], "STOP_MARKET")
        self.assertEqual(stop_order[2]["triggerPrice"], "99.5")
        self.assertNotIn("stopPrice", stop_order[2])

        take_profit_order = client.calls[1]
        self.assertEqual(take_profit_order[1], "/fapi/v1/algoOrder")
        self.assertEqual(take_profit_order[2]["type"], "TAKE_PROFIT_MARKET")
        self.assertEqual(take_profit_order[2]["triggerPrice"], "102.5")

    def test_user_trades_uses_signed_trade_history_endpoint(self) -> None:
        client = RecordingBinanceClient()
        trades = client.user_trades("btcusdt", limit=250, start_time=1, end_time=2)

        self.assertEqual(trades, [{"ok": True}])
        request = client.calls[0]
        self.assertEqual(request[0], "GET")
        self.assertEqual(request[1], "/fapi/v1/userTrades")
        self.assertEqual(request[2]["symbol"], "BTCUSDT")
        self.assertEqual(request[2]["limit"], 250)
        self.assertEqual(request[2]["startTime"], 1)
        self.assertEqual(request[2]["endTime"], 2)

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
        self.assertEqual(reason, "ok")
        notional = float(qty) * 100.0
        self.assertLessEqual(notional, config.risk.max_position_notional_usdt * config.trading.initial_entry_fraction)
        self.assertGreaterEqual(notional, account.equity * config.risk.min_symbol_margin_pct * config.trading.leverage)

    def test_high_equity_min_margin_does_not_block_fixed_notional_cap(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        account = AccountSnapshot(
            equity=200_000.0,
            wallet_balance=200_000.0,
            available_balance=200_000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        signal = Signal(Direction.LONG, 1.0, "test", 0.005, 0.01)

        qty, reason = trader._size_order("BTCUSDT", 100.0, signal, account)

        self.assertEqual(reason, "ok")
        self.assertLessEqual(float(qty) * 100.0, config.risk.max_position_notional_usdt)

    def test_live_soft_drawdown_brake_reduces_and_stops_new_size(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        trader._peak_equity = 1_000.0
        signal = Signal(Direction.LONG, 1.0, "test", 0.005, 0.01, risk_multiplier=1.0)
        normal = AccountSnapshot(
            equity=1_000.0,
            wallet_balance=1_000.0,
            available_balance=1_000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        stressed = AccountSnapshot(
            equity=905.0,
            wallet_balance=905.0,
            available_balance=905.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        stopped = AccountSnapshot(
            equity=900.0,
            wallet_balance=900.0,
            available_balance=900.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )

        normal_qty, normal_reason = trader._size_order("BTCUSDT", 100.0, signal, normal)
        stressed_qty, stressed_reason = trader._size_order("BTCUSDT", 100.0, signal, stressed)
        stopped_qty, stopped_reason = trader._size_order("BTCUSDT", 100.0, signal, stopped)

        self.assertEqual(normal_reason, "ok")
        self.assertEqual(stressed_reason, "ok")
        self.assertEqual(stopped_reason, "ok")
        self.assertLess(float(stressed_qty), float(normal_qty))
        self.assertLess(float(stopped_qty), float(stressed_qty))

    def test_entry_scan_interval_and_symbol_reentry_cooldown(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())

        self.assertTrue(trader._entry_scan_due())
        trader._last_entry_scan_ts = time.time()
        self.assertFalse(trader._entry_scan_due())

        trader._mark_symbol_reentry_cooldown("BTCUSDT", "test")
        self.assertFalse(trader._symbol_reentry_allowed("BTCUSDT"))
        trader._symbol_reentry_block_until["BTCUSDT"] = time.time() - 1.0
        trader._expire_symbol_reentry_blocks()
        self.assertTrue(trader._symbol_reentry_allowed("BTCUSDT"))

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
        self.assertAlmostEqual(position.entry_price, (100.0 * 0.1 + 101.0 * 0.05) / 0.15)
        self.assertGreaterEqual(trader._sim_positions["BTCUSDT"].stop_price, 99.0)

    def test_false_breakout_detection_rejects_failed_short_breakdown(self) -> None:
        config = default_live_config()
        client = FakeClient()
        trader = BinanceAutoTrader(config, client)
        candles = _flat_candles(60)
        candles.append(Candle(datetime(2025, 1, 1, 5, 0), 100.0, 101.0, 95.0, 100.8, 1800.0))

        reason = trader._false_breakout_reason(Direction.SHORT, candles)

        self.assertIsNotNone(reason)
        self.assertIn("false_short_breakdown", reason or "")

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

    def test_dry_run_open_uses_live_open_time_for_management_guard(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            trading=replace(base.trading, min_managed_exit_bars=1),
            strategy=replace(base.strategy, stale_position_exit_enabled=True, stale_observation_bars=1, stale_force_exit_bars=2),
        )
        candles = [
            Candle(datetime(2025, 1, 1) + timedelta(minutes=30 * index), 100.0, 101.0, 99.0, 100.0, 1000.0)
            for index in range(80)
        ]
        client = SequenceClient(candles)
        trader = BinanceAutoTrader(config, client)
        signal = Signal(Direction.LONG, 1.0, "long_breakout score=0.90", 0.10, 0.10, max_holding_bars=100)

        trader._enter_position("BTCUSDT", signal, candles[10], "1")

        position = trader._sim_positions["BTCUSDT"]
        self.assertEqual(position.bars_held, 0)
        self.assertEqual(position.last_checked_time, candles[-2].timestamp)
        snapshot = trader.snapshot_account()
        self.assertFalse(trader._managed_exit_allowed(snapshot.positions["BTCUSDT"]))
        trader._manage_sim_positions()
        self.assertIn("BTCUSDT", trader._sim_positions)
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
        client = SequenceClient([entry, open_candle])
        trader = BinanceAutoTrader(config, client)
        signal = Signal(Direction.LONG, 1.0, "test", 0.05, 0.10)
        trader._enter_position("BTCUSDT", signal, entry, "0.1")
        client.candles = [entry, profit, pullback, open_candle]
        trader._manage_sim_positions()
        snapshot = trader.snapshot_account()
        self.assertNotIn("BTCUSDT", snapshot.positions)
        self.assertEqual(trader.stats.closed_trades, 1)
        self.assertEqual(trader.stats.winning_trades, 1)
        self.assertGreater(trader.stats.realized_pnl, 0.0)

    def test_profit_exit_requires_fee_and_slippage_buffer(self) -> None:
        config = default_live_config()
        trader = BinanceAutoTrader(config, FakeClient())
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            max_holding_bars=0,
            entry_time=datetime(2025, 1, 1),
            last_checked_time=datetime(2025, 1, 1),
            best_price=100.4,
        )
        candle = Candle(datetime(2025, 1, 1, 0, 1), 100.0, 100.4, 99.9, 100.1, 1000.0)

        reason = trader._profit_exit_reason(position, [candle], current_candle=candle)

        self.assertIsNone(reason)

    def test_prepare_symbol_skips_margin_type_blocked_by_open_orders(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False))
        client = MarginBlockedClient()
        trader = BinanceAutoTrader(config, client)

        trader._prepare_symbol("BNBUSDT")

        self.assertTrue(client.leverage_set)

    def test_prepare_symbol_skips_invalid_leverage_symbol(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False))
        client = InvalidLeverageClient()
        trader = BinanceAutoTrader(config, client)

        prepared = trader._prepare_symbol("VVVUSDT")

        self.assertFalse(prepared)
        self.assertIn("VVVUSDT", trader._unsupported_symbols)

    def test_live_exit_cleans_orders_before_and_after_market_close(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False))
        client = OrderCleanupClient()
        trader = BinanceAutoTrader(config, client)
        position = LivePosition(
            symbol="BTCUSDT",
            position_side="BOTH",
            direction=Direction.LONG,
            quantity=0.1,
            entry_price=100.0,
            mark_price=101.0,
            notional=10.1,
            unrealized_pnl=0.1,
            leverage=30,
            margin_type="ISOLATED",
            liquidation_price=None,
        )

        trader._exit_position("BTCUSDT", position, "test_exit")

        self.assertEqual(
            client.calls,
            [
                ("cancel_open", "BTCUSDT"),
                ("cancel_algo", "BTCUSDT"),
                ("market", "BTCUSDT", "SELL", "0.1", True),
                ("cancel_open", "BTCUSDT"),
                ("cancel_algo", "BTCUSDT"),
            ],
        )

    def test_observed_exchange_close_cleans_leftover_orders(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False))
        client = OrderCleanupClient()
        trader = BinanceAutoTrader(config, client)
        trader._known_active_symbols.add("BTCUSDT")
        trader._scale_in_counts["BTCUSDT"] = 1
        trader._last_scale_in_ts["BTCUSDT"] = 123.0

        trader._mark_observed_position_closures(set())

        self.assertEqual(client.calls, [("cancel_open", "BTCUSDT"), ("cancel_algo", "BTCUSDT")])
        self.assertNotIn("BTCUSDT", trader._scale_in_counts)
        self.assertNotIn("BTCUSDT", trader._last_scale_in_ts)

    def test_orphan_order_cleanup_cancels_symbol_without_position(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False))
        client = OrderCleanupClient()
        client.regular_open_orders = [
            {"symbol": "BTCUSDT"},
            {"symbol": "NOTINCONFIGUSDT"},
        ]
        client.algo_open_orders = [
            {"symbol": "ETHUSDT"},
            {"symbol": "BNBUSDT"},
        ]
        trader = BinanceAutoTrader(config, client)

        trader._cleanup_orphan_symbol_orders({"ETHUSDT"})

        self.assertEqual(
            client.calls,
            [
                ("cancel_open", "BNBUSDT"),
                ("cancel_algo", "BNBUSDT"),
                ("cancel_open", "BTCUSDT"),
                ("cancel_algo", "BTCUSDT"),
            ],
        )
        self.assertFalse(trader._symbol_reentry_allowed("BTCUSDT"))
        self.assertTrue(trader._symbol_reentry_allowed("ETHUSDT"))

    def test_live_new_entry_cleans_stale_orders_before_open(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, dry_run=False, use_protective_orders=False))
        client = OrderCleanupClient()
        trader = BinanceAutoTrader(config, client)
        candle = Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
        signal = Signal(Direction.LONG, 1.0, "test", 0.01, 0.02)

        trader._enter_position("BTCUSDT", signal, candle, "0.1")

        self.assertEqual(
            client.calls,
            [
                ("margin", "BTCUSDT", "CROSSED"),
                ("leverage", "BTCUSDT", 30),
                ("cancel_open", "BTCUSDT"),
                ("cancel_algo", "BTCUSDT"),
                ("market", "BTCUSDT", "BUY", "0.1", False),
            ],
        )

    def test_entry_rank_prefers_strong_momentum_and_volume(self) -> None:
        trader = BinanceAutoTrader(default_live_config(), FakeClient())
        start = datetime(2025, 1, 1)
        quiet = [
            Candle(start + timedelta(hours=index), 100.0, 101.0, 99.0, 100.0, 1000.0)
            for index in range(30)
        ]
        mild = quiet + [Candle(start + timedelta(hours=30), 100.0, 102.5, 99.8, 102.0, 1200.0)]
        strong = quiet + [Candle(start + timedelta(hours=30), 100.0, 121.0, 99.8, 120.0, 5000.0)]
        signal = Signal(Direction.LONG, 0.8, "test", 0.01, 0.02, risk_multiplier=0.8)

        mild_score = trader._entry_rank_metrics(signal, mild)[0]
        strong_score = trader._entry_rank_metrics(signal, strong)[0]

        self.assertGreater(strong_score, mild_score)

    def test_indicator_dead_cross_above_zero_creates_short_signal(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            strategy=replace(base.strategy, atr_period=5),
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
        candles = _upper_reversal_candles()[:71]
        trader = BinanceAutoTrader(config, SequenceClient(candles))
        signal = trader._indicator_reversal_signal(candles)
        self.assertEqual(signal.direction, Direction.SHORT)
        self.assertIn("indicator_short", signal.reason)
        self.assertGreater(signal.stop_loss_pct, 0.0)

    def test_stale_position_exits_when_observed_profit_returns(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            strategy=replace(base.strategy, stale_position_exit_enabled=True, stale_observation_bars=3, stale_force_exit_bars=6),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=103.0,
            max_holding_bars=18,
            entry_time=datetime(2025, 1, 1),
            last_checked_time=datetime(2025, 1, 1),
            best_price=100.0,
            bars_held=3,
            entry_reason="long_breakout score=0.90",
        )

        reason = trader._stale_position_exit_reason(position, 100.30)

        self.assertIsNotNone(reason)
        self.assertIn("stale_profit_exit", reason or "")

    def test_stale_position_force_exits_after_hard_limit(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            strategy=replace(base.strategy, stale_position_exit_enabled=True, stale_observation_bars=3, stale_force_exit_bars=6),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=103.0,
            max_holding_bars=18,
            entry_time=datetime(2025, 1, 1),
            last_checked_time=datetime(2025, 1, 1),
            best_price=100.0,
            bars_held=6,
            entry_reason="long_breakout_super_volume volume=4.0x",
        )

        reason = trader._stale_position_exit_reason(position, 99.70)

        self.assertIsNotNone(reason)
        self.assertIn("stale_force_exit", reason or "")

    def test_managed_exit_waits_for_minimum_holding_bars(self) -> None:
        base = default_live_config()
        config = replace(base, trading=replace(base.trading, min_managed_exit_bars=2))
        trader = BinanceAutoTrader(config, FakeClient())
        position = SimPosition(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=103.0,
            max_holding_bars=18,
            entry_time=datetime(2025, 1, 1),
            last_checked_time=datetime(2025, 1, 1),
            best_price=100.0,
            bars_held=1,
            entry_reason="long_breakout score=0.90",
        )

        self.assertFalse(trader._managed_exit_allowed(position))
        position.bars_held = 2
        self.assertTrue(trader._managed_exit_allowed(position))

    def test_profit_pullback_does_not_pause_new_entries(self) -> None:
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

    def test_live_global_risk_uses_principal_and_weekly_drawdown_limits(self) -> None:
        base = default_live_config()
        config = replace(
            base,
            risk=replace(
                base.risk,
                starting_capital_usdt=10_000.0,
                starting_capital_drawdown_stop_pct=0.20,
                weekly_profit_drawdown_stop_pct=0.15,
                max_daily_loss_pct=0.50,
            ),
        )
        trader = BinanceAutoTrader(config, FakeClient())
        trader._day_start_equity = 10_000.0
        trader._peak_equity = 12_000.0
        trader._week_start_equity = 10_500.0
        trader._week_peak_equity = 12_000.0

        old_peak_pullback = AccountSnapshot(
            equity=10_300.0,
            wallet_balance=10_300.0,
            available_balance=10_300.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        weekly_stop = AccountSnapshot(
            equity=10_200.0,
            wallet_balance=10_200.0,
            available_balance=10_200.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )
        principal_stop = AccountSnapshot(
            equity=8_000.0,
            wallet_balance=8_000.0,
            available_balance=8_000.0,
            initial_margin=0.0,
            maintenance_margin=0.0,
            total_unrealized_pnl=0.0,
            positions={},
        )

        self.assertTrue(trader._global_risk_allows_trading(old_peak_pullback))
        self.assertFalse(trader._global_risk_allows_trading(weekly_stop))
        trader._week_peak_equity = 10_000.0
        trader._week_start_equity = 9_800.0
        self.assertFalse(trader._global_risk_allows_trading(principal_stop))


class FakeClient:
    api_key = "key"
    api_secret = "secret"

    def klines(self, symbol: str, interval: str, limit: int = 200):
        return [Candle(datetime(2025, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)]

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")


class MarginBlockedClient(FakeClient):
    def __init__(self) -> None:
        self.leverage_set = False

    def set_margin_type(self, symbol: str, margin_type: str):
        raise BinanceApiError(
            400,
            "Margin type cannot be changed if there exists open orders.",
            {"code": -4047, "msg": "Margin type cannot be changed if there exists open orders."},
        )

    def set_leverage(self, symbol: str, leverage: int):
        self.leverage_set = True
        return {"symbol": symbol, "leverage": leverage}


class InvalidLeverageClient(FakeClient):
    def set_margin_type(self, symbol: str, margin_type: str):
        return {"symbol": symbol, "marginType": margin_type}

    def set_leverage(self, symbol: str, leverage: int):
        raise BinanceApiError(400, f"Leverage {leverage} is not valid", {"msg": f"Leverage {leverage} is not valid"})


class OrderCleanupClient(FakeClient):
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.regular_open_orders: list[dict] = []
        self.algo_open_orders: list[dict] = []

    def set_margin_type(self, symbol: str, margin_type: str):
        self.calls.append(("margin", symbol.upper(), margin_type.upper()))
        return {"symbol": symbol, "marginType": margin_type}

    def set_leverage(self, symbol: str, leverage: int):
        self.calls.append(("leverage", symbol.upper(), leverage))
        return {"symbol": symbol, "leverage": leverage}

    def cancel_all_open_orders(self, symbol: str):
        self.calls.append(("cancel_open", symbol.upper()))
        return {"ok": True}

    def cancel_all_algo_open_orders(self, symbol: str):
        self.calls.append(("cancel_algo", symbol.upper()))
        return {"ok": True}

    def open_orders(self, symbol: str | None = None):
        if symbol:
            return [order for order in self.regular_open_orders if order.get("symbol") == symbol.upper()]
        return self.regular_open_orders

    def open_algo_orders(self, symbol: str | None = None):
        if symbol:
            return [order for order in self.algo_open_orders if order.get("symbol") == symbol.upper()]
        return self.algo_open_orders

    def new_market_order(self, symbol: str, side: str, quantity: str, reduce_only: bool = False):
        self.calls.append(("market", symbol.upper(), side.upper(), quantity, reduce_only))
        return {"avgPrice": "100", "executedQty": quantity, "cumQuote": str(float(quantity) * 100.0)}


class RecordingBinanceClient(BinanceFuturesClient):
    def __init__(self) -> None:
        super().__init__(api_key="key", api_secret="secret")
        self.calls: list[tuple[str, str, dict]] = []

    def _signed_request(self, method: str, path: str, params=None):
        self.calls.append((method, path, dict(params or {})))
        return {"ok": True}


class SequenceClient(FakeClient):
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = candles

    def klines(self, symbol: str, interval: str, limit: int = 200):
        return self.candles[-limit:]


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


def _upper_reversal_candles() -> list[Candle]:
    candles: list[Candle] = []
    price = 90.0
    start = datetime(2025, 1, 1)
    for index in range(70):
        open_price = price
        close_price = price + 0.12
        candles.append(Candle(start + timedelta(minutes=index), open_price, close_price + 0.3, open_price - 0.3, close_price, 1000.0))
        price = close_price
    timestamp = candles[-1].timestamp
    for index in range(20):
        open_price = price
        close_price = price - 0.18
        candles.append(Candle(timestamp + timedelta(minutes=index + 1), open_price, open_price + 0.1, close_price - 0.25, close_price, 1200.0))
        price = close_price
    return candles


def _flat_candles(count: int) -> list[Candle]:
    candles: list[Candle] = []
    price = 100.0
    start = datetime(2025, 1, 1)
    for index in range(count):
        candles.append(Candle(start + timedelta(minutes=index * 5), price, price + 1.0, price - 1.0, price, 1000.0))
    return candles


if __name__ == "__main__":
    unittest.main()
