from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from datetime import timedelta

from crypto_scalper.config import StrategyConfig
from crypto_scalper.live_config import default_live_config
from crypto_scalper.live_execution_backtest import _mtf_adjust_candidate_for_fill
from crypto_scalper.live_execution_backtest import _build_point_in_time_universe
from crypto_scalper.live_execution_backtest import _cached_mtf_4h_rsi_candidate
from crypto_scalper.live_execution_backtest import _mtf_exit_reason_for_position
from crypto_scalper.live_execution_backtest import _initialize_mtf_position
from crypto_scalper.live_execution_backtest import _mtf_stop_exit_reason
from crypto_scalper.live_execution_backtest import _mtf_take_profit_action
from crypto_scalper.live_execution_backtest import _mtf_signal_config_hash
from crypto_scalper.live_execution_backtest import _update_mtf_profit_protection
from crypto_scalper.live_portfolio_backtest import _exit_order_type
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.live_trader import EntryCandidate
from crypto_scalper.models import Candle
from crypto_scalper.models import Direction
from crypto_scalper.models import Signal
from crypto_scalper.mtf_4h_rsi_regime import (
    MTF_REASON_TOKEN,
    Mtf4hRsiRegimePullbackStrategy,
    MtfRegimeResult,
    MtfSetupResult,
    MtfTriggerResult,
    available_candle_end,
    build_oi_features,
    closed_candles_for_decision,
    mtf_min_rank_score_for_direction,
    mtf_report_from_summary,
    oi_change_at,
)
from crypto_scalper.risk import BacktestExecutionConfig


class Mtf4hRsiRegimeTests(unittest.TestCase):
    def test_side_specific_rank_floor_keeps_short_independent(self) -> None:
        strategy = StrategyConfig(
            mtf_min_rank_score=3.0,
            mtf_long_min_rank_score=3.5,
            mtf_short_min_rank_score=3.2,
        )

        self.assertEqual(mtf_min_rank_score_for_direction(strategy, Direction.LONG), 3.5)
        self.assertEqual(mtf_min_rank_score_for_direction(strategy, Direction.SHORT), 3.2)

        global_floor = replace(strategy, mtf_min_rank_score=3.8)
        self.assertEqual(mtf_min_rank_score_for_direction(global_floor, Direction.LONG), 3.8)
        self.assertEqual(mtf_min_rank_score_for_direction(global_floor, Direction.SHORT), 3.8)

    def test_4h_candle_unclosed_is_not_available_for_regime(self) -> None:
        timestamps = [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 4, 0)]

        self.assertEqual(available_candle_end(timestamps, datetime(2026, 1, 1, 7, 59), "4h"), 1)
        self.assertEqual(available_candle_end(timestamps, datetime(2026, 1, 1, 8, 0), "4h"), 2)

    def test_feature_mapping_to_1m_has_no_future_fill(self) -> None:
        start = datetime(2026, 1, 1, 10, 0)
        candles = [_candle(start + timedelta(minutes=15 * i), 100 + i) for i in range(3)]
        timestamps = [item.timestamp for item in candles]

        self.assertEqual(closed_candles_for_decision(candles, timestamps, start + timedelta(minutes=29), "15m", 10)[-1].timestamp, start)
        self.assertEqual(closed_candles_for_decision(candles, timestamps, start + timedelta(minutes=30), "15m", 10)[-1].timestamp, start + timedelta(minutes=15))

    def test_15m_trigger_fills_next_1m_open_not_same_bar(self) -> None:
        signal_time = datetime(2026, 1, 1, 10, 15)
        signal_available = signal_time + timedelta(minutes=15)

        self.assertEqual(signal_available, datetime(2026, 1, 1, 10, 30))

    def test_point_in_time_warmup_only_applies_to_symbols_first_seen_after_data_start(self) -> None:
        start = datetime(2026, 1, 1)
        established = [_candle(start + timedelta(days=index), 100.0) for index in range(8)]
        newly_seen = [_candle(start + timedelta(days=index), 10.0) for index in range(5, 8)]
        config = _config()
        config = replace(
            config,
            risk=replace(
                config.risk,
                point_in_time_universe_enabled=True,
                point_in_time_universe_top_n=10,
                universe_lookback_days=1,
                new_symbol_warmup_days=20,
            ),
        )

        universe = _build_point_in_time_universe(
            config,
            {"ESTABLISHEDUSDT": established, "NEWUSDT": newly_seen},
        )

        self.assertIn("ESTABLISHEDUSDT", universe[(start + timedelta(days=1)).date()])
        self.assertNotIn("NEWUSDT", universe[(start + timedelta(days=6)).date()])

    def test_long_bias_conditions(self) -> None:
        strategy = _strategy(mtf_4h_require_macd_hist_turn=False, mtf_4h_max_distance_from_ema21_pct=1.0)
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [100 - i * 1.0 for i in range(24)] + [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "LONG_BIAS")

    def test_short_bias_conditions(self) -> None:
        strategy = _strategy(mtf_4h_require_macd_hist_turn=False, mtf_4h_max_distance_from_ema21_pct=1.0)
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [100 + i * 1.0 for i in range(24)] + [123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "SHORT_BIAS")

    def test_trend_pullback_mode_detects_closed_4h_uptrend(self) -> None:
        strategy = _strategy(
            mtf_regime_mode="trend_pullback",
            mtf_trend_long_min_ema_slope_atr=0.0,
            mtf_trend_long_rsi_max=100.0,
            mtf_trend_require_macd_support=False,
            mtf_trend_max_distance_from_ema21_pct=1.0,
        )
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [100.0 + index for index in range(40)])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "LONG_BIAS")
        self.assertGreater(regime.ema_slope_atr, 0.0)
        self.assertGreater(regime.ema_spread_atr, 0.0)

    def test_trend_pullback_mode_detects_closed_4h_downtrend(self) -> None:
        strategy = _strategy(
            mtf_regime_mode="trend_pullback",
            mtf_trend_short_max_ema_slope_atr=0.0,
            mtf_trend_short_rsi_min=0.0,
            mtf_trend_require_macd_support=False,
            mtf_trend_max_distance_from_ema21_pct=1.0,
        )
        candles = _trend_candles(datetime(2026, 1, 1), "4h", [140.0 - index for index in range(40)])

        regime = strategy.regime_4h(candles)

        self.assertEqual(regime.regime, "SHORT_BIAS")
        self.assertLess(regime.ema_slope_atr, 0.0)
        self.assertLess(regime.ema_spread_atr, 0.0)

    def test_long_sweep_reclaim_trigger(self) -> None:
        strategy = _strategy()
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.2)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=15), 99.8, 100.6, 99.4, 100.5, 1))

        trigger, reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "15m_long_sweep")

    def test_5m_volume_filter_rejects_low_volume_trigger(self) -> None:
        strategy = _strategy(
            mtf_trigger_timeframe="5m",
            mtf_trigger_volume_filter_enabled=True,
            mtf_trigger_min_volume_ratio=1.2,
            mtf_trigger_volume_period=5,
        )
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_tf(datetime(2026, 1, 1), 24, 5, 100.2, 100.0)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=5), 99.8, 100.6, 99.4, 100.5, 80.0))

        trigger, reason = strategy.trigger_timeframe(Direction.LONG, candles, setup, "5m")

        self.assertIsNone(trigger)
        self.assertEqual(reason, "5m_volume_too_low")

    def test_5m_volume_sweep_trigger(self) -> None:
        strategy = _strategy(
            mtf_trigger_timeframe="5m",
            mtf_trigger_volume_filter_enabled=True,
            mtf_trigger_min_volume_ratio=1.2,
            mtf_trigger_volume_period=5,
        )
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_tf(datetime(2026, 1, 1), 24, 5, 100.2, 100.0)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=5), 99.8, 100.6, 99.4, 100.5, 180.0))

        trigger, reason = strategy.trigger_timeframe(Direction.LONG, candles, setup, "5m")

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "5m_long_sweep")
        self.assertGreater(trigger.volume_ratio, 1.2)

    def test_short_sweep_reject_trigger(self) -> None:
        strategy = _strategy()
        setup = MtfSetupResult(Direction.SHORT, 98.0, 100.0, 100.0, 100.0, 98.0, 100.5, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 99.8)
        candles.append(Candle(candles[-1].timestamp + timedelta(minutes=15), 100.2, 100.7, 99.4, 99.5, 1))

        trigger, reason = strategy.trigger_15m(Direction.SHORT, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "sweep")
        self.assertEqual(reason, "15m_short_sweep")

    def test_long_structure_break_trigger(self) -> None:
        strategy = _strategy(mtf_15m_trigger_mode="structure_break")
        setup = MtfSetupResult(Direction.LONG, 100.0, 102.0, 100.0, 100.0, 99.5, 102.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.0, 100.4, 99.7, 100.1, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 100.2, 100.8, 99.8, 100.7, 1),
        ])

        trigger, _reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "structure_break")

    def test_short_structure_break_trigger(self) -> None:
        strategy = _strategy(mtf_15m_trigger_mode="structure_break")
        setup = MtfSetupResult(Direction.SHORT, 98.0, 100.0, 100.0, 100.0, 98.0, 100.5, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.1, 100.3, 99.7, 99.9, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 99.8, 100.0, 99.0, 99.1, 1),
        ])

        trigger, _reason = strategy.trigger_15m(Direction.SHORT, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "structure_break")

    def test_long_ema_reclaim_trigger_requires_closed_cross_and_quality(self) -> None:
        strategy = _strategy(
            mtf_15m_trigger_mode="ema_reclaim",
            mtf_reclaim_min_body_atr=0.20,
            mtf_reclaim_min_volume_ratio=0.0,
            mtf_reclaim_require_macd_improvement=False,
        )
        setup = MtfSetupResult(Direction.LONG, 98.0, 103.0, 100.0, 100.0, 98.0, 103.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.0, 100.2, 99.2, 99.4, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 99.4, 101.2, 99.3, 101.0, 1),
        ])

        trigger, reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "ema_reclaim")
        self.assertEqual(reason, "15m_long_ema_reclaim")

    def test_short_ema_reject_trigger_requires_closed_cross_and_quality(self) -> None:
        strategy = _strategy(
            mtf_15m_trigger_mode="ema_reclaim",
            mtf_reclaim_min_body_atr=0.20,
            mtf_reclaim_min_volume_ratio=0.0,
            mtf_reclaim_require_macd_improvement=False,
        )
        setup = MtfSetupResult(Direction.SHORT, 97.0, 102.0, 100.0, 100.0, 97.0, 102.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 100.0, 100.8, 99.8, 100.6, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 100.6, 100.7, 99.0, 99.2, 1),
        ])

        trigger, reason = strategy.trigger_15m(Direction.SHORT, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "ema_reclaim")
        self.assertEqual(reason, "15m_short_ema_reclaim")

    def test_structure_or_reclaim_preserves_structure_break_priority(self) -> None:
        strategy = _strategy(
            mtf_15m_trigger_mode="structure_or_reclaim",
            mtf_reclaim_min_body_atr=0.0,
            mtf_reclaim_min_volume_ratio=0.0,
            mtf_reclaim_require_macd_improvement=False,
        )
        setup = MtfSetupResult(Direction.LONG, 98.0, 103.0, 100.0, 100.0, 98.0, 103.0, 50.0)
        candles = _flat_15m(datetime(2026, 1, 1), 23, 100.0)
        candles.extend([
            Candle(candles[-1].timestamp + timedelta(minutes=15), 99.8, 100.0, 99.2, 99.4, 1),
            Candle(candles[-1].timestamp + timedelta(minutes=30), 99.4, 101.2, 99.3, 101.0, 1),
        ])

        trigger, _reason = strategy.trigger_15m(Direction.LONG, candles, setup)

        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.mode, "structure_break")

    def test_stop_too_wide_rejects_trade(self) -> None:
        config = _config(mtf_max_stop_pct=0.001, mtf_use_oi_filter=False, mtf_use_funding_filter=False)
        strategy = Mtf4hRsiRegimePullbackStrategy(config)
        strategy.regime = lambda *_args: MtfRegimeResult("LONG_BIAS", 42.0, "test")  # type: ignore[method-assign]
        strategy.confirm_1h = lambda *_args: (True, "ok", 55.0, 100.0)  # type: ignore[method-assign]
        strategy.setup_30m = lambda *_args: (MtfSetupResult(Direction.LONG, 98.0, 101.0, 100.0, 100.0, 98.0, 101.0, 50.0), "ok")  # type: ignore[method-assign]
        strategy.trigger_timeframe = lambda *_args: (MtfTriggerResult(Direction.LONG, "sweep", Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.5, 1), 1.0), "ok")  # type: ignore[method-assign]

        decision = strategy.build_signal("BTCUSDT", _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 40, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), _flat_15m(datetime(2026, 1, 1), 30, 100), 0.0, 0.0)

        self.assertIsNone(decision.signal)
        self.assertEqual(decision.reject_reason, "mtf_stop_too_wide")

    def test_fill_guard_preserves_structural_stop_and_cost_edge(self) -> None:
        config = _config(
            mtf_min_stop_pct=0.0045,
            mtf_max_stop_pct=0.0065,
            mtf_min_target_to_cost_ratio=6.0,
        )
        candle = Candle(datetime(2026, 1, 1), 100.0, 100.5, 99.5, 100.0, 1)
        candidate = EntryCandidate(
            "BTCUSDT",
            Signal(Direction.LONG, 0.62, MTF_REASON_TOKEN, 0.006, 0.012),
            candle,
            1.0,
            0.0,
            1.0,
            "mtf",
            metadata={
                "trigger_atr": 1.0,
                "trigger_close": 100.0,
                "structural_stop_price": 99.4,
                "take_profit_r": 2.0,
            },
        )
        stats: dict[str, int] = {}

        adjusted = _mtf_adjust_candidate_for_fill(
            config,
            candidate,
            100.0,
            BacktestExecutionConfig(),
            stats,
        )

        self.assertIsNotNone(adjusted)
        assert adjusted is not None
        self.assertAlmostEqual(adjusted.signal.stop_loss_pct, 0.006)
        self.assertGreater(adjusted.metadata["target_to_cost_ratio"], 6.0)
        rejected = _mtf_adjust_candidate_for_fill(
            config,
            candidate,
            100.2,
            BacktestExecutionConfig(),
            stats,
        )
        self.assertIsNone(rejected)
        self.assertEqual(stats["mtf_fill_stop_too_wide"], 1)

    def test_funding_filter_direction(self) -> None:
        strategy = _strategy(mtf_long_min_funding_rate=0.0, mtf_short_max_funding_rate=0.0)

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, -0.0001, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0003, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, -0.0003, 0.0, 0.0), "mtf_rejected_funding")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, 0.0001, 0.0, 0.0), "mtf_rejected_funding")

    def test_oi_filter(self) -> None:
        strategy = _strategy()

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, None, 0.0, 0.0, 0.0), "mtf_rejected_oi_missing")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.03, 0.0, 0.0, 0.0), "mtf_rejected_oi")

    def test_oi_feature_expires_instead_of_forward_filling_forever(self) -> None:
        rows = [
            {
                "timestamp": (datetime(2026, 1, 1) + timedelta(minutes=5 * index)).isoformat(),
                "sumOpenInterestValue": str(100.0 + index),
            }
            for index in range(7)
        ]
        oi = build_oi_features(rows)
        features = {"oi": oi, "oi_times": [item.available_time for item in oi]}

        self.assertIsNotNone(oi_change_at(features, datetime(2026, 1, 1, 0, 35), max_age_minutes=15))
        self.assertIsNone(oi_change_at(features, datetime(2026, 1, 1, 0, 51), max_age_minutes=15))

    def test_oi_change_uses_contract_quantity_not_price_sensitive_notional(self) -> None:
        rows = [
            {
                "timestamp": (datetime(2026, 1, 1) + timedelta(minutes=5 * index)).isoformat(),
                "sumOpenInterest": "100",
                "sumOpenInterestValue": str(10_000 + index * 1_000),
            }
            for index in range(7)
        ]

        features = build_oi_features(rows)

        self.assertEqual(features[-1].oi_chg_30m, 0.0)

    def test_oi_features_deduplicate_overlapping_download_files(self) -> None:
        rows = [
            {
                "timestamp": (datetime(2026, 1, 1) + timedelta(minutes=5 * index)).isoformat(),
                "sumOpenInterest": str(100 + index),
            }
            for index in range(7)
        ]
        rows.append(dict(rows[-1]))

        features = build_oi_features(rows)

        self.assertEqual(len(features), 7)
        self.assertAlmostEqual(features[-1].oi_chg_30m or 0.0, 0.06)

    def test_btc_opposite_filter(self) -> None:
        strategy = _strategy()

        self.assertEqual(strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0, 0.0, -0.02), "mtf_rejected_btc")
        self.assertEqual(strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, 0.0, 0.0, 0.02), "mtf_rejected_btc")

    def test_long_btc_4h_regime_floor_does_not_change_short_filter(self) -> None:
        strategy = _strategy(
            mtf_use_funding_filter=False,
            mtf_use_oi_filter=False,
            mtf_btc_4h_long_min_return_pct=0.004,
        )

        self.assertEqual(
            strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0, 0.0, 0.0039),
            "mtf_rejected_btc_4h_long_regime",
        )
        self.assertIsNone(
            strategy._futures_filter_reject_reason(Direction.LONG, 0.0, 0.0, 0.0, 0.004)
        )
        self.assertIsNone(
            strategy._futures_filter_reject_reason(Direction.SHORT, 0.0, 0.0, 0.0, 0.004)
        )

    def test_fail_fast_exit(self) -> None:
        config = _config()
        position = _position(entry_time=datetime(2026, 1, 1, 0, 0), mfe=0.1)
        candle = Candle(datetime(2026, 1, 1, 2, 0), 100, 100.2, 99.8, 100.0, 1)

        self.assertEqual(_mtf_exit_reason_for_position(config, position, candle, {}, {}), "mtf_fail_fast")

    def test_time_stop_exit(self) -> None:
        config = _config(mtf_fail_fast_min_r=0.0, mtf_max_holding_minutes=60)
        position = _position(entry_time=datetime(2026, 1, 1, 0, 0), mfe=100.0)
        candle = Candle(datetime(2026, 1, 1, 1, 1), 100, 100.2, 99.8, 100.0, 1)

        self.assertEqual(_mtf_exit_reason_for_position(config, position, candle, {}, {}), "mtf_time_stop")

    def test_one_hour_confirmation_exit_can_be_disabled(self) -> None:
        disabled = _config(
            mtf_exit_on_1h_confirm_lost=False,
            mtf_fail_fast_minutes=999,
            mtf_max_holding_minutes=999,
        )
        enabled = _config(
            mtf_exit_on_1h_confirm_lost=True,
            mtf_fail_fast_minutes=999,
            mtf_max_holding_minutes=999,
        )
        start = datetime(2026, 1, 1)
        one_h = [
            Candle(start + timedelta(hours=index), 120.0 - index, 120.2 - index, 118.8 - index, 119.0 - index, 1)
            for index in range(30)
        ]
        decision_time = one_h[-1].timestamp + timedelta(hours=1)
        position = _position(entry_time=decision_time - timedelta(minutes=30), mfe=0.0)
        candle = Candle(decision_time, 90.0, 90.2, 89.8, 90.0, 1)
        candles = {"1h": {"BTCUSDT": one_h}}
        timestamps = {"1h": {"BTCUSDT": [item.timestamp for item in one_h]}}

        self.assertIsNone(_mtf_exit_reason_for_position(disabled, position, candle, candles, timestamps))
        self.assertEqual(
            _mtf_exit_reason_for_position(enabled, position, candle, candles, timestamps),
            "mtf_1h_confirm_lost",
        )

    def test_thirty_minute_confirmation_exit_uses_only_closed_candles(self) -> None:
        config = _config(
            mtf_exit_on_30m_confirm_lost=True,
            mtf_30m_exit_confirm_bars=2,
            mtf_30m_exit_require_macd_adverse=False,
            mtf_exit_on_1h_confirm_lost=False,
            mtf_fail_fast_minutes=999,
            mtf_max_holding_minutes=999,
        )
        start = datetime(2026, 1, 1)
        closes = [100.0] * 25 + [98.0, 97.0]
        thirty_m = [
            Candle(
                start + timedelta(minutes=30 * index),
                close,
                close + 0.2,
                close - 0.2,
                close,
                1,
            )
            for index, close in enumerate(closes)
        ]
        timestamps = [item.timestamp for item in thirty_m]
        candles = {"30m": {"BTCUSDT": thirty_m}}
        timestamps_by_timeframe = {"30m": {"BTCUSDT": timestamps}}
        last_close_time = thirty_m[-1].timestamp + timedelta(minutes=30)
        position = _position(entry_time=last_close_time - timedelta(minutes=30), mfe=100.0)

        before_close = Candle(
            last_close_time - timedelta(minutes=1),
            97.0,
            97.2,
            96.8,
            97.0,
            1,
        )
        after_close = replace(before_close, timestamp=last_close_time)

        self.assertIsNone(
            _mtf_exit_reason_for_position(
                config,
                position,
                before_close,
                candles,
                timestamps_by_timeframe,
            )
        )
        self.assertEqual(
            _mtf_exit_reason_for_position(
                config,
                position,
                after_close,
                candles,
                timestamps_by_timeframe,
            ),
            "mtf_30m_confirm_lost",
        )

    def test_thirty_minute_macd_filter_requires_adverse_momentum(self) -> None:
        start = datetime(2026, 1, 1)
        closes = [100.0] * 30 + [90.0, 96.0]
        candles = [
            Candle(
                start + timedelta(minutes=30 * index),
                close,
                close + 0.2,
                close - 0.2,
                close,
                1,
            )
            for index, close in enumerate(closes)
        ]

        self.assertTrue(
            _strategy(
                mtf_30m_exit_confirm_bars=2,
                mtf_30m_exit_require_macd_adverse=False,
            ).thirty_minute_confirm_lost(Direction.LONG, candles)
        )
        self.assertFalse(
            _strategy(
                mtf_30m_exit_confirm_bars=2,
                mtf_30m_exit_require_macd_adverse=True,
            ).thirty_minute_confirm_lost(Direction.LONG, candles)
        )

    def test_profit_protection_uses_previous_mfe_and_full_cost_buffer(self) -> None:
        config = _config(
            mtf_profit_protection_enabled=True,
            mtf_move_stop_to_breakeven_r=1.0,
            mtf_trailing_mode="none",
        )
        position = _position(entry_time=datetime(2026, 1, 1), mfe=1.5)
        position.entry_fee = 0.05
        position.entry_slippage_cost = 0.02
        position.initial_stop_price = 99.0
        candle = Candle(datetime(2026, 1, 1, 1), 101.0, 102.0, 100.5, 101.5, 1)
        costs = BacktestExecutionConfig(taker_fee_rate=0.0005, stop_slippage_bps=5.0)

        _update_mtf_profit_protection(config, position, 0.99, costs, candle, {}, {})
        self.assertEqual(position.stop_price, 99.0)

        _update_mtf_profit_protection(config, position, 1.0, costs, candle, {}, {})
        self.assertAlmostEqual(position.stop_price, 100.17, places=6)
        self.assertEqual(_mtf_stop_exit_reason(position), "mtf_breakeven_stop")
        self.assertEqual(_exit_order_type("mtf_breakeven_stop"), "stop_market")

    def test_profit_giveback_stop_only_uses_completed_path(self) -> None:
        config = _config(
            mtf_profit_protection_enabled=True,
            mtf_move_stop_to_breakeven_r=1.0,
            mtf_trailing_mode="giveback",
            mtf_trailing_start_r=1.5,
            mtf_profit_giveback_r=0.75,
        )
        position = _position(entry_time=datetime(2026, 1, 1), mfe=2.0)
        position.initial_stop_price = 99.0
        candle = Candle(datetime(2026, 1, 1, 1), 102.0, 103.0, 101.0, 102.0, 1)

        _update_mtf_profit_protection(
            config,
            position,
            2.0,
            BacktestExecutionConfig(taker_fee_rate=0.0, stop_slippage_bps=0.0),
            candle,
            {},
            {},
        )

        self.assertAlmostEqual(position.stop_price, 101.25)
        self.assertEqual(_mtf_stop_exit_reason(position), "mtf_giveback_stop")

    def test_candidate_cache_hash_ignores_exit_parameters(self) -> None:
        strategy = default_live_config().strategy
        changed_exit = replace(
            strategy,
            mtf_profit_protection_enabled=True,
            mtf_move_stop_to_breakeven_r=0.8,
            mtf_exit_on_30m_confirm_lost=True,
            mtf_30m_exit_confirm_bars=3,
            mtf_30m_exit_require_macd_adverse=False,
            mtf_exit_on_1h_confirm_lost=False,
            mtf_exit_mode="partial_runner",
            mtf_partial_take_profit_r=1.5,
            mtf_partial_take_profit_fraction=0.3,
        )
        changed_signal = replace(strategy, mtf_max_stop_pct=0.015)

        self.assertEqual(_mtf_signal_config_hash(strategy), _mtf_signal_config_hash(changed_exit))
        self.assertNotEqual(_mtf_signal_config_hash(strategy), _mtf_signal_config_hash(changed_signal))

    def test_candidate_cache_hash_ignores_portfolio_scheduling(self) -> None:
        strategy = default_live_config().strategy
        changed = replace(
            strategy,
            mtf_max_open_positions=3,
            mtf_max_daily_trades=5,
            mtf_symbol_cooldown_hours=2,
        )

        self.assertEqual(_mtf_signal_config_hash(strategy), _mtf_signal_config_hash(changed))

    def test_candidate_cache_hash_is_memoized(self) -> None:
        strategy = default_live_config().strategy
        _mtf_signal_config_hash.cache_clear()

        first = _mtf_signal_config_hash(strategy)
        second = _mtf_signal_config_hash(strategy)

        self.assertEqual(first, second)
        self.assertEqual(_mtf_signal_config_hash.cache_info().hits, 1)

    def test_partial_runner_uses_separate_partial_target_and_disables_second_tp(self) -> None:
        config = _config(
            mtf_exit_mode="partial_runner",
            mtf_partial_take_profit_r=1.5,
            mtf_partial_take_profit_fraction=0.3,
        )
        start = datetime(2026, 1, 1)
        candidate = EntryCandidate(
            "BTCUSDT",
            Signal(Direction.LONG, 0.62, MTF_REASON_TOKEN, 0.01, 0.02),
            _candle(start, 100.0),
            1.0,
            0.0,
            1.0,
            "mtf",
            metadata={"structural_stop_price": 99.0, "take_profit_r": 2.0},
        )
        position = _position(entry_time=start, mfe=0.0)

        _initialize_mtf_position(config, position, candidate)

        self.assertAlmostEqual(position.take_profit_price, 101.5)
        self.assertEqual(_mtf_take_profit_action(config, position), "partial")
        position.strategy_metadata["mtf_partial_take_profit_done"] = True
        self.assertEqual(_mtf_take_profit_action(config, position), "disabled")

    def test_runner_mode_disables_fixed_take_profit(self) -> None:
        position = _position(entry_time=datetime(2026, 1, 1), mfe=0.0)

        self.assertEqual(_mtf_take_profit_action(_config(mtf_exit_mode="runner"), position), "disabled")

    def test_cached_candidate_returns_before_rebuilding_other_timeframes(self) -> None:
        config = _config(mtf_4h_rsi_regime_enabled=True)
        start = datetime(2026, 1, 1)
        candidate = EntryCandidate(
            "BTCUSDT",
            Signal(Direction.LONG, 0.5, MTF_REASON_TOKEN, 0.01, 0.02),
            _candle(start, 100.0),
            1.0,
            0.0,
            1.0,
            "mtf",
        )
        cache = {
            (
                _mtf_signal_config_hash(config.strategy),
                "BTCUSDT",
                "15m",
                start,
            ): candidate
        }

        actual = _cached_mtf_4h_rsi_candidate(
            None,  # type: ignore[arg-type]
            config,
            "BTCUSDT",
            start + timedelta(minutes=15),
            {},
            {"15m": {"BTCUSDT": [start]}},
            {},
            {},
            {},
            {},
            cache,
            {},
        )

        self.assertIs(actual, candidate)

    def test_report_contains_mtf_dimensions(self) -> None:
        trade = _trade("LONG", "long_mtf_4h_rsi_regime_pullback regime=LONG_BIAS regime_tf=2h rsi_bucket=rsi_38_45 trigger=sweep trigger_tf=5m vol=vol_1p5_2x funding=funding_0_2bps oi=oi_flat_or_down btc=btc_neutral")
        report = mtf_report_from_summary({"trades": [trade]}, {"mtf_stop_too_wide": 2})

        self.assertIn("by_side", report)
        self.assertIn("by_trigger_mode", report)
        self.assertIn("by_regime_timeframe", report)
        self.assertIn("by_trigger_timeframe", report)
        self.assertIn("by_trigger_volume_bucket", report)
        self.assertIn("by_4h_regime", report)
        self.assertEqual(report["rejected_stop_too_wide_count"], 2)

    def test_oi_feature_available_after_bucket_end(self) -> None:
        start = datetime(2026, 1, 1)
        features = build_oi_features([
            {"timestamp": (start + timedelta(minutes=5 * index)).isoformat(), "sumOpenInterestValue": str(100 + index)}
            for index in range(7)
        ])

        self.assertEqual(features[0].available_time, start + timedelta(minutes=5))
        self.assertAlmostEqual(features[6].oi_chg_30m, 106 / 100 - 1.0)


def _config(**kwargs: object):
    base = default_live_config()
    strategy = replace(base.strategy, **kwargs)
    return replace(base, strategy=strategy)


def _strategy(**kwargs: object) -> Mtf4hRsiRegimePullbackStrategy:
    return Mtf4hRsiRegimePullbackStrategy(_config(**kwargs))


def _candle(timestamp: datetime, price: float) -> Candle:
    return Candle(timestamp, price, price + 0.5, price - 0.5, price, 1)


def _trend_candles(start: datetime, timeframe: str, closes: list[float]) -> list[Candle]:
    minutes = 240 if timeframe == "4h" else 15
    candles = []
    for index, close in enumerate(closes):
        timestamp = start + timedelta(minutes=minutes * index)
        open_price = closes[index - 1] if index else close
        high = max(open_price, close) + 0.4
        low = min(open_price, close) - 0.4
        candles.append(Candle(timestamp, open_price, high, low, close, 1))
    return candles


def _flat_15m(start: datetime, count: int, price: float) -> list[Candle]:
    return [Candle(start + timedelta(minutes=15 * index), price, price + 0.5, price - 0.5, price, 1) for index in range(count)]


def _flat_tf(start: datetime, count: int, minutes: int, price: float, volume: float) -> list[Candle]:
    return [
        Candle(start + timedelta(minutes=minutes * index), price, price + 0.5, price - 0.5, price, volume)
        for index in range(count)
    ]


def _position(entry_time: datetime, mfe: float) -> PortfolioPosition:
    return PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        99.0,
        102.0,
        0.0,
        0,
        24,
        100.0,
        entry_reason=MTF_REASON_TOKEN,
        entry_time=entry_time,
        raw_entry_price=100.0,
        mfe=mfe,
    )


def _trade(side: str, reason: str) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "strategy_bucket": "mtf_4h_rsi_regime_pullback",
        "side": side,
        "direction": side,
        "entry_time": "2026-05-20T00:00:00",
        "entry_reason": reason,
        "net_pnl": 1.0,
        "gross_pnl": 1.2,
        "fee": 0.1,
        "slippage_cost": 0.1,
        "funding": 0.0,
        "mfe": 2.0,
        "mae": -0.5,
        "hold_minutes": 30.0,
    }


if __name__ == "__main__":
    unittest.main()
