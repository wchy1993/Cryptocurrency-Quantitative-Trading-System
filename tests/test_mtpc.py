from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.indicators import ema
from crypto_scalper.live_config import default_live_config, load_live_config, write_live_config
from crypto_scalper.live_execution_backtest import (
    _cmipr_initial_quantity_within_campaign_budget,
    _candidate_signal_times,
    _manage_mtpc_position_1m,
    _merge_candidate_sleeves,
    _mtpc_adjust_candidate_for_fill,
    _mtpc_total_open_risk_allows_entry,
    _mtpc_trade_risk_budget,
)
from crypto_scalper.live_portfolio_backtest import PortfolioPosition
from crypto_scalper.live_trader import EntryCandidate
from crypto_scalper.models import Candle, Direction, Signal
from crypto_scalper.mtpc import (
    MTPC_REASON_TOKEN,
    MtpcEngine,
    MtpcImpulseSnapshot,
    MtpcRegime,
    MtpcState,
)
from crypto_scalper.mtpc_optimize import configs_for_stage
from crypto_scalper.risk import (
    BacktestExecutionConfig,
    BacktestExecutionStats,
    market_entry_fill,
    market_exit_fill,
)


def _candles(start: datetime, count: int, minutes: int, slope: float = 0.001) -> list[Candle]:
    rows = []
    price = 100.0
    for index in range(count):
        open_price = price
        close = open_price * (1.0 + slope)
        rows.append(
            Candle(
                start + timedelta(minutes=minutes * index),
                open_price,
                max(open_price, close) * 1.002,
                min(open_price, close) * 0.998,
                close,
                1000.0 + index,
            )
        )
        price = close
    return rows


def _config():
    base = default_live_config()
    symbols = ("BTCUSDT", "ETHUSDT")
    mtpc = replace(base.mtpc, enabled=True, enabled_symbols=symbols)
    return replace(
        base,
        trading=replace(base.trading, symbols=symbols, entry_symbols=symbols, leverage=5, max_open_positions=1),
        risk=replace(base.risk, risk_per_trade_pct=mtpc.risk_control.trade_risk_pct),
        mtpc=mtpc,
    )


def _engine(config=None) -> MtpcEngine:
    config = config or _config()
    start = datetime(2025, 1, 1)
    candles = {
        timeframe: {
            symbol: _candles(start, count, minutes)
            for symbol in config.trading.symbols
        }
        for timeframe, count, minutes in (
            ("5m", 800, 5),
            ("15m", 500, 15),
            ("1h", 200, 60),
            ("4h", 100, 240),
        )
    }
    return MtpcEngine(config, candles)


def test_mtpc_config_round_trip(tmp_path) -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            combine_with_mtf=True,
            regime=replace(
                base.mtpc.regime,
                allow_neutral_symbol_long=True,
                allow_neutral_symbol_short=True,
            ),
        ),
    )
    path = tmp_path / "mtpc.json"
    write_live_config(path, config)
    loaded = load_live_config(path)
    assert loaded.mtpc.enabled
    assert loaded.mtpc.combine_with_mtf
    assert loaded.mtpc.enabled_symbols == config.mtpc.enabled_symbols
    assert loaded.mtpc.pullback.confirmation_timeframe == "5m"
    assert loaded.mtpc.exit.r_basis == "initial_leg"
    assert loaded.mtpc.regime.allow_neutral_symbol_long
    assert loaded.mtpc.regime.allow_neutral_symbol_short
    assert loaded.mtpc.pullback.trend_ema_lookback_bars == 12
    assert loaded.mtpc.pullback.trend_ema_min_reclaim_extension_atr == 0.0


def test_combined_sleeves_keep_primary_priority_and_dedupe_symbol() -> None:
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.5, 1000.0)
    primary_btc = EntryCandidate(
        "BTCUSDT",
        Signal(Direction.LONG, 0.7, "primary", 0.01, 0.02),
        candle,
        4.0,
        1.0,
        1.0,
        "primary",
    )
    secondary_btc = replace(
        primary_btc,
        signal=replace(primary_btc.signal, reason=MTPC_REASON_TOKEN),
        rank_score=99.0,
        filter_reason="secondary",
    )
    secondary_eth = replace(secondary_btc, symbol="ETHUSDT")

    merged = _merge_candidate_sleeves(
        [primary_btc],
        [secondary_btc, secondary_eth],
    )

    assert merged == [primary_btc, secondary_eth]


def test_mtpc_neutral_regime_routes_each_symbol_to_its_ranked_direction() -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            allow_long=True,
            allow_short=True,
            regime=replace(
                base.mtpc.regime,
                allow_neutral_symbol_long=True,
                allow_neutral_symbol_short=True,
            ),
        ),
    )
    engine = _engine(config)
    decision_time = datetime(2025, 1, 3)
    engine._update_regime = lambda _: MtpcRegime.NEUTRAL
    engine._rankings = lambda _: {
        "BTCUSDT": {"percentile": 0.8},
        "ETHUSDT": {"percentile": 0.2},
    }
    observed = []

    def scan_symbol(symbol, timestamp, rank, direction):
        observed.append((symbol, direction))
        return None

    engine._scan_symbol = scan_symbol

    assert engine.scan(decision_time, set()) == []
    assert observed == [
        ("BTCUSDT", Direction.LONG),
        ("ETHUSDT", Direction.SHORT),
    ]


def test_mtpc_neutral_regime_is_blocked_by_default() -> None:
    engine = _engine()
    decision_time = datetime(2025, 1, 3)
    engine._update_regime = lambda _: MtpcRegime.NEUTRAL
    engine._rankings = lambda _: {
        symbol: {"percentile": 0.8} for symbol in engine.trade_symbols
    }
    engine._scan_symbol = lambda *_: (_ for _ in ()).throw(AssertionError("must not scan"))

    assert engine.scan(decision_time, set()) == []
    assert engine.reject_reasons["regime_neutral"] == 1


def test_mtpc_trend_maturity_and_prior_move_bounds_are_configurable(tmp_path) -> None:
    path = tmp_path / "mtpc.json"
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            regime=replace(
                base.mtpc.regime,
                max_4h_fast_slope_atr=0.8,
                max_1h_fast_slope_atr=0.7,
                max_1h_ema_gap_atr=1.2,
            ),
            impulse=replace(base.mtpc.impulse, min_prior_move_atr=0.0),
        ),
    )
    write_live_config(path, config)
    loaded = load_live_config(path)

    assert loaded.mtpc.regime.max_4h_fast_slope_atr == 0.8
    assert loaded.mtpc.regime.max_1h_fast_slope_atr == 0.7
    assert loaded.mtpc.regime.max_1h_ema_gap_atr == 1.2
    assert loaded.mtpc.impulse.min_prior_move_atr == 0.0


def test_mtpc_observation_universe_is_separate_from_trade_universe() -> None:
    base = _config()
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    config = replace(
        base,
        trading=replace(base.trading, symbols=symbols, entry_symbols=("BTCUSDT", "ETHUSDT")),
        mtpc=replace(base.mtpc, enabled_symbols=("BTCUSDT", "ETHUSDT")),
    )
    engine = _engine(config)

    assert engine.symbols == symbols
    assert engine.trade_symbols == ("BTCUSDT", "ETHUSDT")
    assert set(engine.runtime) == {"BTCUSDT", "ETHUSDT"}


def test_mtpc_high_timeframe_uses_only_closed_candles() -> None:
    engine = _engine()
    start = datetime(2025, 1, 1)
    at_close = engine._closed("4h", "BTCUSDT", start + timedelta(hours=8), 10)
    before_close = engine._closed("4h", "BTCUSDT", start + timedelta(hours=7, minutes=59), 10)
    assert [row.timestamp for row in at_close] == [start, start + timedelta(hours=4)]
    assert [row.timestamp for row in before_close] == [start]


def test_mtpc_pending_cancel_enters_cooldown() -> None:
    engine = _engine()
    runtime = engine.runtime["BTCUSDT"]
    runtime.state = MtpcState.ENTRY_READY
    runtime.setup_id = "setup:test"
    timestamp = datetime(2026, 1, 1)
    engine.mark_order_pending("BTCUSDT", timestamp)
    engine.mark_cancelled("BTCUSDT", timestamp + timedelta(minutes=1), "missed_next_1m_open")
    assert runtime.state == MtpcState.COOLDOWN
    assert runtime.setup_id is None


def test_mtpc_setup_and_entry_cancel_use_independent_cooldowns() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            risk_control=replace(
                config.mtpc.risk_control,
                symbol_cooldown_hours=12,
                setup_invalidation_cooldown_minutes=15,
                entry_cancel_cooldown_minutes=30,
            ),
        ),
    )
    engine = _engine(config)
    timestamp = datetime(2026, 1, 1)

    engine.runtime["BTCUSDT"].setup_id = "setup:invalid"
    engine._invalidate("BTCUSDT", timestamp, "impulse_expired")
    assert engine.runtime["BTCUSDT"].cooldown_until == timestamp + timedelta(minutes=15)

    engine.runtime["BTCUSDT"].state = MtpcState.ENTRY_READY
    engine.runtime["BTCUSDT"].setup_id = "setup:cancel"
    engine.mark_cancelled("BTCUSDT", timestamp + timedelta(hours=1), "fill_revalidation")
    assert engine.runtime["BTCUSDT"].cooldown_until == timestamp + timedelta(hours=1, minutes=30)


def test_mtpc_stage6_changes_only_cost_guard_from_frozen_exit() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 6)

    assert [name for name, _ in rows] == [
        "cost_guard_4.0x",
        "cost_guard_3.5x",
        "cost_guard_3.0x",
    ]
    assert [row.mtpc.pullback.min_target_to_cost_ratio for _, row in rows] == [4.0, 3.5, 3.0]
    for _, row in rows:
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert not row.mtpc.exit.runner_enabled
        assert row.mtpc.risk_control.setup_invalidation_cooldown_minutes == 720
        assert row.mtpc.risk_control.entry_cancel_cooldown_minutes == 720


def test_mtpc_stage7_changes_only_trade_risk_from_selected_cost_guard() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 7)

    assert [row.mtpc.risk_control.trade_risk_pct for _, row in rows] == [0.01, 0.0125, 0.015, 0.02]
    for _, row in rows:
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert not row.mtpc.exit.runner_enabled
        assert row.mtpc.risk_control.max_trade_risk_pct == 0.02
        assert row.mtpc.risk_control.max_total_open_risk_pct == 0.02


def test_mtpc_stage8_freezes_selected_validation_parameters() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    [(name, selected)] = configs_for_stage(config, 8)

    assert name == "selected_1_5pct_diagnostic"
    assert selected.mtpc.pullback.min_target_to_cost_ratio == 3.5
    assert selected.mtpc.exit.take_profit_1_r == 1.5
    assert selected.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage9_changes_only_pullback_volume_ceiling() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 9)

    assert [row.mtpc.pullback.max_volume_to_impulse for _, row in rows] == [1.0, 1.05, 1.10, 1.20]
    for _, row in rows:
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage10_changes_only_ranking_floor() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 10)

    assert [row.mtpc.ranking.min_percentile for _, row in rows] == [0.55, 0.50, 0.45, 0.40]
    for _, row in rows:
        assert row.mtpc.ranking.max_percentile == 0.95
        assert row.mtpc.pullback.max_volume_to_impulse == 1.0
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage11_changes_only_ranking_ceiling() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 11)

    assert [row.mtpc.ranking.max_percentile for _, row in rows] == [0.95, 0.98, 1.0]
    for _, row in rows:
        assert row.mtpc.ranking.min_percentile == 0.55
        assert row.mtpc.pullback.max_volume_to_impulse == 1.0
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage12_scales_position_and_total_risk_limits_together() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 12)

    assert [row.trading.max_open_positions for _, row in rows] == [1, 2, 3]
    assert [row.mtpc.risk_control.max_total_open_risk_pct for _, row in rows] == [0.015, 0.03, 0.045]
    for _, row in rows:
        assert row.mtpc.risk_control.max_open_positions == row.trading.max_open_positions
        assert row.mtpc.risk_control.max_same_direction_positions == row.trading.max_open_positions
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage13_changes_only_trend_maturity_cap() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 13)

    assert [row.mtpc.regime.max_1h_ema_gap_atr for _, row in rows] == [1.2, 1.5, 1.8, 2.4]
    for _, row in rows:
        assert row.trading.max_open_positions == 1
        assert row.mtpc.risk_control.max_open_positions == 1
        assert row.mtpc.risk_control.max_total_open_risk_pct == 0.015
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage14_compares_closed_pullback_timeframes() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 14)

    assert [row.mtpc.pullback.pullback_timeframe for _, row in rows] == ["15m", "5m", "5m"]
    assert [row.mtpc.pullback.max_bars_after_impulse for _, row in rows] == [10, 18, 30]
    for _, row in rows:
        assert row.mtpc.impulse.timeframe == "15m"
        assert row.mtpc.pullback.confirmation_timeframe == "5m"
        assert row.mtpc.regime.max_1h_ema_gap_atr == 1.5
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_stage15_isolates_five_minute_confirmation_quality() -> None:
    config = load_live_config("config.mtpc.stage8-conditional-only-funded.json")
    rows = configs_for_stage(config, 15)

    assert [name for name, _ in rows] == [
        "five_minute_current",
        "five_minute_macd_improving",
        "five_minute_break_previous_high",
        "five_minute_close_position_0_60",
        "five_minute_volume_0_80",
    ]
    for _, row in rows:
        assert row.mtpc.pullback.pullback_timeframe == "5m"
        assert row.mtpc.pullback.max_bars_after_impulse == 18
        assert row.mtpc.regime.max_1h_ema_gap_atr == 1.5
        assert row.mtpc.pullback.min_target_to_cost_ratio == 3.5
        assert row.mtpc.exit.take_profit_1_r == 1.5
        assert row.mtpc.risk_control.trade_risk_pct == 0.015


def test_mtpc_signal_availability_uses_closed_5m_confirmation() -> None:
    config = _config()
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.5, 100.8, 1000.0)
    signal = Signal(Direction.LONG, 0.7, MTPC_REASON_TOKEN, 0.02, 0.02, 1.0)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.7,
        0.01,
        1.2,
        "ok",
        {"trigger_timeframe": "5m"},
    )
    signal_time, available = _candidate_signal_times(config, candidate, candle.timestamp)
    assert signal_time == candle.timestamp
    assert available == candle.timestamp + timedelta(minutes=5)


def test_mtpc_direct_impulse_availability_uses_closed_15m_candle() -> None:
    config = _config()
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.5, 100.8, 1000.0)
    signal = Signal(Direction.LONG, 0.7, MTPC_REASON_TOKEN, 0.02, 0.03, 1.0)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.7,
        0.01,
        1.2,
        "ok",
        {"trigger_timeframe": "15m"},
    )

    signal_time, available = _candidate_signal_times(config, candidate, candle.timestamp)

    assert signal_time == candle.timestamp
    assert available == candle.timestamp + timedelta(minutes=15)


def test_mtpc_first_post_impulse_candle_can_remain_pullback_trough() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            pullback=replace(
                config.mtpc.pullback,
                max_depth_atr=2.0,
                max_retrace_fraction=0.90,
                max_volume_to_impulse=1.0,
                ema_proximity_atr=5.0,
            ),
        ),
    )
    engine = _engine(config)
    start = datetime(2026, 1, 1)
    candles = _candles(start, 40, 15, slope=0.0)
    impulse_candle = Candle(start + timedelta(minutes=600), 107.0, 110.0, 106.0, 109.0, 1000.0)
    first_pullback = Candle(start + timedelta(minutes=615), 109.0, 109.2, 107.5, 108.0, 500.0)
    recovery = Candle(start + timedelta(minutes=630), 108.0, 109.4, 108.0, 109.1, 600.0)
    candles.extend((impulse_candle, first_pullback, recovery))
    impulse = MtpcImpulseSnapshot(
        candle=impulse_candle,
        breakout_level=105.0,
        impulse_high=110.0,
        impulse_low=106.0,
        atr_value=2.0,
        breakout_distance_atr=2.0,
        body_atr=1.0,
        close_position=0.75,
        upper_wick_ratio=0.25,
        volume_ratio=2.0,
        prior_move_atr=0.5,
        ema21_extension_atr=1.0,
        quality_score=0.8,
    )

    pullback, reason = engine._detect_pullback(candles, impulse)

    assert reason == ""
    assert pullback is not None
    assert pullback.candle.timestamp == first_pullback.timestamp
    assert pullback.bars_after_impulse == 1


def test_mtpc_five_minute_pullback_ignores_impulse_sub_bars() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            pullback=replace(
                config.mtpc.pullback,
                pullback_timeframe="5m",
                min_depth_atr=0.1,
                max_depth_atr=3.0,
                max_retrace_fraction=2.0,
                max_volume_to_impulse=2.0,
                ema_proximity_atr=100.0,
            ),
        ),
    )
    engine = _engine(config)
    start = datetime(2026, 1, 1)
    impulse_candle = Candle(start, 100.0, 110.0, 99.0, 109.0, 1000.0)
    impulse = MtpcImpulseSnapshot(
        candle=impulse_candle,
        breakout_level=100.0,
        impulse_high=110.0,
        impulse_low=99.0,
        atr_value=2.0,
        breakout_distance_atr=1.0,
        body_atr=1.0,
        close_position=0.8,
        upper_wick_ratio=0.2,
        volume_ratio=1.5,
        prior_move_atr=0.5,
        ema21_extension_atr=0.5,
        quality_score=0.8,
    )
    candles = _candles(start - timedelta(hours=3), 36, 5, slope=0.0)
    candles.extend(
        (
            Candle(start + timedelta(minutes=5), 109.0, 109.5, 101.0, 102.0, 400.0),
            Candle(start + timedelta(minutes=10), 102.0, 109.0, 101.5, 108.5, 400.0),
            Candle(start + timedelta(minutes=15), 108.5, 109.0, 107.0, 108.0, 400.0),
        )
    )

    pullback, reason = engine._detect_pullback(candles, impulse)

    assert reason == ""
    assert pullback is not None
    assert pullback.candle.timestamp == start + timedelta(minutes=15)


def test_mtpc_pullback_wait_reason_is_diagnostic() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            pullback=replace(config.mtpc.pullback, min_depth_atr=2.0),
        ),
    )
    engine = _engine(config)
    start = datetime(2026, 1, 1)
    impulse_candle = Candle(start, 100.0, 102.0, 99.5, 101.5, 1000.0)
    impulse = MtpcImpulseSnapshot(
        candle=impulse_candle,
        breakout_level=100.0,
        impulse_high=102.0,
        impulse_low=99.5,
        atr_value=2.0,
        breakout_distance_atr=0.75,
        body_atr=0.75,
        close_position=0.8,
        upper_wick_ratio=0.2,
        volume_ratio=1.5,
        prior_move_atr=0.5,
        ema21_extension_atr=0.5,
        quality_score=0.8,
    )
    candles = [
        impulse_candle,
        Candle(start + timedelta(minutes=15), 101.5, 101.8, 101.2, 101.4, 500.0),
    ]

    pullback, reason = engine._detect_pullback(candles, impulse)

    assert pullback is None
    assert reason == "pullback_wait_too_shallow"


def test_mtpc_scan_uses_configured_pullback_timeframe() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            ranking=replace(config.mtpc.ranking, enabled=False),
            pullback=replace(config.mtpc.pullback, pullback_timeframe="5m"),
        ),
    )
    engine = _engine(config)
    symbol = "BTCUSDT"
    decision_time = datetime(2025, 1, 3)
    impulse_candles = engine._closed("15m", symbol, decision_time, 80)
    impulse_candle = impulse_candles[-2]
    engine.runtime[symbol].impulse = MtpcImpulseSnapshot(
        candle=impulse_candle,
        breakout_level=impulse_candle.low,
        impulse_high=impulse_candle.high,
        impulse_low=0.0,
        atr_value=1.0,
        breakout_distance_atr=0.5,
        body_atr=0.5,
        close_position=0.8,
        upper_wick_ratio=0.2,
        volume_ratio=1.5,
        prior_move_atr=0.5,
        ema21_extension_atr=0.5,
        quality_score=0.8,
    )
    engine.runtime[symbol].event_id = "event:test-5m-pullback"
    engine.runtime[symbol].expire_time = decision_time + timedelta(hours=1)
    observed_intervals = []

    engine._symbol_trend = lambda *_: (True, {})

    def detect(candles, impulse, direction=Direction.LONG):
        observed_intervals.append(candles[-1].timestamp - candles[-2].timestamp)
        return None, "pullback_wait_too_shallow"

    engine._detect_pullback = detect
    engine._scan_symbol(
        symbol,
        decision_time,
        {"percentile": 0.75, "score": 1.0, "extension_atr": 0.5, "return_4h_pct": 0.01},
    )

    assert observed_intervals == [timedelta(minutes=5)]


def test_mtpc_short_impulse_is_directionally_mirrored() -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            allow_short=True,
            impulse=replace(
                base.mtpc.impulse,
                max_breakout_distance_atr=10.0,
                min_body_atr=0.1,
                min_close_position=0.5,
                min_volume_ratio=1.0,
                max_prior_move_atr=10.0,
                max_ema21_extension_atr=10.0,
            ),
        ),
    )
    engine = _engine(config)
    start = datetime(2026, 1, 1)
    candles = _candles(start, 50, 15, slope=0.0)
    candles[-1] = Candle(candles[-1].timestamp, 100.0, 100.2, 96.0, 96.5, 2500.0)

    impulse = engine._detect_impulse(candles, Direction.SHORT)

    assert impulse is not None
    assert impulse.direction == Direction.SHORT
    assert impulse.breakout_level == min(row.low for row in candles[-13:-1])
    assert impulse.breakout_distance_atr > 0.0
    assert impulse.body_atr > 0.0


def test_mtpc_trend_ema_pullback_detects_closed_long_setup() -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            pullback=replace(
                base.mtpc.pullback,
                trend_ema_min_depth_atr=0.1,
                trend_ema_max_depth_atr=5.0,
                trend_ema_min_prior_extension_atr=0.1,
                trend_ema_max_volume_ratio=2.0,
                trend_ema_proximity_atr=2.0,
                trend_ema_max_adverse_close_atr=2.0,
                trend_ema_max_reclaim_extension_atr=2.0,
                trend_ema_min_alignment_atr=-2.0,
            ),
        ),
    )
    engine = _engine(config)
    rows = _candles(datetime(2026, 1, 1), 60, 15, slope=0.002)
    ema21 = ema([row.close for row in rows], 21)[-1]
    last = rows[-1]
    rows[-1] = Candle(
        last.timestamp,
        last.open,
        max(last.open, ema21 * 1.002),
        ema21 * 0.999,
        ema21 * 1.001,
        900.0,
    )

    setup = engine._detect_trend_ema_pullback(rows, Direction.LONG)

    assert setup is not None
    impulse, pullback = setup
    assert impulse.direction == Direction.LONG
    assert pullback.direction == Direction.LONG
    assert pullback.low == rows[-1].low
    assert impulse.impulse_high > pullback.low


def test_mtpc_trend_ema_pullback_is_directionally_mirrored_for_short() -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            allow_short=True,
            pullback=replace(
                base.mtpc.pullback,
                trend_ema_min_depth_atr=0.1,
                trend_ema_max_depth_atr=5.0,
                trend_ema_min_prior_extension_atr=0.1,
                trend_ema_max_volume_ratio=2.0,
                trend_ema_proximity_atr=2.0,
                trend_ema_max_adverse_close_atr=2.0,
                trend_ema_max_reclaim_extension_atr=2.0,
                trend_ema_min_alignment_atr=-2.0,
            ),
        ),
    )
    engine = _engine(config)
    rows = _candles(datetime(2026, 1, 1), 60, 15, slope=-0.002)
    ema21 = ema([row.close for row in rows], 21)[-1]
    last = rows[-1]
    rows[-1] = Candle(
        last.timestamp,
        last.open,
        ema21 * 1.001,
        min(last.open, ema21 * 0.998),
        ema21 * 0.999,
        900.0,
    )

    setup = engine._detect_trend_ema_pullback(rows, Direction.SHORT)

    assert setup is not None
    impulse, pullback = setup
    assert impulse.direction == Direction.SHORT
    assert pullback.direction == Direction.SHORT
    assert pullback.low == rows[-1].high
    assert impulse.impulse_low < pullback.low


def test_mtpc_short_fill_revalidation_builds_targets_below_entry() -> None:
    base = _config()
    config = replace(
        base,
        mtpc=replace(
            base.mtpc,
            allow_short=True,
            pullback=replace(
                base.mtpc.pullback,
                max_entry_chase_atr=1.0,
                min_target_to_cost_ratio=1.0,
            ),
        ),
    )
    candle = Candle(datetime(2026, 1, 1), 99.0, 99.2, 98.8, 99.0, 10_000.0)
    signal = Signal(Direction.SHORT, 0.8, MTPC_REASON_TOKEN, 0.03, 0.045, 1.0, 48)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.8,
        0.01,
        2.0,
        "mtpc_ok",
        {
            "trigger_atr_15m": 2.0,
            "trigger_close": 100.0,
            "structural_stop_price": 103.0,
        },
    )

    adjusted = _mtpc_adjust_candidate_for_fill(
        config,
        candidate,
        99.0,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0),
        {},
    )

    assert adjusted is not None
    assert adjusted.signal.direction == Direction.SHORT
    assert adjusted.metadata["target_1_price"] < 99.0
    assert adjusted.metadata["target_2_price"] < adjusted.metadata["target_1_price"]


def test_mtpc_risk_budget_obeys_hard_cap() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            risk_control=replace(config.mtpc.risk_control, trade_risk_pct=0.03, max_trade_risk_pct=0.02),
        ),
    )
    assert _mtpc_trade_risk_budget(config, 200.0) == 4.0


def test_mtpc_total_open_risk_cap_counts_existing_campaign_budget() -> None:
    config = _config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        103.0,
        0.05,
        0,
        100,
        100.0,
        entry_reason=MTPC_REASON_TOKEN,
        campaign_risk_budget_usdt=3.0,
    )
    positions = {"BTCUSDT": position}
    limited = replace(
        config,
        mtpc=replace(
            config.mtpc,
            risk_control=replace(config.mtpc.risk_control, max_total_open_risk_pct=0.02),
        ),
    )
    expanded = replace(
        limited,
        mtpc=replace(
            limited.mtpc,
            risk_control=replace(limited.mtpc.risk_control, max_total_open_risk_pct=0.03),
        ),
    )

    assert not _mtpc_total_open_risk_allows_entry(limited, positions, 200.0, 3.0)
    assert _mtpc_total_open_risk_allows_entry(expanded, positions, 200.0, 3.0)


def test_mtpc_quantity_cap_uses_exact_structural_stop() -> None:
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
    signal = Signal(Direction.LONG, 0.7, MTPC_REASON_TOKEN, 0.02, 0.02, 1.0)
    rules = SymbolRules("BTCUSDT", "0.1", "0.001", "0.001", "5")
    execution = BacktestExecutionConfig(
        taker_fee_rate=0.0005,
        market_slippage_bps=2.0,
        stop_slippage_bps=5.0,
    )

    quantity = _cmipr_initial_quantity_within_campaign_budget(
        signal,
        candle,
        requested_quantity=2.0,
        campaign_risk_budget_usdt=2.0,
        execution_config=execution,
        rules=rules,
        exact_stop_price=97.0,
    )
    entry = market_entry_fill(execution, rules, Direction.LONG, quantity, 100.0, 100_000.0)
    stop = market_exit_fill(execution, rules, Direction.LONG, quantity, 97.0, "stop_market", 100_000.0)
    full_cost_risk = quantity * (entry.price - stop.price) + entry.fee + stop.fee

    assert quantity > 0.0
    assert full_cost_risk <= 2.0 + 1e-9


def test_mtpc_fill_guard_rejects_liquidation_before_stop() -> None:
    config = _config()
    candle = Candle(datetime(2026, 1, 1), 100.0, 101.0, 99.0, 100.0, 1000.0)
    signal = Signal(Direction.LONG, 0.7, MTPC_REASON_TOKEN, 0.30, 0.30, 1.0)
    candidate = EntryCandidate(
        "BTCUSDT",
        signal,
        candle,
        0.7,
        0.01,
        1.2,
        "ok",
        {
            "trigger_atr_15m": 2.0,
            "trigger_close": 100.0,
            "structural_stop_price": 70.0,
        },
    )
    stats = {}
    adjusted = _mtpc_adjust_candidate_for_fill(
        config,
        candidate,
        100.0,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        stats,
    )
    assert adjusted is None
    assert stats.get("reject_fill_stop_too_wide", 0) + stats.get("reject_liquidation_before_stop", 0) == 1


def test_mtpc_same_bar_stop_has_priority_over_target() -> None:
    config = _config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        102.0,
        0.05,
        0,
        1000,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        initial_stop_price=98.0,
        campaign_risk_budget_usdt=2.2,
        initial_leg_full_cost_risk_usdt=2.2,
        strategy_metadata={
            "event_id": "event:test",
            "target_1_price": 102.0,
            "target_2_price": 104.0,
            "initial_quantity": 1.0,
            "pullback_low": 98.5,
            "breakout_level": 99.0,
        },
    )
    positions = {"BTCUSDT": position}
    trades = []
    stats = {}
    execution_stats = BacktestExecutionStats()
    trader = SimpleNamespace(
        client=SimpleNamespace(
            symbol_rules=lambda symbol: SymbolRules(symbol, "0.1", "0.001", "0.001", "5")
        )
    )
    candle = Candle(datetime(2026, 1, 1, 0, 1), 100.0, 103.0, 97.0, 101.0, 1000.0)
    _, closed = _manage_mtpc_position_1m(
        trader,
        config,
        200.0,
        positions,
        trades,
        position,
        candle,
        1,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        execution_stats,
        {},
        {},
        stats,
    )
    assert closed
    assert trades[0]["exit_reason"] == "mtpc_stop_loss_1m"
    assert execution_stats.same_bar_tp_sl_conflict_count == 1


def test_mtpc_short_same_bar_stop_has_priority_over_target() -> None:
    config = _config()
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.SHORT,
        1.0,
        100.0,
        102.0,
        98.0,
        0.05,
        0,
        1000,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        initial_stop_price=102.0,
        campaign_risk_budget_usdt=2.2,
        initial_leg_full_cost_risk_usdt=2.2,
        strategy_metadata={
            "event_id": "event:test-short",
            "target_1_price": 98.0,
            "target_2_price": 96.0,
            "initial_quantity": 1.0,
            "pullback_high": 101.5,
            "breakout_level": 101.0,
        },
    )
    positions = {"BTCUSDT": position}
    trades = []
    execution_stats = BacktestExecutionStats()
    trader = SimpleNamespace(
        client=SimpleNamespace(
            symbol_rules=lambda symbol: SymbolRules(symbol, "0.1", "0.001", "0.001", "5")
        )
    )
    candle = Candle(datetime(2026, 1, 1, 0, 1), 100.0, 103.0, 97.0, 99.0, 1000.0)

    _, closed = _manage_mtpc_position_1m(
        trader,
        config,
        200.0,
        positions,
        trades,
        position,
        candle,
        1,
        BacktestExecutionConfig(taker_fee_rate=0.0005, market_slippage_bps=2.0, stop_slippage_bps=5.0),
        execution_stats,
        {},
        {},
        {},
    )

    assert closed
    assert trades[0]["exit_reason"] == "mtpc_stop_loss_1m"
    assert execution_stats.same_bar_tp_sl_conflict_count == 1


def test_mtpc_breakeven_uses_actual_initial_leg_full_cost_r() -> None:
    config = _config()
    config = replace(
        config,
        mtpc=replace(
            config.mtpc,
            exit=replace(config.mtpc.exit, structural_fail_fast_enabled=False),
        ),
    )
    position = PortfolioPosition(
        "BTCUSDT",
        Direction.LONG,
        1.0,
        100.0,
        98.0,
        104.0,
        0.05,
        0,
        1000,
        100.0,
        entry_time=datetime(2026, 1, 1),
        raw_entry_price=100.0,
        initial_stop_price=98.0,
        campaign_risk_budget_usdt=4.0,
        initial_leg_full_cost_risk_usdt=2.0,
        strategy_metadata={
            "event_id": "event:test-r-basis",
            "target_1_price": 104.0,
            "target_2_price": 106.0,
            "initial_quantity": 1.0,
            "pullback_low": 98.5,
            "breakout_level": 99.0,
        },
    )
    positions = {"BTCUSDT": position}
    execution_config = BacktestExecutionConfig(
        taker_fee_rate=0.0005,
        market_slippage_bps=2.0,
        stop_slippage_bps=5.0,
    )
    trader = SimpleNamespace(
        client=SimpleNamespace(
            symbol_rules=lambda symbol: SymbolRules(symbol, "0.1", "0.001", "0.001", "5")
        )
    )
    candle = Candle(datetime(2026, 1, 1, 0, 5), 100.0, 101.8, 99.5, 101.2, 1000.0)

    _, closed = _manage_mtpc_position_1m(
        trader,
        config,
        200.0,
        positions,
        [],
        position,
        candle,
        1,
        execution_config,
        BacktestExecutionStats(),
        {},
        {},
        {},
    )

    assert not closed
    assert position.mtpc_max_initial_leg_executable_r >= config.mtpc.exit.breakeven_trigger_r
    assert position.mtpc_max_campaign_executable_r < config.mtpc.exit.breakeven_trigger_r
    assert position.stop_price >= position.entry_price * (1.0 + config.mtpc.exit.breakeven_cost_buffer_pct)
