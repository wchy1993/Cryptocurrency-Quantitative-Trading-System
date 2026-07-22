from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Direction
from crypto_scalper.models import Candle
from crypto_scalper.mtf_candidate_quality import (
    can_be_structure_break,
    candidate_metrics,
    full_cost_stop_risk_usdt,
    mtf_candidate_event_id,
    mtf_candidate_quality,
    next_1m_execution_index,
)
from crypto_scalper.live_execution_backtest import _mtf_quality_sleeve_allows
from crypto_scalper.risk import BacktestExecutionConfig


def test_candidate_event_id_is_stable_and_direction_specific() -> None:
    timestamp = datetime(2026, 1, 2, 3, 45)
    first = mtf_candidate_event_id("BTCUSDT", Direction.LONG, timestamp)
    assert first == mtf_candidate_event_id("btcusdt", "long", timestamp)
    assert first != mtf_candidate_event_id("BTCUSDT", Direction.SHORT, timestamp)


def test_quality_score_is_directionally_symmetric_and_ignores_outcome_fields() -> None:
    long_metadata = {
        "rank_score": 4.0,
        "trigger_close_position": 0.8,
        "trigger_body_atr": 0.6,
        "btc_1h_return": 0.003,
        "btc_4h_return": 0.01,
        "1h_rsi": 56.0,
        "target_to_cost_ratio": 8.0,
    }
    short_metadata = {
        **long_metadata,
        "trigger_close_position": 0.2,
        "btc_1h_return": -0.003,
        "btc_4h_return": -0.01,
        "1h_rsi": 44.0,
    }
    long_score = mtf_candidate_quality(long_metadata, Direction.LONG).score
    short_score = mtf_candidate_quality(short_metadata, Direction.SHORT).score
    assert short_score == pytest.approx(long_score)
    assert mtf_candidate_quality({**long_metadata, "shadow_net_r": 99.0}, Direction.LONG).score == long_score


def test_next_execution_is_one_full_minute_after_confirmation() -> None:
    available = datetime(2026, 1, 2, 12, 0)
    timestamps = [available + timedelta(minutes=offset) for offset in range(4)]
    assert next_1m_execution_index(timestamps, available) == 1
    assert next_1m_execution_index(timestamps[:1], available) is None


def test_structure_break_prefilter_accepts_both_directions_and_rejects_inside_bar() -> None:
    start = datetime(2026, 1, 2, 12, 0)
    previous = [
        Candle(start + timedelta(minutes=15 * index), 100.0, 101.0, 99.0, 100.0, 10.0)
        for index in range(3)
    ]
    long_break = Candle(start + timedelta(minutes=45), 100.0, 102.0, 99.5, 101.5, 10.0)
    short_break = Candle(start + timedelta(minutes=45), 100.0, 100.5, 98.0, 98.5, 10.0)
    inside = Candle(start + timedelta(minutes=45), 100.0, 100.5, 99.5, 100.2, 10.0)
    assert can_be_structure_break([*previous, long_break], 3, 3)
    assert can_be_structure_break([*previous, short_break], 3, 3)
    assert not can_be_structure_break([*previous, inside], 3, 3)


def test_full_cost_stop_risk_includes_both_fees_and_slippage() -> None:
    rules = SymbolRules("TESTUSDT", "0.001", "0.001", "0.01", "5")
    execution = BacktestExecutionConfig(
        taker_fee_rate=0.0005,
        market_slippage_bps=2.0,
        stop_slippage_bps=5.0,
    )
    position = SimpleNamespace(
        direction=Direction.LONG,
        quantity=10.0,
        entry_price=100.02,
        raw_entry_price=100.0,
        initial_stop_price=99.0,
        stop_price=99.0,
        entry_fee=0.5001,
        entry_slippage_cost=0.2,
        liquidity_reference_quote_volume=1_000_000.0,
    )
    risk = full_cost_stop_risk_usdt(position, execution, rules)
    assert risk > 10.0
    doubled = SimpleNamespace(**{**position.__dict__, "quantity": 20.0, "entry_fee": 1.0002, "entry_slippage_cost": 0.4})
    assert full_cost_stop_risk_usdt(doubled, execution, rules) == pytest.approx(risk * 2.0)


def test_candidate_metrics_uses_shadow_r_not_currency_sizing() -> None:
    metrics = candidate_metrics(
        [
            {"side": "LONG", "shadow_net_r": 2.0, "shadow_cost_r": 0.1, "hit_1r": True},
            {"side": "SHORT", "shadow_net_r": -1.0, "shadow_cost_r": 0.2, "stop_first": True},
        ]
    )
    assert metrics["expectancy_r"] == pytest.approx(0.5)
    assert metrics["profit_factor_r"] == pytest.approx(2.0)
    assert metrics["long_count"] == 1
    assert metrics["short_count"] == 1


def test_quality_sleeve_is_short_only_and_respects_hard_frequency_limits() -> None:
    strategy = SimpleNamespace(
        mtf_quality_sleeve_enabled=True,
        mtf_quality_sleeve_allow_long=False,
        mtf_quality_sleeve_allow_short=True,
        mtf_quality_sleeve_ranking_mode="rank",
        mtf_quality_sleeve_long_min_score=999.0,
        mtf_quality_sleeve_short_min_score=3.7,
        mtf_quality_sleeve_max_open_positions=1,
        mtf_quality_sleeve_max_daily_trades=4,
        mtf_quality_sleeve_symbol_cooldown_hours=0,
        mtf_quality_sleeve_max_signal_age_minutes=0,
        mtf_symbol_cooldown_hours=12,
    )
    config = SimpleNamespace(strategy=strategy)
    timestamp = datetime(2026, 1, 2, 12, 0)
    common = (config, Direction.SHORT, 3.7, 70.0, timestamp, "BTCUSDT")
    assert _mtf_quality_sleeve_allows(*common, 0, {}, {}) == (True, "")
    assert _mtf_quality_sleeve_allows(config, Direction.LONG, 10.0, 100.0, timestamp, "BTCUSDT", 0, {}, {}) == (
        False,
        "mtf_quality_sleeve_side_filtered",
    )
    assert _mtf_quality_sleeve_allows(*common, 1, {}, {}) == (
        False,
        "mtf_quality_sleeve_max_open_positions",
    )
    assert _mtf_quality_sleeve_allows(*common, 0, {timestamp.date(): 4}, {}) == (
        False,
        "mtf_quality_sleeve_daily_trade_limit",
    )
    assert _mtf_quality_sleeve_allows(config, Direction.SHORT, 3.69, 99.0, timestamp, "BTCUSDT", 0, {}, {}) == (
        False,
        "mtf_quality_sleeve_score_low",
    )
    assert _mtf_quality_sleeve_allows(*common, 0, {}, {}, 1.0) == (
        False,
        "mtf_quality_sleeve_stale_signal",
    )
