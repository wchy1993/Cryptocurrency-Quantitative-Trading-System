from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from crypto_scalper.models import Candle, Direction
from crypto_scalper.mtf_momentum_reset import (
    STAGE18_EXPERIMENTS,
    STAGE16_EXPERIMENTS,
    build_release_signal,
    can_be_contraction_release,
    contraction_release_features,
    detect_momentum_reset_release,
    momentum_reset_features,
    momentum_reset_config_from_strategy,
    mtf_momentum_reset_event_id,
)
from scripts.mtf_momentum_reset_research import _dedupe_event_clusters
from scripts.mtf_momentum_reset_selection import _path_metrics


def _one_hour_reset() -> list[Candle]:
    start = datetime(2026, 1, 1)
    closes = [100.0 + index * 0.15 for index in range(52)]
    closes += [107.5, 107.0, 106.6, 106.4, 106.5, 106.7, 107.0, 107.5]
    return [
        Candle(
            start + timedelta(hours=index),
            close - 0.05,
            close + 0.15,
            close - 0.15,
            close,
            100.0,
        )
        for index, close in enumerate(closes)
    ]


def _thirty_minute_release() -> list[Candle]:
    start = datetime(2026, 1, 1)
    candles: list[Candle] = []
    for index in range(53):
        close = 100.0 + index * 0.005
        candles.append(
            Candle(
                start + timedelta(minutes=30 * index),
                close - 0.05,
                close + 0.25,
                close - 0.25,
                close,
                100.0,
            )
        )
    for close in (100.25, 100.27, 100.24, 100.28, 100.26, 100.29):
        index = len(candles)
        candles.append(
            Candle(
                start + timedelta(minutes=30 * index),
                close - 0.01,
                close + 0.06,
                close - 0.06,
                close,
                60.0,
            )
        )
    index = len(candles)
    candles.append(
        Candle(
            start + timedelta(minutes=30 * index),
            100.30,
            100.85,
            100.25,
            100.80,
            130.0,
        )
    )
    return candles


def _mirror(candles: list[Candle], axis: float = 200.0) -> list[Candle]:
    return [
        Candle(
            item.timestamp,
            axis - item.open,
            axis - item.low,
            axis - item.high,
            axis - item.close,
            item.volume,
        )
        for item in candles
    ]


def test_momentum_reset_and_contraction_release_use_closed_history() -> None:
    one_hour = _one_hour_reset()
    thirty_minute = _thirty_minute_release()

    reset = momentum_reset_features(Direction.LONG, one_hour)
    release = contraction_release_features(Direction.LONG, thirty_minute)

    assert reset is not None
    assert reset["macd_reset_present"] is True
    assert reset["rsi_reset_present"] is True
    assert reset["reset_component_count"] == 3
    assert release is not None
    assert release["contraction_tr_ratio"] < 1.0
    assert release["contraction_volume_ratio"] < 1.0
    assert release["breakout_distance_atr"] > 0.0
    assert can_be_contraction_release(thirty_minute, len(thirty_minute) - 1, 6)


def test_long_and_short_features_are_symmetric() -> None:
    long_reset = momentum_reset_features(Direction.LONG, _one_hour_reset())
    short_reset = momentum_reset_features(Direction.SHORT, _mirror(_one_hour_reset()))
    long_release = contraction_release_features(Direction.LONG, _thirty_minute_release())
    short_release = contraction_release_features(Direction.SHORT, _mirror(_thirty_minute_release()))

    assert long_reset is not None and short_reset is not None
    assert long_release is not None and short_release is not None
    for field in ("rsi_recovery", "directional_ema_slope_atr", "directional_ema_distance_atr"):
        assert short_reset[field] == pytest.approx(long_reset[field])
    for field in (
        "contraction_range_atr",
        "contraction_tr_ratio",
        "contraction_volume_ratio",
        "breakout_distance_atr",
    ):
        assert short_release[field] == pytest.approx(long_release[field])


def test_release_event_stop_and_signal_are_structural() -> None:
    event = detect_momentum_reset_release(Direction.LONG, _one_hour_reset(), _thirty_minute_release())
    assert event is not None
    assert event.structural_stop < event.metadata["contraction_low"]
    config = SimpleNamespace(
        strategy=SimpleNamespace(
            mtf_min_stop_pct=0.0,
            mtf_max_stop_pct=0.02,
            mtf_take_profit_r=2.0,
            mtf_max_holding_minutes=720,
            mtf_risk_multiplier=1.0,
        )
    )
    built = build_release_signal(config, event)
    assert built is not None
    signal, metadata = built
    assert signal.direction is Direction.LONG
    assert signal.take_profit_pct == pytest.approx(signal.stop_loss_pct * 2.0)
    assert metadata["trigger_mode"] == "30m_contraction_release"
    assert metadata["structural_stop_price"] == event.structural_stop


def test_live_strategy_fields_map_to_detector_config() -> None:
    strategy = SimpleNamespace(
        mtf_momentum_reset_reset_lookback_1h=10,
        mtf_momentum_reset_min_release_close_position=0.61,
        mtf_momentum_reset_stop_atr_buffer=0.4,
    )

    config = momentum_reset_config_from_strategy(strategy)

    assert config.reset_lookback_1h == 10
    assert config.min_release_close_position == pytest.approx(0.61)
    assert config.stop_atr_buffer == pytest.approx(0.4)


def test_stage16_experiments_are_bounded_and_do_not_change_rank_policy() -> None:
    assert len(STAGE16_EXPERIMENTS) == 6
    names = {item.name for item in STAGE16_EXPERIMENTS}
    assert len(names) == 6
    row = {
        "reset_component_count": 3,
        "macd_reset_present": True,
        "rsi_reset_present": True,
        "ema_reset_present": True,
        "rsi_recovery": 5.0,
        "directional_ema_slope_atr": 0.1,
        "directional_ema_distance_atr": 1.0,
        "contraction_range_atr": 2.0,
        "contraction_tr_ratio": 0.7,
        "contraction_volume_ratio": 0.8,
        "release_body_atr": 0.4,
        "release_directional_close_position": 0.7,
        "release_volume_ratio": 1.2,
        "breakout_distance_atr": 0.4,
        "release_extension_atr": 1.0,
    }
    assert all(item.accepts(row) for item in STAGE16_EXPERIMENTS)


def test_stage18_selected_rule_is_long_rank_cost_and_btc_guard() -> None:
    selected = STAGE18_EXPERIMENTS[-1]
    row = {
        "side": "LONG",
        "rank_score": 4.25,
        "target_to_cost_ratio": 12.0,
        "btc_4h_return": -0.001,
        "reset_component_count": 1,
        "rsi_recovery": 1.0,
        "directional_ema_slope_atr": 0.0,
        "directional_ema_distance_atr": 1.0,
        "contraction_range_atr": 2.0,
        "contraction_tr_ratio": 1.0,
        "contraction_volume_ratio": 1.0,
        "release_body_atr": 0.2,
        "release_directional_close_position": 0.6,
        "release_volume_ratio": 1.0,
        "breakout_distance_atr": 0.4,
        "release_extension_atr": 1.0,
    }
    assert selected.accepts(row)
    assert not selected.accepts({**row, "side": "SHORT"})
    assert not selected.accepts({**row, "rank_score": 4.24})
    assert not selected.accepts({**row, "target_to_cost_ratio": 11.99})
    assert not selected.accepts({**row, "btc_4h_return": -0.0011})


def test_event_ids_include_side_and_cluster_dedupe_is_point_in_time() -> None:
    timestamp = datetime(2026, 1, 2)
    long_id = mtf_momentum_reset_event_id("BTCUSDT", Direction.LONG, timestamp)
    short_id = mtf_momentum_reset_event_id("BTCUSDT", Direction.SHORT, timestamp)
    assert long_id != short_id
    rows = [
        {"event_id": "a", "symbol": "BTCUSDT", "side": "LONG", "release_time": timestamp.isoformat()},
        {
            "event_id": "b",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "release_time": (timestamp + timedelta(hours=2)).isoformat(),
        },
        {
            "event_id": "c",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "release_time": (timestamp + timedelta(hours=5)).isoformat(),
        },
        {
            "event_id": "d",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "release_time": (timestamp + timedelta(hours=2)).isoformat(),
        },
    ]
    assert [item["event_id"] for item in _dedupe_event_clusters(rows, 4)] == ["a", "d", "c"]


def test_normalized_path_uses_rank_priority_and_old_position_exit_first() -> None:
    start = datetime(2026, 1, 1)
    shared = {
        "side": "LONG",
        "signal_available_time": start.isoformat(),
        "shadow_entry_time": (start + timedelta(minutes=1)).isoformat(),
        "shadow_exit_time": (start + timedelta(hours=1)).isoformat(),
        "shadow_net_r": 1.0,
    }
    rows = [
        {**shared, "event_id": "low", "symbol": "ETHUSDT", "rank_score": 4.3},
        {**shared, "event_id": "high", "symbol": "BTCUSDT", "rank_score": 5.0},
        {
            **shared,
            "event_id": "next",
            "symbol": "SOLUSDT",
            "rank_score": 4.5,
            "signal_available_time": (start + timedelta(hours=1)).isoformat(),
            "shadow_entry_time": (start + timedelta(hours=1)).isoformat(),
            "shadow_exit_time": (start + timedelta(hours=2)).isoformat(),
        },
    ]
    report = _path_metrics(rows, 200.0, 0.01, False)
    assert [item["event_id"] for item in report["trades"]] == ["high", "next"]
    assert report["final_equity"] == pytest.approx(204.0)


def test_normalized_path_can_use_point_in_time_trend_priority() -> None:
    start = datetime(2026, 1, 1)
    shared = {
        "side": "LONG",
        "signal_available_time": start.isoformat(),
        "shadow_entry_time": (start + timedelta(minutes=1)).isoformat(),
        "shadow_exit_time": (start + timedelta(hours=1)).isoformat(),
    }
    rows = [
        {
            **shared,
            "event_id": "rank",
            "symbol": "AAAUSDT",
            "rank_score": 9.0,
            "trend_score": 50.0,
            "shadow_net_r": -1.0,
        },
        {
            **shared,
            "event_id": "trend",
            "symbol": "BBBUSDT",
            "rank_score": 4.5,
            "trend_score": 75.0,
            "shadow_net_r": 2.0,
        },
    ]

    report = _path_metrics(
        rows,
        200.0,
        0.01,
        False,
        priority_fields=("trend_score", "rank_score"),
    )

    assert report["trade_count"] == 1
    assert report["trades"][0]["event_id"] == "trend"
    assert report["priority_fields"] == ["trend_score", "rank_score"]
