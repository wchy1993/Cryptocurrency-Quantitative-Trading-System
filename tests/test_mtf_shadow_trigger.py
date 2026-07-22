from __future__ import annotations

from datetime import datetime, timedelta

from crypto_scalper.models import Candle, Direction
from crypto_scalper.mtf_4h_rsi_regime import MtfSetupResult, MtfTriggerResult
from crypto_scalper.mtf_shadow_trigger import (
    MtfNativeShadowTriggerConfig,
    MtfNativeShadowTriggerDetector,
    can_be_native_shadow_trigger,
    classify_structure_overlap,
    mtf_shadow_trigger_event_id,
)


def _candle(index: int, open_: float, high: float, low: float, close: float, volume: float = 100.0) -> Candle:
    return Candle(
        datetime(2026, 1, 1) + timedelta(minutes=15 * index),
        open_,
        high,
        low,
        close,
        volume,
    )


def _flat_history(count: int = 24) -> list[Candle]:
    return [
        _candle(index, 100.0, 100.6, 99.5, 100.0 + (0.02 if index % 2 else -0.02))
        for index in range(count)
    ]


def _setup(direction: Direction, ema21: float = 100.5) -> MtfSetupResult:
    return MtfSetupResult(
        direction=direction,
        support=98.0,
        resistance=102.0,
        ema21=ema21,
        ema34=100.0,
        setup_low=98.0,
        setup_high=102.0,
        rsi=50.0,
    )


def test_shadow_event_id_is_mode_and_direction_specific() -> None:
    timestamp = datetime(2026, 1, 2, 12, 0)
    base = mtf_shadow_trigger_event_id("BTCUSDT", Direction.LONG, "two_bar_false_break", timestamp)
    assert base == mtf_shadow_trigger_event_id("btcusdt", "long", "two_bar_false_break", timestamp)
    assert base != mtf_shadow_trigger_event_id("BTCUSDT", Direction.SHORT, "two_bar_false_break", timestamp)
    assert base != mtf_shadow_trigger_event_id("BTCUSDT", Direction.LONG, "higher_low_break", timestamp)


def test_two_bar_false_break_detects_long_and_short_symmetrically() -> None:
    detector = MtfNativeShadowTriggerDetector()

    long_rows = _flat_history()
    long_rows.append(_candle(24, 100.4, 101.0, 98.5, 100.0))
    long_rows.append(_candle(25, 100.0, 101.5, 99.8, 101.3, 120.0))
    long_trigger, long_reason = detector(Direction.LONG, long_rows, _setup(Direction.LONG), "15m")
    assert long_trigger is not None, long_reason
    assert long_trigger.mode == "two_bar_false_break"
    assert long_trigger.metadata["shadow_probe_wick_ratio"] >= 0.20

    short_rows = _flat_history()
    short_rows.append(_candle(24, 99.6, 101.5, 99.0, 100.0))
    short_rows.append(_candle(25, 100.0, 100.2, 98.5, 98.7, 120.0))
    short_trigger, short_reason = detector(Direction.SHORT, short_rows, _setup(Direction.SHORT, 99.5), "15m")
    assert short_trigger is not None, short_reason
    assert short_trigger.mode == "two_bar_false_break"


def test_pivot_break_requires_new_higher_low_or_lower_high() -> None:
    detector = MtfNativeShadowTriggerDetector(
        MtfNativeShadowTriggerConfig(allow_two_bar_false_break=False)
    )
    rows = _flat_history()
    rows[-1] = _candle(23, 100.0, 100.5, 99.8, 100.1)
    rows.append(_candle(24, 100.1, 100.4, 99.6, 100.0))
    rows.append(_candle(25, 100.0, 101.2, 99.8, 101.0, 120.0))
    trigger, reason = detector(Direction.LONG, rows, _setup(Direction.LONG), "15m")
    assert trigger is not None, reason
    assert trigger.mode == "higher_low_break"


def test_prefilter_rejects_plain_directional_candle_without_probe_structure() -> None:
    rows = _flat_history()
    rows.append(_candle(24, 100.0, 100.6, 99.5, 100.1))
    rows.append(_candle(25, 100.1, 100.8, 99.7, 100.7))
    assert not can_be_native_shadow_trigger(rows, len(rows) - 1, 4)


def test_structure_overlap_separates_precursor_from_independent_event() -> None:
    trigger = datetime(2026, 1, 2, 12, 0)
    precursor = classify_structure_overlap(
        trigger,
        [trigger + timedelta(minutes=90)],
        120,
    )
    assert precursor["structure_followed_within_window"] is True
    assert precursor["independent_from_structure_window"] is False
    assert precursor["minutes_to_next_structure"] == 90.0

    independent = classify_structure_overlap(
        trigger,
        [trigger + timedelta(minutes=135)],
        120,
    )
    assert independent["independent_from_structure_window"] is True


def test_existing_trigger_result_defaults_to_empty_metadata() -> None:
    candle = _candle(0, 100.0, 101.0, 99.0, 100.5)
    trigger = MtfTriggerResult(Direction.LONG, "structure_break", candle, 1.0)
    assert trigger.metadata == {}
