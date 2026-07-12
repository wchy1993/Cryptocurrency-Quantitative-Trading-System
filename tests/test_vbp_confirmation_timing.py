from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from crypto_scalper.live_execution_backtest import (
    VbpBreakoutState,
    VbpConfirmationMetrics,
    VbpPullbackMetrics,
    _vbp_confirmation_quality_allows,
    _vbp_pending_timing_phase,
    _vbp_pullback_quality_action,
)


def _pending() -> VbpBreakoutState:
    return VbpBreakoutState(
        symbol="BTCUSDT",
        breakout_index=10,
        breakout_time=datetime(2026, 1, 1),
        breakout_level=100.0,
        consolidation_bottom=98.0,
        pullback_target=100.0,
        breakout_volume=1000.0,
        breakout_close=101.0,
        tp2_price=105.0,
        breakout_atr=1.0,
    )


def test_confirmation_requires_a_bar_after_pullback_touch() -> None:
    pending = _pending()
    pending.pullback_touch_index = 12
    pending.touched_pullback = True

    assert _vbp_pending_timing_phase(pending, 12) == "wait_after_pullback"
    assert _vbp_pending_timing_phase(pending, 13) == "confirm"


def test_execution_is_only_next_open_after_confirmation() -> None:
    pending = _pending()
    pending.pullback_touch_index = 12
    pending.confirmation_index = 13
    pending.confirmation_price = 101.5

    assert _vbp_pending_timing_phase(pending, 13) == "wait_confirmation_close"
    assert _vbp_pending_timing_phase(pending, 14) == "execute"
    assert _vbp_pending_timing_phase(pending, 15) == "execution_missed"


def test_pullback_quality_distinguishes_wait_from_reject() -> None:
    config = SimpleNamespace(
        pullback_quality_enabled=True,
        pullback_depth_atr_min=0.5,
        pullback_depth_atr_max=2.0,
        pullback_depth_to_breakout_max=3.0,
        pullback_bars_max=10,
        pullback_require_hold_breakout_level=True,
    )

    assert _vbp_pullback_quality_action(config, VbpPullbackMetrics(0.25, 0.5, 3, False))[0] == "wait"
    assert _vbp_pullback_quality_action(config, VbpPullbackMetrics(2.5, 2.0, 3, False))[0] == "reject"
    assert _vbp_pullback_quality_action(config, VbpPullbackMetrics(1.0, 2.0, 3, False)) == ("allow", "ok")


def test_confirmation_quality_is_configurable() -> None:
    config = SimpleNamespace(
        confirmation_quality_enabled=True,
        confirmation_body_atr_min=0.25,
        confirmation_close_position_min=0.70,
        confirmation_upper_wick_ratio_max=0.25,
    )

    assert _vbp_confirmation_quality_allows(config, VbpConfirmationMetrics(0.5, 0.8, 0.1)) == (True, "ok")
    assert _vbp_confirmation_quality_allows(config, VbpConfirmationMetrics(0.1, 0.8, 0.1))[0] is False
