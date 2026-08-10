from __future__ import annotations

from BreakoutV13GridRelativeSqueezeStage29Freqtrade import (
    BreakoutV13GridRelativeSqueeze20WeakStructureBreadth45GridV9Freqtrade,
)


class BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade(
    BreakoutV13GridRelativeSqueeze20WeakStructureBreadth45GridV9Freqtrade,
):
    """Frozen V13 research selection; intentionally not wired to GUI/LIVE.

    The inherited execution path remains the frozen Breakout/Grid Max2 path.
    V13 adds only completed-candle, side-specific stake sizing and the
    restart-safe realized-equity high-water governor.
    """

    STRATEGY_VERSION = (
        "breakout_v13_bull_capture_grid_relative_squeeze_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v13_bull_capture_grid_relative_squeeze_high_water"
    )

    # Bull-rotation Breakout-long convexity.
    V13_BULL_ROTATION_SCALE = 1.50
    V13_BULL_ROTATION_MIN_BREADTH_TRAVEL_24H = 2.40
    V13_BULL_ROTATION_MAX_BTC_TREND_4H = 0.65

    # Directional defenses inherited by the selected V13 core.
    V13_BO_SHORT_EXHAUSTION_SCALE = 0.35
    V13_BO_SHORT_EXHAUSTION_MAX_BREADTH = 0.08
    V13_BO_SHORT_EXHAUSTION_MIN_EMA55_ATR = -4.38
    V13_GRID_ROTATION_SCALE = 0.25
    V13_GRID_ROTATION_MIN_BREADTH_CHANGE_24H = 0.02
    V13_GRID_ROTATION_MAX_MEDIAN_RETURN_24H = 0.00
    V13_BO_LONG_CHOP_SCALE = 0.35
    V13_BO_LONG_CHOP_MAX_BREADTH_TRAVEL_24H = 2.63
    V13_BO_LONG_CHOP_MAX_MARKET_EFFICIENCY = 0.262

    # Underwater Grid-short relative-squeeze defense.
    V13_GRID_SQUEEZE_MIN_DRAWDOWN = 0.03
    V13_GRID_SQUEEZE_SCALE = 0.20
    V13_GRID_SQUEEZE_MIN_SYMBOL_RETURN_4H = 0.0075
    V13_GRID_SQUEEZE_MAX_MARKET_REGIME = 0.00
    V13_GRID_SQUEEZE_MAX_SCORE = 5.0
    V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H = -0.30
    V13_GRID_SQUEEZE_MIN_VOLUME_RATIO = 1.50
    V13_GRID_SQUEEZE_MIN_BREADTH = 0.45
    V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE = True
    V13_GRID_SQUEEZE_MAX_ALIGNMENT = 0.70
    V13_GRID_SQUEEZE_MAX_EXTENSION = 0.25
