from __future__ import annotations

from BreakoutV12RegimeAdaptiveGridV9Freqtrade import (
    BreakoutV12MultiRegimeMatureBull060RelativeGuardFreqtrade,
)


class BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade(
    BreakoutV12MultiRegimeMatureBull060RelativeGuardFreqtrade,
):
    """Frozen research selection for multi-regime Active50 validation.

    Entry, campaign management, stop-loss, take-profit, leverage and Max2
    behavior remain inherited from frozen Breakout-v11/Grid-v8.  This entry
    point freezes only the causal, side-specific sizing layers validated over
    2024, both halves of 2025 and 2026 YTD.

    It is intentionally not referenced by the existing GUI or LIVE config.
    """

    STRATEGY_VERSION = "breakout_v12_regime_adaptive_grid_v9_20260809"

    # Keep runtime high-water state independent from the frozen v11 strategy.
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v12_regime_adaptive_grid_v9_high_water"
    )

    # Existing realized-equity portfolio governor, explicitly frozen here.
    PORTFOLIO_GOVERNOR_TRIGGER = 0.18
    PORTFOLIO_RECENT_WINDOW = 8
    PORTFOLIO_DEFENSIVE_SCALE = 0.15
    PORTFOLIO_RECOVERY_SCALE = 0.60
    PORTFOLIO_RECOVERY_MIN_WINS = 4
    PORTFOLIO_RECOVERY_MIN_RETURN = 0.0
    PORTFOLIO_FULL_RISK_MIN_WINS = 5
    PORTFOLIO_FULL_RISK_MIN_RETURN = 0.60

    # Grid-v9 short: persistent weak-bull and confirmed rebound conflicts.
    GRID_V9_NEUTRAL_SHORT_MIN_PERSISTENCE = 0.90
    GRID_V9_NEUTRAL_SHORT_MIN_REGIME = 0.05
    GRID_V9_NEUTRAL_SHORT_MAX_REGIME = 0.25
    GRID_V9_NEUTRAL_SHORT_SCALE = 0.00
    GRID_V9_REBOUND_MIN_RETURN_4H = 0.0075
    GRID_V9_REBOUND_MIN_PERSISTENCE = 0.75
    GRID_V9_REBOUND_MIN_ETH_RETURN_4H = 0.005
    GRID_V9_REBOUND_MIN_BREADTH = 0.50
    GRID_V9_REBOUND_MIN_REGIME = 0.00
    GRID_V9_REBOUND_REQUIRE_MARKET_CONFIRMATION = True
    GRID_V9_REBOUND_SHORT_SCALE = 0.00

    # Grid-v9 long: intervene only when structure is already weakening.
    GRID_V9_LONG_MAX_ALIGNMENT_ATR = 1.25
    GRID_V9_LONG_MIN_MARKET_EFFICIENCY = 0.23
    GRID_V9_LONG_MAX_BREADTH_CHANGE_24H = -0.25
    GRID_V9_LONG_OVEREXTENSION_SCALE = 0.00

    # Grid-v9 short squeeze: weak broad tape but resilient target symbol.
    GRID_V9_RS_SHORT_MIN_PERSISTENCE = 0.50
    GRID_V9_RS_SHORT_MAX_PERSISTENCE = 0.75
    GRID_V9_RS_SHORT_MAX_REGIME = -0.15
    GRID_V9_RS_SHORT_MIN_SYMBOL_RETURN_4H = 0.00
    GRID_V9_RS_SHORT_SCALE = 0.35

    # Independent Breakout long/short recovery sleeves.
    BO_V12_LONG_SLEEVE_WINDOW = 6
    BO_V12_LONG_SLEEVE_MIN_LOSSES = 4
    BO_V12_LONG_SLEEVE_MAX_RETURN = -0.10
    BO_V12_LONG_SLEEVE_MIN_SYMBOL_RETURN_4H = 0.06
    BO_V12_LONG_SLEEVE_MAX_MARKET_RETURN_24H = 0.01
    BO_V12_LONG_SLEEVE_SCALE = 0.35
    BO_V12_LONG_SLEEVE_PORTFOLIO_CAP = None
    BO_V12_SHORT_SLEEVE_WINDOW = 3
    BO_V12_SHORT_SLEEVE_MIN_LOSSES = 3
    BO_V12_SHORT_SLEEVE_MAX_RETURN = -0.20
    BO_V12_SHORT_SLEEVE_MIN_LOW_OPPORTUNITY = 0.50
    BO_V12_SHORT_SLEEVE_SCALE = 0.35
    BO_V12_SHORT_SLEEVE_PORTFOLIO_CAP = None

    # Mature bull chase: price impulse without continuing breadth expansion.
    BO_V12_MATURE_BULL_CHASE_MIN_PERSISTENCE = 0.75
    BO_V12_MATURE_BULL_CHASE_MIN_REGIME = 0.20
    BO_V12_MATURE_BULL_CHASE_MIN_SYMBOL_RETURN_4H = 0.04
    BO_V12_MATURE_BULL_CHASE_MAX_BREADTH_CHANGE_4H = 0.05
    BO_V12_MATURE_BULL_CHASE_SCALE = 0.60
