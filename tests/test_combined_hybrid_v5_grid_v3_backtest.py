from __future__ import annotations

from pathlib import Path

from crypto_scalper.combined_hybrid_v5_grid_v3_backtest import build_frozen_configs
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)


ROOT = Path(__file__).resolve().parents[1]
BREAKOUT_CONFIG = (
    ROOT / "config.volatility-breakout.hybrid-v5-balanced-expansion-runner-50.json"
)
GRID_CONFIG = ROOT / "config.trend-grid.v3-optimized-50.json"


def test_combined_frozen_configs_are_v5_plus_grid_v3_with_max_two() -> None:
    configs = build_frozen_configs(BREAKOUT_CONFIG, GRID_CONFIG)

    assert configs["breakout_signal"].long_k == 0.7
    assert configs["breakout_signal"].short_k == 0.6
    assert configs["breakout_signal"].stop_atr_multiple == 1.0
    assert configs["breakout_signal"].take_profit_r == 60.0
    assert configs["breakout_signal"].max_holding_minutes == 960
    assert configs["breakout_portfolio"].risk_per_trade_pct == 0.025
    assert configs["breakout_portfolio"].max_open_positions == 1
    assert configs["breakout_exit"].partial_take_profit_r == 8.0
    assert configs["breakout_exit"].partial_take_profit_fraction == 0.05

    assert configs["grid_signal"].timeframe_minutes == 60
    assert configs["grid_signal"].grid_spacing_atr == 0.6
    assert configs["grid_signal"].grid_levels == 2
    assert configs["grid_signal"].grid_target_spacing == 2.0
    assert configs["grid_portfolio"].max_open_campaigns == 1

    combined = configs["combined"]
    assert combined.max_open_positions == 2
    assert combined.allow_same_symbol_across_strategies is False
    assert combined.entry_priority == (BREAKOUT_KEY, GRID_KEY)
