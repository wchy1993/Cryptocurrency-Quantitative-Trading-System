from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from crypto_scalper.combined_hybrid_v5_grid_v3_20x_backtest import (
    NOTIONAL_CAP_MULTIPLE,
    build_20x_configs,
    build_parser,
)
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


def test_20x_variant_changes_only_notional_capacity() -> None:
    frozen = build_frozen_configs(BREAKOUT_CONFIG, GRID_CONFIG)
    variant = build_20x_configs(BREAKOUT_CONFIG, GRID_CONFIG)

    assert variant["breakout_portfolio"].max_notional_multiple == NOTIONAL_CAP_MULTIPLE
    assert variant["grid_portfolio"].max_notional_multiple == NOTIONAL_CAP_MULTIPLE
    assert variant["combined"].max_gross_notional_multiple == NOTIONAL_CAP_MULTIPLE

    assert asdict(variant["breakout_portfolio"]) == {
        **asdict(frozen["breakout_portfolio"]),
        "max_notional_multiple": NOTIONAL_CAP_MULTIPLE,
    }
    assert asdict(variant["grid_portfolio"]) == {
        **asdict(frozen["grid_portfolio"]),
        "max_notional_multiple": NOTIONAL_CAP_MULTIPLE,
    }
    assert asdict(variant["combined"]) == {
        **asdict(frozen["combined"]),
        "max_gross_notional_multiple": NOTIONAL_CAP_MULTIPLE,
    }

    for key in (
        "breakout_build_signal",
        "breakout_signal",
        "breakout_regime",
        "breakout_exit",
        "grid_signal",
        "grid_overlay",
    ):
        assert variant[key] == frozen[key]


def test_20x_variant_preserves_max_two_and_risk_budgets() -> None:
    variant = build_20x_configs(BREAKOUT_CONFIG, GRID_CONFIG)

    assert variant["breakout_portfolio"].risk_per_trade_pct == 0.025
    assert variant["breakout_portfolio"].max_open_positions == 1
    assert variant["grid_portfolio"].risk_per_campaign_pct == 0.10
    assert variant["grid_portfolio"].max_open_campaigns == 1
    assert variant["combined"].max_open_positions == 2
    assert variant["combined"].allow_same_symbol_across_strategies is False
    assert variant["combined"].entry_priority == (BREAKOUT_KEY, GRID_KEY)


def test_20x_parser_uses_independent_output_paths() -> None:
    args = build_parser().parse_args([])

    assert "20x" in args.output
    assert "20x" in args.summary
    assert "20x" in args.config_output
    assert "20x" in args.manifest
