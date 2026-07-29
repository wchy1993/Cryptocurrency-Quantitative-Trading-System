from __future__ import annotations

import json
from pathlib import Path

from crypto_scalper.combined_breakout_v9_grid_v7_backtest import (
    COMBINED_V9_GRID_V7_NAME,
    build_v9_v7_configs,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v9_v7_shared_config_freezes_profit_protection_and_max_two() -> None:
    configs = build_v9_v7_configs(
        ROOT / "config.volatility-breakout.v9-shared-balanced-risk-50.json",
        ROOT / "config.trend-grid.v7-drawdown-refined-50.json",
        ROOT / "config.combined-breakout-v7-grid-v5-max2.json",
    )

    managed = configs["breakout_managed"]
    campaign = configs["grid_campaign"]
    assert managed.core_partial_r == 2.5
    assert managed.core_partial_fraction == 0.1
    assert managed.core_long_profit_floor_activation_r == 6.5
    assert campaign.cycle_profit_floor_min_take_profits == 2
    assert campaign.cycle_profit_floor_activation_r == 0.2
    assert campaign.cycle_profit_floor_r == 0.03
    assert configs["combined"].max_open_positions == 2


def test_v9_v7_shared_candidate_improves_both_windows_and_reconciles() -> None:
    payload = json.loads(
        (
            ROOT
            / "reports/"
            "combined_breakout_v9_shared_balanced_grid_v7_max2_3m_6m.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["strategy_name"] == COMBINED_V9_GRID_V7_NAME
    for period in ("three_month", "six_month"):
        selected = payload["periods"][period]["combined"]
        baseline = payload["baseline_breakout_v8_grid_v6"][period]
        assert selected["net_profit"] > baseline["net_profit"]
        assert selected["profit_factor"] > baseline["profit_factor"]
        assert selected["max_drawdown_pct"] < baseline["max_drawdown_pct"]
        assert selected["win_rate"] > baseline["win_rate"]
        full = payload["periods"][period]["full_combined_result"]
        assert full["breakout_profile_selector_enabled"] is True
        assert full["max_concurrent_positions"] <= 2
        assert abs(full["pnl_reconciliation_error"]) <= 1e-6
        assert (
            payload["periods"][period]["reverse_priority"]
            == selected
        )
