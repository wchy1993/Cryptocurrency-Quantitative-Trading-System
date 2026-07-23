from __future__ import annotations

import json
from pathlib import Path

from crypto_scalper.combined_breakout_v8_grid_v6_backtest import (
    COMBINED_V8_GRID_V6_NAME,
    build_v8_v6_configs,
)
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)


ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return build_v8_v6_configs(
        ROOT / "config.volatility-breakout.v8-score-convex-refined-50.json",
        ROOT / "config.trend-grid.v6-profit-protected-50.json",
        ROOT / "config.combined-breakout-v7-grid-v5-max2.json",
    )


def test_v8_v6_shared_config_freezes_selected_sources_and_max_two() -> None:
    configs = _configs()

    assert configs["breakout_allocation"].score_1_short_factor == 0.5
    assert configs["breakout_allocation"].score_3_short_factor == 0.85
    assert configs["breakout_allocation"].score_5_short_factor == 2.6
    assert configs["breakout_managed_signal"].stop_atr_multiple == 0.77
    assert configs["breakout_exit"].breakeven_trigger_r == 0.0
    assert configs["grid_campaign"].minimum_actual_extension_atr == 0.05
    assert configs["grid_campaign"].maximum_campaign_risk_pct == 0.11
    assert configs["combined"].max_open_positions == 2
    assert configs["combined"].entry_priority == (BREAKOUT_KEY, GRID_KEY)


def test_v8_v6_shared_report_strictly_improves_v7_v5_baseline() -> None:
    payload = json.loads(
        (
            ROOT / "reports/combined_breakout_v8_grid_v6_max2_3m_6m.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["strategy_name"] == COMBINED_V8_GRID_V6_NAME
    for period in ("three_month", "six_month"):
        selected = payload["periods"][period]["combined"]
        baseline = payload["baseline_breakout_v7_grid_v5"][period]
        assert selected["net_profit"] > baseline["net_profit"]
        assert selected["profit_factor"] > baseline["profit_factor"]
        assert selected["max_drawdown_pct"] <= baseline["max_drawdown_pct"]
        full = payload["periods"][period]["full_combined_result"]
        assert full["max_concurrent_positions"] <= 2
        assert abs(full["pnl_reconciliation_error"]) <= 1e-6
