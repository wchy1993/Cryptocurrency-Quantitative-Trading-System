from __future__ import annotations

import argparse

from .trend_grid_v6_optimize import (
    _strict_improvement,
    build_parser as build_v6_parser,
)
from .trend_grid_v6_optimize import run as run_v6_search
from .trend_grid_v7 import (
    TREND_GRID_V7_NAME,
    cycle_profit_floor_drawdown_variants,
    identity_allocation_variants,
)


def _strict_v7_improvement(
    six: dict,
    three: dict,
    base_six: dict,
    base_three: dict,
) -> bool:
    return (
        _strict_improvement(six, three, base_six, base_three)
        and six["max_drawdown_pct"] < base_six["max_drawdown_pct"]
        and three["max_drawdown_pct"] < base_three["max_drawdown_pct"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = build_v6_parser()
    parser.description = "Grid v7 attributed drawdown refinement"
    parser.set_defaults(
        anchor_config="config.trend-grid.v6-profit-protected-50.json",
        refine_from_config="config.trend-grid.v6-profit-protected-50.json",
        seed=20260730,
        structural_budget=31,
        recent_finalists=24,
        structural_finalists=4,
        allocation_budget=1,
        allocation_recent_finalists=4,
        robust_finalists=4,
        output=(
            "reports/trend_grid_v7_"
            "drawdown_refined_3m_6m.json"
        ),
        summary=(
            "reports/trend_grid_v7_"
            "drawdown_refined_3m_6m.md"
        ),
        config_output=(
            "config.trend-grid.v7-drawdown-refined-50.json"
        ),
        manifest=(
            "config.trend-grid."
            "v7-drawdown-refined-50-manifest.json"
        ),
    )
    return parser


def run(args: argparse.Namespace):
    return run_v6_search(
        args,
        strategy_name=TREND_GRID_V7_NAME,
        strategy_label="Grid v7 drawdown refined",
        result_version="v7_cycle_profit_floor_drawdown_refined",
        structural_variant_builder=cycle_profit_floor_drawdown_variants,
        allocation_variant_builder=identity_allocation_variants,
        strict_improvement_predicate=_strict_v7_improvement,
        baseline_strategy_label="Grid v6 frozen",
        improvement_reference_slug="grid_v6",
        manifest_source_files=(
            "crypto_scalper/trend_grid.py",
            "crypto_scalper/trend_grid_optimize.py",
            "crypto_scalper/trend_grid_v5_engine.py",
            "crypto_scalper/trend_grid_v6.py",
            "crypto_scalper/trend_grid_v6_optimize.py",
            "crypto_scalper/trend_grid_v7.py",
            "crypto_scalper/trend_grid_v7_drawdown_refine.py",
        ),
    )


if __name__ == "__main__":
    run(build_parser().parse_args())
