from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .combined_hybrid_v5_grid_v3_backtest import _write_json
from .volatility_breakout_optimize import sha256_file
from .volatility_breakout_v6_core_runner_optimize import ManagedLaneProfile
from .volatility_breakout_v8_optimize import (
    _strict_improvement,
    build_parser as build_v8_parser,
)
from .volatility_breakout_v8_optimize import run as run_v8_search
from .volatility_breakout_v9 import VOLATILITY_BREAKOUT_V9_NAME


RISK_BALANCE_SCALE = 1.0 / 1.02


def _profile_at_scale(
    source: ManagedLaneProfile,
    scale: float,
) -> ManagedLaneProfile:
    return replace(
        source,
        core_risk_pct=source.core_risk_pct * scale,
        core_strong_risk_pct=source.core_strong_risk_pct * scale,
        runner_risk_pct=source.runner_risk_pct * scale,
        runner_strong_risk_pct=source.runner_strong_risk_pct * scale,
    )


def balanced_risk_profile(
    source: ManagedLaneProfile,
    _seed: int,
    budget: int,
) -> list[ManagedLaneProfile]:
    if budget <= 0:
        return []
    return [_profile_at_scale(source, RISK_BALANCE_SCALE)]


def build_parser() -> argparse.ArgumentParser:
    parser = build_v8_parser()
    parser.description = "Breakout v9 balanced-risk shared-account candidate"
    parser.add_argument(
        "--improvement-reference-config",
        default="config.volatility-breakout.v8-score-convex-refined-50.json",
    )
    parser.add_argument(
        "--v8-risk-buffer",
        type=float,
        default=1.0,
        help="Multiplier applied after restoring the frozen v8 risk level",
    )
    parser.set_defaults(
        anchor_config=(
            "config.volatility-breakout.v9-long-floor-refined-50.json"
        ),
        refine_from_config=(
            "config.volatility-breakout.v9-long-floor-refined-50.json"
        ),
        seed=20260731,
        allocation_budget=1,
        recent_finalists=1,
        allocation_finalists=1,
        profile_budget=1,
        profile_recent_finalists=1,
        robust_finalists=1,
        output=(
            "reports/volatility_breakout_"
            "v9_balanced_risk_3m_6m.json"
        ),
        summary=(
            "reports/volatility_breakout_"
            "v9_balanced_risk_3m_6m.md"
        ),
        config_output=(
            "config.volatility-breakout.v9-balanced-risk-50.json"
        ),
        manifest=(
            "config.volatility-breakout."
            "v9-balanced-risk-50-manifest.json"
        ),
    )
    return parser


def _metric_line(
    period: str,
    version: str,
    values: dict[str, Any],
) -> str:
    pf = values.get("profit_factor")
    pf_text = "inf" if pf is None else f"{float(pf):.3f}"
    return (
        f"| {period} | {version} | {values['trade_count']} | "
        f"{values['net_profit']:+.2f}U | {pf_text} | "
        f"{values['win_rate']:.2%} | "
        f"{values['max_drawdown_pct']:.2%} |"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.v8_risk_buffer <= 1.0:
        raise ValueError("v8 risk buffer must be in (0, 1]")
    selected_scale = RISK_BALANCE_SCALE * args.v8_risk_buffer

    def selected_risk_profile(
        source: ManagedLaneProfile,
        _seed: int,
        budget: int,
    ) -> list[ManagedLaneProfile]:
        return (
            [_profile_at_scale(source, selected_scale)]
            if budget > 0
            else []
        )

    reference = json.loads(
        Path(args.improvement_reference_config).read_text(encoding="utf-8")
    )
    reference_six = reference["results"]["six_month"]
    reference_three = reference["results"]["three_month"]

    def clears_v8(
        six: dict[str, Any],
        three: dict[str, Any],
        _anchor_six: dict[str, Any],
        _anchor_three: dict[str, Any],
    ) -> bool:
        return (
            _strict_improvement(
                six,
                three,
                reference_six,
                reference_three,
            )
            and six["max_drawdown_pct"]
            < reference_six["max_drawdown_pct"]
            and three["max_drawdown_pct"]
            < reference_three["max_drawdown_pct"]
        )

    report = run_v8_search(
        args,
        strategy_name=VOLATILITY_BREAKOUT_V9_NAME,
        strategy_label="Breakout v9 balanced risk",
        result_version="v9_profit_ladder_balanced_risk",
        profile_variant_builder=selected_risk_profile,
        strict_improvement_predicate=clears_v8,
        baseline_strategy_label="Breakout v9 high-compound anchor",
        improvement_reference_slug="breakout_v8",
        manifest_source_files=(
            "crypto_scalper/volatility_breakout_exit_protection.py",
            "crypto_scalper/volatility_breakout_v8_optimize.py",
            "crypto_scalper/volatility_breakout_v9.py",
            "crypto_scalper/volatility_breakout_v9_balanced_risk.py",
        ),
    )
    report["improvement_reference"] = {
        "strategy": "Breakout v8 frozen",
        "source_config": args.improvement_reference_config,
        "six_month": reference_six,
        "three_month": reference_three,
    }
    _write_json(args.output, report)

    candidate_config = json.loads(
        Path(args.config_output).read_text(encoding="utf-8")
    )
    candidate_config["improvement_reference"] = {
        "strategy": "Breakout v8 frozen",
        "source_config": args.improvement_reference_config,
    }
    _write_json(args.config_output, candidate_config)

    selected = report["selected"]
    lines = [
        "# Breakout v9 balanced-risk validation",
        "",
        f"- Risk scale versus high-compound v9: `{selected_scale:.6f}`",
        f"- Risk buffer versus frozen v8: `{args.v8_risk_buffer:.4f}`",
        "- Gap-free 1m execution with fees, slippage and funding",
        f"- Selection status: `{report['selection_status']}`",
        "",
        "| Period | Version | Trades | Net | PF | Win | Max DD |",
        "|---|---|---:|---:|---:|---:|---:|",
        _metric_line(
            "3 months", "Breakout v8 frozen", reference_three
        ),
        _metric_line(
            "3 months", "Breakout v9 balanced", selected["three_month"]
        ),
        _metric_line(
            "6 months", "Breakout v8 frozen", reference_six
        ),
        _metric_line(
            "6 months", "Breakout v9 balanced", selected["six_month"]
        ),
    ]
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest["hashes"] = {
        artifact: sha256_file(artifact)
        for artifact in (
            "crypto_scalper/volatility_breakout_exit_protection.py",
            "crypto_scalper/volatility_breakout_v8_optimize.py",
            "crypto_scalper/volatility_breakout_v9.py",
            "crypto_scalper/volatility_breakout_v9_balanced_risk.py",
            args.config_output,
            args.output,
            args.summary,
        )
    }
    _write_json(args.manifest, manifest)
    return report


if __name__ == "__main__":
    run(build_parser().parse_args())
