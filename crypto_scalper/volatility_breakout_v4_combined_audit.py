from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_v4_backtest import (
    simulate_combined_v4_portfolio,
)
from .data import parse_timestamp
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import GridPortfolioConfig, build_grid_research_timeline
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import PortfolioSearchConfig, UNIVERSE_50, build_candidates, sha256_file
from .volatility_breakout_v4_research import (
    V4RegimeConfig,
    V4Variant,
    _deduplicate_variants,
    _json_safe,
    _write_json,
    build_v4_families,
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
    result_quality,
)


AUDIT_NAME = "volatility_breakout_v4_shared_max2_finalist_audit"


def _variant_from_public(payload: dict[str, Any]) -> V4Variant:
    return V4Variant(
        name=payload["name"],
        family_key=payload["family_key"],
        signal=VolatilityBreakoutConfig(**payload["signal"]),
        portfolio=PortfolioSearchConfig(**payload["portfolio"]),
        regime=V4RegimeConfig(**payload["regime"]),
        exit=ExitProtectionConfig(**payload["exit"]),
    )


def _compact_combined(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "candidate_count",
            "trade_count",
            "initial_equity",
            "final_equity",
            "net_profit",
            "return_pct",
            "win_rate",
            "profit_factor",
            "max_drawdown_pct",
            "fee",
            "slippage",
            "funding",
            "full_cost",
            "top5_profit_contribution",
            "hard_drawdown_stopped",
            "hard_drawdown_stopped_at",
            "max_concurrent_positions",
            "max_entry_committed_notional_multiple",
            "by_strategy",
            "by_month",
        )
    }


def _strict_combined(row: dict[str, Any]) -> bool:
    result = row["result"]
    quality = row["fold_quality"]
    return (
        result["trade_count"] >= 60
        and not result["hard_drawdown_stopped"]
        and result["profit_factor"] > 1.10
        and result["max_drawdown_pct"] <= 0.40
        and quality["all_folds_positive"]
        and result["top5_profit_contribution"] <= 0.80
    )


def _public(row: dict[str, Any]) -> dict[str, Any]:
    variant: V4Variant = row["variant"]
    return {
        "name": variant.name,
        "family_key": variant.family_key,
        "signal": variant.signal.as_dict(),
        "portfolio": asdict(variant.portfolio),
        "regime": asdict(variant.regime),
        "exit": asdict(variant.exit),
        "result": _compact_combined(row["result"]),
        "fold_quality": row["fold_quality"],
        "strict_combined_eligible": _strict_combined(row),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
    start = parse_timestamp(source["period"]["start"])
    end = parse_timestamp(source["period"]["end"])
    symbols = tuple(UNIVERSE_50)
    folds = (
        (start, start + timedelta(days=30)),
        (start + timedelta(days=30), start + timedelta(days=61)),
        (start + timedelta(days=61), end),
    )
    public_rows: list[dict[str, Any]] = []
    for rows in source["search_leaderboard"].values():
        public_rows.extend(rows)
    public_rows.extend(
        (
            source["finalists"]["historical_profit_champion"]["variant"],
            source["finalists"]["robust_candidate"]["variant"],
        )
    )
    variants = _deduplicate_variants(_variant_from_public(row) for row in public_rows)
    print(f"combined audit finalists={len(variants)}", flush=True)

    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols,
        source["data"]["one_minute_roots"],
        source["data"]["funding_roots"],
        source["cost_model"]["source_config"],
        start,
        end,
    )
    context = build_v4_market_context(symbols, signal_data)
    families = build_v4_families()
    raw_by_family = {}
    for key in sorted({variant.family_key for variant in variants}):
        raw = build_candidates(
            symbols,
            signal_data,
            execution_data,
            families[key],
            start,
            end,
        )
        raw_by_family[key] = enrich_candidates_v4(raw, context)
        print(f"combined audit raw {key}", flush=True)

    grid_path = Path(args.grid_config)
    grid_hash_before = sha256_file(grid_path)
    grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_signal = TrendGridConfig(
        **grid_payload.get("validation_selected_signal", grid_payload["signal"])
    )
    grid_portfolio = GridPortfolioConfig(
        **grid_payload.get("validation_selected_portfolio", grid_payload["portfolio"])
    )
    grid_candidates, grid_snapshots = build_grid_research_timeline(
        symbols,
        signal_data,
        execution_data,
        grid_signal,
        start,
        end,
    )
    combined_config = CombinedPortfolioConfig(
        max_open_positions=2,
        max_gross_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60,
        allow_same_symbol_across_strategies=False,
        entry_priority=(BREAKOUT_KEY, GRID_KEY),
    )
    rows: list[dict[str, Any]] = []
    for number, variant in enumerate(variants, start=1):
        candidates = filter_candidates_v4(
            raw_by_family[variant.family_key],
            variant.signal,
            variant.regime,
            context,
        )
        result = simulate_combined_v4_portfolio(
            candidates,
            grid_candidates,
            grid_snapshots,
            symbols,
            execution_data,
            rules,
            variant.signal,
            variant.portfolio,
            variant.exit,
            grid_signal,
            grid_portfolio,
            combined_config,
            execution,
            start,
            end,
            source["initial_equity"],
        )
        rows.append(
            {
                "variant": variant,
                "result": result,
                "fold_quality": result_quality(result, folds),
            }
        )
        if number == 1 or number % 10 == 0 or number == len(variants):
            print(
                f"combined audit {number}/{len(variants)} {variant.name} "
                f"net={result['net_profit']:+.2f} pf={result['profit_factor']:.3f} "
                f"dd={result['max_drawdown_pct']:.2%}",
                flush=True,
            )

    unconstrained = max(rows, key=lambda row: row["result"]["net_profit"])
    no_hard_pool = [row for row in rows if not row["result"]["hard_drawdown_stopped"]]
    no_hard = max(no_hard_pool or rows, key=lambda row: row["result"]["net_profit"])
    strict_pool = [row for row in rows if _strict_combined(row)]
    robust = max(
        strict_pool or no_hard_pool or rows,
        key=lambda row: (
            row["result"]["net_profit"],
            row["fold_quality"]["minimum_fold_net_profit"],
            row["result"]["profit_factor"],
        ),
    )
    grid_hash_after = sha256_file(grid_path)
    leaderboard = sorted(rows, key=lambda row: row["result"]["net_profit"], reverse=True)
    report = {
        "strategy_name": AUDIT_NAME,
        "research_status": "historical_shared_account_finalist_rerank_not_future_oos",
        "source_report": args.source_report,
        "source_report_sha256": sha256_file(args.source_report),
        "period": source["period"],
        "initial_equity": source["initial_equity"],
        "finalist_count": len(rows),
        "strict_combined_pool_count": len(strict_pool),
        "execution_rules": source["execution_rules"],
        "coverage": {
            "minimum_ratio": metadata["minimum_coverage_ratio"],
            "maximum_missing_minutes": metadata["maximum_missing_minutes"],
            "funding_missing_symbols": metadata["funding_missing_symbols"],
        },
        "grid_freeze": {
            "path": args.grid_config,
            "sha256_before": grid_hash_before,
            "sha256_after": grid_hash_after,
            "unchanged": grid_hash_before == grid_hash_after,
        },
        "source_current_v2_combined": _compact_combined(
            source["baseline_current_v2"]["combined"]
        ),
        "maximum_net_unconstrained": _public(unconstrained),
        "maximum_net_without_hard_stop": _public(no_hard),
        "strict_robust_combined": _public(robust),
        "leaderboard": [_public(row) for row in leaderboard],
    }
    _write_json(args.output, report)
    lines = [
        "# Breakout v4 + frozen Grid shared max-2 finalist audit",
        "",
        f"- Finalists: `{len(rows)}`; strict combined pool: `{len(strict_pool)}`.",
        f"- Grid hash unchanged: `{report['grid_freeze']['unchanged']}`.",
        "",
    ]
    for label, row in (
        ("Maximum net (unconstrained)", unconstrained),
        ("Maximum net without hard stop", no_hard),
        ("Strict robust combined", robust),
    ):
        result = row["result"]
        lines.append(
            f"- {label}: `{row['variant'].name}`, `{result['net_profit']:+.2f}U`, "
            f"PF `{result['profit_factor']:.3f}`, DD `{result['max_drawdown_pct']:.2%}`, "
            f"folds `{row['fold_quality']['fold_net_profit']}`."
        )
    target = Path(args.summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rerank v4 finalists in the frozen-Grid shared account")
    parser.add_argument(
        "--source-report", default="reports/volatility_breakout_v4_regime_exit_3m.json"
    )
    parser.add_argument("--grid-config", default="config.trend-grid.v2-optimized-50.json")
    parser.add_argument(
        "--output", default="reports/volatility_breakout_v4_combined_finalist_audit_3m.json"
    )
    parser.add_argument(
        "--summary", default="reports/volatility_breakout_v4_combined_finalist_audit_3m.md"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_audit(build_parser().parse_args(argv))
    concise = {
        key: report[key]
        for key in (
            "finalist_count",
            "strict_combined_pool_count",
            "maximum_net_unconstrained",
            "maximum_net_without_hard_stop",
            "strict_robust_combined",
        )
    }
    print(json.dumps(_json_safe(concise), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
