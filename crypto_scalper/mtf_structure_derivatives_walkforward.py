from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .mtf_structure_derivatives import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SignalFilterConfig,
    load_extended_feature_bundle,
    select_setups,
)
from .mtf_structure_derivatives_backtest import (
    BacktestResult,
    ExitConfig,
    PortfolioConfig,
    build_15m_execution_series,
    build_atr_timelines,
    default_execution_config,
    inferred_rules,
    load_1m_execution_series,
    run_backtest,
)
from .mtf_structure_derivatives_optimize import (
    compact_result,
    sample_exit_config,
    _write_trades,
)


FOLDS: tuple[tuple[str, datetime, datetime], ...] = (
    ("2025_q3", datetime(2025, 7, 1), datetime(2025, 10, 1)),
    ("2025_q4", datetime(2025, 10, 1), datetime(2026, 1, 1)),
    ("2026_q1", datetime(2026, 1, 1), datetime(2026, 4, 1)),
    ("2026_q2_revealed", datetime(2026, 4, 1), datetime(2026, 6, 7)),
)
DEVELOPMENT = (FOLDS[0][1], FOLDS[-1][2])
FORWARD_HOLDOUT = (datetime(2026, 6, 12), datetime(2026, 7, 19))


def sample_walkforward_signal(rng: random.Random) -> SignalFilterConfig:
    side_mode = rng.choices(("both", "long", "short"), weights=(8, 1, 1), k=1)[0]
    trigger_mode = rng.choice(
        ("all", "structural", "continuation", "reversal", "micro", "bos_micro")
    )
    allow_bos = trigger_mode in {"all", "structural", "continuation", "bos_micro"}
    allow_choch = trigger_mode in {"all", "structural", "reversal"}
    allow_micro = trigger_mode in {"all", "micro", "bos_micro"}
    return SignalFilterConfig(
        allow_long=side_mode != "short",
        allow_short=side_mode != "long",
        allow_bos_trigger=allow_bos,
        allow_choch_trigger=allow_choch,
        allow_micro_bos_trigger=allow_micro,
        max_sweep_age_bars=rng.choice((1, 2, 3, 4, 5, 6)),
        max_four_h_bos_age=rng.choice((12, 20, 30, 40, 48, 64, 80)),
        max_one_h_structure_age=rng.choice((8, 12, 16, 20, 24, 30)),
        min_four_h_spread_atr=rng.choice((0.05, 0.10, 0.15, 0.25, 0.40, 0.60)),
        min_four_h_efficiency=rng.choice((0.05, 0.08, 0.12, 0.18, 0.25, 0.35)),
        min_four_h_slope_atr=rng.choice((-0.08, -0.03, 0.0, 0.03, 0.07, 0.12)),
        require_four_h_structure_alignment=rng.random() < 0.55,
        min_one_h_spread_atr=rng.choice((0.0, 0.05, 0.10, 0.20, 0.30, 0.45)),
        min_one_h_efficiency=rng.choice((0.03, 0.05, 0.08, 0.12, 0.18, 0.25)),
        min_one_h_slope_atr=rng.choice((-0.15, -0.10, -0.05, -0.02, 0.0, 0.05)),
        require_one_h_structure_alignment=rng.random() < 0.70,
        min_entry_extension_atr=rng.choice((-3.0, -2.0, -1.5, -1.0, -0.5, 0.0)),
        max_entry_extension_atr=rng.choice((0.50, 0.75, 1.0, 1.5, 2.0, 2.5)),
        min_volume_ratio=rng.choice((0.70, 0.80, 0.90, 1.0, 1.20, 1.50)),
        max_volume_ratio=rng.choice((2.0, 2.5, 3.0, 4.0, 6.0, 999.0)),
        min_oi_change_30m=rng.choice((-0.03, -0.02, -0.01, -0.005, 0.0, 0.003)),
        max_oi_change_30m=rng.choice((0.015, 0.02, 0.03, 0.05, 0.08)),
        min_directional_taker_imbalance=rng.choice((-0.08, -0.05, -0.03, 0.0, 0.03, 0.06)),
        min_directional_cvd=rng.choice((-0.08, -0.05, -0.03, 0.0, 0.03, 0.06)),
        max_directional_funding_rate=rng.choice((0.00010, 0.00020, 0.00030, 0.00050)),
        min_quality_score=rng.choice((2.5, 3.0, 3.5, 4.0, 4.5)),
    )


def aggregate(results: dict[str, BacktestResult]) -> dict[str, float]:
    rows = list(results.values())
    gross_profit = sum(item.gross_profit for item in rows)
    gross_loss = sum(item.gross_loss for item in rows)
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else (math.inf if gross_profit > 0 else 0.0)
    return {
        "trade_count": sum(item.trade_count for item in rows),
        "net_profit": sum(item.net_profit for item in rows),
        "sum_return_pct": sum(item.return_pct for item in rows),
        "mean_return_pct": sum(item.return_pct for item in rows) / max(1, len(rows)),
        "worst_return_pct": min((item.return_pct for item in rows), default=0.0),
        "profit_factor": pf,
        "worst_drawdown_pct": max((item.max_drawdown_pct for item in rows), default=0.0),
        "positive_folds": sum(item.net_profit > 0.0 for item in rows),
        "minimum_fold_trades": min((item.trade_count for item in rows), default=0),
    }


def walkforward_score(results: dict[str, BacktestResult]) -> float:
    stats = aggregate(results)
    if stats["trade_count"] < 48 or stats["minimum_fold_trades"] < 7:
        return -1e9 + stats["trade_count"]
    bounded_pf = min(3.0, stats["profit_factor"] if math.isfinite(stats["profit_factor"]) else 3.0)
    negative_penalty = sum(min(0.0, item.return_pct) for item in results.values())
    return (
        0.35 * stats["sum_return_pct"]
        + 0.65 * stats["worst_return_pct"]
        + 0.40 * negative_penalty
        - 0.30 * stats["worst_drawdown_pct"]
        + 8.0 * math.log(max(0.20, bounded_pf))
        + 2.0 * stats["positive_folds"]
    )


def walkforward_robust(results: dict[str, BacktestResult]) -> bool:
    stats = aggregate(results)
    return (
        stats["trade_count"] >= 48
        and stats["minimum_fold_trades"] >= 7
        and stats["net_profit"] > 0.0
        and stats["profit_factor"] >= 1.05
        and stats["positive_folds"] >= 3
        and stats["worst_return_pct"] >= -8.0
        and stats["worst_drawdown_pct"] <= 35.0
    )


def evaluate_folds(
    bundle: Any,
    signal: SignalFilterConfig,
    exits: ExitConfig,
    portfolio: PortfolioConfig,
    series_by_fold: dict[str, Any],
    resolution: str,
    execution: Any,
    atr_timelines: Any,
    rules: Any,
) -> dict[str, BacktestResult]:
    return {
        name: run_backtest(
            bundle,
            signal,
            exits,
            portfolio,
            start,
            end,
            resolution=resolution,
            execution_series=series_by_fold[name],
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        for name, start, end in FOLDS
    }


def run_walkforward(
    trials: int,
    exact_finalists: int,
    seed: int,
    output_config: Path,
    output_report: Path,
    output_markdown: Path,
    output_trades: Path,
) -> dict[str, Any]:
    rng = random.Random(seed)
    print("loading extended no-gap prices and independently downloaded derivatives data", flush=True)
    bundle = load_extended_feature_bundle()
    atr_timelines = build_atr_timelines(bundle)
    rules = inferred_rules(bundle)
    execution = default_execution_config(bundle)
    base_portfolio = PortfolioConfig(risk_per_trade_pct=0.02)
    series_15m = {
        name: build_15m_execution_series(bundle, start, end)
        for name, start, end in FOLDS
    }

    rows: list[dict[str, Any]] = []
    for attempt in range(1, trials + 1):
        signal = sample_walkforward_signal(rng)
        fold_setup_counts = {
            name: len(select_setups(bundle.raw_setups, signal, start, end))
            for name, start, end in FOLDS
        }
        total_setups = sum(fold_setup_counts.values())
        if min(fold_setup_counts.values()) < 8 or not 50 <= total_setups <= 750:
            if attempt % 250 == 0:
                print(f"walk search {attempt}/{trials}: eligible={len(rows)}", flush=True)
            continue
        exits = sample_exit_config(rng)
        fold_results = evaluate_folds(
            bundle,
            signal,
            exits,
            base_portfolio,
            series_15m,
            "15m",
            execution,
            atr_timelines,
            rules,
        )
        rows.append(
            {
                "signal": signal,
                "exit": exits,
                "setup_counts": fold_setup_counts,
                "folds_15m": fold_results,
                "score_15m": walkforward_score(fold_results),
                "robust_15m": walkforward_robust(fold_results),
            }
        )
        if attempt % 250 == 0:
            best = max(item["score_15m"] for item in rows)
            robust_count = sum(item["robust_15m"] for item in rows)
            print(
                f"walk search {attempt}/{trials}: eligible={len(rows)} "
                f"robust={robust_count} best={best:.3f}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("no walk-forward candidates met setup-count constraints")
    robust_rows = [item for item in rows if item["robust_15m"]]
    ranked = robust_rows or rows
    ranked.sort(key=lambda item: item["score_15m"], reverse=True)
    finalists = ranked[: min(exact_finalists, len(ranked))]
    print(
        f"15m eligible={len(rows)} robust={len(robust_rows)}; exact-checking {len(finalists)}",
        flush=True,
    )

    development_1m = load_1m_execution_series(
        bundle.symbols,
        DEVELOPMENT[0],
        DEVELOPMENT[1],
        "data/binance_1m_365d_top100",
    )
    series_1m = {name: development_1m for name, _, _ in FOLDS}
    exact_rows: list[dict[str, Any]] = []
    for index, item in enumerate(finalists, 1):
        exact_folds = evaluate_folds(
            bundle,
            item["signal"],
            item["exit"],
            base_portfolio,
            series_1m,
            "1m",
            execution,
            atr_timelines,
            rules,
        )
        exact_rows.append(
            {
                **item,
                "folds_1m": exact_folds,
                "score_1m": walkforward_score(exact_folds),
                "robust_1m": walkforward_robust(exact_folds),
            }
        )
        stats = aggregate(exact_folds)
        print(
            f"exact {index}/{len(finalists)} net={stats['net_profit']:.2f}U "
            f"positive_folds={stats['positive_folds']:.0f}/4 worst={stats['worst_return_pct']:.2f}%",
            flush=True,
        )
    exact_robust = [item for item in exact_rows if item["robust_1m"]]
    exact_ranked = exact_robust or exact_rows
    exact_ranked.sort(key=lambda item: item["score_1m"], reverse=True)
    chosen = exact_ranked[0]

    print("choosing the highest development profit inside a 25% fold drawdown cap", flush=True)
    risk_rows: list[dict[str, Any]] = []
    for risk in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        portfolio = replace(base_portfolio, risk_per_trade_pct=risk)
        results = evaluate_folds(
            bundle,
            chosen["signal"],
            chosen["exit"],
            portfolio,
            series_1m,
            "1m",
            execution,
            atr_timelines,
            rules,
        )
        risk_rows.append(
            {
                "risk": risk,
                "portfolio": portfolio,
                "folds": results,
                "aggregate": aggregate(results),
                "score": walkforward_score(results),
                "robust": walkforward_robust(results),
            }
        )
    eligible_risk = [
        item
        for item in risk_rows
        if item["robust"] and item["aggregate"]["worst_drawdown_pct"] <= 25.0
    ]
    if eligible_risk:
        eligible_risk.sort(key=lambda item: item["aggregate"]["net_profit"], reverse=True)
        selected_risk = eligible_risk[0]
    else:
        risk_rows.sort(key=lambda item: item["score"], reverse=True)
        selected_risk = risk_rows[0]
    final_portfolio = selected_risk["portfolio"]

    development_result = run_backtest(
        bundle,
        chosen["signal"],
        chosen["exit"],
        final_portfolio,
        *DEVELOPMENT,
        resolution="1m",
        execution_series=development_1m,
        execution=execution,
        atr_timelines=atr_timelines,
        rules=rules,
        include_trades=False,
    )

    print("parameters frozen; opening the newly downloaded forward holdout exactly once", flush=True)
    holdout_1m = load_1m_execution_series(
        bundle.symbols,
        FORWARD_HOLDOUT[0],
        FORWARD_HOLDOUT[1],
        "data/binance_1m_v3_exit_holdout_20260522_20260719",
    )
    holdout_result = run_backtest(
        bundle,
        chosen["signal"],
        chosen["exit"],
        final_portfolio,
        *FORWARD_HOLDOUT,
        resolution="1m",
        execution_series=holdout_1m,
        execution=execution,
        atr_timelines=atr_timelines,
        rules=rules,
        include_trades=True,
    )

    config = {
        "strategy": STRATEGY_NAME,
        "version": f"{STRATEGY_VERSION}_walkforward_v2",
        "research_status": (
            "passed_new_forward_holdout"
            if holdout_result.net_profit > 0.0 and holdout_result.profit_factor > 1.0
            else "rejected_new_forward_holdout"
        ),
        "deployment_approved": False,
        "independent_from_gui_strategy": True,
        "universe": list(bundle.symbols),
        "timeframes": {"regime": "4h", "structure": "1h", "entry": "15m", "execution": "1m"},
        "signal": chosen["signal"].as_dict(),
        "exit": chosen["exit"].as_dict(),
        "portfolio": final_portfolio.as_dict(),
        "execution_costs": {
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
        },
        "data_contract": {
            "closed_candles_only": True,
            "entry_at_next_1m_open": True,
            "derivatives_publication_lag_minutes": 5,
            "cvd_definition": bundle.cvd_definition,
            "price_gap_count": 0,
            "derivatives_zero_fill_used": False,
        },
        "selection": {
            "seed": seed,
            "attempted_trials": trials,
            "eligible_trials": len(rows),
            "robust_15m_trials": len(robust_rows),
            "exact_1m_finalists": len(finalists),
            "robust_1m_finalists": len(exact_robust),
            "forward_holdout_used_for_selection": False,
        },
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    first_report_path = Path("reports/mtf_structure_derivatives_10coin_optimization.json")
    first_failure = None
    if first_report_path.exists():
        first = json.loads(first_report_path.read_text(encoding="utf-8"))
        first_failure = first.get("selected", {}).get("sealed_test_1m")
    report = {
        "strategy": config["strategy"],
        "version": config["version"],
        "config_path": str(output_config),
        "first_round_preserved_failure": first_failure,
        "assessment": {
            "research_status": config["research_status"],
            "deployment_approved": False,
            "reason": (
                "The new forward holdout was profitable, but research never auto-approves deployment."
                if holdout_result.net_profit > 0.0 and holdout_result.profit_factor > 1.0
                else "The frozen walk-forward configuration lost money in the newly sealed holdout."
            ),
        },
        "methodology": {
            "development_folds": {
                name: [start.isoformat(), end.isoformat()] for name, start, end in FOLDS
            },
            "forward_holdout": [item.isoformat() for item in FORWARD_HOLDOUT],
            "holdout_opened_after_configuration_freeze": True,
            "holdout_price_data_preexisting": True,
            "holdout_derivatives_downloaded_before_model_selection": True,
            "holdout_returns_inspected_before_freeze": False,
        },
        "search": {
            "seed": seed,
            "attempted_trials": trials,
            "eligible_trials": len(rows),
            "robust_15m_trials": len(robust_rows),
            "robust_1m_finalists": len(exact_robust),
            "top_15m": [_row_payload(item, False) for item in ranked[:10]],
            "exact_finalists": [_row_payload(item, True) for item in exact_ranked],
            "risk_sweep": [
                {
                    "risk_per_trade_pct": item["risk"],
                    "aggregate": item["aggregate"],
                    "folds": {name: compact_result(value) for name, value in item["folds"].items()},
                    "robust": item["robust"],
                    "score": item["score"],
                }
                for item in risk_rows
            ],
        },
        "selected": {
            "signal": chosen["signal"].as_dict(),
            "exit": chosen["exit"].as_dict(),
            "portfolio": final_portfolio.as_dict(),
            "development_fold_1m": {
                name: compact_result(value) for name, value in selected_risk["folds"].items()
            },
            "development_fold_aggregate": selected_risk["aggregate"],
            "development_contiguous_1m": development_result.as_dict(include_trades=False),
            "forward_holdout_1m": holdout_result.as_dict(include_trades=False),
        },
        "forward_holdout_trades": [item.as_dict() for item in holdout_result.trades],
        "cvd_definition": bundle.cvd_definition,
        "source_files": bundle.source_files,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_trades.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_trades(output_trades, holdout_result)
    output_markdown.write_text(
        _markdown(report, output_config, output_report, output_trades), encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2, ensure_ascii=False), flush=True)
    return report


def _row_payload(item: dict[str, Any], exact: bool) -> dict[str, Any]:
    payload = {
        "signal": item["signal"].as_dict(),
        "exit": item["exit"].as_dict(),
        "setup_counts": item["setup_counts"],
        "folds_15m": {name: compact_result(value) for name, value in item["folds_15m"].items()},
        "aggregate_15m": aggregate(item["folds_15m"]),
        "score_15m": item["score_15m"],
        "robust_15m": item["robust_15m"],
    }
    if exact:
        payload.update(
            folds_1m={name: compact_result(value) for name, value in item["folds_1m"].items()},
            aggregate_1m=aggregate(item["folds_1m"]),
            score_1m=item["score_1m"],
            robust_1m=item["robust_1m"],
        )
    return payload


def _markdown(
    report: dict[str, Any],
    config_path: Path,
    report_path: Path,
    trades_path: Path,
) -> str:
    selected = report["selected"]
    aggregate_row = selected["development_fold_aggregate"]
    holdout = selected["forward_holdout_1m"]
    fold_lines = "\n".join(
        f"| {name} | {row['trade_count']} | {row['net_profit']:.2f} | {row['return_pct']:.2f}% | "
        f"{row['profit_factor']:.3f} | {row['win_rate_pct']:.2f}% | {row['max_drawdown_pct']:.2f}% |"
        for name, row in selected["development_fold_1m"].items()
    )
    return f"""# 10-Coin MTF Structure + Derivatives Walk-Forward v2

This remains independent from the current GUI strategy. The first frozen test failure is preserved in the JSON report and was not rewritten.

> Research status: **{report['assessment']['research_status']}**. Deployment is not approved.

## Exact 1-minute development folds

| Fold | Trades | Net PnL (U) | Return | PF | Win rate | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
{fold_lines}

Fold-reset aggregate: {aggregate_row['trade_count']:.0f} trades, {aggregate_row['net_profit']:.2f} U, PF {aggregate_row['profit_factor']:.3f}, {aggregate_row['positive_folds']:.0f}/4 positive folds, worst fold {aggregate_row['worst_return_pct']:.2f}%.

## Newly sealed forward holdout

2026-06-12 through 2026-07-19 was opened once after configuration freeze: {holdout['trade_count']} trades, {holdout['net_profit']:.2f} U ({holdout['return_pct']:.2f}%), PF {holdout['profit_factor']:.3f}, win rate {holdout['win_rate_pct']:.2f}%, max drawdown {holdout['max_drawdown_pct']:.2f}%.

The CVD field is a documented taker-ratio/volume proxy, not raw trade CVD. Full-cost execution includes fees, adverse slippage and historical funding. Maximum concurrent positions: {selected['portfolio']['max_open_positions']}.

Artifacts: `{config_path}`, `{report_path}`, `{trades_path}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward optimize the independent 10-coin MTF strategy")
    parser.add_argument("--trials", type=int, default=6_000)
    parser.add_argument("--exact-finalists", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config.mtf-structure-derivatives.10coin-walkforward-v2.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_walkforward_v2.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_walkforward_v2.md"),
    )
    parser.add_argument(
        "--output-trades",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_forward_holdout_trades_v2.csv"),
    )
    args = parser.parse_args()
    if args.trials <= 0 or args.exact_finalists <= 0:
        parser.error("trials and exact-finalists must be positive")
    run_walkforward(
        args.trials,
        args.exact_finalists,
        args.seed,
        args.output_config,
        args.output_report,
        args.output_markdown,
        args.output_trades,
    )


if __name__ == "__main__":
    main()
