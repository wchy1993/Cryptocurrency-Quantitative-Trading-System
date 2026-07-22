from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .mtf_structure_derivatives import (
    STRATEGY_NAME,
    STRATEGY_VERSION,
    SignalFilterConfig,
    load_feature_bundle,
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


TRAIN = (datetime(2025, 7, 1), datetime(2026, 1, 1))
VALIDATION = (datetime(2026, 1, 1), datetime(2026, 4, 1))
SEALED_TEST = (datetime(2026, 4, 1), datetime(2026, 6, 7))


def sample_signal_config(rng: random.Random) -> SignalFilterConfig:
    side_mode = rng.choices(("both", "long", "short"), weights=(8, 1, 1), k=1)[0]
    return SignalFilterConfig(
        allow_long=side_mode != "short",
        allow_short=side_mode != "long",
        max_sweep_age_bars=rng.choice((1, 2, 3, 4, 5, 6)),
        max_four_h_bos_age=rng.choice((12, 20, 30, 40, 48, 64)),
        max_one_h_structure_age=rng.choice((8, 12, 16, 20, 24, 30)),
        min_four_h_spread_atr=rng.choice((0.05, 0.10, 0.15, 0.25, 0.40, 0.60)),
        min_four_h_efficiency=rng.choice((0.05, 0.08, 0.12, 0.18, 0.25, 0.35)),
        min_four_h_slope_atr=rng.choice((-0.08, -0.03, 0.0, 0.03, 0.07, 0.12)),
        min_one_h_spread_atr=rng.choice((0.0, 0.05, 0.10, 0.20, 0.30, 0.45)),
        min_one_h_efficiency=rng.choice((0.03, 0.05, 0.08, 0.12, 0.18, 0.25)),
        min_one_h_slope_atr=rng.choice((-0.15, -0.10, -0.05, -0.02, 0.0, 0.05)),
        require_one_h_structure_alignment=rng.random() < 0.75,
        max_entry_extension_atr=rng.choice((0.50, 0.75, 1.0, 1.5, 2.0, 2.5)),
        min_volume_ratio=rng.choice((0.70, 0.80, 0.90, 1.0, 1.20, 1.50)),
        min_oi_change_30m=rng.choice((-0.03, -0.02, -0.01, -0.005, 0.0, 0.003)),
        max_oi_change_30m=rng.choice((0.02, 0.03, 0.05, 0.08, 0.12)),
        min_directional_taker_imbalance=rng.choice((-0.08, -0.05, -0.03, 0.0, 0.03, 0.06, 0.10)),
        min_directional_cvd=rng.choice((-0.08, -0.05, -0.03, 0.0, 0.03, 0.06, 0.10)),
        max_directional_funding_rate=rng.choice((0.00010, 0.00020, 0.00030, 0.00050)),
        min_quality_score=rng.choice((2.5, 3.0, 3.5, 4.0, 4.5, 5.0)),
    )


def sample_exit_config(rng: random.Random) -> ExitConfig:
    minimum = rng.choice((0.50, 0.75, 1.0, 1.25))
    maximum = rng.choice((2.0, 2.5, 3.0, 3.5, 4.0))
    maximum = max(minimum, maximum)
    tp1 = rng.choice((0.75, 1.0, 1.25, 1.5, 2.0))
    tp2 = rng.choice(tuple(value for value in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0) if value > tp1))
    tp1_fraction = rng.choice((0.35, 0.40, 0.50, 0.60))
    tp2_fraction = rng.choice(tuple(value for value in (0.15, 0.20, 0.25, 0.30, 0.35) if value + tp1_fraction < 0.95))
    return ExitConfig(
        min_stop_atr=minimum,
        max_stop_atr=maximum,
        stop_buffer_atr=rng.choice((0.10, 0.20, 0.30, 0.50, 0.75)),
        max_stop_pct=rng.choice((0.04, 0.06, 0.08, 0.10)),
        tp1_r=tp1,
        tp1_fraction=tp1_fraction,
        tp2_r=tp2,
        tp2_fraction=tp2_fraction,
        breakeven_after_tp1=rng.random() < 0.85,
        breakeven_offset_r=rng.choice((0.0, 0.05, 0.10, 0.20)),
        locked_r_after_tp2=rng.choice((0.25, 0.50, 0.75, 1.0, 1.25)),
        trailing_activation_r=rng.choice((1.5, 2.0, 2.5, 3.0, 4.0)),
        trailing_atr_multiple=rng.choice((1.25, 1.5, 2.0, 2.5, 3.0, 4.0)),
        trend_exit_mode=rng.choice(("fast", "slow", "structure", "hybrid")),
        trend_exit_confirm_bars=rng.choice((1, 2, 3, 4)),
        max_holding_minutes=rng.choice((720, 1_440, 2_880, 4_320, 7_200)),
    )


def compact_result(result: BacktestResult) -> dict[str, Any]:
    return {
        "trade_count": result.trade_count,
        "net_profit": result.net_profit,
        "return_pct": result.return_pct,
        "profit_factor": result.profit_factor,
        "win_rate_pct": result.win_rate_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "fees": result.fees,
        "funding": result.funding,
        "slippage": result.slippage,
        "positive_months": sum(row["net_profit"] > 0.0 for row in result.by_month.values()),
        "month_count": len(result.by_month),
    }


def development_score(result: BacktestResult) -> float:
    if result.trade_count < 30:
        return -1e9 + result.trade_count
    positive_ratio = (
        sum(row["net_profit"] > 0.0 for row in result.by_month.values())
        / max(1, len(result.by_month))
    )
    bounded_pf = min(3.0, result.profit_factor if math.isfinite(result.profit_factor) else 3.0)
    return (
        result.return_pct
        - 0.45 * result.max_drawdown_pct
        + 7.5 * math.log(max(0.20, bounded_pf))
        + 8.0 * positive_ratio
        + min(4.0, result.trade_count / 50.0)
    )


def cross_period_score(train: BacktestResult, validation: BacktestResult) -> float:
    if train.trade_count < 30 or validation.trade_count < 12:
        return -1e9 + train.trade_count + validation.trade_count
    weakest_return = min(train.return_pct, validation.return_pct)
    average_return = (train.return_pct + validation.return_pct) / 2.0
    worst_drawdown = max(train.max_drawdown_pct, validation.max_drawdown_pct)
    weakest_pf = min(train.profit_factor, validation.profit_factor)
    weakest_pf = min(3.0, weakest_pf if math.isfinite(weakest_pf) else 3.0)
    positive_months = sum(row["net_profit"] > 0.0 for row in train.by_month.values())
    positive_months += sum(row["net_profit"] > 0.0 for row in validation.by_month.values())
    month_count = len(train.by_month) + len(validation.by_month)
    return (
        0.75 * weakest_return
        + 0.25 * average_return
        - 0.35 * worst_drawdown
        + 8.0 * math.log(max(0.20, weakest_pf))
        + 8.0 * positive_months / max(1, month_count)
    )


def is_robust(train: BacktestResult, validation: BacktestResult) -> bool:
    return (
        train.trade_count >= 30
        and validation.trade_count >= 12
        and train.net_profit > 0.0
        and validation.net_profit > 0.0
        and train.profit_factor >= 1.03
        and validation.profit_factor >= 1.03
        and train.max_drawdown_pct <= 40.0
        and validation.max_drawdown_pct <= 40.0
    )


def run_optimization(
    trials: int,
    validation_finalists: int,
    exact_finalists: int,
    seed: int,
    output_config: Path,
    output_report: Path,
    output_markdown: Path,
    output_trades: Path,
) -> dict[str, Any]:
    rng = random.Random(seed)
    print("loading closed-bar 4H/1H/15M features and derivatives data", flush=True)
    bundle = load_feature_bundle()
    atr_timelines = build_atr_timelines(bundle)
    rules = inferred_rules(bundle)
    execution = default_execution_config(bundle)
    portfolio = PortfolioConfig()
    series_15m = {
        "train": build_15m_execution_series(bundle, *TRAIN),
        "validation": build_15m_execution_series(bundle, *VALIDATION),
    }
    train_trials: list[dict[str, Any]] = []
    attempted = 0
    while attempted < trials:
        attempted += 1
        signal = sample_signal_config(rng)
        raw_count = len(select_setups(bundle.raw_setups, signal, *TRAIN))
        if raw_count < 38 or raw_count > 800:
            if attempted % 100 == 0:
                print(f"search {attempted}/{trials}: eligible={len(train_trials)}", flush=True)
            continue
        exits = sample_exit_config(rng)
        result = run_backtest(
            bundle,
            signal,
            exits,
            portfolio,
            *TRAIN,
            resolution="15m",
            execution_series=series_15m["train"],
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        train_trials.append(
            {
                "signal": signal,
                "exit": exits,
                "raw_setup_count": raw_count,
                "train": result,
                "train_score": development_score(result),
            }
        )
        if attempted % 100 == 0:
            best = max(item["train_score"] for item in train_trials)
            print(
                f"search {attempted}/{trials}: eligible={len(train_trials)} best_train_score={best:.3f}",
                flush=True,
            )
    if not train_trials:
        raise RuntimeError("no optimization trial met the minimum setup-count constraint")
    train_trials.sort(key=lambda item: item["train_score"], reverse=True)
    shortlist = train_trials[: min(validation_finalists, len(train_trials))]
    print(f"validating {len(shortlist)} train-only finalists", flush=True)
    for index, item in enumerate(shortlist, 1):
        validation = run_backtest(
            bundle,
            item["signal"],
            item["exit"],
            portfolio,
            *VALIDATION,
            resolution="15m",
            execution_series=series_15m["validation"],
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        item["validation"] = validation
        item["cross_score"] = cross_period_score(item["train"], validation)
        item["robust"] = is_robust(item["train"], validation)
        if index % 20 == 0:
            print(f"validation {index}/{len(shortlist)}", flush=True)
    robust_15m = [item for item in shortlist if item["robust"]]
    ranked = robust_15m or shortlist
    ranked.sort(key=lambda item: item["cross_score"], reverse=True)
    finalists = ranked[: min(exact_finalists, len(ranked))]
    print(
        f"15m robust finalists={len(robust_15m)}; loading 1m data for {len(finalists)} exact checks",
        flush=True,
    )
    one_minute = load_1m_execution_series(bundle.symbols, TRAIN[0], VALIDATION[1])
    exact_rows: list[dict[str, Any]] = []
    for index, item in enumerate(finalists, 1):
        exact_train = run_backtest(
            bundle,
            item["signal"],
            item["exit"],
            portfolio,
            *TRAIN,
            resolution="1m",
            execution_series=one_minute,
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        exact_validation = run_backtest(
            bundle,
            item["signal"],
            item["exit"],
            portfolio,
            *VALIDATION,
            resolution="1m",
            execution_series=one_minute,
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        exact_rows.append(
            {
                **item,
                "exact_train": exact_train,
                "exact_validation": exact_validation,
                "exact_score": cross_period_score(exact_train, exact_validation),
                "exact_robust": is_robust(exact_train, exact_validation),
            }
        )
        print(
            f"exact {index}/{len(finalists)} train={exact_train.return_pct:.2f}% "
            f"validation={exact_validation.return_pct:.2f}%",
            flush=True,
        )
    exact_robust = [item for item in exact_rows if item["exact_robust"]]
    exact_ranked = exact_robust or exact_rows
    exact_ranked.sort(key=lambda item: item["exact_score"], reverse=True)
    chosen = exact_ranked[0]

    print("sizing risk on development periods only", flush=True)
    risk_rows: list[dict[str, Any]] = []
    for risk in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
        sized = replace(portfolio, risk_per_trade_pct=risk)
        train_result = run_backtest(
            bundle,
            chosen["signal"],
            chosen["exit"],
            sized,
            *TRAIN,
            resolution="1m",
            execution_series=one_minute,
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        validation_result = run_backtest(
            bundle,
            chosen["signal"],
            chosen["exit"],
            sized,
            *VALIDATION,
            resolution="1m",
            execution_series=one_minute,
            execution=execution,
            atr_timelines=atr_timelines,
            rules=rules,
            include_trades=False,
        )
        risk_rows.append(
            {
                "risk": risk,
                "portfolio": sized,
                "train": train_result,
                "validation": validation_result,
                "score": cross_period_score(train_result, validation_result),
                "robust": is_robust(train_result, validation_result),
            }
        )
    eligible_risk = [
        item
        for item in risk_rows
        if item["robust"]
        and item["train"].max_drawdown_pct <= 35.0
        and item["validation"].max_drawdown_pct <= 35.0
    ]
    if eligible_risk:
        eligible_risk.sort(
            key=lambda item: (
                item["validation"].net_profit,
                item["train"].net_profit,
            ),
            reverse=True,
        )
        chosen_risk = eligible_risk[0]
    else:
        risk_rows.sort(key=lambda item: item["score"], reverse=True)
        chosen_risk = risk_rows[0]
    final_portfolio = chosen_risk["portfolio"]

    print("configuration frozen; opening sealed 2026-04-01..2026-06-07 test once", flush=True)
    test_series = load_1m_execution_series(bundle.symbols, *SEALED_TEST)
    sealed_result = run_backtest(
        bundle,
        chosen["signal"],
        chosen["exit"],
        final_portfolio,
        *SEALED_TEST,
        resolution="1m",
        execution_series=test_series,
        execution=execution,
        atr_timelines=atr_timelines,
        rules=rules,
        include_trades=True,
    )
    final_train = chosen_risk["train"]
    final_validation = chosen_risk["validation"]

    config_payload = {
        "strategy": STRATEGY_NAME,
        "version": STRATEGY_VERSION,
        "research_status": (
            "passed_initial_forward_test"
            if sealed_result.net_profit > 0.0 and sealed_result.profit_factor > 1.0
            else "rejected_initial_forward_test"
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
        },
        "selection": {
            "seed": seed,
            "attempted_trials": trials,
            "eligible_train_trials": len(train_trials),
            "validation_finalists": len(shortlist),
            "exact_1m_finalists": len(finalists),
            "sealed_test_used_for_selection": False,
        },
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "config_path": str(output_config),
        "strategy": STRATEGY_NAME,
        "version": STRATEGY_VERSION,
        "universe": list(bundle.symbols),
        "cvd_definition": bundle.cvd_definition,
        "assessment": {
            "research_status": config_payload["research_status"],
            "deployment_approved": False,
            "reason": (
                "Forward test was profitable, but this research script never auto-approves deployment."
                if sealed_result.net_profit > 0.0 and sealed_result.profit_factor > 1.0
                else "The frozen configuration lost money in the sealed forward test."
            ),
        },
        "methodology": {
            "train": [item.isoformat() for item in TRAIN],
            "validation": [item.isoformat() for item in VALIDATION],
            "sealed_test": [item.isoformat() for item in SEALED_TEST],
            "test_opened_after_configuration_frozen": True,
            "candidate_execution_resolution": "15m conservative",
            "finalist_and_final_resolution": "1m conservative",
            "same_bar_order": "open exits, entry, stop before target; only one new target tier per bar",
        },
        "search": {
            "seed": seed,
            "attempted_trials": trials,
            "eligible_train_trials": len(train_trials),
            "robust_15m_finalists": len(robust_15m),
            "robust_1m_finalists": len(exact_robust),
            "train_only_champion": _trial_payload(train_trials[0]),
            "top_validated_15m": [_trial_payload(item) for item in ranked[:10]],
            "exact_1m_finalists": [_exact_payload(item) for item in exact_ranked],
            "risk_sweep": [
                {
                    "risk_per_trade_pct": item["risk"],
                    "train": compact_result(item["train"]),
                    "validation": compact_result(item["validation"]),
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
            "train_1m": compact_result(final_train),
            "validation_1m": compact_result(final_validation),
            "sealed_test_1m": sealed_result.as_dict(include_trades=False),
        },
        "sealed_test_trades": [item.as_dict() for item in sealed_result.trades],
        "source_files": bundle.source_files,
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_trades(output_trades, sealed_result)
    output_markdown.write_text(
        _markdown_report(report, output_config, output_report, output_trades), encoding="utf-8"
    )
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2), flush=True)
    return report


def _trial_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "signal": item["signal"].as_dict(),
        "exit": item["exit"].as_dict(),
        "raw_setup_count": item["raw_setup_count"],
        "train": compact_result(item["train"]),
        "train_score": item["train_score"],
    }
    if "validation" in item:
        payload.update(
            validation=compact_result(item["validation"]),
            cross_score=item["cross_score"],
            robust=item["robust"],
        )
    return payload


def _exact_payload(item: dict[str, Any]) -> dict[str, Any]:
    payload = _trial_payload(item)
    payload.update(
        exact_train=compact_result(item["exact_train"]),
        exact_validation=compact_result(item["exact_validation"]),
        exact_score=item["exact_score"],
        exact_robust=item["exact_robust"],
    )
    return payload


def _write_trades(path: Path, result: BacktestResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.as_dict() for item in result.trades]
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("event_id,symbol,direction,entry_time,exit_time,net_pnl\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    report: dict[str, Any],
    config_path: Path,
    report_path: Path,
    trades_path: Path,
) -> str:
    selected = report["selected"]
    train = selected["train_1m"]
    validation = selected["validation_1m"]
    test = selected["sealed_test_1m"]
    return f"""# 10-Coin MTF Structure + Derivatives Strategy Research

This is an independent research strategy. It does not modify or replace the current GUI strategy.

> Research status: **{report['assessment']['research_status']}**. Deployment is not approved.

## Data and anti-leakage contract

- Fixed universe: {', '.join(report['universe'])}
- Closed-bar hierarchy: 4H regime -> 1H structure -> 15M sweep/trigger -> next 1M open execution.
- OI and taker-ratio observations are delayed by five minutes before use.
- CVD is explicitly a proxy, not raw trade CVD: {report['cvd_definition']}
- Train: {report['methodology']['train'][0]} to {report['methodology']['train'][1]}
- Validation: {report['methodology']['validation'][0]} to {report['methodology']['validation'][1]}
- Sealed test: {report['methodology']['sealed_test'][0]} to {report['methodology']['sealed_test'][1]}; opened once after configuration freeze.

## Exact 1-minute full-cost results

| Period | Trades | Net PnL (U) | Return | PF | Win rate | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Train | {train['trade_count']} | {train['net_profit']:.2f} | {train['return_pct']:.2f}% | {train['profit_factor']:.3f} | {train['win_rate_pct']:.2f}% | {train['max_drawdown_pct']:.2f}% |
| Validation | {validation['trade_count']} | {validation['net_profit']:.2f} | {validation['return_pct']:.2f}% | {validation['profit_factor']:.3f} | {validation['win_rate_pct']:.2f}% | {validation['max_drawdown_pct']:.2f}% |
| Sealed test | {test['trade_count']} | {test['net_profit']:.2f} | {test['return_pct']:.2f}% | {test['profit_factor']:.3f} | {test['win_rate_pct']:.2f}% | {test['max_drawdown_pct']:.2f}% |

Initial equity is {selected['portfolio']['initial_equity']:.2f} U. Costs include taker fees, adverse market/stop/target slippage, and historical funding.

## Search audit

- Attempted deterministic trials: {report['search']['attempted_trials']}
- Eligible training trials: {report['search']['eligible_train_trials']}
- Robust 15M finalists: {report['search']['robust_15m_finalists']}
- Robust exact-1M finalists: {report['search']['robust_1m_finalists']}
- Maximum concurrent positions: {selected['portfolio']['max_open_positions']}
- Risk per trade: {selected['portfolio']['risk_per_trade_pct'] * 100:.2f}%

Artifacts: `{config_path}`, `{report_path}`, `{trades_path}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the independent 10-coin MTF structure strategy")
    parser.add_argument("--trials", type=int, default=1_500)
    parser.add_argument("--validation-finalists", type=int, default=100)
    parser.add_argument("--exact-finalists", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config.mtf-structure-derivatives.10coin-optimized.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_optimization.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_optimization.md"),
    )
    parser.add_argument(
        "--output-trades",
        type=Path,
        default=Path("reports/mtf_structure_derivatives_10coin_sealed_test_trades.csv"),
    )
    args = parser.parse_args()
    if args.trials <= 0 or args.validation_finalists <= 0 or args.exact_finalists <= 0:
        parser.error("trial and finalist counts must be positive")
    run_optimization(
        args.trials,
        args.validation_finalists,
        args.exact_finalists,
        args.seed,
        args.output_config,
        args.output_report,
        args.output_markdown,
        args.output_trades,
    )


if __name__ == "__main__":
    main()
