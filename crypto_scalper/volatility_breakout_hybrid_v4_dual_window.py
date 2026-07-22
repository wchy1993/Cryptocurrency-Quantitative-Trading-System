from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_volatility_trend_grid_backtest import BREAKOUT_KEY, GRID_KEY, CombinedPortfolioConfig
from .combined_volatility_trend_grid_v4_backtest import simulate_combined_v4_portfolio
from .data import parse_timestamp
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import GridPortfolioConfig
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_hybrid_research import HybridRiskProfile, _selector
from .volatility_breakout_optimize import PortfolioSearchConfig, UNIVERSE_50, build_candidates
from .volatility_breakout_v4_research import (
    V4RegimeConfig,
    _json_safe,
    build_v4_families,
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
)


STRATEGY_NAME = "dual_thrust_volatility_breakout_hybrid_v4_dual_window"


@dataclass(frozen=True)
class EntryProfile:
    min_alignment: float
    max_alignment: float
    min_body: float
    max_body: float
    min_volume: float
    max_volume: float
    min_close_position: float
    min_range: float
    max_range: float
    min_extension: float
    max_extension: float
    min_breadth_change: float
    min_market_efficiency: float
    min_ema55_alignment: float
    max_ema55_alignment: float
    min_regime_score: float
    max_regime_score: float
    max_directional_breadth: float


@dataclass(frozen=True)
class ExitRiskProfile:
    normal_risk_pct: float
    strong_risk_pct: float
    weak_risk_pct: float
    strong_regime_score: float
    weak_regime_score: float
    strong_alignment_atr: float
    partial_take_profit_r: float
    partial_fraction: float
    max_holding_minutes: int
    fail_fast_minutes: int
    fail_fast_min_mfe_r: float
    fail_fast_max_current_r: float


def _entry_variants(seed: int, budget: int) -> list[EntryProfile]:
    rows = [
        EntryProfile(0.0, 3.0, 0.0, 1.5, 0.0, 4.0, 0.50, 2.0, 6.0, -0.25, 999.0, 0.0, 0.10, 0.5, 3.0, 0.25, 1.5, 1.0),
        EntryProfile(1.5, 3.0, 0.50, 1.00, 0.0, 4.0, 0.55, 2.0, 6.0, 0.0, 0.75, 0.0, 0.10, 0.5, 3.0, 0.25, 1.5, 1.0),
        EntryProfile(1.0, 3.0, 0.40, 1.10, 0.0, 1.5, 0.55, 2.0, 6.0, 0.0, 0.75, 0.0, 0.15, 0.75, 3.0, 0.30, 1.5, 0.95),
        EntryProfile(1.5, 2.75, 0.50, 1.00, 0.0, 1.5, 0.60, 2.5, 5.5, 0.05, 0.60, 0.0, 0.15, 0.75, 2.75, 0.35, 1.25, 0.95),
    ]
    baseline = rows[0]
    # Local one-axis mutations preserve the runner set and reveal which gate removes
    # the weak January-April trades without discarding the May-July trend winners.
    axes = {
        "min_alignment": (0.25, 0.50, 0.75, 1.0, 1.25, 1.50),
        "max_alignment": (2.50, 2.60, 2.65, 2.70, 2.75, 2.80, 2.85, 2.90, 3.25),
        "min_body": (0.10, 0.20, 0.30, 0.40, 0.50),
        "max_body": (0.90, 1.00, 1.10, 1.20, 1.30, 1.40),
        "min_volume": (0.25, 0.50, 0.75, 1.00),
        "max_volume": (1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50),
        "min_close_position": (0.52, 0.55, 0.58, 0.60, 0.65),
        "min_range": (2.25, 2.50, 2.75, 3.00),
        "max_range": (4.50, 5.00, 5.25, 5.50, 5.75),
        "min_extension": (-0.10, 0.0, 0.05, 0.10, 0.15),
        "max_extension": (0.40, 0.50, 0.60, 0.75, 1.0),
        "min_breadth_change": (0.01, 0.02, 0.03, 0.05, 0.08),
        "min_market_efficiency": (0.11, 0.12, 0.15, 0.18, 0.20),
        "min_ema55_alignment": (0.60, 0.75, 0.90, 1.00, 1.25),
        "max_ema55_alignment": (2.25, 2.50, 2.75),
        "min_regime_score": (0.28, 0.30, 0.35, 0.40, 0.50),
        "max_regime_score": (1.0, 1.25, 1.75, 2.0),
        "max_directional_breadth": (0.80, 0.85, 0.90, 0.95),
    }
    for field, values in axes.items():
        for value in values:
            rows.append(replace(baseline, **{field: value}))
    rng = random.Random(seed)
    while len(rows) < budget:
        changes: dict[str, float] = {}
        for field in rng.sample(tuple(axes), rng.choice((2, 2, 3))):
            changes[field] = rng.choice(axes[field])
        row = replace(baseline, **changes)
        if row.min_alignment <= row.max_alignment and row.min_body <= row.max_body and row.min_extension <= row.max_extension:
            rows.append(row)
    return rows[:budget]


def _exit_risk_variants(seed: int, budget: int) -> list[ExitRiskProfile]:
    rows = [
        ExitRiskProfile(0.020, 0.055, 0.008, 0.75, 0.20, 1.50, 2.0, 0.10, 720, 120, 0.10, -0.50),
        ExitRiskProfile(0.018, 0.050, 0.006, 0.75, 0.20, 1.50, 2.0, 0.10, 960, 120, 0.10, -0.50),
        ExitRiskProfile(0.015, 0.044, 0.006, 0.75, 0.20, 1.50, 2.0, 0.20, 720, 120, 0.10, -0.50),
        ExitRiskProfile(0.020, 0.050, 0.006, 1.00, 0.25, 1.50, 1.5, 0.15, 1440, 180, 0.10, -0.50),
    ]
    baseline = rows[0]
    axes = {
        "normal_risk_pct": (0.012, 0.015, 0.018, 0.022, 0.025),
        "strong_risk_pct": (0.044, 0.050, 0.060, 0.065),
        "weak_risk_pct": (0.004, 0.006, 0.010, 0.012),
        "strong_regime_score": (0.50, 1.00, 1.25),
        "weak_regime_score": (0.15, 0.25, 0.40),
        "strong_alignment_atr": (1.0, 2.0, 2.5),
        "partial_take_profit_r": (2.25, 2.50, 2.75, 3.0, 3.25, 3.50, 4.0),
        "partial_fraction": (0.05, 0.075, 0.125, 0.15, 0.20),
        "max_holding_minutes": (960, 1440, 2160),
        "fail_fast_minutes": (60, 180, 240),
        "fail_fast_min_mfe_r": (0.20, 0.30),
        "fail_fast_max_current_r": (-0.60, -0.35, -0.20),
    }
    for field, values in axes.items():
        for value in values:
            rows.append(replace(baseline, **{field: value}))
    rng = random.Random(seed)
    while len(rows) < budget:
        changes: dict[str, float | int] = {}
        for field in rng.sample(tuple(axes), rng.choice((2, 2, 3))):
            changes[field] = rng.choice(axes[field])
        row = replace(baseline, **changes)
        if row.normal_risk_pct <= row.strong_risk_pct and row.weak_risk_pct <= row.normal_risk_pct:
            rows.append(row)
    return rows[:budget]


def _configs(base_signal: VolatilityBreakoutConfig, entry: EntryProfile, risk: ExitRiskProfile):
    signal = replace(
        base_signal,
        min_trend_alignment_atr=entry.min_alignment,
        max_trend_alignment_atr=entry.max_alignment,
        min_body_atr=entry.min_body,
        max_body_atr=entry.max_body,
        min_volume_ratio=entry.min_volume,
        max_volume_ratio=entry.max_volume,
        min_directional_close_position=entry.min_close_position,
        min_range_atr=entry.min_range,
        max_range_atr=entry.max_range,
        min_breakout_extension_atr=entry.min_extension,
        max_breakout_extension_atr=entry.max_extension,
        max_signals_per_symbol_day=2,
        take_profit_r=60.0,
        max_holding_minutes=risk.max_holding_minutes,
        fail_fast_minutes=risk.fail_fast_minutes,
        fail_fast_min_mfe_r=risk.fail_fast_min_mfe_r,
        fail_fast_max_current_r=risk.fail_fast_max_current_r,
    )
    regime = V4RegimeConfig(
        min_directional_breadth_change_4h=entry.min_breadth_change,
        min_market_efficiency_12h=entry.min_market_efficiency,
        min_directional_symbol_ema55_atr=entry.min_ema55_alignment,
        max_directional_symbol_ema55_atr=entry.max_ema55_alignment,
        min_regime_score=entry.min_regime_score,
        max_regime_score=entry.max_regime_score,
        max_directional_breadth=entry.max_directional_breadth,
    )
    adaptive = HybridRiskProfile(
        risk.normal_risk_pct, risk.strong_risk_pct, risk.weak_risk_pct,
        risk.strong_regime_score, risk.weak_regime_score, risk.strong_alignment_atr,
        risk.partial_fraction, risk.max_holding_minutes,
    )
    exit_config = ExitProtectionConfig(
        partial_take_profit_r=risk.partial_take_profit_r,
        partial_take_profit_fraction=risk.partial_fraction,
    )
    return signal, regime, adaptive, exit_config


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in (
        "candidate_count", "trade_count", "initial_equity", "final_equity", "net_profit",
        "return_pct", "win_rate", "profit_factor", "max_drawdown_pct",
        "max_drawdown_duration_minutes", "fee", "slippage", "funding", "full_cost",
        "top5_profit_contribution", "hard_drawdown_stopped", "by_month", "by_strategy",
    )}


def _rolling_months(result: dict[str, Any], start: datetime, end: datetime) -> list[float]:
    output: list[float] = []
    cursor = start
    while cursor < end:
        boundary = min(cursor + timedelta(days=30), end)
        output.append(sum(
            float(t["net_pnl"]) for t in result["trades"]
            if cursor <= parse_timestamp(t["exit_time"]) < boundary
        ))
        cursor = boundary
    return output


def _score(result_6m: dict[str, Any], result_3m: dict[str, Any]) -> float:
    if result_6m["hard_drawdown_stopped"] or result_3m["hard_drawdown_stopped"]:
        return -1e9
    if result_6m["trade_count"] < 25 or result_3m["trade_count"] < 18:
        return -1e8
    n6 = max(-0.95, result_6m["net_profit"] / 200.0)
    n3 = max(-0.95, result_3m["net_profit"] / 200.0)
    pf6 = min(float(result_6m["profit_factor"]), 4.0)
    pf3 = min(float(result_3m["profit_factor"]), 4.0)
    dd6 = float(result_6m["max_drawdown_pct"])
    dd3 = float(result_3m["max_drawdown_pct"])
    # Reward the weaker window heavily so one exceptional recent trend cannot hide a weak 6m result.
    balance = min(n6 / 1.6135, n3 / 2.9403)
    return 2.5 * balance + 0.8 * n6 + 0.6 * n3 + 0.9 * pf6 + 0.7 * pf3 - 2.2 * dd6 - 1.2 * dd3


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_6m = parse_timestamp(args.start_6m)
    start_3m = parse_timestamp(args.start_3m)
    end = parse_timestamp(args.end)
    symbols = tuple(UNIVERSE_50)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols, args.one_minute_roots, args.funding_roots, args.cost_config, start_6m, end
    )
    if metadata["minimum_coverage_ratio"] < 0.999 or metadata["maximum_missing_minutes"] > 0:
        raise RuntimeError(f"full coverage required: {metadata['minimum_coverage_ratio']=} {metadata['maximum_missing_minutes']=}")
    context = build_v4_market_context(symbols, signal_data)
    family = build_v4_families()["tf60_lb5_lk0.6_sk0.6"]
    raw = enrich_candidates_v4(build_candidates(symbols, signal_data, execution_data, family, start_6m, end), context)
    print(f"raw candidates={sum(map(len, raw.values()))}", flush=True)

    source = json.loads(Path(args.base_config).read_text(encoding="utf-8"))["robust_candidate"]
    base_signal = VolatilityBreakoutConfig(**source["signal"])
    base_portfolio = PortfolioSearchConfig(**source["portfolio"])
    grid_signal = TrendGridConfig()
    grid_portfolio = GridPortfolioConfig()
    global_config = CombinedPortfolioConfig(
        max_open_positions=1, max_gross_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60, allow_same_symbol_across_strategies=False,
        entry_priority=(BREAKOUT_KEY, GRID_KEY),
    )

    def simulate(entry: EntryProfile, risk: ExitRiskProfile):
        signal, regime, adaptive, exit_config = _configs(base_signal, entry, risk)
        candidates = filter_candidates_v4(raw, signal, regime, context)
        selector = _selector(adaptive, base_portfolio, context)
        common = (candidates, {}, {}, symbols, execution_data, rules, signal, base_portfolio,
                  exit_config, grid_signal, grid_portfolio, global_config, execution)
        r6 = simulate_combined_v4_portfolio(*common, start_6m, end, args.initial_equity, selector)
        r3 = simulate_combined_v4_portfolio(*common, start_3m, end, args.initial_equity, selector)
        return signal, regime, adaptive, exit_config, r6, r3

    baseline_risk = _exit_risk_variants(args.seed + 1, 1)[0]
    stage1: list[dict[str, Any]] = []
    for number, entry in enumerate(_entry_variants(args.seed, args.entry_budget), 1):
        signal, regime, adaptive, exit_config, r6, r3 = simulate(entry, baseline_risk)
        row = {"entry": entry, "risk": baseline_risk, "signal": signal, "regime": regime,
               "adaptive": adaptive, "exit": exit_config, "r6": r6, "r3": r3,
               "score": _score(r6, r3)}
        stage1.append(row)
        if number == 1 or number % 10 == 0 or number == args.entry_budget:
            print(f"entry {number}/{args.entry_budget} score={row['score']:.3f} 6m={r6['net_profit']:+.1f}/PF{r6['profit_factor']:.2f}/DD{r6['max_drawdown_pct']:.1%} 3m={r3['net_profit']:+.1f}/PF{r3['profit_factor']:.2f}/DD{r3['max_drawdown_pct']:.1%}", flush=True)

    finalists = sorted(stage1, key=lambda x: x["score"], reverse=True)[:args.entry_finalists]
    exit_rows = _exit_risk_variants(args.seed + 2, args.exit_budget)
    stage2: list[dict[str, Any]] = []
    for entry_number, source_row in enumerate(finalists, 1):
        for risk_number, risk in enumerate(exit_rows, 1):
            signal, regime, adaptive, exit_config, r6, r3 = simulate(source_row["entry"], risk)
            stage2.append({"entry": source_row["entry"], "risk": risk, "signal": signal,
                           "regime": regime, "adaptive": adaptive, "exit": exit_config,
                           "r6": r6, "r3": r3, "score": _score(r6, r3)})
        print(f"exit/risk entry {entry_number}/{len(finalists)} complete", flush=True)

    all_rows = stage1 + stage2
    eligible = [row for row in all_rows if not row["r6"]["hard_drawdown_stopped"] and not row["r3"]["hard_drawdown_stopped"]]
    champion = max(eligible, key=lambda x: x["score"])
    profit_pf_pool = [row for row in eligible if row["r6"]["net_profit"] > 322.71 and row["r3"]["net_profit"] > 588.07 and row["r6"]["profit_factor"] > 1.722 and row["r3"]["profit_factor"] > 2.377]
    profit_pf_champion = max(profit_pf_pool, key=lambda x: (x["r6"]["net_profit"] + x["r3"]["net_profit"], -x["r6"]["max_drawdown_pct"])) if profit_pf_pool else None
    pareto = []
    for row in eligible:
        dominated = any(
            other is not row
            and other["r6"]["net_profit"] >= row["r6"]["net_profit"]
            and other["r3"]["net_profit"] >= row["r3"]["net_profit"]
            and other["r6"]["profit_factor"] >= row["r6"]["profit_factor"]
            and other["r3"]["profit_factor"] >= row["r3"]["profit_factor"]
            and other["r6"]["max_drawdown_pct"] <= row["r6"]["max_drawdown_pct"]
            and other["r3"]["max_drawdown_pct"] <= row["r3"]["max_drawdown_pct"]
            for other in eligible
        )
        if not dominated:
            pareto.append(row)
    pareto.sort(key=lambda x: x["score"], reverse=True)

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        value = {
            "score": row["score"], "entry_profile": asdict(row["entry"]),
            "exit_risk_profile": asdict(row["risk"]), "signal": row["signal"].as_dict(),
            "regime": asdict(row["regime"]), "adaptive_risk": asdict(row["adaptive"]),
            "exit": asdict(row["exit"]), "six_month": _metrics(row["r6"]),
            "three_month": _metrics(row["r3"]),
            "six_month_30d_folds": _rolling_months(row["r6"], start_6m, end),
            "three_month_30d_folds": _rolling_months(row["r3"], start_3m, end),
        }
        if full:
            value["six_month_full"] = row["r6"]
            value["three_month_full"] = row["r3"]
        return value

    report = {
        "strategy_name": STRATEGY_NAME,
        "status": "independent_dual_window_historical_research_not_live_gui_unchanged",
        "periods": {"six_month": [start_6m.isoformat(), end.isoformat()], "three_month": [start_3m.isoformat(), end.isoformat()]},
        "data_coverage": {"minimum_ratio": metadata["minimum_coverage_ratio"], "maximum_missing_minutes": metadata["maximum_missing_minutes"]},
        "search": {"entry_budget": args.entry_budget, "entry_finalists": args.entry_finalists, "exit_budget": args.exit_budget, "evaluations": len(all_rows), "pareto_count": len(pareto)},
        "v3_baseline": {"six_month": {"net_profit": 322.7125153151833, "profit_factor": 1.7216896392180763, "max_drawdown_pct": 0.5878898520215038}, "three_month": {"net_profit": 588.0662213043076, "profit_factor": 2.376563938314969, "max_drawdown_pct": 0.3288706302197589}},
        "champion": public(champion, True),
        "strict_profit_pf_champion": public(profit_pf_champion, True) if profit_pf_champion else None,
        "pareto_front": [public(row) for row in pareto[:20]],
        "leaderboard": [public(row) for row in sorted(eligible, key=lambda x: x["score"], reverse=True)[:30]],
    }
    _write_json(args.output, report)
    chosen = profit_pf_champion or champion
    _write_json(args.config_output, {
        "strategy_name": STRATEGY_NAME, "status": "research_not_live_gui_unchanged",
        "selected_reason": "strict_profit_pf_champion" if profit_pf_champion else "balanced_multi_window_champion",
        "signal": chosen["signal"].as_dict(), "portfolio": asdict(base_portfolio),
        "regime": asdict(chosen["regime"]), "adaptive_risk": asdict(chosen["adaptive"]),
        "exit": asdict(chosen["exit"]), "six_month": _metrics(chosen["r6"]), "three_month": _metrics(chosen["r3"]),
    })
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    c6, c3 = champion["r6"], champion["r3"]
    lines = ["# Hybrid v4 dual-window optimization", "",
             f"- 6m: `{c6['net_profit']:+.2f}U`, PF `{c6['profit_factor']:.3f}`, DD `{c6['max_drawdown_pct']:.2%}`, trades `{c6['trade_count']}`.",
             f"- 3m: `{c3['net_profit']:+.2f}U`, PF `{c3['profit_factor']:.3f}`, DD `{c3['max_drawdown_pct']:.2%}`, trades `{c3['trade_count']}`.",
             f"- Strictly beats v3 profit+PF in both windows: `{'yes' if profit_pf_champion else 'no'}`.",
             "- GUI and v3 artifacts unchanged."]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"champion 6m={c6['net_profit']:+.2f} PF={c6['profit_factor']:.3f} DD={c6['max_drawdown_pct']:.2%}; 3m={c3['net_profit']:+.2f} PF={c3['profit_factor']:.3f} DD={c3['max_drawdown_pct']:.2%}", flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dual-window Hybrid v4 optimizer")
    p.add_argument("--start-6m", default="2026-01-19T00:00:00")
    p.add_argument("--start-3m", default="2026-04-19T00:00:00")
    p.add_argument("--end", default="2026-07-19T00:00:00")
    p.add_argument("--initial-equity", type=float, default=200.0)
    p.add_argument("--one-minute-roots", nargs="+", default=("data/binance_1m_365d_top100", "data/binance_1m_v3_exit_holdout_20260522_20260719"))
    p.add_argument("--funding-roots", nargs="+", default=("data/binance_funding_365d_top100", "data/binance_funding_v3_exit_holdout_20260612_20260719"))
    p.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    p.add_argument("--base-config", default="config.volatility-breakout.v4-regime-exit-50.json")
    p.add_argument("--entry-budget", type=int, default=48)
    p.add_argument("--entry-finalists", type=int, default=6)
    p.add_argument("--exit-budget", type=int, default=14)
    p.add_argument("--seed", type=int, default=20260723)
    p.add_argument("--output", default="reports/volatility_breakout_hybrid_v4_dual_window.json")
    p.add_argument("--summary", default="reports/volatility_breakout_hybrid_v4_dual_window.md")
    p.add_argument("--config-output", default="config.volatility-breakout.hybrid-v4-dual-window-50.json")
    return p


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
