from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_volatility_trend_grid_backtest import BREAKOUT_KEY, GRID_KEY, CombinedPortfolioConfig
from .combined_volatility_trend_grid_v4_backtest import simulate_combined_v4_portfolio
from .data import parse_timestamp
from .models import Direction
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import GridPortfolioConfig
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_hybrid_research import HybridRiskProfile, _selector
from .volatility_breakout_optimize import Candidate, PortfolioSearchConfig, UNIVERSE_50, build_candidates, minute_datetime
from .volatility_breakout_v4_research import (
    V4RegimeConfig,
    _json_safe,
    build_v4_families,
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
)


STRATEGY_NAME = "dual_thrust_volatility_breakout_hybrid_v5_expansion_runner"


@dataclass(frozen=True)
class ExpansionEntry:
    family_key: str
    include_stable_lane: bool
    min_alignment: float
    max_alignment: float
    max_body: float
    max_volume: float
    max_range: float
    max_extension: float
    min_close_position: float
    min_market_efficiency: float
    min_ema55_alignment: float
    max_ema55_alignment: float
    min_regime_score: float
    max_directional_btc_return_4h: float


@dataclass(frozen=True)
class RunnerRiskExit:
    expansion_risk_pct: float
    expansion_long_multiplier: float
    stop_atr_multiple: float
    partial_take_profit_r: float
    partial_fraction: float
    max_holding_minutes: int
    ranking_mode: str


def _entry_variants(seed: int, budget: int) -> list[ExpansionEntry]:
    rows = [
        # Exact old-style permissive lane, retained as a historical reference.
        ExpansionEntry("tf60_lb5_lk0.7_sk0.6", False, -999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 0.0, 0.0, -999.0, 999.0, -999.0, 0.02),
        # Stable core plus a broad expansion lane capable of admitting B/APE/HYPE/XLM.
        ExpansionEntry("tf60_lb5_lk0.7_sk0.6", True, 1.0, 6.0, 5.0, 25.0, 10.0, 3.0, 0.50, 0.10, 0.5, 6.0, 0.25, 0.02),
        ExpansionEntry("tf60_lb5_lk0.7_sk0.6", True, 1.5, 6.0, 5.0, 25.0, 10.0, 3.0, 0.50, 0.10, 0.5, 6.0, 0.40, 0.02),
        ExpansionEntry("tf60_lb5_lk0.7_sk0.6", True, 2.0, 6.0, 5.0, 25.0, 10.0, 3.0, 0.50, 0.10, 0.5, 6.0, 0.50, 0.02),
        ExpansionEntry("tf60_lb5_lk0.6_sk0.6", True, 1.0, 6.0, 5.0, 25.0, 10.0, 3.0, 0.50, 0.10, 0.5, 6.0, 0.25, 0.02),
    ]
    rng = random.Random(seed)
    while len(rows) < budget:
        rows.append(ExpansionEntry(
            family_key=rng.choice(("tf60_lb5_lk0.7_sk0.6", "tf60_lb5_lk0.6_sk0.6")),
            include_stable_lane=True,
            min_alignment=rng.choice((0.5, 1.0, 1.5, 2.0, 2.5)),
            max_alignment=rng.choice((5.0, 6.0, 8.0, 999.0)),
            max_body=rng.choice((3.0, 5.0, 8.0, 999.0)),
            max_volume=rng.choice((8.0, 12.0, 25.0, 999.0)),
            max_range=rng.choice((8.0, 10.0, 12.0, 999.0)),
            max_extension=rng.choice((1.5, 3.0, 5.0, 999.0)),
            min_close_position=rng.choice((0.45, 0.50, 0.55, 0.60)),
            min_market_efficiency=rng.choice((0.05, 0.08, 0.10, 0.15)),
            min_ema55_alignment=rng.choice((0.0, 0.5, 0.75, 1.0)),
            max_ema55_alignment=rng.choice((4.0, 6.0, 8.0, 999.0)),
            min_regime_score=rng.choice((0.15, 0.25, 0.40, 0.50, 0.75)),
            max_directional_btc_return_4h=rng.choice((0.02, 0.03, 999.0)),
        ))
    return rows[:budget]


def _runner_variants(seed: int, budget: int) -> list[RunnerRiskExit]:
    rows = [
        RunnerRiskExit(0.040, 1.0, 0.75, 0.0, 0.0, 960, "quality_desc"),
        RunnerRiskExit(0.044, 1.0, 0.75, 3.5, 0.05, 960, "quality_desc"),
        RunnerRiskExit(0.044, 1.0, 0.75, 3.5, 0.10, 960, "quality_desc"),
        RunnerRiskExit(0.040, 1.3, 1.0, 3.5, 0.10, 960, "trend_alignment_desc"),
        RunnerRiskExit(0.035, 1.0, 1.0, 5.0, 0.05, 1440, "quality_desc"),
        RunnerRiskExit(0.025, 1.0, 1.0, 8.0, 0.05, 960, "quality_desc"),
        RunnerRiskExit(0.030, 1.0, 1.0, 8.0, 0.05, 960, "trend_alignment_desc"),
    ]
    rng = random.Random(seed)
    while len(rows) < budget:
        partial_fraction = rng.choice((0.0, 0.05, 0.10, 0.15))
        rows.append(RunnerRiskExit(
            expansion_risk_pct=rng.choice((0.020, 0.025, 0.030, 0.035, 0.040, 0.044, 0.050, 0.055)),
            expansion_long_multiplier=rng.choice((1.0, 1.15, 1.30)),
            stop_atr_multiple=rng.choice((0.75, 1.0, 1.25)),
            partial_take_profit_r=0.0 if partial_fraction == 0.0 else rng.choice((3.5, 5.0, 8.0)),
            partial_fraction=partial_fraction,
            max_holding_minutes=rng.choice((720, 960, 1440, 2160, 2880)),
            ranking_mode=rng.choice(("quality_desc", "trend_alignment_desc")),
        ))
    return rows[:budget]


def _union_with_daily_cap(
    stable: dict[int, list[Candidate]],
    expansion: dict[int, list[Candidate]],
    cap: int = 2,
) -> tuple[dict[int, list[Candidate]], set[str]]:
    stable_ids = {row.signal.event_id for rows in stable.values() for row in rows}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    output: dict[int, list[Candidate]] = {}
    expansion_only: set[str] = set()
    for minute in sorted(set(stable) | set(expansion)):
        unique: dict[tuple[str, Direction], Candidate] = {}
        for row in expansion.get(minute, ()):
            unique[(row.signal.symbol, row.signal.direction)] = row
        for row in stable.get(minute, ()):
            unique[(row.signal.symbol, row.signal.direction)] = row
        selected: list[Candidate] = []
        for row in sorted(unique.values(), key=lambda item: (-item.signal.quality_score, item.signal.symbol)):
            key = (row.signal.symbol, minute_datetime(minute).date().isoformat())
            if counts[key] >= cap:
                continue
            counts[key] += 1
            selected.append(row)
            if row.signal.event_id not in stable_ids:
                expansion_only.add(row.signal.event_id)
        if selected:
            output[minute] = selected
    return output, expansion_only


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    trades = result["trades"]
    winners = sorted((float(t["net_pnl"]) for t in trades if float(t["net_pnl"]) > 0.0), reverse=True)
    output = {
        key: result.get(key) for key in (
            "candidate_count", "trade_count", "initial_equity", "final_equity", "net_profit", "return_pct",
            "win_rate", "profit_factor", "max_drawdown_pct", "max_drawdown_duration_minutes", "full_cost",
            "top5_profit_contribution", "hard_drawdown_stopped", "by_month", "by_strategy",
        )
    }
    output.update({
        "largest_trade_net_profit": max((float(t["net_pnl"]) for t in trades), default=0.0),
        "largest_trade_r": max((float(t["pnl_r"]) for t in trades), default=0.0),
        "five_r_winner_count": sum(float(t["pnl_r"]) >= 5.0 for t in trades),
        "top5_winner_net_profit": sum(winners[:5]),
    })
    return output


def _score(r6: dict[str, Any], r3: dict[str, Any], rold: dict[str, Any]) -> float:
    if any(r["hard_drawdown_stopped"] for r in (r6, r3, rold)):
        return -1e9
    if min(r6["trade_count"], r3["trade_count"], rold["trade_count"]) < 20:
        return -1e8
    def growth(result: dict[str, Any]) -> float:
        return math.log1p(max(0.0, float(result["net_profit"])) / 200.0)
    top5 = sorted((max(0.0, float(t["net_pnl"])) for t in rold["trades"]), reverse=True)[:5]
    convexity = math.log1p(sum(top5) / 200.0)
    return (
        2.0 * growth(r6) + 1.6 * growth(r3) + 1.8 * growth(rold)
        + 0.65 * min(float(r6["profit_factor"]), 4.0)
        + 0.55 * min(float(r3["profit_factor"]), 4.0)
        + 0.55 * min(float(rold["profit_factor"]), 4.0)
        + 0.45 * convexity
        - 2.0 * float(r6["max_drawdown_pct"])
        - 1.2 * float(r3["max_drawdown_pct"])
        - 1.0 * float(rold["max_drawdown_pct"])
    )


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    start6, start_old, end_old, start3, end = map(parse_timestamp, (args.start_6m, args.start_old, args.end_old, args.start_3m, args.end))
    symbols = tuple(UNIVERSE_50)
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols, args.one_minute_roots, args.funding_roots, args.cost_config, start6, end
    )
    if metadata["minimum_coverage_ratio"] < 0.999 or metadata["maximum_missing_minutes"]:
        raise RuntimeError("v5 requires complete 1m coverage")
    context = build_v4_market_context(symbols, signal_data)
    families = build_v4_families()
    raw_by_family: dict[str, dict[int, list[Candidate]]] = {}
    for key in ("tf60_lb5_lk0.6_sk0.6", "tf60_lb5_lk0.7_sk0.6"):
        raw_by_family[key] = enrich_candidates_v4(
            build_candidates(symbols, signal_data, execution_data, families[key], start6, end), context
        )
        print(f"{key} raw={sum(map(len, raw_by_family[key].values()))}", flush=True)

    stable_payload = json.loads(Path(args.stable_config).read_text(encoding="utf-8"))
    stable_signal = VolatilityBreakoutConfig(**stable_payload["signal"])
    stable_regime = V4RegimeConfig(**stable_payload["regime"])
    stable_filter_signal = replace(stable_signal, max_signals_per_symbol_day=24)
    stable_candidates = filter_candidates_v4(raw_by_family["tf60_lb5_lk0.6_sk0.6"], stable_filter_signal, stable_regime, context)
    stable_profile = HybridRiskProfile(
        normal_risk_pct=stable_payload["adaptive_risk"]["normal_risk_pct"],
        strong_risk_pct=stable_payload["adaptive_risk"]["strong_risk_pct"],
        weak_risk_pct=stable_payload["adaptive_risk"]["weak_risk_pct"],
        strong_regime_score=stable_payload["adaptive_risk"]["strong_regime_score"],
        weak_regime_score=stable_payload["adaptive_risk"]["weak_regime_score"],
        strong_alignment_atr=stable_payload["adaptive_risk"]["strong_alignment_atr"],
        partial_fraction=0.10,
        max_holding_minutes=720,
    )
    base_portfolio = PortfolioSearchConfig(**stable_payload["portfolio"])
    dummy_grid_signal, dummy_grid_portfolio = TrendGridConfig(), GridPortfolioConfig()
    global_config = CombinedPortfolioConfig(1, 9.0, 0.60, False, (BREAKOUT_KEY, GRID_KEY))

    entry_cache: dict[ExpansionEntry, tuple[dict[int, list[Candidate]], set[str]]] = {}
    def candidates_for(entry: ExpansionEntry):
        cached = entry_cache.get(entry)
        if cached is not None:
            return cached
        signal_filter = replace(
            families[entry.family_key],
            min_trend_alignment_atr=entry.min_alignment,
            max_trend_alignment_atr=entry.max_alignment,
            min_volume_ratio=0.0,
            max_volume_ratio=entry.max_volume,
            min_body_atr=0.0,
            max_body_atr=entry.max_body,
            min_directional_close_position=entry.min_close_position,
            min_range_atr=0.0,
            max_range_atr=entry.max_range,
            min_breakout_extension_atr=-999.0,
            max_breakout_extension_atr=entry.max_extension,
            max_signals_per_symbol_day=24,
        )
        regime = V4RegimeConfig(
            max_directional_btc_return_4h=entry.max_directional_btc_return_4h,
            min_market_efficiency_12h=entry.min_market_efficiency,
            min_directional_symbol_ema55_atr=entry.min_ema55_alignment,
            max_directional_symbol_ema55_atr=entry.max_ema55_alignment,
            min_regime_score=entry.min_regime_score,
        )
        expansion = filter_candidates_v4(raw_by_family[entry.family_key], signal_filter, regime, context)
        stable = stable_candidates if entry.include_stable_lane else {}
        cached = _union_with_daily_cap(stable, expansion)
        entry_cache[entry] = cached
        return cached

    def simulate(entry: ExpansionEntry, runner: RunnerRiskExit):
        candidates, expansion_ids = candidates_for(entry)
        signal = replace(
            stable_signal,
            stop_atr_multiple=runner.stop_atr_multiple,
            take_profit_r=60.0,
            max_holding_minutes=runner.max_holding_minutes,
        )
        portfolio = replace(base_portfolio, ranking_mode=runner.ranking_mode)
        stable_selector = _selector(stable_profile, portfolio, context)
        def select(candidate: Candidate, minute: int, equity: float) -> PortfolioSearchConfig:
            if candidate.signal.event_id in expansion_ids:
                return replace(
                    portfolio,
                    risk_per_trade_pct=runner.expansion_risk_pct,
                    long_risk_multiplier=runner.expansion_long_multiplier,
                    short_risk_multiplier=1.0,
                )
            return stable_selector(candidate, minute, equity)
        exit_config = ExitProtectionConfig(
            partial_take_profit_r=runner.partial_take_profit_r,
            partial_take_profit_fraction=runner.partial_fraction,
        )
        common = (candidates, {}, {}, symbols, execution_data, rules, signal, portfolio, exit_config,
                  dummy_grid_signal, dummy_grid_portfolio, global_config, execution)
        return (
            simulate_combined_v4_portfolio(*common, start6, end, args.initial_equity, select),
            simulate_combined_v4_portfolio(*common, start3, end, args.initial_equity, select),
            simulate_combined_v4_portfolio(*common, start_old, end_old, args.initial_equity, select),
            len(expansion_ids),
        )

    # Low-risk discovery pass prevents permissive entry structures from all tying
    # at the 60% hard stop before their signal quality can be compared.
    baseline_runner = RunnerRiskExit(0.020, 1.0, 1.0, 8.0, 0.05, 960, "quality_desc")
    stage1: list[dict[str, Any]] = []
    for number, entry in enumerate(_entry_variants(args.seed, args.entry_budget), 1):
        r6, r3, rold, expansion_count = simulate(entry, baseline_runner)
        row = {"entry": entry, "runner": baseline_runner, "r6": r6, "r3": r3, "rold": rold,
               "expansion_candidate_count": expansion_count, "score": _score(r6, r3, rold)}
        stage1.append(row)
        if number == 1 or number % 10 == 0 or number == args.entry_budget:
            print(f"entry {number}/{args.entry_budget} score={row['score']:.2f} 6m={r6['net_profit']:+.0f}/PF{r6['profit_factor']:.2f}/DD{r6['max_drawdown_pct']:.0%} 3m={r3['net_profit']:+.0f} old={rold['net_profit']:+.0f} maxR={max((t['pnl_r'] for t in rold['trades']),default=0):.1f}", flush=True)

    finalists = sorted(stage1, key=lambda x: x["score"], reverse=True)[:args.entry_finalists]
    stage2: list[dict[str, Any]] = []
    for entry_number, source in enumerate(finalists, 1):
        for runner in _runner_variants(args.seed + 2, args.runner_budget):
            r6, r3, rold, expansion_count = simulate(source["entry"], runner)
            stage2.append({"entry": source["entry"], "runner": runner, "r6": r6, "r3": r3, "rold": rold,
                           "expansion_candidate_count": expansion_count, "score": _score(r6, r3, rold)})
        print(f"runner entry {entry_number}/{len(finalists)} complete", flush=True)

    rows = [row for row in stage1 + stage2 if row["score"] > -1e8]
    rows.sort(key=lambda x: x["score"], reverse=True)
    champion = rows[0]
    conservative_pool = [row for row in rows if row["r6"]["max_drawdown_pct"] <= 0.55 and row["r3"]["max_drawdown_pct"] <= 0.45 and row["rold"]["max_drawdown_pct"] <= 0.55]
    conservative = max(conservative_pool or rows, key=lambda x: (x["rold"]["net_profit"] + x["r6"]["net_profit"] + x["r3"]["net_profit"], x["score"]))

    def public(row: dict[str, Any], full: bool = False) -> dict[str, Any]:
        result = {"score": row["score"], "entry": asdict(row["entry"]), "runner": asdict(row["runner"]),
                  "expansion_candidate_count": row["expansion_candidate_count"],
                  "six_month": _metrics(row["r6"]), "three_month": _metrics(row["r3"]), "old_reference_period": _metrics(row["rold"])}
        if full:
            result.update(six_month_full=row["r6"], three_month_full=row["r3"], old_reference_full=row["rold"])
        return result

    old_report = json.loads(Path(args.old_report).read_text(encoding="utf-8"))["universes"]["50"]["balanced_result"]
    report = {
        "strategy_name": STRATEGY_NAME,
        "status": "independent_expansion_runner_research_not_live_gui_unchanged",
        "periods": {"six_month": [start6.isoformat(), end.isoformat()], "three_month": [start3.isoformat(), end.isoformat()], "old_reference": [start_old.isoformat(), end_old.isoformat()]},
        "coverage": {"minimum_ratio": metadata["minimum_coverage_ratio"], "maximum_missing_minutes": metadata["maximum_missing_minutes"]},
        "search": {"entry_budget": args.entry_budget, "entry_finalists": args.entry_finalists, "runner_budget": args.runner_budget, "evaluations": len(stage1) + len(stage2)},
        "old_v2_reference": _metrics(old_report),
        "v4_balanced_reference": {"six_month": stable_payload["six_month_result"], "three_month": stable_payload["three_month_result"]},
        "champion": public(champion, True),
        "conservative_runner": public(conservative, True),
        "leaderboard": [public(row) for row in rows[:30]],
    }
    _write_json(args.output, report)
    _write_json(args.config_output, {
        "strategy_name": STRATEGY_NAME, "status": "research_not_live_gui_unchanged",
        "selected": "conservative_runner", "entry": asdict(conservative["entry"]), "runner": asdict(conservative["runner"]),
        "six_month": _metrics(conservative["r6"]), "three_month": _metrics(conservative["r3"]), "old_reference_period": _metrics(conservative["rold"]),
    })
    summary = Path(args.summary); summary.parent.mkdir(parents=True, exist_ok=True)
    c6, c3, co = conservative["r6"], conservative["r3"], conservative["rold"]
    summary.write_text(
        "# Hybrid v5 expansion runner\n\n"
        f"- 6m: `{c6['net_profit']:+.2f}U`, PF `{c6['profit_factor']:.3f}`, DD `{c6['max_drawdown_pct']:.2%}`.\n"
        f"- 3m: `{c3['net_profit']:+.2f}U`, PF `{c3['profit_factor']:.3f}`, DD `{c3['max_drawdown_pct']:.2%}`.\n"
        f"- Old reference window: `{co['net_profit']:+.2f}U`, PF `{co['profit_factor']:.3f}`, DD `{co['max_drawdown_pct']:.2%}`.\n"
        f"- Largest old-window trade: `{_metrics(co)['largest_trade_net_profit']:+.2f}U`, `{_metrics(co)['largest_trade_r']:.2f}R`.\n"
        "- Grid excluded; GUI and v4 unchanged.\n",
        encoding="utf-8",
    )
    print(f"selected 6m={c6['net_profit']:+.2f}/PF{c6['profit_factor']:.2f}/DD{c6['max_drawdown_pct']:.1%} 3m={c3['net_profit']:+.2f}/PF{c3['profit_factor']:.2f}/DD{c3['max_drawdown_pct']:.1%} old={co['net_profit']:+.2f} maxR={_metrics(co)['largest_trade_r']:.1f}", flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid v5 stable core plus expansion runner research")
    p.add_argument("--start-6m", default="2026-01-19T00:00:00")
    p.add_argument("--start-old", default="2026-03-06T00:00:00")
    p.add_argument("--end-old", default="2026-06-06T00:00:00")
    p.add_argument("--start-3m", default="2026-04-19T00:00:00")
    p.add_argument("--end", default="2026-07-19T00:00:00")
    p.add_argument("--initial-equity", type=float, default=200.0)
    p.add_argument("--one-minute-roots", nargs="+", default=("data/binance_1m_365d_top100", "data/binance_1m_v3_exit_holdout_20260522_20260719"))
    p.add_argument("--funding-roots", nargs="+", default=("data/binance_funding_365d_top100", "data/binance_funding_v3_exit_holdout_20260612_20260719"))
    p.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    p.add_argument("--stable-config", default="config.volatility-breakout.hybrid-v4-dual-window-balanced-50.json")
    p.add_argument("--old-report", default="reports/volatility_breakout_v2_3m_30_vs_50.json")
    p.add_argument("--entry-budget", type=int, default=36)
    p.add_argument("--entry-finalists", type=int, default=5)
    p.add_argument("--runner-budget", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260727)
    p.add_argument("--output", default="reports/volatility_breakout_hybrid_v5_expansion_runner.json")
    p.add_argument("--summary", default="reports/volatility_breakout_hybrid_v5_expansion_runner.md")
    p.add_argument("--config-output", default="config.volatility-breakout.hybrid-v5-expansion-runner-50.json")
    return p


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
