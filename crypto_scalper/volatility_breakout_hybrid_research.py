from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .combined_volatility_trend_grid_backtest import BREAKOUT_KEY, GRID_KEY, CombinedPortfolioConfig
from .combined_volatility_trend_grid_v4_backtest import simulate_combined_v4_portfolio
from .data import parse_timestamp
from .risk import BacktestExecutionConfig
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import GridPortfolioConfig, build_grid_research_timeline
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import Candidate, CompactSeries, PortfolioSearchConfig, UNIVERSE_50, build_candidates, compact_summary, sha256_file
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    V4RegimeConfig,
    V4Variant,
    _json_safe,
    _regime_score,
    _write_json,
    build_v4_families,
    build_v4_market_context,
    enrich_candidates_v4,
    filter_candidates_v4,
    load_v4_runtime_inputs,
    result_quality,
)


HYBRID_NAME = "dual_thrust_volatility_breakout_hybrid_adaptive_runner_v1"


@dataclass(frozen=True)
class HybridRiskProfile:
    normal_risk_pct: float
    strong_risk_pct: float
    weak_risk_pct: float
    strong_regime_score: float
    weak_regime_score: float
    strong_alignment_atr: float
    partial_fraction: float
    max_holding_minutes: int


def _snapshot_for(
    context: dict[int, dict[str, V4MarketSnapshot]], minute: int, symbol: str
) -> V4MarketSnapshot | None:
    return context.get(minute - minute % 60, {}).get(symbol)


def _selector(
    profile: HybridRiskProfile,
    base: PortfolioSearchConfig,
    context: dict[int, dict[str, V4MarketSnapshot]],
):
    def choose(candidate: Candidate, minute: int, _equity: float) -> PortfolioSearchConfig:
        snapshot = _snapshot_for(context, minute, candidate.signal.symbol)
        score = (
            _regime_score(snapshot, candidate.signal.direction)
            if snapshot is not None
            else 0.0
        )
        side = float(candidate.signal.direction.value)
        alignment = (
            side * snapshot.symbol_ema55_atr if snapshot is not None else -999.0
        )
        if score >= profile.strong_regime_score and alignment >= profile.strong_alignment_atr:
            risk = profile.strong_risk_pct
        elif score < profile.weak_regime_score:
            risk = profile.weak_risk_pct
        else:
            risk = profile.normal_risk_pct
        return replace(base, risk_per_trade_pct=risk)

    return choose


def _profile_variants(seed: int, budget: int) -> list[HybridRiskProfile]:
    rows = [
        # v3 PF leader: +588.07U / PF 2.377 / DD 32.89% on the prior 3m holdout.
        HybridRiskProfile(0.020, 0.055, 0.008, 0.75, 0.20, 1.50, 0.10, 720),
        HybridRiskProfile(0.025, 0.040, 0.012, 0.75, 0.25, 0.50, 0.25, 720),
        HybridRiskProfile(0.030, 0.044, 0.020, 0.50, 0.25, 0.50, 0.25, 960),
        HybridRiskProfile(0.025, 0.035, 0.015, 1.00, 0.40, 1.00, 0.33, 720),
        HybridRiskProfile(0.035, 0.044, 0.015, 0.75, 0.25, 0.50, 0.25, 960),
        HybridRiskProfile(0.020, 0.035, 0.010, 0.75, 0.25, 0.50, 0.25, 720),
    ]
    rng = random.Random(seed)
    while len(rows) < budget:
        rows.append(
            HybridRiskProfile(
                normal_risk_pct=rng.choice((0.020, 0.025, 0.030, 0.035, 0.040)),
                strong_risk_pct=rng.choice((0.035, 0.040, 0.044, 0.050, 0.055)),
                weak_risk_pct=rng.choice((0.008, 0.010, 0.012, 0.015, 0.020)),
                strong_regime_score=rng.choice((0.40, 0.50, 0.75, 1.00)),
                weak_regime_score=rng.choice((0.15, 0.20, 0.25, 0.40)),
                strong_alignment_atr=rng.choice((0.50, 1.00, 1.50, 2.00)),
                partial_fraction=rng.choice((0.10, 0.20, 0.25, 0.33)),
                max_holding_minutes=rng.choice((720, 960, 1440)),
            )
        )
    return rows


def _variant_public(profile: HybridRiskProfile, result: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": asdict(profile),
        "result": compact_summary(result),
        "fold_quality": quality,
    }


def _strict(row: dict[str, Any]) -> bool:
    result = row["result"]
    quality = row["quality"]
    return (
        result["trade_count"] >= 40
        and not result["hard_drawdown_stopped"]
        and result["profit_factor"] > 1.20
        and result["max_drawdown_pct"] <= 0.40
        and quality["all_folds_positive"]
    )


def run_hybrid(args: argparse.Namespace) -> dict[str, Any]:
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    symbols = tuple(UNIVERSE_50)
    folds = (
        (start, start + timedelta(days=30)),
        (start + timedelta(days=30), start + timedelta(days=61)),
        (start + timedelta(days=61), end),
    )
    signal_data, execution_data, rules, execution, metadata = load_v4_runtime_inputs(
        symbols, args.one_minute_roots, args.funding_roots, args.cost_config, start, end
    )
    context = build_v4_market_context(symbols, signal_data)
    source = json.loads(Path(args.robust_config).read_text(encoding="utf-8"))
    robust = source["robust_candidate"]
    signal = VolatilityBreakoutConfig(**robust["signal"])
    base_portfolio = PortfolioSearchConfig(**robust["portfolio"])
    regime = V4RegimeConfig(**robust["regime"])
    family_key = f"tf{signal.timeframe_minutes}_lb{signal.lookback_days}_lk{signal.long_k:g}_sk{signal.short_k:g}"
    family = build_v4_families()[family_key]
    raw = enrich_candidates_v4(
        build_candidates(symbols, signal_data, execution_data, family, start, end), context
    )
    candidates = filter_candidates_v4(raw, signal, regime, context)
    print(f"hybrid candidates={sum(len(rows) for rows in candidates.values())}", flush=True)

    grid_path = Path(args.grid_config)
    grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_signal = TrendGridConfig(**grid_payload.get("validation_selected_signal", grid_payload["signal"]))
    grid_portfolio = GridPortfolioConfig(**grid_payload.get("validation_selected_portfolio", grid_payload["portfolio"]))
    grid_candidates, grid_snapshots = build_grid_research_timeline(
        symbols, signal_data, execution_data, grid_signal, start, end
    )
    combined_config = CombinedPortfolioConfig(
        max_open_positions=2, max_gross_notional_multiple=9.0,
        hard_drawdown_stop_pct=0.60, allow_same_symbol_across_strategies=False,
        entry_priority=(BREAKOUT_KEY, GRID_KEY),
    )
    profiles = _profile_variants(args.seed, args.profile_budget)
    rows: list[dict[str, Any]] = []
    for number, profile in enumerate(profiles, start=1):
        portfolio_selector = _selector(profile, base_portfolio, context)
        exit_config = ExitProtectionConfig(
            partial_take_profit_r=2.0,
            partial_take_profit_fraction=profile.partial_fraction,
        )
        signal_for_profile = replace(signal, max_holding_minutes=profile.max_holding_minutes)
        standalone = simulate_combined_v4_portfolio(
            candidates, {}, {}, symbols, execution_data, rules,
            signal_for_profile, base_portfolio, exit_config,
            grid_signal, grid_portfolio, combined_config, execution,
            start, end, args.initial_equity, portfolio_selector,
        )
        row = {"profile": profile, "result": standalone, "quality": result_quality(standalone, folds), "signal": signal_for_profile.as_dict(), "exit": asdict(exit_config)}
        rows.append(row)
        if number == 1 or number % 10 == 0 or number == len(profiles):
            print(f"hybrid {number}/{len(profiles)} net={standalone['net_profit']:+.2f} pf={standalone['profit_factor']:.3f} dd={standalone['max_drawdown_pct']:.2%}", flush=True)

    champion = max(rows, key=lambda row: (row["result"]["net_profit"], -row["result"]["max_drawdown_pct"]))
    strict_pool = [row for row in rows if _strict(row)]
    stable = max(strict_pool or rows, key=lambda row: (row["result"]["net_profit"], row["quality"]["minimum_fold_net_profit"]))

    def combined_for(row: dict[str, Any]) -> dict[str, Any]:
        profile = row["profile"]
        selector = _selector(profile, base_portfolio, context)
        exit_config = ExitProtectionConfig(**row["exit"])
        signal_for_profile = VolatilityBreakoutConfig(**row["signal"])
        return simulate_combined_v4_portfolio(
            candidates, grid_candidates, grid_snapshots, symbols, execution_data, rules,
            signal_for_profile, base_portfolio, exit_config, grid_signal, grid_portfolio,
            combined_config, execution, start, end, args.initial_equity, selector,
        )

    combined_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in sorted(rows, key=lambda x: x["result"]["net_profit"], reverse=True)[:args.combined_finalists]:
        combined_rows.append((row, combined_for(row)))
    combined_champion = max(combined_rows, key=lambda pair: pair[1]["net_profit"])
    combined_stable = max(
        combined_rows,
        key=lambda pair: (pair[1]["profit_factor"] > 1.1 and pair[1]["max_drawdown_pct"] <= 0.4, pair[1]["net_profit"]),
    )
    report = {
        "strategy_name": HYBRID_NAME,
        "research_status": "independent_hybrid_historical_3m_full_cost_not_live",
        "gui_modified": False,
        "grid_modified": False,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": args.initial_equity,
        "symbols": list(symbols),
        "data_coverage": {"minimum_ratio": metadata["minimum_coverage_ratio"], "maximum_missing_minutes": metadata["maximum_missing_minutes"]},
        "design": {
            "old_advantage": "60R runner, compounding, strong-trend risk tier",
            "stable_advantage": "robust structure/regime entry filters, weak-state risk reduction",
            "profit_protection": "25-33% partial exit at 2R, remainder runner",
            "adaptive_risk": "weak/normal/strong regime risk tiers",
        },
        "baseline_old_balanced": json.loads(Path(args.old_report).read_text(encoding="utf-8"))["universes"]["50"]["balanced_result"],
        "baseline_stable_v4": json.loads(Path(args.stable_report).read_text(encoding="utf-8"))["finalists"]["robust_candidate"]["standalone"],
        "profile_count": len(rows),
        "strict_profile_count": len(strict_pool),
        "hybrid_standalone_champion": {"profile": asdict(champion["profile"]), "signal": champion["signal"], "exit": champion["exit"], "result": champion["result"], "fold_quality": champion["quality"]},
        "hybrid_standalone_stable": {"profile": asdict(stable["profile"]), "signal": stable["signal"], "exit": stable["exit"], "result": stable["result"], "fold_quality": stable["quality"]},
        "hybrid_combined_champion": {"profile": asdict(combined_champion[0]["profile"]), "result": combined_champion[1], "fold_quality": result_quality(combined_champion[1], folds)},
        "hybrid_combined_stable": {"profile": asdict(combined_stable[0]["profile"]), "result": combined_stable[1], "fold_quality": result_quality(combined_stable[1], folds)},
        "combined_finalist_count": len(combined_rows),
        "grid_config_sha256": sha256_file(grid_path),
        "leaderboard": [{"profile": asdict(row["profile"]), "result": {key: row["result"].get(key) for key in ("candidate_count", "trade_count", "initial_equity", "final_equity", "net_profit", "return_pct", "win_rate", "profit_factor", "max_drawdown_pct", "max_drawdown_duration_minutes", "fee", "slippage", "funding", "full_cost", "hard_drawdown_stopped", "positive_months", "negative_months", "by_month", "by_side", "by_exit_reason")}, "fold_quality": row["quality"]} for row in sorted(rows, key=lambda x: x["result"]["net_profit"], reverse=True)],
    }
    _write_json(args.output, report)
    _write_json(args.config_output, {
        "strategy_name": HYBRID_NAME, "status": "research_not_live_gui_unchanged", "period": report["period"],
        "signal": stable["signal"], "portfolio": asdict(base_portfolio), "regime": asdict(regime),
        "exit": stable["exit"], "adaptive_risk_profile": asdict(stable["profile"]),
        "grid_unchanged": True, "grid_config_sha256": sha256_file(grid_path),
    })
    summary = Path(args.summary); summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Breakout Hybrid v1 — old runner + stable entry/regime + adaptive risk",
        "",
        f"- Old balanced standalone: `{report['baseline_old_balanced']['net_profit']:+.2f}U`, PF `{report['baseline_old_balanced']['profit_factor']:.3f}`, DD `{report['baseline_old_balanced']['max_drawdown_pct']:.2%}`.",
        f"- Stable v4 standalone: `{report['baseline_stable_v4']['net_profit']:+.2f}U`, PF `{report['baseline_stable_v4']['profit_factor']:.3f}`, DD `{report['baseline_stable_v4']['max_drawdown_pct']:.2%}`.",
        f"- Hybrid standalone champion: `{champion['result']['net_profit']:+.2f}U`, PF `{champion['result']['profit_factor']:.3f}`, DD `{champion['result']['max_drawdown_pct']:.2%}`.",
        f"- Hybrid standalone stable: `{stable['result']['net_profit']:+.2f}U`, PF `{stable['result']['profit_factor']:.3f}`, DD `{stable['result']['max_drawdown_pct']:.2%}`.",
        f"- Hybrid + frozen Grid champion: `{combined_champion[1]['net_profit']:+.2f}U`, PF `{combined_champion[1]['profit_factor']:.3f}`, DD `{combined_champion[1]['max_drawdown_pct']:.2%}`.",
        f"- Hybrid + frozen Grid stable: `{combined_stable[1]['net_profit']:+.2f}U`, PF `{combined_stable[1]['profit_factor']:.3f}`, DD `{combined_stable[1]['max_drawdown_pct']:.2%}`.",
        "",
        "该版本只生成新代码和研究配置，没有修改 GUI 或 Grid。",
    ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest an adaptive hybrid Breakout strategy")
    p.add_argument("--start", default="2026-04-19T00:00:00")
    p.add_argument("--end", default="2026-07-19T00:00:00")
    p.add_argument("--initial-equity", type=float, default=200.0)
    p.add_argument("--one-minute-roots", nargs="+", default=("data/binance_1m_3m_to_20260622_top100", "data/binance_1m_v3_exit_holdout_20260522_20260719"))
    p.add_argument("--funding-roots", nargs="+", default=("data/binance_funding_365d_top100", "data/binance_funding_v3_exit_holdout_20260612_20260719"))
    p.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    p.add_argument("--robust-config", default="config.volatility-breakout.v4-regime-exit-50.json")
    p.add_argument("--old-report", default="reports/volatility_breakout_v2_3m_30_vs_50.json")
    p.add_argument("--stable-report", default="reports/volatility_breakout_v4_regime_exit_3m.json")
    p.add_argument("--grid-config", default="config.trend-grid.v2-optimized-50.json")
    p.add_argument("--profile-budget", type=int, default=48)
    p.add_argument("--combined-finalists", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--output", default="reports/volatility_breakout_hybrid_3m.json")
    p.add_argument("--summary", default="reports/volatility_breakout_hybrid_3m.md")
    p.add_argument("--config-output", default="config.volatility-breakout.hybrid-adaptive-50.json")
    return p


def main(argv: list[str] | None = None) -> int:
    report = run_hybrid(parser().parse_args(argv))
    for key in ("hybrid_standalone_champion", "hybrid_standalone_stable", "hybrid_combined_champion", "hybrid_combined_stable"):
        x=report[key]; print(key, x["result"]["net_profit"], x["result"]["profit_factor"], x["result"]["max_drawdown_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
