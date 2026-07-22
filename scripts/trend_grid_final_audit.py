from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from crypto_scalper.trend_grid import TrendGridConfig
from crypto_scalper.trend_grid_optimize import (
    GridPortfolioConfig,
    _load_runtime_inputs,
    _scaled_execution,
    _shift_candidates,
    _write_json,
    build_grid_research_timeline,
    compact_grid_summary,
    simulate_grid_portfolio,
)
from crypto_scalper.volatility_breakout_optimize import UNIVERSE_30, UNIVERSE_50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-universe and severe-cost audit for trend grid v2")
    parser.add_argument("--signal-data-dir", default="data/binance_15m_365d_top100")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--funding-data-dir", default="data/binance_funding_365d_top100")
    parser.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    parser.add_argument("--start", default="2026-03-06T00:00:00")
    parser.add_argument("--end", default="2026-06-06T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--report", default="reports/trend_grid_v2_3m_30_vs_50.json")
    parser.add_argument("--output", default="reports/trend_grid_v2_final_audit.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start, end, signal_data, execution_data, rules, execution, _ = _load_runtime_inputs(args)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    configs = {
        size: (
            TrendGridConfig(**report["universes"][size]["selected_signal_config"]),
            GridPortfolioConfig(**report["universes"][size]["selected_portfolio_config"]),
        )
        for size in ("30", "50")
    }
    universes = {"30": UNIVERSE_30, "50": UNIVERSE_50}
    results: dict[str, Any] = {}
    timelines: dict[tuple[str, str], Any] = {}
    for config_size, (signal, portfolio) in configs.items():
        for universe_size, universe in universes.items():
            key = (config_size, universe_size)
            candidates, snapshots = build_grid_research_timeline(
                universe, signal_data, execution_data, signal, start, end
            )
            timelines[key] = (candidates, snapshots)
            result = simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules,
                signal, portfolio, execution, start, end, args.initial_equity,
            )
            results[f"config{config_size}_universe{universe_size}"] = compact_grid_summary(result)
            print(
                f"config{config_size}/universe{universe_size}: "
                f"trades={result['trade_count']} net={result['net_profit']:.2f} "
                f"pf={result['profit_factor']:.3f} dd={result['max_drawdown_pct']:.2%}",
                flush=True,
            )

    stress: dict[str, Any] = {}
    for size, universe in universes.items():
        signal, portfolio = configs[size]
        candidates, snapshots = timelines[(size, size)]
        for label, stressed_candidates, stressed_execution in (
            ("cost_2.0x", candidates, _scaled_execution(execution, 2.0)),
            ("entry_delay_5m", _shift_candidates(candidates, 5, execution_data), execution),
        ):
            result = simulate_grid_portfolio(
                stressed_candidates, snapshots, universe, execution_data, rules,
                signal, portfolio, stressed_execution, start, end, args.initial_equity,
            )
            stress[f"{size}_{label}"] = compact_grid_summary(result)
    payload = {
        "strategy_name": "dynamic_trend_following_grid_v2_final_audit",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "cross_universe": results,
        "stress": stress,
    }
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
