from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .indicators import ema
from .models import Candle, Direction
from .risk import BacktestExecutionConfig
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_optimize import (
    UNIVERSE_30,
    UNIVERSE_50,
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    _entry_cache_key,
    _load_runtime_inputs,
    _shift_candidates,
    _without_symbols,
    _write_json,
    build_candidates,
    compact_summary,
    minute_token,
    sha256_file,
    signal_candles_for_timeframe,
    simulate_portfolio,
)


V2_RESEARCH_NAME = "dual_thrust_volatility_breakout_v2"


def build_market_context(
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
) -> dict[int, dict[str, float]]:
    per_minute: dict[int, dict[str, tuple[float, bool]]] = defaultdict(dict)
    for symbol in universe:
        source = signal_data.get(symbol)
        if not source:
            continue
        candles = signal_candles_for_timeframe(source, 60)
        closes = [candle.close for candle in candles]
        ema21 = ema(closes, 21)
        for index in range(4, len(candles)):
            minute = minute_token(candles[index].timestamp + timedelta(minutes=60))
            return_4h = closes[index] / max(closes[index - 4], 1e-12) - 1.0
            per_minute[minute][symbol] = (return_4h, closes[index] > ema21[index])

    context: dict[int, dict[str, float]] = {}
    for minute, rows in per_minute.items():
        eligible = len(rows)
        if eligible < max(5, len(universe) // 2):
            continue
        context[minute] = {
            "btc_return_4h": rows.get("BTCUSDT", (math.nan, False))[0],
            "eth_return_4h": rows.get("ETHUSDT", (math.nan, False))[0],
            "breadth_above_ema21": sum(int(above) for _, above in rows.values()) / eligible,
            "eligible_symbols": float(eligible),
        }
    return context


def enrich_candidates(
    candidates: dict[int, list[Candidate]],
    context: dict[int, dict[str, float]],
) -> dict[int, list[Candidate]]:
    output: dict[int, list[Candidate]] = {}
    for minute, rows in candidates.items():
        snapshot = context.get(minute)
        if snapshot is None:
            output[minute] = list(rows)
            continue
        btc = snapshot["btc_return_4h"]
        eth = snapshot["eth_return_4h"]
        output[minute] = [
            replace(
                candidate,
                btc_return_4h=None if not math.isfinite(btc) else btc,
                eth_return_4h=None if not math.isfinite(eth) else eth,
                breadth_above_ema21=snapshot["breadth_above_ema21"],
            )
            for candidate in rows
        ]
    return output


def _passes_signal_filters(signal: Any, config: VolatilityBreakoutConfig) -> bool:
    return (
        config.min_trend_alignment_atr <= signal.trend_alignment_atr <= config.max_trend_alignment_atr
        and config.min_volume_ratio <= signal.volume_ratio <= config.max_volume_ratio
        and config.min_body_atr <= signal.body_atr <= config.max_body_atr
        and signal.directional_close_position >= config.min_directional_close_position
        and config.min_range_atr <= signal.range_atr <= config.max_range_atr
        and config.min_breakout_extension_atr
        <= signal.breakout_extension_atr
        <= config.max_breakout_extension_atr
    )


def filter_candidates(
    candidates: dict[int, list[Candidate]],
    config: VolatilityBreakoutConfig,
) -> dict[int, list[Candidate]]:
    output: dict[int, list[Candidate]] = {}
    for minute, rows in candidates.items():
        selected = [candidate for candidate in rows if _passes_signal_filters(candidate.signal, config)]
        if selected:
            output[minute] = selected
    return output


def _valid_result(result: dict[str, Any], minimum_trades: int) -> bool:
    return (
        result["trade_count"] >= minimum_trades
        and result["profit_factor"] > 1.05
        and result["max_drawdown_pct"] <= 0.55
        and not result["hard_drawdown_stopped"]
    )


def _select_best(rows: list[dict[str, Any]], minimum_trades: int) -> dict[str, Any]:
    valid = [row for row in rows if _valid_result(row["full_result"], minimum_trades)]
    pool = valid or rows
    return max(
        pool,
        key=lambda row: (
            _valid_result(row["full_result"], minimum_trades),
            row["full_result"]["net_profit"],
            row["full_result"]["profit_factor"],
            -row["full_result"]["max_drawdown_pct"],
        ),
    )


def _variant_record(
    name: str,
    signal: VolatilityBreakoutConfig,
    portfolio: PortfolioSearchConfig,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "signal_config": signal.as_dict(),
        "portfolio_config": asdict(portfolio),
        "result": compact_summary(result),
        "full_result": result,
    }


def _public_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "signal_config": row["signal_config"],
            "portfolio_config": row["portfolio_config"],
            "result": row["result"],
        }
        for row in rows
    ]


def optimize_v2_universe(
    universe: tuple[str, ...],
    frozen_signal: VolatilityBreakoutConfig,
    frozen_portfolio: PortfolioSearchConfig,
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, Any],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    frozen_result: dict[str, Any],
) -> dict[str, Any]:
    raw_candidates = build_candidates(
        universe, signal_data, execution_data, frozen_signal, start, end
    )
    context = build_market_context(universe, signal_data)
    candidates = enrich_candidates(raw_candidates, context)
    filtered_cache: dict[str, dict[int, list[Candidate]]] = {}

    def run(
        name: str,
        signal: VolatilityBreakoutConfig,
        portfolio: PortfolioSearchConfig,
        candidate_override: dict[int, list[Candidate]] | None = None,
        execution_override: BacktestExecutionConfig | None = None,
        skip_event_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        key = _entry_cache_key(signal)
        selected_candidates = candidate_override
        if selected_candidates is None:
            selected_candidates = filtered_cache.get(key)
            if selected_candidates is None:
                selected_candidates = filter_candidates(candidates, signal)
                filtered_cache[key] = selected_candidates
        result = simulate_portfolio(
            selected_candidates,
            universe,
            execution_data,
            rules,
            signal,
            portfolio,
            execution_override or execution,
            start,
            end,
            initial_equity,
            skip_event_ids=skip_event_ids,
        )
        return _variant_record(name, signal, portfolio, result)

    stages: dict[str, list[dict[str, Any]]] = {}
    baseline = run("v1_frozen", frozen_signal, frozen_portfolio)
    baseline_signature = [
        (trade["event_id"], trade["entry_time"], trade["exit_time"], trade["exit_reason"])
        for trade in baseline["full_result"]["trades"]
    ]
    frozen_signature = [
        (trade["event_id"], trade["entry_time"], trade["exit_time"], trade["exit_reason"])
        for trade in frozen_result["trades"]
    ]
    invariance = {
        "trade_path_equal": baseline_signature == frozen_signature,
        "trade_count_equal": baseline["full_result"]["trade_count"] == frozen_result["trade_count"],
        "net_profit_delta": baseline["full_result"]["net_profit"] - frozen_result["net_profit"],
    }
    if not invariance["trade_path_equal"] or abs(invariance["net_profit_delta"]) > 1e-6:
        raise RuntimeError(f"v1 baseline invariance failed for {len(universe)} symbols: {invariance}")

    minimum_trades = max(40, int(frozen_result["trade_count"] * 0.50))

    ranking_rows = [
        run(mode, frozen_signal, replace(frozen_portfolio, ranking_mode=mode))
        for mode in (
            "quality_desc",
            "range_asc",
            "breakout_extension_desc",
            "extension_to_range_desc",
            "trend_alignment_desc",
            "directional_breadth_desc",
            "directional_btc_4h_desc",
        )
    ]
    stages["stage1_candidate_ranking"] = _public_stage(ranking_rows)
    selected = _select_best(ranking_rows, minimum_trades)
    selected_signal = VolatilityBreakoutConfig(**selected["signal_config"])
    selected_portfolio = PortfolioSearchConfig(**selected["portfolio_config"])

    fail_fast_rows = [run("disabled", selected_signal, selected_portfolio)]
    for minutes in (120, 240, 360, 480, 600):
        for minimum_mfe in (0.10, 0.25, 0.50, 0.75):
            for current_r in (0.0, -0.25, -0.50):
                signal = replace(
                    selected_signal,
                    fail_fast_minutes=minutes,
                    fail_fast_min_mfe_r=minimum_mfe,
                    fail_fast_max_current_r=current_r,
                )
                fail_fast_rows.append(
                    run(f"ff_{minutes}m_mfe{minimum_mfe:g}_cur{current_r:g}", signal, selected_portfolio)
                )
    stages["stage2_conditional_fail_fast"] = _public_stage(fail_fast_rows)
    selected = _select_best(fail_fast_rows, minimum_trades)
    selected_signal = VolatilityBreakoutConfig(**selected["signal_config"])

    btc_rows = [run("btc_disabled", selected_signal, selected_portfolio)]
    for minimum in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01):
        btc_rows.append(
            run(
                f"btc_directional_min_{minimum:g}",
                selected_signal,
                replace(selected_portfolio, min_directional_btc_return_4h=minimum),
            )
        )
    for maximum in (0.02, 0.03, 0.05, 0.08):
        btc_rows.append(
            run(
                f"btc_directional_max_{maximum:g}",
                selected_signal,
                replace(selected_portfolio, max_directional_btc_return_4h=maximum),
            )
        )
    stages["stage3_btc_regime"] = _public_stage(btc_rows)
    selected = _select_best(btc_rows, minimum_trades)
    selected_portfolio = PortfolioSearchConfig(**selected["portfolio_config"])

    eth_rows = [run("eth_disabled", selected_signal, selected_portfolio)]
    for minimum in (-0.02, -0.01, -0.005, 0.0, 0.005):
        eth_rows.append(
            run(
                f"eth_directional_min_{minimum:g}",
                selected_signal,
                replace(selected_portfolio, min_directional_eth_return_4h=minimum),
            )
        )
    for maximum in (0.03, 0.05, 0.08):
        eth_rows.append(
            run(
                f"eth_directional_max_{maximum:g}",
                selected_signal,
                replace(selected_portfolio, max_directional_eth_return_4h=maximum),
            )
        )
    stages["stage4_eth_regime"] = _public_stage(eth_rows)
    selected = _select_best(eth_rows, minimum_trades)
    selected_portfolio = PortfolioSearchConfig(**selected["portfolio_config"])

    breadth_rows = [run("breadth_disabled", selected_signal, selected_portfolio)]
    for minimum in (0.25, 0.35, 0.45, 0.55, 0.65):
        breadth_rows.append(
            run(
                f"breadth_directional_min_{minimum:g}",
                selected_signal,
                replace(selected_portfolio, min_directional_breadth=minimum),
            )
        )
    for maximum in (0.60, 0.70, 0.80, 0.90):
        breadth_rows.append(
            run(
                f"breadth_directional_max_{maximum:g}",
                selected_signal,
                replace(selected_portfolio, max_directional_breadth=maximum),
            )
        )
    stages["stage5_market_breadth"] = _public_stage(breadth_rows)
    selected = _select_best(breadth_rows, minimum_trades)
    selected_portfolio = PortfolioSearchConfig(**selected["portfolio_config"])

    alpha_rows = [run("alpha_unfiltered", selected_signal, selected_portfolio)]
    for minimum in (0.05, 0.10, 0.20, 0.50, 0.75, 1.0):
        alpha_rows.append(
            run(
                f"breakout_extension_min_{minimum:g}",
                replace(selected_signal, min_breakout_extension_atr=minimum),
                selected_portfolio,
            )
        )
    for maximum in (4.5, 5.0, 6.0, 7.0, 8.0):
        alpha_rows.append(
            run(
                f"range_atr_max_{maximum:g}",
                replace(selected_signal, max_range_atr=maximum),
                selected_portfolio,
            )
        )
    for maximum in (1.5, 2.0, 3.0, 5.0):
        alpha_rows.append(
            run(
                f"body_atr_max_{maximum:g}",
                replace(selected_signal, max_body_atr=maximum),
                selected_portfolio,
            )
        )
    stages["stage6_breakout_structure"] = _public_stage(alpha_rows)
    selected = _select_best(alpha_rows, minimum_trades)
    selected_signal = VolatilityBreakoutConfig(**selected["signal_config"])

    balanced = run("balanced", selected_signal, selected_portfolio)
    risk_rows = [balanced]
    base_risk = frozen_portfolio.risk_per_trade_pct
    for risk in sorted({base_risk * 0.80, base_risk, base_risk * 1.10}):
        for long_multiplier in (1.0, 1.15, 1.30):
            for short_multiplier in (0.60, 0.80, 1.0):
                portfolio = replace(
                    selected_portfolio,
                    risk_per_trade_pct=risk,
                    long_risk_multiplier=long_multiplier,
                    short_risk_multiplier=short_multiplier,
                )
                risk_rows.append(
                    run(
                        f"risk{risk:.4f}_long{long_multiplier:.2f}_short{short_multiplier:.2f}",
                        selected_signal,
                        portfolio,
                    )
                )
    stages["stage7_directional_risk"] = _public_stage(risk_rows)
    aggressive = _select_best(risk_rows, minimum_trades)
    final_signal = VolatilityBreakoutConfig(**aggressive["signal_config"])
    final_portfolio = PortfolioSearchConfig(**aggressive["portfolio_config"])
    final_result = aggressive["full_result"]

    final_candidates = filter_candidates(candidates, final_signal)
    stress: dict[str, Any] = {}
    stress["fixed_risk_no_compounding"] = compact_summary(
        run(
            "fixed_risk_no_compounding",
            final_signal,
            replace(final_portfolio, compound=False),
            candidate_override=final_candidates,
        )["full_result"]
    )
    for delay in (1, 2, 5):
        shifted = _shift_candidates(final_candidates, delay, execution_data)
        stress[f"entry_delay_{delay}m"] = compact_summary(
            run(
                f"entry_delay_{delay}m",
                final_signal,
                final_portfolio,
                candidate_override=shifted,
            )["full_result"]
        )
    for multiplier in (1.5, 2.0):
        stressed_execution = replace(
            execution,
            market_slippage_bps=execution.market_slippage_bps * multiplier,
            stop_slippage_bps=execution.stop_slippage_bps * multiplier,
            take_profit_slippage_bps=execution.take_profit_slippage_bps * multiplier,
            taker_fee_rate=execution.taker_fee_rate * multiplier,
        )
        stress[f"cost_{multiplier:.1f}x"] = compact_summary(
            run(
                f"cost_{multiplier:.1f}x",
                final_signal,
                final_portfolio,
                candidate_override=final_candidates,
                execution_override=stressed_execution,
            )["full_result"]
        )
    ranked_trades = sorted(final_result["trades"], key=lambda trade: trade["net_pnl"], reverse=True)
    for count in (1, 3, 5):
        excluded = frozenset(trade["event_id"] for trade in ranked_trades[:count])
        stress[f"exclude_top_{count}_path"] = compact_summary(
            run(
                f"exclude_top_{count}_path",
                final_signal,
                final_portfolio,
                candidate_override=final_candidates,
                skip_event_ids=excluded,
            )["full_result"]
        )
    top_symbol = max(final_result["by_symbol"], key=lambda key: final_result["by_symbol"][key]["net_pnl"])
    stress["exclude_top_symbol"] = {
        "excluded_symbol": top_symbol,
        "result": compact_summary(
            run(
                "exclude_top_symbol",
                final_signal,
                final_portfolio,
                candidate_override=_without_symbols(final_candidates, frozenset({top_symbol})),
            )["full_result"]
        ),
    }

    return {
        "universe_size": len(universe),
        "symbols": list(universe),
        "minimum_selection_trades": minimum_trades,
        "baseline_invariance": invariance,
        "market_context": {
            "snapshot_count": len(context),
            "definition": "closed 60m BTC/ETH 4h return and point-in-time universe EMA21 breadth",
        },
        "selected_signal_config": final_signal.as_dict(),
        "selected_portfolio_config": asdict(final_portfolio),
        "balanced_signal_config": balanced["signal_config"],
        "balanced_portfolio_config": balanced["portfolio_config"],
        "balanced_result": balanced["full_result"],
        "stages": stages,
        "final_result": final_result,
        "stress_tests": stress,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Dual Thrust Volatility Breakout v2 - 3 Month Full-Cost Research",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        "- v1 artifacts remain frozen; v2 uses separate outputs",
        "- Historical staged optimization only; not untouched OOS or a live recommendation",
        "",
    ]
    for size in ("30", "50"):
        old = report["v1_comparison"][size]
        result = report["universes"][size]["final_result"]
        balanced = report["universes"][size]["balanced_result"]
        lines.extend(
            [
                f"## {size} symbols",
                "",
                f"- v1: `{old['trade_count']}` trades, `{old['net_profit']:+.2f}U`, PF `{old['profit_factor']:.3f}`, DD `{old['max_drawdown_pct']:.2%}`",
                f"- v2 balanced: `{balanced['trade_count']}` trades, `{balanced['net_profit']:+.2f}U`, PF `{balanced['profit_factor']:.3f}`, DD `{balanced['max_drawdown_pct']:.2%}`",
                f"- v2 aggressive: `{result['trade_count']}` trades, `{result['net_profit']:+.2f}U`, PF `{result['profit_factor']:.3f}`, DD `{result['max_drawdown_pct']:.2%}`",
                f"- Long: `{result['by_side'].get('LONG', {}).get('net_pnl', 0.0):+.2f}U`; Short: `{result['by_side'].get('SHORT', {}).get('net_pnl', 0.0):+.2f}U`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continue frozen Dual Thrust v1 with bounded v2 research")
    parser.add_argument("--signal-data-dir", default="data/binance_15m_365d_top100")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--funding-data-dir", default="data/binance_funding_365d_top100")
    parser.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    parser.add_argument("--start", default="2026-03-06T00:00:00")
    parser.add_argument("--end", default="2026-06-06T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--v1-report", default="reports/volatility_breakout_v1_frozen_3m_30_vs_50.json")
    parser.add_argument("--v1-config30", default="config.volatility-breakout.v1-frozen-30.json")
    parser.add_argument("--v1-config50", default="config.volatility-breakout.v1-frozen-50.json")
    parser.add_argument("--output", default="reports/volatility_breakout_v2_3m_30_vs_50.json")
    parser.add_argument("--summary", default="reports/volatility_breakout_v2_3m_30_vs_50.md")
    parser.add_argument("--config30", default="config.volatility-breakout.v2-optimized-30.json")
    parser.add_argument("--config50", default="config.volatility-breakout.v2-optimized-50.json")
    parser.add_argument("--manifest", default="config.volatility-breakout.v2-3m-manifest.json")
    return parser


def run_v2(args: argparse.Namespace) -> dict[str, Any]:
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)
    frozen_report = json.loads(Path(args.v1_report).read_text(encoding="utf-8"))
    frozen_configs = {
        "30": json.loads(Path(args.v1_config30).read_text(encoding="utf-8")),
        "50": json.loads(Path(args.v1_config50).read_text(encoding="utf-8")),
    }
    results: dict[str, Any] = {}
    for size, universe in (("30", UNIVERSE_30), ("50", UNIVERSE_50)):
        print(f"starting v2 staged research for {size} symbols", flush=True)
        results[size] = optimize_v2_universe(
            universe,
            VolatilityBreakoutConfig(**frozen_configs[size]["signal"]),
            PortfolioSearchConfig(**frozen_configs[size]["portfolio"]),
            signal_data,
            execution_data,
            rules,
            execution,
            start,
            end,
            args.initial_equity,
            frozen_report["universes"][size]["final_result"],
        )
        final = results[size]["final_result"]
        print(
            f"completed {size}: trades={final['trade_count']} net={final['net_profit']:.2f} "
            f"pf={final['profit_factor']:.3f} dd={final['max_drawdown_pct']:.2%}",
            flush=True,
        )

    report = {
        "strategy_name": V2_RESEARCH_NAME,
        "research_status": "historical_in_sample_staged_optimization_not_untouched_oos",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": args.initial_equity,
        "v1_frozen_manifest": "config.volatility-breakout.v1-frozen-manifest.json",
        "cost_model": frozen_report["cost_model"],
        "execution_rules": frozen_report["execution_rules"],
        "selection_rule": {
            "primary": "maximum historical net profit",
            "minimum_trades": "50% of v1 count, floor 40",
            "minimum_profit_factor": 1.05,
            "maximum_drawdown_pct": 0.55,
            "experiment_budget": "bounded staged search; one module per stage",
        },
        "v1_comparison": {
            size: compact_summary(frozen_report["universes"][size]["final_result"])
            for size in ("30", "50")
        },
        "universes": results,
    }
    _write_json(Path(args.output), report)
    _write_markdown(Path(args.summary), report)

    for size, target in (("30", Path(args.config30)), ("50", Path(args.config50))):
        result = results[size]
        _write_json(
            target,
            {
                "strategy_name": V2_RESEARCH_NAME,
                "status": "historical_research_not_live",
                "period": report["period"],
                "universe_size": int(size),
                "symbols": result["symbols"],
                "signal": result["selected_signal_config"],
                "portfolio": result["selected_portfolio_config"],
                "balanced_signal": result["balanced_signal_config"],
                "balanced_portfolio": result["balanced_portfolio_config"],
                "cost_model": report["cost_model"],
                "execution_rules": report["execution_rules"],
                "stress_tests": result["stress_tests"],
            },
        )

    artifacts = (
        args.output,
        args.summary,
        args.config30,
        args.config50,
        "crypto_scalper/volatility_breakout.py",
        "crypto_scalper/volatility_breakout_optimize.py",
        "crypto_scalper/volatility_breakout_v2_optimize.py",
        "tests/test_volatility_breakout.py",
        "config.volatility-breakout.v1-frozen-manifest.json",
    )
    _write_json(
        Path(args.manifest),
        {
            "strategy_name": V2_RESEARCH_NAME,
            "status": "historical_research_frozen_separately_from_v1",
            "report": args.output,
            "summary": args.summary,
            "configs": [args.config30, args.config50],
            "v1_preserved": True,
            "funding_missing_symbols": metadata["funding_missing"],
            "hashes": {path: sha256_file(path) for path in artifacts if Path(path).exists()},
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_v2(args)
    concise = {
        size: compact_summary(report["universes"][size]["final_result"])
        for size in ("30", "50")
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
