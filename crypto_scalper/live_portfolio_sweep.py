from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

from .live_config import load_live_config, write_live_config
from .live_portfolio_backtest import _load_symbol_data, run_portfolio_backtest_config


def run_sweep(config_path: str, data_dir: str, max_drawdown_pct: float, write_best: str | None = None) -> dict[str, Any]:
    base = load_live_config(config_path)
    candles_by_symbol = _load_symbol_data(data_dir, tuple(base.trading.symbols), base.trading.timeframe)
    if not candles_by_symbol:
        raise RuntimeError(f"no data loaded from {data_dir}")

    rows = []
    for name, candidate in _focused_candidates(base):
        summary = run_portfolio_backtest_config(candidate, candles_by_symbol)
        eligible = _eligible(summary, max_drawdown_pct)
        rows.append({"name": name, "eligible": eligible, "summary": summary, "config": candidate})
        print(
            f"{name}: net={summary['net_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"pf={_fmt(summary.get('profit_factor'))} trades={summary['total_trades']} eligible={eligible}",
            file=sys.stderr,
            flush=True,
        )

    rows.sort(key=lambda row: _score(row["summary"], max_drawdown_pct), reverse=True)
    best = rows[0]
    if write_best and best["eligible"]:
        write_live_config(write_best, best["config"])

    return {
        "datasets": len(candles_by_symbol),
        "best": best["name"],
        "best_eligible": best["eligible"],
        "best_summary": best["summary"],
        "candidates": [
            {
                "name": row["name"],
                "eligible": row["eligible"],
                "net_return_pct": row["summary"]["net_return_pct"],
                "max_drawdown_pct": row["summary"]["max_drawdown_pct"],
                "profit_factor": row["summary"]["profit_factor"],
                "total_trades": row["summary"]["total_trades"],
                "final_equity": row["summary"]["final_equity"],
            }
            for row in rows
        ],
    }


def _focused_candidates(base: Any) -> list[tuple[str, Any]]:
    return [
        (
            "risk_0_05",
            replace(base, risk=replace(base.risk, risk_per_trade_pct=0.05)),
        ),
        (
            "risk_0_048",
            replace(base, risk=replace(base.risk, risk_per_trade_pct=0.048)),
        ),
        (
            "risk_0_045",
            replace(base, risk=replace(base.risk, risk_per_trade_pct=0.045)),
        ),
        (
            "risk_0_03",
            replace(base, risk=replace(base.risk, risk_per_trade_pct=0.03)),
        ),
        (
            "long_regime_mild",
            replace(
                base,
                strategy=replace(
                    base.strategy,
                    regime_filter_enabled=True,
                    regime_lookback=24,
                    long_min_slow_slope_atr=-0.25,
                    short_max_slow_slope_atr=1.5,
                ),
            ),
        ),
        (
            "long_regime_strict",
            replace(
                base,
                strategy=replace(
                    base.strategy,
                    regime_filter_enabled=True,
                    regime_lookback=24,
                    long_min_slow_slope_atr=0.0,
                    short_max_slow_slope_atr=1.5,
                ),
            ),
        ),
        (
            "long_threshold_0_90",
            replace(base, strategy=replace(base.strategy, long_score_threshold=0.90)),
        ),
        (
            "long_threshold_0_95",
            replace(base, strategy=replace(base.strategy, long_score_threshold=0.95)),
        ),
        (
            "long_bias_0_55",
            replace(base, strategy=replace(base.strategy, long_risk_bias=0.55)),
        ),
        (
            "long_bias_0_45",
            replace(base, strategy=replace(base.strategy, long_risk_bias=0.45)),
        ),
        (
            "long_guard_combo",
            replace(
                base,
                strategy=replace(
                    base.strategy,
                    regime_filter_enabled=True,
                    regime_lookback=24,
                    long_min_slow_slope_atr=-0.25,
                    long_score_threshold=0.90,
                    long_risk_bias=0.55,
                    short_max_slow_slope_atr=1.5,
                ),
            ),
        ),
        (
            "min_score_6",
            replace(base, filters=replace(base.filters, min_score=6)),
        ),
        (
            "super_volume_loose",
            replace(
                base,
                strategy=replace(
                    base.strategy,
                    super_volume_min_ratio=2.8,
                    super_volume_min_breakout_atr=0.65,
                    super_volume_min_body_atr=0.2,
                    super_volume_risk_multiplier=1.35,
                ),
            ),
        ),
        (
            "faster_profit",
            replace(
                base,
                trading=replace(base.trading, quick_take_profit_pct=0.006, strong_take_profit_pct=0.018),
                strategy=replace(base.strategy, take_profit_atr=1.6, max_holding_bars=18),
            ),
        ),
        (
            "scale_more",
            replace(
                base,
                trading=replace(
                    base.trading,
                    scale_in_entry_fraction=0.25,
                    max_scale_ins_per_symbol=2,
                    scale_in_min_profit_pct=0.006,
                    scale_in_cooldown_seconds=3600,
                ),
            ),
        ),
    ]


def _eligible(summary: dict[str, Any], max_drawdown_pct: float) -> bool:
    profit_factor = summary.get("profit_factor")
    return summary["net_return_pct"] > 0 and summary["max_drawdown_pct"] <= max_drawdown_pct and profit_factor is not None and profit_factor > 1.0


def _score(summary: dict[str, Any], max_drawdown_pct: float) -> float:
    score = float(summary["net_return_pct"])
    if summary["max_drawdown_pct"] > max_drawdown_pct:
        score -= 1000.0 + (summary["max_drawdown_pct"] - max_drawdown_pct) * 50.0
    if summary.get("profit_factor") is None or summary["profit_factor"] <= 1.0:
        score -= 500.0
    return score


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.live.optimized_super_volume.json")
    parser.add_argument("--data-dir", default="data/binance_30m_180d")
    parser.add_argument("--max-drawdown-pct", type=float, default=20.0)
    parser.add_argument("--write-best", default=None)
    args = parser.parse_args()
    print(json.dumps(run_sweep(args.config, args.data_dir, args.max_drawdown_pct, args.write_best), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
