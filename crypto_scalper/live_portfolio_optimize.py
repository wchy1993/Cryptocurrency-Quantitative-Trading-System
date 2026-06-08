from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .config import StrategyConfig
from .live_config import ExchangeConfig, LiveAppConfig, LiveRiskConfig, LiveTradingConfig, MultiTimeframeFilterConfig, load_live_config, write_live_config
from .live_portfolio_backtest import _load_symbol_data, run_portfolio_backtest_config


def optimize_live_portfolio(
    config_path: str,
    data_dir: str,
    trials: int,
    seed: int,
    max_drawdown_pct: float,
    min_trades: int,
    top: int,
    report_path: str,
    write_config_path: str | None,
    initial_equity: float | None = None,
) -> dict[str, Any]:
    base = load_live_config(config_path)
    candles_by_symbol = _load_symbol_data(data_dir, tuple(base.trading.symbols), base.trading.timeframe)
    if not candles_by_symbol:
        raise RuntimeError(f"no data loaded from {data_dir}")

    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(_sample_live_configs(base, trials, seed, max_drawdown_pct), 1):
        summary = run_portfolio_backtest_config(candidate, candles_by_symbol, initial_equity)
        row = {
            "trial": index,
            "score": _score(summary, max_drawdown_pct, min_trades),
            "eligible": _eligible(summary, max_drawdown_pct, min_trades),
            "summary": summary,
            "config": _config_payload(candidate),
        }
        rows.append(row)
        print(
            f"trial={index}/{trials} score={row['score']:.2f} "
            f"net={summary['net_return_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}% "
            f"pf={_fmt_optional(summary.get('profit_factor'))} trades={summary['total_trades']}",
            file=sys.stderr,
            flush=True,
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    eligible_rows = [row for row in rows if row["eligible"]]
    best = eligible_rows[0] if eligible_rows else rows[0]
    Path(report_path).write_text(_build_report(rows[: max(top, 1)], best, max_drawdown_pct, min_trades, data_dir), encoding="utf-8")

    if write_config_path and best:
        write_live_config(write_config_path, _config_from_payload(best["config"]))

    return {
        "datasets": len(candles_by_symbol),
        "evaluated_candidates": len(rows),
        "eligible_candidates": len(eligible_rows),
        "best_score": best["score"],
        "best_eligible": best["eligible"],
        "best_summary": best["summary"],
        "report": report_path,
        "write_config": write_config_path,
    }


def _sample_live_configs(base: LiveAppConfig, trials: int, seed: int, max_drawdown_pct: float) -> Iterable[LiveAppConfig]:
    rng = random.Random(seed)
    max_drawdown = max_drawdown_pct / 100.0
    yielded = 0
    seen: set[tuple[Any, ...]] = set()

    if trials > 0:
        yielded += 1
        yield replace(
            base,
            risk=replace(base.risk, max_drawdown_pct=max_drawdown, soft_drawdown_stop_pct=min(base.risk.soft_drawdown_stop_pct, max_drawdown)),
        )

    strategy_space = {
        "fast_ema": (5, 8, 12, 18),
        "slow_ema": (34, 55, 89),
        "atr_period": (10, 14, 21),
        "channel_period": (20, 30, 48, 72),
        "min_atr_pct": (0.001, 0.0025, 0.004, 0.006),
        "max_atr_pct": (0.02, 0.035, 0.05),
        "breakout_buffer_atr": (0.0, 0.1, 0.2, 0.35),
        "ema_gap_atr": (0.0, 0.1, 0.2, 0.35),
        "volume_period": (20, 30, 48),
        "min_volume_ratio": (0.8, 1.0, 1.2, 1.5),
        "stop_loss_atr": (0.9, 1.2, 1.6, 2.0, 2.5),
        "take_profit_atr": (0.9, 1.2, 1.6, 2.4, 3.0, 4.0),
        "breakeven_atr": (0.5, 0.8, 1.0, 1.5),
        "trailing_activation_atr": (0.5, 0.8, 1.0, 1.5),
        "trailing_stop_atr": (0.5, 0.8, 1.0),
        "max_holding_bars": (12, 24, 36, 48, 72),
        "spike_guard_enabled": (True, False),
        "spike_trade_enabled": (False, True),
        "spike_min_range_atr": (2.5, 3.0, 4.0),
        "spike_min_wick_atr": (1.0, 1.4, 2.0),
        "spike_min_wick_ratio": (0.5, 0.6, 0.7),
        "spike_min_volume_ratio": (1.0, 1.2, 1.6),
        "spike_recovery_ratio": (0.35, 0.45, 0.6),
        "spike_stop_atr": (0.5, 0.7, 1.0),
        "spike_take_profit_atr": (0.7, 0.9, 1.2),
        "spike_risk_multiplier": (0.2, 0.35, 0.5),
        "spike_max_holding_bars": (4, 8, 12),
        "long_score_threshold": (0.55, 0.65, 0.75, 0.85),
        "short_score_threshold": (0.45, 0.55, 0.65, 0.75, 0.85),
        "long_risk_bias": (0.0, 0.15, 0.35, 0.5, 0.75),
        "short_risk_bias": (0.2, 0.35, 0.5, 0.75, 1.0),
        "regime_filter_enabled": (True, False),
        "regime_lookback": (12, 24, 48),
        "long_min_slow_slope_atr": (-0.75, -0.25, 0.0, 0.25),
        "short_max_slow_slope_atr": (0.0, 0.25, 0.75, 1.5),
        "super_volume_breakout_enabled": (True,),
        "super_volume_min_ratio": (2.2, 2.8, 3.5, 4.5),
        "super_volume_min_breakout_atr": (0.45, 0.65, 0.85, 1.1),
        "super_volume_min_body_atr": (0.2, 0.35, 0.5),
        "super_volume_confidence_boost": (0.1, 0.15, 0.2),
        "super_volume_risk_multiplier": (1.15, 1.35, 1.5),
        "super_volume_take_profit_multiplier": (1.15, 1.35, 1.6),
    }
    filter_space = {
        "min_score": (4, 5, 6),
        "extreme_reversal_entry_enabled": (True, False),
        "pre_cross_entry_enabled": (True, False),
        "reversal_cross_lookback_bars": (2, 3, 4),
        "confirmed_cross_risk_multiplier": (0.5, 0.65, 0.8),
        "pre_cross_risk_multiplier": (0.15, 0.25, 0.35),
        "rsi_long_floor": (30.0, 32.0, 35.0),
        "rsi_long_ceiling": (68.0, 72.0, 76.0),
        "rsi_short_floor": (24.0, 28.0, 32.0),
        "rsi_short_ceiling": (64.0, 68.0, 70.0),
    }
    trading_space = {
        "max_open_positions": (3, 4, 5),
        "max_new_entries_per_cycle": (1, 2),
        "symbol_reentry_cooldown_seconds": (1800, 3600, 7200),
        "initial_entry_fraction": (0.45, 0.6, 0.75),
        "scale_in_entry_fraction": (0.15, 0.25, 0.35),
        "max_scale_ins_per_symbol": (0, 1, 2),
        "scale_in_min_profit_pct": (0.003, 0.005, 0.008),
        "scale_in_cooldown_seconds": (1800, 3600, 7200),
        "breakeven_trigger_pct": (0.002, 0.003, 0.0045),
        "breakeven_lock_pct": (0.0008, 0.0012, 0.0018),
        "trailing_activation_pct": (0.005, 0.008, 0.012),
        "trailing_pullback_pct": (0.0025, 0.0035, 0.005),
        "momentum_exit_min_profit_pct": (0.002, 0.003, 0.0045),
        "quick_take_profit_pct": (0.005, 0.0075, 0.01),
        "strong_take_profit_pct": (0.014, 0.018, 0.024),
    }
    risk_space = {
        "max_account_margin_usage_pct": (0.05, 0.06, 0.08, 0.10),
        "max_symbol_margin_pct": (0.02, 0.03, 0.04),
        "risk_per_trade_pct": (0.03, 0.04, 0.05, 0.06),
        "max_daily_loss_pct": (0.08, 0.10, 0.15),
        "soft_drawdown_reduce_pct": (0.04, 0.05, 0.07),
        "soft_drawdown_stop_pct": (0.08, 0.10, 0.15),
        "min_profit_after_cost_pct": (0.001, 0.0015, 0.002),
    }

    max_attempts = max(trials * 80, 500)
    attempts = 0
    while yielded < max(0, trials) and attempts < max_attempts:
        attempts += 1
        strategy_values = {name: rng.choice(values) for name, values in strategy_space.items()}
        if strategy_values["fast_ema"] >= strategy_values["slow_ema"]:
            continue
        if strategy_values["take_profit_atr"] < strategy_values["stop_loss_atr"] * 0.3:
            continue
        filter_values = {name: rng.choice(values) for name, values in filter_space.items()}
        trading_values = {name: rng.choice(values) for name, values in trading_space.items()}
        risk_values = {name: rng.choice(values) for name, values in risk_space.items()}
        risk_values["max_drawdown_pct"] = max_drawdown
        risk_values["soft_drawdown_stop_pct"] = min(risk_values["soft_drawdown_stop_pct"], max_drawdown)

        key = (
            tuple(strategy_values[name] for name in sorted(strategy_values))
            + tuple(filter_values[name] for name in sorted(filter_values))
            + tuple(trading_values[name] for name in sorted(trading_values))
            + tuple(risk_values[name] for name in sorted(risk_values))
        )
        if key in seen:
            continue
        seen.add(key)
        yielded += 1
        yield replace(
            base,
            trading=replace(base.trading, **trading_values),
            strategy=replace(base.strategy, allow_short=True, **strategy_values),
            filters=replace(base.filters, enabled=True, **filter_values),
            risk=replace(base.risk, **risk_values),
        )


def _score(summary: dict[str, Any], max_drawdown_pct: float, min_trades: int) -> float:
    net_return = float(summary["net_return_pct"])
    drawdown = float(summary["max_drawdown_pct"])
    win_rate = float(summary["win_rate_pct"])
    trades = int(summary["total_trades"])
    profit_factor = _none_to_zero(summary.get("profit_factor"))
    avg_trade_return = float(summary.get("avg_trade_return_pct", 0.0))

    score = (
        net_return * 7.0
        + (profit_factor - 1.0) * 95.0
        + win_rate * 0.02
        + avg_trade_return * 35.0
        + min(3.0, math.log(max(trades, 1)) / 2.5)
        - drawdown * 1.5
    )
    if trades < min_trades:
        score -= (min_trades - trades) * 0.05
    if summary["long"]["trades"] > 0 and summary["long"]["net_pnl"] <= 0:
        score -= min(25.0, abs(summary["long"]["net_pnl"]) * 0.7)
    if summary["short"]["trades"] > 0 and summary["short"]["net_pnl"] <= 0:
        score -= min(25.0, abs(summary["short"]["net_pnl"]) * 0.7)
    if drawdown > max_drawdown_pct:
        score -= 500.0 + (drawdown - max_drawdown_pct) * 50.0
    if net_return <= 0.0 or profit_factor <= 1.0:
        score -= 25.0
    if profit_factor < 1.08:
        score -= (1.08 - profit_factor) * 120.0
    return score


def _eligible(summary: dict[str, Any], max_drawdown_pct: float, min_trades: int) -> bool:
    return (
        summary["max_drawdown_pct"] <= max_drawdown_pct
        and summary["net_return_pct"] > 0.0
        and _none_to_zero(summary.get("profit_factor")) >= 1.08
        and summary["total_trades"] >= min_trades
    )


def _config_payload(config: LiveAppConfig) -> dict[str, Any]:
    return {
        "exchange": asdict(config.exchange),
        "trading": asdict(config.trading),
        "strategy": asdict(config.strategy),
        "filters": asdict(config.filters),
        "risk": asdict(config.risk),
    }


def _config_from_payload(payload: dict[str, Any]) -> LiveAppConfig:
    trading = dict(payload["trading"])
    trading["symbols"] = tuple(trading["symbols"])
    trading["entry_symbols"] = tuple(trading["entry_symbols"])
    filters = dict(payload["filters"])
    filters["timeframes"] = tuple(filters["timeframes"])
    return LiveAppConfig(
        exchange=ExchangeConfig(**payload["exchange"]),
        trading=LiveTradingConfig(**trading),
        strategy=StrategyConfig(**payload["strategy"]),
        filters=MultiTimeframeFilterConfig(**filters),
        risk=LiveRiskConfig(**payload["risk"]),
    )


def _build_report(rows: list[dict[str, Any]], best: dict[str, Any], max_drawdown_pct: float, min_trades: int, data_dir: str) -> str:
    lines = [
        f"# {best['config']['trading']['timeframe']} 组合优化报告",
        "",
        f"- 最大回撤硬约束: {max_drawdown_pct:.2f}%",
        f"- 最小交易数: {min_trades}",
        f"- 最优候选是否满足约束: {'是' if best['eligible'] else '否'}",
        f"- 数据: `{data_dir}` Binance USD-M {best['config']['trading']['timeframe']} CSV。",
        "- 执行限制: 本轮只做本地组合回测，没有连接真实账户，也没有真实下单。",
        "",
        "## 最优结果",
        _summary_lines(best["summary"]),
        "",
        "## 最优参数",
        "```json",
        json.dumps(
            {
                "trading": _compact(best["config"]["trading"], _TRADING_KEYS),
                "strategy": best["config"]["strategy"],
                "filters": _compact(best["config"]["filters"], _FILTER_KEYS),
                "risk": _compact(best["config"]["risk"], _RISK_KEYS),
            },
            indent=2,
            ensure_ascii=False,
        ),
        "```",
        "",
        "## 候选排名",
        "| 排名 | Trial | Score | 合格 | 收益% | 月化% | 回撤% | 胜率% | PF | 交易 | 多头PnL | 空头PnL |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        summary = row["summary"]
        lines.append(
            f"| {index} | {row['trial']} | {row['score']:.2f} | {'Y' if row['eligible'] else 'N'} | "
            f"{summary['net_return_pct']:.2f} | {summary['monthly_return_pct']:.2f} | "
            f"{summary['max_drawdown_pct']:.2f} | {summary['win_rate_pct']:.2f} | "
            f"{_fmt_optional(summary.get('profit_factor'))} | {summary['total_trades']} | "
            f"{summary['long']['net_pnl']:.2f} | {summary['short']['net_pnl']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _summary_lines(summary: dict[str, Any]) -> str:
    return (
        f"- 初始权益: {summary['initial_equity']:.2f} U\n"
        f"- 最终权益: {summary['final_equity']:.2f} U\n"
        f"- 净收益: {summary['net_return_pct']:.2f}%\n"
        f"- 折算月收益: {summary['monthly_return_pct']:.2f}%\n"
        f"- 最大回撤: {summary['max_drawdown_pct']:.2f}%\n"
        f"- 总交易: {summary['total_trades']}\n"
        f"- 胜率: {summary['win_rate_pct']:.2f}%\n"
        f"- Profit Factor: {_fmt_optional(summary.get('profit_factor'))}\n"
        f"- 多头: {summary['long']['trades']} 笔，PnL {summary['long']['net_pnl']:.2f} U，PF {_fmt_optional(summary['long'].get('profit_factor'))}\n"
        f"- 空头: {summary['short']['trades']} 笔，PnL {summary['short']['net_pnl']:.2f} U，PF {_fmt_optional(summary['short'].get('profit_factor'))}"
    )


def _compact(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: values[key] for key in keys}


def _none_to_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}"


_TRADING_KEYS = (
    "timeframe",
    "max_open_positions",
    "max_new_entries_per_cycle",
    "symbol_reentry_cooldown_seconds",
    "initial_entry_fraction",
    "scale_in_entry_fraction",
    "max_scale_ins_per_symbol",
    "scale_in_min_profit_pct",
    "scale_in_cooldown_seconds",
    "breakeven_trigger_pct",
    "breakeven_lock_pct",
    "trailing_activation_pct",
    "trailing_pullback_pct",
    "momentum_exit_min_profit_pct",
    "quick_take_profit_pct",
    "strong_take_profit_pct",
)
_FILTER_KEYS = (
    "enabled",
    "timeframes",
    "min_score",
    "extreme_reversal_entry_enabled",
    "pre_cross_entry_enabled",
    "reversal_cross_lookback_bars",
    "confirmed_cross_risk_multiplier",
    "pre_cross_risk_multiplier",
    "rsi_long_floor",
    "rsi_long_ceiling",
    "rsi_short_floor",
    "rsi_short_ceiling",
)
_RISK_KEYS = (
    "starting_capital_usdt",
    "max_account_margin_usage_pct",
    "max_symbol_margin_pct",
    "risk_per_trade_pct",
    "max_daily_loss_pct",
    "max_drawdown_pct",
    "starting_capital_drawdown_stop_pct",
    "weekly_profit_drawdown_stop_pct",
    "soft_drawdown_reduce_pct",
    "soft_drawdown_stop_pct",
    "min_profit_after_cost_pct",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.live.json")
    parser.add_argument("--data-dir", default="data/binance_15m_100d")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-drawdown-pct", type=float, default=15.0)
    parser.add_argument("--min-trades", type=int, default=300)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--report", default="report_live_opt_15m_60.md")
    parser.add_argument("--write-config", default="config.live.optimized.json")
    parser.add_argument("--initial-equity", type=float, default=None)
    args = parser.parse_args()
    payload = optimize_live_portfolio(
        config_path=args.config,
        data_dir=args.data_dir,
        trials=args.trials,
        seed=args.seed,
        max_drawdown_pct=args.max_drawdown_pct,
        min_trades=args.min_trades,
        top=args.top,
        report_path=args.report,
        write_config_path=args.write_config,
        initial_equity=args.initial_equity,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
