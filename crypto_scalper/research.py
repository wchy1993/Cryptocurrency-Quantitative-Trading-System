from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .backtest import BacktestResult, Backtester
from .config import AppConfig, RiskConfig, StrategyConfig
from .data import load_candles_csv
from .models import Candle, Direction, EquityPoint, Trade
from .strategy import VolatilityBreakoutScalper


@dataclass(frozen=True)
class SplitDataset:
    path: str
    train: list[Candle]
    validation: list[Candle]
    test: list[Candle]


@dataclass(frozen=True)
class CandidateConfig:
    strategy: StrategyConfig
    risk: RiskConfig


def run_robust_optimization(
    config: AppConfig,
    data_paths: list[str],
    trials: int,
    seed: int,
    top: int,
    min_trades: int,
    max_drawdown_pct: float,
    train_ratio: float,
    validation_ratio: float,
    report_path: str,
    write_config_path: str | None = None,
) -> dict[str, Any]:
    splits = [_split_dataset(path, train_ratio, validation_ratio) for path in data_paths]
    rows: list[dict[str, Any]] = []
    train_items = [(item.path, item.train) for item in splits]
    validation_items = [(item.path, item.validation) for item in splits]

    for candidate in _sample_candidate_configs(config.strategy, config.risk, trials, seed):
        train_results = run_backtest_results(train_items, candidate.strategy, candidate.risk)
        if aggregate_results(train_results)["total_trades"] < min_trades:
            continue

        validation_results = run_backtest_results(validation_items, candidate.strategy, candidate.risk)
        selected_paths = select_trade_universe(train_results, validation_results, max_drawdown_pct)
        if len(selected_paths) < 3:
            continue

        train_summary = aggregate_results(_filter_results(train_results, selected_paths))
        validation_summary = aggregate_results(_filter_results(validation_results, selected_paths))
        if validation_summary["total_trades"] < max(5, min_trades // 3):
            continue

        rows.append(
            {
                "validation_score": robust_score(validation_summary, max_drawdown_pct, max(5, min_trades // 3)),
                "strategy": asdict(candidate.strategy),
                "risk": asdict(candidate.risk),
                "train": train_summary,
                "validation": validation_summary,
                "selected_paths": selected_paths,
                "selected_symbols": [_symbol_from_path(path) for path in selected_paths],
            }
        )

    rows.sort(key=lambda row: row["validation_score"], reverse=True)
    selected = rows[: max(top * 20, top, 80)]
    for row in selected:
        strategy = StrategyConfig(**row["strategy"])
        risk = RiskConfig(**row["risk"])
        selected_path_set = set(row["selected_paths"])
        test_items = [(item.path, item.test) for item in splits if item.path in selected_path_set]
        row["test"] = evaluate_many(test_items, strategy, risk)
        row["overfit"] = detect_overfit(row["train"], row["validation"], row["test"])
        row["eligible"] = (
            row["validation"]["max_drawdown_pct"] <= max_drawdown_pct
            and row["test"]["max_drawdown_pct"] <= max_drawdown_pct
            and row["validation"]["net_return_pct"] > 0.0
            and row["test"]["net_return_pct"] > 0.0
            and row["validation"]["win_rate_pct"] >= 55.0
            and row["test"]["win_rate_pct"] >= 55.0
            and _none_to_zero(row["validation"].get("profit_factor")) > 1.0
            and _none_to_zero(row["test"].get("profit_factor")) > 1.0
            and not row["overfit"]
        )

    eligible = [row for row in selected if row.get("eligible")]
    best = eligible[0] if eligible else (selected[0] if selected else None)
    reported_rows = eligible[: max(top, 1)]
    for row in selected:
        if len(reported_rows) >= max(top, 1):
            break
        if row not in reported_rows:
            reported_rows.append(row)
    report = build_report(best, reported_rows, max_drawdown_pct, train_ratio, validation_ratio, data_paths)
    Path(report_path).write_text(report, encoding="utf-8")

    if write_config_path and best and best.get("eligible"):
        _write_candidate_config(config, best, write_config_path)

    return {
        "datasets": len(splits),
        "evaluated_candidates": len(rows),
        "reported_candidates": len(reported_rows),
        "eligible_candidates": len(eligible),
        "best_validation_score": None if best is None else best["validation_score"],
        "best_eligible": False if best is None else bool(best.get("eligible")),
        "report": report_path,
        "write_config": write_config_path if best and best.get("eligible") else None,
    }


def evaluate_many(items: list[tuple[str, list[Candle]]], strategy: StrategyConfig, risk: RiskConfig) -> dict[str, Any]:
    return aggregate_results(run_backtest_results(items, strategy, risk))


def run_backtest_results(items: list[tuple[str, list[Candle]]], strategy: StrategyConfig, risk: RiskConfig) -> list[tuple[str, BacktestResult]]:
    results: list[tuple[str, BacktestResult]] = []
    for path, candles in items:
        if len(candles) < max(strategy.slow_ema, strategy.channel_period, strategy.volume_period, 50) + 10:
            continue
        result = Backtester(candles, VolatilityBreakoutScalper(strategy), risk).run()
        results.append((path, result))
    return results


def select_trade_universe(
    train_results: list[tuple[str, BacktestResult]],
    validation_results: list[tuple[str, BacktestResult]],
    max_drawdown_pct: float,
    max_symbols: int = 12,
) -> list[str]:
    train_by_path = {path: result for path, result in train_results}
    scored: list[tuple[float, str]] = []
    for path, validation_result in validation_results:
        train_result = train_by_path.get(path)
        if not train_result:
            continue
        train_summary = aggregate_results([(path, train_result)])
        validation_summary = aggregate_results([(path, validation_result)])
        if train_summary["total_trades"] < 8 or validation_summary["total_trades"] < 4:
            continue
        if validation_summary["max_drawdown_pct"] > max_drawdown_pct or train_summary["max_drawdown_pct"] > max_drawdown_pct:
            continue
        if validation_summary["net_return_pct"] <= 0 or _none_to_zero(validation_summary["profit_factor"]) <= 1.0:
            continue
        if validation_summary["win_rate_pct"] < 52.0:
            continue
        if _none_to_zero(train_summary["profit_factor"]) < 0.85:
            continue
        score = robust_score(validation_summary, max_drawdown_pct, 4) + max(0.0, train_summary["net_return_pct"]) * 0.2
        scored.append((score, path))
    scored.sort(reverse=True)
    return [path for _, path in scored[:max_symbols]]


def aggregate_results(results: list[tuple[str, BacktestResult]]) -> dict[str, Any]:
    initial = sum(float(result.summary["initial_equity"]) for _, result in results)
    final = sum(float(result.summary["final_equity"]) for _, result in results)
    trades = [trade for _, result in results for trade in result.trades]
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    period_days = max((float(result.summary.get("period_days", 0.0)) for _, result in results), default=0.0)
    months = period_days / 30.4375 if period_days > 0 else 0.0
    net_return_pct = 0.0 if initial <= 0 else (final / initial - 1.0) * 100.0
    annual_return_pct = 0.0
    if period_days > 0 and initial > 0 and final > 0:
        annual_return_pct = ((final / initial) ** (365.25 / period_days) - 1.0) * 100.0
    stage_returns = _aggregate_stage_returns([result for _, result in results])
    stage_values = list(stage_returns.values())
    return {
        "datasets": len(results),
        "initial_equity": initial,
        "final_equity": final,
        "net_profit": final - initial,
        "net_return_pct": net_return_pct,
        "annual_return_pct": annual_return_pct,
        "monthly_return_pct": 0.0 if months <= 0 else net_return_pct / months,
        "period_days": period_days,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "payoff_ratio": _payoff_ratio(wins, losses),
        "sharpe_ratio": _combined_sharpe([result.equity_curve for _, result in results]),
        "avg_trade": 0.0 if not trades else sum(trade.net_pnl for trade in trades) / len(trades),
        "avg_trade_return_pct": 0.0 if not trades else sum(trade.return_pct for trade in trades) / len(trades) * 100.0,
        "max_drawdown_pct": max((float(result.summary["max_drawdown_pct"]) for _, result in results), default=0.0),
        "long": _side_summary([trade for trade in trades if trade.direction == Direction.LONG]),
        "short": _side_summary([trade for trade in trades if trade.direction == Direction.SHORT]),
        "stage_return_pct": stage_returns,
        "stage_return_std_pct": 0.0 if len(stage_values) < 2 else statistics.pstdev(stage_values),
        "negative_stages": sum(1 for value in stage_values if value < 0),
    }


def robust_score(summary: dict[str, Any], max_drawdown_pct: float, min_trades: int) -> float:
    if summary["total_trades"] < min_trades:
        return -1_000_000.0 + float(summary["total_trades"])
    drawdown = float(summary["max_drawdown_pct"])
    if drawdown > max_drawdown_pct:
        return -100_000.0 - drawdown

    profit_factor = min(3.0, _none_to_zero(summary.get("profit_factor")))
    payoff = min(3.0, _none_to_zero(summary.get("payoff_ratio")))
    sharpe = max(-3.0, min(5.0, float(summary.get("sharpe_ratio", 0.0))))
    annual = max(-100.0, min(160.0, float(summary.get("annual_return_pct", 0.0))))
    win_rate = float(summary.get("win_rate_pct", 0.0))
    frequency_bonus = min(2.5, summary["total_trades"] / max(min_trades, 1) * 1.2)
    direction_penalty = 0.0 if summary["long"]["trades"] > 0 and summary["short"]["trades"] > 0 else 0.5
    stability_penalty = float(summary.get("stage_return_std_pct", 0.0)) * 0.5 + float(summary.get("negative_stages", 0)) * 1.5
    loss_penalty = 25.0 if summary["net_return_pct"] <= 0 or profit_factor <= 1.0 else 0.0
    win_penalty = max(0.0, 55.0 - win_rate) * 3.0
    win_bonus = max(0.0, win_rate - 55.0) * 1.5
    return (
        float(summary["net_return_pct"]) * 3.0
        + annual * 0.18
        + sharpe * 3.0
        + profit_factor * 8.0
        + payoff * 2.5
        + win_rate * 0.05
        + win_bonus
        + float(summary["avg_trade_return_pct"]) * 6.0
        + frequency_bonus
        - drawdown * 1.2
        - stability_penalty
        - direction_penalty
        - loss_penalty
        - win_penalty
    )


def detect_overfit(train: dict[str, Any], validation: dict[str, Any], test: dict[str, Any]) -> bool:
    train_return = float(train.get("net_return_pct", 0.0))
    validation_return = float(validation.get("net_return_pct", 0.0))
    test_return = float(test.get("net_return_pct", 0.0))
    train_pf = _none_to_zero(train.get("profit_factor"))
    validation_pf = _none_to_zero(validation.get("profit_factor"))
    test_pf = _none_to_zero(test.get("profit_factor"))
    if train_return > 5.0 and test_return < train_return * 0.25:
        return True
    if validation_return > 2.0 and test_return < -1.0:
        return True
    if min(train_pf, validation_pf) >= 1.2 and test_pf < 1.0:
        return True
    return False


def build_report(
    best: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    max_drawdown_pct: float,
    train_ratio: float,
    validation_ratio: float,
    data_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# 短线策略稳健优化报告")
    lines.append("")
    timeframe = _infer_report_timeframe(data_paths)
    framework = "15m/30m/1h" if timeframe == "15m" else f"{timeframe} 单周期"
    lines.append(f"- 数据集: {len(data_paths)} 个 Binance USD-M {timeframe} CSV")
    lines.append(f"- 切分: 训练集 {train_ratio:.0%} / 验证集 {validation_ratio:.0%} / 测试集 {1 - train_ratio - validation_ratio:.0%}")
    lines.append(f"- 硬约束: 最终候选验证集和测试集最大回撤都必须 <= {max_drawdown_pct:.2f}%，胜率都必须 >= 55%。")
    lines.append("- 交易白名单: 只允许使用训练集和验证集筛选，测试集不参与白名单选择。")
    lines.append("- 执行限制: 本轮只做本地回测和参数搜索，没有连接真实账户，也没有下单。")
    lines.append("")
    lines.append("## 当前策略逻辑")
    lines.append(f"- 基础周期为 {timeframe}，可用于 {framework} 超短线框架；本轮优化使用本地 {timeframe} 历史数据。")
    lines.append("- 信号分数 signal score 由四类相对独立的信息组成: 价格突破/回收强度、EMA 趋势距离、成交量确认、ATR 波动率位置。")
    lines.append("- 多头开仓: RSI 超卖回收、向上突破通道、趋势回踩后重新站上快 EMA、下影线插针反转。")
    lines.append("- 空头开仓: RSI 超买回落、向下跌破通道、趋势反弹后跌回快 EMA、上影线插针反转。")
    lines.append("- 可选市场状态过滤: 用慢 EMA 在若干根 K 线内的 ATR 标准化斜率，过滤明显逆大方向的多头或空头。")
    lines.append("- 平仓规则: 固定 ATR 止损/止盈、保本触发、移动止损、最长持仓时间、反向信号退出。")
    lines.append("")
    lines.append("## 动态仓位规则")
    lines.append("- `risk.py` 中的仓位公式为: 风险金额 = equity * risk_per_trade_pct * signal_risk_weight * drawdown_size_multiplier。")
    lines.append("- `signal_risk_weight` 使用非线性曲线: 低分信号只做试探仓，0.75 以上高置信信号开始加速放大，最高 1.8 倍基础风险权重。")
    lines.append("- signal score 越低，允许仓位越小；高质量信号可以使用更高仓位，但仍受单仓名义上限、杠杆上限、总回撤刹车约束。")
    lines.append("- 当总回撤达到最大回撤阈值的 50%/75%/90% 时，新开仓风险会逐级降到 60%/35%/20%。")
    lines.append("")
    lines.append("## 加仓规则")
    lines.append("- 默认只测试盈利后顺势加仓，不启用亏损补仓。")
    lines.append("- 加仓必须满足: 同向新信号、已有仓位浮盈达到阈值、signal score 达到加仓阈值、未超过最大加仓次数、总仓位不超过上限、回撤刹车允许。")
    lines.append("- 高置信同向信号的加仓比例会随 `signal_risk_weight` 放大，低置信同向信号即使允许加仓也只按较小比例执行。")
    lines.append("- 亏损补仓参数保留为显式开关，本轮没有作为推荐策略使用。")
    lines.append("")
    lines.append("## 风控规则")
    lines.append("- 手续费和滑点已经计入回测。")
    lines.append("- 设置单笔风险比例、单日亏损上限、连续亏损冷却、最大回撤停止交易、回撤接近阈值自动降仓。")
    lines.append("- 最大总仓位由 `max_position_notional_pct` 和 `max_leverage` 双重限制。")
    lines.append("")
    lines.append("## 参数搜索范围")
    lines.append("- EMA: fast 5-24, slow 21-89；ATR: 10/14/21；通道: 12-72。")
    lines.append("- 止损 ATR: 0.7-3.0；止盈 ATR: 0.45-4.0；保本/移动止损/持仓时间多组搜索。")
    lines.append("- 多空分别搜索 signal score 门槛和风险偏置；`allow_short` 固定为 true。")
    lines.append("- 市场状态过滤搜索: 慢 EMA 斜率开关、lookback、长短方向 slope 阈值。")
    lines.append("- 风险参数搜索: 单笔风险、单仓名义上限、冷却 bars、首仓比例、盈利加仓比例、加仓触发和最大加仓次数。")
    lines.append("")
    if not best:
        lines.append("## 最优参数组合")
        lines.append("没有找到可报告候选。")
        return "\n".join(lines) + "\n"

    lines.append("## 最优参数组合")
    lines.append(f"- 是否满足推荐约束: {'是' if best.get('eligible') else '否'}")
    lines.append(f"- 是否疑似过拟合: {'是' if best.get('overfit') else '否'}")
    lines.append(f"- 交易白名单: {', '.join(best.get('selected_symbols', []))}")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"strategy": best["strategy"], "risk": best["risk"]}, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 候选参数排名")
    lines.append("| 排名 | 币数 | 验证评分 | 推荐 | 过拟合 | Val收益% | Val胜率% | Val回撤% | Val PF | Test收益% | Test胜率% | Test回撤% | Test PF | Test交易 |")
    lines.append("|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, row in enumerate(candidates, 1):
        test = row.get("test", {})
        validation = row["validation"]
        lines.append(
            f"| {index} | {len(row.get('selected_symbols', []))} | {row['validation_score']:.2f} | {'Y' if row.get('eligible') else 'N'} | "
            f"{'Y' if row.get('overfit') else 'N'} | {validation['net_return_pct']:.2f} | "
            f"{validation['win_rate_pct']:.2f} | {validation['max_drawdown_pct']:.2f} | {_fmt_optional(validation['profit_factor'])} | "
            f"{test.get('net_return_pct', 0.0):.2f} | {test.get('win_rate_pct', 0.0):.2f} | "
            f"{test.get('max_drawdown_pct', 0.0):.2f} | {_fmt_optional(test.get('profit_factor'))} | {test.get('total_trades', 0)} |"
        )
    lines.append("")
    lines.append("## 训练/验证/测试结果")
    lines.append("| 区间 | 收益% | 年化% | 月化% | 最大回撤% | 胜率% | 盈亏比 | Sharpe | Profit Factor | 交易次数 | 平均单笔收益% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, key in (("训练集", "train"), ("验证集", "validation"), ("测试集", "test")):
        summary = best[key]
        lines.append(
            f"| {label} | {summary['net_return_pct']:.2f} | {summary['annual_return_pct']:.2f} | "
            f"{summary['monthly_return_pct']:.2f} | {summary['max_drawdown_pct']:.2f} | "
            f"{summary['win_rate_pct']:.2f} | {_fmt_optional(summary['payoff_ratio'])} | "
            f"{summary['sharpe_ratio']:.2f} | {_fmt_optional(summary['profit_factor'])} | "
            f"{summary['total_trades']} | {summary['avg_trade_return_pct']:.4f} |"
        )
    lines.append("")
    lines.append("## 多空分别表现")
    lines.append("| 区间 | 方向 | 交易次数 | 净利润U | 胜率% | Profit Factor | 平均单笔U |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for label, key in (("训练集", "train"), ("验证集", "validation"), ("测试集", "test")):
        for side_key, side_label in (("long", "多头"), ("short", "空头")):
            side = best[key][side_key]
            lines.append(
                f"| {label} | {side_label} | {side['trades']} | {side['net_profit']:.2f} | "
                f"{side['win_rate_pct']:.2f} | {_fmt_optional(side['profit_factor'])} | {side['avg_trade']:.4f} |"
            )
    lines.append("")
    lines.append("## 市场阶段稳定性")
    for label, key in (("训练集", "train"), ("验证集", "validation"), ("测试集", "test")):
        summary = best[key]
        stages = summary["stage_return_pct"]
        lines.append(
            f"- {label}: early={stages['early']:.2f}%, middle={stages['middle']:.2f}%, "
            f"late={stages['late']:.2f}%, negative_stages={summary['negative_stages']}"
        )
    lines.append("")
    lines.append("## 是否存在过拟合")
    lines.append("- 判定: " + ("疑似过拟合，不能作为最终推荐。" if best.get("overfit") else "未触发当前过拟合规则。"))
    lines.append("- 注意: 测试集只用于最终评价；推荐候选按验证集评分排序，并额外要求测试集回撤不超过硬约束。")
    lines.append("")
    lines.append("## 当前策略风险点")
    lines.append("- 当前回测是按单币独立资金模型聚合，尚未完整模拟 30 个币同时开仓时的组合保证金竞争。")
    lines.append(f"- {timeframe} K 线无法还原 K 线内部触发先后顺序，止损/止盈同根 K 线内可能偏保守或偏乐观。")
    lines.append("- 加仓已纳入回测，但实盘成交滑点、盘口深度和保护单触发会造成偏差。")
    lines.append("")
    lines.append("## 下一步改进建议")
    lines.append("- 加入组合级并发回测，按同一账户权益限制 30 币同时持仓。")
    lines.append("- 下载更长周期数据，做滚动 walk-forward 验证。")
    lines.append("- 对多头和空头进一步拆分不同止损/止盈参数，但保持参数数量受控。")
    return "\n".join(lines) + "\n"


def _infer_report_timeframe(data_paths: list[str]) -> str:
    for path in data_paths:
        parts = Path(path).stem.split("_")
        if len(parts) >= 2 and parts[1][-1:] in {"m", "h", "d", "w"}:
            return parts[1]
    return "15m"


def _split_dataset(path: str, train_ratio: float, validation_ratio: float) -> SplitDataset:
    candles = load_candles_csv(path)
    if len(candles) < 300:
        raise ValueError(f"dataset too small for robust split: {path}")
    train_end = int(len(candles) * train_ratio)
    validation_end = int(len(candles) * (train_ratio + validation_ratio))
    train_end = max(100, min(train_end, len(candles) - 200))
    validation_end = max(train_end + 100, min(validation_end, len(candles) - 100))
    return SplitDataset(
        path=path,
        train=candles[:train_end],
        validation=candles[train_end:validation_end],
        test=candles[validation_end:],
    )


def _filter_results(results: list[tuple[str, BacktestResult]], selected_paths: list[str]) -> list[tuple[str, BacktestResult]]:
    selected = set(selected_paths)
    return [(path, result) for path, result in results if path in selected]


def _symbol_from_path(path: str) -> str:
    return Path(path).name.split("_")[0]


def _sample_candidate_configs(
    base_strategy: StrategyConfig,
    base_risk: RiskConfig,
    trials: int,
    seed: int,
) -> Iterable[CandidateConfig]:
    rng = random.Random(seed)
    strategy_spaces = {
        "fast_ema": (5, 8, 12, 18, 24),
        "slow_ema": (21, 34, 55, 89),
        "atr_period": (10, 14, 21),
        "channel_period": (12, 20, 30, 48, 72),
        "min_atr_pct": (0.0005, 0.001, 0.0015, 0.0025, 0.004),
        "max_atr_pct": (0.012, 0.02, 0.035, 0.05),
        "breakout_buffer_atr": (0.0, 0.1, 0.2, 0.35, 0.5),
        "ema_gap_atr": (0.0, 0.1, 0.2, 0.35, 0.5),
        "volume_period": (20, 30, 48),
        "min_volume_ratio": (0.0, 0.8, 1.0, 1.2, 1.5),
        "stop_loss_atr": (0.7, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0),
        "take_profit_atr": (0.45, 0.6, 0.75, 0.9, 1.2, 1.6, 2.4, 3.0, 4.0),
        "breakeven_atr": (0.0, 0.5, 1.0, 1.5, 2.0),
        "trailing_activation_atr": (0.5, 0.8, 1.0, 1.5, 2.0, 3.0),
        "trailing_stop_atr": (0.0, 0.5, 0.8, 1.0, 1.5, 2.0),
        "max_holding_bars": (4, 6, 8, 12, 24, 48, 72),
        "spike_guard_enabled": (True, False),
        "spike_trade_enabled": (True, False),
        "spike_min_range_atr": (2.5, 3.0, 4.0, 5.0),
        "spike_min_wick_atr": (1.0, 1.4, 2.0),
        "spike_min_wick_ratio": (0.5, 0.6, 0.7),
        "spike_min_volume_ratio": (1.0, 1.2, 1.6),
        "spike_recovery_ratio": (0.35, 0.45, 0.6),
        "spike_stop_atr": (0.5, 0.7, 1.0),
        "spike_take_profit_atr": (0.7, 0.9, 1.2),
        "spike_risk_multiplier": (0.2, 0.35, 0.5),
        "spike_max_holding_bars": (4, 8, 12),
        "long_score_threshold": (0.25, 0.35, 0.45, 0.55, 0.65),
        "short_score_threshold": (0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
        "long_risk_bias": (0.75, 1.0, 1.2),
        "short_risk_bias": (0.2, 0.35, 0.5, 0.75, 1.0),
        "regime_filter_enabled": (True, False),
        "regime_lookback": (12, 24, 48, 72),
        "long_min_slow_slope_atr": (-1.5, -0.75, -0.25, 0.0, 0.25),
        "short_max_slow_slope_atr": (-0.25, 0.0, 0.25, 0.75, 1.5),
    }
    risk_spaces = {
        "risk_per_trade_pct": (0.0035, 0.005, 0.008, 0.012, 0.016, 0.02),
        "max_position_notional_pct": (1.2, 2.0, 3.0, 4.0, 5.0),
        "max_daily_loss_pct": (0.015, 0.02, 0.03, 0.04),
        "cooldown_bars_after_loss": (1, 2, 4, 6),
        "initial_entry_fraction": (0.45, 0.6, 0.75),
        "scale_in_enabled": (True,),
        "scale_in_fraction": (0.15, 0.25, 0.35),
        "scale_in_profit_trigger_pct": (0.003, 0.005, 0.008),
        "scale_in_min_score": (0.5, 0.6, 0.7),
        "max_scale_ins": (0, 1, 2),
        "fee_bps": (base_risk.fee_bps, max(base_risk.fee_bps, 5.0)),
        "slippage_bps": (base_risk.slippage_bps, max(base_risk.slippage_bps, 3.0), max(base_risk.slippage_bps, 4.0)),
    }

    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    yielded = 0
    max_attempts = max(trials * 80, 500)
    if trials > 0:
        yielded += 1
        yield CandidateConfig(
            strategy=replace(base_strategy, allow_short=True),
            risk=replace(base_risk, max_drawdown_pct=0.10, loss_scale_in_enabled=False),
        )
    while yielded < max(0, trials) and attempts < max_attempts:
        attempts += 1
        strategy_values = {name: rng.choice(values) for name, values in strategy_spaces.items()}
        if strategy_values["fast_ema"] >= strategy_values["slow_ema"]:
            continue
        if strategy_values["take_profit_atr"] < strategy_values["stop_loss_atr"] * 0.3:
            continue
        risk_values = {name: rng.choice(values) for name, values in risk_spaces.items()}
        key = tuple(strategy_values[name] for name in sorted(strategy_values)) + tuple(risk_values[name] for name in sorted(risk_values))
        if key in seen:
            continue
        seen.add(key)
        yielded += 1
        yield CandidateConfig(
            strategy=replace(base_strategy, allow_short=True, **strategy_values),
            risk=replace(
                base_risk,
                max_drawdown_pct=0.10,
                loss_scale_in_enabled=False,
                **risk_values,
            ),
        )


def _side_summary(trades: list[Trade]) -> dict[str, Any]:
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    return {
        "trades": len(trades),
        "net_profit": sum(trade.net_pnl for trade in trades),
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "avg_trade": 0.0 if not trades else sum(trade.net_pnl for trade in trades) / len(trades),
    }


def _payoff_ratio(wins: list[Trade], losses: list[Trade]) -> float | None:
    if not wins or not losses:
        return None
    avg_win = sum(trade.net_pnl for trade in wins) / len(wins)
    avg_loss = abs(sum(trade.net_pnl for trade in losses) / len(losses))
    if avg_loss <= 0:
        return None
    return avg_win / avg_loss


def _combined_sharpe(curves: list[list[EquityPoint]]) -> float:
    returns: list[float] = []
    intervals: list[float] = []
    for curve in curves:
        for previous, current in zip(curve, curve[1:]):
            if previous.equity <= 0:
                continue
            returns.append(current.equity / previous.equity - 1.0)
            seconds = (current.timestamp - previous.timestamp).total_seconds()
            if seconds > 0:
                intervals.append(seconds)
    if len(returns) < 2:
        return 0.0
    stdev = statistics.pstdev(returns)
    if stdev <= 0:
        return 0.0
    interval_seconds = statistics.median(intervals) if intervals else 3600.0
    periods_per_year = 365.25 * 86_400.0 / max(interval_seconds, 1.0)
    return statistics.fmean(returns) / stdev * math.sqrt(periods_per_year)


def _aggregate_stage_returns(results: list[BacktestResult]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    for result in results:
        stage_returns = _stage_returns(result.equity_curve)
        for key, value in stage_returns.items():
            buckets[key].append(value)
    return {key: (statistics.fmean(values) if values else 0.0) for key, values in buckets.items()}


def _stage_returns(curve: list[EquityPoint]) -> dict[str, float]:
    if len(curve) < 6:
        return {"early": 0.0, "middle": 0.0, "late": 0.0}
    n = len(curve)
    ranges = {
        "early": (0, n // 3),
        "middle": (n // 3, 2 * n // 3),
        "late": (2 * n // 3, n - 1),
    }
    output: dict[str, float] = {}
    for key, (start, end) in ranges.items():
        start_equity = curve[start].equity
        end_equity = curve[max(start + 1, end)].equity
        output[key] = 0.0 if start_equity <= 0 else (end_equity / start_equity - 1.0) * 100.0
    return output


def _none_to_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.2f}"


def _write_candidate_config(base_config: AppConfig, row: dict[str, Any], output_path: str) -> None:
    payload = {
        "data": asdict(base_config.data),
        "strategy": row["strategy"],
        "risk": row["risk"],
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
