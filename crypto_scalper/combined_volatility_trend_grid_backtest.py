from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .binance_client import SymbolRules
from .risk import BacktestExecutionConfig
from .trend_grid import TREND_GRID_STRATEGY_NAME, TrendGridConfig, TrendGridSnapshot
from .trend_grid_optimize import (
    GridCampaign,
    GridCandidate,
    GridPortfolioConfig,
    _close_campaign,
    _create_campaign,
    _mark_equity as _mark_grid_equity,
    _process_campaign_bar,
    build_grid_research_timeline,
    compact_grid_summary,
    simulate_grid_portfolio,
)
from .volatility_breakout import VOLATILITY_BREAKOUT_STRATEGY_NAME, VolatilityBreakoutConfig
from .volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    OpenPosition,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _candidate_sort_key,
    _entry_position,
    _exit_position,
    _load_runtime_inputs,
    _market_filter_reject_reason,
    _mark_equity as _mark_breakout_equity,
    _process_position_bar,
    build_candidates,
    compact_summary,
    minute_datetime,
    minute_token,
    sha256_file,
    simulate_portfolio,
)
from .volatility_breakout_v2_optimize import (
    build_market_context,
    enrich_candidates,
    filter_candidates,
)


BREAKOUT_KEY = "volatility_breakout"
GRID_KEY = "dynamic_trend_following_grid"
STRATEGY_KEYS = (BREAKOUT_KEY, GRID_KEY)
COMBINED_STRATEGY_NAME = "volatility_breakout_v2_balanced_plus_dynamic_trend_grid_v2"
NOTIONAL_HEADROOM_SAFETY = 0.999


@dataclass(frozen=True)
class CombinedPortfolioConfig:
    """Shared constraints applied above both unchanged strategy engines."""

    max_open_positions: int = 2
    max_gross_notional_multiple: float = 9.0
    hard_drawdown_stop_pct: float = 0.60
    allow_same_symbol_across_strategies: bool = False
    entry_priority: tuple[str, str] = (BREAKOUT_KEY, GRID_KEY)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CombinedPortfolioConfig:
        values = dict(payload)
        values["entry_priority"] = tuple(values.get("entry_priority", STRATEGY_KEYS))
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if self.max_gross_notional_multiple <= 0.0:
            raise ValueError("max_gross_notional_multiple must be positive")
        if not 0.0 < self.hard_drawdown_stop_pct <= 1.0:
            raise ValueError("hard_drawdown_stop_pct must be in (0, 1]")
        if len(self.entry_priority) != len(STRATEGY_KEYS) or set(self.entry_priority) != set(STRATEGY_KEYS):
            raise ValueError(f"entry_priority must contain exactly {STRATEGY_KEYS}")


def _combined_equity(
    cash: float,
    breakout_positions: dict[str, OpenPosition],
    grid_campaigns: dict[str, GridCampaign],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    minute: int,
    execution: BacktestExecutionConfig,
) -> float:
    return (
        cash
        + _mark_breakout_equity(0.0, breakout_positions, execution_data, minute, execution)
        + _mark_grid_equity(0.0, grid_campaigns, execution_data, rules, minute, execution)
    )


def _committed_notional(
    breakout_positions: dict[str, OpenPosition],
    grid_campaigns: dict[str, GridCampaign],
) -> float:
    breakout = sum(abs(position.quantity * position.entry_price) for position in breakout_positions.values())
    # Reserve all simultaneously fillable grid levels, not just currently filled lots.
    grid = sum(
        abs(level.quantity * level.raw_price)
        for campaign in grid_campaigns.values()
        for level in campaign.levels
    )
    return breakout + grid


def _available_notional_multiple(
    equity: float,
    breakout_positions: dict[str, OpenPosition],
    grid_campaigns: dict[str, GridCampaign],
    config: CombinedPortfolioConfig,
) -> float:
    if equity <= 0.0:
        return 0.0
    headroom = equity * config.max_gross_notional_multiple - _committed_notional(
        breakout_positions, grid_campaigns
    )
    return max(0.0, headroom / equity)


def _tag_trade(trade: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {"strategy": strategy, **trade}


def _trade_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if row["net_pnl"] > 0.0]
    losses = [row for row in rows if row["net_pnl"] <= 0.0]
    gross_profit = sum(row["net_pnl"] for row in wins)
    gross_loss = abs(sum(row["net_pnl"] for row in losses))
    net_pnl = sum(row["net_pnl"] for row in rows)
    fee = sum(row["fee"] for row in rows)
    slippage = sum(row["slippage"] for row in rows)
    funding = sum(row["funding"] for row in rows)
    return {
        "trade_count": len(rows),
        "net_pnl": net_pnl,
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0.0
            else (math.inf if gross_profit > 0.0 else 0.0)
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy_usdt": net_pnl / len(rows) if rows else 0.0,
        "expectancy_r": statistics.mean(row["pnl_r"] for row in rows) if rows else 0.0,
        "fee": fee,
        "slippage": slippage,
        "funding": funding,
        "full_cost": fee + slippage - funding,
    }


def _summarize_combined_result(
    initial_equity: float,
    final_equity: float,
    trades: list[dict[str, Any]],
    max_drawdown: float,
    max_drawdown_duration: int,
    candidate_counts: dict[str, int],
    rejected: dict[str, dict[str, int]],
    hard_stopped: bool,
    hard_stopped_at: str | None,
    max_concurrent_positions: int,
    concurrency_minutes: dict[int, int],
    position_minutes_by_strategy: dict[str, int],
    max_entry_committed_notional_multiple: float,
    daily_equity: dict[str, float],
    combined_config: CombinedPortfolioConfig,
    breakout_signal_config: VolatilityBreakoutConfig,
    breakout_portfolio_config: PortfolioSearchConfig,
    grid_signal_config: TrendGridConfig,
    grid_portfolio_config: GridPortfolioConfig,
) -> dict[str, Any]:
    overall = _trade_stats(trades)
    net_profit = final_equity - initial_equity
    by_strategy = {
        strategy: _trade_stats([row for row in trades if row["strategy"] == strategy])
        for strategy in STRATEGY_KEYS
    }
    by_month = {
        month: _trade_stats([row for row in trades if row["exit_time"][:7] == month])
        for month in sorted({row["exit_time"][:7] for row in trades})
    }
    total_minutes = sum(concurrency_minutes.values())
    sorted_wins = sorted((row["net_pnl"] for row in trades if row["net_pnl"] > 0.0), reverse=True)
    raw_gross_profit = sum(max(0.0, row["gross_pnl"]) for row in trades)
    return {
        "strategy": COMBINED_STRATEGY_NAME,
        "global_portfolio_config": asdict(combined_config),
        "source_strategy_configs": {
            BREAKOUT_KEY: {
                "signal": breakout_signal_config.as_dict(),
                "portfolio": asdict(breakout_portfolio_config),
            },
            GRID_KEY: {
                "signal": grid_signal_config.as_dict(),
                "portfolio": asdict(grid_portfolio_config),
            },
        },
        "candidate_count": sum(candidate_counts.values()),
        "candidate_count_by_strategy": candidate_counts,
        "trade_count": overall["trade_count"],
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_profit": net_profit,
        "return_pct": net_profit / initial_equity if initial_equity else 0.0,
        "win_rate": overall["win_rate"],
        "profit_factor": overall["profit_factor"],
        "expectancy_usdt": overall["expectancy_usdt"],
        "expectancy_r": overall["expectancy_r"],
        "max_drawdown_pct": max_drawdown,
        "max_drawdown_duration_minutes": max_drawdown_duration,
        "fee": overall["fee"],
        "slippage": overall["slippage"],
        "funding": overall["funding"],
        "full_cost": overall["full_cost"],
        "cost_to_raw_gross_profit_ratio": (
            overall["full_cost"] / raw_gross_profit if raw_gross_profit > 0.0 else math.inf
        ),
        "top5_profit_contribution": sum(sorted_wins[:5]) / net_profit if net_profit > 0.0 else math.inf,
        "hard_drawdown_stopped": hard_stopped,
        "hard_drawdown_stopped_at": hard_stopped_at,
        "max_concurrent_positions": max_concurrent_positions,
        "max_entry_committed_notional_multiple": max_entry_committed_notional_multiple,
        "concurrency_minutes": {str(key): value for key, value in sorted(concurrency_minutes.items())},
        "concurrency_share": {
            str(key): value / total_minutes if total_minutes else 0.0
            for key, value in sorted(concurrency_minutes.items())
        },
        "position_minutes_by_strategy": position_minutes_by_strategy,
        "pnl_reconciliation_error": net_profit - overall["net_pnl"],
        "rejected": rejected,
        "by_strategy": by_strategy,
        "by_month": by_month,
        "daily_equity": daily_equity,
        "trades": trades,
    }


def simulate_combined_portfolio(
    breakout_candidates: dict[int, list[Candidate]],
    grid_candidates: dict[int, list[GridCandidate]],
    grid_snapshots: dict[str, dict[int, TrendGridSnapshot]],
    symbols: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    breakout_signal_config: VolatilityBreakoutConfig,
    breakout_portfolio_config: PortfolioSearchConfig,
    grid_signal_config: TrendGridConfig,
    grid_portfolio_config: GridPortfolioConfig,
    combined_config: CombinedPortfolioConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    combined_config.validate()
    if start >= end:
        raise ValueError("start must be before end")

    start_minute = minute_token(start)
    end_minute = minute_token(end)
    symbol_set = frozenset(symbols)
    cash = float(initial_equity)
    breakout_positions: dict[str, OpenPosition] = {}
    grid_campaigns: dict[str, GridCampaign] = {}
    trades: list[dict[str, Any]] = []
    breakout_cooldown_until: dict[str, int] = {}
    grid_cooldown_until: dict[str, int] = {}
    breakout_daily_entries: dict[str, int] = defaultdict(int)
    grid_daily_entries: dict[str, int] = defaultdict(int)
    rejected: dict[str, dict[str, int]] = {
        BREAKOUT_KEY: defaultdict(int),
        GRID_KEY: defaultdict(int),
    }
    peak_equity = cash
    max_drawdown = 0.0
    max_drawdown_duration = 0
    drawdown_start: int | None = None
    hard_stopped = False
    hard_stopped_at: str | None = None
    max_concurrent_positions = 0
    max_entry_committed_notional_multiple = 0.0
    concurrency_minutes: dict[int, int] = defaultdict(int)
    position_minutes_by_strategy: dict[str, int] = defaultdict(int)
    daily_equity: dict[str, float] = {}

    def open_count() -> int:
        return len(breakout_positions) + len(grid_campaigns)

    def symbol_is_open(symbol: str) -> bool:
        return symbol in breakout_positions or symbol in grid_campaigns

    def observe_entry_exposure(entry_equity: float) -> None:
        nonlocal max_concurrent_positions, max_entry_committed_notional_multiple
        max_concurrent_positions = max(max_concurrent_positions, open_count())
        if entry_equity > 0.0:
            max_entry_committed_notional_multiple = max(
                max_entry_committed_notional_multiple,
                _committed_notional(breakout_positions, grid_campaigns) / entry_equity,
            )

    for minute in range(start_minute, end_minute):
        # Both source engines preserve their original rule: exits are processed before new entries.
        for symbol, position in list(breakout_positions.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            closed = _process_position_bar(
                position,
                minute,
                series,
                index,
                breakout_signal_config,
                execution,
                rules[symbol],
            )
            if closed is None:
                continue
            trade, cash_delta = closed
            cash += cash_delta
            trades.append(_tag_trade(trade, BREAKOUT_KEY))
            breakout_positions.pop(symbol, None)
            breakout_cooldown_until[symbol] = minute + breakout_portfolio_config.symbol_cooldown_minutes

        for symbol, campaign in list(grid_campaigns.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            cash_delta, close_reason = _process_campaign_bar(
                campaign,
                minute,
                series,
                index,
                grid_snapshots.get(symbol, {}).get(minute),
                grid_signal_config,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if close_reason is None:
                continue
            report = getattr(campaign, "_pending_report", None)
            if report is not None:
                trades.append(_tag_trade(report, GRID_KEY))
            else:
                rejected[GRID_KEY]["campaign_without_fill"] += 1
            grid_campaigns.pop(symbol, None)
            grid_cooldown_until[symbol] = minute + grid_portfolio_config.symbol_cooldown_minutes

        day_key = minute_datetime(minute).date().isoformat()
        if not hard_stopped:
            for strategy in combined_config.entry_priority:
                if strategy == BREAKOUT_KEY:
                    rows = sorted(
                        breakout_candidates.get(minute, ()),
                        key=lambda row: _candidate_sort_key(
                            row, breakout_portfolio_config.ranking_mode
                        ),
                    )
                    for candidate in rows:
                        symbol = candidate.signal.symbol
                        if symbol not in symbol_set or symbol not in execution_data:
                            rejected[BREAKOUT_KEY]["outside_universe_or_missing_data"] += 1
                            continue
                        market_reject = _market_filter_reject_reason(
                            candidate, breakout_portfolio_config
                        )
                        if market_reject is not None:
                            rejected[BREAKOUT_KEY][market_reject] += 1
                            continue
                        if len(breakout_positions) >= breakout_portfolio_config.max_open_positions:
                            rejected[BREAKOUT_KEY]["strategy_position_limit"] += 1
                            continue
                        if open_count() >= combined_config.max_open_positions:
                            rejected[BREAKOUT_KEY]["global_position_limit"] += 1
                            continue
                        if (
                            not combined_config.allow_same_symbol_across_strategies
                            and symbol_is_open(symbol)
                        ):
                            rejected[BREAKOUT_KEY]["global_symbol_already_open"] += 1
                            continue
                        if symbol in breakout_positions:
                            rejected[BREAKOUT_KEY]["symbol_already_open"] += 1
                            continue
                        if breakout_cooldown_until.get(symbol, -1) > minute:
                            rejected[BREAKOUT_KEY]["symbol_cooldown"] += 1
                            continue
                        if breakout_daily_entries[day_key] >= breakout_portfolio_config.max_daily_trades:
                            rejected[BREAKOUT_KEY]["daily_trade_limit"] += 1
                            continue
                        entry_equity = _combined_equity(
                            cash,
                            breakout_positions,
                            grid_campaigns,
                            execution_data,
                            rules,
                            minute,
                            execution,
                        )
                        available_multiple = _available_notional_multiple(
                            entry_equity,
                            breakout_positions,
                            grid_campaigns,
                            combined_config,
                        )
                        if available_multiple <= 0.0:
                            rejected[BREAKOUT_KEY]["global_notional_limit"] += 1
                            continue
                        entry_portfolio = replace(
                            breakout_portfolio_config,
                            max_notional_multiple=min(
                                breakout_portfolio_config.max_notional_multiple,
                                available_multiple * NOTIONAL_HEADROOM_SAFETY,
                            ),
                        )
                        opened = _entry_position(
                            candidate,
                            execution_data[symbol],
                            rules[symbol],
                            breakout_signal_config,
                            entry_portfolio,
                            execution,
                            entry_equity if breakout_portfolio_config.compound else initial_equity,
                        )
                        if opened is None:
                            rejected[BREAKOUT_KEY]["sizing_or_data"] += 1
                            continue
                        position, entry_fee = opened
                        breakout_positions[symbol] = position
                        cash -= entry_fee
                        breakout_daily_entries[day_key] += 1
                        observe_entry_exposure(entry_equity)

                        index = execution_data[symbol].index_at(minute)
                        if index is not None:
                            closed = _process_position_bar(
                                position,
                                minute,
                                execution_data[symbol],
                                index,
                                breakout_signal_config,
                                execution,
                                rules[symbol],
                            )
                            if closed is not None:
                                trade, cash_delta = closed
                                cash += cash_delta
                                trades.append(_tag_trade(trade, BREAKOUT_KEY))
                                breakout_positions.pop(symbol, None)
                                breakout_cooldown_until[symbol] = (
                                    minute + breakout_portfolio_config.symbol_cooldown_minutes
                                )
                        if (
                            len(breakout_positions) >= breakout_portfolio_config.max_open_positions
                            or open_count() >= combined_config.max_open_positions
                        ):
                            break

                elif strategy == GRID_KEY:
                    for candidate in grid_candidates.get(minute, ()):
                        symbol = candidate.signal.symbol
                        if symbol not in symbol_set or symbol not in execution_data:
                            rejected[GRID_KEY]["outside_universe_or_missing_data"] += 1
                            continue
                        if len(grid_campaigns) >= grid_portfolio_config.max_open_campaigns:
                            rejected[GRID_KEY]["strategy_position_limit"] += 1
                            continue
                        if open_count() >= combined_config.max_open_positions:
                            rejected[GRID_KEY]["global_position_limit"] += 1
                            continue
                        if (
                            not combined_config.allow_same_symbol_across_strategies
                            and symbol_is_open(symbol)
                        ):
                            rejected[GRID_KEY]["global_symbol_already_open"] += 1
                            continue
                        if symbol in grid_campaigns:
                            rejected[GRID_KEY]["symbol_campaign_open"] += 1
                            continue
                        if grid_cooldown_until.get(symbol, -1) > minute:
                            rejected[GRID_KEY]["symbol_cooldown"] += 1
                            continue
                        if grid_daily_entries[day_key] >= grid_portfolio_config.max_daily_campaigns:
                            rejected[GRID_KEY]["daily_campaign_limit"] += 1
                            continue
                        entry_equity = _combined_equity(
                            cash,
                            breakout_positions,
                            grid_campaigns,
                            execution_data,
                            rules,
                            minute,
                            execution,
                        )
                        available_multiple = _available_notional_multiple(
                            entry_equity,
                            breakout_positions,
                            grid_campaigns,
                            combined_config,
                        )
                        if available_multiple <= 0.0:
                            rejected[GRID_KEY]["global_notional_limit"] += 1
                            continue
                        entry_portfolio = replace(
                            grid_portfolio_config,
                            max_notional_multiple=min(
                                grid_portfolio_config.max_notional_multiple,
                                available_multiple * NOTIONAL_HEADROOM_SAFETY,
                            ),
                        )
                        opened = _create_campaign(
                            candidate,
                            execution_data[symbol],
                            rules[symbol],
                            grid_signal_config,
                            entry_portfolio,
                            execution,
                            entry_equity if grid_portfolio_config.compound else initial_equity,
                        )
                        if opened is None:
                            rejected[GRID_KEY]["sizing_or_structure"] += 1
                            continue
                        campaign, initial_fee = opened
                        grid_campaigns[symbol] = campaign
                        cash -= initial_fee
                        grid_daily_entries[day_key] += 1
                        observe_entry_exposure(entry_equity)

                        index = execution_data[symbol].index_at(minute)
                        if index is not None:
                            cash_delta, close_reason = _process_campaign_bar(
                                campaign,
                                minute,
                                execution_data[symbol],
                                index,
                                grid_snapshots.get(symbol, {}).get(minute),
                                grid_signal_config,
                                execution,
                                rules[symbol],
                            )
                            cash += cash_delta
                            if close_reason is not None:
                                report = getattr(campaign, "_pending_report", None)
                                if report is not None:
                                    trades.append(_tag_trade(report, GRID_KEY))
                                grid_campaigns.pop(symbol, None)
                                grid_cooldown_until[symbol] = (
                                    minute + grid_portfolio_config.symbol_cooldown_minutes
                                )
                        if (
                            len(grid_campaigns) >= grid_portfolio_config.max_open_campaigns
                            or open_count() >= combined_config.max_open_positions
                        ):
                            break

        equity = _combined_equity(
            cash,
            breakout_positions,
            grid_campaigns,
            execution_data,
            rules,
            minute,
            execution,
        )
        concurrent = open_count()
        max_concurrent_positions = max(max_concurrent_positions, concurrent)
        concurrency_minutes[concurrent] += 1
        position_minutes_by_strategy[BREAKOUT_KEY] += len(breakout_positions)
        position_minutes_by_strategy[GRID_KEY] += len(grid_campaigns)
        daily_equity[day_key] = equity

        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 1.0
        if drawdown > 0.0:
            drawdown_start = minute if drawdown_start is None else drawdown_start
            max_drawdown_duration = max(max_drawdown_duration, minute - drawdown_start)
        else:
            drawdown_start = None
        max_drawdown = max(max_drawdown, drawdown)
        if not hard_stopped and (
            drawdown >= combined_config.hard_drawdown_stop_pct or equity <= 0.0
        ):
            hard_stopped = True
            hard_stopped_at = minute_datetime(minute).isoformat()

    force_minute = end_minute - 1
    for symbol, position in list(breakout_positions.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        trade, cash_delta = _exit_position(
            position,
            series.closes[index],
            "end_of_backtest",
            int(series.minutes[index]),
            execution,
            rules[symbol],
        )
        cash += cash_delta
        trades.append(_tag_trade(trade, BREAKOUT_KEY))

    for symbol, campaign in list(grid_campaigns.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        report, cash_delta = _close_campaign(
            campaign,
            float(series.closes[index]),
            int(series.minutes[index]),
            "end_of_backtest",
            execution,
            rules[symbol],
        )
        cash += cash_delta
        if report is not None:
            trades.append(_tag_trade(report, GRID_KEY))

    if daily_equity:
        daily_equity[minute_datetime(force_minute).date().isoformat()] = cash

    return _summarize_combined_result(
        initial_equity,
        cash,
        trades,
        max_drawdown,
        max_drawdown_duration,
        {
            BREAKOUT_KEY: sum(len(rows) for rows in breakout_candidates.values()),
            GRID_KEY: sum(len(rows) for rows in grid_candidates.values()),
        },
        {strategy: dict(values) for strategy, values in rejected.items()},
        hard_stopped,
        hard_stopped_at,
        max_concurrent_positions,
        dict(concurrency_minutes),
        dict(position_minutes_by_strategy),
        max_entry_committed_notional_multiple,
        daily_equity,
        combined_config,
        breakout_signal_config,
        breakout_portfolio_config,
        grid_signal_config,
        grid_portfolio_config,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_pf(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    combined = report["combined_result"]
    standalone = report["standalone_reference"]
    by_strategy = combined["by_strategy"]
    lines = [
        "# Volatility Breakout + Dynamic Trend-Following Grid (Max 2)",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial shared equity: `{report['initial_equity']:.2f}U`",
        "- One shared account; at most two campaigns/positions total and at most one per source strategy",
        "- No simultaneous duplicate symbol; 9x aggregate committed-notional entry cap",
        "- Closed 60m signal, next 1m open, conservative stop-first path, full fees/slippage/funding",
        "- Historical in-sample research only; not an untouched holdout or live-return forecast",
        "",
        "| Run | Trades | Final equity | Net | Return | PF | Win rate | Max DD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("Volatility Breakout standalone", standalone[BREAKOUT_KEY]),
        ("Trend Grid standalone", standalone[GRID_KEY]),
        ("Combined max-2 shared account", combined),
    ):
        lines.append(
            f"| {label} | {row['trade_count']} | {row['final_equity']:.2f}U | "
            f"{row['net_profit']:+.2f}U | {row['return_pct']:.2%} | "
            f"{_format_pf(row['profit_factor'])} | {row['win_rate']:.2%} | "
            f"{row['max_drawdown_pct']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Combined-account attribution",
            "",
            "| Strategy | Closed campaigns/trades | Realized net | PF | Win rate | Full cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGY_KEYS:
        row = by_strategy[strategy]
        lines.append(
            f"| {strategy} | {row['trade_count']} | {row['net_pnl']:+.2f}U | "
            f"{_format_pf(row['profit_factor'])} | {row['win_rate']:.2%} | {row['full_cost']:.2f}U |"
        )
    comparison = report["comparison_vs_breakout_standalone"]
    lines.extend(
        [
            "",
            "## Shared constraints and interaction",
            "",
            f"- Observed maximum concurrent positions: `{combined['max_concurrent_positions']}`.",
            f"- Observed maximum committed-notional multiple at entry: "
            f"`{combined['max_entry_committed_notional_multiple']:.3f}x`.",
            f"- Two-position occupancy: `{combined['concurrency_share'].get('2', 0.0):.2%}` of test minutes.",
            f"- Incremental net versus the same-200U Breakout-only run: "
            f"`{comparison['net_profit_delta']:+.2f}U`.",
            f"- Max-drawdown change versus Breakout-only: "
            f"`{comparison['max_drawdown_pct_delta']:+.2%}`.",
            f"- Reversed entry-priority sensitivity net: "
            f"`{report['priority_sensitivity']['net_profit']:+.2f}U` "
            f"(PF `{_format_pf(report['priority_sensitivity']['profit_factor'])}`).",
            "",
            "Standalone rows each start with 200U and are references, not additive sleeves. "
            "The combined row is the executable shared-capital simulation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact_combined(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trade_count",
        "initial_equity",
        "final_equity",
        "net_profit",
        "return_pct",
        "win_rate",
        "profit_factor",
        "expectancy_usdt",
        "expectancy_r",
        "max_drawdown_pct",
        "max_drawdown_duration_minutes",
        "fee",
        "slippage",
        "funding",
        "full_cost",
        "hard_drawdown_stopped",
        "max_concurrent_positions",
        "max_entry_committed_notional_multiple",
        "concurrency_share",
        "by_strategy",
        "by_month",
    )
    return {key: result[key] for key in keys}


def _verify_shadow_matches_balanced(
    shadow_path: Path,
    signal_config: VolatilityBreakoutConfig,
    portfolio_config: PortfolioSearchConfig,
) -> None:
    payload = json.loads(shadow_path.read_text(encoding="utf-8"))["dual_thrust_shadow"]
    expected = {**signal_config.as_dict(), **asdict(portfolio_config)}
    mismatches = {
        key: {"shadow": payload[key], "balanced": value}
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if mismatches:
        raise RuntimeError(f"frozen shadow and balanced source config differ: {mismatches}")


def run_combined_backtest(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data"]
    period = config["period"]
    runtime_args = argparse.Namespace(
        signal_data_dir=data["signal_data_dir"],
        execution_data_dir=data["execution_data_dir"],
        funding_data_dir=data["funding_data_dir"],
        cost_config=data["cost_config"],
        start=period["start"],
        end=period["end"],
    )
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(
        runtime_args
    )

    sources = config["source_configs"]
    breakout_path = Path(sources[BREAKOUT_KEY]["path"])
    breakout_payload = json.loads(breakout_path.read_text(encoding="utf-8"))
    breakout_signal_config = VolatilityBreakoutConfig(**breakout_payload["balanced_signal"])
    breakout_portfolio_config = PortfolioSearchConfig(**breakout_payload["balanced_portfolio"])
    shadow_path = Path(sources[BREAKOUT_KEY]["frozen_shadow_path"])
    _verify_shadow_matches_balanced(
        shadow_path, breakout_signal_config, breakout_portfolio_config
    )

    grid_path = Path(sources[GRID_KEY]["path"])
    grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
    grid_signal_config = TrendGridConfig(**grid_payload["signal"])
    grid_signal_config.validate()
    grid_portfolio_config = GridPortfolioConfig(**grid_payload["portfolio"])

    symbols = tuple(breakout_payload.get("symbols", UNIVERSE_50))
    if set(symbols) != set(grid_payload["symbols"]):
        raise RuntimeError("source strategies must use the same symbol universe")
    combined_config = CombinedPortfolioConfig.from_dict(config["portfolio"])
    initial_equity = float(config["initial_equity"])

    print("building Volatility Breakout candidates", flush=True)
    raw_breakout_candidates = build_candidates(
        symbols,
        signal_data,
        execution_data,
        breakout_signal_config,
        start,
        end,
    )
    market_context = build_market_context(symbols, signal_data)
    breakout_candidates = filter_candidates(
        enrich_candidates(raw_breakout_candidates, market_context),
        breakout_signal_config,
    )
    print("building Dynamic Trend-Following Grid timeline", flush=True)
    grid_candidates, grid_snapshots = build_grid_research_timeline(
        symbols,
        signal_data,
        execution_data,
        grid_signal_config,
        start,
        end,
    )

    print("running standalone references", flush=True)
    standalone_breakout = simulate_portfolio(
        breakout_candidates,
        symbols,
        execution_data,
        rules,
        breakout_signal_config,
        breakout_portfolio_config,
        execution,
        start,
        end,
        initial_equity,
    )
    standalone_grid = simulate_grid_portfolio(
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        grid_signal_config,
        grid_portfolio_config,
        execution,
        start,
        end,
        initial_equity,
    )

    print("running shared-account max-2 combination", flush=True)
    combined = simulate_combined_portfolio(
        breakout_candidates,
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        breakout_signal_config,
        breakout_portfolio_config,
        grid_signal_config,
        grid_portfolio_config,
        combined_config,
        execution,
        start,
        end,
        initial_equity,
    )
    reversed_priority = replace(
        combined_config, entry_priority=tuple(reversed(combined_config.entry_priority))
    )
    print("running reversed-priority sensitivity", flush=True)
    priority_sensitivity = simulate_combined_portfolio(
        breakout_candidates,
        grid_candidates,
        grid_snapshots,
        symbols,
        execution_data,
        rules,
        breakout_signal_config,
        breakout_portfolio_config,
        grid_signal_config,
        grid_portfolio_config,
        reversed_priority,
        execution,
        start,
        end,
        initial_equity,
    )

    standalone_reference = {
        BREAKOUT_KEY: compact_summary(standalone_breakout),
        GRID_KEY: compact_grid_summary(standalone_grid),
    }
    report = {
        "strategy_name": COMBINED_STRATEGY_NAME,
        "research_status": "historical_in_sample_combination_not_untouched_oos",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": initial_equity,
        "symbols": list(symbols),
        "source_configs": {
            BREAKOUT_KEY: {
                **sources[BREAKOUT_KEY],
                "sha256": sha256_file(breakout_path),
                "frozen_shadow_sha256": sha256_file(shadow_path),
            },
            GRID_KEY: {
                **sources[GRID_KEY],
                "sha256": sha256_file(grid_path),
            },
        },
        "cost_model": {
            "source_config": data["cost_config"],
            "source_config_sha256": sha256_file(data["cost_config"]),
            "mode": execution.mode,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
            "funding_missing_symbols": sorted(set(symbols) - set(metadata["funding"])),
        },
        "execution_rules": {
            "signal": "closed_60m",
            "entry": "next_1m_open",
            "same_bar_conflict": "stop_before_take_profit",
            "exits_before_entries": True,
            "shared_account": True,
            "global_position_limit": combined_config.max_open_positions,
            "same_symbol_concurrent_positions": combined_config.allow_same_symbol_across_strategies,
            "aggregate_committed_notional_entry_cap": combined_config.max_gross_notional_multiple,
        },
        "standalone_reference": standalone_reference,
        "combined_result": combined,
        "priority_sensitivity": _compact_combined(priority_sensitivity),
        "comparison_vs_breakout_standalone": {
            "final_equity_delta": combined["final_equity"] - standalone_breakout["final_equity"],
            "net_profit_delta": combined["net_profit"] - standalone_breakout["net_profit"],
            "return_pct_delta": combined["return_pct"] - standalone_breakout["return_pct"],
            "max_drawdown_pct_delta": (
                combined["max_drawdown_pct"] - standalone_breakout["max_drawdown_pct"]
            ),
            "profit_factor_delta": combined["profit_factor"] - standalone_breakout["profit_factor"],
        },
    }

    output = Path(config["output"]["report"])
    summary = Path(config["output"]["summary"])
    manifest = Path(config["output"]["manifest"])
    _write_json(output, report)
    _write_summary(summary, report)
    artifacts = (
        path,
        output,
        summary,
        breakout_path,
        shadow_path,
        grid_path,
        Path("crypto_scalper/combined_volatility_trend_grid_backtest.py"),
        Path("tests/test_combined_volatility_trend_grid_backtest.py"),
    )
    _write_json(
        manifest,
        {
            "strategy_name": COMBINED_STRATEGY_NAME,
            "status": "historical_research_frozen_separately_from_source_strategies",
            "source_strategies_modified": False,
            "report": str(output),
            "summary": str(summary),
            "hashes": {
                str(artifact): sha256_file(artifact)
                for artifact in artifacts
                if artifact.exists()
            },
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest frozen Volatility Breakout and Dynamic Trend Grid in one max-2 account"
    )
    parser.add_argument(
        "--config",
        default="config.combined-volatility-trend-grid.max2.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_combined_backtest(args.config)
    output = {
        "standalone_reference": report["standalone_reference"],
        "combined": _compact_combined(report["combined_result"]),
        "comparison_vs_breakout_standalone": report["comparison_vs_breakout_standalone"],
        "priority_sensitivity": report["priority_sensitivity"],
    }
    print(json.dumps(_json_safe(output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
