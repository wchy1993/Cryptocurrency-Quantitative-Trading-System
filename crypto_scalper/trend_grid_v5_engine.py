from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from .binance_client import SymbolRules
from .risk import BacktestExecutionConfig
from .trend_grid import TrendGridConfig, TrendGridSnapshot
from .trend_grid_optimize import (
    CompactSeries,
    GridCampaign,
    GridCandidate,
    GridPortfolioConfig,
    _close_campaign,
    _create_campaign,
    _mark_equity,
    _process_campaign_bar,
    minute_datetime,
    minute_token,
    summarize_grid_result,
)


GRID_V5_ENGINE_NAME = "dynamic_trend_grid_v5_managed_campaigns"


@dataclass(frozen=True)
class GridV5ExecutionProfile:
    tier: str
    signal: TrendGridConfig
    portfolio: GridPortfolioConfig


@dataclass
class _ManagedCampaign:
    campaign: GridCampaign
    profile: GridV5ExecutionProfile


ProfileSelector = Callable[
    [GridCandidate, int, float], Optional[GridV5ExecutionProfile]
]


def _tier_metrics(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for trade in trades:
        tier = str(trade.get("v5_tier", "unknown"))
        row = output.setdefault(
            tier,
            {"trade_count": 0, "wins": 0, "net_pnl": 0.0,
             "gross_profit": 0.0, "gross_loss": 0.0},
        )
        pnl = float(trade["net_pnl"])
        row["trade_count"] += 1
        row["wins"] += int(pnl > 0.0)
        row["net_pnl"] += pnl
        if pnl > 0.0:
            row["gross_profit"] += pnl
        else:
            row["gross_loss"] += abs(pnl)
    for row in output.values():
        count = int(row["trade_count"])
        loss = float(row["gross_loss"])
        row["win_rate"] = row.pop("wins") / count if count else 0.0
        row["profit_factor"] = (
            row["gross_profit"] / loss
            if loss > 0.0
            else (float("inf") if row["gross_profit"] > 0.0 else 0.0)
        )
    return output


def simulate_grid_v5_portfolio(
    candidates: dict[int, list[GridCandidate]],
    snapshots: dict[str, dict[int, TrendGridSnapshot]],
    universe: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    operational_signal: TrendGridConfig,
    operational_portfolio: GridPortfolioConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    profile_selector: ProfileSelector | None = None,
    skip_symbols: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Grid simulator with a profile frozen independently at campaign entry."""

    start_minute = minute_token(start)
    end_minute = minute_token(end)
    cash = float(initial_equity)
    campaigns: dict[str, _ManagedCampaign] = {}
    trades: list[dict[str, Any]] = []
    cooldown_until: dict[str, int] = {}
    daily_entries: dict[str, int] = defaultdict(int)
    rejected: dict[str, int] = defaultdict(int)
    peak_equity = cash
    max_drawdown = 0.0
    max_drawdown_duration = 0
    drawdown_start: int | None = None
    hard_stopped = False
    universe_set = frozenset(universe)
    default_profile = GridV5ExecutionProfile(
        "default", operational_signal, operational_portfolio
    )

    def plain_campaigns() -> dict[str, GridCampaign]:
        return {symbol: managed.campaign for symbol, managed in campaigns.items()}

    def append_report(managed: _ManagedCampaign) -> bool:
        report = getattr(managed.campaign, "_pending_report", None)
        if report is None:
            rejected["campaign_without_fill"] += 1
            return False
        report["v5_tier"] = managed.profile.tier
        trades.append(report)
        return True

    for minute in range(start_minute, end_minute):
        for symbol, managed in list(campaigns.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            cash_delta, close_reason = _process_campaign_bar(
                managed.campaign,
                minute,
                series,
                index,
                snapshots.get(symbol, {}).get(minute),
                managed.profile.signal,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if close_reason is None:
                continue
            append_report(managed)
            campaigns.pop(symbol, None)
            cooldown_until[symbol] = (
                minute + operational_portfolio.symbol_cooldown_minutes
            )

        equity_before_entry = _mark_equity(
            cash, plain_campaigns(), execution_data, rules, minute, execution
        )
        day_key = minute_datetime(minute).date().isoformat()
        if not hard_stopped:
            for candidate in candidates.get(minute, ()):
                symbol = candidate.signal.symbol
                if symbol not in universe_set or symbol in skip_symbols:
                    continue
                if symbol in campaigns:
                    rejected["symbol_campaign_open"] += 1
                    continue
                if cooldown_until.get(symbol, -1) > minute:
                    rejected["symbol_cooldown"] += 1
                    continue
                if len(campaigns) >= operational_portfolio.max_open_campaigns:
                    rejected["campaign_limit"] += 1
                    continue
                if daily_entries[day_key] >= operational_portfolio.max_daily_campaigns:
                    rejected["daily_campaign_limit"] += 1
                    continue
                profile = (
                    profile_selector(candidate, minute, equity_before_entry)
                    if profile_selector is not None
                    else default_profile
                )
                if profile is None:
                    rejected["profile_rejected"] += 1
                    continue
                opened = _create_campaign(
                    candidate,
                    execution_data[symbol],
                    rules[symbol],
                    profile.signal,
                    profile.portfolio,
                    execution,
                    equity_before_entry
                    if profile.portfolio.compound
                    else initial_equity,
                )
                if opened is None:
                    rejected["sizing_or_structure"] += 1
                    continue
                campaign, initial_fee = opened
                managed = _ManagedCampaign(campaign, profile)
                campaigns[symbol] = managed
                cash -= initial_fee
                daily_entries[day_key] += 1
                index = execution_data[symbol].index_at(minute)
                if index is not None:
                    cash_delta, close_reason = _process_campaign_bar(
                        campaign,
                        minute,
                        execution_data[symbol],
                        index,
                        snapshots.get(symbol, {}).get(minute),
                        profile.signal,
                        execution,
                        rules[symbol],
                    )
                    cash += cash_delta
                    if close_reason is not None:
                        append_report(managed)
                        campaigns.pop(symbol, None)
                        cooldown_until[symbol] = (
                            minute + operational_portfolio.symbol_cooldown_minutes
                        )
                if len(campaigns) >= operational_portfolio.max_open_campaigns:
                    break

        equity = _mark_equity(
            cash, plain_campaigns(), execution_data, rules, minute, execution
        )
        peak_equity = max(peak_equity, equity)
        drawdown = (
            (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 1.0
        )
        if drawdown > 0.0:
            drawdown_start = minute if drawdown_start is None else drawdown_start
            max_drawdown_duration = max(
                max_drawdown_duration, minute - drawdown_start
            )
        else:
            drawdown_start = None
        max_drawdown = max(max_drawdown, drawdown)
        if (
            drawdown >= operational_portfolio.hard_drawdown_stop_pct
            or equity <= 0.0
        ):
            hard_stopped = True

    force_minute = end_minute - 1
    for symbol, managed in list(campaigns.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        report, cash_delta = _close_campaign(
            managed.campaign,
            float(series.closes[index]),
            int(series.minutes[index]),
            "end_of_backtest",
            execution,
            rules[symbol],
        )
        cash += cash_delta
        if report is not None:
            report["v5_tier"] = managed.profile.tier
            trades.append(report)

    result = summarize_grid_result(
        initial_equity,
        cash,
        trades,
        max_drawdown,
        max_drawdown_duration,
        sum(len(rows) for rows in candidates.values()),
        dict(rejected),
        hard_stopped,
        operational_signal,
        operational_portfolio,
    )
    result.update(
        {
            "strategy": GRID_V5_ENGINE_NAME,
            "strategy_version": "v5_independent_managed_campaigns",
            "by_v5_tier": _tier_metrics(trades),
            "operational_portfolio_config": asdict(operational_portfolio),
        }
    )
    return result
