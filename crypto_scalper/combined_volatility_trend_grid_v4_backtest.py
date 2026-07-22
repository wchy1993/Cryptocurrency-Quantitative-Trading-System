from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any, Callable, Iterable

from .binance_client import SymbolRules
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    NOTIONAL_HEADROOM_SAFETY,
    CombinedPortfolioConfig,
    _summarize_combined_result,
    _tag_trade,
)
from .risk import BacktestExecutionConfig
from .trend_grid import TrendGridConfig, TrendGridSnapshot
from .trend_grid_optimize import (
    GridCampaign,
    GridCandidate,
    GridPortfolioConfig,
    _close_campaign,
    _create_campaign,
    _mark_equity as _mark_grid_equity,
    _process_campaign_bar,
)
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    ProtectedPosition,
    _close_protected_position,
    _process_protected_bar,
    _protected_position,
)
from .volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    _candidate_sort_key,
    _entry_position,
    _mark_equity as _mark_breakout_equity,
    _market_filter_reject_reason,
    minute_datetime,
    minute_token,
)


COMBINED_V4_STRATEGY_NAME = (
    "volatility_breakout_v4_regime_exit_plus_unchanged_dynamic_trend_grid_v2"
)


def _open_breakout_positions(
    positions: dict[str, ProtectedPosition],
) -> dict[str, Any]:
    return {symbol: protected.position for symbol, protected in positions.items()}


def _combined_equity_v4(
    cash: float,
    breakout_positions: dict[str, ProtectedPosition],
    grid_campaigns: dict[str, GridCampaign],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    minute: int,
    execution: BacktestExecutionConfig,
) -> float:
    return (
        cash
        + _mark_breakout_equity(
            0.0,
            _open_breakout_positions(breakout_positions),
            execution_data,
            minute,
            execution,
        )
        + _mark_grid_equity(
            0.0, grid_campaigns, execution_data, rules, minute, execution
        )
    )


def _committed_notional_v4(
    breakout_positions: dict[str, ProtectedPosition],
    grid_campaigns: dict[str, GridCampaign],
) -> float:
    breakout = sum(
        abs(protected.position.quantity * protected.position.entry_price)
        for protected in breakout_positions.values()
    )
    grid = sum(
        abs(level.quantity * level.raw_price)
        for campaign in grid_campaigns.values()
        for level in campaign.levels
    )
    return breakout + grid


def _available_notional_multiple_v4(
    equity: float,
    breakout_positions: dict[str, ProtectedPosition],
    grid_campaigns: dict[str, GridCampaign],
    config: CombinedPortfolioConfig,
) -> float:
    if equity <= 0.0:
        return 0.0
    headroom = equity * config.max_gross_notional_multiple - _committed_notional_v4(
        breakout_positions, grid_campaigns
    )
    return max(0.0, headroom / equity)


def simulate_combined_v4_portfolio(
    breakout_candidates: dict[int, list[Candidate]],
    grid_candidates: dict[int, list[GridCandidate]],
    grid_snapshots: dict[str, dict[int, TrendGridSnapshot]],
    symbols: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    breakout_signal_config: VolatilityBreakoutConfig,
    breakout_portfolio_config: PortfolioSearchConfig,
    breakout_exit_config: ExitProtectionConfig,
    grid_signal_config: TrendGridConfig,
    grid_portfolio_config: GridPortfolioConfig,
    combined_config: CombinedPortfolioConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    breakout_portfolio_selector: Callable[
        [Candidate, int, float], PortfolioSearchConfig
    ] | None = None,
) -> dict[str, Any]:
    """Shared max-position simulation with a v4 protected Breakout sleeve.

    Grid campaign creation, fills and exits are imported unchanged from the
    frozen Grid engine. The only new path is the independent Breakout overlay.
    """

    combined_config.validate()
    breakout_exit_config.validate()
    if start >= end:
        raise ValueError("start must be before end")

    start_minute = minute_token(start)
    end_minute = minute_token(end)
    symbol_set = frozenset(symbols)
    cash = float(initial_equity)
    breakout_positions: dict[str, ProtectedPosition] = {}
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

    def equity_at(minute: int) -> float:
        return _combined_equity_v4(
            cash,
            breakout_positions,
            grid_campaigns,
            execution_data,
            rules,
            minute,
            execution,
        )

    def observe_entry_exposure(entry_equity: float) -> None:
        nonlocal max_concurrent_positions, max_entry_committed_notional_multiple
        max_concurrent_positions = max(max_concurrent_positions, open_count())
        if entry_equity > 0.0:
            max_entry_committed_notional_multiple = max(
                max_entry_committed_notional_multiple,
                _committed_notional_v4(breakout_positions, grid_campaigns)
                / entry_equity,
            )

    for minute in range(start_minute, end_minute):
        # Preserve the live-like source rule: all old exits before any new entry.
        for symbol, protected in list(breakout_positions.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            trade, cash_delta = _process_protected_bar(
                protected,
                minute,
                series,
                index,
                breakout_signal_config,
                breakout_exit_config,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if trade is None:
                continue
            trades.append(_tag_trade(trade, BREAKOUT_KEY))
            breakout_positions.pop(symbol, None)
            breakout_cooldown_until[symbol] = (
                minute + breakout_portfolio_config.symbol_cooldown_minutes
            )

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
                        if (
                            breakout_daily_entries[day_key]
                            >= breakout_portfolio_config.max_daily_trades
                        ):
                            rejected[BREAKOUT_KEY]["daily_trade_limit"] += 1
                            continue
                        entry_equity = equity_at(minute)
                        available_multiple = _available_notional_multiple_v4(
                            entry_equity,
                            breakout_positions,
                            grid_campaigns,
                            combined_config,
                        )
                        if available_multiple <= 0.0:
                            rejected[BREAKOUT_KEY]["global_notional_limit"] += 1
                            continue
                        selected_breakout_portfolio = (
                            breakout_portfolio_selector(candidate, minute, entry_equity)
                            if breakout_portfolio_selector is not None
                            else breakout_portfolio_config
                        )
                        entry_portfolio = replace(
                            selected_breakout_portfolio,
                            max_notional_multiple=min(
                                selected_breakout_portfolio.max_notional_multiple,
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
                            entry_equity
                            if selected_breakout_portfolio.compound
                            else initial_equity,
                        )
                        if opened is None:
                            rejected[BREAKOUT_KEY]["sizing_or_data"] += 1
                            continue
                        position, entry_fee = opened
                        protected = _protected_position(
                            position, breakout_exit_config, execution
                        )
                        breakout_positions[symbol] = protected
                        cash -= entry_fee
                        breakout_daily_entries[day_key] += 1
                        observe_entry_exposure(entry_equity)

                        index = execution_data[symbol].index_at(minute)
                        if index is not None:
                            trade, cash_delta = _process_protected_bar(
                                protected,
                                minute,
                                execution_data[symbol],
                                index,
                                breakout_signal_config,
                                breakout_exit_config,
                                execution,
                                rules[symbol],
                            )
                            cash += cash_delta
                            if trade is not None:
                                trades.append(_tag_trade(trade, BREAKOUT_KEY))
                                breakout_positions.pop(symbol, None)
                                breakout_cooldown_until[symbol] = (
                                    minute
                                    + breakout_portfolio_config.symbol_cooldown_minutes
                                )
                        if (
                            len(breakout_positions)
                            >= breakout_portfolio_config.max_open_positions
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
                        if (
                            grid_daily_entries[day_key]
                            >= grid_portfolio_config.max_daily_campaigns
                        ):
                            rejected[GRID_KEY]["daily_campaign_limit"] += 1
                            continue
                        entry_equity = equity_at(minute)
                        available_multiple = _available_notional_multiple_v4(
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
                            entry_equity
                            if grid_portfolio_config.compound
                            else initial_equity,
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

        equity = equity_at(minute)
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
            max_drawdown_duration = max(
                max_drawdown_duration, minute - drawdown_start
            )
        else:
            drawdown_start = None
        max_drawdown = max(max_drawdown, drawdown)
        if not hard_stopped and (
            drawdown >= combined_config.hard_drawdown_stop_pct or equity <= 0.0
        ):
            hard_stopped = True
            hard_stopped_at = minute_datetime(minute).isoformat()

    force_minute = end_minute - 1
    for symbol, protected in list(breakout_positions.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        trade, cash_delta = _close_protected_position(
            protected,
            float(series.closes[index]),
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

    result = _summarize_combined_result(
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
    result.update(
        {
            "strategy": COMBINED_V4_STRATEGY_NAME,
            "breakout_exit_protection_config": asdict(breakout_exit_config),
            "breakout_partial_exit_count": sum(
                int(trade.get("partial_exit_count", 0))
                for trade in trades
                if trade["strategy"] == BREAKOUT_KEY
            ),
        }
    )
    return result
