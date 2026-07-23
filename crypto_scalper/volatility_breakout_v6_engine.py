from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Tuple

from .binance_client import SymbolRules
from .risk import BacktestExecutionConfig
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
    _mark_equity,
    _market_filter_reject_reason,
    minute_datetime,
    minute_token,
    summarize_result,
)


V6_ENGINE_NAME = "dual_thrust_volatility_breakout_v6_managed_lanes"


@dataclass(frozen=True)
class BreakoutV6ExecutionProfile:
    """Entry sizing and exits selected with data available at entry time."""

    lane: str
    signal: VolatilityBreakoutConfig
    portfolio: PortfolioSearchConfig
    exit_protection: ExitProtectionConfig

    def validate(self) -> None:
        if not self.lane:
            raise ValueError("v6 execution lane cannot be empty")
        self.exit_protection.validate()


@dataclass
class _ManagedPosition:
    protected: ProtectedPosition
    profile: BreakoutV6ExecutionProfile


ProfileSelector = Callable[
    [Candidate, int, float], Optional[BreakoutV6ExecutionProfile]
]
PrioritySelector = Callable[[Candidate], Tuple[Any, ...]]
ProfileGovernor = Callable[
    [BreakoutV6ExecutionProfile, float, int, float],
    BreakoutV6ExecutionProfile,
]


def _lane_metrics(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for trade in trades:
        lane = str(trade.get("v6_lane", "unknown"))
        row = output.setdefault(
            lane,
            {
                "trade_count": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            },
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
        profit = float(row["gross_profit"])
        row["win_rate"] = row.pop("wins") / count if count else 0.0
        row["profit_factor"] = (
            profit / loss if loss > 0.0 else (float("inf") if profit > 0.0 else 0.0)
        )
    return output


def simulate_v6_managed_portfolio(
    candidates: dict[int, list[Candidate]],
    symbols: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    operational_signal: VolatilityBreakoutConfig,
    operational_portfolio: PortfolioSearchConfig,
    operational_exit: ExitProtectionConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    profile_selector: ProfileSelector | None = None,
    priority_selector: PrioritySelector | None = None,
    profile_governor: ProfileGovernor | None = None,
) -> dict[str, Any]:
    """Simulate point-in-time lane-specific risk and exits.

    Portfolio limits, cooldown, hard drawdown stop and the default ranking stay
    global.  Only sizing and position-management parameters may vary by signal.
    With no selector this is path-equivalent to the frozen protected simulator.
    """

    operational_exit.validate()
    if start >= end:
        raise ValueError("start must be before end")

    default_profile = BreakoutV6ExecutionProfile(
        lane="default",
        signal=operational_signal,
        portfolio=operational_portfolio,
        exit_protection=operational_exit,
    )
    default_profile.validate()
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    cash = float(initial_equity)
    positions: dict[str, _ManagedPosition] = {}
    trades: list[dict[str, Any]] = []
    cooldown_until: dict[str, int] = {}
    daily_entries: dict[str, int] = defaultdict(int)
    rejected: dict[str, int] = defaultdict(int)
    peak_equity = cash
    max_drawdown = 0.0
    max_drawdown_duration = 0
    drawdown_start: int | None = None
    hard_stopped = False
    symbol_set = frozenset(symbols)

    def marked_equity(minute: int) -> float:
        return _mark_equity(
            cash,
            {
                symbol: managed.protected.position
                for symbol, managed in positions.items()
            },
            execution_data,
            minute,
            execution,
        )

    def tag_trade(
        trade: dict[str, Any], profile: BreakoutV6ExecutionProfile
    ) -> dict[str, Any]:
        trade["v6_lane"] = profile.lane
        return trade

    for minute in range(start_minute, end_minute):
        for symbol, managed in list(positions.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            trade, cash_delta = _process_protected_bar(
                managed.protected,
                minute,
                series,
                index,
                managed.profile.signal,
                managed.profile.exit_protection,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if trade is None:
                continue
            trades.append(tag_trade(trade, managed.profile))
            positions.pop(symbol, None)
            cooldown_until[symbol] = (
                minute + operational_portfolio.symbol_cooldown_minutes
            )

        equity_before_entry = marked_equity(minute)
        day_key = minute_datetime(minute).date().isoformat()
        rows = list(candidates.get(minute, ()))
        if priority_selector is None:
            rows.sort(
                key=lambda row: _candidate_sort_key(
                    row, operational_portfolio.ranking_mode
                )
            )
        else:
            rows.sort(key=priority_selector)
        if not hard_stopped:
            for candidate in rows:
                symbol = candidate.signal.symbol
                if symbol not in symbol_set or symbol not in execution_data:
                    rejected["outside_universe_or_missing_data"] += 1
                    continue
                market_reject = _market_filter_reject_reason(
                    candidate, operational_portfolio
                )
                if market_reject is not None:
                    rejected[market_reject] += 1
                    continue
                if symbol in positions:
                    rejected["symbol_already_open"] += 1
                    continue
                if cooldown_until.get(symbol, -1) > minute:
                    rejected["symbol_cooldown"] += 1
                    continue
                if len(positions) >= operational_portfolio.max_open_positions:
                    rejected["position_limit"] += 1
                    continue
                if daily_entries[day_key] >= operational_portfolio.max_daily_trades:
                    rejected["daily_trade_limit"] += 1
                    continue

                profile = (
                    profile_selector(candidate, minute, equity_before_entry)
                    if profile_selector is not None
                    else default_profile
                )
                if profile is None:
                    rejected["profile_rejected"] += 1
                    continue
                entry_drawdown = (
                    (peak_equity - equity_before_entry) / peak_equity
                    if peak_equity > 0.0
                    else 1.0
                )
                if profile_governor is not None:
                    profile = profile_governor(
                        profile, entry_drawdown, minute, equity_before_entry
                    )
                profile.validate()
                opened = _entry_position(
                    candidate,
                    execution_data[symbol],
                    rules[symbol],
                    profile.signal,
                    profile.portfolio,
                    execution,
                    (
                        equity_before_entry
                        if profile.portfolio.compound
                        else initial_equity
                    ),
                )
                if opened is None:
                    rejected["sizing_or_data"] += 1
                    continue
                position, entry_fee = opened
                protected = _protected_position(
                    position, profile.exit_protection, execution
                )
                managed = _ManagedPosition(protected=protected, profile=profile)
                positions[symbol] = managed
                cash -= entry_fee
                daily_entries[day_key] += 1

                index = execution_data[symbol].index_at(minute)
                if index is not None:
                    trade, cash_delta = _process_protected_bar(
                        protected,
                        minute,
                        execution_data[symbol],
                        index,
                        profile.signal,
                        profile.exit_protection,
                        execution,
                        rules[symbol],
                    )
                    cash += cash_delta
                    if trade is not None:
                        trades.append(tag_trade(trade, profile))
                        positions.pop(symbol, None)
                        cooldown_until[symbol] = (
                            minute
                            + operational_portfolio.symbol_cooldown_minutes
                        )
                if len(positions) >= operational_portfolio.max_open_positions:
                    break

        equity = marked_equity(minute)
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
    for symbol, managed in list(positions.items()):
        series = execution_data[symbol]
        index = series.last_index_at_or_before(force_minute)
        if index is None:
            continue
        trade, cash_delta = _close_protected_position(
            managed.protected,
            float(series.closes[index]),
            "end_of_backtest",
            int(series.minutes[index]),
            execution,
            rules[symbol],
        )
        cash += cash_delta
        trades.append(tag_trade(trade, managed.profile))
        positions.pop(symbol, None)

    result = summarize_result(
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
            "strategy": V6_ENGINE_NAME,
            "strategy_version": "v6_independent_managed_lanes",
            "operational_exit_protection_config": asdict(operational_exit),
            "by_v6_lane": _lane_metrics(trades),
            "partial_exit_count": sum(
                int(trade.get("partial_exit_count", 0)) for trade in trades
            ),
            "trades_with_partial_exit": sum(
                int(int(trade.get("partial_exit_count", 0)) > 0)
                for trade in trades
            ),
            "cash_reconciliation_error": (
                cash
                - initial_equity
                - sum(float(trade["net_pnl"]) for trade in trades)
            ),
        }
    )
    return result
