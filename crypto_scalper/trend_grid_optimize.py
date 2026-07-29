from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .binance_client import SymbolRules
from .models import Candle, Direction
from .risk import (
    BacktestExecutionConfig,
    conservative_quantity,
    limit_fill,
    limit_order_filled,
    market_entry_fill,
    market_exit_fill,
)
from .trend_grid import (
    TREND_GRID_STRATEGY_NAME,
    TREND_GRID_VERSION,
    TrendGridConfig,
    TrendGridSignal,
    TrendGridSnapshot,
    build_trend_grid_timeline,
)
from .volatility_breakout_optimize import (
    UNIVERSE_30,
    UNIVERSE_50,
    CompactSeries,
    _load_runtime_inputs,
    _write_json,
    minute_datetime,
    minute_token,
    sha256_file,
    signal_candles_for_timeframe,
)


@dataclass(frozen=True)
class GridPortfolioConfig:
    risk_per_campaign_pct: float = 0.02
    max_campaign_risk_pct: float = 0.10
    max_open_campaigns: int = 1
    max_daily_campaigns: int = 8
    symbol_cooldown_minutes: int = 240
    max_notional_multiple: float = 9.0
    hard_drawdown_stop_pct: float = 0.60
    compound: bool = True
    long_risk_multiplier: float = 1.0
    short_risk_multiplier: float = 1.0


@dataclass(frozen=True)
class GridCandidate:
    signal: TrendGridSignal
    entry_minute: int


@dataclass
class GridLevel:
    index: int
    raw_price: float
    quantity: float
    cycles: int = 0


@dataclass
class GridLot:
    level_index: int
    quantity: float
    raw_entry_price: float
    entry_price: float
    entry_fee: float
    entry_slippage: float
    entry_minute: int
    target_price: float
    liquidity: str


@dataclass
class GridCampaign:
    candidate: GridCandidate
    start_minute: int
    anchor_price: float
    hard_stop: float
    risk_budget: float
    levels: list[GridLevel]
    lots: dict[int, GridLot] = field(default_factory=dict)
    regime_invalid_bars: int = 0
    entry_count: int = 0
    grid_take_profit_count: int = 0
    raw_gross_pnl: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    funding: float = 0.0
    net_pnl: float = 0.0
    max_open_lots: int = 0
    best_equity_pnl: float = 0.0
    worst_equity_pnl: float = 0.0


_FUNDING_INDEX: dict[int, tuple[list[datetime], list[float]]] = {}


def build_grid_research_timeline(
    universe: Iterable[str],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    config: TrendGridConfig,
    start: datetime,
    end: datetime,
) -> tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]]:
    candidates: dict[int, list[GridCandidate]] = defaultdict(list)
    snapshots: dict[str, dict[int, TrendGridSnapshot]] = {}
    for symbol in universe:
        source = signal_data.get(symbol)
        series = execution_data.get(symbol)
        if not source or series is None:
            continue
        candles = signal_candles_for_timeframe(source, config.timeframe_minutes)
        symbol_snapshots, signals = build_trend_grid_timeline(symbol, candles, config, start, end)
        snapshots[symbol] = {
            minute_token(item.available_time): item
            for item in symbol_snapshots
            if start <= item.available_time < end
        }
        for signal in signals:
            entry_minute = minute_token(signal.signal_available_time)
            if series.index_at(entry_minute) is None:
                continue
            candidates[entry_minute].append(GridCandidate(signal=signal, entry_minute=entry_minute))
    for rows in candidates.values():
        rows.sort(key=lambda item: (-item.signal.quality_score, item.signal.symbol))
    return dict(candidates), snapshots


def _funding_for_lot(
    execution: BacktestExecutionConfig,
    symbol: str,
    direction: Direction,
    lot: GridLot,
    exit_minute: int,
) -> float:
    if not execution.funding_enabled:
        return 0.0
    rates = (execution.funding_rates_by_symbol or {}).get(symbol, ())
    cache_key = id(rates)
    indexed = _FUNDING_INDEX.get(cache_key)
    if indexed is None:
        indexed = ([item.timestamp for item in rates], [float(item.rate) for item in rates])
        _FUNDING_INDEX[cache_key] = indexed
    timestamps, values = indexed
    entry_time = minute_datetime(lot.entry_minute)
    exit_time = minute_datetime(exit_minute)
    start = bisect.bisect_right(timestamps, entry_time)
    end = bisect.bisect_right(timestamps, exit_time)
    return -direction.value * abs(lot.quantity * lot.entry_price) * sum(values[start:end])


def _lot_worst_loss_per_unit(
    direction: Direction,
    entry_price: float,
    entry_fee: float,
    hard_stop: float,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    stop_fill = market_exit_fill(execution, rules, direction, 1.0, hard_stop, "stop_market")
    return -(
        direction.value * (stop_fill.price - entry_price)
        - entry_fee
        - stop_fill.fee
    )


def _create_campaign(
    candidate: GridCandidate,
    series: CompactSeries,
    rules: SymbolRules,
    signal_config: TrendGridConfig,
    portfolio_config: GridPortfolioConfig,
    execution: BacktestExecutionConfig,
    sizing_equity: float,
) -> tuple[GridCampaign, float] | None:
    index = series.index_at(candidate.entry_minute)
    if index is None or sizing_equity <= 0.0:
        return None
    raw_entry = float(series.opens[index])
    direction = candidate.signal.direction
    atr_value = candidate.signal.atr_value
    spacing = atr_value * signal_config.grid_spacing_atr
    if raw_entry <= 0.0 or spacing <= 0.0:
        return None

    distance_stop = raw_entry - direction.value * atr_value * signal_config.hard_stop_atr_multiple
    structural_stop = (
        candidate.signal.slow_ema
        - direction.value * atr_value * signal_config.hard_stop_slow_ema_buffer_atr
    )
    hard_stop = min(distance_stop, structural_stop) if direction == Direction.LONG else max(distance_stop, structural_stop)
    deepest = raw_entry - direction.value * spacing * signal_config.grid_levels
    tick = float(rules.price_tick)
    if direction == Direction.LONG:
        hard_stop = min(hard_stop, deepest - tick)
    else:
        hard_stop = max(hard_stop, deepest + tick)
    if hard_stop <= 0.0:
        return None

    level_prices = [raw_entry - direction.value * spacing * level for level in range(signal_config.grid_levels + 1)]
    if any(price <= 0.0 for price in level_prices):
        return None
    weights = [signal_config.deeper_level_size_multiplier**level for level in range(len(level_prices))]
    unit_risks: list[float] = []
    unit_entries: list[float] = []
    for level, raw_price in enumerate(level_prices):
        if level == 0 and signal_config.initial_entry_enabled:
            fill = market_entry_fill(execution, rules, direction, 1.0, raw_price)
        else:
            side = "buy" if direction == Direction.LONG else "sell"
            fill = limit_fill(execution, rules, direction, side, 1.0, raw_price)
        unit_entries.append(fill.price)
        unit_risks.append(
            _lot_worst_loss_per_unit(direction, fill.price, fill.fee, hard_stop, execution, rules)
        )
    weighted_risk = sum(weight * risk for weight, risk in zip(weights, unit_risks))
    if weighted_risk <= 0.0:
        return None

    side_multiplier = (
        portfolio_config.long_risk_multiplier
        if direction == Direction.LONG
        else portfolio_config.short_risk_multiplier
    )
    risk_pct = min(
        max(0.0, portfolio_config.max_campaign_risk_pct),
        max(0.0, portfolio_config.risk_per_campaign_pct) * max(0.0, side_multiplier),
    )
    risk_budget = sizing_equity * risk_pct
    base_quantity = risk_budget / weighted_risk
    weighted_notional = sum(weight * price for weight, price in zip(weights, unit_entries))
    max_notional = sizing_equity * max(0.0, portfolio_config.max_notional_multiple)
    base_quantity = min(base_quantity, max_notional / max(weighted_notional, 1e-12))
    levels: list[GridLevel] = []
    for level, (raw_price, weight) in enumerate(zip(level_prices, weights)):
        quantity = conservative_quantity(rules, base_quantity * weight)
        if quantity <= 0.0 or quantity * raw_price < max(5.0, float(rules.min_notional)):
            return None
        levels.append(GridLevel(index=level, raw_price=raw_price, quantity=quantity))

    campaign = GridCampaign(
        candidate=candidate,
        start_minute=candidate.entry_minute,
        anchor_price=raw_entry,
        hard_stop=hard_stop,
        risk_budget=sum(level.quantity * unit_risks[level.index] for level in levels),
        levels=levels,
    )
    initial_fee = 0.0
    if signal_config.initial_entry_enabled:
        fill = market_entry_fill(
            execution,
            rules,
            direction,
            levels[0].quantity,
            raw_entry,
            float(series.volumes[index]) * raw_entry,
        )
        campaign.lots[0] = GridLot(
            level_index=0,
            quantity=levels[0].quantity,
            raw_entry_price=raw_entry,
            entry_price=fill.price,
            entry_fee=fill.fee,
            entry_slippage=fill.slippage_cost,
            entry_minute=candidate.entry_minute,
            target_price=fill.price + direction.value * spacing * signal_config.grid_target_spacing,
            liquidity=fill.liquidity,
        )
        campaign.entry_count = 1
        campaign.max_open_lots = 1
        campaign.fee += fill.fee
        campaign.slippage += fill.slippage_cost
        initial_fee = fill.fee
    return campaign, initial_fee


def _exit_lot(
    campaign: GridCampaign,
    lot: GridLot,
    raw_exit: float,
    exit_minute: int,
    order_type: str,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    direction = campaign.candidate.signal.direction
    if order_type == "limit":
        side = "sell" if direction == Direction.LONG else "buy"
        fill = limit_fill(execution, rules, direction, side, lot.quantity, raw_exit)
    else:
        fill = market_exit_fill(execution, rules, direction, lot.quantity, raw_exit, order_type)
    funding = _funding_for_lot(
        execution,
        campaign.candidate.signal.symbol,
        direction,
        lot,
        exit_minute,
    )
    raw_gross = direction.value * lot.quantity * (raw_exit - lot.raw_entry_price)
    execution_gross = direction.value * lot.quantity * (fill.price - lot.entry_price)
    net = execution_gross - lot.entry_fee - fill.fee + funding
    campaign.raw_gross_pnl += raw_gross
    campaign.fee += fill.fee
    campaign.slippage += fill.slippage_cost
    campaign.funding += funding
    campaign.net_pnl += net
    return execution_gross - fill.fee + funding


def _close_campaign(
    campaign: GridCampaign,
    raw_exit: float,
    exit_minute: int,
    reason: str,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[dict[str, Any] | None, float]:
    cash_delta = 0.0
    order_type = "stop_market" if "stop" in reason else "market"
    for lot in list(campaign.lots.values()):
        cash_delta += _exit_lot(campaign, lot, raw_exit, exit_minute, order_type, execution, rules)
    campaign.lots.clear()
    if campaign.entry_count <= 0:
        return None, cash_delta
    signal = campaign.candidate.signal
    report = {
        "event_id": signal.event_id,
        "symbol": signal.symbol,
        "side": signal.direction.name,
        "entry_time": minute_datetime(campaign.start_minute).isoformat(),
        "exit_time": minute_datetime(exit_minute).isoformat(),
        "anchor_price": campaign.anchor_price,
        "hard_stop": campaign.hard_stop,
        "risk_usdt": campaign.risk_budget,
        "entry_count": campaign.entry_count,
        "grid_take_profit_count": campaign.grid_take_profit_count,
        "max_open_lots": campaign.max_open_lots,
        "gross_pnl": campaign.raw_gross_pnl,
        "fee": campaign.fee,
        "slippage": campaign.slippage,
        "funding": campaign.funding,
        "net_pnl": campaign.net_pnl,
        "pnl_r": campaign.net_pnl / max(campaign.risk_budget, 1e-12),
        "mfe_usdt": campaign.best_equity_pnl,
        "mae_usdt": campaign.worst_equity_pnl,
        "holding_minutes": exit_minute - campaign.start_minute,
        "exit_reason": reason,
        "quality_score": signal.quality_score,
        "alignment_atr": signal.alignment_atr,
        "extension_atr": signal.extension_atr,
        "volume_ratio": signal.volume_ratio,
    }
    return report, cash_delta


def _campaign_unrealized(
    campaign: GridCampaign,
    raw_price: float,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> float:
    result = campaign.net_pnl
    for lot in campaign.lots.values():
        fill = market_exit_fill(
            execution,
            rules,
            campaign.candidate.signal.direction,
            lot.quantity,
            raw_price,
            "market",
        )
        result += (
            campaign.candidate.signal.direction.value * lot.quantity * (fill.price - lot.entry_price)
            - lot.entry_fee
            - fill.fee
        )
    return result


def _process_campaign_bar(
    campaign: GridCampaign,
    minute: int,
    series: CompactSeries,
    index: int,
    snapshot: TrendGridSnapshot | None,
    signal_config: TrendGridConfig,
    execution: BacktestExecutionConfig,
    rules: SymbolRules,
) -> tuple[float, str | None]:
    direction = campaign.candidate.signal.direction
    open_price = float(series.opens[index])
    high = float(series.highs[index])
    low = float(series.lows[index])
    gap_stop = open_price <= campaign.hard_stop if direction == Direction.LONG else open_price >= campaign.hard_stop
    if gap_stop:
        report, cash_delta = _close_campaign(
            campaign, open_price, minute, "hard_stop_gap", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "hard_stop_gap"

    current_open_pnl = _campaign_unrealized(campaign, open_price, execution, rules)
    campaign.best_equity_pnl = max(campaign.best_equity_pnl, current_open_pnl)
    campaign.worst_equity_pnl = min(campaign.worst_equity_pnl, current_open_pnl)
    if (
        signal_config.campaign_loss_limit_r > 0.0
        and current_open_pnl <= -campaign.risk_budget * signal_config.campaign_loss_limit_r
    ):
        report, cash_delta = _close_campaign(
            campaign, open_price, minute, "campaign_loss_limit", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "campaign_loss_limit"
    if (
        signal_config.campaign_take_profit_r > 0.0
        and current_open_pnl >= campaign.risk_budget * signal_config.campaign_take_profit_r
    ):
        report, cash_delta = _close_campaign(
            campaign, open_price, minute, "campaign_take_profit", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "campaign_take_profit"
    if (
        signal_config.profit_lock_activation_r > 0.0
        and signal_config.profit_giveback_r > 0.0
        and campaign.best_equity_pnl >= campaign.risk_budget * signal_config.profit_lock_activation_r
        and current_open_pnl
        <= campaign.best_equity_pnl - campaign.risk_budget * signal_config.profit_giveback_r
    ):
        report, cash_delta = _close_campaign(
            campaign, open_price, minute, "profit_giveback", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "profit_giveback"
    if (
        signal_config.cycle_profit_floor_min_take_profits > 0
        and campaign.grid_take_profit_count
        >= signal_config.cycle_profit_floor_min_take_profits
        and campaign.best_equity_pnl
        >= (
            campaign.risk_budget
            * signal_config.cycle_profit_floor_activation_r
        )
        and current_open_pnl
        <= campaign.risk_budget * signal_config.cycle_profit_floor_r
    ):
        report, cash_delta = _close_campaign(
            campaign,
            open_price,
            minute,
            "cycle_profit_floor",
            execution,
            rules,
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "cycle_profit_floor"

    if snapshot is not None:
        if snapshot.exit_invalid(direction, signal_config.regime_exit_mode):
            campaign.regime_invalid_bars += 1
        else:
            campaign.regime_invalid_bars = 0
        if campaign.regime_invalid_bars >= signal_config.regime_exit_confirm_bars:
            report, cash_delta = _close_campaign(
                campaign, open_price, minute, "ema_regime_exit", execution, rules
            )
            campaign._pending_report = report  # type: ignore[attr-defined]
            return cash_delta, "ema_regime_exit"

    if minute - campaign.start_minute >= signal_config.max_campaign_minutes:
        report, cash_delta = _close_campaign(
            campaign, open_price, minute, "campaign_time_stop", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "campaign_time_stop"

    hard_stop_hit = low <= campaign.hard_stop if direction == Direction.LONG else high >= campaign.hard_stop
    if hard_stop_hit:
        report, cash_delta = _close_campaign(
            campaign, campaign.hard_stop, minute, "hard_stop", execution, rules
        )
        campaign._pending_report = report  # type: ignore[attr-defined]
        return cash_delta, "hard_stop"

    cash_delta = 0.0
    for level_index, lot in list(campaign.lots.items()):
        side = "sell" if direction == Direction.LONG else "buy"
        _, filled = limit_order_filled(
            rules,
            side,
            Candle(minute_datetime(minute), open_price, high, low, float(series.closes[index]), float(series.volumes[index])),
            lot.target_price,
        )
        if not filled:
            continue
        cash_delta += _exit_lot(campaign, lot, lot.target_price, minute, "limit", execution, rules)
        campaign.lots.pop(level_index, None)
        campaign.levels[level_index].cycles += 1
        campaign.grid_take_profit_count += 1

    entry_side = "buy" if direction == Direction.LONG else "sell"
    candle = Candle(
        minute_datetime(minute),
        open_price,
        high,
        low,
        float(series.closes[index]),
        float(series.volumes[index]),
    )
    spacing = campaign.candidate.signal.atr_value * signal_config.grid_spacing_atr
    pause_new_fills = False
    if snapshot is not None and signal_config.pause_new_fills_on_fast_breach:
        pause_new_fills = (
            snapshot.close < snapshot.fast_ema
            if direction == Direction.LONG
            else snapshot.close > snapshot.fast_ema
        )
    for level in campaign.levels:
        if pause_new_fills:
            break
        if signal_config.max_total_entries > 0 and campaign.entry_count >= signal_config.max_total_entries:
            break
        if level.index in campaign.lots or level.cycles >= signal_config.max_cycles_per_level:
            continue
        if level.index == 0 and signal_config.initial_entry_enabled and level.cycles == 0:
            continue
        _, filled = limit_order_filled(rules, entry_side, candle, level.raw_price)
        if not filled:
            continue
        fill = limit_fill(execution, rules, direction, entry_side, level.quantity, level.raw_price)
        campaign.lots[level.index] = GridLot(
            level_index=level.index,
            quantity=level.quantity,
            raw_entry_price=level.raw_price,
            entry_price=fill.price,
            entry_fee=fill.fee,
            entry_slippage=fill.slippage_cost,
            entry_minute=minute,
            target_price=fill.price + direction.value * spacing * signal_config.grid_target_spacing,
            liquidity=fill.liquidity,
        )
        campaign.entry_count += 1
        campaign.fee += fill.fee
        campaign.slippage += fill.slippage_cost
        campaign.max_open_lots = max(campaign.max_open_lots, len(campaign.lots))
        cash_delta -= fill.fee

    mark = _campaign_unrealized(campaign, float(series.closes[index]), execution, rules)
    campaign.best_equity_pnl = max(campaign.best_equity_pnl, mark)
    campaign.worst_equity_pnl = min(campaign.worst_equity_pnl, mark)
    return cash_delta, None


def _mark_equity(
    cash: float,
    campaigns: dict[str, GridCampaign],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    minute: int,
    execution: BacktestExecutionConfig,
) -> float:
    equity = cash
    for symbol, campaign in campaigns.items():
        index = execution_data[symbol].last_index_at_or_before(minute)
        if index is None:
            continue
        raw_price = float(execution_data[symbol].closes[index])
        for lot in campaign.lots.values():
            fill = market_exit_fill(
                execution,
                rules[symbol],
                campaign.candidate.signal.direction,
                lot.quantity,
                raw_price,
                "market",
            )
            equity += (
                campaign.candidate.signal.direction.value * lot.quantity * (fill.price - lot.entry_price)
                - fill.fee
            )
    return equity


def simulate_grid_portfolio(
    candidates: dict[int, list[GridCandidate]],
    snapshots: dict[str, dict[int, TrendGridSnapshot]],
    universe: Iterable[str],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    signal_config: TrendGridConfig,
    portfolio_config: GridPortfolioConfig,
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
    skip_symbols: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    start_minute = minute_token(start)
    end_minute = minute_token(end)
    cash = float(initial_equity)
    campaigns: dict[str, GridCampaign] = {}
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

    for minute in range(start_minute, end_minute):
        for symbol, campaign in list(campaigns.items()):
            series = execution_data[symbol]
            index = series.index_at(minute)
            if index is None:
                continue
            cash_delta, close_reason = _process_campaign_bar(
                campaign,
                minute,
                series,
                index,
                snapshots.get(symbol, {}).get(minute),
                signal_config,
                execution,
                rules[symbol],
            )
            cash += cash_delta
            if close_reason is None:
                continue
            report = getattr(campaign, "_pending_report", None)
            if report is not None:
                trades.append(report)
            else:
                rejected["campaign_without_fill"] += 1
            campaigns.pop(symbol, None)
            cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes

        equity_before_entry = _mark_equity(cash, campaigns, execution_data, rules, minute, execution)
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
                if len(campaigns) >= portfolio_config.max_open_campaigns:
                    rejected["campaign_limit"] += 1
                    continue
                if daily_entries[day_key] >= portfolio_config.max_daily_campaigns:
                    rejected["daily_campaign_limit"] += 1
                    continue
                opened = _create_campaign(
                    candidate,
                    execution_data[symbol],
                    rules[symbol],
                    signal_config,
                    portfolio_config,
                    execution,
                    equity_before_entry if portfolio_config.compound else initial_equity,
                )
                if opened is None:
                    rejected["sizing_or_structure"] += 1
                    continue
                campaign, initial_fee = opened
                campaigns[symbol] = campaign
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
                        signal_config,
                        execution,
                        rules[symbol],
                    )
                    cash += cash_delta
                    if close_reason is not None:
                        report = getattr(campaign, "_pending_report", None)
                        if report is not None:
                            trades.append(report)
                        campaigns.pop(symbol, None)
                        cooldown_until[symbol] = minute + portfolio_config.symbol_cooldown_minutes
                if len(campaigns) >= portfolio_config.max_open_campaigns:
                    break

        equity = _mark_equity(cash, campaigns, execution_data, rules, minute, execution)
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0.0 else 1.0
        if drawdown > 0.0:
            drawdown_start = minute if drawdown_start is None else drawdown_start
            max_drawdown_duration = max(max_drawdown_duration, minute - drawdown_start)
        else:
            drawdown_start = None
        max_drawdown = max(max_drawdown, drawdown)
        if drawdown >= portfolio_config.hard_drawdown_stop_pct or equity <= 0.0:
            hard_stopped = True

    force_minute = end_minute - 1
    for symbol, campaign in list(campaigns.items()):
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
            trades.append(report)

    return summarize_grid_result(
        initial_equity,
        cash,
        trades,
        max_drawdown,
        max_drawdown_duration,
        sum(len(rows) for rows in candidates.values()),
        dict(rejected),
        hard_stopped,
        signal_config,
        portfolio_config,
    )


def summarize_grid_result(
    initial_equity: float,
    final_equity: float,
    trades: list[dict[str, Any]],
    max_drawdown: float,
    max_drawdown_duration: int,
    candidate_count: int,
    rejected: dict[str, int],
    hard_stopped: bool,
    signal_config: TrendGridConfig,
    portfolio_config: GridPortfolioConfig,
) -> dict[str, Any]:
    wins = [trade for trade in trades if trade["net_pnl"] > 0.0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0.0]
    net_profit = final_equity - initial_equity
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    raw_gross_profit = sum(max(0.0, trade["gross_pnl"]) for trade in trades)
    fees = sum(trade["fee"] for trade in trades)
    slippage = sum(trade["slippage"] for trade in trades)
    funding = sum(trade["funding"] for trade in trades)
    by_month: dict[str, dict[str, Any]] = {}
    by_symbol: dict[str, dict[str, Any]] = {}
    by_side: dict[str, dict[str, Any]] = {}
    by_exit_reason: dict[str, dict[str, Any]] = {}
    for trade in trades:
        for output, key in (
            (by_month, trade["exit_time"][:7]),
            (by_symbol, trade["symbol"]),
            (by_side, trade["side"]),
            (by_exit_reason, trade["exit_reason"]),
        ):
            row = output.setdefault(
                key,
                {"trade_count": 0, "wins": 0, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0},
            )
            row["trade_count"] += 1
            row["wins"] += int(trade["net_pnl"] > 0.0)
            row["net_pnl"] += trade["net_pnl"]
            if trade["net_pnl"] > 0.0:
                row["gross_profit"] += trade["net_pnl"]
            else:
                row["gross_loss"] += abs(trade["net_pnl"])
    for output in (by_month, by_symbol, by_side, by_exit_reason):
        for row in output.values():
            row["win_rate"] = row.pop("wins") / row["trade_count"]
            row["profit_factor"] = (
                row["gross_profit"] / row["gross_loss"]
                if row["gross_loss"] > 0.0
                else (math.inf if row["gross_profit"] > 0.0 else 0.0)
            )
    average_win = statistics.mean(trade["net_pnl"] for trade in wins) if wins else 0.0
    average_loss = statistics.mean(trade["net_pnl"] for trade in losses) if losses else 0.0
    sorted_wins = sorted((trade["net_pnl"] for trade in wins), reverse=True)
    return {
        "strategy": TREND_GRID_STRATEGY_NAME,
        "strategy_version": TREND_GRID_VERSION,
        "signal_config": signal_config.as_dict(),
        "portfolio_config": asdict(portfolio_config),
        "candidate_count": candidate_count,
        "trade_count": len(trades),
        "grid_entry_count": sum(trade["entry_count"] for trade in trades),
        "grid_take_profit_count": sum(trade["grid_take_profit_count"] for trade in trades),
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "net_profit": net_profit,
        "return_pct": net_profit / initial_equity if initial_equity else 0.0,
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (math.inf if gross_profit > 0.0 else 0.0),
        "expectancy_usdt": net_profit / len(trades) if trades else 0.0,
        "expectancy_r": statistics.mean(trade["pnl_r"] for trade in trades) if trades else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "average_win_loss_ratio": average_win / abs(average_loss) if average_loss < 0.0 else 0.0,
        "max_drawdown_pct": max_drawdown,
        "max_drawdown_duration_minutes": max_drawdown_duration,
        "fee": fees,
        "slippage": slippage,
        "funding": funding,
        "full_cost": fees + slippage - funding,
        "cost_to_raw_gross_profit_ratio": (fees + slippage - funding) / raw_gross_profit if raw_gross_profit > 0.0 else math.inf,
        "top5_profit_contribution": sum(sorted_wins[:5]) / net_profit if net_profit > 0.0 else math.inf,
        "positive_months": sum(1 for row in by_month.values() if row["net_pnl"] > 0.0),
        "negative_months": sum(1 for row in by_month.values() if row["net_pnl"] < 0.0),
        "hard_drawdown_stopped": hard_stopped,
        "rejected": rejected,
        "by_month": by_month,
        "by_symbol": by_symbol,
        "by_side": by_side,
        "by_exit_reason": by_exit_reason,
        "trades": trades,
    }


def compact_grid_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_count", "trade_count", "grid_entry_count", "grid_take_profit_count",
        "initial_equity", "final_equity", "net_profit", "return_pct", "win_rate",
        "profit_factor", "expectancy_usdt", "expectancy_r", "average_win", "average_loss",
        "average_win_loss_ratio", "max_drawdown_pct", "max_drawdown_duration_minutes",
        "fee", "slippage", "funding", "full_cost", "cost_to_raw_gross_profit_ratio",
        "top5_profit_contribution", "positive_months", "negative_months",
        "hard_drawdown_stopped", "by_month", "by_side", "by_exit_reason",
    )
    return {key: result[key] for key in keys}


def _signal_cache_key(config: TrendGridConfig) -> str:
    fields = config.as_dict()
    for key in (
        "grid_spacing_atr", "grid_levels", "grid_target_spacing",
        "deeper_level_size_multiplier", "initial_entry_enabled",
        "hard_stop_atr_multiple", "hard_stop_slow_ema_buffer_atr",
        "regime_exit_mode", "regime_exit_confirm_bars", "max_campaign_minutes",
        "max_cycles_per_level", "max_total_entries", "pause_new_fills_on_fast_breach",
        "campaign_loss_limit_r", "campaign_take_profit_r", "profit_lock_activation_r",
        "profit_giveback_r", "cycle_profit_floor_min_take_profits",
        "cycle_profit_floor_activation_r", "cycle_profit_floor_r",
    ):
        fields.pop(key, None)
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _shift_candidates(
    candidates: dict[int, list[GridCandidate]],
    delay_minutes: int,
    execution_data: dict[str, CompactSeries],
) -> dict[int, list[GridCandidate]]:
    if delay_minutes <= 0:
        return candidates
    output: dict[int, list[GridCandidate]] = defaultdict(list)
    for rows in candidates.values():
        for candidate in rows:
            minute = candidate.entry_minute + delay_minutes
            series = execution_data.get(candidate.signal.symbol)
            if series is None or series.index_at(minute) is None:
                continue
            output[minute].append(replace(candidate, entry_minute=minute))
    for rows in output.values():
        rows.sort(key=lambda item: (-item.signal.quality_score, item.signal.symbol))
    return dict(output)


def _scaled_execution(execution: BacktestExecutionConfig, multiplier: float) -> BacktestExecutionConfig:
    return replace(
        execution,
        maker_fee_rate=execution.maker_fee_rate * multiplier,
        taker_fee_rate=execution.taker_fee_rate * multiplier,
        market_slippage_bps=execution.market_slippage_bps * multiplier,
        stop_slippage_bps=execution.stop_slippage_bps * multiplier,
        take_profit_slippage_bps=execution.take_profit_slippage_bps * multiplier,
    )


def _evaluate_variant(
    name: str,
    signal_config: TrendGridConfig,
    portfolio_config: GridPortfolioConfig,
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    periods: dict[str, tuple[datetime, datetime]],
    initial_equity: float,
    timeline_cache: dict[str, tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]]],
) -> dict[str, Any]:
    cache_key = _signal_cache_key(signal_config)
    timeline = timeline_cache.get(cache_key)
    if timeline is None:
        full_start, full_end = periods["full"]
        timeline = build_grid_research_timeline(
            universe, signal_data, execution_data, signal_config, full_start, full_end
        )
        timeline_cache[cache_key] = timeline
    candidates, snapshots = timeline
    period_start, period_end = periods["full"]
    result = simulate_grid_portfolio(
        candidates,
        snapshots,
        universe,
        execution_data,
        rules,
        signal_config,
        portfolio_config,
        execution,
        period_start,
        period_end,
        initial_equity,
    )
    return {
        "name": name,
        "signal_config": signal_config.as_dict(),
        "portfolio_config": asdict(portfolio_config),
        "full": compact_grid_summary(result),
        "_full_result": result,
    }


def _selection_score(row: dict[str, Any], historical_max: bool = False) -> tuple[Any, ...]:
    full = row["full"]
    valid = (
        full["trade_count"] >= 12
        and full["profit_factor"] > 1.0
        and not full["hard_drawdown_stopped"]
    )
    if historical_max or "validation" not in row:
        return (
            valid,
            full["net_profit"],
            full["profit_factor"],
            -full["max_drawdown_pct"],
        )
    validation = row["validation"]
    train = row["train"]
    return (
        valid,
        validation["trade_count"] >= 3,
        validation["net_profit"] > 0.0,
        train["net_profit"] > 0.0,
        validation["net_profit"],
        full["net_profit"],
        validation["profit_factor"],
        -full["max_drawdown_pct"],
    )


def _select(rows: list[dict[str, Any]], historical_max: bool = False) -> dict[str, Any]:
    return max(rows, key=lambda row: _selection_score(row, historical_max))


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _attach_split_results(
    rows: list[dict[str, Any]],
    universe: tuple[str, ...],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    periods: dict[str, tuple[datetime, datetime]],
    initial_equity: float,
    timeline_cache: dict[str, tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]]],
) -> None:
    for row in rows:
        signal = TrendGridConfig(**row["signal_config"])
        portfolio = GridPortfolioConfig(**row["portfolio_config"])
        candidates, snapshots = timeline_cache[_signal_cache_key(signal)]
        for period_name in ("train", "validation", "test"):
            period_start, period_end = periods[period_name]
            result = simulate_grid_portfolio(
                candidates,
                snapshots,
                universe,
                execution_data,
                rules,
                signal,
                portfolio,
                execution,
                period_start,
                period_end,
                initial_equity,
            )
            row[period_name] = compact_grid_summary(result)


def _run_stage(
    stage: str,
    variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]],
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    periods: dict[str, tuple[datetime, datetime]],
    initial_equity: float,
    timeline_cache: dict[str, tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, (name, signal, portfolio) in enumerate(variants, start=1):
        row = _evaluate_variant(
            name,
            signal,
            portfolio,
            universe,
            signal_data,
            execution_data,
            rules,
            execution,
            periods,
            initial_equity,
            timeline_cache,
        )
        rows.append(row)
        if number == 1 or number % 10 == 0:
            full = row["full"]
            print(
                f"{len(universe)} {stage} {number}/{len(variants)} {name}: "
                f"trades={full['trade_count']} net={full['net_profit']:.2f} "
                f"pf={full['profit_factor']:.3f} dd={full['max_drawdown_pct']:.2%}",
                flush=True,
            )
    return rows


def optimize_grid_universe(
    universe: tuple[str, ...],
    signal_data: dict[str, list[Candle]],
    execution_data: dict[str, CompactSeries],
    rules: dict[str, SymbolRules],
    execution: BacktestExecutionConfig,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> dict[str, Any]:
    split1 = start + timedelta(days=31)
    split2 = start + timedelta(days=61)
    if split2 >= end:
        raise ValueError("three-month optimization requires at least 62 days")
    periods = {
        "train": (start, split1),
        "validation": (split1, split2),
        "test": (split2, end),
        "full": (start, end),
    }
    timeline_cache: dict[str, tuple[dict[int, list[GridCandidate]], dict[str, dict[int, TrendGridSnapshot]]]] = {}
    stages: dict[str, list[dict[str, Any]]] = {}
    base_signal = TrendGridConfig()
    base_portfolio = GridPortfolioConfig()

    stage0 = _run_stage(
        "stage0_baseline",
        [("baseline", base_signal, base_portfolio)],
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage0_baseline"] = [_public_row(row) for row in stage0]

    regime_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for timeframe in (15, 30, 60):
        for fast, slow in ((9, 36), (13, 55), (21, 55), (21, 99)):
            for mode in ("continuous", "fast_reclaim", "pullback_touch", "hybrid"):
                signal = replace(
                    base_signal,
                    timeframe_minutes=timeframe,
                    fast_ema_period=fast,
                    slow_ema_period=slow,
                    entry_mode=mode,
                )
                regime_variants.append((f"tf{timeframe}_ema{fast}_{slow}_{mode}", signal, base_portfolio))
    stage1 = _run_stage(
        "stage1_regime_entry", regime_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage1_regime_entry"] = [_public_row(row) for row in stage1]
    selected = _select(stage1)
    selected_signal = TrendGridConfig(**selected["signal_config"])

    geometry_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for spacing in (0.25, 0.40, 0.55, 0.70):
        for levels in (2, 3, 4):
            for target in (0.75, 1.0, 1.25):
                hard_stop = max(2.0, spacing * levels + 0.75)
                signal = replace(
                    selected_signal,
                    grid_spacing_atr=spacing,
                    grid_levels=levels,
                    grid_target_spacing=target,
                    hard_stop_atr_multiple=hard_stop,
                )
                geometry_variants.append((f"space{spacing:.2f}_levels{levels}_tp{target:.2f}", signal, base_portfolio))
    for multiplier in (0.75, 0.85):
        geometry_variants.append(
            (
                f"deeper_size_{multiplier:.2f}",
                replace(selected_signal, deeper_level_size_multiplier=multiplier),
                base_portfolio,
            )
        )
    geometry_variants.append(("limit_grid_without_initial", replace(selected_signal, initial_entry_enabled=False), base_portfolio))
    stage2 = _run_stage(
        "stage2_grid_geometry", geometry_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage2_grid_geometry"] = [_public_row(row) for row in stage2]
    selected = _select(stage2)
    selected_signal = TrendGridConfig(**selected["signal_config"])

    stop_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    minimum_stop = selected_signal.grid_spacing_atr * selected_signal.grid_levels + 0.25
    for stop in sorted({max(minimum_stop, value) for value in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)}):
        stop_variants.append((f"hard_stop_{stop:.2f}", replace(selected_signal, hard_stop_atr_multiple=stop), base_portfolio))
    stage3a = _run_stage(
        "stage3a_hard_stop", stop_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage3a_hard_stop"] = [_public_row(row) for row in stage3a]
    selected = _select(stage3a)
    selected_signal = TrendGridConfig(**selected["signal_config"])

    exit_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for mode in ("fast_ema", "slow_ema", "ema_cross", "fast_or_cross"):
        for confirm in (1, 2):
            for holding in (720, 1_440, 2_880):
                signal = replace(
                    selected_signal,
                    regime_exit_mode=mode,
                    regime_exit_confirm_bars=confirm,
                    max_campaign_minutes=holding,
                )
                exit_variants.append((f"{mode}_confirm{confirm}_hold{holding}", signal, base_portfolio))
    stage3b = _run_stage(
        "stage3b_exit", exit_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage3b_exit"] = [_public_row(row) for row in stage3b]
    selected = _select(stage3b)
    selected_signal = TrendGridConfig(**selected["signal_config"])

    quality_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for side in ("both", "long", "short"):
        for extension in (0.50, 0.80, 1.20, 1.60):
            signal = replace(
                selected_signal,
                allow_long=side != "short",
                allow_short=side != "long",
                max_entry_extension_atr=extension,
            )
            quality_variants.append((f"{side}_extension{extension:.2f}", signal, base_portfolio))
    stage4 = _run_stage(
        "stage4_direction_quality", quality_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    stages["stage4_direction_quality"] = [_public_row(row) for row in stage4]
    selected = _select(stage4)
    selected_signal = TrendGridConfig(**selected["signal_config"])

    risk_variants: list[tuple[str, TrendGridConfig, GridPortfolioConfig]] = []
    for risk in (0.02, 0.04, 0.06, 0.08, 0.10):
        for max_campaigns in (1, 2, 3):
            portfolio = replace(
                base_portfolio,
                risk_per_campaign_pct=risk,
                max_open_campaigns=max_campaigns,
                max_daily_campaigns=12,
                hard_drawdown_stop_pct=0.70,
            )
            risk_variants.append((f"risk{risk:.2f}_campaigns{max_campaigns}", selected_signal, portfolio))
    stage5 = _run_stage(
        "stage5_risk", risk_variants,
        universe, signal_data, execution_data, rules, execution, periods, initial_equity, timeline_cache,
    )
    split_candidates = sorted(
        stage5,
        key=lambda row: _selection_score(row, historical_max=True),
        reverse=True,
    )[:5]
    _attach_split_results(
        split_candidates,
        universe,
        execution_data,
        rules,
        execution,
        periods,
        initial_equity,
        timeline_cache,
    )
    stages["stage5_risk"] = [_public_row(row) for row in stage5]
    validation_selected = _select(split_candidates)
    historical_selected = _select(stage5, historical_max=True)

    selected_signal = TrendGridConfig(**historical_selected["signal_config"])
    selected_portfolio = GridPortfolioConfig(**historical_selected["portfolio_config"])
    cache_key = _signal_cache_key(selected_signal)
    candidates, snapshots = timeline_cache[cache_key]
    full_result = simulate_grid_portfolio(
        candidates, snapshots, universe, execution_data, rules, selected_signal,
        selected_portfolio, execution, start, end, initial_equity,
    )
    validation_signal = TrendGridConfig(**validation_selected["signal_config"])
    validation_portfolio = GridPortfolioConfig(**validation_selected["portfolio_config"])
    validation_candidates, validation_snapshots = timeline_cache[_signal_cache_key(validation_signal)]
    validation_frozen_result = simulate_grid_portfolio(
        validation_candidates, validation_snapshots, universe, execution_data, rules,
        validation_signal, validation_portfolio, execution, start, end, initial_equity,
    )

    top_symbol = max(
        full_result["by_symbol"],
        key=lambda symbol: full_result["by_symbol"][symbol]["net_pnl"],
        default="",
    )
    stress = {
        "fixed_risk_no_compounding": compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                replace(selected_portfolio, compound=False), execution, start, end, initial_equity,
            )
        ),
        "entry_delay_1m": compact_grid_summary(
            simulate_grid_portfolio(
                _shift_candidates(candidates, 1, execution_data), snapshots, universe,
                execution_data, rules, selected_signal, selected_portfolio, execution,
                start, end, initial_equity,
            )
        ),
        "cost_1.5x": compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                selected_portfolio, _scaled_execution(execution, 1.5), start, end, initial_equity,
            )
        ),
    }
    if top_symbol:
        stress["exclude_top_symbol"] = compact_grid_summary(
            simulate_grid_portfolio(
                candidates, snapshots, universe, execution_data, rules, selected_signal,
                selected_portfolio, execution, start, end, initial_equity,
                skip_symbols=frozenset({top_symbol}),
            )
        )
        stress["exclude_top_symbol"]["excluded_symbol"] = top_symbol

    return {
        "universe_size": len(universe),
        "symbols": list(universe),
        "periods": {key: {"start": value[0].isoformat(), "end": value[1].isoformat()} for key, value in periods.items()},
        "selected_signal_config": selected_signal.as_dict(),
        "selected_portfolio_config": asdict(selected_portfolio),
        "validation_selected_signal_config": validation_signal.as_dict(),
        "validation_selected_portfolio_config": asdict(validation_portfolio),
        "stages": stages,
        "final_result": full_result,
        "validation_frozen_full_result": validation_frozen_result,
        "historical_test_result": historical_selected["test"],
        "validation_selected_historical_test": validation_selected["test"],
        "stress_tests": stress,
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Dynamic Trend-Following Grid - 3 Month Full-Cost Research",
        "",
        f"- Period: `{report['period']['start']}` to `{report['period']['end']}`",
        f"- Initial equity: `{report['initial_equity']:.2f}U`",
        "- Execution: closed high-timeframe signal, next 1m open, stop-first path, maker grid fills, full cost and actual funding",
        "- The historical maximum inspects all three months and is not an untouched OOS result",
        "",
        "| Universe | Campaigns | Grid fills | Grid TP | Net | Return | PF | Win rate | Max DD | Test net |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in ("30", "50"):
        result = report["universes"][size]["final_result"]
        test = report["universes"][size]["historical_test_result"]
        lines.append(
            f"| {size} | {result['trade_count']} | {result['grid_entry_count']} | "
            f"{result['grid_take_profit_count']} | {result['net_profit']:+.2f}U | "
            f"{result['return_pct']:.2%} | {result['profit_factor']:.3f} | "
            f"{result['win_rate']:.2%} | {result['max_drawdown_pct']:.2%} | "
            f"{test['net_profit']:+.2f}U |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize a full-cost dynamic trend-following grid")
    parser.add_argument("--signal-data-dir", default="data/binance_15m_365d_top100")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--funding-data-dir", default="data/binance_funding_365d_top100")
    parser.add_argument("--cost-config", default="config.gui.mtf-momentum-reset-stage21.json")
    parser.add_argument("--start", default="2026-03-06T00:00:00")
    parser.add_argument("--end", default="2026-06-06T00:00:00")
    parser.add_argument("--initial-equity", type=float, default=200.0)
    parser.add_argument("--output", default="reports/trend_grid_3m_30_vs_50.json")
    parser.add_argument("--summary", default="reports/trend_grid_3m_30_vs_50.md")
    parser.add_argument("--config30", default="config.trend-grid.optimized-30.json")
    parser.add_argument("--config50", default="config.trend-grid.optimized-50.json")
    parser.add_argument("--manifest", default="config.trend-grid.3m-optimized-manifest.json")
    return parser


def run_optimization(args: argparse.Namespace) -> dict[str, Any]:
    start, end, signal_data, execution_data, rules, execution, metadata = _load_runtime_inputs(args)
    universes: dict[str, Any] = {}
    for size, universe in (("30", UNIVERSE_30), ("50", UNIVERSE_50)):
        print(f"starting trend-grid research for {size} symbols", flush=True)
        universes[size] = optimize_grid_universe(
            universe, signal_data, execution_data, rules, execution,
            start, end, args.initial_equity,
        )
        result = universes[size]["final_result"]
        print(
            f"completed {size}: campaigns={result['trade_count']} net={result['net_profit']:.2f} "
            f"pf={result['profit_factor']:.3f} dd={result['max_drawdown_pct']:.2%}",
            flush=True,
        )
    report = {
        "strategy_name": TREND_GRID_STRATEGY_NAME,
        "strategy_version": TREND_GRID_VERSION,
        "research_status": "historical_staged_optimization_not_untouched_oos",
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "initial_equity": args.initial_equity,
        "preserved_gui_strategy": "config.volatility-breakout.v2-balanced-50-shadow.json",
        "cost_model": {
            "source_config": args.cost_config,
            "source_config_sha256": sha256_file(args.cost_config),
            "mode": execution.mode,
            "market_slippage_bps": execution.market_slippage_bps,
            "stop_slippage_bps": execution.stop_slippage_bps,
            "take_profit_slippage_bps": execution.take_profit_slippage_bps,
            "maker_fee_rate": execution.maker_fee_rate,
            "taker_fee_rate": execution.taker_fee_rate,
            "funding_enabled": execution.funding_enabled,
            "funding_missing_symbols": metadata["funding_missing"],
        },
        "execution_rules": {
            "signal": "closed 15m/30m/60m only",
            "entry": "next 1m open",
            "grid_limit_fill": "must trade through by at least one tick",
            "same_bar_conflict": "hard stop before grid take-profit; new fills cannot take profit in same minute",
            "old_campaign_exit_before_new_entry": True,
        },
        "universes": universes,
    }
    _write_json(Path(args.output), report)
    _write_summary(Path(args.summary), report)
    for size, config_path in (("30", args.config30), ("50", args.config50)):
        result = universes[size]
        _write_json(
            Path(config_path),
            {
                "strategy_name": TREND_GRID_STRATEGY_NAME,
                "status": "historical_maximum_not_live",
                "period": report["period"],
                "symbols": result["symbols"],
                "signal": result["selected_signal_config"],
                "portfolio": result["selected_portfolio_config"],
                "validation_selected_signal": result["validation_selected_signal_config"],
                "validation_selected_portfolio": result["validation_selected_portfolio_config"],
                "cost_model": report["cost_model"],
                "execution_rules": report["execution_rules"],
            },
        )
    artifacts = (
        args.output,
        args.summary,
        args.config30,
        args.config50,
        "crypto_scalper/trend_grid.py",
        "crypto_scalper/trend_grid_optimize.py",
        "tests/test_trend_grid.py",
        "config.volatility-breakout.v2-balanced-50-shadow-manifest.json",
    )
    _write_json(
        Path(args.manifest),
        {
            "strategy_name": TREND_GRID_STRATEGY_NAME,
            "status": "historical_research_frozen_separately_from_gui_strategy",
            "report": args.output,
            "summary": args.summary,
            "configs": [args.config30, args.config50],
            "preserved_gui_strategy": "config.volatility-breakout.v2-balanced-50-shadow.json",
            "hashes": {path: sha256_file(path) for path in artifacts if Path(path).exists()},
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_optimization(args)
    output = {
        size: compact_grid_summary(report["universes"][size]["final_result"])
        for size in ("30", "50")
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
