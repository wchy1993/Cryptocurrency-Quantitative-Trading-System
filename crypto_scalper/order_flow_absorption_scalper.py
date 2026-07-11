from __future__ import annotations

import json
import math
import statistics
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .binance_client import SymbolRules
from .models import Direction


STRATEGY_NAME = "order_flow_absorption_scalper"
STRATEGY_CODE = "OFAS"
EPSILON = 1e-12


class OfasState(str, Enum):
    IDLE = "IDLE"
    SELL_SHOCK_PENDING = "SELL_SHOCK_PENDING"
    LONG_ABSORPTION_CONFIRMING = "LONG_ABSORPTION_CONFIRMING"
    LONG_ENTRY_READY = "LONG_ENTRY_READY"
    LONG_ORDER_WORKING = "LONG_ORDER_WORKING"
    LONG_POSITION_OPEN = "LONG_POSITION_OPEN"
    BUY_SHOCK_PENDING = "BUY_SHOCK_PENDING"
    SHORT_ABSORPTION_CONFIRMING = "SHORT_ABSORPTION_CONFIRMING"
    SHORT_ENTRY_READY = "SHORT_ENTRY_READY"
    SHORT_ORDER_WORKING = "SHORT_ORDER_WORKING"
    SHORT_POSITION_OPEN = "SHORT_POSITION_OPEN"
    POSITION_EXITING = "POSITION_EXITING"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class OfasConfig:
    enabled: bool = False
    symbols: tuple[str, ...] = ()
    allow_long: bool = True
    allow_short: bool = True
    evaluation_interval_ms: int = 250
    trade_stale_ms: int = 5_000
    depth_stale_ms: int = 1_000
    btc_stale_ms: int = 1_500
    baseline_seconds: int = 600
    baseline_min_samples: int = 60
    sell_shock_window_seconds: int = 5
    buy_shock_window_seconds: int = 5
    sell_shock_imbalance_threshold: float = 0.70
    buy_shock_imbalance_threshold: float = 0.72
    sell_shock_volume_zscore: float = 2.0
    buy_shock_volume_zscore: float = 2.5
    sell_shock_price_sigma: float = 2.0
    buy_shock_price_sigma: float = 2.5
    sell_shock_min_return_bps: float = 4.0
    buy_shock_min_return_bps: float = 5.0
    pending_expiry_seconds: int = 12
    event_merge_seconds: int = 15
    event_merge_price_bps: float = 8.0
    no_new_low_seconds: float = 3.0
    no_new_high_seconds: float = 3.0
    replenishment_window_seconds: float = 3.0
    bid_replenishment_min: float = 0.65
    ask_replenishment_min: float = 0.70
    normalized_ofi_long_min: float = 0.08
    normalized_ofi_short_max: float = -0.10
    microprice_edge_long_bps: float = 0.5
    microprice_edge_short_bps: float = -0.6
    sell_exhaustion_threshold: float = 0.62
    buy_exhaustion_threshold: float = 0.62
    impact_efficiency_decay_ratio: float = 0.80
    max_long_entry_rebound_bps: float = 12.0
    max_short_entry_pullback_bps: float = 10.0
    max_spread_bps: float = 4.0
    max_spread_vs_median: float = 2.0
    max_spread_percentile: float = 0.95
    min_l10_depth_usdt: float = 50_000.0
    min_trades_per_second: float = 2.0
    max_order_to_l10_depth_ratio: float = 0.03
    target_move_bps: float = 20.0
    max_spread_to_target_ratio: float = 0.20
    min_expected_move_to_cost_multiple: float = 4.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    estimated_slippage_bps: float = 1.5
    stop_extra_slippage_bps: float = 3.0
    entry_mode: str = "maker-first"
    maker_wait_ms: int = 800
    maker_price_offset_ticks: int = 0
    taker_min_expected_move_to_cost_multiple: float = 6.0
    stop_short_vol_mult: float = 1.0
    stop_event_range_mult: float = 0.20
    min_stop_bps: float = 5.0
    max_stop_bps: float = 30.0
    take_profit_r: float = 1.8
    ofi_exit_threshold: float = 0.12
    microprice_reversal_bps: float = 0.4
    microprice_reversal_hold_ms: int = 500
    replenishment_exit_min: float = 0.20
    spread_emergency_bps: float = 10.0
    fail_fast_seconds: int = 60
    fail_fast_min_r: float = 0.20
    time_stop_seconds: int = 180
    base_risk_pct: float = 0.0015
    max_trade_risk_pct: float = 0.0020
    short_risk_multiplier: float = 0.60
    quality_b_risk_multiplier: float = 0.60
    quality_a_min_score: float = 75.0
    quality_b_min_score: float = 60.0
    max_notional_usdt: float = 5_000.0
    max_margin_usage_pct: float = 0.10
    max_symbol_margin_pct: float = 0.05
    leverage: float = 5.0
    max_open_positions: int = 1
    max_daily_trades: int = 10
    global_min_entry_interval_seconds: int = 60
    symbol_cooldown_minutes: int = 20
    symbol_loss_cooldown_minutes: int = 60
    consecutive_loss_limit: int = 3
    consecutive_loss_pause_minutes: int = 120
    direction_consecutive_loss_limit: int = 3
    direction_pause_minutes: int = 180
    symbol_consecutive_loss_limit: int = 2
    daily_loss_stop_pct: float = 0.0075
    soft_drawdown_pct: float = 0.020
    soft_drawdown_risk_multiplier: float = 0.60
    hard_drawdown_pct: float = 0.050
    hard_drawdown_pause_minutes: int = 1_440
    btc_adverse_return_2s_bps: float = 8.0
    btc_adverse_return_5s_bps: float = 15.0
    short_min_funding_rate: float = -0.0002
    short_max_recent_drop_bps: float = 40.0
    lower_wick_short_block_ratio: float = 0.60
    quality_weights: dict[str, float] = field(default_factory=lambda: {
        "shock": 0.25,
        "absorption": 0.30,
        "book": 0.20,
        "market": 0.10,
        "execution": 0.15,
    })

    def __post_init__(self) -> None:
        symbols = tuple(dict.fromkeys(str(s).upper() for s in self.symbols if str(s).strip()))
        object.__setattr__(self, "symbols", symbols)
        if self.entry_mode not in {"maker-first", "taker-confirmation"}:
            raise ValueError("OFAS entry_mode must be maker-first or taker-confirmation")
        if self.sell_shock_window_seconds not in {5, 15} or self.buy_shock_window_seconds not in {5, 15}:
            raise ValueError("OFAS shock windows must be 5 or 15 seconds")
        if self.max_open_positions != 1:
            raise ValueError("OFAS v1 requires max_open_positions=1")
        if not 0 < self.base_risk_pct <= self.max_trade_risk_pct:
            raise ValueError("OFAS base_risk_pct must be positive and capped by max_trade_risk_pct")
        if not 0 < self.short_risk_multiplier <= 1:
            raise ValueError("OFAS short_risk_multiplier must be in (0, 1]")
        required_weights = {"shock", "absorption", "book", "market", "execution"}
        if set(self.quality_weights) != required_weights or sum(self.quality_weights.values()) <= 0:
            raise ValueError("OFAS quality_weights must define shock/absorption/book/market/execution")


def load_ofas_config(path: str | Path) -> OfasConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("ofas_strategy", raw)
    allowed = {item.name for item in fields(OfasConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown OFAS config keys: {', '.join(unknown)}")
    return OfasConfig(**values)


@dataclass(frozen=True)
class AggTradeEvent:
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    buyer_is_maker: bool
    aggregate_trade_id: int | None = None

    @property
    def quote_volume(self) -> float:
        return self.price * self.quantity

    @property
    def taker_direction(self) -> Direction:
        return Direction.SHORT if self.buyer_is_maker else Direction.LONG


@dataclass(frozen=True)
class DepthUpdateEvent:
    symbol: str
    timestamp: datetime
    first_update_id: int
    final_update_id: int
    previous_final_update_id: int | None
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class BookSnapshot:
    symbol: str
    timestamp: datetime
    last_update_id: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MarketContext:
    timestamp: datetime
    equity: float
    available_balance: float
    open_positions: int
    btc_return_2s_bps: float = 0.0
    btc_return_5s_bps: float = 0.0
    btc_data_age_ms: int = 0
    funding_rate: float = 0.0
    recent_return_60s_bps: float = 0.0
    lower_wick_ratio: float = 0.0
    market_volatility_score: float = 0.0


@dataclass(frozen=True)
class OfasAction:
    action: str
    symbol: str
    side: Direction = Direction.FLAT
    reason: str = ""
    event_id: str = ""
    order_type: str = ""
    price: float = 0.0
    quantity: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    reduce_only: bool = False
    post_only: bool = False
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OfasFeatures:
    timestamp: datetime
    mid_price: float
    best_bid: float
    best_ask: float
    spread_bps: float
    spread_vs_recent_median: float
    spread_percentile: float
    bid_depth_l1: float
    ask_depth_l1: float
    bid_depth_l5: float
    ask_depth_l5: float
    bid_depth_l10: float
    ask_depth_l10: float
    buy_imbalance_2s: float
    buy_imbalance_5s: float
    buy_imbalance_15s: float
    sell_imbalance_2s: float
    sell_imbalance_5s: float
    sell_imbalance_15s: float
    buy_volume_zscore: float
    sell_volume_zscore: float
    total_volume_zscore: float
    return_2s: float
    return_5s: float
    return_15s: float
    realized_volatility_60s: float
    price_shock_sigma: float
    ofi_l1: float
    ofi_l5: float
    ofi_l10: float
    normalized_ofi_l1: float
    normalized_ofi_l5: float
    normalized_ofi_l10: float
    microprice: float
    microprice_edge_bps: float
    sell_impact_efficiency: float
    buy_impact_efficiency: float
    sell_impact_efficiency_decreasing: bool
    buy_impact_efficiency_decreasing: bool
    trades_per_second: float


class LocalOrderBook:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int | None = None
        self.last_timestamp: datetime | None = None
        self.valid = False
        self.invalid_reason = "snapshot_required"
        self._updates_since_snapshot = 0

    def apply_snapshot(self, snapshot: BookSnapshot) -> None:
        if snapshot.symbol.upper() != self.symbol:
            raise ValueError("snapshot symbol mismatch")
        self.bids = {float(p): float(q) for p, q in snapshot.bids if p > 0 and q > 0}
        self.asks = {float(p): float(q) for p, q in snapshot.asks if p > 0 and q > 0}
        self.last_update_id = snapshot.last_update_id
        self.last_timestamp = snapshot.timestamp
        self._updates_since_snapshot = 0
        self.valid = self._shape_valid()
        self.invalid_reason = "" if self.valid else "invalid_snapshot"

    def apply_update(self, update: DepthUpdateEvent) -> bool:
        if update.symbol.upper() != self.symbol or self.last_update_id is None or not self.valid:
            self.invalidate("snapshot_required")
            return False
        expected = self.last_update_id + 1
        if self._updates_since_snapshot == 0:
            continuous = update.first_update_id <= expected <= update.final_update_id
        else:
            continuous = update.previous_final_update_id == self.last_update_id
        if not continuous:
            self.invalidate("update_id_gap")
            return False
        for price, quantity in update.bids:
            self._apply_level(self.bids, price, quantity)
        for price, quantity in update.asks:
            self._apply_level(self.asks, price, quantity)
        self.last_update_id = update.final_update_id
        self.last_timestamp = update.timestamp
        self._updates_since_snapshot += 1
        self.valid = self._shape_valid()
        self.invalid_reason = "" if self.valid else "crossed_or_empty_book"
        return self.valid

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.invalid_reason = reason

    def levels(self, side: str, limit: int = 10) -> tuple[tuple[float, float], ...]:
        source = self.bids if side == "bid" else self.asks
        return tuple(sorted(source.items(), reverse=side == "bid")[:limit])

    @property
    def best_bid(self) -> tuple[float, float]:
        rows = self.levels("bid", 1)
        return rows[0] if rows else (0.0, 0.0)

    @property
    def best_ask(self) -> tuple[float, float]:
        rows = self.levels("ask", 1)
        return rows[0] if rows else (0.0, 0.0)

    def quote_depth(self, side: str, levels: int) -> float:
        return sum(price * quantity for price, quantity in self.levels(side, levels))

    def _shape_valid(self) -> bool:
        bid, _ = self.best_bid
        ask, _ = self.best_ask
        return bid > 0 and ask > 0 and bid < ask

    @staticmethod
    def _apply_level(levels: dict[float, float], price: float, quantity: float) -> None:
        price = float(price)
        quantity = float(quantity)
        if price <= 0 or quantity < 0:
            return
        if quantity == 0:
            levels.pop(price, None)
        else:
            levels[price] = quantity


@dataclass
class PendingAbsorption:
    event_id: str
    symbol: str
    side: Direction
    shock_time: datetime
    expire_time: datetime
    shock_start_price: float
    shock_extreme: float
    last_extreme_time: datetime
    shock_return_bps: float
    shock_imbalance: float
    shock_volume_zscore: float
    spread_bps: float
    initial_same_side_depth: float
    min_same_side_depth: float
    min_depth_time: datetime
    short_term_volatility: float
    btc_return_5s_bps: float
    consumed: bool = False


@dataclass
class OfasPosition:
    event_id: str
    side: Direction
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_price: float
    take_profit_price: float
    risk_per_unit: float
    quality_score: float
    quality_bucket: str
    entry_order_type: str
    peak_price: float
    trough_price: float
    microprice_reversal_since: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymbolRuntime:
    state: OfasState = OfasState.IDLE
    pending: PendingAbsorption | None = None
    position: OfasPosition | None = None
    working_order: OfasAction | None = None
    cooldown_until: datetime | None = None
    last_evaluation: datetime | None = None
    last_trade_time: datetime | None = None
    last_depth_time: datetime | None = None
    last_event_signature: tuple[Direction, datetime, float] | None = None
    used_event_ids: set[str] = field(default_factory=set)


@dataclass
class OfasRiskState:
    starting_equity: float
    peak_equity: float
    day: Any = None
    day_start_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    direction_losses: dict[Direction, int] = field(default_factory=lambda: {Direction.LONG: 0, Direction.SHORT: 0})
    symbol_losses: Counter[str] = field(default_factory=Counter)
    global_pause_until: datetime | None = None
    direction_pause_until: dict[Direction, datetime] = field(default_factory=dict)
    hard_pause_until: datetime | None = None
    last_entry_time: datetime | None = None

    def roll_day(self, now: datetime, equity: float) -> None:
        if self.day != now.date():
            self.day = now.date()
            self.day_start_equity = equity
            self.daily_realized_pnl = 0.0
            self.daily_trades = 0
        self.peak_equity = max(self.peak_equity, equity)


class RollingOrderFlow:
    def __init__(self, config: OfasConfig) -> None:
        self.config = config
        self.trades: deque[AggTradeEvent] = deque()
        self.prices: deque[tuple[datetime, float]] = deque()
        self.ofi_events: deque[tuple[datetime, tuple[float, float, float]]] = deque()
        self.spreads: deque[tuple[datetime, float]] = deque()
        self.volume_baseline: deque[tuple[datetime, float, float, float, float]] = deque()
        self._last_baseline_sample: datetime | None = None
        self.previous_levels: dict[int, tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]] = {}
        self.last_trade_id: int | None = None
        self.last_trade_timestamp: datetime | None = None

    def add_trade(self, event: AggTradeEvent) -> bool:
        if self.last_trade_timestamp and event.timestamp < self.last_trade_timestamp:
            return False
        if event.aggregate_trade_id is not None and self.last_trade_id is not None and event.aggregate_trade_id <= self.last_trade_id:
            return False
        self.trades.append(event)
        self.prices.append((event.timestamp, event.price))
        self.last_trade_timestamp = event.timestamp
        if event.aggregate_trade_id is not None:
            self.last_trade_id = event.aggregate_trade_id
        self._prune(event.timestamp)
        return True

    def add_book(self, now: datetime, book: LocalOrderBook) -> None:
        bid, _ = book.best_bid
        ask, _ = book.best_ask
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) / 2.0
        self.prices.append((now, mid))
        self.spreads.append((now, (ask - bid) / mid * 10_000.0))
        ofi = []
        for depth in (1, 5, 10):
            current = (book.levels("bid", depth), book.levels("ask", depth))
            previous = self.previous_levels.get(depth)
            ofi.append(_book_ofi(previous, current))
            self.previous_levels[depth] = current
        self.ofi_events.append((now, (ofi[0], ofi[1], ofi[2])))
        self._sample_baseline(now)
        self._prune(now)

    def features(self, now: datetime, book: LocalOrderBook) -> OfasFeatures | None:
        if not book.valid:
            return None
        bid, bid_qty = book.best_bid
        ask, ask_qty = book.best_ask
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        volumes = {window: self._trade_volume(now, window) for window in (2, 5, 15)}
        returns = {window: self._return(now, window, mid) for window in (2, 5, 15)}
        buy5, sell5 = volumes[5]
        total5 = buy5 + sell5
        buy_slot = 1 if self.config.buy_shock_window_seconds == 5 else 3
        sell_slot = 2 if self.config.sell_shock_window_seconds == 5 else 4
        baseline_buy = [row[buy_slot] for row in self.volume_baseline]
        baseline_sell = [row[sell_slot] for row in self.volume_baseline]
        current_buy = volumes[self.config.buy_shock_window_seconds][0]
        current_sell = volumes[self.config.sell_shock_window_seconds][1]
        return_samples = self._one_second_returns(now)
        return_sigma = statistics.pstdev(return_samples) if len(return_samples) >= 5 else 0.0
        shock_sigma = abs(returns[5]) / max(return_sigma * math.sqrt(5), EPSILON)
        ofi = self._ofi(now, 2)
        depths = {
            ("bid", 1): book.quote_depth("bid", 1),
            ("ask", 1): book.quote_depth("ask", 1),
            ("bid", 5): book.quote_depth("bid", 5),
            ("ask", 5): book.quote_depth("ask", 5),
            ("bid", 10): book.quote_depth("bid", 10),
            ("ask", 10): book.quote_depth("ask", 10),
        }
        normalized = (
            ofi[0] / max(depths[("bid", 1)] + depths[("ask", 1)], EPSILON),
            ofi[1] / max(depths[("bid", 5)] + depths[("ask", 5)], EPSILON),
            ofi[2] / max(depths[("bid", 10)] + depths[("ask", 10)], EPSILON),
        )
        microprice = (ask * bid_qty + bid * ask_qty) / max(bid_qty + ask_qty, EPSILON)
        spreads = [value for ts, value in self.spreads if ts >= now - timedelta(seconds=60)]
        spread = (ask - bid) / mid * 10_000.0
        median_spread = statistics.median(spreads) if spreads else spread
        sell_eff2 = max(0.0, -self._return(now, 2, mid) * 10_000.0) / max(self._trade_volume(now, 2)[1], EPSILON)
        sell_eff5 = max(0.0, -returns[5] * 10_000.0) / max(sell5, EPSILON)
        buy_eff2 = max(0.0, self._return(now, 2, mid) * 10_000.0) / max(self._trade_volume(now, 2)[0], EPSILON)
        buy_eff5 = max(0.0, returns[5] * 10_000.0) / max(buy5, EPSILON)
        return OfasFeatures(
            timestamp=now,
            mid_price=mid,
            best_bid=bid,
            best_ask=ask,
            spread_bps=spread,
            spread_vs_recent_median=spread / max(median_spread, EPSILON),
            spread_percentile=_percentile_rank(spreads, spread),
            bid_depth_l1=depths[("bid", 1)], ask_depth_l1=depths[("ask", 1)],
            bid_depth_l5=depths[("bid", 5)], ask_depth_l5=depths[("ask", 5)],
            bid_depth_l10=depths[("bid", 10)], ask_depth_l10=depths[("ask", 10)],
            buy_imbalance_2s=_ratio(volumes[2][0], sum(volumes[2])),
            buy_imbalance_5s=_ratio(buy5, total5),
            buy_imbalance_15s=_ratio(volumes[15][0], sum(volumes[15])),
            sell_imbalance_2s=_ratio(volumes[2][1], sum(volumes[2])),
            sell_imbalance_5s=_ratio(sell5, total5),
            sell_imbalance_15s=_ratio(volumes[15][1], sum(volumes[15])),
            buy_volume_zscore=_robust_zscore(baseline_buy, current_buy),
            sell_volume_zscore=_robust_zscore(baseline_sell, current_sell),
            total_volume_zscore=_robust_zscore([row[1] + row[2] for row in self.volume_baseline], total5),
            return_2s=returns[2], return_5s=returns[5], return_15s=returns[15],
            realized_volatility_60s=math.sqrt(sum(x * x for x in return_samples)),
            price_shock_sigma=shock_sigma,
            ofi_l1=ofi[0], ofi_l5=ofi[1], ofi_l10=ofi[2],
            normalized_ofi_l1=normalized[0], normalized_ofi_l5=normalized[1], normalized_ofi_l10=normalized[2],
            microprice=microprice,
            microprice_edge_bps=(microprice - mid) / mid * 10_000.0,
            sell_impact_efficiency=sell_eff2,
            buy_impact_efficiency=buy_eff2,
            sell_impact_efficiency_decreasing=sell_eff2 <= sell_eff5 * self.config.impact_efficiency_decay_ratio,
            buy_impact_efficiency_decreasing=buy_eff2 <= buy_eff5 * self.config.impact_efficiency_decay_ratio,
            trades_per_second=sum(1 for row in self.trades if row.timestamp >= now - timedelta(seconds=5)) / 5.0,
        )

    def baseline_ready(self) -> bool:
        return len(self.volume_baseline) >= self.config.baseline_min_samples

    def _sample_baseline(self, now: datetime) -> None:
        if self._last_baseline_sample and (now - self._last_baseline_sample).total_seconds() < 1:
            return
        buy5, sell5 = self._trade_volume(now, 5)
        buy15, sell15 = self._trade_volume(now, 15)
        self.volume_baseline.append((now, buy5, sell5, buy15, sell15))
        self._last_baseline_sample = now

    def _trade_volume(self, now: datetime, seconds: int) -> tuple[float, float]:
        start = now - timedelta(seconds=seconds)
        buy = sum(row.quote_volume for row in self.trades if row.timestamp >= start and row.taker_direction == Direction.LONG)
        sell = sum(row.quote_volume for row in self.trades if row.timestamp >= start and row.taker_direction == Direction.SHORT)
        return buy, sell

    def _return(self, now: datetime, seconds: int, current: float) -> float:
        start = now - timedelta(seconds=seconds)
        eligible = [price for ts, price in self.prices if ts <= start]
        reference = eligible[-1] if eligible else (self.prices[0][1] if self.prices else current)
        return current / max(reference, EPSILON) - 1.0

    def _ofi(self, now: datetime, seconds: int) -> tuple[float, float, float]:
        start = now - timedelta(seconds=seconds)
        rows = [values for ts, values in self.ofi_events if ts >= start]
        return tuple(sum(row[i] for row in rows) for i in range(3))  # type: ignore[return-value]

    def _one_second_returns(self, now: datetime) -> list[float]:
        rows = [(ts, price) for ts, price in self.prices if ts >= now - timedelta(seconds=60)]
        sampled: list[float] = []
        last_second = None
        last_price = None
        for ts, price in rows:
            second = ts.replace(microsecond=0)
            if second == last_second:
                if sampled and last_price:
                    sampled[-1] = price / last_price - 1.0
                continue
            if last_price:
                sampled.append(price / last_price - 1.0)
            last_second = second
            last_price = price
        return sampled

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=max(60, self.config.baseline_seconds))
        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()
        for queue in (self.prices, self.ofi_events, self.spreads, self.volume_baseline):
            while queue and queue[0][0] < cutoff:
                queue.popleft()


class OrderFlowAbsorptionScalper:
    def __init__(self, config: OfasConfig, rules: dict[str, SymbolRules]) -> None:
        self.config = config
        self.rules = {symbol.upper(): value for symbol, value in rules.items()}
        self.books = {symbol: LocalOrderBook(symbol) for symbol in config.symbols}
        self.flows = {symbol: RollingOrderFlow(config) for symbol in config.symbols}
        self.runtime = {symbol: SymbolRuntime() for symbol in config.symbols}
        self.risk: OfasRiskState | None = None
        self.reject_reasons: Counter[str] = Counter()
        self.stats: Counter[str] = Counter()
        self.closed_trades: list[dict[str, Any]] = []

    def on_book_snapshot(self, snapshot: BookSnapshot) -> None:
        symbol = snapshot.symbol.upper()
        if symbol not in self.books:
            return
        self.books[symbol].apply_snapshot(snapshot)
        self.runtime[symbol].last_depth_time = snapshot.timestamp
        self.flows[symbol].add_book(snapshot.timestamp, self.books[symbol])

    def on_depth_update(self, update: DepthUpdateEvent) -> list[OfasAction]:
        symbol = update.symbol.upper()
        if symbol not in self.books:
            return []
        book = self.books[symbol]
        if not book.apply_update(update):
            runtime = self.runtime[symbol]
            actions = self._cancel_pending_or_order(symbol, "order_book_invalid")
            if runtime.position:
                runtime.state = OfasState.POSITION_EXITING
                actions.append(self._exit_action(symbol, runtime.position, "order_book_invalid"))
            actions.append(OfasAction("RESYNC_BOOK", symbol, reason=book.invalid_reason))
            self.stats["order_book_invalid"] += 1
            return actions
        self.runtime[symbol].last_depth_time = update.timestamp
        self.flows[symbol].add_book(update.timestamp, book)
        return []

    def on_order_book_invalid(self, symbol: str, reason: str = "order_book_invalid") -> list[OfasAction]:
        symbol = symbol.upper()
        if symbol not in self.books:
            return []
        self.books[symbol].invalidate(reason)
        runtime = self.runtime[symbol]
        actions = self._cancel_pending_or_order(symbol, "order_book_invalid")
        if runtime.position:
            runtime.state = OfasState.POSITION_EXITING
            actions.append(self._exit_action(symbol, runtime.position, "order_book_invalid"))
        self.stats["order_book_invalid"] += 1
        return actions

    def on_agg_trade(self, event: AggTradeEvent) -> None:
        symbol = event.symbol.upper()
        if symbol not in self.flows or event.price <= 0 or event.quantity <= 0:
            return
        if not self.flows[symbol].add_trade(event):
            self.stats["duplicate_or_out_of_order_trade"] += 1
            return
        self.runtime[symbol].last_trade_time = event.timestamp

    def evaluate(self, symbol: str, context: MarketContext) -> list[OfasAction]:
        symbol = symbol.upper()
        if not self.config.enabled or symbol not in self.runtime:
            return []
        runtime = self.runtime[symbol]
        if runtime.last_evaluation and (context.timestamp - runtime.last_evaluation).total_seconds() * 1000 < self.config.evaluation_interval_ms:
            return []
        runtime.last_evaluation = context.timestamp
        self._ensure_risk(context)
        assert self.risk is not None
        self.risk.roll_day(context.timestamp, context.equity)
        if runtime.state == OfasState.COOLDOWN:
            if runtime.cooldown_until and context.timestamp < runtime.cooldown_until:
                return []
            runtime.state = OfasState.IDLE
            runtime.cooldown_until = None
        stale = self._data_reject_reason(symbol, context)
        if stale:
            self._reject(stale)
            actions = self._cancel_pending_or_order(symbol, stale)
            return actions
        features = self.flows[symbol].features(context.timestamp, self.books[symbol])
        if features is None:
            self._reject("order_book_invalid")
            return []
        if runtime.position:
            return self._manage_position(symbol, runtime, features, context)
        if runtime.working_order:
            return self._manage_working_order(symbol, runtime, features, context)
        if runtime.pending:
            return self._manage_pending(symbol, runtime, features, context)
        if runtime.state == OfasState.IDLE:
            return self._detect_shock(symbol, runtime, features, context)
        return []

    def on_entry_fill(self, symbol: str, timestamp: datetime, price: float, quantity: float) -> list[OfasAction]:
        symbol = symbol.upper()
        runtime = self.runtime[symbol]
        order = runtime.working_order
        pending = runtime.pending
        if not order or not pending or pending.event_id in runtime.used_event_ids or quantity <= 0:
            return []
        stop_price = order.stop_price
        tp_price = order.take_profit_price
        risk_per_unit = abs(price - stop_price)
        metadata = order.metadata
        runtime.position = OfasPosition(
            event_id=pending.event_id, side=order.side, entry_time=timestamp, entry_price=price,
            quantity=quantity, stop_price=stop_price, take_profit_price=tp_price,
            risk_per_unit=risk_per_unit, quality_score=float(metadata["quality_score"]),
            quality_bucket=str(metadata["quality_bucket"]), entry_order_type=order.order_type,
            peak_price=price, trough_price=price, metadata=dict(metadata),
        )
        runtime.used_event_ids.add(pending.event_id)
        pending.consumed = True
        runtime.working_order = None
        runtime.state = OfasState.LONG_POSITION_OPEN if order.side == Direction.LONG else OfasState.SHORT_POSITION_OPEN
        self.stats["entry_count"] += 1
        if order.order_type == "LIMIT_MAKER":
            self.stats["maker_filled"] += 1
        exit_side = Direction.SHORT if order.side == Direction.LONG else Direction.LONG
        actions: list[OfasAction] = []
        if order.order_type == "LIMIT_MAKER":
            actions.append(OfasAction("CANCEL_ENTRY_REMAINDER", symbol, order.side, "maker_partial_or_complete_fill", pending.event_id))
        actions.extend([
            OfasAction("PLACE_PROTECTIVE_STOP", symbol, exit_side, "protective_stop", pending.event_id,
                       "STOP_MARKET", stop_price, quantity, reduce_only=True, metadata={"working_type": "MARK_PRICE"}),
            OfasAction("PLACE_TAKE_PROFIT", symbol, exit_side, "take_profit", pending.event_id,
                       "TAKE_PROFIT_MARKET", tp_price, quantity, reduce_only=True, metadata={"working_type": "MARK_PRICE"}),
        ])
        return actions

    def on_protection_rejected(self, symbol: str, reason: str = "protective_stop_rejected") -> list[OfasAction]:
        symbol = symbol.upper()
        runtime = self.runtime[symbol]
        if not runtime.position:
            return []
        runtime.state = OfasState.POSITION_EXITING
        self.stats["protection_failure"] += 1
        return [self._exit_action(symbol, runtime.position, reason)]

    def on_entry_rejected(self, symbol: str, reason: str, now: datetime) -> None:
        runtime = self.runtime[symbol.upper()]
        self._reject(reason)
        runtime.working_order = None
        self._start_cooldown(runtime, now)

    def on_position_closed(
        self,
        symbol: str,
        now: datetime,
        net_pnl: float,
        *,
        exit_price: float | None = None,
        gross_pnl: float | None = None,
        fee: float = 0.0,
        slippage: float = 0.0,
        exit_reason: str = "exchange_position_closed",
    ) -> None:
        symbol = symbol.upper()
        runtime = self.runtime[symbol]
        position = runtime.position
        if not position:
            return
        assert self.risk is not None
        resolved_exit = float(exit_price if exit_price is not None else position.entry_price)
        resolved_gross = float(gross_pnl if gross_pnl is not None else net_pnl + fee + slippage)
        mfe_r = ((position.peak_price - position.entry_price) if position.side == Direction.LONG else (position.entry_price - position.trough_price)) / max(position.risk_per_unit, EPSILON)
        mae_r = ((position.entry_price - position.trough_price) if position.side == Direction.LONG else (position.peak_price - position.entry_price)) / max(position.risk_per_unit, EPSILON)
        pnl_r = position.side.value * (resolved_exit - position.entry_price) / max(position.risk_per_unit, EPSILON)
        self.closed_trades.append({
            **position.metadata,
            "event_id": position.event_id,
            "symbol": symbol,
            "side": position.side.name,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": now.isoformat(),
            "entry_price": position.entry_price,
            "exit_price": resolved_exit,
            "quantity": position.quantity,
            "entry_order_type": position.entry_order_type,
            "maker_filled": position.entry_order_type == "LIMIT_MAKER",
            "mfe_r": mfe_r,
            "mae_r": mae_r,
            "pnl_r": pnl_r,
            "gross_pnl": resolved_gross,
            "fee": fee,
            "slippage": slippage,
            "net_pnl": net_pnl,
            "exit_reason": exit_reason,
            "holding_seconds": (now - position.entry_time).total_seconds(),
        })
        self.risk.daily_realized_pnl += net_pnl
        if net_pnl < 0:
            self.risk.consecutive_losses += 1
            self.risk.direction_losses[position.side] += 1
            self.risk.symbol_losses[symbol] += 1
            if self.risk.consecutive_losses >= self.config.consecutive_loss_limit:
                self.risk.global_pause_until = now + timedelta(minutes=self.config.consecutive_loss_pause_minutes)
            if self.risk.direction_losses[position.side] >= self.config.direction_consecutive_loss_limit:
                self.risk.direction_pause_until[position.side] = now + timedelta(minutes=self.config.direction_pause_minutes)
        else:
            self.risk.consecutive_losses = 0
            self.risk.direction_losses[position.side] = 0
            self.risk.symbol_losses[symbol] = 0
        runtime.position = None
        runtime.pending = None
        runtime.working_order = None
        extra = self.config.symbol_loss_cooldown_minutes if net_pnl < 0 else 0
        runtime.cooldown_until = now + timedelta(minutes=self.config.symbol_cooldown_minutes + extra)
        runtime.state = OfasState.COOLDOWN

    def summary(self) -> dict[str, Any]:
        trades = self.closed_trades
        long_trades = [row for row in trades if row["side"] == "LONG"]
        short_trades = [row for row in trades if row["side"] == "SHORT"]
        wins = [float(row["net_pnl"]) for row in trades if float(row["net_pnl"]) > 0]
        losses = [float(row["net_pnl"]) for row in trades if float(row["net_pnl"]) < 0]
        return {
            "strategy": STRATEGY_NAME,
            "strategy_code": STRATEGY_CODE,
            "candidate_count": self.stats["candidate_count"],
            "pending_count": self.stats["pending_count"],
            "confirmed_count": self.stats["confirmed_count"],
            "entry_count": self.stats["entry_count"],
            "maker_fill_rate": self.stats["maker_filled"] / max(self.stats["maker_submitted"], 1),
            "reject_count": sum(self.reject_reasons.values()),
            "reject_reasons": dict(self.reject_reasons),
            "long_trade_count": len(long_trades),
            "short_trade_count": len(short_trades),
            "long_profit_factor": _profit_factor(long_trades),
            "short_profit_factor": _profit_factor(short_trades),
            "avg_win": statistics.mean(wins) if wins else 0.0,
            "avg_loss": statistics.mean(losses) if losses else 0.0,
            "avg_win_loss_ratio": (statistics.mean(wins) / abs(statistics.mean(losses))) if wins and losses else 0.0,
            "expectancy_per_trade": statistics.mean(float(row["net_pnl"]) for row in trades) if trades else 0.0,
            "max_drawdown": _trade_max_drawdown(trades, self.risk.starting_equity if self.risk else 0.0),
            "average_holding_seconds": statistics.mean(float(row["holding_seconds"]) for row in trades) if trades else 0.0,
            "by_symbol": _group_trade_metrics(trades, "symbol"),
            "by_hour": _group_trade_metrics(trades, "entry_hour"),
            "by_quality_bucket": _group_trade_metrics(trades, "quality_bucket"),
            "by_exit_reason": _group_trade_metrics(trades, "exit_reason"),
            "states": {symbol: runtime.state.value for symbol, runtime in self.runtime.items()},
            "trades": list(trades),
        }

    def _detect_shock(self, symbol: str, runtime: SymbolRuntime, f: OfasFeatures, context: MarketContext) -> list[OfasAction]:
        if not self.flows[symbol].baseline_ready():
            self._reject("ofas_baseline_not_ready")
            return []
        liquidity = self._liquidity_reject_reason(f)
        if liquidity:
            self._reject(liquidity)
            return []
        sell_imbalance = f.sell_imbalance_5s if self.config.sell_shock_window_seconds == 5 else f.sell_imbalance_15s
        buy_imbalance = f.buy_imbalance_5s if self.config.buy_shock_window_seconds == 5 else f.buy_imbalance_15s
        sell_return = f.return_5s if self.config.sell_shock_window_seconds == 5 else f.return_15s
        buy_return = f.return_5s if self.config.buy_shock_window_seconds == 5 else f.return_15s
        sell_shock = (
            self.config.allow_long
            and sell_imbalance >= self.config.sell_shock_imbalance_threshold
            and f.sell_volume_zscore >= self.config.sell_shock_volume_zscore
            and sell_return * 10_000 <= -self.config.sell_shock_min_return_bps
            and f.price_shock_sigma >= self.config.sell_shock_price_sigma
            and context.btc_return_5s_bps > -self.config.btc_adverse_return_5s_bps
        )
        buy_shock = (
            self.config.allow_short
            and buy_imbalance >= self.config.buy_shock_imbalance_threshold
            and f.buy_volume_zscore >= self.config.buy_shock_volume_zscore
            and buy_return * 10_000 >= self.config.buy_shock_min_return_bps
            and f.price_shock_sigma >= self.config.buy_shock_price_sigma
            and context.btc_return_5s_bps < self.config.btc_adverse_return_5s_bps
        )
        if not sell_shock and not buy_shock:
            return []
        side = Direction.LONG if sell_shock else Direction.SHORT
        if side == Direction.SHORT and self._short_reject_reason(context):
            self._reject(self._short_reject_reason(context) or "ofas_short_rejected")
            return []
        extreme = f.mid_price
        event_id = self._event_id(symbol, runtime, side, context.timestamp, extreme)
        imbalance = sell_imbalance if side == Direction.LONG else buy_imbalance
        zscore = f.sell_volume_zscore if side == Direction.LONG else f.buy_volume_zscore
        depth = f.bid_depth_l10 if side == Direction.LONG else f.ask_depth_l10
        shock_return = sell_return if side == Direction.LONG else buy_return
        shock_start_price = f.mid_price / max(1.0 + shock_return, EPSILON)
        runtime.pending = PendingAbsorption(
            event_id, symbol, side, context.timestamp, context.timestamp + timedelta(seconds=self.config.pending_expiry_seconds),
            shock_start_price, extreme, context.timestamp, shock_return * 10_000, imbalance, zscore,
            f.spread_bps, depth, depth, context.timestamp, f.realized_volatility_60s, context.btc_return_5s_bps,
        )
        runtime.state = OfasState.SELL_SHOCK_PENDING if side == Direction.LONG else OfasState.BUY_SHOCK_PENDING
        self.stats["candidate_count"] += 1
        self.stats["pending_count"] += 1
        return [OfasAction("PENDING_CREATED", symbol, side, "sell_shock" if side == Direction.LONG else "buy_shock", event_id,
                           metadata=self._candidate_metadata(runtime.pending, f))]

    def _manage_pending(self, symbol: str, runtime: SymbolRuntime, f: OfasFeatures, context: MarketContext) -> list[OfasAction]:
        pending = runtime.pending
        assert pending is not None
        now = context.timestamp
        same_depth = f.bid_depth_l10 if pending.side == Direction.LONG else f.ask_depth_l10
        if same_depth < pending.min_same_side_depth:
            pending.min_same_side_depth = same_depth
            pending.min_depth_time = now
        if pending.side == Direction.LONG and f.mid_price < pending.shock_extreme:
            pending.shock_extreme = f.mid_price
            pending.last_extreme_time = now
        elif pending.side == Direction.SHORT and f.mid_price > pending.shock_extreme:
            pending.shock_extreme = f.mid_price
            pending.last_extreme_time = now
        runtime.state = OfasState.LONG_ABSORPTION_CONFIRMING if pending.side == Direction.LONG else OfasState.SHORT_ABSORPTION_CONFIRMING
        no_extreme = (now - pending.last_extreme_time).total_seconds()
        replenishment = _replenishment(pending.initial_same_side_depth, pending.min_same_side_depth, same_depth)
        checks = self._confirmation_checks(pending, f, context, no_extreme, replenishment)
        if now >= pending.expire_time:
            runtime.pending = None
            runtime.state = OfasState.IDLE
            self._reject("ofas_pending_expired")
            return [OfasAction(
                "PENDING_CANCELLED",
                symbol,
                pending.side,
                "ofas_pending_expired",
                pending.event_id,
                metadata={
                    "failed_checks": [reason for ok, reason in checks if not ok],
                    "no_new_extreme_seconds": no_extreme,
                    "replenishment_ratio": replenishment,
                    "normalized_ofi_l5": f.normalized_ofi_l5,
                    "microprice_edge_bps": f.microprice_edge_bps,
                },
            )]
        failed_hard = next((reason for ok, reason in checks if not ok and reason in {
            "ofas_spread_rejected", "ofas_btc_adverse_move", "ofas_entry_chased", "ofas_liquidity_insufficient"
        }), None)
        if failed_hard:
            return self._cancel_pending_or_order(symbol, failed_hard)
        if not all(ok for ok, _ in checks):
            return []
        score, bucket, parts = self._quality_score(pending, f, context, no_extreme, replenishment)
        if bucket == "C":
            self._reject("ofas_quality_c")
            return self._cancel_pending_or_order(symbol, "ofas_quality_c")
        runtime.state = OfasState.LONG_ENTRY_READY if pending.side == Direction.LONG else OfasState.SHORT_ENTRY_READY
        entry = self._build_entry(symbol, pending, f, context, score, bucket, parts)
        if isinstance(entry, str):
            self._reject(entry)
            return self._cancel_pending_or_order(symbol, entry)
        runtime.working_order = entry
        runtime.state = OfasState.LONG_ORDER_WORKING if pending.side == Direction.LONG else OfasState.SHORT_ORDER_WORKING
        self.stats["confirmed_count"] += 1
        if entry.order_type == "LIMIT_MAKER":
            self.stats["maker_submitted"] += 1
        assert self.risk is not None
        self.risk.daily_trades += 1
        self.risk.last_entry_time = now
        return [entry]

    def _confirmation_checks(self, p: PendingAbsorption, f: OfasFeatures, context: MarketContext,
                             no_extreme: float, replenishment: float) -> list[tuple[bool, str]]:
        long = p.side == Direction.LONG
        rebound = (f.mid_price / p.shock_extreme - 1.0) * 10_000.0 if long else (p.shock_extreme / f.mid_price - 1.0) * 10_000.0
        return [
            (no_extreme >= (self.config.no_new_low_seconds if long else self.config.no_new_high_seconds), "ofas_no_new_extreme"),
            (replenishment >= (self.config.bid_replenishment_min if long else self.config.ask_replenishment_min), "ofas_replenishment_weak"),
            ((context.timestamp - p.min_depth_time).total_seconds() <= self.config.replenishment_window_seconds, "ofas_replenishment_too_slow"),
            ((f.normalized_ofi_l5 >= self.config.normalized_ofi_long_min) if long else (f.normalized_ofi_l5 <= self.config.normalized_ofi_short_max), "ofas_ofi_not_confirmed"),
            ((f.microprice_edge_bps >= self.config.microprice_edge_long_bps) if long else (f.microprice_edge_bps <= self.config.microprice_edge_short_bps), "ofas_microprice_not_confirmed"),
            ((f.sell_imbalance_2s <= self.config.sell_exhaustion_threshold) if long else (f.buy_imbalance_2s <= self.config.buy_exhaustion_threshold), "ofas_aggression_not_exhausted"),
            (f.sell_impact_efficiency_decreasing if long else f.buy_impact_efficiency_decreasing, "ofas_impact_not_decaying"),
            (f.spread_bps <= self.config.max_spread_bps and f.spread_vs_recent_median <= self.config.max_spread_vs_median, "ofas_spread_rejected"),
            ((context.btc_return_2s_bps > -self.config.btc_adverse_return_2s_bps) if long else (context.btc_return_2s_bps < self.config.btc_adverse_return_2s_bps), "ofas_btc_adverse_move"),
            (rebound <= (self.config.max_long_entry_rebound_bps if long else self.config.max_short_entry_pullback_bps), "ofas_entry_chased"),
            (min(f.bid_depth_l10, f.ask_depth_l10) >= self.config.min_l10_depth_usdt, "ofas_liquidity_insufficient"),
        ]

    def _build_entry(self, symbol: str, p: PendingAbsorption, f: OfasFeatures, context: MarketContext,
                     score: float, bucket: str, parts: dict[str, float]) -> OfasAction | str:
        risk_check = self._account_reject_reason(symbol, p.side, context)
        if risk_check:
            return risk_check
        long = p.side == Direction.LONG
        tick = float(self.rules[symbol].price_tick)
        volatility_bps = max(f.realized_volatility_60s * 10_000.0, self.config.min_stop_bps)
        event_range_bps = abs(p.shock_start_price / max(p.shock_extreme, EPSILON) - 1.0) * 10_000.0
        buffer_bps = max(tick / f.mid_price * 10_000.0, f.spread_bps,
                         volatility_bps * self.config.stop_short_vol_mult,
                         event_range_bps * self.config.stop_event_range_mult)
        raw_entry = (f.best_bid - tick * self.config.maker_price_offset_ticks) if long else (f.best_ask + tick * self.config.maker_price_offset_ticks)
        entry_price = float(self.rules[symbol].round_price(raw_entry))
        stop_price = p.shock_extreme * (1.0 - buffer_bps / 10_000.0 if long else 1.0 + buffer_bps / 10_000.0)
        stop_bps = abs(entry_price - stop_price) / entry_price * 10_000.0
        if stop_bps > self.config.max_stop_bps:
            return "ofas_stop_too_wide"
        if stop_bps < self.config.min_stop_bps:
            return "ofas_stop_too_tight"
        full_cost = self.config.maker_fee_bps + self.config.taker_fee_bps + f.spread_bps + self.config.estimated_slippage_bps + self.config.stop_extra_slippage_bps
        remaining_move = max(0.0, self.config.target_move_bps - abs(f.mid_price / p.shock_extreme - 1.0) * 10_000.0)
        cost_multiple = remaining_move / max(full_cost, EPSILON)
        required = self.config.taker_min_expected_move_to_cost_multiple if self.config.entry_mode == "taker-confirmation" else self.config.min_expected_move_to_cost_multiple
        if cost_multiple < required or f.spread_bps / max(self.config.target_move_bps, EPSILON) > self.config.max_spread_to_target_ratio:
            return "ofas_cost_edge_insufficient"
        risk_multiplier = 1.0 if bucket == "A" else self.config.quality_b_risk_multiplier
        if not long:
            risk_multiplier *= self.config.short_risk_multiplier
        assert self.risk is not None
        drawdown = max(0.0, (self.risk.peak_equity - context.equity) / max(self.risk.peak_equity, EPSILON))
        if drawdown >= self.config.soft_drawdown_pct:
            risk_multiplier *= self.config.soft_drawdown_risk_multiplier
        risk_pct = min(self.config.base_risk_pct * risk_multiplier, self.config.max_trade_risk_pct)
        risk_usdt = context.equity * risk_pct
        notional = min(risk_usdt / (stop_bps / 10_000.0), self.config.max_notional_usdt,
                       context.equity * self.config.leverage * self.config.max_margin_usage_pct,
                       context.equity * self.config.leverage * self.config.max_symbol_margin_pct,
                       min(f.bid_depth_l10, f.ask_depth_l10) * self.config.max_order_to_l10_depth_ratio)
        quantity = float(self.rules[symbol].round_quantity(notional / entry_price))
        if quantity < float(self.rules[symbol].min_quantity) or quantity * entry_price < float(self.rules[symbol].min_notional):
            return "ofas_below_exchange_minimum"
        tp_price = entry_price + (entry_price - stop_price) * self.config.take_profit_r if long else entry_price - (stop_price - entry_price) * self.config.take_profit_r
        order_type = "LIMIT_MAKER" if self.config.entry_mode == "maker-first" else "MARKET"
        expires = context.timestamp + timedelta(milliseconds=self.config.maker_wait_ms) if order_type == "LIMIT_MAKER" else None
        metadata = {
            **self._candidate_metadata(p, f), **parts,
            "quality_score": score, "quality_bucket": bucket,
            "expected_move_to_cost_multiple": cost_multiple, "stop_loss_bps": stop_bps,
            "take_profit_r": self.config.take_profit_r, "maker_wait_ms": self.config.maker_wait_ms,
            "confirmation_time": context.timestamp.isoformat(), "entry_hour": context.timestamp.hour,
        }
        return OfasAction("PLACE_ENTRY", symbol, p.side, "ofas_absorption_confirmed", p.event_id,
                          order_type, entry_price, quantity, stop_price, tp_price,
                          post_only=order_type == "LIMIT_MAKER", expires_at=expires, metadata=metadata)

    def _manage_working_order(self, symbol: str, runtime: SymbolRuntime, f: OfasFeatures, context: MarketContext) -> list[OfasAction]:
        order = runtime.working_order
        assert order is not None
        invalid = self._working_signal_invalid(order.side, f, context)
        expired = order.expires_at is not None and context.timestamp >= order.expires_at
        if not invalid and not expired:
            return []
        reason = invalid or "ofas_maker_timeout"
        runtime.working_order = None
        runtime.pending = None
        self._start_cooldown(runtime, context.timestamp)
        self._reject(reason)
        return [OfasAction("CANCEL_ENTRY", symbol, order.side, reason, order.event_id)]

    def _manage_position(self, symbol: str, runtime: SymbolRuntime, f: OfasFeatures, context: MarketContext) -> list[OfasAction]:
        p = runtime.position
        assert p is not None
        p.peak_price = max(p.peak_price, f.mid_price)
        p.trough_price = min(p.trough_price, f.mid_price)
        long = p.side == Direction.LONG
        current_r = p.side.value * (f.mid_price - p.entry_price) / max(p.risk_per_unit, EPSILON)
        mfe_r = ((p.peak_price - p.entry_price) if long else (p.entry_price - p.trough_price)) / max(p.risk_per_unit, EPSILON)
        reason = ""
        if (long and f.mid_price <= p.stop_price) or (not long and f.mid_price >= p.stop_price):
            reason = "ofas_event_extreme_broken"
        elif (long and f.mid_price >= p.take_profit_price) or (not long and f.mid_price <= p.take_profit_price):
            reason = "ofas_take_profit"
        elif (long and f.normalized_ofi_l5 <= -self.config.ofi_exit_threshold) or (not long and f.normalized_ofi_l5 >= self.config.ofi_exit_threshold):
            reason = "ofas_ofi_reversal"
        elif (long and context.btc_return_2s_bps <= -self.config.btc_adverse_return_2s_bps) or (not long and context.btc_return_2s_bps >= self.config.btc_adverse_return_2s_bps):
            reason = "ofas_btc_adverse_move"
        elif f.spread_bps >= self.config.spread_emergency_bps:
            reason = "ofas_spread_emergency"
        else:
            reversed_microprice = f.microprice_edge_bps <= -self.config.microprice_reversal_bps if long else f.microprice_edge_bps >= self.config.microprice_reversal_bps
            if reversed_microprice:
                p.microprice_reversal_since = p.microprice_reversal_since or context.timestamp
                if (context.timestamp - p.microprice_reversal_since).total_seconds() * 1000 >= self.config.microprice_reversal_hold_ms:
                    reason = "ofas_microprice_reversal"
            else:
                p.microprice_reversal_since = None
        held = (context.timestamp - p.entry_time).total_seconds()
        pending = runtime.pending
        if not reason and pending and held >= self.config.replenishment_window_seconds:
            same_depth = f.bid_depth_l10 if long else f.ask_depth_l10
            replenishment = _replenishment(pending.initial_same_side_depth, pending.min_same_side_depth, same_depth)
            if replenishment < self.config.replenishment_exit_min:
                reason = "ofas_replenishment_failed"
        if not reason and held >= 1.0:
            aggression = f.sell_imbalance_2s if long else f.buy_imbalance_2s
            if aggression >= (self.config.sell_shock_imbalance_threshold if long else self.config.buy_shock_imbalance_threshold):
                reason = "ofas_order_flow_invalidated"
        if not reason and held >= self.config.fail_fast_seconds and mfe_r < self.config.fail_fast_min_r:
            reason = "ofas_fail_fast"
        if not reason and held >= self.config.time_stop_seconds:
            reason = "ofas_time_stop"
        if not reason:
            return []
        runtime.state = OfasState.POSITION_EXITING
        action = self._exit_action(symbol, p, reason)
        action.metadata.update({"mfe_r": mfe_r, "pnl_r": current_r, "holding_seconds": held})
        return [action]

    def _quality_score(self, p: PendingAbsorption, f: OfasFeatures, context: MarketContext,
                       no_extreme: float, replenishment: float) -> tuple[float, str, dict[str, float]]:
        long = p.side == Direction.LONG
        shock = _clamp(35 * (p.shock_imbalance - 0.5) / 0.3 + 35 * p.shock_volume_zscore / 4 + 30 * f.price_shock_sigma / 4, 0, 100)
        absorption = _clamp(35 * replenishment + 30 * no_extreme / 5 + 35 * float(f.sell_impact_efficiency_decreasing if long else f.buy_impact_efficiency_decreasing), 0, 100)
        ofi_strength = f.normalized_ofi_l5 if long else -f.normalized_ofi_l5
        micro_strength = f.microprice_edge_bps if long else -f.microprice_edge_bps
        book = _clamp(40 * ofi_strength / 0.25 + 35 * micro_strength / 1.5 + 25 / max(f.spread_vs_recent_median, 1), 0, 100)
        btc_adverse = -context.btc_return_5s_bps if long else context.btc_return_5s_bps
        market = _clamp(100 - max(0.0, btc_adverse) * 4 - context.market_volatility_score * 20, 0, 100)
        full_cost = self.config.maker_fee_bps + self.config.taker_fee_bps + f.spread_bps + self.config.estimated_slippage_bps
        execution = _clamp(60 * self.config.target_move_bps / max(full_cost * self.config.min_expected_move_to_cost_multiple, EPSILON) + 40 * min(f.bid_depth_l10, f.ask_depth_l10) / max(self.config.min_l10_depth_usdt * 2, EPSILON), 0, 100)
        parts = {"shock_score": shock, "absorption_score": absorption, "book_score": book, "market_score": market, "execution_score": execution}
        weight_sum = sum(self.config.quality_weights.values())
        score = sum(parts[f"{name}_score"] * weight for name, weight in self.config.quality_weights.items()) / weight_sum
        bucket = "A" if score >= self.config.quality_a_min_score else "B" if score >= self.config.quality_b_min_score else "C"
        return score, bucket, parts

    def _liquidity_reject_reason(self, f: OfasFeatures) -> str | None:
        if f.spread_bps > self.config.max_spread_bps or f.spread_vs_recent_median > self.config.max_spread_vs_median or f.spread_percentile > self.config.max_spread_percentile:
            return "ofas_spread_rejected"
        if min(f.bid_depth_l10, f.ask_depth_l10) < self.config.min_l10_depth_usdt:
            return "ofas_liquidity_insufficient"
        if f.trades_per_second < self.config.min_trades_per_second:
            return "ofas_trade_frequency_insufficient"
        return None

    def _account_reject_reason(self, symbol: str, side: Direction, context: MarketContext) -> str | None:
        assert self.risk is not None
        r = self.risk
        now = context.timestamp
        if context.open_positions >= self.config.max_open_positions:
            return "ofas_max_open_positions"
        if r.daily_trades >= self.config.max_daily_trades:
            return "ofas_max_daily_trades"
        if r.last_entry_time and (now - r.last_entry_time).total_seconds() < self.config.global_min_entry_interval_seconds:
            return "ofas_global_entry_interval"
        if r.global_pause_until and now < r.global_pause_until:
            return "ofas_consecutive_loss_pause"
        if r.direction_pause_until.get(side) and now < r.direction_pause_until[side]:
            return "ofas_direction_pause"
        if r.hard_pause_until and now < r.hard_pause_until:
            return "ofas_hard_drawdown_pause"
        if r.daily_realized_pnl <= -r.day_start_equity * self.config.daily_loss_stop_pct:
            return "ofas_daily_loss_stop"
        drawdown = max(0.0, (r.peak_equity - context.equity) / max(r.peak_equity, EPSILON))
        if drawdown >= self.config.hard_drawdown_pct:
            r.hard_pause_until = now + timedelta(minutes=self.config.hard_drawdown_pause_minutes)
            return "ofas_hard_drawdown_stop"
        if r.symbol_losses[symbol] >= self.config.symbol_consecutive_loss_limit:
            return "ofas_symbol_loss_pause"
        return None

    def _data_reject_reason(self, symbol: str, context: MarketContext) -> str | None:
        runtime = self.runtime[symbol]
        book = self.books[symbol]
        if not book.valid:
            return "order_book_invalid"
        if runtime.last_trade_time is None or (context.timestamp - runtime.last_trade_time).total_seconds() * 1000 > self.config.trade_stale_ms:
            return "stale_market_data"
        if runtime.last_depth_time is None or (context.timestamp - runtime.last_depth_time).total_seconds() * 1000 > self.config.depth_stale_ms:
            return "stale_market_data"
        if context.btc_data_age_ms > self.config.btc_stale_ms:
            return "stale_btc_data"
        return None

    def _short_reject_reason(self, context: MarketContext) -> str | None:
        if context.btc_return_2s_bps >= self.config.btc_adverse_return_2s_bps or context.btc_return_5s_bps >= self.config.btc_adverse_return_5s_bps:
            return "ofas_btc_strong_short_block"
        if context.funding_rate < self.config.short_min_funding_rate:
            return "ofas_negative_funding_short_block"
        if context.recent_return_60s_bps <= -self.config.short_max_recent_drop_bps:
            return "ofas_short_chase_block"
        if context.lower_wick_ratio >= self.config.lower_wick_short_block_ratio:
            return "ofas_lower_wick_short_block"
        return None

    def _working_signal_invalid(self, side: Direction, f: OfasFeatures, context: MarketContext) -> str | None:
        if f.spread_bps > self.config.max_spread_bps:
            return "ofas_spread_rejected"
        if side == Direction.LONG and (f.normalized_ofi_l5 < 0 or f.microprice_edge_bps < 0):
            return "ofas_order_flow_invalidated"
        if side == Direction.SHORT and (f.normalized_ofi_l5 > 0 or f.microprice_edge_bps > 0):
            return "ofas_order_flow_invalidated"
        if side == Direction.LONG and context.btc_return_2s_bps <= -self.config.btc_adverse_return_2s_bps:
            return "ofas_btc_adverse_move"
        if side == Direction.SHORT and context.btc_return_2s_bps >= self.config.btc_adverse_return_2s_bps:
            return "ofas_btc_adverse_move"
        return None

    def _cancel_pending_or_order(self, symbol: str, reason: str) -> list[OfasAction]:
        runtime = self.runtime[symbol]
        actions: list[OfasAction] = []
        if runtime.working_order:
            actions.append(OfasAction("CANCEL_ENTRY", symbol, runtime.working_order.side, reason, runtime.working_order.event_id))
        elif runtime.pending:
            actions.append(OfasAction("PENDING_CANCELLED", symbol, runtime.pending.side, reason, runtime.pending.event_id))
        runtime.working_order = None
        runtime.pending = None
        if runtime.position is None:
            runtime.state = OfasState.IDLE
        self._reject(reason)
        return actions

    def _exit_action(self, symbol: str, position: OfasPosition, reason: str) -> OfasAction:
        exit_side = Direction.SHORT if position.side == Direction.LONG else Direction.LONG
        return OfasAction("EXIT_MARKET", symbol, exit_side, reason, position.event_id, "MARKET",
                          quantity=position.quantity, reduce_only=True)

    def _event_id(self, symbol: str, runtime: SymbolRuntime, side: Direction, now: datetime, price: float) -> str:
        signature = runtime.last_event_signature
        if signature:
            old_side, old_time, old_price = signature
            within_time = (now - old_time).total_seconds() <= self.config.event_merge_seconds
            within_price = abs(price / max(old_price, EPSILON) - 1.0) * 10_000 <= self.config.event_merge_price_bps
            if old_side == side and within_time and within_price and runtime.pending:
                return runtime.pending.event_id
        event_id = f"OFAS-{symbol}-{int(now.timestamp() * 1000)}-{uuid.uuid4().hex[:8]}"
        runtime.last_event_signature = (side, now, price)
        return event_id

    def _candidate_metadata(self, p: PendingAbsorption, f: OfasFeatures) -> dict[str, Any]:
        same_depth = f.bid_depth_l10 if p.side == Direction.LONG else f.ask_depth_l10
        return {
            "strategy": STRATEGY_NAME, "event_id": p.event_id, "symbol": p.symbol, "side": p.side.name,
            "shock_time": p.shock_time.isoformat(), "shock_imbalance": p.shock_imbalance,
            "shock_volume_zscore": p.shock_volume_zscore, "shock_return_bps": p.shock_return_bps,
            "no_new_extreme_seconds": max(0.0, (f.timestamp - p.last_extreme_time).total_seconds()),
            "replenishment_ratio": _replenishment(p.initial_same_side_depth, p.min_same_side_depth, same_depth),
            "normalized_ofi": f.normalized_ofi_l5, "microprice_edge_bps": f.microprice_edge_bps,
            "spread_bps": f.spread_bps, "depth_l10": same_depth,
        }

    def _ensure_risk(self, context: MarketContext) -> None:
        if self.risk is None:
            self.risk = OfasRiskState(context.equity, context.equity, day_start_equity=context.equity)

    def _start_cooldown(self, runtime: SymbolRuntime, now: datetime) -> None:
        runtime.state = OfasState.COOLDOWN
        runtime.cooldown_until = now + timedelta(minutes=self.config.symbol_cooldown_minutes)

    def _reject(self, reason: str) -> None:
        self.reject_reasons[reason] += 1


def _book_ofi(previous: Any, current: Any) -> float:
    if previous is None:
        return 0.0
    previous_bids, previous_asks = previous
    current_bids, current_asks = current
    return _side_ofi(previous_bids, current_bids, is_bid=True) + _side_ofi(previous_asks, current_asks, is_bid=False)


def _side_ofi(previous: Iterable[tuple[float, float]], current: Iterable[tuple[float, float]], is_bid: bool) -> float:
    old = dict(previous)
    new = dict(current)
    total = 0.0
    for price in set(old) | set(new):
        delta_quote = price * (new.get(price, 0.0) - old.get(price, 0.0))
        total += delta_quote if is_bid else -delta_quote
    return total


def _robust_zscore(values: list[float], current: float) -> float:
    if len(values) < 5:
        return 0.0
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    if mad <= EPSILON:
        std = statistics.pstdev(values)
        return (current - median) / max(std, EPSILON)
    return 0.67448975 * (current - median) / mad


def _percentile_rank(values: list[float], current: float) -> float:
    if not values:
        return 0.0
    return sum(value <= current for value in values) / len(values)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, EPSILON)


def _replenishment(initial: float, minimum: float, current: float) -> float:
    consumed = max(0.0, initial - minimum)
    if consumed <= EPSILON:
        return 0.0
    return _clamp((current - minimum) / consumed, 0.0, 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _profit_factor(trades: list[dict[str, Any]]) -> float:
    gross_profit = sum(max(0.0, float(row["net_pnl"])) for row in trades)
    gross_loss = abs(sum(min(0.0, float(row["net_pnl"])) for row in trades))
    return gross_profit / gross_loss if gross_loss > EPSILON else (math.inf if gross_profit > 0 else 0.0)


def _group_trade_metrics(trades: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in trades:
        value = str(row.get(key, "unknown"))
        groups.setdefault(value, []).append(row)
    return {
        value: {
            "trades": len(rows),
            "net_pnl": sum(float(row["net_pnl"]) for row in rows),
            "win_rate": sum(float(row["net_pnl"]) > 0 for row in rows) / len(rows),
            "profit_factor": _profit_factor(rows),
        }
        for value, rows in groups.items()
    }


def _trade_max_drawdown(trades: list[dict[str, Any]], starting_equity: float) -> float:
    equity = peak = max(0.0, starting_equity)
    max_drawdown = 0.0
    for row in trades:
        equity += float(row["net_pnl"])
        peak = max(peak, equity)
        if peak > EPSILON:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
    return max_drawdown


def config_as_dict(config: OfasConfig) -> dict[str, Any]:
    return asdict(config)
