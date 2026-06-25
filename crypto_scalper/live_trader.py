from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .binance_client import BinanceApiError, BinanceFuturesClient
from .data import interval_to_milliseconds
from .indicators import atr, ema, kdj, macd, rolling_high, rolling_low, rsi
from .live_config import LiveAppConfig
from .macro_events import MacroEvent, load_macro_events
from .market_filters import MultiTimeframeFilter, TimeframeSignal
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import (
    MTF_REASON_TOKEN,
    Mtf4hRsiRegimePullbackStrategy,
    funding_at,
    load_auxiliary_features,
    oi_change_at,
)
from .risk import signal_risk_weight
from .strategy import VolatilityBreakoutScalper


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class LivePosition:
    symbol: str
    position_side: str
    direction: Direction
    quantity: float
    entry_price: float
    mark_price: float
    notional: float
    unrealized_pnl: float
    leverage: int
    margin_type: str
    liquidation_price: float | None
    entry_reason: str = ""
    opened_at: datetime | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    wallet_balance: float
    available_balance: float
    initial_margin: float
    maintenance_margin: float
    total_unrealized_pnl: float
    positions: dict[str, LivePosition]
    position_rows: tuple[LivePosition, ...] = ()
    position_mode: str = "unknown"

    @property
    def margin_usage_pct(self) -> float:
        return self.initial_margin_usage_pct

    @property
    def initial_margin_usage_pct(self) -> float:
        if self.equity <= 0:
            return 1.0
        return self.initial_margin / self.equity

    @property
    def maintenance_margin_ratio_pct(self) -> float:
        if self.equity <= 0:
            return 1.0
        return self.maintenance_margin / self.equity


@dataclass
class SimPosition:
    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    max_holding_bars: int
    entry_time: datetime
    last_checked_time: datetime
    best_price: float
    leverage: int = 0
    bars_held: int = 0
    scale_ins: int = 0
    entry_reason: str = ""


@dataclass
class SessionStats:
    started_at: datetime
    starting_equity: float
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    realized_pnl: float = 0.0

    @property
    def win_rate_pct(self) -> float:
        if self.closed_trades == 0:
            return 0.0
        return self.winning_trades / self.closed_trades * 100.0


@dataclass
class ProfitState:
    direction: Direction
    entry_price: float
    best_price: float


@dataclass(frozen=True)
class EntryCandidate:
    symbol: str
    signal: Signal
    candle: Candle
    rank_score: float
    directional_momentum_pct: float
    volume_ratio: float
    filter_reason: str


@dataclass
class VbpLiveBreakoutState:
    symbol: str
    breakout_time: datetime
    breakout_level: float
    consolidation_bottom: float
    pullback_target: float
    breakout_volume: float
    breakout_close: float
    tp2_price: float
    touched_pullback: bool = False


def _portfolio_bucket_from_reason(reason: str) -> str:
    lowered = str(reason or "").lower()
    if "vbp_" in lowered or "volume_breakout_pullback" in lowered:
        return "vbp"
    if "indicator_" in lowered or "macd_golden_cross" in lowered or "macd_dead_cross" in lowered:
        return "indicator"
    return "other"


def _vbp_symbol_enabled(config: LiveAppConfig, symbol: str) -> bool:
    enabled_symbols = tuple(getattr(config.vbp_strategy, "enabled_symbols", ()) or ())
    if not enabled_symbols:
        return symbol in set(config.trading.entry_symbols or config.trading.symbols)
    return symbol in set(enabled_symbols)


def _entry_position_limit(config: LiveAppConfig) -> int:
    base_limit = max(0, int(config.trading.max_open_positions))
    if not config.trading.super_volume_extra_slot_enabled:
        return base_limit
    extra_limit = max(base_limit, int(config.trading.super_volume_extra_max_open_positions))
    return min(extra_limit, base_limit + 1)


def _candidate_requires_extra_slot(config: LiveAppConfig, projected_open_positions: int) -> bool:
    return projected_open_positions >= max(0, int(config.trading.max_open_positions))


def _super_volume_extra_slot_candidate_allowed(config: LiveAppConfig, candidate: EntryCandidate) -> bool:
    trading = config.trading
    if not trading.super_volume_extra_slot_enabled:
        return False
    if "super_volume" not in candidate.signal.reason and "startup_breakout" not in candidate.signal.reason:
        return False
    if candidate.rank_score < trading.super_volume_extra_min_rank_score:
        return False
    if candidate.directional_momentum_pct < trading.super_volume_extra_min_momentum_pct:
        return False
    if candidate.volume_ratio < trading.super_volume_extra_min_volume_ratio:
        return False
    return True


@dataclass(frozen=True)
class MarketRegime:
    weak: bool
    breadth_pct: float
    avg_return_pct: float
    reason: str


@dataclass(frozen=True)
class BtcMarketState:
    direction: Direction
    return_pct: float
    slope_atr: float
    reason: str


class BinanceAutoTrader:
    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: LogCallback | None = None,
        account_callback: Callable[[AccountSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.logger = logger or (lambda message: None)
        self.account_callback = account_callback
        self._prepared_symbols: set[tuple[str, int]] = set()
        self._unsupported_symbols: set[str] = set()
        self._unsupported_symbol_leverages: set[tuple[str, int]] = set()
        self._active_symbol_leverage: dict[str, int] = {}
        self._day = datetime.now(timezone.utc).date()
        self._day_start_equity = config.risk.starting_capital_usdt
        self._peak_equity = config.risk.starting_capital_usdt
        self._week = datetime.now(timezone.utc).isocalendar()[:2]
        self._week_start_equity = config.risk.starting_capital_usdt
        self._week_peak_equity = config.risk.starting_capital_usdt
        self._cooldown_until = 0.0
        self._sim_positions: dict[str, SimPosition] = {}
        self._scale_in_counts: dict[str, int] = {}
        self._last_scale_in_ts: dict[str, float] = {}
        self._symbol_reentry_block_until: dict[str, float] = {}
        self._known_active_symbols: set[str] = set()
        self._entry_reasons: dict[str, str] = {}
        self._position_opened_at: dict[str, datetime] = {}
        self._portfolio_symbol_cooldown_until: dict[str, float] = {}
        self._portfolio_btc_weak_cache: tuple[float, float] | None = None
        self._portfolio_week_key: tuple[int, int] | None = None
        self._portfolio_week_start_equity = config.risk.starting_capital_usdt
        self._portfolio_week_peak_equity = config.risk.starting_capital_usdt
        self._vbp_live_week_key: tuple[int, int] | None = None
        self._vbp_live_week_start_equity = config.risk.starting_capital_usdt
        self._vbp_live_week_peak_equity = config.risk.starting_capital_usdt
        self._vbp_live_loss_streak = 0
        self._vbp_live_loss_reduce_until = 0.0
        self._vbp_watch_until: dict[str, datetime] = {}
        self._vbp_breakouts: dict[str, VbpLiveBreakoutState] = {}
        self._vbp_stats: dict[str, int] = {}
        self._last_entry_scan_ts = 0.0
        self._mtf_filter = MultiTimeframeFilter(config.filters)
        self._mtf_candle_cache: dict[tuple[str, str], tuple[float, list[Candle]]] = {}
        self._market_regime_cache: tuple[float, MarketRegime] | None = None
        self._btc_market_state_cache: tuple[float, BtcMarketState] | None = None
        self._profit_states: dict[str, ProfitState] = {}
        self._macro_events = self._load_macro_events()
        self._macro_event_traded_ids: set[str] = set()
        self._macro_event_positions: dict[str, float] = {}
        self._session_peak_pnl = 0.0
        self._indicator_reversal_loss_streak = 0
        self._indicator_reversal_pause_until = 0.0
        self._indicator_reversal_side_loss_streak: dict[Direction, int] = {
            Direction.LONG: 0,
            Direction.SHORT: 0,
        }
        self._indicator_reversal_side_pause_until: dict[Direction, float] = {
            Direction.LONG: 0.0,
            Direction.SHORT: 0.0,
        }
        self._accounted_trade_ids: set[str] = set()
        self._mtf_aux_feature_cache: tuple[float, dict[str, dict[str, object]]] | None = None
        self.short_guard_stats: dict[str, int] = {}
        self.low_base_ignition_stats: dict[str, int] = {}
        self.stats = SessionStats(datetime.now(), config.risk.starting_capital_usdt)
        self._last_stats_log_ts = 0.0
        self._last_position_diagnostics_log_ts: dict[str, float] = {}

    def run_forever(self, stop_event: threading.Event) -> None:
        self.validate_startup()
        self.log("交易循环已启动")
        self._log_session_stats(self.snapshot_account(), force=True)
        while not stop_event.is_set():
            started = time.time()
            try:
                self.run_once()
            except BinanceApiError as exc:
                self.log(f"Binance API 错误: {exc}")
                backoff_seconds = _rate_limit_backoff_seconds(str(exc))
                if backoff_seconds > 0:
                    self.log(f"触发 Binance 限流，等待 {int(backoff_seconds)}s 后再继续请求")
                    stop_event.wait(backoff_seconds)
                    continue
            except Exception as exc:
                self.log(f"运行错误: {type(exc).__name__}: {exc}")

            elapsed = time.time() - started
            stop_event.wait(max(1.0, self.config.trading.poll_seconds - elapsed))
        self._log_session_stats(self.snapshot_account(), force=True)
        self.log("交易循环已停止")

    def validate_startup(self) -> None:
        trading = self.config.trading
        exchange = self.config.exchange
        if exchange.environment == "mainnet" and not trading.dry_run:
            if trading.require_mainnet_confirmation and trading.mainnet_confirmation_text != "CONFIRM_MAINNET":
                raise RuntimeError("mainnet live trading requires CONFIRM_MAINNET")

        if trading.dry_run:
            self.log("当前为 dry-run，会记录本地虚拟仓，不会发送真实订单")
            return

        dual_side = self.client.position_mode()
        if trading.require_one_way_mode and dual_side:
            raise RuntimeError("账户当前是 Hedge Mode 双向持仓；真实下单前请改成 One-way 单向持仓")

        prepared = 0
        for symbol in trading.entry_symbols or trading.symbols:
            if self._prepare_symbol(symbol):
                prepared += 1
        if prepared <= 0:
            raise RuntimeError("没有可交易币种通过杠杆/保证金模式准备")

    def run_once(self) -> None:
        account = self.snapshot_account()
        self._update_loss_limits(account)
        self.log(
            f"权益={account.equity:.2f}U 可用={account.available_balance:.2f}U "
            f"仓位占用={account.initial_margin_usage_pct * 100:.2f}% "
            f"强平保证金率={account.maintenance_margin_ratio_pct * 100:.2f}% "
            f"持仓={len(account.positions)}"
        )

        if self.config.trading.dry_run:
            self._manage_sim_positions()
            account = self.snapshot_account()

        active_symbols = set(account.positions)
        self._mark_observed_position_closures(active_symbols)
        self._cleanup_orphan_symbol_orders(active_symbols)
        for stale_symbol in set(self._scale_in_counts) - active_symbols:
            self._scale_in_counts.pop(stale_symbol, None)
            self._last_scale_in_ts.pop(stale_symbol, None)
        for stale_symbol in set(self._profit_states) - active_symbols:
            self._profit_states.pop(stale_symbol, None)
        for stale_symbol in set(self._macro_event_positions) - active_symbols:
            self._macro_event_positions.pop(stale_symbol, None)
        self._expire_symbol_reentry_blocks()
        self._known_active_symbols = set(active_symbols)

        if self.account_callback:
            self.account_callback(account)

        self._log_session_stats(account)

        macro_action = self._handle_macro_event_window(account)
        if macro_action == "flattened":
            return

        if not self._global_risk_allows_trading(account):
            return

        for symbol in self.config.trading.symbols:
            if symbol in account.positions:
                if symbol in self._macro_event_positions:
                    self._manage_macro_event_position(symbol, account.positions[symbol])
                    continue
                self._manage_existing_position(symbol, account.positions[symbol], account)

        if macro_action == "lockout":
            self._maybe_open_macro_event_position(account)
            return

        position_limit = _entry_position_limit(self.config)
        available_slots = max(0, position_limit - len(account.positions))
        cycle_slots = min(available_slots, max(1, self.config.trading.max_new_entries_per_cycle))
        if available_slots <= 0:
            self.log("已达到最大同时持仓数量，跳过新开仓扫描")
        elif not self._entry_scan_due():
            pass
        else:
            self._last_entry_scan_ts = time.time()
            entry_symbols = set(self.config.trading.entry_symbols or self.config.trading.symbols)
            candidates: list[EntryCandidate] = []
            for symbol in self.config.trading.symbols:
                if symbol in account.positions or symbol not in entry_symbols:
                    continue
                if symbol in self._unsupported_symbols:
                    continue
                if not self._symbol_reentry_allowed(symbol):
                    continue
                candidate = self._entry_candidate(symbol)
                if candidate:
                    candidates.append(candidate)
                vbp_candidate = self._vbp_entry_candidate(symbol)
                if vbp_candidate:
                    candidates.append(vbp_candidate)

            candidates.sort(key=lambda item: item.rank_score, reverse=True)
            deduped_candidates: list[EntryCandidate] = []
            seen_candidate_symbols: set[str] = set()
            for candidate in candidates:
                if candidate.symbol in seen_candidate_symbols:
                    continue
                seen_candidate_symbols.add(candidate.symbol)
                deduped_candidates.append(candidate)
            candidates = deduped_candidates
            if getattr(self.config.portfolio_control, "enabled", False):
                adjusted_candidates: list[EntryCandidate] = []
                for candidate in candidates:
                    adjusted = self._portfolio_control_filter_candidate(candidate, account)
                    if adjusted is not None:
                        adjusted_candidates.append(adjusted)
                candidates = adjusted_candidates
            if candidates:
                preview = ", ".join(
                    f"{item.symbol}:{item.rank_score:.2f}/动能{item.directional_momentum_pct * 100:+.2f}%/量{item.volume_ratio:.2f}x"
                    for item in candidates[: min(5, len(candidates))]
                )
                self.log(f"开仓候选排序: {preview}")

            opened = 0
            scan_start_positions = len(account.positions)
            for candidate in candidates:
                if opened >= cycle_slots:
                    break
                if _candidate_requires_extra_slot(self.config, scan_start_positions + opened):
                    if not _super_volume_extra_slot_candidate_allowed(self.config, candidate):
                        continue
                if self._open_entry_candidate(candidate, account):
                    opened += 1
                    if self.config.trading.dry_run:
                        account = self.snapshot_account()
                    elif opened < available_slots:
                        try:
                            account = self.snapshot_account()
                        except Exception as exc:
                            self.log(f"开仓后刷新账户失败，按本轮剩余名额继续 ({type(exc).__name__}: {exc})")

        if self.config.trading.dry_run:
            account = self.snapshot_account()
            if self.account_callback:
                self.account_callback(account)
            self._log_session_stats(account)
            self._known_active_symbols = set(account.positions)

    def _entry_scan_due(self) -> bool:
        interval = max(1, self.config.trading.entry_scan_seconds)
        return self._last_entry_scan_ts <= 0.0 or time.time() - self._last_entry_scan_ts >= interval

    def _indicator_reversal_entry_paused(self, symbol: str, signal: Signal) -> bool:
        if not _is_indicator_reversal_entry_reason(signal.reason):
            return False
        direction = signal.direction
        if direction not in (Direction.LONG, Direction.SHORT):
            return False
        if not self._indicator_reversal_side_pause_enabled(direction):
            return False
        now = time.time()
        pause_until = self._indicator_reversal_side_pause_until.get(direction, 0.0)
        if now >= pause_until:
            return False
        remaining = int(pause_until - now)
        side = "LONG" if direction == Direction.LONG else "SHORT"
        self.log(
            f"{symbol}: 指标反转{side}连亏临时暂停中，剩余约{max(1, remaining // 60)}分钟，"
            f"跳过 {side} ({signal.reason})"
        )
        return True

    def _record_indicator_reversal_result(self, net_pnl: float, entry_reason: str) -> None:
        if not _is_indicator_reversal_entry_reason(entry_reason):
            return
        direction = _indicator_reversal_direction_from_reason(entry_reason)
        if direction not in (Direction.LONG, Direction.SHORT):
            return
        if not self._indicator_reversal_side_pause_enabled(direction):
            return
        side = "LONG" if direction == Direction.LONG else "SHORT"

        if net_pnl > 0:
            if self._indicator_reversal_side_loss_streak.get(direction, 0) > 0:
                self.log(f"指标反转{side}单盈利，{side}连亏计数清零")
            self._indicator_reversal_side_loss_streak[direction] = 0
            return

        self._indicator_reversal_side_loss_streak[direction] = self._indicator_reversal_side_loss_streak.get(direction, 0) + 1
        trigger_losses = self._indicator_reversal_side_pause_losses(direction)
        if self._indicator_reversal_side_loss_streak[direction] < trigger_losses:
            self.log(f"指标反转{side}单亏损，{side}连亏={self._indicator_reversal_side_loss_streak[direction]}/{trigger_losses}")
            return

        pause_bars = self._indicator_reversal_side_pause_bars(direction)
        pause_seconds = pause_bars * max(1.0, interval_to_milliseconds(self.config.trading.timeframe) / 1000.0)
        self._indicator_reversal_side_pause_until[direction] = max(
            self._indicator_reversal_side_pause_until.get(direction, 0.0),
            time.time() + pause_seconds,
        )
        self._indicator_reversal_side_loss_streak[direction] = 0
        self.log(f"指标反转{side}连续{trigger_losses}次亏损，{side}临时暂停{pause_bars}根K线后自动恢复")

    def _indicator_reversal_side_pause_enabled(self, direction: Direction) -> bool:
        strategy = self.config.strategy
        side = "long" if direction == Direction.LONG else "short"
        side_attr = f"indicator_{side}_loss_pause_enabled"
        if hasattr(strategy, side_attr):
            return bool(getattr(strategy, side_attr))
        return bool(getattr(strategy, "indicator_reversal_loss_pause_enabled", False))

    def _indicator_reversal_side_pause_losses(self, direction: Direction) -> int:
        strategy = self.config.strategy
        side = "long" if direction == Direction.LONG else "short"
        return max(1, int(getattr(strategy, f"indicator_{side}_loss_pause_losses", getattr(strategy, "indicator_reversal_loss_pause_losses", 2))))

    def _indicator_reversal_side_pause_bars(self, direction: Direction) -> int:
        strategy = self.config.strategy
        side = "long" if direction == Direction.LONG else "short"
        return max(1, int(getattr(strategy, f"indicator_{side}_loss_pause_bars", getattr(strategy, "indicator_reversal_loss_pause_bars", 8))))

    def _mark_observed_position_closures(self, active_symbols: set[str]) -> None:
        if not self._known_active_symbols:
            return
        for symbol in self._known_active_symbols - active_symbols:
            self._scale_in_counts.pop(symbol, None)
            self._last_scale_in_ts.pop(symbol, None)
            self._profit_states.pop(symbol, None)
            if not self.config.trading.dry_run:
                self._cancel_all_symbol_orders(symbol)
                self._sync_closed_symbol_trade_pnl(symbol, "position_closed_on_exchange")
            self._mark_symbol_reentry_cooldown(symbol, "position_closed_on_exchange")

    def _cleanup_orphan_symbol_orders(self, active_symbols: set[str]) -> None:
        if self.config.trading.dry_run:
            return
        configured_symbols = {symbol.upper() for symbol in self.config.trading.symbols}
        orphan_symbols: set[str] = set()
        try:
            open_orders = self.client.open_orders()
        except BinanceApiError as exc:
            self.log(f"检查普通挂单失败，跳过残留委托清理 ({exc})")
            open_orders = []
        for order in open_orders:
            symbol = str(order.get("symbol", "")).upper()
            if symbol in configured_symbols and symbol not in active_symbols:
                orphan_symbols.add(symbol)

        try:
            open_algo_orders = self.client.open_algo_orders()
        except BinanceApiError as exc:
            self.log(f"检查条件挂单失败，跳过条件单残留清理 ({exc})")
            open_algo_orders = []
        for order in open_algo_orders:
            symbol = str(order.get("symbol", "")).upper()
            if symbol in configured_symbols and symbol not in active_symbols:
                orphan_symbols.add(symbol)

        for symbol in sorted(orphan_symbols):
            self.log(f"{symbol}: 当前无持仓但仍有残留委托，自动撤同币种委托")
            self._cancel_all_symbol_orders(symbol)
            self._mark_symbol_reentry_cooldown(symbol, "orphan_orders_cleanup")

    def _mark_symbol_reentry_cooldown(self, symbol: str, reason: str) -> None:
        cooldown = max(0, self.config.trading.symbol_reentry_cooldown_seconds)
        if cooldown <= 0:
            return
        until = time.time() + cooldown
        previous = self._symbol_reentry_block_until.get(symbol, 0.0)
        if until > previous:
            self._symbol_reentry_block_until[symbol] = until
            self.log(f"{symbol}: 平仓后进入再开仓冷却 {cooldown}s ({reason})")

    def _expire_symbol_reentry_blocks(self) -> None:
        now = time.time()
        for symbol, until in list(self._symbol_reentry_block_until.items()):
            if until <= now:
                self._symbol_reentry_block_until.pop(symbol, None)

    def _symbol_reentry_allowed(self, symbol: str) -> bool:
        until = self._symbol_reentry_block_until.get(symbol, 0.0)
        return until <= time.time()

    def _load_macro_events(self) -> tuple[MacroEvent, ...]:
        macro = self.config.macro_events
        if not macro.enabled:
            return ()
        path = Path(macro.events_path)
        try:
            events = load_macro_events(path)
        except Exception as exc:
            self.log(f"宏观事件数据加载失败，宏观事件策略不会启用: {path} ({type(exc).__name__}: {exc})")
            return ()
        allowed_types = set(macro.event_types)
        filtered = tuple(event for event in events if event.event_type in allowed_types)
        self.log(f"已加载宏观事件 {len(filtered)} 条: {path}")
        return filtered

    def _active_macro_event(self) -> MacroEvent | None:
        macro = self.config.macro_events
        if not macro.enabled or not self._macro_events:
            return None
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        pre_seconds = max(0, macro.pre_event_flatten_seconds)
        lockout_seconds = max(0, macro.post_event_lockout_seconds)
        for event in self._macro_events:
            window_start = event.timestamp - timedelta(seconds=pre_seconds)
            window_end = event.timestamp + timedelta(seconds=lockout_seconds)
            if window_start <= now <= window_end:
                return event
        return None

    def _handle_macro_event_window(self, account: AccountSnapshot) -> str:
        event = self._active_macro_event()
        if event is None:
            return "none"

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if now < event.timestamp:
            if account.positions:
                self.log(
                    f"宏观事件 {event.event_type} {event.reference} 即将公布，"
                    f"提前清空 {len(account.positions)} 个持仓"
                )
                for symbol, position in list(account.positions.items()):
                    self._exit_position(symbol, position, f"macro_event_pre_flatten_{event.event_type}")
                return "flattened"
            return "lockout"

        return "lockout"

    def _maybe_open_macro_event_position(self, account: AccountSnapshot) -> bool:
        macro = self.config.macro_events
        event = self._active_macro_event()
        if event is None:
            return False
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        entry_time = event.timestamp + timedelta(seconds=max(0, macro.post_event_entry_delay_seconds))
        if now < entry_time:
            return False

        event_id = _macro_event_id(event)
        if event_id in self._macro_event_traded_ids:
            return False

        if account.positions:
            self.log(f"宏观事件 {event.event_type}: 当前已有持仓，事件后一小时内不再新开仓")
            return False

        direction = self._macro_event_direction(event)
        if direction == Direction.FLAT:
            self._macro_event_traded_ids.add(event_id)
            self.log(
                f"宏观事件 {event.event_type} {event.reference}: 偏离不足，"
                f"actual={event.actual:g}{event.unit} forecast={event.forecast:g}{event.unit}，不开仓"
            )
            return False

        symbol = macro.primary_symbol or (macro.symbols[0] if macro.symbols else "")
        symbol = symbol.upper()
        if not symbol:
            self.log("宏观事件策略没有配置交易币种，跳过")
            return False
        if symbol not in self.config.trading.symbols:
            self.log(f"宏观事件币种 {symbol} 不在当前 symbols 中，跳过以避免仓位监控遗漏")
            return False

        candle = self._latest_macro_candle(symbol)
        if candle is None:
            self.log(f"{symbol}: 宏观事件开仓失败，无法取得最新 1m 价格")
            return False

        signal = Signal(
            direction=direction,
            confidence=0.9,
            reason=(
                f"macro_event_{event.event_type} reference={event.reference} "
                f"actual={event.actual:g}{event.unit} forecast={event.forecast:g}{event.unit} "
                f"surprise={event.surprise:+g}{event.unit}"
            ),
            stop_loss_pct=max(0.0001, macro.stop_loss_pct),
            take_profit_pct=max(0.0001, macro.take_profit_pct),
            risk_multiplier=1.0,
            max_holding_bars=1,
        )
        quantity, reason = self._size_macro_event_order(symbol, candle.close, signal, account)
        if float(quantity) <= 0:
            self.log(f"{symbol}: 宏观事件跳过开仓 ({reason})")
            return False

        self.log(
            f"{symbol}: 宏观事件开仓 {direction.name} 保证金目标={macro.margin_pct * 100:.2f}% "
            f"reason={signal.reason}"
        )
        self._enter_position(symbol, signal, candle, quantity, leverage_override=macro.leverage)
        self._macro_event_traded_ids.add(event_id)
        self._macro_event_positions[symbol] = time.time()
        return True

    def _macro_event_direction(self, event: MacroEvent) -> Direction:
        macro = self.config.macro_events
        surprise = event.surprise
        if event.event_type == "NFP":
            threshold = abs(macro.nfp_min_surprise_k)
        elif event.event_type == "CPI_YOY":
            threshold = abs(macro.cpi_min_surprise_pct)
        else:
            return Direction.FLAT
        if abs(surprise) < threshold:
            return Direction.FLAT
        return Direction.SHORT if surprise > 0 else Direction.LONG

    def _latest_macro_candle(self, symbol: str) -> Candle | None:
        for timeframe in ("1m", self.config.trading.timeframe):
            try:
                candles = self.client.klines(symbol, timeframe, 2)
            except Exception:
                continue
            if candles:
                return candles[-1]
        return None

    def _size_macro_event_order(
        self,
        symbol: str,
        price: float,
        signal: Signal,
        account: AccountSnapshot,
    ) -> tuple[str, str]:
        macro = self.config.macro_events
        if price <= 0:
            return "0", "bad_price"
        if account.available_balance < self.config.risk.min_available_balance_usdt:
            return "0", "available_balance_too_low"

        max_account_margin = account.equity * self.config.risk.max_account_margin_usage_pct
        remaining_margin = max_account_margin - account.initial_margin
        desired_margin = account.equity * max(0.0, macro.margin_pct)
        usable_margin = min(desired_margin, remaining_margin, account.available_balance)
        if usable_margin <= 0:
            return "0", "margin_usage_limit"

        notional = usable_margin * max(1, macro.leverage)
        if self.config.risk.max_position_notional_usdt > 0:
            notional = min(notional, self.config.risk.max_position_notional_usdt)
        if notional < self.config.risk.min_order_notional_usdt:
            return "0", "below_min_notional"

        rules = self.client.symbol_rules(symbol)
        quantity = rules.round_quantity(notional / price)
        rounded_notional = float(quantity) * price
        min_required = max(float(rules.min_notional), self.config.risk.min_order_notional_usdt)
        if DecimalCompat.less_than(quantity, rules.min_quantity):
            return "0", "below_min_quantity"
        if rounded_notional < min_required:
            return "0", "below_exchange_min_notional"
        return quantity, "ok"

    def _manage_macro_event_position(self, symbol: str, position: LivePosition) -> None:
        opened_at = self._macro_event_positions.get(symbol)
        if opened_at is None:
            return
        max_holding = max(1, self.config.macro_events.max_holding_seconds)
        if time.time() - opened_at >= max_holding:
            self.log(f"{symbol}: 宏观事件仓达到最长持仓 {max_holding}s，准备平仓")
            self._exit_position(symbol, position, "macro_event_time_stop")

    def snapshot_account(self) -> AccountSnapshot:
        if self.config.trading.dry_run:
            return self._sim_snapshot()

        payload = self.client.account()
        equity = float(payload.get("totalMarginBalance", payload.get("totalWalletBalance", 0.0)))
        wallet = float(payload.get("totalWalletBalance", equity))
        available = float(payload.get("availableBalance", 0.0))
        initial_margin = float(payload.get("totalInitialMargin", 0.0))
        maintenance_margin = float(payload.get("totalMaintMargin", 0.0))
        total_unrealized = float(payload.get("totalUnrealizedProfit", 0.0))
        position_mode = "Hedge Mode 双向" if self.client.position_mode() else "One-way 单向"
        positions: dict[str, LivePosition] = {}
        position_rows: list[LivePosition] = []
        for raw in payload.get("positions", []):
            symbol = str(raw.get("symbol", "")).upper()
            if symbol not in self.config.trading.symbols:
                continue
            amount = float(raw.get("positionAmt", 0.0))
            if abs(amount) <= 0:
                continue
            entry = float(raw.get("entryPrice", 0.0))
            mark = float(raw.get("markPrice", 0.0) or raw.get("breakEvenPrice", 0.0) or entry)
            unrealized = float(raw.get("unrealizedProfit", 0.0))
            notional = abs(float(raw.get("notional", 0.0)) or amount * entry)
            position = LivePosition(
                symbol=symbol,
                position_side=str(raw.get("positionSide", "BOTH")),
                direction=Direction.LONG if amount > 0 else Direction.SHORT,
                quantity=abs(amount),
                entry_price=entry,
                mark_price=mark,
                notional=notional,
                unrealized_pnl=unrealized,
                leverage=int(float(raw.get("leverage", self.config.trading.leverage))),
                margin_type=str(raw.get("marginType", self.config.trading.margin_type)),
                liquidation_price=_optional_float(raw.get("liquidationPrice")),
                entry_reason=self._entry_reasons.get(symbol, ""),
                opened_at=self._position_opened_at.get(symbol),
            )
            positions.setdefault(symbol, position)
            position_rows.append(position)
        return AccountSnapshot(
            equity=equity,
            wallet_balance=wallet,
            available_balance=available,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            total_unrealized_pnl=total_unrealized,
            positions=positions,
            position_rows=tuple(position_rows),
            position_mode=position_mode,
        )

    def _sim_snapshot(self) -> AccountSnapshot:
        rows: list[LivePosition] = []
        positions: dict[str, LivePosition] = {}
        total_unrealized = 0.0
        initial_margin = 0.0
        maintenance_margin = 0.0
        for sim in self._sim_positions.values():
            mark = self._latest_close(sim.symbol)
            unrealized = sim.direction.value * sim.quantity * (mark - sim.entry_price)
            notional = abs(sim.quantity * mark)
            leverage = sim.leverage or self.config.trading.leverage
            margin = notional / max(leverage, 1)
            total_unrealized += unrealized
            initial_margin += margin
            maintenance_margin += notional * 0.005
            live = LivePosition(
                symbol=sim.symbol,
                position_side="SIM",
                direction=sim.direction,
                quantity=sim.quantity,
                entry_price=sim.entry_price,
                mark_price=mark,
                notional=notional,
                unrealized_pnl=unrealized,
                leverage=leverage,
                margin_type=self.config.trading.margin_type,
                liquidation_price=None,
                entry_reason=sim.entry_reason,
                opened_at=sim.entry_time,
            )
            rows.append(live)
            positions[sim.symbol] = live

        wallet = self.config.risk.starting_capital_usdt + self.stats.realized_pnl
        equity = wallet + total_unrealized
        available = max(0.0, equity - initial_margin)
        return AccountSnapshot(
            equity=equity,
            wallet_balance=wallet,
            available_balance=available,
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            total_unrealized_pnl=total_unrealized,
            positions=positions,
            position_rows=tuple(rows),
            position_mode="dry-run 模拟仓",
        )

    def _entry_candidate(self, symbol: str) -> EntryCandidate | None:
        if getattr(self.config.strategy, "low_base_volume_ignition_enabled", False):
            candidate = self._low_base_volume_ignition_entry_candidate(symbol)
            if candidate is not None or getattr(self.config.strategy, "low_base_volume_ignition_disable_legacy", False):
                return candidate

        if getattr(self.config.strategy, "mtf_4h_rsi_regime_enabled", False):
            candidate = self._mtf_4h_rsi_regime_entry_candidate(symbol)
            if candidate is not None or getattr(self.config.strategy, "mtf_disable_legacy_strategies", False):
                return candidate

        candles = self._closed_candles(symbol)
        if len(candles) < VolatilityBreakoutScalper(self.config.strategy).warmup_bars:
            self.log(f"{symbol}: K 线不足，等待")
            return None

        strategy = VolatilityBreakoutScalper(self.config.strategy)
        strategy.prepare(candles)
        signal = strategy.signal(len(candles) - 1, candles)
        if signal.direction == Direction.FLAT:
            indicator_signal = self._indicator_reversal_signal(candles)
            if indicator_signal.direction == Direction.FLAT:
                fast_candidate = self._fast_breakout_entry_candidate(symbol)
                if fast_candidate:
                    return fast_candidate
                self.log(f"{symbol}: 无开仓信号 ({signal.reason}; {indicator_signal.reason})")
                return None
            else:
                signal = indicator_signal

        if self._indicator_reversal_entry_paused(symbol, signal):
            return None

        reference_guard_reason = self._indicator_reference_guard_reason(symbol, signal)
        if reference_guard_reason:
            self.log(f"{symbol}: 1h 指标反转过滤拒绝 {signal.direction.name} ({reference_guard_reason})")
            return None
        short_guard_reason = self._indicator_short_guard_reason(symbol, signal, candles)
        if short_guard_reason:
            self._record_short_guard_reject(short_guard_reason)
            self.log(f"{symbol}: 指标空头专用过滤拒绝 SHORT ({short_guard_reason})")
            return None

        allowed, filter_reason = self._passes_multi_timeframe_filter(symbol, signal.direction)
        if not allowed:
            self.log(f"{symbol}: 多周期过滤拒绝 {signal.direction.name} ({filter_reason})")
            return None
        self.log(f"{symbol}: 多周期过滤通过 {signal.direction.name} ({filter_reason})")
        rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._weak_market_adjusted_signal(signal, rank_score)
        if adjusted_signal is None:
            self.log(f"{symbol}: 弱势行情多头过滤拒绝 ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._strong_market_adjusted_signal(signal, rank_score)
        if adjusted_signal is None:
            self.log(f"{symbol}: 强多头行情空头过滤拒绝 ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._btc_market_adjusted_signal(signal, rank_score, momentum_pct, volume_ratio)
        if adjusted_signal is None:
            self.log(f"{symbol}: BTC大盘方向过滤拒绝 {signal.direction.name} ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._trend_reference_adjusted_signal(symbol, signal)
        if adjusted_signal is None:
            self.log(f"{symbol}: 1h趋势方向保护拒绝 {signal.direction.name} ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = _ordinary_breakout_adjusted_signal(self.config, signal, rank_score)
        if adjusted_signal is None:
            self.log(f"{symbol}: 普通突破质量过滤拒绝 {signal.direction.name} ({signal.reason}, rank={rank_score:.2f})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        candidate = EntryCandidate(symbol, signal, candles[-1], rank_score, momentum_pct, volume_ratio, filter_reason)
        quality_guard_reason = _entry_quality_guard_reason(self.config, candidate)
        if quality_guard_reason:
            self.log(f"{symbol}: 开仓质量过滤拒绝 {signal.direction.name} ({quality_guard_reason})")
            return None
        return candidate

    def _mtf_4h_rsi_regime_entry_candidate(self, symbol: str) -> EntryCandidate | None:
        strategy_config = self.config.strategy
        if not getattr(strategy_config, "mtf_4h_rsi_regime_enabled", False):
            return None
        if not _mtf_symbol_allowed(self.config, symbol):
            return None
        regime_timeframe = _valid_mtf_timeframe(getattr(strategy_config, "mtf_regime_timeframe", "4h"), "4h")
        trigger_timeframe = _valid_mtf_timeframe(getattr(strategy_config, "mtf_trigger_timeframe", "15m"), "15m")
        try:
            candles_trigger = self._closed_candles_for_timeframe(symbol, trigger_timeframe, 220)
            candles_30m = self._closed_candles_for_timeframe(symbol, "30m", 160)
            candles_1h = self._closed_candles_for_timeframe(symbol, "1h", 160)
            candles_regime = self._closed_candles_for_timeframe(symbol, regime_timeframe, 180)
            btc_1h = self._closed_candles_for_timeframe("BTCUSDT", "1h", 80)
            btc_4h = self._closed_candles_for_timeframe("BTCUSDT", "4h", 80)
        except Exception as exc:
            self.log(f"{symbol}: MTF策略数据不可用 ({exc})")
            return None
        if not candles_trigger or not candles_30m or not candles_1h or not candles_regime or not btc_1h or not btc_4h:
            return None

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
        aux = self._mtf_aux_features().get(symbol, {})
        mtf_strategy = Mtf4hRsiRegimePullbackStrategy(self.config)
        decision = mtf_strategy.build_signal(
            symbol,
            candles_trigger,
            candles_30m,
            candles_1h,
            candles_regime,
            btc_1h,
            btc_4h,
            oi_change_at(aux, timestamp),
            funding_at(aux, timestamp, float(getattr(self.config.risk, "funding_default_rate", 0.0))),
            candles_regime=candles_regime,
            candles_trigger=candles_trigger,
            regime_timeframe=regime_timeframe,
            trigger_timeframe=trigger_timeframe,
        )
        if (
            decision.signal is None
            and getattr(strategy_config, "mtf_secondary_2h_enabled", False)
            and regime_timeframe != "2h"
        ):
            try:
                candles_secondary_regime = self._closed_candles_for_timeframe(symbol, "2h", 180)
            except Exception:
                candles_secondary_regime = []
            if candles_secondary_regime:
                decision = mtf_strategy.build_signal(
                    symbol,
                    candles_trigger,
                    candles_30m,
                    candles_1h,
                    candles_secondary_regime,
                    btc_1h,
                    btc_4h,
                    oi_change_at(aux, timestamp),
                    funding_at(aux, timestamp, float(getattr(self.config.risk, "funding_default_rate", 0.0))),
                    candles_regime=candles_secondary_regime,
                    candles_trigger=candles_trigger,
                    regime_timeframe="2h",
                    trigger_timeframe=trigger_timeframe,
                )
        if decision.signal is None or decision.candle is None:
            return None
        rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(decision.signal, decision.rank_candles)
        candidate = EntryCandidate(symbol, decision.signal, decision.candle, rank_score, momentum_pct, volume_ratio, "mtf_4h_rsi_regime")
        self.log(
            f"{symbol}: MTF {regime_timeframe}/{trigger_timeframe} RSI候选 {decision.signal.direction.name} rank={rank_score:.2f} "
            f"动能={momentum_pct * 100:+.2f}% reason={decision.signal.reason}"
        )
        return candidate

    def _mtf_aux_features(self) -> dict[str, dict[str, object]]:
        now = time.time()
        cached = self._mtf_aux_feature_cache
        if cached and now - cached[0] < 600.0:
            return cached[1]
        features = load_auxiliary_features(
            tuple(self.config.trading.symbols),
            str(getattr(self.config.strategy, "mtf_oi_data_dir", "data/binance_oi_taker_5m")),
            str(getattr(self.config.strategy, "mtf_funding_data_dir", "data/binance_oi_flush_funding")),
        )
        self._mtf_aux_feature_cache = (now, features)
        return features

    def _fast_breakout_entry_candidate(self, symbol: str) -> EntryCandidate | None:
        strategy = self.config.strategy
        if not getattr(strategy, "fast_breakout_enabled", False):
            return None

        timeframe = str(getattr(strategy, "fast_breakout_timeframe", "5m"))
        limit = max(
            80,
            int(getattr(strategy, "fast_breakout_channel_period", 18))
            + int(getattr(strategy, "fast_breakout_volume_period", 24))
            + self.config.strategy.atr_period
            + 10,
        )
        try:
            candles = self._closed_candles_for_timeframe(symbol, timeframe, limit)
        except Exception as exc:
            self.log(f"{symbol}: 5m早期突破数据不可用 ({exc})")
            return None
        if not candles:
            return None

        signal = _fast_breakout_signal_for_candles(self.config, candles)
        if signal.direction == Direction.FLAT:
            return None

        allowed, filter_reason = self._passes_multi_timeframe_filter(symbol, signal.direction)
        if not allowed:
            self.log(f"{symbol}: 5m早期突破多周期过滤拒绝 {signal.direction.name} ({filter_reason})")
            return None
        rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._weak_market_adjusted_signal(signal, rank_score)
        if adjusted_signal is None:
            self.log(f"{symbol}: 5m早期突破弱势行情过滤拒绝 ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._strong_market_adjusted_signal(signal, rank_score)
        if adjusted_signal is None:
            self.log(f"{symbol}: 5m早期突破强多头行情空头过滤拒绝 ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._btc_market_adjusted_signal(signal, rank_score, momentum_pct, volume_ratio)
        if adjusted_signal is None:
            self.log(f"{symbol}: 5m早期突破BTC大盘方向过滤拒绝 {signal.direction.name} ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        adjusted_signal = self._trend_reference_adjusted_signal(symbol, signal)
        if adjusted_signal is None:
            self.log(f"{symbol}: 5m早期突破1h趋势方向保护拒绝 {signal.direction.name} ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)

        candidate = EntryCandidate(symbol, signal, candles[-1], rank_score, momentum_pct, volume_ratio, filter_reason)
        quality_guard_reason = _entry_quality_guard_reason(self.config, candidate)
        if quality_guard_reason:
            self.log(f"{symbol}: 5m早期突破开仓质量过滤拒绝 {signal.direction.name} ({quality_guard_reason})")
            return None
        self.log(
            f"{symbol}: 5m早期突破候选 {signal.direction.name} rank={rank_score:.2f} "
            f"动能={momentum_pct * 100:+.2f}% 量能={volume_ratio:.2f}x ({signal.reason})"
        )
        return candidate

    def _vbp_entry_candidate(self, symbol: str) -> EntryCandidate | None:
        vbp = self.config.vbp_strategy
        if not getattr(vbp, "enabled", False):
            return None
        if not _vbp_symbol_enabled(self.config, symbol):
            return None
        limit = max(
            180,
            int(vbp.structure_filter.consolidation_bars)
            + int(vbp.universe.rank_window_minutes) * 3
            + int(vbp.entry.timeout_bars)
            + 20,
        )
        try:
            candles = self._closed_candles_for_timeframe(symbol, "1m", limit)
        except Exception as exc:
            self._record_vbp_stat("reject_live_1m_unavailable")
            self.log(f"{symbol}: VBP 1m数据不可用 ({exc})")
            return None
        if len(candles) < int(vbp.structure_filter.consolidation_bars) + 65:
            self._record_vbp_stat("reject_live_warmup")
            return None
        now = candles[-1].timestamp
        pending = self._vbp_breakouts.get(symbol)
        if pending is not None:
            return self._vbp_pending_candidate(symbol, candles, pending)

        rvol_1h = self._vbp_live_rvol_1h(symbol, candles)
        if rvol_1h >= float(vbp.universe.rvol_entry_threshold):
            self._vbp_watch_until[symbol] = now + timedelta(minutes=max(1, int(vbp.universe.watchlist_ttl_minutes)))
            self._record_vbp_stat("watchlist_added")
            self.log(f"{symbol}: VBP加入观察池 rvol_1h={rvol_1h:.2f}x")
        watch_until = self._vbp_watch_until.get(symbol)
        if watch_until is None or now > watch_until:
            return None

        structure = self._vbp_live_consolidation(candles)
        if structure is None:
            self._record_vbp_stat("reject_not_consolidating_near_top")
            return None
        zone_top, zone_bottom = structure
        if not self._vbp_live_daily_high_ok(symbol, candles[-1].close):
            self._record_vbp_stat("reject_daily_high_zone")
            self.log(f"{symbol}: VBP拒绝 daily_high_zone")
            return None
        funding_rate = float(getattr(self.config.risk, "funding_default_rate", 0.0))
        if funding_rate > float(vbp.structure_filter.funding_rate_max):
            self._record_vbp_stat("reject_funding_rate")
            return None

        candle = candles[-1]
        previous_close = candles[-2].close
        candle_rvol = self._vbp_live_rvol_1m(candles)
        if candle_rvol < float(vbp.universe.rvol_trigger_threshold):
            self._record_vbp_stat("reject_no_breakout_rvol")
            return None
        if not (previous_close <= zone_top and candle.close > zone_top):
            self._record_vbp_stat("reject_no_zone_breakout")
            return None

        pullback_target = zone_top
        if bool(vbp.entry.use_vwap_as_pullback_target):
            pullback_target = max(zone_top, self._vbp_live_vwap(candles[-int(vbp.structure_filter.consolidation_bars):]))
        tp2_price = self._vbp_live_tp2_price(candles, candle.close, zone_bottom)
        self._vbp_breakouts[symbol] = VbpLiveBreakoutState(
            symbol=symbol,
            breakout_time=now,
            breakout_level=zone_top,
            consolidation_bottom=zone_bottom,
            pullback_target=pullback_target,
            breakout_volume=candle.volume,
            breakout_close=candle.close,
            tp2_price=tp2_price,
        )
        self._record_vbp_stat("breakout_detected")
        self.log(
            f"{symbol}: VBP突破确认 level={zone_top:.6g} bottom={zone_bottom:.6g} "
            f"target={pullback_target:.6g} rvol_1m={candle_rvol:.2f}x rvol_1h={rvol_1h:.2f}x"
        )
        return None

    def _vbp_pending_candidate(
        self,
        symbol: str,
        candles: list[Candle],
        pending: VbpLiveBreakoutState,
    ) -> EntryCandidate | None:
        vbp = self.config.vbp_strategy
        candle = candles[-1]
        age_bars = max(0, int((candle.timestamp - pending.breakout_time).total_seconds() // 60))
        if age_bars <= 0:
            return None
        if age_bars > int(vbp.entry.timeout_bars):
            self._vbp_breakouts.pop(symbol, None)
            self._record_vbp_stat("pending_timeout")
            self.log(f"{symbol}: VBP pending超时 age={age_bars}")
            return None
        if candle.close < pending.consolidation_bottom:
            self._vbp_breakouts.pop(symbol, None)
            self._record_vbp_stat("pending_failed_back_inside")
            self.log(f"{symbol}: VBP失败 跌回整理区 bottom={pending.consolidation_bottom:.6g}")
            return None

        touched_target = candle.low <= pending.pullback_target <= candle.high or candle.low <= pending.pullback_target
        pullback_volume_ok = candle.volume <= pending.breakout_volume * float(vbp.entry.pullback_volume_ratio)
        if touched_target and not pullback_volume_ok:
            self._vbp_breakouts.pop(symbol, None)
            self._record_vbp_stat("pending_reject_high_volume_pullback")
            self.log(
                f"{symbol}: VBP拒绝 放量回踩 vol={candle.volume:.4g} "
                f"breakout_vol={pending.breakout_volume:.4g}"
            )
            return None
        if touched_target:
            pending.touched_pullback = True
            self._record_vbp_stat("pullback_touched")
        if not pending.touched_pullback:
            return None
        if not (candle.close > candle.open and candle.close >= pending.pullback_target):
            self._record_vbp_stat("pending_wait_bull_reclaim")
            return None

        entry_price = candle.close
        stop_price = max(pending.consolidation_bottom, entry_price * (1.0 - float(vbp.exit.stop_loss_pct)))
        stop_loss_pct = (entry_price - stop_price) / max(entry_price, 1e-12)
        if stop_loss_pct <= 0:
            self._vbp_breakouts.pop(symbol, None)
            self._record_vbp_stat("reject_invalid_stop")
            return None
        risk = entry_price - stop_price
        tp1_price = entry_price + risk * float(vbp.exit.tp1_rr_ratio)
        tp2_price = max(pending.tp2_price, tp1_price + risk)
        take_profit_pct = max((tp2_price - entry_price) / max(entry_price, 1e-12), stop_loss_pct * float(vbp.exit.tp1_rr_ratio))
        reason = (
            "vbp_volume_breakout_pullback "
            f"level={pending.breakout_level:.8g} bottom={pending.consolidation_bottom:.8g} "
            f"target={pending.pullback_target:.8g} "
            f"tp1={tp1_price:.8g} tp1_ratio={float(vbp.exit.tp1_close_ratio):.3f} stop={stop_loss_pct * 100:.3f}%"
        )
        signal = Signal(
            Direction.LONG,
            0.75,
            reason,
            stop_loss_pct,
            take_profit_pct,
            risk_multiplier=float(vbp.position.size_multiplier),
            max_holding_bars=max(1, int(vbp.entry.timeout_bars) * 4),
        )
        momentum_pct = candle.close / max(pending.breakout_close, 1e-12) - 1.0
        volume_ratio = candle.volume / max(pending.breakout_volume, 1e-12)
        rank_score = 5.0 + max(0.0, momentum_pct * 100.0) + max(0.0, 1.0 - volume_ratio)
        self._vbp_breakouts.pop(symbol, None)
        self._record_vbp_stat("entry_count")
        self.log(
            f"{symbol}: VBP入场确认 age={age_bars} pullback_target={pending.pullback_target:.6g} "
            f"entry={entry_price:.6g} stop={stop_loss_pct * 100:.2f}%"
        )
        return EntryCandidate(symbol, signal, candle, rank_score, momentum_pct, volume_ratio, "vbp_volume_breakout_pullback")

    def _vbp_live_consolidation(self, candles: list[Candle]) -> tuple[float, float] | None:
        vbp = self.config.vbp_strategy
        n_bars = max(2, int(vbp.structure_filter.consolidation_bars))
        if len(candles) <= n_bars:
            return None
        window = candles[-n_bars - 1:-1]
        top = max(item.high for item in window)
        bottom = min(item.low for item in window)
        close = candles[-1].close
        if top <= bottom or close <= 0:
            return None
        threshold = float(vbp.structure_filter.consolidation_threshold_pct)
        near_top = (top - close) / close <= threshold
        inside = bottom <= close <= top * (1.0 + threshold)
        if not (inside and near_top):
            return None
        return top, bottom

    def _vbp_live_daily_high_ok(self, symbol: str, close: float) -> bool:
        vbp = self.config.vbp_strategy
        try:
            daily = self._closed_candles_for_timeframe(symbol, "1d", max(10, int(vbp.structure_filter.daily_high_lookback_days) + 5))
        except Exception:
            self._record_vbp_stat("reject_daily_data_unavailable")
            return False
        lookback = max(7, int(vbp.structure_filter.daily_high_lookback_days))
        if len(daily) < min(lookback, 7):
            return False
        window = daily[-lookback:]
        high = max(item.high for item in window)
        return close < high * float(vbp.structure_filter.daily_high_zone_pct)

    def _vbp_live_rvol_1m(self, candles: list[Candle]) -> float:
        window = candles[-61:-1]
        if not window:
            return 0.0
        avg = sum(item.volume for item in window) / len(window)
        return candles[-1].volume / max(avg, 1e-12)

    def _vbp_live_rvol_1h(self, symbol: str, candles_1m: list[Candle]) -> float:
        vbp = self.config.vbp_strategy
        current_volume = sum(max(item.volume, 0.0) * max(item.close, 0.0) for item in candles_1m[-60:])
        try:
            hourly = self._closed_candles_for_timeframe(symbol, "1h", max(30, int(vbp.universe.rvol_lookback_days) * 24 + 2))
        except Exception:
            hourly = []
        if len(hourly) < 24:
            return 0.0
        quote_volumes = [max(item.volume, 0.0) * max(item.close, 0.0) for item in hourly[:-1]]
        if not quote_volumes:
            return 0.0
        average = sum(quote_volumes[-int(vbp.universe.rvol_lookback_days) * 24:]) / max(1, min(len(quote_volumes), int(vbp.universe.rvol_lookback_days) * 24))
        return current_volume / max(average, 1e-12)

    def _vbp_live_vwap(self, candles: list[Candle]) -> float:
        volume = sum(max(item.volume, 0.0) for item in candles)
        if volume <= 0:
            return candles[-1].close if candles else 0.0
        return sum(((item.high + item.low + item.close) / 3.0) * max(item.volume, 0.0) for item in candles) / volume

    def _vbp_live_tp2_price(self, candles: list[Candle], entry_price: float, stop_price: float) -> float:
        risk = max(entry_price - stop_price, entry_price * 0.005)
        recent_high = max((item.high for item in candles), default=entry_price)
        tp1_rr = float(self.config.vbp_strategy.exit.tp1_rr_ratio)
        if recent_high > entry_price + risk * tp1_rr:
            return recent_high
        return entry_price + risk * max(tp1_rr * 1.5, 2.0)

    def _record_vbp_stat(self, key: str) -> None:
        self._vbp_stats[key] = self._vbp_stats.get(key, 0) + 1

    def _open_entry_candidate(self, candidate: EntryCandidate, account: AccountSnapshot) -> bool:
        adjusted_candidate = self._portfolio_control_filter_candidate(candidate, account)
        if adjusted_candidate is None:
            return False
        candidate = adjusted_candidate
        chase_guard_reason = self._super_volume_chase_guard_reason(candidate)
        if chase_guard_reason:
            self.log(f"{candidate.symbol}: 跳过强突破追价开仓 ({chase_guard_reason})")
            return False
        timing_guard_reason = self._entry_timing_guard_reason(candidate)
        if timing_guard_reason:
            self.log(f"{candidate.symbol}: 跳过15m进场过滤 ({timing_guard_reason})")
            return False
        execution_guard_reason = self._entry_execution_guard_reason(candidate)
        if execution_guard_reason:
            self.log(f"{candidate.symbol}: 跳过5m执行确认 ({execution_guard_reason})")
            return False

        execution_price = self._entry_execution_reference_price(candidate)
        execution_candle = replace(
            candidate.candle,
            high=max(candidate.candle.high, execution_price),
            low=min(candidate.candle.low, execution_price),
            close=execution_price,
        )
        quantity, reason = self._size_order(candidate.symbol, execution_price, candidate.signal, account)
        if float(quantity) <= 0:
            self.log(f"{candidate.symbol}: 跳过开仓 ({reason})")
            return False

        self.log(
            f"{candidate.symbol}: 选择开仓候选 rank={candidate.rank_score:.2f} "
            f"动能={candidate.directional_momentum_pct * 100:+.2f}% 量能={candidate.volume_ratio:.2f}x "
            f"参考价={execution_price:.6g}"
        )
        self._enter_position(candidate.symbol, candidate.signal, execution_candle, quantity)
        return True

    def _portfolio_control_filter_candidate(self, candidate: EntryCandidate, account: AccountSnapshot) -> EntryCandidate | None:
        control = self.config.portfolio_control
        if not getattr(control, "enabled", False):
            return candidate
        now = time.time()
        cooldown_until = self._portfolio_symbol_cooldown_until.get(candidate.symbol)
        if cooldown_until is not None and now < cooldown_until:
            self.log(f"{candidate.symbol}: 组合风控拒绝 symbol_cooldown")
            return None
        if bool(control.prevent_same_symbol_overlap) and candidate.symbol in account.positions:
            self.log(f"{candidate.symbol}: 组合风控拒绝 same_symbol_overlap")
            return None
        max_open = max(0, int(control.max_open_positions))
        if max_open > 0 and len(account.positions) >= max_open:
            self.log(f"{candidate.symbol}: 组合风控拒绝 max_open_positions")
            return None
        bucket = _portfolio_bucket_from_reason(candidate.signal.reason)
        vbp_week_max_positions = max(0, int(control.max_vbp_positions))
        vbp_week_multiplier = 1.0
        if bucket == "vbp":
            vbp_week_max_positions, vbp_week_multiplier, vbp_week_reason = self._vbp_live_weekly_limits(account)
            if vbp_week_max_positions <= 0 or vbp_week_multiplier <= 0:
                self.log(f"{candidate.symbol}: 组合风控拒绝 {vbp_week_reason}")
                return None
        if bucket == "vbp" and self._portfolio_live_bucket_count(account, "vbp") >= max(0, min(int(control.max_vbp_positions), vbp_week_max_positions)):
            self.log(f"{candidate.symbol}: 组合风控拒绝 max_vbp_positions")
            return None
        if bucket == "indicator" and self._portfolio_live_bucket_count(account, "indicator") >= max(0, int(control.max_indicator_positions)):
            self.log(f"{candidate.symbol}: 组合风控拒绝 max_indicator_positions")
            return None
        if candidate.symbol not in {"BTCUSDT", "ETHUSDT"}:
            max_alt = max(0, int(control.max_altcoin_positions))
            if max_alt > 0 and self._portfolio_live_altcoin_count(account) >= max_alt:
                self.log(f"{candidate.symbol}: 组合风控拒绝 max_altcoin_positions")
                return None

        multiplier = 1.0
        if bucket == "vbp":
            multiplier *= max(0.0, float(control.vbp_risk_multiplier))
            multiplier *= vbp_week_multiplier
            multiplier *= self._portfolio_live_btc_weak_multiplier()
        elif bucket == "indicator":
            multiplier *= max(0.0, float(control.indicator_risk_multiplier))
        weekly_multiplier, weekly_reason = self._portfolio_live_weekly_multiplier(account)
        if weekly_multiplier <= 0:
            self.log(f"{candidate.symbol}: 组合风控拒绝 {weekly_reason}")
            return None
        multiplier *= weekly_multiplier
        if multiplier <= 0:
            self.log(f"{candidate.symbol}: 组合风控拒绝 zero_risk")
            return None
        if multiplier < 0.999:
            adjusted_signal = replace(candidate.signal, risk_multiplier=candidate.signal.risk_multiplier * multiplier)
            return replace(candidate, signal=adjusted_signal)
        return candidate

    def _portfolio_live_weekly_multiplier(self, account: AccountSnapshot) -> tuple[float, str]:
        control = self.config.portfolio_control
        if not getattr(control, "enabled", False) or not getattr(control, "weekly_drawdown_control_enabled", False):
            return 1.0, "portfolio_weekly_disabled"
        now = datetime.now(timezone.utc)
        key = now.isocalendar()[:2]
        if self._portfolio_week_key != key:
            self._portfolio_week_key = key
            self._portfolio_week_start_equity = account.equity
            self._portfolio_week_peak_equity = account.equity
        self._portfolio_week_peak_equity = max(self._portfolio_week_peak_equity, account.equity)
        drawdown = max(0.0, 1.0 - account.equity / max(self._portfolio_week_peak_equity, 1e-12))
        loss = max(0.0, 1.0 - account.equity / max(self._portfolio_week_start_equity, 1e-12))
        if drawdown >= float(getattr(control, "weekly_drawdown_stop_pct", 0.0)) > 0:
            return 0.0, "portfolio_weekly_drawdown_stop"
        if loss >= float(getattr(control, "weekly_loss_stop_pct", 0.0)) > 0:
            return 0.0, "portfolio_weekly_loss_stop"
        multiplier = 1.0
        if drawdown >= float(getattr(control, "weekly_drawdown_reduce_pct", 0.0)) > 0:
            multiplier = min(multiplier, max(0.0, min(1.0, float(getattr(control, "weekly_drawdown_risk_multiplier", 0.5)))))
        if loss >= float(getattr(control, "weekly_loss_reduce_pct", 0.0)) > 0:
            multiplier = min(multiplier, max(0.0, min(1.0, float(getattr(control, "weekly_loss_risk_multiplier", 0.5)))))
        return multiplier, "portfolio_weekly_risk_reduced" if multiplier < 0.999 else "portfolio_weekly_base"

    def _vbp_live_weekly_limits(self, account: AccountSnapshot) -> tuple[int, float, str]:
        risk = self.config.vbp_strategy.risk_control
        base_max = max(1, int(self.config.vbp_strategy.position.max_positions))
        base_multiplier = 1.0
        if not bool(getattr(risk, "enabled", False)):
            return base_max, base_multiplier, "vbp_weekly_disabled"
        now = datetime.now(timezone.utc)
        key = now.isocalendar()[:2]
        if self._vbp_live_week_key != key:
            self._vbp_live_week_key = key
            self._vbp_live_week_start_equity = account.equity
            self._vbp_live_week_peak_equity = account.equity
            self._vbp_live_loss_streak = 0
        self._vbp_live_week_peak_equity = max(self._vbp_live_week_peak_equity, account.equity)
        if bool(getattr(risk, "weekly_drawdown_control_enabled", False)):
            drawdown = max(0.0, 1.0 - account.equity / max(self._vbp_live_week_peak_equity, 1e-12))
            loss = max(0.0, 1.0 - account.equity / max(self._vbp_live_week_start_equity, 1e-12))
            if drawdown >= float(getattr(risk, "weekly_drawdown_stop_pct", 0.0)) > 0:
                return 0, 0.0, "vbp_weekly_drawdown_stop"
            if loss >= float(getattr(risk, "weekly_loss_stop_pct", 0.0)) > 0:
                return 0, 0.0, "vbp_weekly_loss_stop"
            if (
                drawdown >= float(getattr(risk, "weekly_drawdown_one_position_pct", 0.0)) > 0
                or loss >= float(getattr(risk, "weekly_loss_one_position_pct", 0.0)) > 0
            ):
                return 1, min(base_multiplier, float(getattr(risk, "reduced_size_multiplier", 0.75))), "vbp_weekly_one_position"
            if (
                drawdown >= float(getattr(risk, "weekly_drawdown_half_size_pct", 0.0)) > 0
                or loss >= float(getattr(risk, "weekly_loss_reduce_pct", 0.0)) > 0
            ):
                return base_max, min(base_multiplier, 0.5), "vbp_weekly_half_size"
        if bool(getattr(risk, "consecutive_loss_reduce_enabled", False)) and time.time() < self._vbp_live_loss_reduce_until:
            return (
                max(1, min(base_max, int(getattr(risk, "consecutive_loss_reduced_max_positions", 1)))),
                min(base_multiplier, float(getattr(risk, "consecutive_loss_reduced_size_multiplier", 0.65))),
                "vbp_consecutive_loss_reduced",
            )
        return base_max, base_multiplier, "vbp_weekly_base"

    def _portfolio_live_bucket_count(self, account: AccountSnapshot, bucket: str) -> int:
        count = 0
        for symbol, position in account.positions.items():
            reason = position.entry_reason or self._entry_reasons.get(symbol, "")
            if _portfolio_bucket_from_reason(reason) == bucket:
                count += 1
        return count

    def _portfolio_live_altcoin_count(self, account: AccountSnapshot) -> int:
        return sum(1 for symbol in account.positions if symbol not in {"BTCUSDT", "ETHUSDT"})

    def _portfolio_live_btc_weak_multiplier(self) -> float:
        control = self.config.portfolio_control
        if not getattr(control, "enabled", False) or not getattr(control, "btc_weak_risk_reduction_enabled", False):
            return 1.0
        now = time.time()
        cached = self._portfolio_btc_weak_cache
        if cached and now - cached[0] < 60.0:
            return cached[1]
        multiplier = 1.0
        try:
            candles = self.client.klines("BTCUSDT", "1m", 260)
        except Exception:
            candles = []
        if len(candles) >= 241:
            ret_1h = candles[-1].close / max(candles[-61].close, 1e-12) - 1.0
            ret_4h = candles[-1].close / max(candles[-241].close, 1e-12) - 1.0
            if ret_1h <= float(control.btc_weak_1h_return_pct) or ret_4h <= float(control.btc_weak_4h_return_pct):
                multiplier = max(0.0, min(1.0, float(control.btc_weak_risk_multiplier)))
        self._portfolio_btc_weak_cache = (now, multiplier)
        return multiplier

    def _entry_execution_reference_price(self, candidate: EntryCandidate) -> float:
        latest = self._latest_close(candidate.symbol)
        return latest if latest > 0 else candidate.candle.close

    def _super_volume_chase_guard_reason(self, candidate: EntryCandidate) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "super_volume_live_chase_guard_enabled", False):
            return None
        reason = candidate.signal.reason.lower()
        if "super_volume" not in reason and "startup_breakout" not in reason:
            return None
        signal_close = candidate.candle.close
        latest = self._entry_execution_reference_price(candidate)
        if signal_close <= 0 or latest <= 0:
            return None
        chase_pct = (latest / signal_close - 1.0) * candidate.signal.direction.value
        max_chase = max(0.0, float(getattr(strategy, "super_volume_max_entry_chase_pct", 0.006)))
        if chase_pct <= max_chase:
            return None
        return (
            f"latest={latest:.6g} signal_close={signal_close:.6g} "
            f"追价={chase_pct * 100:.2f}% > {max_chase * 100:.2f}%"
        )

    def _entry_timing_guard_reason(self, candidate: EntryCandidate) -> str | None:
        if MTF_REASON_TOKEN in candidate.signal.reason:
            return None
        strategy = self.config.strategy
        if not getattr(strategy, "entry_timing_filter_enabled", False):
            return None

        timeframe = str(getattr(strategy, "entry_timing_timeframe", "15m"))
        fast_period = max(2, int(getattr(strategy, "entry_timing_rsi_fast_period", 6)))
        mid_period = max(fast_period, int(getattr(strategy, "entry_timing_rsi_mid_period", 12)))
        limit = max(80, mid_period + 10)
        try:
            candles = self.client.klines(candidate.symbol, timeframe, limit)
        except Exception as exc:
            self.log(f"{candidate.symbol}: 15m进场过滤数据不可用 ({exc})")
            return None
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(milliseconds=interval_to_milliseconds(timeframe))
        candles = [candle for candle in candles if candle.timestamp <= cutoff]
        if len(candles) < mid_period + 3:
            return None

        reclaim_reason = _indicator_long_reclaim_guard_reason_for_candles(self.config, candidate, candles, timeframe)
        if reclaim_reason:
            return reclaim_reason

        closes = [candle.close for candle in candles]
        fast_rsi = rsi(closes, fast_period)[-1]
        mid_rsi = rsi(closes, mid_period)[-1]
        latest = closes[-1]
        signal_close = candidate.candle.close
        if latest <= 0 or signal_close <= 0:
            return None

        direction = candidate.signal.direction
        chase_pct = (latest / signal_close - 1.0) * direction.value
        max_chase = max(0.0, float(getattr(strategy, "entry_timing_max_chase_pct", 0.010)))
        if chase_pct > max_chase:
            return (
                f"{timeframe}_chase latest={latest:.6g} signal_close={signal_close:.6g} "
                f"追价={chase_pct * 100:.2f}% > {max_chase * 100:.2f}%"
            )

        reversal_pct = (latest / candles[-2].close - 1.0) * direction.value if candles[-2].close > 0 else 0.0
        max_reversal = max(0.0, float(getattr(strategy, "entry_timing_reversal_pct", 0.006)))
        if reversal_pct < -max_reversal:
            return f"{timeframe}_短线反向={reversal_pct * 100:.2f}% < -{max_reversal * 100:.2f}%"

        if direction == Direction.LONG:
            fast_ceiling = float(getattr(strategy, "entry_timing_long_rsi_fast_ceiling", 82.0))
            mid_ceiling = float(getattr(strategy, "entry_timing_long_rsi_mid_ceiling", 76.0))
            if fast_rsi >= fast_ceiling and mid_rsi >= mid_ceiling:
                return f"{timeframe}_long_rsi_hot rsi{fast_period}={fast_rsi:.1f} rsi{mid_period}={mid_rsi:.1f}"
        elif direction == Direction.SHORT:
            fast_floor = float(getattr(strategy, "entry_timing_short_rsi_fast_floor", 18.0))
            mid_floor = float(getattr(strategy, "entry_timing_short_rsi_mid_floor", 24.0))
            if fast_rsi <= fast_floor and mid_rsi <= mid_floor:
                return f"{timeframe}_short_rsi_cold rsi{fast_period}={fast_rsi:.1f} rsi{mid_period}={mid_rsi:.1f}"
        return None

    def _entry_execution_guard_reason(self, candidate: EntryCandidate) -> str | None:
        if MTF_REASON_TOKEN in candidate.signal.reason:
            return None
        strategy = self.config.strategy
        if not getattr(strategy, "entry_execution_filter_enabled", False):
            return None
        if getattr(strategy, "entry_execution_filter_trend_only", True) and not _is_trend_entry_reason(candidate.signal.reason):
            return None

        timeframe = str(getattr(strategy, "entry_execution_timeframe", "5m"))
        fast_period = max(2, int(getattr(strategy, "entry_execution_rsi_fast_period", 6)))
        mid_period = max(fast_period, int(getattr(strategy, "entry_execution_rsi_mid_period", 12)))
        limit = max(80, mid_period + 10)
        try:
            candles = self.client.klines(candidate.symbol, timeframe, limit)
        except Exception as exc:
            self.log(f"{candidate.symbol}: 5m执行确认数据不可用 ({exc})")
            return None
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(milliseconds=interval_to_milliseconds(timeframe))
        candles = [candle for candle in candles if candle.timestamp <= cutoff]
        if len(candles) < mid_period + 3:
            return None

        return _entry_execution_guard_reason_for_candles(self.config, candidate, candles)

    def _trend_reference_adjusted_signal(self, symbol: str, signal: Signal) -> Signal | None:
        strategy = self.config.strategy
        if not getattr(strategy, "trend_reference_filter_enabled", False):
            return signal
        if signal.direction == Direction.FLAT or not _is_trend_entry_reason(signal.reason):
            return signal

        timeframe = str(getattr(strategy, "trend_reference_timeframe", "1h"))
        lookback = max(1, int(getattr(strategy, "trend_reference_lookback_bars", 6)))
        limit = max(strategy.slow_ema + lookback + 5, strategy.atr_period + lookback + 5, 120)
        try:
            candles = self._closed_candles_for_timeframe(symbol, timeframe, limit)
        except Exception as exc:
            self.log(f"{symbol}: 1h趋势方向保护数据不可用 ({exc})")
            return signal
        adjusted, reason = _trend_reference_adjusted_signal_for_candles(self.config, signal, candles)
        if adjusted is None:
            self.log(f"{symbol}: 1h趋势方向保护拒绝 {signal.direction.name} ({reason})")
        elif adjusted is not signal:
            self.log(f"{symbol}: 1h趋势方向保护降仓 {signal.direction.name} ({reason})")
        return adjusted

    def _low_base_volume_ignition_entry_candidate(self, symbol: str) -> EntryCandidate | None:
        strategy = self.config.strategy
        if not getattr(strategy, "low_base_volume_ignition_enabled", False):
            return None
        try:
            candles_30m = self._closed_candles_for_timeframe(symbol, "30m", 260)
        except Exception:
            self._record_low_base_reject("reject_data_unavailable")
            return None
        if len(candles_30m) < 120:
            self._record_low_base_reject("reject_warmup")
            return None
        closes_30 = [item.close for item in candles_30m]
        highs_30 = [item.high for item in candles_30m]
        lows_30 = [item.low for item in candles_30m]
        volumes_30 = [item.volume for item in candles_30m]
        ma25_30 = _sma_values(closes_30, 25)
        ma99_30 = _sma_values(closes_30, 99)
        atr_30 = atr(candles_30m, self.config.strategy.atr_period)
        rsi14_30 = rsi(closes_30, 14)
        rsi6_30 = rsi(closes_30, 6)
        _, _, hist_30 = macd(closes_30, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)
        ignition_index, _, _ = self._low_base_find_recent_ignition(
            candles_30m,
            highs_30,
            lows_30,
            closes_30,
            volumes_30,
            ma25_30,
            ma99_30,
            atr_30,
            rsi14_30,
            rsi6_30,
            hist_30,
        )
        if ignition_index is None:
            self._record_low_base_reject("reject_no_volume_ignition")
            return None
        try:
            candles_15m = self._closed_candles_for_timeframe(symbol, "15m", 220)
            lookback_days = max(10, int(getattr(strategy, "low_base_range_lookback_days", 30)))
            candles_1h = self._closed_candles_for_timeframe(symbol, "1h", min(1500, lookback_days * 24 + 80))
            candles_4h = self._closed_candles_for_timeframe(symbol, "4h", 120)
            btc_15m = self._closed_candles_for_timeframe("BTCUSDT", "15m", 80)
            btc_1h = self._closed_candles_for_timeframe("BTCUSDT", "1h", 120)
            btc_4h = self._closed_candles_for_timeframe("BTCUSDT", "4h", 80)
        except Exception:
            self._record_low_base_reject("reject_data_unavailable")
            return None
        result = self._low_base_volume_ignition_signal(symbol, candles_1h, candles_4h, candles_30m, candles_15m, btc_15m, btc_1h, btc_4h)
        if result is None:
            return None
        signal, candidate_candle = result
        rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles_30m)
        return EntryCandidate(symbol, signal, candidate_candle, rank_score, momentum_pct, volume_ratio, "low_base_volume_ignition_long")

    def _low_base_volume_ignition_signal(
        self,
        symbol: str,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        candles_30m: list[Candle],
        candles_15m: list[Candle],
        btc_15m: list[Candle],
        btc_1h: list[Candle],
        btc_4h: list[Candle],
    ) -> tuple[Signal, Candle] | None:
        strategy = self.config.strategy
        if len(candles_1h) < 240 or len(candles_4h) < 20 or len(candles_30m) < 120 or len(candles_15m) < 120:
            self._record_low_base_reject("reject_warmup")
            return None
        if self._low_base_btc_reject_reason(btc_15m, btc_1h, btc_4h):
            self._record_low_base_reject("reject_btc")
            return None
        relative_reason = self._low_base_relative_strength_reject_reason(candles_1h, candles_4h, btc_1h, btc_4h)
        if relative_reason:
            self._record_low_base_reject(relative_reason)
            return None
        breadth_reason = self._low_base_market_breadth_reject_reason()
        if breadth_reason:
            self._record_low_base_reject(breadth_reason)
            return None

        close_1h = candles_1h[-1].close
        highs_1h = [item.high for item in candles_1h]
        lows_1h = [item.low for item in candles_1h]
        range_low = min(lows_1h)
        range_high = max(highs_1h)
        range_width = max(range_high - range_low, 1e-12)
        low_position = (close_1h - range_low) / range_width
        if low_position > float(getattr(strategy, "low_base_position_max", 0.55)):
            self._record_low_base_reject("reject_not_low_base")
            return None

        closes_30 = [item.close for item in candles_30m]
        highs_30 = [item.high for item in candles_30m]
        lows_30 = [item.low for item in candles_30m]
        volumes_30 = [item.volume for item in candles_30m]
        ma7_30 = _sma_values(closes_30, 7)
        ma25_30 = _sma_values(closes_30, 25)
        ma99_30 = _sma_values(closes_30, 99)
        atr_30 = atr(candles_30m, self.config.strategy.atr_period)
        rsi14_30 = rsi(closes_30, 14)
        rsi6_30 = rsi(closes_30, 6)
        _, _, hist_30 = macd(closes_30, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)

        pump_lookback = min(max(2, int(getattr(strategy, "low_base_recent_pump_lookback_30m", 48))), len(closes_30) - 1)
        recent_return = closes_30[-1] / max(closes_30[-1 - pump_lookback], 1e-12) - 1.0
        if recent_return > float(getattr(strategy, "low_base_recent_pump_return", 0.12)):
            self._record_low_base_reject("reject_recent_pump")
            return None
        if abs(closes_30[-1] - ma25_30[-1]) / max(ma25_30[-1], 1e-12) > float(getattr(strategy, "low_base_max_distance_ma25_pct", 0.05)):
            self._record_low_base_reject("reject_too_far_from_ma")
            return None

        ignition_index, breakout_level, base_score = self._low_base_find_recent_ignition(
            candles_30m,
            highs_30,
            lows_30,
            closes_30,
            volumes_30,
            ma25_30,
            ma99_30,
            atr_30,
            rsi14_30,
            rsi6_30,
            hist_30,
        )
        if ignition_index is None or breakout_level is None:
            self._record_low_base_reject("reject_no_volume_ignition")
            return None

        trigger = self._low_base_trigger_mode(candles_15m, candles_30m[ignition_index], breakout_level)
        if trigger is None:
            self._record_low_base_reject("reject_no_entry_trigger")
            return None
        mode, trigger_candle, stop_anchor = trigger

        closes_15 = [item.close for item in candles_15m]
        ma7_15 = _sma_values(closes_15, 7)
        ma25_15 = _sma_values(closes_15, 25)
        atr_15 = atr(candles_15m, self.config.strategy.atr_period)
        entry = trigger_candle.close
        if atr_15[-1] > 0 and (entry - ma7_15[-1]) / atr_15[-1] > float(getattr(strategy, "low_base_max_distance_ma7_atr", 2.5)):
            self._record_low_base_reject("reject_direct_entry_overextended")
            return None
        buffer = float(getattr(strategy, "low_base_structure_stop_buffer_atr", 0.2)) * atr_15[-1]
        stop_level = min(stop_anchor, breakout_level, ma25_15[-1]) - buffer
        stop_pct = (entry - stop_level) / max(entry, 1e-12)
        max_stop = float(getattr(strategy, "low_base_max_initial_stop_pct", 0.025))
        if stop_pct <= 0.002 or stop_pct > max_stop:
            self._record_low_base_reject("reject_stop_distance")
            return None
        take_profit_pct = stop_pct * max(1.8, float(getattr(strategy, "low_base_min_reward_risk", 1.8)))

        score = base_score
        score += 1.5 if mode == "pullback_reclaim_v2" else (0.8 if mode == "flag_breakout_v2" else 0.3)
        score += 1.0 if closes_30[ignition_index] > ma99_30[ignition_index] else 0.0
        score += min(2.0, volumes_30[ignition_index] / max(sum(volumes_30[max(0, ignition_index - 30):ignition_index]) / max(1, min(30, ignition_index)), 1e-12) - 1.0)
        score += 1.0 if hist_30[ignition_index] > hist_30[ignition_index - 1] else 0.0
        score -= 1.0 if rsi6_30[ignition_index] > float(getattr(strategy, "low_base_rsi6_overheat_block", 88.0)) - 5.0 else 0.0
        if score < float(getattr(strategy, "low_base_quality_score_threshold", 6.0)):
            self._record_low_base_reject("reject_low_quality_score")
            return None

        risk_multiplier = float(getattr(strategy, "low_base_risk_multiplier", 0.55))
        if score >= float(getattr(strategy, "low_base_high_score_threshold", 9.0)):
            risk_multiplier = float(getattr(strategy, "low_base_high_score_risk_multiplier", 0.75))
        signal = Signal(
            Direction.LONG,
            min(1.0, 0.55 + score / 20.0),
            f"low_base_volume_ignition_long {mode} score={score:.1f} level={breakout_level:.8g} stop={stop_pct * 100:.2f}%",
            stop_pct,
            take_profit_pct,
            risk_multiplier=risk_multiplier,
            max_holding_bars=max(1, int(getattr(strategy, "low_base_max_holding_bars", 32))),
        )
        return signal, trigger_candle

    def _record_low_base_reject(self, reason: str) -> None:
        self.low_base_ignition_stats[reason] = self.low_base_ignition_stats.get(reason, 0) + 1

    def _low_base_btc_reject_reason(self, btc_15m: list[Candle], btc_1h: list[Candle], btc_4h: list[Candle]) -> str | None:
        strategy = self.config.strategy
        if len(btc_15m) >= 5:
            ret_15m = btc_15m[-1].close / max(btc_15m[-5].close, 1e-12) - 1.0
            if ret_15m <= float(getattr(strategy, "low_base_btc_15m_drop_block", -0.008)):
                return "reject_btc_fast_drop"
        if len(btc_1h) >= 30:
            closes = [item.close for item in btc_1h]
            ret_1h = btc_1h[-1].close / max(btc_1h[-2].close, 1e-12) - 1.0
            ema9 = ema(closes, 9)
            ema21 = ema(closes, 21)
            _, _, hist = macd(closes, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)
            if ret_1h <= float(getattr(strategy, "low_base_btc_1h_drop_block", -0.015)):
                return "reject_btc_fast_drop"
            if bool(getattr(strategy, "low_base_btc_require_1h_not_bear", True)) and closes[-1] < ema21[-1] and ema9[-1] < ema21[-1] and hist[-1] < hist[-2]:
                return "reject_btc_1h_bear"
        if len(btc_4h) >= 30 and bool(getattr(strategy, "low_base_btc_require_4h_not_crash", True)):
            closes = [item.close for item in btc_4h]
            ema21 = ema(closes, 21)
            if closes[-1] < ema21[-1] and closes[-1] / max(closes[-4].close if hasattr(closes[-4], "close") else closes[-4], 1e-12) - 1.0 < -0.025:
                return "reject_btc_4h_crash"
        return None

    def _low_base_relative_strength_reject_reason(
        self,
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        btc_1h: list[Candle],
        btc_4h: list[Candle],
    ) -> str | None:
        strategy = self.config.strategy
        if not bool(getattr(strategy, "low_base_relative_strength_enabled", False)):
            return None
        if len(candles_1h) < 2 or len(candles_4h) < 2 or len(btc_1h) < 2 or len(btc_4h) < 2:
            return "reject_relative_strength_data"
        symbol_ret_1h = candles_1h[-1].close / max(candles_1h[-2].close, 1e-12) - 1.0
        btc_ret_1h = btc_1h[-1].close / max(btc_1h[-2].close, 1e-12) - 1.0
        symbol_ret_4h = candles_4h[-1].close / max(candles_4h[-2].close, 1e-12) - 1.0
        btc_ret_4h = btc_4h[-1].close / max(btc_4h[-2].close, 1e-12) - 1.0
        if symbol_ret_1h - btc_ret_1h < float(getattr(strategy, "low_base_min_relative_1h", 0.0015)):
            return "reject_relative_strength_1h"
        if symbol_ret_4h - btc_ret_4h < float(getattr(strategy, "low_base_min_relative_4h", 0.0025)):
            return "reject_relative_strength_4h"
        return None

    def _low_base_market_breadth_reject_reason(self) -> str | None:
        strategy = self.config.strategy
        if not bool(getattr(strategy, "low_base_market_breadth_enabled", False)):
            return None
        symbols = tuple(getattr(self.config.trading, "symbols", ()))[:100]
        checked = 0
        up_1h = 0
        up_15m = 0
        above_ema21 = 0
        returns_1h = []
        for symbol in symbols:
            try:
                candles_1h = self._closed_candles_for_timeframe(symbol, "1h", 25)
                candles_15m = self._closed_candles_for_timeframe(symbol, "15m", 6)
            except Exception:
                continue
            if len(candles_1h) < 22 or len(candles_15m) < 2:
                continue
            checked += 1
            ret_1h = candles_1h[-1].close / max(candles_1h[-2].close, 1e-12) - 1.0
            ret_15m = candles_15m[-1].close / max(candles_15m[-2].close, 1e-12) - 1.0
            returns_1h.append(ret_1h)
            up_1h += 1 if ret_1h > 0 else 0
            up_15m += 1 if ret_15m > 0 else 0
            above_ema21 += 1 if candles_1h[-1].close > ema([item.close for item in candles_1h], 21)[-1] else 0
        if checked < 20:
            return None
        if up_1h / checked < float(getattr(strategy, "low_base_market_breadth_min_1h_up_pct", 0.45)):
            return "reject_market_breadth_1h"
        if up_15m / checked < float(getattr(strategy, "low_base_market_breadth_min_15m_up_pct", 0.42)):
            return "reject_market_breadth_15m"
        if above_ema21 / checked < float(getattr(strategy, "low_base_market_breadth_min_ema21_pct", 0.42)):
            return "reject_market_breadth_ema21"
        if sum(returns_1h) / max(1, len(returns_1h)) < float(getattr(strategy, "low_base_market_breadth_min_avg_1h_return", -0.003)):
            return "reject_market_breadth_avg_return"
        return None

    def _low_base_find_recent_ignition(
        self,
        candles: list[Candle],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        ma25: list[float],
        ma99: list[float],
        atr_values: list[float],
        rsi14_values: list[float],
        rsi6_values: list[float],
        hist: list[float],
    ) -> tuple[int | None, float | None, float]:
        strategy = self.config.strategy
        breakout_lookback = max(12, int(getattr(strategy, "low_base_breakout_lookback_30m", 36)))
        base_lookback = max(12, int(getattr(strategy, "low_base_lookback_bars_30m", 48)))
        for index in range(len(candles) - 1, max(110, len(candles) - 10), -1):
            if index <= max(breakout_lookback, base_lookback, 100):
                continue
            base_start = index - base_lookback
            breakout_start = index - breakout_lookback
            base_high = max(highs[base_start:index])
            base_low = min(lows[base_start:index])
            base_range = (base_high - base_low) / max(closes[index], 1e-12)
            if base_range > float(getattr(strategy, "low_base_max_range_pct", 0.08)):
                continue
            if bool(getattr(strategy, "low_base_atr_contraction_required", True)):
                recent_atr = sum(atr_values[index - 10:index]) / 10
                previous_atr = sum(atr_values[index - 40:index - 10]) / 30
                if previous_atr > 0 and recent_atr > previous_atr * 1.10:
                    continue
            if bool(getattr(strategy, "low_base_volume_contraction_required", False)):
                recent_vol = sum(volumes[index - 10:index]) / 10
                previous_vol = sum(volumes[index - 40:index - 10]) / 30
                if previous_vol > 0 and recent_vol > previous_vol * 1.15:
                    continue
            breakout_level = max(highs[breakout_start:index])
            volume_avg = sum(volumes[index - 30:index]) / 30
            volume_ratio = volumes[index] / max(volume_avg, 1e-12)
            candle = candles[index]
            candle_range = max(candle.high - candle.low, 1e-12)
            close_position = (candle.close - candle.low) / candle_range
            upper_wick = (candle.high - max(candle.open, candle.close)) / candle_range
            if candle.close <= breakout_level:
                continue
            if volume_ratio < float(getattr(strategy, "low_base_volume_multiplier_min", 2.0)):
                continue
            if close_position < float(getattr(strategy, "low_base_close_position_min", 0.70)):
                continue
            if upper_wick > float(getattr(strategy, "low_base_upper_wick_max_ratio", 0.45)):
                continue
            if bool(getattr(strategy, "low_base_require_close_above_ma25", True)) and candle.close < ma25[index]:
                continue
            if bool(getattr(strategy, "low_base_require_close_above_ma99", False)) and candle.close < ma99[index]:
                continue
            if bool(getattr(strategy, "low_base_macd_hist_expand_required", True)) and not (hist[index] > hist[index - 1] and hist[index] > hist[index - 2]):
                continue
            if rsi14_values[index] < float(getattr(strategy, "low_base_rsi14_min", 55.0)):
                continue
            if rsi14_values[index] > float(getattr(strategy, "low_base_rsi14_max", 78.0)) or rsi6_values[index] > float(getattr(strategy, "low_base_rsi6_overheat_block", 88.0)):
                continue
            score = 3.0
            score += 1.0 if base_range <= float(getattr(strategy, "low_base_max_range_pct", 0.08)) * 0.75 else 0.0
            score += 1.0 if ma25[index] >= ma25[index - 4] else 0.0
            ma7_values = _sma_values(closes[: index + 1], 7)
            score += 1.0 if len(ma7_values) >= 4 and ma7_values[-1] >= ma7_values[-4] else 0.0
            score += 1.0 if volume_ratio >= float(getattr(strategy, "low_base_volume_multiplier_min", 2.0)) * 1.25 else 0.0
            score += 1.0 if close_position >= 0.80 else 0.0
            return index, breakout_level, score
        return None, None, 0.0

    def _low_base_trigger_mode(self, candles_15m: list[Candle], ignition_candle: Candle, breakout_level: float) -> tuple[str, Candle, float] | None:
        strategy = self.config.strategy
        post = [item for item in candles_15m if item.timestamp >= ignition_candle.timestamp]
        if len(post) < 2:
            return None
        closes = [item.close for item in candles_15m]
        volumes = [item.volume for item in candles_15m]
        ma7 = _sma_values(closes, 7)
        ma25 = _sma_values(closes, 25)
        current = candles_15m[-1]
        candle_range = max(current.high - current.low, 1e-12)
        close_position = (current.close - current.low) / candle_range
        volume_avg = sum(volumes[-25:-1]) / max(1, min(24, len(volumes) - 1))
        volume_ratio = current.volume / max(volume_avg, 1e-12)

        if bool(getattr(strategy, "low_base_pullback_enabled", True)) and bool(getattr(strategy, "low_base_pullback_v2_enabled", False)) and len(post) >= 4:
            pullback_low = min(item.low for item in post)
            recent_lows = [item.low for item in candles_15m[-3:]]
            low_not_breaking = recent_lows[-1] >= min(recent_lows[:-1]) * 0.998
            touched_support = min(item.low for item in post[-4:]) <= max(breakout_level, ma7[-1], ma25[-1]) * 1.006
            held_level = pullback_low >= breakout_level * (1.0 - float(getattr(strategy, "low_base_pullback_max_depth_pct", 0.025)))
            reclaimed = current.close >= max(breakout_level, ma7[-1])
            quiet_pullback = volume_ratio <= float(getattr(strategy, "low_base_pullback_volume_max_ratio", 1.2))
            strong_candle = current.close > current.open and close_position >= float(getattr(strategy, "low_base_pullback_reclaim_close_position_min", 0.60))
            if touched_support and held_level and low_not_breaking and reclaimed and quiet_pullback and strong_candle:
                return "pullback_reclaim_v2", current, pullback_low

        if bool(getattr(strategy, "low_base_pullback_enabled", True)) and not bool(getattr(strategy, "low_base_pullback_v2_enabled", False)) and len(post) >= 3:
            pullback_low = min(item.low for item in post)
            held_level = pullback_low >= breakout_level * (1.0 - float(getattr(strategy, "low_base_pullback_max_depth_pct", 0.025)))
            reclaimed = current.close >= max(breakout_level, ma7[-1])
            quiet_pullback = volume_ratio <= float(getattr(strategy, "low_base_pullback_volume_max_ratio", 1.2)) or current.close > current.open
            if held_level and reclaimed and quiet_pullback and close_position >= float(getattr(strategy, "low_base_pullback_reclaim_close_position_min", 0.60)):
                return "pullback_reclaim", current, pullback_low

        if bool(getattr(strategy, "low_base_flag_enabled", True)):
            min_bars = max(2, int(getattr(strategy, "low_base_flag_bars_min", 3)))
            max_bars = max(min_bars, int(getattr(strategy, "low_base_flag_bars_max", 8)))
            flag = post[-max_bars:]
            if len(flag) >= min_bars:
                previous = flag[:-1]
                flag_high = max(item.high for item in previous)
                flag_low = min(item.low for item in previous)
                flag_range = (flag_high - flag_low) / max(current.close, 1e-12)
                if (
                    flag_range <= float(getattr(strategy, "low_base_flag_max_range_pct", 0.04))
                    and current.close > flag_high
                    and min(item.low for item in flag) >= min(ma7[-1], ma25[-1]) * 0.995
                    and volume_ratio >= float(getattr(strategy, "low_base_flag_breakout_volume_multiplier", 1.5))
                ):
                    return "flag_breakout_v2", current, flag_low

        if bool(getattr(strategy, "low_base_direct_entry_enabled", True)) and candles_15m[-1].timestamp <= ignition_candle.timestamp:
            return None
        if bool(getattr(strategy, "low_base_direct_entry_enabled", True)) and len(post) <= 3:
            if (
                current.close > breakout_level
                and volume_ratio >= float(getattr(strategy, "low_base_direct_entry_volume_multiplier_min", 3.0))
                and close_position >= float(getattr(strategy, "low_base_direct_entry_close_position_min", 0.80))
            ):
                return "direct_ignition", current, min(item.low for item in post)
        return None

    def _entry_rank_metrics(self, signal: Signal, candles: list[Candle]) -> tuple[float, float, float]:
        candle = candles[-1]
        lookback = min(4, len(candles) - 1)
        if lookback > 0 and candles[-1 - lookback].close > 0:
            raw_momentum = candle.close / candles[-1 - lookback].close - 1.0
        else:
            raw_momentum = 0.0
        directional_momentum = raw_momentum * signal.direction.value

        volume_window = candles[-25:-1] if len(candles) >= 25 else candles[:-1]
        average_volume = sum(item.volume for item in volume_window) / len(volume_window) if volume_window else 0.0
        volume_ratio = candle.volume / average_volume if average_volume > 0 else 1.0

        candle_body_pct = (candle.close - candle.open) / max(candle.open, 1e-12) * signal.direction.value
        risk_weight = signal_risk_weight(signal.confidence, signal.risk_multiplier)
        momentum_score = max(-1.0, min(3.0, directional_momentum / 0.04))
        volume_score = max(0.0, min(2.0, volume_ratio - 1.0))
        candle_score = max(-1.0, min(1.5, candle_body_pct / 0.01))
        rank_score = risk_weight * 3.0 + momentum_score * 2.0 + volume_score * 1.5 + candle_score
        return rank_score, directional_momentum, volume_ratio

    def _weak_market_adjusted_signal(self, signal: Signal, rank_score: float) -> Signal | None:
        strategy = self.config.strategy
        if signal.direction != Direction.LONG or not getattr(strategy, "weak_market_long_filter_enabled", False):
            return signal

        regime = self._market_regime()
        if not regime.weak:
            return signal

        if getattr(strategy, "weak_market_long_super_volume_only", True) and "super_volume" not in signal.reason:
            return None
        min_rank = max(0.0, float(getattr(strategy, "weak_market_long_min_rank_score", 4.8)))
        if rank_score < min_rank:
            return None

        multiplier = max(0.0, min(1.0, float(getattr(strategy, "weak_market_long_risk_multiplier", 0.55))))
        return Signal(
            signal.direction,
            signal.confidence,
            f"{signal.reason}_weak_market_guard {regime.reason}",
            signal.stop_loss_pct,
            signal.take_profit_pct,
            risk_multiplier=signal.risk_multiplier * multiplier,
            max_holding_bars=signal.max_holding_bars,
        )

    def _strong_market_adjusted_signal(self, signal: Signal, rank_score: float) -> Signal | None:
        strategy = self.config.strategy
        if signal.direction != Direction.SHORT or not getattr(strategy, "strong_market_short_filter_enabled", False):
            return signal

        regime = self._market_regime()
        strong = (
            regime.breadth_pct >= max(0.0, min(1.0, float(getattr(strategy, "strong_market_breadth_threshold", 0.58))))
            and regime.avg_return_pct >= float(getattr(strategy, "strong_market_avg_return_threshold", 0.006)) * 100.0
        )
        if not strong:
            return signal

        min_rank = max(0.0, float(getattr(strategy, "strong_market_short_min_rank_score", 6.2)))
        is_exceptional = "super_volume" in signal.reason or "startup_breakout" in signal.reason
        if not is_exceptional and rank_score < min_rank:
            return None

        multiplier = max(0.0, min(1.0, float(getattr(strategy, "strong_market_short_risk_multiplier", 0.45))))
        return Signal(
            signal.direction,
            signal.confidence,
            f"{signal.reason}_strong_market_short_guard {regime.reason}",
            signal.stop_loss_pct,
            signal.take_profit_pct,
            risk_multiplier=signal.risk_multiplier * multiplier,
            max_holding_bars=signal.max_holding_bars,
        )

    def _market_regime(self) -> MarketRegime:
        now = time.time()
        cached = self._market_regime_cache
        if cached and now - cached[0] < 30.0:
            return cached[1]

        strategy = self.config.strategy
        if not (
            getattr(strategy, "weak_market_long_filter_enabled", False)
            or getattr(strategy, "strong_market_short_filter_enabled", False)
        ):
            regime = MarketRegime(False, 1.0, 0.0, "weak_market_disabled")
            self._market_regime_cache = (now, regime)
            return regime

        symbols = tuple(self.config.trading.entry_symbols or self.config.trading.symbols)
        weak_lookback = int(getattr(strategy, "weak_market_lookback_bars", 48))
        strong_lookback = int(getattr(strategy, "strong_market_lookback_bars", weak_lookback))
        lookback = max(1, weak_lookback, strong_lookback)
        limit = max(strategy.slow_ema + lookback + 5, lookback + 5, 120)
        constructive = 0
        valid = 0
        returns = []
        for symbol in symbols:
            candles = self._closed_candles_for_timeframe(symbol, self.config.trading.timeframe, limit)
            if len(candles) <= lookback or len(candles) < strategy.slow_ema:
                continue
            closes = [candle.close for candle in candles]
            previous = closes[-1 - lookback]
            current = closes[-1]
            if previous <= 0:
                continue
            valid += 1
            returns.append(current / previous - 1.0)
            slow = ema(closes, strategy.slow_ema)
            if current > previous and current > slow[-1]:
                constructive += 1

        breadth = 1.0 if valid <= 0 else constructive / valid
        avg_return = 0.0 if not returns else sum(returns) / len(returns)
        weak = (
            breadth <= max(0.0, min(1.0, float(getattr(strategy, "weak_market_breadth_threshold", 0.42))))
            and avg_return <= float(getattr(strategy, "weak_market_avg_return_threshold", -0.012))
        )
        regime = MarketRegime(weak, breadth, avg_return * 100.0, f"breadth={breadth:.2f} avg={avg_return * 100.0:.2f}%")
        self._market_regime_cache = (now, regime)
        return regime

    def _btc_market_adjusted_signal(
        self,
        signal: Signal,
        rank_score: float,
        momentum_pct: float,
        volume_ratio: float,
    ) -> Signal | None:
        strategy = self.config.strategy
        if not getattr(strategy, "btc_market_filter_enabled", False):
            return signal
        if not self._btc_market_filter_applies(signal):
            return signal
        state = self._btc_market_state()
        if state.direction == Direction.FLAT or state.direction == signal.direction:
            return signal

        if (
            signal.direction == Direction.LONG
            and state.direction == Direction.SHORT
            and getattr(strategy, "btc_counter_trend_reduce_only_enabled", False)
        ):
            multiplier = max(0.0, min(1.0, float(getattr(strategy, "btc_weak_long_risk_multiplier", getattr(strategy, "btc_counter_trend_risk_multiplier", 0.45)))))
            return replace(
                signal,
                risk_multiplier=signal.risk_multiplier * multiplier,
                reason=f"{signal.reason}_btc_weak_long_reduce {state.reason}",
            )

        is_exceptional = "super_volume" in signal.reason or "startup_breakout" in signal.reason
        if getattr(strategy, "btc_counter_trend_super_volume_only", True) and not is_exceptional:
            return None
        if rank_score < float(getattr(strategy, "btc_counter_trend_min_rank_score", 6.2)):
            return None
        if momentum_pct < float(getattr(strategy, "btc_counter_trend_min_momentum_pct", 0.025)):
            return None
        if volume_ratio < float(getattr(strategy, "btc_counter_trend_min_volume_ratio", 2.2)):
            return None

        multiplier = max(0.0, min(1.0, float(getattr(strategy, "btc_counter_trend_risk_multiplier", 0.45))))
        return replace(
            signal,
            risk_multiplier=signal.risk_multiplier * multiplier,
            reason=f"{signal.reason}_btc_counter_guard {state.reason}",
        )

    def _btc_market_filter_applies(self, signal: Signal) -> bool:
        strategy = self.config.strategy
        if not getattr(strategy, "btc_market_filter_trend_only", True):
            return True
        reason = signal.reason.lower()
        trend_like = any(
            token in reason
            for token in ("breakout", "breakdown", "pullback", "startup_breakout", "super_volume")
        )
        holding_bars = signal.max_holding_bars or self.config.strategy.max_holding_bars
        long_holding = holding_bars >= max(1, int(getattr(strategy, "btc_market_filter_min_holding_bars", 18)))
        return trend_like or long_holding

    def _btc_market_state(self) -> BtcMarketState:
        now = time.time()
        cached = self._btc_market_state_cache
        if cached and now - cached[0] < 30.0:
            return cached[1]

        strategy = self.config.strategy
        if not getattr(strategy, "btc_market_filter_enabled", False):
            state = BtcMarketState(Direction.FLAT, 0.0, 0.0, "btc_market_disabled")
            self._btc_market_state_cache = (now, state)
            return state

        symbol = str(getattr(strategy, "btc_market_symbol", "BTCUSDT")).upper()
        timeframe = str(getattr(strategy, "btc_market_timeframe", self.config.trading.timeframe))
        lookback = max(1, int(getattr(strategy, "btc_market_lookback_bars", 12)))
        state = self._btc_market_state_for_timeframe(symbol, timeframe, lookback)
        if getattr(strategy, "btc_market_confirmation_enabled", False):
            confirmation = self._btc_market_state_for_timeframe(
                symbol,
                str(getattr(strategy, "btc_market_confirmation_timeframe", "8h")),
                max(1, int(getattr(strategy, "btc_market_confirmation_lookback_bars", 2))),
            )
            state = _combine_btc_market_states(state, confirmation)
        self._btc_market_state_cache = (now, state)
        return state

    def _btc_market_state_for_timeframe(self, symbol: str, timeframe: str, lookback: int) -> BtcMarketState:
        strategy = self.config.strategy
        ema_period = max(2, int(getattr(strategy, "btc_market_ema_period", self.config.strategy.slow_ema)))
        limit = max(ema_period + lookback + 5, self.config.strategy.atr_period + lookback + 5, 120)
        candles = self._closed_candles_for_timeframe(symbol, timeframe, limit)
        if len(candles) <= lookback or len(candles) < max(ema_period, self.config.strategy.atr_period):
            return BtcMarketState(Direction.FLAT, 0.0, 0.0, f"btc_{timeframe}_warmup")

        closes = [candle.close for candle in candles]
        slow = ema(closes, ema_period)
        atr_values = atr(candles, self.config.strategy.atr_period)
        current = closes[-1]
        previous = closes[-1 - lookback]
        atr_value = atr_values[-1]
        return_pct = current / previous - 1.0 if previous > 0 else 0.0
        slope_atr = (slow[-1] - slow[-1 - lookback]) / atr_value if atr_value > 0 else 0.0
        bull_return = float(getattr(strategy, "btc_bull_return_threshold", 0.006))
        bear_return = float(getattr(strategy, "btc_bear_return_threshold", -0.006))
        slope_threshold = float(getattr(strategy, "btc_market_slope_atr_threshold", 0.20))

        if return_pct >= bull_return and slope_atr >= slope_threshold and current >= slow[-1]:
            direction = Direction.LONG
            label = "btc_bull"
        elif return_pct <= bear_return and slope_atr <= -slope_threshold and current <= slow[-1]:
            direction = Direction.SHORT
            label = "btc_bear"
        else:
            direction = Direction.FLAT
            label = "btc_neutral"
        return BtcMarketState(direction, return_pct * 100.0, slope_atr, f"{label}_{timeframe} ret={return_pct * 100:.2f}% slope_atr={slope_atr:.2f}")

    def _indicator_reversal_signal(self, candles: list[Candle]) -> Signal:
        config = self.config.filters
        if not config.extreme_reversal_entry_enabled:
            return Signal(Direction.FLAT, 0.0, "indicator_reversal_disabled", 0.0, 0.0)
        minimum = max(config.rsi_period + 2, config.macd_slow + config.macd_signal + 3, config.kdj_period + 2, self.config.strategy.atr_period + 2)
        if len(candles) < minimum:
            return Signal(Direction.FLAT, 0.0, "indicator_warmup", 0.0, 0.0)

        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, config.rsi_period)
        macd_line, macd_signal_line, macd_histogram = macd(closes, config.macd_fast, config.macd_slow, config.macd_signal)
        k_values, d_values, _ = kdj(candles, config.kdj_period)
        atr_values = atr(candles, self.config.strategy.atr_period)
        slow_values = ema(closes, self.config.strategy.slow_ema)

        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        current_macd = macd_line[-1]
        current_signal = macd_signal_line[-1]
        previous_macd = macd_line[-2]
        previous_signal = macd_signal_line[-2]
        current_hist = macd_histogram[-1]
        previous_hist = macd_histogram[-2]
        current_k = k_values[-1]
        current_d = d_values[-1]
        previous_k = k_values[-2]
        previous_d = d_values[-2]
        candle = candles[-1]

        atr_pct = max(atr_values[-1] / candle.close, 0.0001)
        indicator_size_multiplier = max(
            0.0,
            float(getattr(self.config.strategy, "indicator_reversal_size_multiplier", 1.0)),
        )

        lookback_start = max(1, len(macd_line) - max(1, config.reversal_cross_lookback_bars))
        long_cross = False
        short_cross = False
        for index in range(lookback_start, len(macd_line)):
            if (
                macd_line[index - 1] <= macd_signal_line[index - 1]
                and macd_line[index] > macd_signal_line[index]
                and (macd_line[index] < 0 or macd_signal_line[index] < 0)
            ):
                long_cross = True
            if (
                macd_line[index - 1] >= macd_signal_line[index - 1]
                and macd_line[index] < macd_signal_line[index]
                and (macd_line[index] > 0 or macd_signal_line[index] > 0)
            ):
                short_cross = True
        long_pre_cross = (
            config.pre_cross_entry_enabled
            and current_macd < current_signal
            and current_hist > previous_hist
            and (current_macd < 0 or current_signal < 0)
            and (current_rsi <= config.long_extreme_rsi or min(current_k, current_d) <= config.long_extreme_kdj)
            and (current_rsi >= previous_rsi or current_k > previous_k)
        )
        short_pre_cross = (
            config.pre_cross_entry_enabled
            and current_macd > current_signal
            and current_hist < previous_hist
            and (current_macd > 0 or current_signal > 0)
            and (current_rsi >= config.short_extreme_rsi or max(current_k, current_d) >= config.short_extreme_kdj)
            and (current_rsi <= previous_rsi or current_k < previous_k)
        )

        if long_cross:
            if self.config.strategy.long_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_long_disabled", 0.0, 0.0)
            strict_guard_reason = self._indicator_confirmed_cross_required_extreme_guard_reason(
                Direction.LONG,
                current_rsi,
                current_k,
                current_d,
            )
            if strict_guard_reason:
                return Signal(Direction.FLAT, 0.0, strict_guard_reason, 0.0, 0.0)
            context_guard_reason = self._indicator_confirmed_cross_context_guard_reason(
                Direction.LONG,
                closes,
                rsi_values,
                k_values,
                d_values,
                slow_values,
                atr_values,
            )
            if context_guard_reason:
                return Signal(Direction.FLAT, 0.0, context_guard_reason, 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.LONG, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            long_stop_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.LONG, "stop_loss_atr", self.config.strategy.stop_loss_atr),
                0.0008,
            )
            long_take_profit_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.LONG, "take_profit_atr", self.config.strategy.take_profit_atr),
                long_stop_pct * 1.05,
            )
            long_size_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.LONG,
                "size_multiplier",
                indicator_size_multiplier,
            )
            long_risk_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.LONG,
                "confirmed_cross_risk_multiplier",
                config.confirmed_cross_risk_multiplier,
            )
            long_holding_bars = _indicator_side_holding_bars(self.config.strategy, Direction.LONG)
            return Signal(
                Direction.LONG,
                0.7,
                f"indicator_long_macd_golden_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                long_stop_pct,
                long_take_profit_pct,
                risk_multiplier=long_risk_multiplier * self.config.strategy.long_risk_bias * long_size_multiplier,
                max_holding_bars=long_holding_bars,
            )
        if short_cross:
            if self.config.strategy.short_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_short_disabled", 0.0, 0.0)
            strict_guard_reason = self._indicator_confirmed_cross_required_extreme_guard_reason(
                Direction.SHORT,
                current_rsi,
                current_k,
                current_d,
            )
            if strict_guard_reason:
                return Signal(Direction.FLAT, 0.0, strict_guard_reason, 0.0, 0.0)
            context_guard_reason = self._indicator_confirmed_cross_context_guard_reason(
                Direction.SHORT,
                closes,
                rsi_values,
                k_values,
                d_values,
                slow_values,
                atr_values,
            )
            if context_guard_reason:
                return Signal(Direction.FLAT, 0.0, context_guard_reason, 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.SHORT, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            close_position_multiplier, close_position_reason = self._indicator_short_close_position_adjustment(candle)
            if close_position_multiplier <= 0:
                return Signal(Direction.FLAT, 0.0, close_position_reason or "indicator_short_blocked_high_close_position", 0.0, 0.0)
            short_stop_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.SHORT, "stop_loss_atr", self.config.strategy.stop_loss_atr),
                0.0008,
            )
            short_take_profit_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.SHORT, "take_profit_atr", self.config.strategy.take_profit_atr),
                short_stop_pct * 1.05,
            )
            short_size_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.SHORT,
                "size_multiplier",
                indicator_size_multiplier,
            )
            short_risk_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.SHORT,
                "confirmed_cross_risk_multiplier",
                config.confirmed_cross_risk_multiplier,
            )
            short_holding_bars = _indicator_side_holding_bars(self.config.strategy, Direction.SHORT)
            return Signal(
                Direction.SHORT,
                0.7,
                (
                    f"indicator_short_macd_dead_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                    + (f" {close_position_reason}" if close_position_reason else "")
                ),
                short_stop_pct,
                short_take_profit_pct,
                risk_multiplier=short_risk_multiplier * self.config.strategy.short_risk_bias * short_size_multiplier * close_position_multiplier,
                max_holding_bars=short_holding_bars,
            )
        if long_pre_cross:
            if self.config.strategy.long_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_long_disabled", 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.LONG, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            long_stop_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.LONG, "stop_loss_atr", self.config.strategy.stop_loss_atr),
                0.0008,
            )
            long_take_profit_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.LONG, "take_profit_atr", self.config.strategy.take_profit_atr),
                long_stop_pct * 1.05,
            )
            long_size_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.LONG,
                "size_multiplier",
                indicator_size_multiplier,
            )
            long_pre_cross_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.LONG,
                "pre_cross_risk_multiplier",
                config.pre_cross_risk_multiplier,
            )
            long_holding_bars = _indicator_side_holding_bars(self.config.strategy, Direction.LONG)
            return Signal(
                Direction.LONG,
                0.45,
                f"indicator_long_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                long_stop_pct,
                long_take_profit_pct,
                risk_multiplier=long_pre_cross_multiplier * self.config.strategy.long_risk_bias * long_size_multiplier,
                max_holding_bars=max(1, min(long_holding_bars, max(6, self.config.strategy.max_holding_bars // 2))),
            )
        if short_pre_cross:
            if self.config.strategy.short_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_short_disabled", 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.SHORT, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            close_position_multiplier, close_position_reason = self._indicator_short_close_position_adjustment(candle)
            if close_position_multiplier <= 0:
                return Signal(Direction.FLAT, 0.0, close_position_reason or "indicator_short_blocked_high_close_position", 0.0, 0.0)
            short_stop_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.SHORT, "stop_loss_atr", self.config.strategy.stop_loss_atr),
                0.0008,
            )
            short_take_profit_pct = max(
                atr_pct * _indicator_side_float(self.config.strategy, Direction.SHORT, "take_profit_atr", self.config.strategy.take_profit_atr),
                short_stop_pct * 1.05,
            )
            short_size_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.SHORT,
                "size_multiplier",
                indicator_size_multiplier,
            )
            short_pre_cross_multiplier = _indicator_side_float(
                self.config.strategy,
                Direction.SHORT,
                "pre_cross_risk_multiplier",
                config.pre_cross_risk_multiplier,
            )
            short_holding_bars = _indicator_side_holding_bars(self.config.strategy, Direction.SHORT)
            return Signal(
                Direction.SHORT,
                0.45,
                (
                    f"indicator_short_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                    + (f" {close_position_reason}" if close_position_reason else "")
                ),
                short_stop_pct,
                short_take_profit_pct,
                risk_multiplier=short_pre_cross_multiplier * self.config.strategy.short_risk_bias * short_size_multiplier * close_position_multiplier,
                max_holding_bars=max(1, min(short_holding_bars, max(6, self.config.strategy.max_holding_bars // 2))),
            )
        return Signal(
            Direction.FLAT,
            0.0,
            f"indicator_no_extreme rsi={current_rsi:.1f} macd={current_macd:.6g}/{current_signal:.6g} kdj={current_k:.1f}/{current_d:.1f}",
            0.0,
            0.0,
        )

    def _indicator_short_close_position_adjustment(self, candle: Candle) -> tuple[float, str | None]:
        threshold = float(getattr(self.config.strategy, "indicator_short_max_close_position", 0.0))
        if threshold <= 0:
            return 1.0, None
        candle_range = max(candle.high - candle.low, 1e-12)
        close_position = (candle.close - candle.low) / candle_range
        if close_position <= threshold:
            return 1.0, None
        multiplier = max(0.0, min(1.0, float(getattr(self.config.strategy, "indicator_short_high_close_risk_multiplier", 1.0))))
        reason = f"indicator_short_high_close_risk close_pos={close_position:.2f}>{threshold:.2f} mult={multiplier:.2f}"
        return multiplier, reason

    def _record_short_guard_reject(self, reason: str) -> None:
        key = reason.split()[0]
        self.short_guard_stats[key] = self.short_guard_stats.get(key, 0) + 1

    def _indicator_short_guard_reason(self, symbol: str, signal: Signal, signal_candles: list[Candle]) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_short_guard_enabled", False):
            return None
        if signal.direction != Direction.SHORT:
            return None
        reason = signal.reason.lower()
        if not reason.startswith("indicator_short_"):
            return None
        if "indicator_short_pre_cross" in reason:
            if getattr(strategy, "indicator_short_confirmed_only", False) or not getattr(strategy, "indicator_short_pre_cross_enabled", True):
                return "short_reject_pre_cross"
        if getattr(strategy, "indicator_short_confirmed_only", False) and "indicator_short_macd_dead_cross" not in reason:
            return "short_reject_not_confirmed_cross"

        if getattr(strategy, "indicator_short_btc_month_guard_enabled", False) and self._indicator_short_btc_month_bull_reason():
            return "short_reject_btc_bull_month"
        if getattr(strategy, "indicator_short_btc_guard_enabled", True) and self._indicator_short_btc_bull_reason():
            return "short_reject_btc_bull"
        if getattr(strategy, "indicator_short_require_btc_bear_enabled", False) and not self._indicator_short_btc_bear_confirmed():
            return "short_reject_btc_not_bear"

        if getattr(strategy, "indicator_short_funding_guard_enabled", False):
            funding_reason = self._indicator_short_funding_reject_reason(symbol, signal_candles[-1].timestamp)
            if funding_reason:
                return funding_reason

        candles_15m = self._safe_closed_candles_for_timeframe(symbol, "15m", 140)
        if getattr(strategy, "indicator_short_15m_guard_enabled", True):
            reason_15m = self._indicator_short_15m_bear_reject_reason(candles_15m)
            if reason_15m:
                return reason_15m

        candles_30m = self._safe_closed_candles_for_timeframe(symbol, "30m", 140)
        if getattr(strategy, "indicator_short_30m_guard_enabled", True):
            reason_30m = self._indicator_short_30m_bear_reject_reason(candles_30m)
            if reason_30m:
                return reason_30m

        overextended_reason = self._indicator_short_overextended_reject_reason(candles_15m, candles_30m)
        if overextended_reason:
            return overextended_reason

        retest_reason = self._indicator_short_retest_reject_reason(candles_15m)
        if retest_reason:
            return retest_reason
        return None

    def _safe_closed_candles_for_timeframe(self, symbol: str, timeframe: str, limit: int) -> list[Candle]:
        try:
            return self._closed_candles_for_timeframe(symbol, timeframe, limit)
        except Exception:
            return []

    def _indicator_short_btc_month_bull_reason(self) -> str | None:
        strategy = self.config.strategy
        btc_symbol = str(getattr(strategy, "btc_market_symbol", "BTCUSDT")).upper()
        timeframe = str(getattr(strategy, "indicator_short_btc_month_timeframe", "1h"))
        lookback = max(2, int(getattr(strategy, "indicator_short_btc_month_lookback_bars", 720)))
        ema_period = max(2, int(getattr(strategy, "indicator_short_btc_month_ema_period", 21)))
        candles = self._safe_closed_candles_for_timeframe(btc_symbol, timeframe, lookback + ema_period + 8)
        if len(candles) <= lookback + ema_period:
            return None
        closes = [item.close for item in candles]
        recent_return = closes[-1] / max(closes[-1 - lookback], 1e-12) - 1.0
        threshold = float(getattr(strategy, "indicator_short_btc_month_return_threshold", 0.04))
        ema_fast = ema(closes, 9)
        ema_slow = ema(closes, ema_period)
        slow_lookback = min(24, len(ema_slow) - 1)
        trend_bull = closes[-1] > ema_slow[-1] and ema_fast[-1] > ema_slow[-1] and ema_slow[-1] > ema_slow[-1 - slow_lookback]
        if recent_return >= threshold or trend_bull:
            return f"btc_month_bull return={recent_return * 100:.2f}%"
        return None

    def _indicator_short_btc_bull_reason(self) -> str | None:
        strategy = self.config.strategy
        btc_symbol = str(getattr(strategy, "btc_market_symbol", "BTCUSDT")).upper()
        fast_timeframe = str(getattr(strategy, "indicator_short_btc_fast_timeframe", "4h"))
        slow_timeframe = str(getattr(strategy, "indicator_short_btc_slow_timeframe", "8h"))
        candles_fast = self._safe_closed_candles_for_timeframe(btc_symbol, fast_timeframe, 80)
        candles_slow = self._safe_closed_candles_for_timeframe(btc_symbol, slow_timeframe, 80)
        if self._indicator_short_btc_frame_bullish(
            candles_fast,
            float(getattr(strategy, "indicator_short_btc_15m_bull_return_pct", 0.003)),
            int(getattr(strategy, "indicator_short_btc_breakout_lookback", 8)),
        ):
            return f"btc_{fast_timeframe}_bull"
        if self._indicator_short_btc_frame_bullish(
            candles_slow,
            float(getattr(strategy, "indicator_short_btc_1h_bull_return_pct", 0.006)),
            int(getattr(strategy, "indicator_short_btc_breakout_lookback", 8)),
        ):
            return f"btc_{slow_timeframe}_bull"
        return None

    def _indicator_short_btc_frame_bullish(self, candles: list[Candle], return_threshold: float, breakout_lookback: int) -> bool:
        if len(candles) < 35:
            return False
        closes = [item.close for item in candles]
        highs = [item.high for item in candles]
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        _, _, hist = macd(closes, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)
        current_return = closes[-1] / max(closes[-2], 1e-12) - 1.0
        ema_bull = closes[-1] > ema21[-1] and ema9[-1] > ema21[-1]
        macd_expanding = hist[-1] > 0 and hist[-1] > hist[-2]
        lookback = max(2, min(breakout_lookback, len(highs) - 1))
        breakout = closes[-1] > max(highs[-1 - lookback:-1])
        return current_return >= return_threshold or ema_bull or macd_expanding or breakout

    def _indicator_short_btc_bear_confirmed(self) -> bool:
        strategy = self.config.strategy
        btc_symbol = str(getattr(strategy, "btc_market_symbol", "BTCUSDT")).upper()
        fast_timeframe = str(getattr(strategy, "indicator_short_btc_fast_timeframe", "4h"))
        slow_timeframe = str(getattr(strategy, "indicator_short_btc_slow_timeframe", "8h"))
        candles_fast = self._safe_closed_candles_for_timeframe(btc_symbol, fast_timeframe, 80)
        candles_slow = self._safe_closed_candles_for_timeframe(btc_symbol, slow_timeframe, 80)
        threshold = float(getattr(strategy, "indicator_short_btc_bear_return_pct", -0.004))
        return (
            self._indicator_short_btc_frame_bearish(candles_fast, threshold)
            or self._indicator_short_btc_frame_bearish(candles_slow, threshold)
        )

    def _indicator_short_btc_frame_bearish(self, candles: list[Candle], return_threshold: float) -> bool:
        if len(candles) < 35:
            return False
        closes = [item.close for item in candles]
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        _, _, hist = macd(closes, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)
        current_return = closes[-1] / max(closes[-2], 1e-12) - 1.0
        ema_bear = closes[-1] < ema21[-1] and ema9[-1] < ema21[-1]
        macd_weak = hist[-1] < 0 and hist[-1] <= hist[-2]
        return current_return <= return_threshold or (ema_bear and macd_weak)

    def _indicator_short_funding_reject_reason(self, symbol: str, timestamp: datetime) -> str | None:
        features = self._mtf_aux_features().get(symbol, {})
        funding_rate = funding_at(features, timestamp, float(getattr(self.config.risk, "funding_default_rate", 0.0)))
        threshold = float(getattr(self.config.strategy, "indicator_short_min_funding_rate", -0.0002))
        if funding_rate < threshold:
            return f"short_reject_funding_too_negative funding={funding_rate:.6g}<{threshold:.6g}"
        return None

    def _indicator_short_15m_bear_reject_reason(self, candles: list[Candle]) -> str | None:
        if len(candles) < 110:
            return "short_reject_15m_not_bear insufficient_data"
        closes = [item.close for item in candles]
        highs = [item.high for item in candles]
        lows = [item.low for item in candles]
        ema21 = ema(closes, 21)
        ma7 = _sma_values(closes, 7)
        ma99 = _sma_values(closes, 99)
        confirmations = 0
        confirmations += int(closes[-1] < ema21[-1])
        confirmations += int(closes[-1] < ma99[-1])
        confirmations += int(ma7[-1] < ma99[-1])
        confirmations += int(ma7[-1] < ma7[-4])
        confirmations += int(highs[-1] < max(highs[-5:-1]))
        confirmations += int(closes[-1] < min(lows[-5:-1]))
        minimum = max(1, int(getattr(self.config.strategy, "indicator_short_15m_min_bear_confirmations", 2)))
        if confirmations < minimum:
            return f"short_reject_15m_not_bear confirmations={confirmations}<{minimum}"
        return None

    def _indicator_short_30m_bear_reject_reason(self, candles: list[Candle]) -> str | None:
        if len(candles) < 40:
            return "short_reject_30m_not_bear insufficient_data"
        closes = [item.close for item in candles]
        highs = [item.high for item in candles]
        ema21 = ema(closes, 21)
        rsi14 = rsi(closes, 14)
        _, _, hist = macd(closes, self.config.filters.macd_fast, self.config.filters.macd_slow, self.config.filters.macd_signal)
        confirmations = 0
        confirmations += int(hist[-1] < hist[-2])
        confirmations += int(hist[-1] < hist[-2] < hist[-3])
        confirmations += int(closes[-1] < ema21[-1])
        confirmations += int(rsi14[-1] < 50.0 or rsi14[-1] < rsi14[-2])
        confirmations += int(highs[-1] <= max(highs[-7:-1]))
        minimum = max(1, int(getattr(self.config.strategy, "indicator_short_30m_min_bear_confirmations", 2)))
        if confirmations < minimum:
            return f"short_reject_30m_not_bear confirmations={confirmations}<{minimum}"
        return None

    def _indicator_short_overextended_reject_reason(self, candles_15m: list[Candle], candles_30m: list[Candle]) -> str | None:
        strategy = self.config.strategy
        if len(candles_15m) < 110 or len(candles_30m) < 30:
            return None
        closes_15 = [item.close for item in candles_15m]
        ma7_15 = _sma_values(closes_15, 7)
        if ma7_15[-1] > 0 and (ma7_15[-1] - closes_15[-1]) / ma7_15[-1] > float(getattr(strategy, "indicator_short_max_distance_ma7_pct", 0.012)):
            return "short_reject_too_far_from_ma ma7"
        closes_30 = [item.close for item in candles_30m]
        ma25_30 = _sma_values(closes_30, 25)
        if ma25_30[-1] > 0 and (ma25_30[-1] - closes_30[-1]) / ma25_30[-1] > float(getattr(strategy, "indicator_short_max_distance_30m_ma25_pct", 0.018)):
            return "short_reject_too_far_from_ma ma25_30m"
        if rsi(closes_15, 6)[-1] <= float(getattr(strategy, "indicator_short_15m_rsi6_floor", 24.0)):
            return "short_reject_rsi_oversold 15m"
        if rsi(closes_30, 6)[-1] <= float(getattr(strategy, "indicator_short_30m_rsi6_floor", 26.0)):
            return "short_reject_rsi_oversold 30m"
        red_bars = max(1, int(getattr(strategy, "indicator_short_consecutive_red_bars", 3)))
        recent = candles_15m[-red_bars:]
        touched_retest = any(item.high >= ma7_15[-1] * (1.0 - float(getattr(strategy, "indicator_short_retest_touch_pct", 0.003))) for item in recent)
        if all(item.close < item.open for item in recent) and not touched_retest:
            return "short_reject_consecutive_red_no_retest"
        candle = candles_15m[-1]
        candle_range = max(candle.high - candle.low, 1e-12)
        lower_wick = (min(candle.open, candle.close) - candle.low) / candle_range
        if lower_wick >= float(getattr(strategy, "indicator_short_lower_wick_ratio", 0.45)):
            return "short_reject_lower_wick_support"
        support_lookback = max(2, int(getattr(strategy, "indicator_short_support_lookback", 8)))
        support = min(item.low for item in candles_15m[-support_lookback - 1:-1])
        if candle.close <= support * (1.0 + float(getattr(strategy, "indicator_short_near_support_pct", 0.003))):
            return "short_reject_near_support"
        return None

    def _indicator_short_retest_reject_reason(self, candles: list[Candle]) -> str | None:
        if not bool(getattr(self.config.strategy, "indicator_short_retest_guard_enabled", True)):
            return None
        if len(candles) < 110:
            return "short_waiting_retest_reject insufficient_data"
        mode = str(getattr(self.config.strategy, "indicator_short_ma_mode", "retest_after_cross")).lower()
        closes = [item.close for item in candles]
        highs = [item.high for item in candles]
        lows = [item.low for item in candles]
        ma7 = _sma_values(closes, 7)
        ma99 = _sma_values(closes, 99)
        candle = candles[-1]
        touch_pct = float(getattr(self.config.strategy, "indicator_short_retest_touch_pct", 0.003))
        close_pos = (candle.close - candle.low) / max(candle.high - candle.low, 1e-12)
        weak_close = close_pos <= float(getattr(self.config.strategy, "indicator_short_max_close_position", 0.50))
        ma7_reject = candle.high >= ma7[-1] * (1.0 - touch_pct) and candle.close < ma7[-1] and candle.close < candle.open and weak_close
        ma99_reject = candle.high >= ma99[-1] * (1.0 - touch_pct) and candle.close < ma99[-1] and ma7[-1] < ma99[-1] and weak_close
        lower_high_breakdown = highs[-1] < max(highs[-5:-1]) and closes[-1] < min(lows[-5:-1])
        if mode == "slope_only":
            if closes[-1] < ma99[-1] and ma7[-1] < ma7[-4]:
                return None
        elif mode == "cross_confirmed":
            prior_cross = any(ma7[index] >= ma99[index] for index in range(max(0, len(ma7) - 14), len(ma7) - 1))
            if ma7[-1] < ma99[-1] and prior_cross and closes[-1] < ma7[-1] and closes[-1] < ma99[-1]:
                return None
        else:
            if ma7[-1] < ma99[-1] and (ma7_reject or ma99_reject or lower_high_breakdown):
                return None
        return "short_waiting_retest_reject"

    def _manage_existing_position(self, symbol: str, position: LivePosition, account: AccountSnapshot) -> None:
        candles = self._closed_candles(symbol)
        if len(candles) < VolatilityBreakoutScalper(self.config.strategy).warmup_bars:
            return
        recent_1m_candles = self._live_recent_1m_candles(symbol)
        exit_candle = self._live_exit_candle(symbol, candles, position, recent_1m_candles)
        exit_mark_price = exit_candle.close
        profit_state = self._profit_state_for(symbol, position)
        self._update_profit_state_with_candle(position, profit_state, exit_candle)
        strategy = VolatilityBreakoutScalper(self.config.strategy)
        strategy.prepare(candles)
        signal = strategy.signal(len(candles) - 1, candles)
        managed_exit_allowed = self._managed_exit_allowed(position)
        if _is_vbp_entry_reason(str(getattr(position, "entry_reason", ""))):
            vbp_exit_reason = self._vbp_live_exit_reason(position, exit_candle, recent_1m_candles, profit_state)
            if vbp_exit_reason:
                self._log_position_exit_diagnostics(
                    symbol,
                    position,
                    profit_state,
                    exit_mark_price,
                    profit_exit_reason=vbp_exit_reason,
                    stale_reason=None,
                    managed_exit_allowed=True,
                )
                self.log(f"{symbol}: VBP动态退出触发，准备平仓 ({vbp_exit_reason})")
                self._exit_position(symbol, position, vbp_exit_reason)
                return
        if not managed_exit_allowed:
            self._log_position_exit_diagnostics(
                symbol,
                position,
                profit_state,
                exit_mark_price,
                profit_exit_reason=None,
                stale_reason=None,
                managed_exit_allowed=False,
            )
            return
        profit_exit_reason = self._profit_exit_reason(position, candles, current_candle=exit_candle, state=profit_state)
        stale_reason = self._stale_position_exit_reason(position, exit_mark_price)
        self._log_position_exit_diagnostics(
            symbol,
            position,
            profit_state,
            exit_mark_price,
            profit_exit_reason=profit_exit_reason,
            stale_reason=stale_reason,
            managed_exit_allowed=True,
        )
        if profit_exit_reason:
            self.log(f"{symbol}: 盈利保护触发，准备平仓 ({profit_exit_reason})")
            self._exit_position(symbol, position, profit_exit_reason)
            return
        account_loss_reason = self._account_loss_exit_reason(position, exit_mark_price, account)
        if account_loss_reason:
            self.log(f"{symbol}: 单仓账户亏损保护触发，准备平仓 ({account_loss_reason})")
            self._exit_position(symbol, position, account_loss_reason)
            return
        fail_fast_reason = self._indicator_long_fail_fast_exit_reason(position, exit_mark_price)
        if fail_fast_reason:
            self.log(f"{symbol}: 指标多头快速失败退出，准备平仓 ({fail_fast_reason})")
            self._exit_position(symbol, position, fail_fast_reason)
            return
        trend_loss_reason = self._trend_loss_exit_reason(position, exit_mark_price)
        if trend_loss_reason:
            self.log(f"{symbol}: 趋势单亏损保护触发，准备平仓 ({trend_loss_reason})")
            self._exit_position(symbol, position, trend_loss_reason)
            return
        if stale_reason:
            self.log(f"{symbol}: 长时间持仓观察退出，准备平仓 ({stale_reason})")
            self._exit_position(symbol, position, stale_reason)
            return
        false_position_reason = self._false_breakout_reason(position.direction, candles)
        if false_position_reason and _position_profit_pct(position, exit_mark_price) < 0:
            self.log(f"{symbol}: 当前持仓方向疑似假突破，准备平仓 ({false_position_reason})")
            self._exit_position(symbol, position, false_position_reason)
            return
        if signal.direction != Direction.FLAT and signal.direction != position.direction:
            position_supported, support_reason = self._passes_multi_timeframe_filter(symbol, position.direction)
            false_opposite_reason = self._false_breakout_reason(signal.direction, candles)
            if false_opposite_reason:
                self.log(
                    f"{symbol}: 5m 反向信号疑似假突破，继续按 {position.direction.name} 管理 "
                    f"({false_opposite_reason})"
                )
                self._maybe_loss_scale_on_supported_direction(
                    symbol,
                    position,
                    account,
                    candles,
                    f"反向假突破 {false_opposite_reason}",
                    position_supported,
                    support_reason,
                )
                return
            if position_supported:
                self.log(f"{symbol}: 5m 反向信号但大周期仍支持 {position.direction.name}，暂不平仓 ({support_reason})")
                self._maybe_loss_scale_on_supported_direction(
                    symbol,
                    position,
                    account,
                    candles,
                    f"5m临时反向 {signal.reason}",
                    position_supported,
                    support_reason,
                )
                return
            self.log(f"{symbol}: 5m 反向信号且大周期不再支持，准备市价平仓 ({support_reason})")
            self._exit_position(symbol, position, "reverse_signal_confirmed")
            return
        if signal.direction == position.direction:
            self._maybe_scale_in_position(symbol, position, account, signal, candles)
            return
        if _position_profit_pct(position, exit_mark_price) >= self.config.trading.scale_in_min_profit_pct:
            supported, support_reason = self._passes_multi_timeframe_filter(symbol, position.direction)
            if supported:
                continuation = self._continuation_signal(
                    position.direction,
                    candles,
                    f"profit_continuation flat_signal={signal.reason}; mtf={support_reason}",
                )
                self._maybe_scale_in_position(symbol, position, account, continuation, candles)
                return
        self._maybe_loss_scale_on_supported_direction(symbol, position, account, candles, "5m无同向突破")

    def _live_recent_1m_candles(self, symbol: str, limit: int = 32) -> list[Candle]:
        try:
            candles = self.client.klines(symbol, "1m", max(3, limit))
        except Exception:
            return []
        if len(candles) > 1:
            return candles[:-1]
        return candles

    def _live_exit_candle(
        self,
        symbol: str,
        candles: list[Candle],
        position: LivePosition,
        recent_1m_candles: list[Candle] | None = None,
    ) -> Candle:
        base = candles[-1]
        mark_price = float(getattr(position, "mark_price", 0.0) or 0.0)
        latest_1m = recent_1m_candles[-1] if recent_1m_candles else None

        close = latest_1m.close if latest_1m is not None else (mark_price if mark_price > 0 else base.close)
        high = max(base.high, close)
        low = min(base.low, close)
        volume = base.volume
        if latest_1m is not None:
            high = max(high, latest_1m.high)
            low = min(low, latest_1m.low)
            volume = max(volume, latest_1m.volume)
        return replace(base, high=high, low=low, close=close, volume=volume)

    def _update_profit_state_with_candle(self, position: LivePosition, state: ProfitState, candle: Candle) -> None:
        if position.direction == Direction.LONG:
            state.best_price = max(state.best_price, candle.high)
        elif position.direction == Direction.SHORT:
            state.best_price = min(state.best_price, candle.low)

    def _vbp_live_exit_reason(
        self,
        position: LivePosition,
        candle: Candle,
        recent_1m_candles: list[Candle],
        state: ProfitState,
    ) -> str | None:
        if position.direction != Direction.LONG:
            return None
        if not _is_vbp_entry_reason(str(getattr(position, "entry_reason", ""))):
            return None
        exit_config = self.config.vbp_strategy.exit
        current_profit = _directional_profit_pct(position.direction, position.entry_price, candle.close)
        peak_profit = _directional_profit_pct(position.direction, position.entry_price, state.best_price)
        pullback = max(0.0, (state.best_price - candle.close) / max(position.entry_price, 1e-12))

        if bool(getattr(exit_config, "peak_giveback_enabled", True)):
            trigger = max(0.0, float(getattr(exit_config, "peak_giveback_trigger_pct", 0.008)))
            floor = float(getattr(exit_config, "peak_giveback_floor_pct", 0.0015))
            retrace = max(0.0, float(getattr(exit_config, "peak_giveback_retrace_pct", 0.005)))
            if peak_profit >= trigger and (current_profit <= floor or pullback >= retrace):
                return (
                    f"vbp_peak_giveback peak={peak_profit * 100:.3f}% "
                    f"now={current_profit * 100:.3f}% pullback={pullback * 100:.3f}%"
                )

        if bool(getattr(exit_config, "large_bear_exit_enabled", True)):
            min_peak = float(getattr(exit_config, "large_bear_min_peak_profit_pct", 0.006))
            min_current = float(getattr(exit_config, "large_bear_min_current_profit_pct", -0.001))
            if peak_profit >= min_peak and current_profit >= min_current:
                bear_reason = _vbp_large_bearish_candle_reason(recent_1m_candles, exit_config)
                if bear_reason:
                    return (
                        f"vbp_high_volume_bear_exit {bear_reason} "
                        f"peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}%"
                    )
        return None

    def _log_position_exit_diagnostics(
        self,
        symbol: str,
        position: LivePosition,
        state: ProfitState,
        mark_price: float,
        profit_exit_reason: str | None,
        stale_reason: str | None,
        managed_exit_allowed: bool,
    ) -> None:
        interval = max(5.0, float(getattr(self.config.trading, "position_diagnostics_interval_seconds", 60.0)))
        now = time.time()
        if now - self._last_position_diagnostics_log_ts.get(symbol, 0.0) < interval:
            return
        self._last_position_diagnostics_log_ts[symbol] = now

        bars_held = self._position_bars_held(position)
        current_profit = _directional_profit_pct(position.direction, position.entry_price, mark_price)
        peak_profit = _directional_profit_pct(position.direction, position.entry_price, state.best_price)
        minimum_profit = self._minimum_profit_exit_pct()
        stale_enabled = bool(getattr(self.config.strategy, "stale_position_exit_enabled", False))
        observation_bars, force_exit_bars = self._stale_exit_thresholds(position)
        if not managed_exit_allowed:
            profit_status = "blocked managed_exit_min"
            stale_status = f"blocked managed_exit_min bars={bars_held}"
        else:
            profit_status = f"YES {profit_exit_reason}" if profit_exit_reason else "NO"
            if not stale_enabled:
                stale_status = "disabled"
            elif stale_reason:
                stale_status = f"YES {stale_reason}"
            else:
                stale_status = f"NO obs={observation_bars} force={force_exit_bars}"

        self.log(
            f"{symbol}: 持仓退出诊断 bars={bars_held} mark={mark_price:.6g} "
            f"profit={current_profit * 100:+.3f}% peak={peak_profit * 100:+.3f}% "
            f"min_exit={minimum_profit * 100:.3f}% profit_exit={profit_status} stale_exit={stale_status}"
        )

    def _maybe_scale_in_position(
        self,
        symbol: str,
        position: LivePosition,
        account: AccountSnapshot,
        signal: Signal,
        candles: list[Candle],
    ) -> None:
        candle = candles[-1]
        trading = self.config.trading
        if trading.max_scale_ins_per_symbol <= 0 or trading.scale_in_entry_fraction <= 0:
            return

        scale_count = self._scale_in_counts.get(symbol, 0)
        if scale_count >= trading.max_scale_ins_per_symbol:
            return

        now = time.time()
        last_scale_in = self._last_scale_in_ts.get(symbol, 0.0)
        if now - last_scale_in < trading.scale_in_cooldown_seconds:
            return

        profit_pct = _position_profit_pct(position, candle.close)
        signal_quality = max(0.25, min(1.0, signal.risk_multiplier))
        if profit_pct >= trading.scale_in_min_profit_pct:
            entry_fraction = _scale_fraction(trading.scale_in_entry_fraction, scale_count) * signal_quality
            scale_label = f"顺势浮盈/质量{signal_quality:.2f}"
        elif trading.allow_loss_scale_in and profit_pct <= -trading.loss_scale_in_trigger_pct:
            false_reason = self._false_breakout_reason(signal.direction, candles)
            if false_reason:
                self.log(f"{symbol}: 暂不亏损补仓，当前方向疑似假突破 ({false_reason})")
                return
            entry_fraction = trading.loss_scale_in_entry_fraction * min(0.7, signal_quality)
            scale_label = f"受限亏损/质量{signal_quality:.2f}"
        else:
            self.log(
                f"{symbol}: 同向信号但暂不补仓 "
                f"(浮动{profit_pct * 100:+.3f}%，未达到浮盈或亏损补仓阈值)"
            )
            return

        confirmed, confirmation_reason = self._scale_in_confirmation_reason(signal.direction, candles, loss_scale=profit_pct < 0)
        if not confirmed:
            self.log(f"{symbol}: 同向信号但指标未确认补仓 ({confirmation_reason})")
            return

        allowed, filter_reason = self._passes_multi_timeframe_filter(symbol, signal.direction)
        if not allowed:
            self.log(f"{symbol}: 同向信号但多周期过滤拒绝补仓 ({filter_reason})")
            return

        quantity, reason = self._size_order(
            symbol,
            candle.close,
            signal,
            account,
            existing_position=position,
            entry_fraction=entry_fraction,
        )
        if float(quantity) <= 0:
            self.log(f"{symbol}: 跳过补仓 ({reason})")
            return

        self._enter_position(symbol, signal, candle, quantity, scale_in=True, scale_label=scale_label)
        self._scale_in_counts[symbol] = scale_count + 1
        self._last_scale_in_ts[symbol] = now

    def _maybe_loss_scale_on_supported_direction(
        self,
        symbol: str,
        position: LivePosition,
        account: AccountSnapshot,
        candles: list[Candle],
        context: str,
        supported: bool | None = None,
        support_reason: str = "",
    ) -> None:
        trading = self.config.trading
        if not trading.allow_loss_scale_in:
            return
        if not candles:
            return
        profit_pct = _position_profit_pct(position, candles[-1].close)
        if profit_pct > -trading.loss_scale_in_trigger_pct:
            return
        false_reason = self._false_breakout_reason(position.direction, candles)
        if false_reason:
            self.log(f"{symbol}: 亏损但不补仓，持仓方向疑似假突破 ({false_reason})")
            return
        if supported is None:
            supported, support_reason = self._passes_multi_timeframe_filter(symbol, position.direction)
        if not supported:
            self.log(f"{symbol}: 亏损但不补仓，大周期不支持 {position.direction.name} ({support_reason})")
            return
        signal = self._continuation_signal(position.direction, candles, f"loss_scale {context}; mtf={support_reason}")
        self._maybe_scale_in_position(symbol, position, account, signal, candles)

    def _continuation_signal(self, direction: Direction, candles: list[Candle], reason: str) -> Signal:
        candle = candles[-1]
        atr_values = atr(candles, self.config.strategy.atr_period)
        atr_value = atr_values[-1] if atr_values else candle.close * 0.001
        atr_pct = max(atr_value / max(candle.close, 1e-12), 0.0008)
        stop_pct = max(atr_pct * self.config.strategy.stop_loss_atr, 0.0008)
        take_profit_pct = max(atr_pct * self.config.strategy.take_profit_atr, stop_pct * 1.05)
        return Signal(
            direction,
            0.55,
            reason,
            stop_pct,
            take_profit_pct,
            risk_multiplier=self.config.filters.confirmed_cross_risk_multiplier,
            max_holding_bars=self.config.strategy.max_holding_bars,
        )

    def _false_breakout_reason(self, direction: Direction, candles: list[Candle]) -> str | None:
        config = self.config.strategy
        minimum = max(config.channel_period, config.atr_period) + 2
        if direction == Direction.FLAT or len(candles) < minimum:
            return None
        index = len(candles) - 1
        candle = candles[-1]
        highs = [item.high for item in candles]
        lows = [item.low for item in candles]
        atr_values = atr(candles, config.atr_period)
        atr_value = atr_values[index - 1] if index > 0 else atr_values[index]
        if atr_value <= 0:
            return None
        upper_channel = rolling_high(highs, index, config.channel_period)
        lower_channel = rolling_low(lows, index, config.channel_period)
        buffer = atr_value * max(config.breakout_buffer_atr, 0.05)
        body = abs(candle.close - candle.open)

        if direction == Direction.LONG:
            upper_wick = candle.high - max(candle.open, candle.close)
            broke_high = candle.high > upper_channel + buffer
            failed_close = candle.close < upper_channel - buffer * 0.25
            if broke_high and failed_close and (candle.close < candle.open or upper_wick > body):
                return f"false_long_breakout close={candle.close:.6g} upper={upper_channel:.6g}"
        elif direction == Direction.SHORT:
            lower_wick = min(candle.open, candle.close) - candle.low
            broke_low = candle.low < lower_channel - buffer
            failed_close = candle.close > lower_channel + buffer * 0.25
            if broke_low and failed_close and (candle.close > candle.open or lower_wick > body):
                return f"false_short_breakdown close={candle.close:.6g} lower={lower_channel:.6g}"
        return None

    def _manage_sim_positions(self) -> None:
        for symbol, position in list(self._sim_positions.items()):
            candles = self._closed_candles(symbol)
            new_candles = [candle for candle in candles if candle.timestamp > position.last_checked_time]
            if not new_candles:
                continue
            for candle in new_candles:
                position.bars_held += 1
                position.last_checked_time = candle.timestamp
                exit_price = None
                reason = ""
                if position.direction == Direction.LONG:
                    if candle.low <= position.stop_price:
                        exit_price = position.stop_price
                        reason = "stop_loss"
                    elif candle.high >= position.take_profit_price:
                        exit_price = position.take_profit_price
                        reason = "take_profit"
                else:
                    if candle.high >= position.stop_price:
                        exit_price = position.stop_price
                        reason = "stop_loss"
                    elif candle.low <= position.take_profit_price:
                        exit_price = position.take_profit_price
                        reason = "take_profit"

                if exit_price is None:
                    self._update_sim_profit_protection(position, candle)
                    if self._managed_exit_allowed(position):
                        profit_reason = self._profit_exit_reason(position, candles, current_candle=candle)
                        if profit_reason:
                            exit_price = candle.close
                            reason = profit_reason
                        else:
                            fail_fast_reason = self._indicator_long_fail_fast_exit_reason(position, candle.close)
                            if fail_fast_reason:
                                exit_price = candle.close
                                reason = fail_fast_reason
                            else:
                                trend_loss_reason = self._trend_loss_exit_reason(position, candle.close)
                                if trend_loss_reason:
                                    exit_price = candle.close
                                    reason = trend_loss_reason
                                else:
                                    stale_reason = self._stale_position_exit_reason(position, candle.close)
                                    if stale_reason:
                                        exit_price = candle.close
                                        reason = stale_reason

                if exit_price is None and position.max_holding_bars > 0 and position.bars_held >= position.max_holding_bars:
                    exit_price = candle.close
                    reason = "time_stop"

                if exit_price is not None:
                    self._close_sim_position(symbol, exit_price, reason)
                    break

    def _enter_position(
        self,
        symbol: str,
        signal: Signal,
        candle: Candle,
        quantity: str,
        scale_in: bool = False,
        scale_label: str = "",
        leverage_override: int | None = None,
    ) -> None:
        side = "BUY" if signal.direction == Direction.LONG else "SELL"
        leverage = _effective_entry_leverage(self.config, symbol, signal, leverage_override)
        notional = float(quantity) * candle.close
        margin = notional / leverage
        action = f"补仓({scale_label})" if scale_in and scale_label else "补仓" if scale_in else "开仓"
        opened_at = datetime.now(timezone.utc)
        self.log(
            f"{symbol}: {action} {signal.direction.name} 仓位≈{notional:.2f}U "
            f"倍率={leverage}x 保证金≈{margin:.2f}U "
            f"qty={quantity} reason={signal.reason}"
        )
        if self.config.trading.dry_run:
            qty = float(quantity)
            if signal.direction == Direction.LONG:
                stop = candle.close * (1.0 - signal.stop_loss_pct)
                take_profit = candle.close * (1.0 + signal.take_profit_pct)
            else:
                stop = candle.close * (1.0 + signal.stop_loss_pct)
                take_profit = candle.close * (1.0 - signal.take_profit_pct)
            existing = self._sim_positions.get(symbol)
            if scale_in and existing and existing.direction == signal.direction:
                total_qty = existing.quantity + qty
                if total_qty <= 0:
                    return
                avg_entry = (existing.entry_price * existing.quantity + candle.close * qty) / total_qty
                if signal.direction == Direction.LONG:
                    merged_stop = max(existing.stop_price, stop)
                    merged_take_profit = min(existing.take_profit_price, take_profit)
                else:
                    merged_stop = min(existing.stop_price, stop)
                    merged_take_profit = max(existing.take_profit_price, take_profit)
                existing.quantity = total_qty
                existing.entry_price = avg_entry
                existing.stop_price = merged_stop
                existing.take_profit_price = merged_take_profit
                existing.scale_ins += 1
                existing.last_checked_time = max(
                    existing.last_checked_time,
                    self._latest_closed_candle_timestamp(symbol, candle.timestamp),
                )
                if signal.direction == Direction.LONG:
                    existing.best_price = max(existing.best_price, candle.close)
                else:
                    existing.best_price = min(existing.best_price, candle.close)
                self.log(
                    f"{symbol}: dry-run 已合并虚拟补仓 avg={avg_entry:.6g} "
                    f"qty={total_qty:.6g} stop={merged_stop:.6g} take_profit={merged_take_profit:.6g}"
                )
                self._known_active_symbols.add(symbol)
                return
            self._sim_positions[symbol] = SimPosition(
                symbol=symbol,
                direction=signal.direction,
                quantity=qty,
                entry_price=candle.close,
                stop_price=stop,
                take_profit_price=take_profit,
                max_holding_bars=signal.max_holding_bars or self.config.strategy.max_holding_bars,
                entry_time=opened_at,
                last_checked_time=self._latest_closed_candle_timestamp(symbol, candle.timestamp),
                best_price=candle.close,
                leverage=leverage,
                entry_reason=signal.reason,
            )
            self._entry_reasons[symbol] = signal.reason
            self._position_opened_at[symbol] = opened_at
            self._known_active_symbols.add(symbol)
            self.log(f"{symbol}: dry-run 已记录虚拟仓 stop={stop:.6g} take_profit={take_profit:.6g}")
            return

        if not self._prepare_symbol(symbol, leverage_override=leverage_override, signal=signal):
            self.log(f"{symbol}: 跳过下单，当前杠杆/保证金模式不支持")
            return
        if not scale_in:
            self._cancel_all_symbol_orders(symbol)
        response = self.client.new_market_order(symbol, side, quantity, reduce_only=False)
        entry_price = self._entry_price_from_response(response, candle.close)
        self._known_active_symbols.add(symbol)
        if not scale_in:
            self._entry_reasons[symbol] = signal.reason
            self._position_opened_at[symbol] = opened_at
        if self.config.trading.use_protective_orders:
            self._place_protective_orders(symbol, signal, quantity, entry_price)

    def _exit_position(self, symbol: str, position: LivePosition, reason: str = "strategy_exit") -> None:
        side = "SELL" if position.direction == Direction.LONG else "BUY"
        rules = self.client.symbol_rules(symbol)
        quantity = rules.round_quantity(position.quantity)
        if float(quantity) <= 0:
            self.log(f"{symbol}: 平仓数量无效")
            return
        if self.config.trading.dry_run:
            self._close_sim_position(symbol, position.mark_price, reason)
            return
        self._cancel_all_symbol_orders(symbol)
        self.client.new_market_order(symbol, side, quantity, reduce_only=self.config.trading.reduce_only_exit)
        self._cancel_all_symbol_orders(symbol)
        self._mark_symbol_reentry_cooldown(symbol, reason)
        entry_reason = position.entry_reason or self._entry_reasons.get(symbol, "")
        self._mark_portfolio_symbol_cooldown(symbol, position.unrealized_pnl)
        self._record_vbp_live_result(position.unrealized_pnl, entry_reason)
        if not self._sync_closed_symbol_trade_pnl(symbol, reason, entry_reason=entry_reason):
            self._record_indicator_reversal_result(position.unrealized_pnl, entry_reason)
        self._scale_in_counts.pop(symbol, None)
        self._last_scale_in_ts.pop(symbol, None)
        self._known_active_symbols.discard(symbol)
        self._profit_states.pop(symbol, None)
        self._entry_reasons.pop(symbol, None)
        self._position_opened_at.pop(symbol, None)
        self.log(f"{symbol}: 已发送 reduce-only 市价平仓 reason={reason}")

    def _close_sim_position(self, symbol: str, exit_price: float, reason: str) -> None:
        position = self._sim_positions.pop(symbol, None)
        if not position:
            return
        pnl = position.direction.value * position.quantity * (exit_price - position.entry_price)
        self.stats.closed_trades += 1
        if pnl > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1
        self.stats.realized_pnl += pnl
        self._record_indicator_reversal_result(pnl, position.entry_reason)
        self._record_vbp_live_result(pnl, position.entry_reason)
        self._scale_in_counts.pop(symbol, None)
        self._last_scale_in_ts.pop(symbol, None)
        self._profit_states.pop(symbol, None)
        self._entry_reasons.pop(symbol, None)
        self._position_opened_at.pop(symbol, None)
        self._mark_symbol_reentry_cooldown(symbol, reason)
        self._mark_portfolio_symbol_cooldown(symbol, pnl)
        self._known_active_symbols.discard(symbol)
        self.log(f"{symbol}: dry-run 虚拟平仓 exit={exit_price:.6g} pnl={pnl:+.4f}U reason={reason}")
        self._log_session_stats(self.snapshot_account(), force=True)

    def _mark_portfolio_symbol_cooldown(self, symbol: str, pnl: float) -> None:
        control = self.config.portfolio_control
        if not getattr(control, "enabled", False):
            return
        minutes = max(0, int(control.symbol_cooldown_minutes))
        if pnl < 0:
            minutes = max(minutes, int(control.symbol_loss_cooldown_minutes))
        if minutes <= 0:
            return
        self._portfolio_symbol_cooldown_until[symbol] = max(
            self._portfolio_symbol_cooldown_until.get(symbol, 0.0),
            time.time() + minutes * 60.0,
        )

    def _record_vbp_live_result(self, pnl: float, entry_reason: str | None) -> None:
        if _portfolio_bucket_from_reason(entry_reason or "") != "vbp":
            return
        if pnl < 0:
            self._vbp_live_loss_streak += 1
            risk = self.config.vbp_strategy.risk_control
            if bool(getattr(risk, "consecutive_loss_reduce_enabled", False)):
                losses = max(1, int(getattr(risk, "consecutive_loss_reduce_losses", 1)))
                if self._vbp_live_loss_streak >= losses:
                    minutes = max(1, int(getattr(risk, "consecutive_loss_reduce_minutes", 1440)))
                    self._vbp_live_loss_reduce_until = max(self._vbp_live_loss_reduce_until, time.time() + minutes * 60.0)
        else:
            self._vbp_live_loss_streak = 0

    def _sync_closed_symbol_trade_pnl(self, symbol: str, reason: str, entry_reason: str | None = None) -> bool:
        if self.config.trading.dry_run:
            return False
        if not hasattr(self.client, "user_trades"):
            return False
        since_ms = int(self.stats.started_at.timestamp() * 1000) - 60_000
        try:
            trades = self.client.user_trades(symbol=symbol, limit=1000, start_time=max(0, since_ms))
        except BinanceApiError as exc:
            self.log(f"{symbol}: 同步平仓成交盈亏失败 ({exc})")
            return False

        grouped: dict[str, float] = {}
        commissions: dict[str, float] = {}
        for trade in trades:
            realized = _float_value(trade.get("realizedPnl"))
            commission = _float_value(trade.get("commission"))
            trade_id = f"{symbol}:{trade.get('id', '')}:{trade.get('orderId', '')}:{trade.get('time', '')}"
            if trade_id in self._accounted_trade_ids or abs(realized) <= 0:
                continue
            self._accounted_trade_ids.add(trade_id)
            order_id = str(trade.get("orderId", trade.get("id", "")))
            grouped[order_id] = grouped.get(order_id, 0.0) + realized
            if str(trade.get("commissionAsset", "")).upper() == "USDT":
                commissions[order_id] = commissions.get(order_id, 0.0) + commission

        if not grouped:
            return False

        total_net = 0.0
        closed_count = 0
        wins = 0
        losses = 0
        for order_id, gross_pnl in grouped.items():
            net_pnl = gross_pnl - commissions.get(order_id, 0.0)
            total_net += net_pnl
            closed_count += 1
            if net_pnl > 0:
                wins += 1
            else:
                losses += 1
            if entry_reason:
                self._record_indicator_reversal_result(net_pnl, entry_reason)

        self.stats.closed_trades += closed_count
        self.stats.winning_trades += wins
        self.stats.losing_trades += losses
        self.stats.realized_pnl += total_net
        self.log(
            f"{symbol}: 已同步交易所平仓盈亏 {total_net:+.4f}U "
            f"平仓订单={closed_count} reason={reason}"
        )
        return True

    def _place_protective_orders(self, symbol: str, signal: Signal, quantity: str, entry_price: float) -> None:
        rules = self.client.symbol_rules(symbol)
        if signal.direction == Direction.LONG:
            exit_side = "SELL"
            stop_price = entry_price * (1.0 - signal.stop_loss_pct)
            take_profit_price = entry_price * (1.0 + signal.take_profit_pct)
        else:
            exit_side = "BUY"
            stop_price = entry_price * (1.0 + signal.stop_loss_pct)
            take_profit_price = entry_price * (1.0 - signal.take_profit_pct)

        rounded_stop = rules.round_price(stop_price)
        rounded_take_profit = rules.round_price(take_profit_price)
        try:
            self.client.new_stop_market_order(symbol, exit_side, rounded_stop, quantity, reduce_only=True, working_type=self.config.trading.working_type)
            self.client.new_take_profit_market_order(symbol, exit_side, rounded_take_profit, quantity, reduce_only=True, working_type=self.config.trading.working_type)
            self.log(f"{symbol}: 已挂保护单 stop={rounded_stop} take_profit={rounded_take_profit}")
        except BinanceApiError:
            self.log(f"{symbol}: 保护单失败，尝试撤单并市价平仓")
            self._cancel_all_symbol_orders(symbol)
            self.client.new_market_order(symbol, exit_side, quantity, reduce_only=True)
            self._cancel_all_symbol_orders(symbol)
            raise

    def _cancel_all_symbol_orders(self, symbol: str) -> None:
        for cancel in (self.client.cancel_all_open_orders, self.client.cancel_all_algo_open_orders):
            try:
                cancel(symbol)
            except BinanceApiError as exc:
                self.log(f"{symbol}: 撤单失败，继续处理 ({exc})")

    def _profit_state_for(self, symbol: str, position: LivePosition) -> ProfitState:
        state = self._profit_states.get(symbol)
        if state is None or state.direction != position.direction or abs(state.entry_price - position.entry_price) > 1e-12:
            state = ProfitState(position.direction, position.entry_price, position.entry_price)
            self._profit_states[symbol] = state
        return state

    def _update_sim_profit_protection(self, position: SimPosition, candle: Candle) -> None:
        if not self.config.trading.profit_exit_enabled:
            return
        previous_stop = position.stop_price
        breakeven_lock_pct = max(self.config.trading.breakeven_lock_pct, self._minimum_profit_exit_pct())
        if position.direction == Direction.LONG:
            position.best_price = max(position.best_price, candle.high)
            peak_profit = _directional_profit_pct(position.direction, position.entry_price, position.best_price)
            if peak_profit >= self.config.trading.breakeven_trigger_pct:
                position.stop_price = max(
                    position.stop_price,
                    position.entry_price * (1.0 + breakeven_lock_pct),
                )
            if peak_profit >= self.config.trading.trailing_activation_pct:
                position.stop_price = max(
                    position.stop_price,
                    position.best_price * (1.0 - self.config.trading.trailing_pullback_pct),
                )
        else:
            position.best_price = min(position.best_price, candle.low)
            peak_profit = _directional_profit_pct(position.direction, position.entry_price, position.best_price)
            if peak_profit >= self.config.trading.breakeven_trigger_pct:
                position.stop_price = min(
                    position.stop_price,
                    position.entry_price * (1.0 - breakeven_lock_pct),
                )
            if peak_profit >= self.config.trading.trailing_activation_pct:
                position.stop_price = min(
                    position.stop_price,
                    position.best_price * (1.0 + self.config.trading.trailing_pullback_pct),
                )
        if abs(position.stop_price - previous_stop) / max(position.entry_price, 1e-12) >= 0.00005:
            self.log(f"{position.symbol}: 盈利保护移动止损 {previous_stop:.6g} -> {position.stop_price:.6g}")

    def _profit_exit_reason(
        self,
        position: LivePosition | SimPosition,
        candles: list[Candle],
        current_candle: Candle | None = None,
        state: ProfitState | None = None,
    ) -> str | None:
        if not self.config.trading.profit_exit_enabled or not candles:
            return None
        candle = current_candle or candles[-1]
        if state is not None:
            if position.direction == Direction.LONG:
                state.best_price = max(state.best_price, candle.high)
                best_price = state.best_price
            else:
                state.best_price = min(state.best_price, candle.low)
                best_price = state.best_price
        else:
            best_price = position.best_price if isinstance(position, SimPosition) else position.entry_price

        current_profit = _directional_profit_pct(position.direction, position.entry_price, candle.close)
        peak_profit = _directional_profit_pct(position.direction, position.entry_price, best_price)
        if position.direction == Direction.LONG:
            pullback = (best_price - candle.close) / max(position.entry_price, 1e-12)
        else:
            pullback = (candle.close - best_price) / max(position.entry_price, 1e-12)

        indicator_peak_reason = self._indicator_peak_giveback_exit_reason(
            position,
            current_profit,
            peak_profit,
            pullback,
        )
        if indicator_peak_reason:
            return indicator_peak_reason

        minimum_profit = self._minimum_profit_exit_pct()
        if current_profit < minimum_profit:
            return None

        if current_profit >= self.config.trading.strong_take_profit_pct:
            return f"strong_take_profit now={current_profit * 100:.3f}%"

        if current_profit >= self.config.trading.quick_take_profit_pct:
            continuation, continuation_reason = self._scale_in_confirmation_reason(
                position.direction,
                _candles_until(candles, candle),
                loss_scale=False,
            )
            if not continuation:
                return f"quick_take_profit now={current_profit * 100:.3f}% no_follow={continuation_reason}"

        if (
            peak_profit >= self.config.trading.trailing_activation_pct
            and current_profit > self.config.trading.breakeven_lock_pct
            and pullback >= self.config.trading.trailing_pullback_pct
        ):
            return f"profit_pullback peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}%"

        if (
            peak_profit >= self.config.trading.breakeven_trigger_pct
            and 0.0 < current_profit <= self.config.trading.breakeven_lock_pct
        ):
            return f"profit_lock now={current_profit * 100:.3f}%"

        if current_profit < self.config.trading.momentum_exit_min_profit_pct:
            return None
        momentum_reason = self._momentum_profit_exit_reason(position.direction, _candles_until(candles, candle))
        if momentum_reason:
            return momentum_reason
        return None

    def _indicator_peak_giveback_exit_reason(
        self,
        position: LivePosition | SimPosition,
        current_profit: float,
        peak_profit: float,
        pullback: float,
    ) -> str | None:
        strategy = self.config.strategy
        if not bool(getattr(strategy, "indicator_peak_giveback_enabled", False)):
            return None
        entry_reason = str(getattr(position, "entry_reason", ""))
        if not _is_indicator_reversal_entry_reason(entry_reason):
            return None
        if peak_profit <= 0:
            return None

        high_trigger = max(0.0, float(getattr(strategy, "indicator_peak_giveback_high_trigger_pct", 0.018)))
        high_floor = max(0.0, float(getattr(strategy, "indicator_peak_giveback_high_floor_pct", 0.0050)))
        if high_trigger > 0 and peak_profit >= high_trigger and current_profit <= high_floor:
            return (
                f"indicator_peak_giveback_high peak={peak_profit * 100:.3f}% "
                f"now={current_profit * 100:.3f}% floor={high_floor * 100:.3f}%"
            )

        mid_trigger = max(0.0, float(getattr(strategy, "indicator_peak_giveback_mid_trigger_pct", 0.012)))
        mid_retrace = max(0.0, float(getattr(strategy, "indicator_peak_giveback_mid_retrace_pct", 0.0045)))
        if mid_trigger > 0 and peak_profit >= mid_trigger and pullback >= mid_retrace:
            return (
                f"indicator_peak_giveback_mid peak={peak_profit * 100:.3f}% "
                f"now={current_profit * 100:.3f}% pullback={pullback * 100:.3f}%"
            )

        low_trigger = max(0.0, float(getattr(strategy, "indicator_peak_giveback_low_trigger_pct", 0.008)))
        low_floor = max(0.0, float(getattr(strategy, "indicator_peak_giveback_low_floor_pct", 0.0010)))
        if low_trigger > 0 and peak_profit >= low_trigger and current_profit <= low_floor:
            return (
                f"indicator_peak_giveback_low peak={peak_profit * 100:.3f}% "
                f"now={current_profit * 100:.3f}% floor={low_floor * 100:.3f}%"
            )

        breakeven_trigger = max(0.0, float(getattr(strategy, "indicator_peak_breakeven_trigger_pct", 0.005)))
        breakeven_floor = max(0.0, float(getattr(strategy, "indicator_peak_breakeven_floor_pct", 0.0008)))
        if breakeven_trigger > 0 and peak_profit >= breakeven_trigger and current_profit <= breakeven_floor:
            return (
                f"indicator_peak_breakeven peak={peak_profit * 100:.3f}% "
                f"now={current_profit * 100:.3f}% floor={breakeven_floor * 100:.3f}%"
            )
        return None

    def _indicator_trend_guard_reason(self, direction: Direction, closes: list[float], atr_values: list[float]) -> str | None:
        strategy = self.config.strategy
        if not strategy.indicator_trend_guard_enabled:
            return None
        minimum = max(strategy.slow_ema, strategy.indicator_trend_guard_lookback_bars + 1)
        if len(closes) < minimum or not atr_values:
            return None

        slow_values = ema(closes, strategy.slow_ema)
        lookback = max(1, min(strategy.indicator_trend_guard_lookback_bars, len(slow_values) - 1))
        current_close = closes[-1]
        current_slow = slow_values[-1]
        atr_value = max(atr_values[-1], 1e-12)
        slope_atr = (slow_values[-1] - slow_values[-1 - lookback]) / atr_value
        buffer = max(0.0, strategy.indicator_trend_guard_buffer_atr) * atr_value
        slope_threshold = max(0.0, strategy.indicator_trend_guard_slope_atr)

        if direction == Direction.SHORT and current_close > current_slow + buffer and slope_atr >= slope_threshold:
            return f"indicator_short_blocked_30m_uptrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
        if direction == Direction.LONG and current_close < current_slow - buffer and slope_atr <= -slope_threshold:
            return f"indicator_long_blocked_30m_downtrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
        return None

    def _trend_loss_exit_reason(self, position: LivePosition | SimPosition, mark_price: float) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "trend_loss_guard_enabled", False):
            return None
        entry_reason = getattr(position, "entry_reason", "")
        if not _is_trend_entry_reason(entry_reason):
            return None

        normalized = entry_reason.lower()
        is_super_volume = "super_volume" in normalized
        min_bars = (
            int(getattr(strategy, "trend_loss_guard_super_volume_bars", 5))
            if is_super_volume
            else int(getattr(strategy, "trend_loss_guard_bars", 3))
        )
        loss_pct = (
            float(getattr(strategy, "trend_loss_guard_super_volume_loss_pct", 0.009))
            if is_super_volume
            else float(getattr(strategy, "trend_loss_guard_loss_pct", 0.006))
        )
        if self._position_bars_held(position) < max(1, min_bars):
            return None

        current_profit = _directional_profit_pct(position.direction, position.entry_price, mark_price)
        if current_profit > -abs(loss_pct):
            return None
        return f"trend_loss_guard loss={current_profit * 100:.3f}% bars={self._position_bars_held(position)}"

    def _indicator_long_fail_fast_exit_reason(self, position: LivePosition | SimPosition, mark_price: float) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_long_fail_fast_enabled", False):
            return None
        if getattr(position, "direction", Direction.FLAT) != Direction.LONG:
            return None
        entry_reason = str(getattr(position, "entry_reason", ""))
        if not _is_indicator_reversal_entry_reason(entry_reason):
            return None

        entry_time = getattr(position, "entry_time", None) or getattr(position, "opened_at", None)
        checked_time = getattr(position, "last_checked_time", None)
        if checked_time is None:
            checked_time = datetime.now(timezone.utc).replace(tzinfo=None)
        if entry_time is None or checked_time is None:
            return None
        try:
            hold_minutes = max(0.0, (checked_time - entry_time).total_seconds() / 60.0)
        except TypeError:
            return None
        fail_fast_minutes = max(1, int(getattr(strategy, "indicator_long_fail_fast_minutes", 120)))
        if hold_minutes < fail_fast_minutes:
            return None

        quantity = abs(float(getattr(position, "quantity", 0.0) or 0.0))
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        stop_price = float(getattr(position, "stop_price", 0.0) or 0.0)
        best_price = getattr(position, "best_price", None)
        if best_price is None:
            state = self._profit_states.get(getattr(position, "symbol", ""))
            best_price = state.best_price if state is not None else entry_price
        best_price = float(best_price or entry_price)
        risk_cash = max(0.0, entry_price - stop_price) * quantity
        if risk_cash <= 0:
            return None
        mfe_cash = max(0.0, best_price - entry_price) * quantity
        min_r = max(0.0, float(getattr(strategy, "indicator_long_fail_fast_min_r", 0.25)))
        if mfe_cash >= risk_cash * min_r:
            return None
        current_profit = _directional_profit_pct(Direction.LONG, entry_price, mark_price)
        return (
            f"indicator_long_fail_fast hold={hold_minutes:.0f}m "
            f"mfe_r={mfe_cash / risk_cash:.2f} < {min_r:.2f} "
            f"profit={current_profit * 100:.3f}%"
        )

    def _account_loss_exit_reason(
        self,
        position: LivePosition | SimPosition,
        mark_price: float,
        account: AccountSnapshot,
    ) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "account_loss_guard_enabled", False):
            return None
        if account.equity <= 0:
            return None
        min_bars = max(0, int(getattr(strategy, "account_loss_guard_min_bars", 1)))
        bars_held = self._position_bars_held(position)
        if bars_held < min_bars:
            return None

        quantity = float(getattr(position, "quantity", 0.0) or 0.0)
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        direction = getattr(position, "direction", Direction.FLAT)
        pnl = direction.value * quantity * (mark_price - entry_price)
        loss_pct = pnl / account.equity
        threshold = abs(float(getattr(strategy, "account_loss_guard_pct", 0.012)))
        if threshold <= 0 or loss_pct > -threshold:
            return None
        return f"account_loss_guard loss={loss_pct * 100:.2f}% pnl={pnl:+.4f}U bars={bars_held}"

    def _stale_position_exit_reason(self, position: LivePosition | SimPosition, mark_price: float) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "stale_position_exit_enabled", False):
            return None

        bars_held = self._position_bars_held(position)
        if bars_held <= 0:
            return None

        observation_bars, force_exit_bars = self._stale_exit_thresholds(position)

        current_profit = _directional_profit_pct(position.direction, position.entry_price, mark_price)
        minimum_profit = self._minimum_profit_exit_pct()
        if bars_held >= force_exit_bars:
            return f"stale_force_exit bars={bars_held} profit={current_profit * 100:.3f}%"
        if bars_held >= observation_bars and current_profit >= minimum_profit:
            return f"stale_profit_exit bars={bars_held} profit={current_profit * 100:.3f}%"
        return None

    def _stale_exit_thresholds(self, position: LivePosition | SimPosition) -> tuple[int, int]:
        strategy = self.config.strategy
        entry_reason = str(getattr(position, "entry_reason", ""))
        is_indicator = _is_indicator_reversal_entry_reason(entry_reason)
        is_super_volume = "super_volume" in entry_reason.lower() or "startup_breakout" in entry_reason.lower()
        observation_bars = (
            int(getattr(strategy, "indicator_stale_observation_bars", 6))
            if is_indicator
            else (
                int(getattr(strategy, "stale_super_volume_observation_bars", 3))
                if is_super_volume
                else int(getattr(strategy, "stale_observation_bars", 8))
            )
        )
        force_exit_bars = (
            int(getattr(strategy, "indicator_stale_force_exit_bars", 8))
            if is_indicator
            else (
                int(getattr(strategy, "stale_super_volume_force_exit_bars", 6))
                if is_super_volume
                else int(getattr(strategy, "stale_force_exit_bars", 12))
            )
        )
        observation_bars = max(1, observation_bars)
        force_exit_bars = max(observation_bars, force_exit_bars)
        return observation_bars, force_exit_bars

    def _position_bars_held(self, position: LivePosition | SimPosition) -> int:
        bars_held = int(getattr(position, "bars_held", 0) or 0)
        if bars_held > 0:
            return bars_held
        opened_at = getattr(position, "opened_at", None)
        if opened_at is None:
            return 0
        now = datetime.now(opened_at.tzinfo) if opened_at.tzinfo else datetime.now()
        elapsed_seconds = max(0.0, (now - opened_at).total_seconds())
        bar_seconds = max(1.0, interval_to_milliseconds(self.config.trading.timeframe) / 1000.0)
        return int(elapsed_seconds // bar_seconds)

    def _managed_exit_allowed(self, position: LivePosition | SimPosition) -> bool:
        minimum = max(0, int(getattr(self.config.trading, "min_managed_exit_bars", 0)))
        if minimum <= 0:
            return True
        return self._position_bars_held(position) >= minimum

    def _minimum_profit_exit_pct(self) -> float:
        round_trip_fee = max(0.0, self.config.risk.estimated_fee_bps) * 2.0 / 10_000.0
        slippage = max(0.0, self.config.risk.estimated_slippage_bps) / 10_000.0
        net_buffer = max(0.0, self.config.risk.min_profit_after_cost_pct)
        return round_trip_fee + slippage + net_buffer

    def _momentum_profit_exit_reason(self, direction: Direction, candles: list[Candle]) -> str | None:
        minimum = max(
            self.config.filters.rsi_period + 2,
            self.config.filters.macd_slow + self.config.filters.macd_signal + 3,
            self.config.filters.kdj_period + 2,
        )
        if len(candles) < minimum:
            return None
        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self.config.filters.rsi_period)
        _, _, macd_histogram = macd(
            closes,
            self.config.filters.macd_fast,
            self.config.filters.macd_slow,
            self.config.filters.macd_signal,
        )
        k_values, d_values, _ = kdj(candles, self.config.filters.kdj_period)
        if direction == Direction.LONG:
            if rsi_values[-1] >= self.config.trading.profit_exit_rsi_long and rsi_values[-1] < rsi_values[-2]:
                return f"profit_rsi_rollover rsi={rsi_values[-1]:.1f}"
            if k_values[-1] < d_values[-1] and k_values[-2] >= d_values[-2] and rsi_values[-1] > 50.0:
                return f"profit_kdj_cross_down k={k_values[-1]:.1f} d={d_values[-1]:.1f}"
            if macd_histogram[-1] < macd_histogram[-2] < macd_histogram[-3] and rsi_values[-1] > 55.0:
                return "profit_macd_fade"
        elif direction == Direction.SHORT:
            if rsi_values[-1] <= self.config.trading.profit_exit_rsi_short and rsi_values[-1] > rsi_values[-2]:
                return f"profit_rsi_rebound rsi={rsi_values[-1]:.1f}"
            if k_values[-1] > d_values[-1] and k_values[-2] <= d_values[-2] and rsi_values[-1] < 50.0:
                return f"profit_kdj_cross_up k={k_values[-1]:.1f} d={d_values[-1]:.1f}"
            if macd_histogram[-1] > macd_histogram[-2] > macd_histogram[-3] and rsi_values[-1] < 45.0:
                return "profit_macd_rebound"
        return None

    def _indicator_confirmed_cross_context_guard_reason(
        self,
        direction: Direction,
        closes: list[float],
        rsi_values: list[float],
        k_values: list[float],
        d_values: list[float],
        slow_values: list[float],
        atr_values: list[float],
    ) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_confirmed_cross_extreme_guard_enabled", False):
            return None
        lookback = max(1, min(int(getattr(strategy, "indicator_confirmed_extreme_lookback_bars", 4)), len(rsi_values)))
        recent_rsi = rsi_values[-lookback:]
        recent_kd = [value for pair in zip(k_values[-lookback:], d_values[-lookback:]) for value in pair]

        if direction == Direction.LONG:
            has_extreme = (
                min(recent_rsi) <= float(getattr(strategy, "indicator_confirmed_long_max_rsi", 42.0))
                or min(recent_kd) <= float(getattr(strategy, "indicator_confirmed_long_max_kdj", 35.0))
            )
            if has_extreme:
                return None
            conflict_reason = self._indicator_confirmed_counter_trend_conflict_reason(direction, closes, slow_values, atr_values)
            if conflict_reason:
                return (
                    f"indicator_long_blocked_no_cold_context "
                    f"rsi_min={min(recent_rsi):.1f} kdj_min={min(recent_kd):.1f} {conflict_reason}"
                )
            return None

        if direction == Direction.SHORT:
            has_extreme = (
                max(recent_rsi) >= float(getattr(strategy, "indicator_confirmed_short_min_rsi", 58.0))
                or max(recent_kd) >= float(getattr(strategy, "indicator_confirmed_short_min_kdj", 65.0))
            )
            if has_extreme:
                return None
            conflict_reason = self._indicator_confirmed_counter_trend_conflict_reason(direction, closes, slow_values, atr_values)
            if conflict_reason:
                return (
                    f"indicator_short_blocked_no_hot_context "
                    f"rsi_max={max(recent_rsi):.1f} kdj_max={max(recent_kd):.1f} {conflict_reason}"
                )
            return None

        return None

    def _indicator_confirmed_cross_required_extreme_guard_reason(
        self,
        direction: Direction,
        current_rsi: float,
        current_k: float,
        current_d: float,
    ) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_confirmed_cross_extreme_required_enabled", False):
            return None

        if direction == Direction.LONG:
            max_rsi = float(getattr(strategy, "indicator_confirmed_cross_long_max_rsi", 45.0))
            max_kdj = float(getattr(strategy, "indicator_confirmed_cross_long_max_kdj", 35.0))
            if current_rsi <= max_rsi or min(current_k, current_d) <= max_kdj:
                return None
            return (
                f"indicator_long_blocked_not_cold_enough "
                f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
            )

        if direction == Direction.SHORT:
            min_rsi = float(getattr(strategy, "indicator_confirmed_cross_short_min_rsi", 60.0))
            min_kdj = float(getattr(strategy, "indicator_confirmed_cross_short_min_kdj", 70.0))
            if current_rsi >= min_rsi or max(current_k, current_d) >= min_kdj:
                return None
            return (
                f"indicator_short_blocked_not_hot_enough "
                f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
            )

        return None

    def _indicator_confirmed_counter_trend_conflict_reason(
        self,
        direction: Direction,
        closes: list[float],
        slow_values: list[float],
        atr_values: list[float],
    ) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_confirmed_trend_fallback_enabled", True):
            return None
        minimum = max(strategy.slow_ema, 2)
        if len(closes) < minimum or len(slow_values) < 2 or not atr_values:
            return None
        lookback = max(1, min(getattr(strategy, "indicator_trend_guard_lookback_bars", 12), len(slow_values) - 1))
        current_close = closes[-1]
        current_slow = slow_values[-1]
        atr_value = max(atr_values[-1], 1e-12)
        slope_atr = (slow_values[-1] - slow_values[-1 - lookback]) / atr_value
        buffer = max(0.0, getattr(strategy, "indicator_confirmed_trend_buffer_atr", 0.15)) * atr_value
        slope_threshold = max(0.0, getattr(strategy, "indicator_confirmed_trend_slope_atr", 0.05))
        if direction == Direction.LONG and current_close < current_slow - buffer and slope_atr <= -slope_threshold:
            return f"against_30m_downtrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
        if direction == Direction.SHORT and current_close > current_slow + buffer and slope_atr >= slope_threshold:
            return f"against_30m_uptrend close={current_close:.6g} slow={current_slow:.6g} slope_atr={slope_atr:.2f}"
        return None

    def _scale_in_confirmation_reason(self, direction: Direction, candles: list[Candle], loss_scale: bool) -> tuple[bool, str]:
        minimum = max(
            self.config.filters.rsi_period + 2,
            self.config.filters.macd_slow + self.config.filters.macd_signal + 3,
            self.config.filters.kdj_period + 2,
        )
        if len(candles) < minimum:
            return False, "indicator_warmup"

        closes = [candle.close for candle in candles]
        rsi_values = rsi(closes, self.config.filters.rsi_period)
        _, _, macd_histogram = macd(
            closes,
            self.config.filters.macd_fast,
            self.config.filters.macd_slow,
            self.config.filters.macd_signal,
        )
        k_values, d_values, _ = kdj(candles, self.config.filters.kdj_period)
        candle = candles[-1]
        score = 0
        reasons: list[str] = []
        if direction == Direction.LONG:
            if candle.close > candle.open:
                score += 1
                reasons.append("close_green")
            if macd_histogram[-1] > macd_histogram[-2]:
                score += 1
                reasons.append("macd_up")
            if k_values[-1] > d_values[-1]:
                score += 1
                reasons.append("kdj_up")
            if rsi_values[-1] < self.config.trading.profit_exit_rsi_long:
                score += 1
                reasons.append("rsi_not_hot")
        elif direction == Direction.SHORT:
            if candle.close < candle.open:
                score += 1
                reasons.append("close_red")
            if macd_histogram[-1] < macd_histogram[-2]:
                score += 1
                reasons.append("macd_down")
            if k_values[-1] < d_values[-1]:
                score += 1
                reasons.append("kdj_down")
            if rsi_values[-1] > self.config.trading.profit_exit_rsi_short:
                score += 1
                reasons.append("rsi_not_cold")
        required_score = 2
        if score < required_score:
            return False, f"score={score}/{required_score} rsi={rsi_values[-1]:.1f}"
        return True, f"score={score} {' '.join(reasons)}"

    def _size_order(
        self,
        symbol: str,
        price: float,
        signal: Signal,
        account: AccountSnapshot,
        existing_position: LivePosition | None = None,
        entry_fraction: float | None = None,
    ) -> tuple[str, str]:
        if price <= 0 or signal.stop_loss_pct <= 0:
            return "0", "bad_price_or_stop"
        if account.available_balance < self.config.risk.min_available_balance_usdt:
            return "0", "available_balance_too_low"

        account_margin_pct = _account_margin_limit_pct(self.config, signal=signal)
        symbol_margin_pct = _symbol_margin_limit_pct(self.config, signal)
        remaining_margin = account.equity * account_margin_pct - account.initial_margin
        if remaining_margin <= 0:
            return "0", "margin_usage_limit"

        existing_notional = existing_position.notional if existing_position else 0.0
        risk_weight = signal_risk_weight(signal.confidence, signal.risk_multiplier)
        drawdown_multiplier = self._soft_drawdown_size_multiplier(account)
        if drawdown_multiplier <= 0:
            return "0", "soft_drawdown_stop"
        risk_notional = account.equity * self.config.risk.risk_per_trade_pct * risk_weight * drawdown_multiplier / signal.stop_loss_pct
        leverage = _effective_entry_leverage(self.config, symbol, signal)
        symbol_margin_notional = account.equity * symbol_margin_pct * leverage * drawdown_multiplier
        policy_notional_cap = self.config.risk.max_position_notional_usdt
        if policy_notional_cap <= 0:
            policy_notional_cap = float("inf")
        min_initial_margin_notional = account.equity * self.config.risk.min_symbol_margin_pct * leverage
        if policy_notional_cap < float("inf"):
            min_initial_margin_notional = min(min_initial_margin_notional, policy_notional_cap)
        remaining_margin_notional = remaining_margin * leverage
        total_cap = min(risk_notional, symbol_margin_notional, policy_notional_cap)
        remaining_risk_notional = risk_notional - existing_notional
        remaining_symbol_notional = symbol_margin_notional - existing_notional
        remaining_policy_notional = policy_notional_cap - existing_notional
        additional_cap = min(
            remaining_risk_notional,
            remaining_symbol_notional,
            remaining_policy_notional,
            remaining_margin_notional,
        )
        if existing_position:
            if existing_position.direction != signal.direction:
                return "0", "position_direction_mismatch"
            requested_fraction = self.config.trading.scale_in_entry_fraction if entry_fraction is None else entry_fraction
            fraction = max(0.0, min(1.0, requested_fraction))
            notional = min(additional_cap, total_cap * fraction)
        else:
            fraction = max(0.0, min(1.0, self.config.trading.initial_entry_fraction))
            notional = additional_cap * fraction
        if 0.0 < notional < min_initial_margin_notional:
            if drawdown_multiplier < 1.0:
                return "0", "soft_drawdown_below_min_margin"
            if additional_cap < min_initial_margin_notional:
                return "0", "below_min_margin_pct"
            notional = min_initial_margin_notional
        if notional < self.config.risk.min_order_notional_usdt:
            return "0", "below_min_notional"

        rules = self.client.symbol_rules(symbol)
        quantity = rules.round_quantity(notional / price)
        rounded_notional = float(quantity) * price
        min_required_notional = max(self.config.risk.min_order_notional_usdt, min_initial_margin_notional)
        if rounded_notional < min_required_notional and min_required_notional <= additional_cap:
            quantity = rules.round_quantity(min_required_notional * 1.001 / price)
            rounded_notional = float(quantity) * price
        if DecimalCompat.less_than(quantity, rules.min_quantity):
            return "0", "below_min_quantity"
        if rounded_notional < max(float(rules.min_notional), min_required_notional):
            return "0", "below_exchange_min_notional"
        return quantity, "ok"

    def _soft_drawdown_size_multiplier(self, account: AccountSnapshot) -> float:
        peak = max(self._peak_equity, account.equity)
        if peak <= 0:
            return 1.0
        stop_at = max(0.0, self.config.risk.soft_drawdown_stop_pct)
        reduce_at = max(0.0, self.config.risk.soft_drawdown_reduce_pct)
        floor = max(0.0, min(1.0, self.config.risk.soft_drawdown_min_size_multiplier))
        if stop_at <= 0:
            return 1.0

        drawdown = max(0.0, (peak - account.equity) / peak)
        if drawdown >= stop_at:
            return floor
        if reduce_at <= 0 or drawdown <= reduce_at:
            return 1.0
        if stop_at <= reduce_at:
            return 1.0

        pressure = (drawdown - reduce_at) / (stop_at - reduce_at)
        return max(floor, 1.0 - pressure * (1.0 - floor))

    def _global_risk_allows_trading(self, account: AccountSnapshot) -> bool:
        now = time.time()
        if now < self._cooldown_until:
            self.log("冷却期内，暂停开仓")
            return False
        total_pnl = account.equity - self.stats.starting_equity
        self._session_peak_pnl = max(self._session_peak_pnl, total_pnl)
        if account.initial_margin_usage_pct >= _account_margin_limit_pct(self.config):
            self.log("保证金占用已达到上限，暂停开仓")
            return False
        if account.equity <= self._day_start_equity * (1.0 - self.config.risk.max_daily_loss_pct):
            self.log("触发当日最大亏损限制，暂停开仓")
            return False
        starting_stop = max(0.0, self.config.risk.starting_capital_drawdown_stop_pct)
        if starting_stop > 0 and account.equity <= self.stats.starting_equity * (1.0 - starting_stop):
            self.log("触发本金最大回撤限制，暂停开仓")
            return False
        weekly_stop = max(0.0, self.config.risk.weekly_profit_drawdown_stop_pct)
        weekly_active = self._week_peak_equity > self.stats.starting_equity or self._week_start_equity > self.stats.starting_equity
        if weekly_active and weekly_stop > 0 and account.equity <= self._week_peak_equity * (1.0 - weekly_stop):
            self.log("触发本周盈利回撤限制，暂停开仓")
            return False
        return True

    def _update_loss_limits(self, account: AccountSnapshot) -> None:
        now = datetime.now(timezone.utc)
        today = now.date()
        if today != self._day:
            self._day = today
            self._day_start_equity = account.equity
        week = now.isocalendar()[:2]
        if week != self._week:
            self._week = week
            self._week_start_equity = account.equity
            self._week_peak_equity = account.equity
        else:
            self._week_peak_equity = max(self._week_peak_equity, account.equity)
        self._peak_equity = max(self._peak_equity, account.equity)

    def _log_session_stats(self, account: AccountSnapshot, force: bool = False) -> None:
        now = time.time()
        interval = max(10, self.config.trading.stats_log_interval_seconds)
        if not force and now - self._last_stats_log_ts < interval:
            return
        self._last_stats_log_ts = now
        runtime = datetime.now() - self.stats.started_at
        runtime_text = _format_runtime(int(runtime.total_seconds()))
        total_pnl = account.equity - self.stats.starting_equity
        self._session_peak_pnl = max(self._session_peak_pnl, total_pnl)
        self.log(
            "统计: "
            f"运行={runtime_text} "
            f"平仓={self.stats.closed_trades} "
            f"胜率={self.stats.win_rate_pct:.2f}% "
            f"已实现={self.stats.realized_pnl:+.4f}U "
            f"未实现={account.total_unrealized_pnl:+.4f}U "
            f"总盈亏={total_pnl:+.4f}U "
            f"峰值={self._session_peak_pnl:+.4f}U "
            f"权益={account.equity:.2f}U"
        )

    def _closed_candles(self, symbol: str) -> list[Candle]:
        candles = self.client.klines(symbol, self.config.trading.timeframe, self.config.trading.kline_limit)
        return candles[:-1] if len(candles) > 1 else candles

    def _latest_closed_candle_timestamp(self, symbol: str, fallback: datetime) -> datetime:
        try:
            candles = self._closed_candles(symbol)
        except Exception:
            return fallback
        if not candles:
            return fallback
        return max(fallback, candles[-1].timestamp)

    def _passes_multi_timeframe_filter(self, symbol: str, direction: Direction) -> tuple[bool, str]:
        if not self.config.filters.enabled:
            return True, "disabled"
        frames: list[TimeframeSignal] = []
        for timeframe in self.config.filters.timeframes:
            candles = self._closed_candles_for_timeframe(symbol, timeframe, self.config.filters.kline_limit)
            if len(candles) < self._mtf_filter.warmup_bars:
                return False, f"{timeframe}_candles_insufficient"
            frames.append(self._mtf_filter.snapshot(timeframe, candles))
        return self._mtf_filter.evaluate(direction, frames)

    def _indicator_reference_guard_reason(self, symbol: str, signal: Signal) -> str | None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_reference_guard_enabled", False):
            return None
        if signal.direction == Direction.FLAT or not signal.reason.lower().startswith("indicator_"):
            return None
        if signal.direction == Direction.LONG and not getattr(strategy, "indicator_reference_guard_long_enabled", True):
            return None
        if signal.direction == Direction.SHORT and not getattr(strategy, "indicator_reference_guard_short_enabled", True):
            return None

        timeframe = str(getattr(strategy, "indicator_reference_timeframe", "1h"))
        lookback = max(1, int(getattr(strategy, "indicator_reference_lookback_bars", 12)))
        limit = max(
            self.config.strategy.slow_ema + lookback + 5,
            self.config.strategy.atr_period + lookback + 5,
            self.config.filters.rsi_period + 5,
            self.config.filters.kdj_period + 5,
            120,
        )
        candles = self._closed_candles_for_timeframe(symbol, timeframe, limit)
        minimum = max(self.config.strategy.slow_ema, self.config.strategy.atr_period, lookback + 1)
        if len(candles) < minimum:
            return None

        closes = [candle.close for candle in candles]
        slow_values = ema(closes, self.config.strategy.slow_ema)
        atr_values = atr(candles, self.config.strategy.atr_period)
        rsi_values = rsi(closes, self.config.filters.rsi_period)
        k_values, d_values, _ = kdj(candles, self.config.filters.kdj_period)

        current_close = closes[-1]
        current_slow = slow_values[-1]
        atr_value = max(atr_values[-1], 1e-12)
        slope_atr = (slow_values[-1] - slow_values[-1 - min(lookback, len(slow_values) - 1)]) / atr_value
        buffer = max(0.0, float(getattr(strategy, "indicator_reference_buffer_atr", 0.10))) * atr_value
        slope_threshold = max(0.0, float(getattr(strategy, "indicator_reference_slope_atr", 0.05)))
        current_rsi = rsi_values[-1]
        current_k = k_values[-1]
        current_d = d_values[-1]
        extreme_override = bool(getattr(strategy, "indicator_reference_extreme_override_enabled", True))

        if signal.direction == Direction.SHORT:
            overhot = (
                current_rsi >= float(getattr(strategy, "indicator_reference_short_extreme_rsi", 62.0))
                or max(current_k, current_d) >= float(getattr(strategy, "indicator_reference_short_extreme_kdj", 70.0))
            )
            if extreme_override and overhot:
                return None
            if current_close > current_slow + buffer and slope_atr >= slope_threshold:
                return (
                    f"{timeframe}_bullish_against_short close={current_close:.6g} "
                    f"slow={current_slow:.6g} slope_atr={slope_atr:.2f} "
                    f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                )
        elif signal.direction == Direction.LONG:
            oversold = (
                current_rsi <= float(getattr(strategy, "indicator_reference_long_extreme_rsi", 38.0))
                or min(current_k, current_d) <= float(getattr(strategy, "indicator_reference_long_extreme_kdj", 30.0))
            )
            if extreme_override and oversold:
                return None
            if current_close < current_slow - buffer and slope_atr <= -slope_threshold:
                return (
                    f"{timeframe}_bearish_against_long close={current_close:.6g} "
                    f"slow={current_slow:.6g} slope_atr={slope_atr:.2f} "
                    f"rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}"
                )
        return None

    def _closed_candles_for_timeframe(self, symbol: str, timeframe: str, limit: int) -> list[Candle]:
        key = (symbol, timeframe)
        now = time.time()
        cached = self._mtf_candle_cache.get(key)
        if cached and now - cached[0] < 30.0:
            return cached[1]
        candles = self.client.klines(symbol, timeframe, limit)
        closed = candles[:-1] if len(candles) > 1 else candles
        self._mtf_candle_cache[key] = (now, closed)
        return closed

    def _latest_close(self, symbol: str) -> float:
        try:
            candles = self.client.klines(symbol, self.config.trading.timeframe, 2)
            return candles[-1].close if candles else 0.0
        except Exception:
            position = self._sim_positions.get(symbol)
            return position.entry_price if position else 0.0

    def _prepare_symbol(self, symbol: str, leverage_override: int | None = None, signal: Signal | None = None) -> bool:
        leverage = _effective_entry_leverage(self.config, symbol, signal, leverage_override)
        prepared_key = (symbol, leverage)
        if prepared_key in self._prepared_symbols and self._active_symbol_leverage.get(symbol) == leverage:
            return True
        if leverage_override is None and symbol in self._unsupported_symbols:
            return False
        if leverage_override is not None and prepared_key in self._unsupported_symbol_leverages:
            return False
        if self.config.trading.dry_run:
            self._prepared_symbols.add(prepared_key)
            self._active_symbol_leverage[symbol] = leverage
            return True
        try:
            try:
                self.client.set_margin_type(symbol, self.config.trading.margin_type)
            except BinanceApiError as exc:
                if _margin_type_change_blocked(exc):
                    self.log(f"{symbol}: 有未成交挂单或持仓，跳过保证金模式切换，沿用交易所当前设置 ({exc})")
                else:
                    raise
            self.client.set_leverage(symbol, leverage)
            self._prepared_symbols.add(prepared_key)
            self._active_symbol_leverage[symbol] = leverage
            self.log(f"{symbol}: 已设置 {self.config.trading.margin_type} / {leverage}x")
            return True
        except BinanceApiError as exc:
            self.log(f"{symbol}: 杠杆或保证金模式设置失败: {exc}")
            if _leverage_not_valid(exc):
                if leverage_override is None:
                    self._unsupported_symbols.add(symbol)
                    self.log(f"{symbol}: 当前不支持 {leverage}x，已从本次开仓扫描中跳过")
                else:
                    self._unsupported_symbol_leverages.add(prepared_key)
                    self.log(f"{symbol}: 当前不支持宏观事件 {leverage}x，本次事件仓跳过")
                return False
            raise

    @staticmethod
    def _entry_price_from_response(response: dict, fallback: float) -> float:
        try:
            avg_price = float(response.get("avgPrice", 0.0))
            if avg_price > 0:
                return avg_price
            executed_qty = float(response.get("executedQty", 0.0))
            cum_quote = float(response.get("cumQuote", 0.0))
            if executed_qty > 0 and cum_quote > 0:
                return cum_quote / executed_qty
        except (TypeError, ValueError):
            pass
        return fallback

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger(f"[{timestamp}] {message}")


class DecimalCompat:
    @staticmethod
    def less_than(quantity: str, minimum: object) -> bool:
        try:
            return float(quantity) < float(str(minimum))
        except ValueError:
            return True


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _rate_limit_backoff_seconds(message: str) -> float:
    match = re.search(r"banned until (\d+)", message)
    if match:
        banned_until_seconds = int(match.group(1)) / 1000.0
        return max(60.0, banned_until_seconds - time.time() + 5.0)
    lowered = message.lower()
    if "too many requests" in lowered or "rate limit" in lowered:
        return 120.0
    return 0.0


def _margin_type_change_blocked(exc: BinanceApiError) -> bool:
    if isinstance(exc.payload, dict):
        try:
            code = int(exc.payload.get("code", 0))
        except (TypeError, ValueError):
            code = 0
        if code in {-4047, -4048}:
            return True
    message = str(exc).lower()
    return (
        "margin type cannot be changed" in message
        or "there exists open orders" in message
        or "there exists quantity" in message
    )


def _leverage_not_valid(exc: BinanceApiError) -> bool:
    message = str(exc).lower()
    return "leverage" in message and "not valid" in message


def _signal_uses_fixed_vbp_leverage(signal: Signal | None) -> bool:
    if signal is None:
        return False
    reason = str(getattr(signal, "reason", "")).lower()
    return "vbp_" in reason or "volume_breakout_pullback" in reason


def _effective_entry_leverage(
    config: LiveAppConfig,
    symbol: str,
    signal: Signal | None = None,
    leverage_override: int | None = None,
) -> int:
    if leverage_override is not None:
        return max(1, int(leverage_override))
    base = max(1, int(getattr(config.trading, "leverage", 1)))
    if _signal_uses_fixed_vbp_leverage(signal):
        return base
    overrides = getattr(config.trading, "symbol_leverage_overrides", {}) or {}
    override = overrides.get(symbol.upper())
    if override is None:
        return base
    try:
        return max(1, min(base, int(float(override))))
    except (TypeError, ValueError):
        return base


def _macro_event_id(event: MacroEvent) -> str:
    return f"{event.event_type}:{event.reference}:{event.timestamp.isoformat()}"


def _position_profit_pct(position: LivePosition, mark_price: float) -> float:
    if position.entry_price <= 0:
        return 0.0
    return _directional_profit_pct(position.direction, position.entry_price, mark_price)


def _directional_profit_pct(direction: Direction, entry_price: float, mark_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return direction.value * (mark_price - entry_price) / entry_price


def _is_trend_entry_reason(reason: str) -> bool:
    normalized = reason.lower()
    return any(token in normalized for token in ("breakout", "breakdown", "pullback", "super_volume", "startup_breakout"))


def _is_vbp_entry_reason(reason: str) -> bool:
    normalized = reason.lower()
    return "vbp_" in normalized or "volume_breakout_pullback" in normalized


def _vbp_large_bearish_candle_reason(candles: list[Candle], exit_config: Any) -> str | None:
    lookback = max(3, int(getattr(exit_config, "large_bear_lookback_bars", 20)))
    if len(candles) < lookback + 1:
        return None
    candle = candles[-1]
    previous = candles[-lookback - 1:-1]
    average_volume = sum(max(item.volume, 0.0) for item in previous) / max(1, len(previous))
    volume_multiplier = candle.volume / max(average_volume, 1e-12)
    candle_range = max(candle.high - candle.low, 1e-12)
    close_position = (candle.close - candle.low) / candle_range
    body_pct = abs(candle.close - candle.open) / max(candle.open, 1e-12)
    required_volume = float(getattr(exit_config, "large_bear_volume_multiplier", 2.0))
    max_close_position = float(getattr(exit_config, "large_bear_max_close_position", 0.35))
    min_body_pct = float(getattr(exit_config, "large_bear_min_body_pct", 0.0015))
    if candle.close >= candle.open:
        return None
    if volume_multiplier < required_volume:
        return None
    if close_position > max_close_position:
        return None
    if body_pct < min_body_pct:
        return None
    return (
        f"vol={volume_multiplier:.2f}x close_pos={close_position:.2f} "
        f"body={body_pct * 100:.3f}%"
    )


def _is_indicator_reversal_entry_reason(reason: str) -> bool:
    return reason.lower().startswith("indicator_")


def _indicator_side_field(direction: Direction, suffix: str) -> str:
    side = "long" if direction == Direction.LONG else "short"
    return f"indicator_{side}_{suffix}"


def _indicator_side_float(strategy: Any, direction: Direction, suffix: str, fallback: float) -> float:
    value = getattr(strategy, _indicator_side_field(direction, suffix), 0.0)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if numeric <= 0:
        return float(fallback)
    return numeric


def _indicator_side_int(strategy: Any, direction: Direction, suffix: str, fallback: int) -> int:
    value = getattr(strategy, _indicator_side_field(direction, suffix), 0)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    if numeric <= 0:
        return int(fallback)
    return numeric


def _indicator_side_holding_bars(strategy: Any, direction: Direction) -> int:
    fallback = int(getattr(strategy, "indicator_max_holding_bars", getattr(strategy, "max_holding_bars", 1)))
    side_value = _indicator_side_int(strategy, direction, "max_holding_bars", fallback)
    max_bars = int(getattr(strategy, "max_holding_bars", side_value))
    return max(1, min(max_bars, max(1, side_value)))


def _entry_quality_guard_reason(config: LiveAppConfig, candidate: EntryCandidate) -> str | None:
    strategy = config.strategy
    if not _is_indicator_reversal_entry_reason(candidate.signal.reason):
        return None
    if not getattr(strategy, "indicator_min_rank_guard_enabled", False):
        return None
    min_rank = max(0.0, float(getattr(strategy, "indicator_min_rank_score", 3.0)))
    if candidate.rank_score >= min_rank:
        return None
    return f"indicator_rank={candidate.rank_score:.2f} < {min_rank:.2f}"


def _ordinary_breakout_adjusted_signal(config: LiveAppConfig, signal: Signal, rank_score: float) -> Signal | None:
    normalized = signal.reason.lower()
    ordinary = normalized.startswith("long_breakout") or normalized.startswith("short_breakdown")
    if not ordinary or "super_volume" in normalized or "startup_breakout" in normalized:
        return signal
    if not getattr(config.strategy, "ordinary_breakout_enabled", True):
        return None
    min_rank = max(0.0, float(getattr(config.strategy, "ordinary_breakout_min_rank_score", 0.0)))
    if rank_score < min_rank:
        return None
    multiplier = max(0.0, min(1.0, float(getattr(config.strategy, "ordinary_breakout_risk_multiplier", 1.0))))
    if multiplier >= 0.999:
        return signal
    return replace(
        signal,
        risk_multiplier=signal.risk_multiplier * multiplier,
        reason=f"{signal.reason}_ordinary_breakout_risk{multiplier:.2f}",
    )


def _fast_breakout_signal_for_candles(config: LiveAppConfig, candles: list[Candle]) -> Signal:
    strategy = config.strategy
    if not getattr(strategy, "fast_breakout_enabled", False):
        return Signal(Direction.FLAT, 0.0, "fast_breakout_disabled", 0.0, 0.0)

    channel_period = max(3, int(getattr(strategy, "fast_breakout_channel_period", 18)))
    volume_period = max(3, int(getattr(strategy, "fast_breakout_volume_period", 24)))
    fast_period = max(2, int(getattr(strategy, "entry_execution_rsi_fast_period", 6)))
    mid_period = max(fast_period, int(getattr(strategy, "entry_execution_rsi_mid_period", 12)))
    minimum = max(channel_period, volume_period, strategy.atr_period, mid_period) + 3
    if len(candles) < minimum:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_warmup", 0.0, 0.0)

    index = len(candles) - 1
    candle = candles[-1]
    highs = [item.high for item in candles]
    lows = [item.low for item in candles]
    closes = [item.close for item in candles]
    atr_values = atr(candles, strategy.atr_period)
    atr_value = atr_values[-2] if len(atr_values) >= 2 else atr_values[-1]
    if candle.close <= 0 or atr_value <= 0:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_zero_atr", 0.0, 0.0)

    atr_pct = atr_value / candle.close
    if atr_pct < strategy.min_atr_pct:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_volatility_too_low", 0.0, 0.0)
    if strategy.max_atr_pct > 0 and atr_pct > strategy.max_atr_pct:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_volatility_too_high", 0.0, 0.0)

    average_volume = sum(item.volume for item in candles[-volume_period - 1:-1]) / volume_period
    if average_volume <= 0:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_zero_volume", 0.0, 0.0)
    volume_ratio = candle.volume / average_volume
    min_volume = max(0.0, float(getattr(strategy, "fast_breakout_min_volume_ratio", 2.5)))
    if volume_ratio < min_volume:
        return Signal(Direction.FLAT, 0.0, "fast_breakout_volume_too_low", 0.0, 0.0)

    upper_channel = rolling_high(highs, index, channel_period)
    lower_channel = rolling_low(lows, index, channel_period)
    long_breakout = candle.close - upper_channel
    short_breakout = lower_channel - candle.close
    min_breakout = atr_value * max(0.0, float(getattr(strategy, "fast_breakout_min_breakout_atr", 0.30)))
    min_body = atr_value * max(0.0, float(getattr(strategy, "fast_breakout_min_body_atr", 0.35)))
    candle_range = max(candle.high - candle.low, 1e-12)
    max_adverse_wick = max(0.0, min(1.0, float(getattr(strategy, "fast_breakout_max_adverse_wick_pct", 0.50))))
    fast_rsi = rsi(closes, fast_period)[-1]
    mid_rsi = rsi(closes, mid_period)[-1]

    def build(direction: Direction, breakout_distance: float, body: float, wick_pct: float) -> Signal:
        breakout_score = min(1.0, max(0.0, breakout_distance / max(atr_value, 1e-12)))
        body_score = min(1.0, max(0.0, body / max(atr_value, 1e-12)))
        volume_score = min(1.0, max(0.0, (volume_ratio - min_volume) / max(min_volume, 1e-12)))
        quality = max(0.65, min(1.0, 0.58 + breakout_score * 0.18 + body_score * 0.14 + volume_score * 0.10))
        bias = strategy.long_risk_bias if direction == Direction.LONG else strategy.short_risk_bias
        if bias <= 0:
            return Signal(Direction.FLAT, 0.0, "fast_breakout_direction_disabled", 0.0, 0.0)
        stop_pct = max(atr_pct * max(0.1, float(getattr(strategy, "fast_breakout_stop_loss_atr", 0.85))), 0.0008)
        take_profit_pct = max(
            atr_pct * max(0.1, float(getattr(strategy, "fast_breakout_take_profit_atr", 1.10))),
            stop_pct * 1.05,
        )
        side = "long" if direction == Direction.LONG else "short"
        risk = max(0.0, min(1.5, quality * max(0.0, float(getattr(strategy, "fast_breakout_risk_multiplier", 0.55))) * bias))
        return Signal(
            direction=direction,
            confidence=quality,
            reason=(
                f"{side}_fast_breakout_super_volume volume={volume_ratio:.2f}x "
                f"wick={wick_pct:.2f} rsi{fast_period}={fast_rsi:.1f} rsi{mid_period}={mid_rsi:.1f}"
            ),
            stop_loss_pct=stop_pct,
            take_profit_pct=take_profit_pct,
            risk_multiplier=risk,
            max_holding_bars=max(1, int(getattr(strategy, "fast_breakout_max_holding_bars", 2))),
        )

    long_body = candle.close - candle.open
    upper_wick_pct = (candle.high - max(candle.open, candle.close)) / candle_range
    if (
        long_breakout >= min_breakout
        and long_body >= min_body
        and upper_wick_pct <= max_adverse_wick
        and fast_rsi < float(getattr(strategy, "fast_breakout_long_rsi_fast_ceiling", 86.0))
        and mid_rsi < float(getattr(strategy, "fast_breakout_long_rsi_mid_ceiling", 80.0))
    ):
        return build(Direction.LONG, long_breakout, long_body, upper_wick_pct)

    short_body = candle.open - candle.close
    lower_wick_pct = (min(candle.open, candle.close) - candle.low) / candle_range
    if (
        strategy.allow_short
        and short_breakout >= min_breakout
        and short_body >= min_body
        and lower_wick_pct <= max_adverse_wick
        and fast_rsi > float(getattr(strategy, "fast_breakout_short_rsi_fast_floor", 14.0))
        and mid_rsi > float(getattr(strategy, "fast_breakout_short_rsi_mid_floor", 20.0))
    ):
        return build(Direction.SHORT, short_breakout, short_body, lower_wick_pct)

    return Signal(Direction.FLAT, 0.0, "fast_breakout_no_breakout", 0.0, 0.0)


def _entry_execution_guard_reason_for_candles(config: LiveAppConfig, candidate: EntryCandidate, candles: list[Candle]) -> str | None:
    strategy = config.strategy
    fast_period = max(2, int(getattr(strategy, "entry_execution_rsi_fast_period", 6)))
    mid_period = max(fast_period, int(getattr(strategy, "entry_execution_rsi_mid_period", 12)))
    if len(candles) < mid_period + 3:
        return None

    reclaim_reason = _indicator_long_reclaim_guard_reason_for_candles(config, candidate, candles, str(getattr(strategy, "entry_execution_timeframe", "5m")))
    if reclaim_reason:
        return reclaim_reason

    window = candles[-max(80, mid_period + 10):]
    latest_candle = window[-1]
    latest = latest_candle.close
    signal_close = candidate.candle.close
    if latest <= 0 or signal_close <= 0:
        return None

    direction = candidate.signal.direction
    chase_pct = (latest / signal_close - 1.0) * direction.value
    max_chase = max(0.0, float(getattr(strategy, "entry_execution_max_chase_pct", 0.004)))
    if chase_pct > max_chase:
        return f"5m_chase {chase_pct * 100:.2f}% > {max_chase * 100:.2f}%"

    previous_close = window[-2].close
    reversal_pct = (latest / previous_close - 1.0) * direction.value if previous_close > 0 else 0.0
    max_reversal = max(0.0, float(getattr(strategy, "entry_execution_reversal_pct", 0.003)))
    if reversal_pct < -max_reversal:
        return f"5m_reversal {reversal_pct * 100:.2f}% < -{max_reversal * 100:.2f}%"

    candle_range = max(latest_candle.high - latest_candle.low, 1e-12)
    max_adverse_wick = max(0.0, min(1.0, float(getattr(strategy, "entry_execution_max_adverse_wick_pct", 0.55))))
    if direction == Direction.LONG:
        upper_wick_pct = (latest_candle.high - max(latest_candle.open, latest_candle.close)) / candle_range
        if upper_wick_pct > max_adverse_wick:
            return f"5m_upper_wick {upper_wick_pct:.2f} > {max_adverse_wick:.2f}"
    elif direction == Direction.SHORT:
        lower_wick_pct = (min(latest_candle.open, latest_candle.close) - latest_candle.low) / candle_range
        if lower_wick_pct > max_adverse_wick:
            return f"5m_lower_wick {lower_wick_pct:.2f} > {max_adverse_wick:.2f}"

    closes = [candle.close for candle in window]
    fast_rsi = rsi(closes, fast_period)[-1]
    mid_rsi = rsi(closes, mid_period)[-1]
    if direction == Direction.LONG:
        fast_ceiling = float(getattr(strategy, "entry_execution_long_rsi_fast_ceiling", 84.0))
        mid_ceiling = float(getattr(strategy, "entry_execution_long_rsi_mid_ceiling", 78.0))
        if fast_rsi >= fast_ceiling and mid_rsi >= mid_ceiling:
            return f"5m_long_rsi_hot rsi{fast_period}={fast_rsi:.1f} rsi{mid_period}={mid_rsi:.1f}"
    elif direction == Direction.SHORT:
        fast_floor = float(getattr(strategy, "entry_execution_short_rsi_fast_floor", 16.0))
        mid_floor = float(getattr(strategy, "entry_execution_short_rsi_mid_floor", 22.0))
        if fast_rsi <= fast_floor and mid_rsi <= mid_floor:
            return f"5m_short_rsi_cold rsi{fast_period}={fast_rsi:.1f} rsi{mid_period}={mid_rsi:.1f}"
    return None


def _indicator_long_reclaim_guard_reason_for_candles(
    config: LiveAppConfig,
    candidate: EntryCandidate,
    candles: list[Candle],
    timeframe: str,
) -> str | None:
    strategy = config.strategy
    if not getattr(strategy, "indicator_long_reclaim_filter_enabled", False):
        return None
    if candidate.signal.direction != Direction.LONG:
        return None
    if not _is_indicator_reversal_entry_reason(candidate.signal.reason):
        return None
    ema_period = max(2, int(getattr(strategy, "indicator_long_reclaim_ema_period", 9)))
    if len(candles) < ema_period + 3:
        return None
    window = candles[-max(ema_period + 10, 40):]
    latest = window[-1]
    closes = [candle.close for candle in window]
    ema_values = ema(closes, ema_period)
    reclaim_level = ema_values[-1]
    if latest.close < reclaim_level:
        return f"{timeframe}_indicator_long_no_reclaim close={latest.close:.6g} ema{ema_period}={reclaim_level:.6g}"
    candle_range = max(latest.high - latest.low, 1e-12)
    close_position = (latest.close - latest.low) / candle_range
    min_close_position = max(0.0, min(1.0, float(getattr(strategy, "indicator_long_reclaim_min_close_position", 0.55))))
    if close_position < min_close_position:
        return f"{timeframe}_indicator_long_weak_close close_pos={close_position:.2f} < {min_close_position:.2f}"
    return None


def _trend_reference_adjusted_signal_for_candles(
    config: LiveAppConfig,
    signal: Signal,
    candles: list[Candle],
) -> tuple[Signal | None, str | None]:
    strategy = config.strategy
    lookback = max(1, int(getattr(strategy, "trend_reference_lookback_bars", 6)))
    minimum = max(strategy.slow_ema, strategy.atr_period, lookback + 1)
    if len(candles) < minimum:
        return signal, None

    closes = [candle.close for candle in candles]
    slow_values = ema(closes, strategy.slow_ema)
    atr_values = atr(candles, strategy.atr_period)
    current_close = closes[-1]
    current_slow = slow_values[-1]
    atr_value = max(atr_values[-1], 1e-12)
    slope_lookback = min(lookback, len(slow_values) - 1)
    slope_atr = (slow_values[-1] - slow_values[-1 - slope_lookback]) / atr_value
    buffer = max(0.0, float(getattr(strategy, "trend_reference_buffer_atr", 0.08))) * atr_value
    slope_threshold = max(0.0, float(getattr(strategy, "trend_reference_slope_atr", 0.04)))

    opposite = False
    if signal.direction == Direction.LONG:
        opposite = current_close < current_slow - buffer and slope_atr <= -slope_threshold
    elif signal.direction == Direction.SHORT:
        opposite = current_close > current_slow + buffer and slope_atr >= slope_threshold
    if not opposite:
        return signal, None

    reason = (
        f"1h_opposite close={current_close:.6g} slow={current_slow:.6g} "
        f"slope_atr={slope_atr:.2f}"
    )
    normalized = signal.reason.lower()
    is_exceptional = "super_volume" in normalized or "startup_breakout" in normalized
    if not is_exceptional:
        return None, reason

    multiplier = max(0.0, min(1.0, float(getattr(strategy, "trend_reference_super_volume_risk_multiplier", 0.60))))
    return replace(
        signal,
        risk_multiplier=signal.risk_multiplier * multiplier,
        reason=f"{signal.reason}_1h_reference_risk{multiplier:.2f}",
    ), reason


def _combine_btc_market_states(primary: BtcMarketState, confirmation: BtcMarketState) -> BtcMarketState:
    if primary.direction == Direction.FLAT:
        return primary
    if confirmation.direction == primary.direction:
        return BtcMarketState(
            primary.direction,
            primary.return_pct,
            primary.slope_atr,
            f"{primary.reason}; confirm={confirmation.reason}",
        )
    return BtcMarketState(
        Direction.FLAT,
        primary.return_pct,
        primary.slope_atr,
        f"btc_mixed primary={primary.reason}; confirm={confirmation.reason}",
    )


def _float_value(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sma_values(values: list[float], period: int) -> list[float]:
    period = max(1, int(period))
    output: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        window = min(index + 1, period)
        output.append(running / window)
    return output


def _valid_mtf_timeframe(value: object, default: str) -> str:
    timeframe = str(value or default).strip().lower()
    try:
        interval_to_milliseconds(timeframe)
    except ValueError:
        return default
    return timeframe


def _mtf_symbol_allowed(config: LiveAppConfig, symbol: str) -> bool:
    mode = str(getattr(config.strategy, "mtf_symbols_mode", "configured")).lower()
    symbols = tuple(config.trading.symbols)
    if mode == "top30":
        return symbol in set(symbols[:30])
    if mode == "top50":
        return symbol in set(symbols[:50])
    return True


def _account_margin_limit_pct(config: LiveAppConfig, signal: Signal | None = None) -> float:
    base = max(0.0, float(config.risk.max_account_margin_usage_pct))
    strategy = config.strategy
    mtf_override = max(0.0, float(getattr(strategy, "mtf_account_margin_usage_pct", 0.0)))
    if signal is not None:
        if MTF_REASON_TOKEN in str(signal.reason) and mtf_override > 0:
            return mtf_override
        return base
    if (
        bool(getattr(strategy, "mtf_4h_rsi_regime_enabled", False))
        and bool(getattr(strategy, "mtf_disable_legacy_strategies", False))
        and mtf_override > 0
    ):
        return mtf_override
    return base


def _symbol_margin_limit_pct(config: LiveAppConfig, signal: Signal) -> float:
    base = max(0.0, float(config.risk.max_symbol_margin_pct))
    if MTF_REASON_TOKEN not in str(signal.reason):
        return base
    override = max(0.0, float(getattr(config.strategy, "mtf_symbol_margin_pct", 0.0)))
    return override if override > 0 else base


def _scale_fraction(base_fraction: float, scale_count: int) -> float:
    return max(0.0, min(1.0, base_fraction * (1.0 + 0.25 * max(0, scale_count))))


def _candles_until(candles: list[Candle], candle: Candle) -> list[Candle]:
    return [candidate for candidate in candles if candidate.timestamp <= candle.timestamp]


def _format_runtime(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
