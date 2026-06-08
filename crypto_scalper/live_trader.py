from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .binance_client import BinanceApiError, BinanceFuturesClient
from .data import interval_to_milliseconds
from .indicators import atr, ema, kdj, macd, rolling_high, rolling_low, rsi
from .live_config import LiveAppConfig
from .macro_events import MacroEvent, load_macro_events
from .market_filters import MultiTimeframeFilter, TimeframeSignal
from .models import Candle, Direction, Signal
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
        self.stats = SessionStats(datetime.now(), config.risk.starting_capital_usdt)
        self._last_stats_log_ts = 0.0

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

            candidates.sort(key=lambda item: item.rank_score, reverse=True)
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
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_reversal_loss_pause_enabled", False):
            return False
        if not _is_indicator_reversal_entry_reason(signal.reason):
            return False
        now = time.time()
        if now >= self._indicator_reversal_pause_until:
            return False
        remaining = int(self._indicator_reversal_pause_until - now)
        self.log(
            f"{symbol}: 指标反转连亏临时暂停中，剩余约{max(1, remaining // 60)}分钟，"
            f"跳过 {signal.direction.name} ({signal.reason})"
        )
        return True

    def _record_indicator_reversal_result(self, net_pnl: float, entry_reason: str) -> None:
        strategy = self.config.strategy
        if not getattr(strategy, "indicator_reversal_loss_pause_enabled", False):
            return
        if not _is_indicator_reversal_entry_reason(entry_reason):
            return

        if net_pnl > 0:
            if self._indicator_reversal_loss_streak > 0:
                self.log("指标反转单盈利，连亏计数清零")
            self._indicator_reversal_loss_streak = 0
            return

        self._indicator_reversal_loss_streak += 1
        trigger_losses = max(1, int(getattr(strategy, "indicator_reversal_loss_pause_losses", 2)))
        if self._indicator_reversal_loss_streak < trigger_losses:
            self.log(f"指标反转单亏损，连亏={self._indicator_reversal_loss_streak}/{trigger_losses}")
            return

        pause_bars = max(1, int(getattr(strategy, "indicator_reversal_loss_pause_bars", 8)))
        pause_seconds = pause_bars * max(1.0, interval_to_milliseconds(self.config.trading.timeframe) / 1000.0)
        self._indicator_reversal_pause_until = max(self._indicator_reversal_pause_until, time.time() + pause_seconds)
        self._indicator_reversal_loss_streak = 0
        self.log(f"指标反转连续{trigger_losses}次亏损，临时暂停{pause_bars}根K线后自动恢复")

    def _mark_observed_position_closures(self, active_symbols: set[str]) -> None:
        if not self._known_active_symbols:
            return
        for symbol in self._known_active_symbols - active_symbols:
            self._scale_in_counts.pop(symbol, None)
            self._last_scale_in_ts.pop(symbol, None)
            self._profit_states.pop(symbol, None)
            if not self.config.trading.dry_run:
                self._cancel_all_symbol_orders(symbol)
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
        adjusted_signal = self._btc_market_adjusted_signal(signal, rank_score, momentum_pct, volume_ratio)
        if adjusted_signal is None:
            self.log(f"{symbol}: BTC大盘方向过滤拒绝 {signal.direction.name} ({signal.reason})")
            return None
        if adjusted_signal is not signal:
            signal = adjusted_signal
            rank_score, momentum_pct, volume_ratio = self._entry_rank_metrics(signal, candles)
        return EntryCandidate(symbol, signal, candles[-1], rank_score, momentum_pct, volume_ratio, filter_reason)

    def _open_entry_candidate(self, candidate: EntryCandidate, account: AccountSnapshot) -> bool:
        quantity, reason = self._size_order(candidate.symbol, candidate.candle.close, candidate.signal, account)
        if float(quantity) <= 0:
            self.log(f"{candidate.symbol}: 跳过开仓 ({reason})")
            return False

        self.log(
            f"{candidate.symbol}: 选择开仓候选 rank={candidate.rank_score:.2f} "
            f"动能={candidate.directional_momentum_pct * 100:+.2f}% 量能={candidate.volume_ratio:.2f}x"
        )
        self._enter_position(candidate.symbol, candidate.signal, candidate.candle, quantity)
        return True

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

    def _market_regime(self) -> MarketRegime:
        now = time.time()
        cached = self._market_regime_cache
        if cached and now - cached[0] < 30.0:
            return cached[1]

        strategy = self.config.strategy
        if not getattr(strategy, "weak_market_long_filter_enabled", False):
            regime = MarketRegime(False, 1.0, 0.0, "weak_market_disabled")
            self._market_regime_cache = (now, regime)
            return regime

        symbols = tuple(self.config.trading.entry_symbols or self.config.trading.symbols)
        lookback = max(1, int(getattr(strategy, "weak_market_lookback_bars", 48)))
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
        limit = max(self.config.strategy.slow_ema + lookback + 5, self.config.strategy.atr_period + lookback + 5, 120)
        candles = self._closed_candles_for_timeframe(symbol, timeframe, limit)
        if len(candles) <= lookback or len(candles) < max(self.config.strategy.slow_ema, self.config.strategy.atr_period):
            state = BtcMarketState(Direction.FLAT, 0.0, 0.0, "btc_market_warmup")
            self._btc_market_state_cache = (now, state)
            return state

        closes = [candle.close for candle in candles]
        slow = ema(closes, self.config.strategy.slow_ema)
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
        state = BtcMarketState(direction, return_pct * 100.0, slope_atr, f"{label} ret={return_pct * 100:.2f}% slope_atr={slope_atr:.2f}")
        self._btc_market_state_cache = (now, state)
        return state

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
        stop_pct = max(atr_pct * self.config.strategy.stop_loss_atr, 0.0008)
        take_profit_pct = max(atr_pct * self.config.strategy.take_profit_atr, stop_pct * 1.05)
        indicator_holding_bars = max(
            1,
            min(
                self.config.strategy.max_holding_bars,
                max(1, self.config.strategy.indicator_max_holding_bars),
            ),
        )
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
            return Signal(
                Direction.LONG,
                0.7,
                f"indicator_long_macd_golden_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.confirmed_cross_risk_multiplier * self.config.strategy.long_risk_bias * indicator_size_multiplier,
                max_holding_bars=indicator_holding_bars,
            )
        if short_cross:
            if self.config.strategy.short_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_short_disabled", 0.0, 0.0)
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
            return Signal(
                Direction.SHORT,
                0.7,
                f"indicator_short_macd_dead_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.confirmed_cross_risk_multiplier * self.config.strategy.short_risk_bias * indicator_size_multiplier,
                max_holding_bars=indicator_holding_bars,
            )
        if long_pre_cross:
            if self.config.strategy.long_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_long_disabled", 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.LONG, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            return Signal(
                Direction.LONG,
                0.45,
                f"indicator_long_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.pre_cross_risk_multiplier * self.config.strategy.long_risk_bias * indicator_size_multiplier,
                max_holding_bars=max(1, min(indicator_holding_bars, max(6, self.config.strategy.max_holding_bars // 2))),
            )
        if short_pre_cross:
            if self.config.strategy.short_risk_bias <= 0:
                return Signal(Direction.FLAT, 0.0, "indicator_short_disabled", 0.0, 0.0)
            guard_reason = self._indicator_trend_guard_reason(Direction.SHORT, closes, atr_values)
            if guard_reason:
                return Signal(Direction.FLAT, 0.0, guard_reason, 0.0, 0.0)
            return Signal(
                Direction.SHORT,
                0.45,
                f"indicator_short_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.pre_cross_risk_multiplier * self.config.strategy.short_risk_bias * indicator_size_multiplier,
                max_holding_bars=max(1, min(indicator_holding_bars, max(6, self.config.strategy.max_holding_bars // 2))),
            )
        return Signal(
            Direction.FLAT,
            0.0,
            f"indicator_no_extreme rsi={current_rsi:.1f} macd={current_macd:.6g}/{current_signal:.6g} kdj={current_k:.1f}/{current_d:.1f}",
            0.0,
            0.0,
        )

    def _manage_existing_position(self, symbol: str, position: LivePosition, account: AccountSnapshot) -> None:
        candles = self._closed_candles(symbol)
        if len(candles) < VolatilityBreakoutScalper(self.config.strategy).warmup_bars:
            return
        strategy = VolatilityBreakoutScalper(self.config.strategy)
        strategy.prepare(candles)
        signal = strategy.signal(len(candles) - 1, candles)
        profit_exit_reason = self._profit_exit_reason(position, candles, state=self._profit_state_for(symbol, position))
        if profit_exit_reason:
            self.log(f"{symbol}: 盈利保护触发，准备平仓 ({profit_exit_reason})")
            self._exit_position(symbol, position, profit_exit_reason)
            return
        trend_loss_reason = self._trend_loss_exit_reason(position, candles[-1].close)
        if trend_loss_reason:
            self.log(f"{symbol}: 趋势单亏损保护触发，准备平仓 ({trend_loss_reason})")
            self._exit_position(symbol, position, trend_loss_reason)
            return
        false_position_reason = self._false_breakout_reason(position.direction, candles)
        if false_position_reason and _position_profit_pct(position, candles[-1].close) < 0:
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
        if _position_profit_pct(position, candles[-1].close) >= self.config.trading.scale_in_min_profit_pct:
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
                    profit_reason = self._profit_exit_reason(position, candles, current_candle=candle)
                    if profit_reason:
                        exit_price = candle.close
                        reason = profit_reason
                    else:
                        trend_loss_reason = self._trend_loss_exit_reason(position, candle.close)
                        if trend_loss_reason:
                            exit_price = candle.close
                            reason = trend_loss_reason

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
        leverage = max(1, int(leverage_override or self.config.trading.leverage))
        notional = float(quantity) * candle.close
        margin = notional / leverage
        action = f"补仓({scale_label})" if scale_in and scale_label else "补仓" if scale_in else "开仓"
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
                existing.last_checked_time = max(existing.last_checked_time, candle.timestamp)
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
                entry_time=candle.timestamp,
                last_checked_time=candle.timestamp,
                best_price=candle.close,
                leverage=leverage,
                entry_reason=signal.reason,
            )
            self._entry_reasons[symbol] = signal.reason
            self._position_opened_at[symbol] = candle.timestamp
            self._known_active_symbols.add(symbol)
            self.log(f"{symbol}: dry-run 已记录虚拟仓 stop={stop:.6g} take_profit={take_profit:.6g}")
            return

        if not self._prepare_symbol(symbol, leverage_override=leverage_override):
            self.log(f"{symbol}: 跳过下单，当前杠杆/保证金模式不支持")
            return
        if not scale_in:
            self._cancel_all_symbol_orders(symbol)
        response = self.client.new_market_order(symbol, side, quantity, reduce_only=False)
        entry_price = self._entry_price_from_response(response, candle.close)
        self._known_active_symbols.add(symbol)
        if not scale_in:
            self._entry_reasons[symbol] = signal.reason
            self._position_opened_at[symbol] = candle.timestamp
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
        self._scale_in_counts.pop(symbol, None)
        self._last_scale_in_ts.pop(symbol, None)
        self._profit_states.pop(symbol, None)
        self._entry_reasons.pop(symbol, None)
        self._position_opened_at.pop(symbol, None)
        self._mark_symbol_reentry_cooldown(symbol, reason)
        self._known_active_symbols.discard(symbol)
        self.log(f"{symbol}: dry-run 虚拟平仓 exit={exit_price:.6g} pnl={pnl:+.4f}U reason={reason}")
        self._log_session_stats(self.snapshot_account(), force=True)

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

        remaining_margin = account.equity * self.config.risk.max_account_margin_usage_pct - account.initial_margin
        if remaining_margin <= 0:
            return "0", "margin_usage_limit"

        existing_notional = existing_position.notional if existing_position else 0.0
        risk_weight = signal_risk_weight(signal.confidence, signal.risk_multiplier)
        drawdown_multiplier = self._soft_drawdown_size_multiplier(account)
        if drawdown_multiplier <= 0:
            return "0", "soft_drawdown_stop"
        risk_notional = account.equity * self.config.risk.risk_per_trade_pct * risk_weight * drawdown_multiplier / signal.stop_loss_pct
        leverage = max(self.config.trading.leverage, 1)
        symbol_margin_notional = account.equity * self.config.risk.max_symbol_margin_pct * leverage * drawdown_multiplier
        policy_notional_cap = self.config.risk.max_position_notional_usdt
        if policy_notional_cap <= 0:
            policy_notional_cap = float("inf")
        min_initial_margin_notional = account.equity * self.config.risk.min_symbol_margin_pct * leverage
        if policy_notional_cap < float("inf"):
            min_initial_margin_notional = min(min_initial_margin_notional, policy_notional_cap)
        remaining_margin_notional = remaining_margin * self.config.trading.leverage
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
        if account.initial_margin_usage_pct >= self.config.risk.max_account_margin_usage_pct:
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

    def _prepare_symbol(self, symbol: str, leverage_override: int | None = None) -> bool:
        leverage = max(1, int(leverage_override or self.config.trading.leverage))
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


def _is_indicator_reversal_entry_reason(reason: str) -> bool:
    return reason.lower().startswith("indicator_")


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
