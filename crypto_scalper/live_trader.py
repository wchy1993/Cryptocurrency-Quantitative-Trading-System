from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .binance_client import BinanceApiError, BinanceFuturesClient
from .indicators import adx, atr, bollinger_bands, ema, kdj, macd, percentile_rank, rolling_high, rolling_low, rsi, vwap
from .live_config import LiveAppConfig
from .market_filters import MultiTimeframeFilter, TimeframeSignal
from .models import Candle, Direction, Signal
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
        if self.equity <= 0:
            return 1.0
        return self.initial_margin / self.equity


@dataclass(frozen=True)
class MarketState:
    mode: str
    direction: Direction
    reason: str


@dataclass(frozen=True)
class EntryCandidate:
    symbol: str
    state: MarketState
    signal: Signal
    candle: Candle
    score: float
    edge_reason: str


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
    entry_fee: float = 0.0
    mode: str = "unknown"
    initial_stop_distance: float = 0.0
    target_price: float | None = None
    bars_held: int = 0
    scale_ins: int = 0


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


@dataclass
class ConditionCounter:
    checked: int = 0
    passed: int = 0


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
        self._prepared_symbols: set[str] = set()
        self._disabled_symbols: set[str] = set()
        self._entry_timestamps: list[float] = []
        self._symbol_entry_timestamps: dict[str, list[float]] = {}
        self._last_entry_candle_time: dict[str, datetime] = {}
        self._last_symbol_exit_ts: dict[str, float] = {}
        self._last_frequency_log_ts = 0.0
        self._day = datetime.now(timezone.utc).date()
        self._day_start_equity = config.risk.starting_capital_usdt
        self._peak_equity = config.risk.starting_capital_usdt
        self._cooldown_until = 0.0
        self._sim_positions: dict[str, SimPosition] = {}
        self._scale_in_counts: dict[str, int] = {}
        self._last_scale_in_ts: dict[str, float] = {}
        self._position_modes: dict[str, str] = {}
        self._position_initial_stop_distances: dict[str, float] = {}
        self._position_entry_timestamps: dict[str, float] = {}
        self._position_max_holding_bars: dict[str, int] = {}
        self._symbol_loss_counts: dict[str, int] = {}
        self._symbol_cooldown_until: dict[str, float] = {}
        self._consecutive_losses = 0
        self._mtf_filter = MultiTimeframeFilter(config.filters)
        self._mtf_candle_cache: dict[tuple[str, str], tuple[float, list[Candle]]] = {}
        self._profit_states: dict[str, ProfitState] = {}
        self._session_peak_pnl = 0.0
        self.stats = SessionStats(datetime.now(), config.risk.starting_capital_usdt)
        self._last_stats_log_ts = 0.0
        self._condition_stats: dict[str, ConditionCounter] = {}
        self._last_condition_stats_log_ts = 0.0

    def run_forever(self, stop_event: threading.Event) -> None:
        self.validate_startup()
        self.log("交易循环已启动")
        self._log_session_stats(self.snapshot_account(), force=True)
        self._log_condition_stats(force=True)
        while not stop_event.is_set():
            started = time.time()
            try:
                self.run_once()
            except BinanceApiError as exc:
                self.log(f"Binance API 错误: {exc}")
            except Exception as exc:
                self.log(f"运行错误: {type(exc).__name__}: {exc}")

            elapsed = time.time() - started
            stop_event.wait(max(1.0, self.config.trading.poll_seconds - elapsed))
        self._log_session_stats(self.snapshot_account(), force=True)
        self._log_condition_stats(force=True)
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
        for symbol in trading.symbols:
            try:
                self._prepare_symbol(symbol)
                prepared += 1
            except BinanceApiError as exc:
                if self._is_unknown_symbol_error(exc):
                    self._disabled_symbols.add(symbol)
                    self.log(f"{symbol}: skipped unsupported symbol ({exc})")
                    continue
                raise
        if prepared <= 0:
            raise RuntimeError("no configured symbols could be prepared")

    def run_once(self) -> None:
        account = self.snapshot_account()
        self._update_loss_limits(account)
        self.log(
            f"权益={account.equity:.2f}U 可用={account.available_balance:.2f}U "
            f"保证金占用={account.margin_usage_pct * 100:.2f}% 持仓={len(account.positions)}"
        )

        if self.config.trading.dry_run:
            self._manage_sim_positions()
            account = self.snapshot_account()

        active_symbols = set(account.positions)
        for stale_symbol in set(self._scale_in_counts) - active_symbols:
            self._scale_in_counts.pop(stale_symbol, None)
            self._last_scale_in_ts.pop(stale_symbol, None)
        for stale_symbol in set(self._profit_states) - active_symbols:
            self._profit_states.pop(stale_symbol, None)
        for stale_symbol in set(self._position_modes) - active_symbols:
            self._position_modes.pop(stale_symbol, None)
            self._position_initial_stop_distances.pop(stale_symbol, None)
            self._position_entry_timestamps.pop(stale_symbol, None)
            self._position_max_holding_bars.pop(stale_symbol, None)

        if self.account_callback:
            self.account_callback(account)

        self._log_session_stats(account)

        for symbol in self.config.trading.symbols:
            if symbol in account.positions:
                self._manage_existing_position(symbol, account.positions[symbol], account)

        account = self.snapshot_account()
        if self.account_callback:
            self.account_callback(account)
        if self._session_profit_guard_closes_positions(account):
            return

        if not self._global_risk_allows_trading(account):
            return

        remaining_slots = max(0, self.config.trading.max_open_positions - len(account.positions))
        if remaining_slots <= 0:
            return

        global_state = self._classify_global_market_state()
        self._record_condition("btc_market_state", global_state.mode in {"trend", "range", "opportunity"})
        candidates: list[EntryCandidate] = []
        for symbol in self.config.trading.symbols:
            if symbol in self._disabled_symbols:
                continue
            if symbol in account.positions:
                continue
            try:
                candidate = self._entry_candidate(symbol, account, global_state)
            except BinanceApiError as exc:
                if self._is_unknown_symbol_error(exc):
                    self._disabled_symbols.add(symbol)
                    self.log(f"{symbol}: skipped unsupported symbol ({exc})")
                    continue
                raise
            if candidate is not None:
                candidates.append(candidate)

        self._log_condition_stats()
        if not candidates:
            return

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        batch_limit = max(1, self.config.trading.candidate_batch_size)
        opened = 0
        planned_directions = [position.direction for position in account.positions.values()]
        max_to_open = min(batch_limit, remaining_slots)
        for candidate in candidates:
            if opened >= max_to_open:
                break
            allowed, reason = self._direction_capacity_allows(candidate.signal.direction, planned_directions)
            if not allowed:
                self.log(f"{candidate.symbol}: 同方向组合限制，跳过候选信号 ({reason})")
                continue
            quantity, reason = self._size_order(candidate.symbol, candidate.candle.close, candidate.signal, account)
            if float(quantity) <= 0:
                self.log(f"{candidate.symbol}: 跳过开仓 ({reason})")
                continue
            self.log(
                f"{candidate.symbol}: 选中候选 score={candidate.score:.1f} 市场模式={candidate.state.mode} "
                f"信号={candidate.signal.reason} ({candidate.state.reason}; {candidate.edge_reason})"
            )
            self.log(f"{candidate.symbol}: 仓位计算 {reason}")
            self._enter_position(candidate.symbol, candidate.signal, candidate.candle, quantity)
            planned_directions.append(candidate.signal.direction)
            opened += 1

        if self.config.trading.dry_run:
            account = self.snapshot_account()
            if self.account_callback:
                self.account_callback(account)
            self._log_session_stats(account)

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
        for sim in self._sim_positions.values():
            mark = self._latest_close(sim.symbol)
            unrealized = sim.direction.value * sim.quantity * (mark - sim.entry_price)
            notional = abs(sim.quantity * mark)
            margin = notional / max(self.config.trading.leverage, 1)
            total_unrealized += unrealized
            initial_margin += margin
            live = LivePosition(
                symbol=sim.symbol,
                position_side="SIM",
                direction=sim.direction,
                quantity=sim.quantity,
                entry_price=sim.entry_price,
                mark_price=mark,
                notional=notional,
                unrealized_pnl=unrealized,
                leverage=self.config.trading.leverage,
                margin_type=self.config.trading.margin_type,
                liquidation_price=None,
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
            maintenance_margin=0.0,
            total_unrealized_pnl=total_unrealized,
            positions=positions,
            position_rows=tuple(rows),
            position_mode="dry-run 模拟仓",
        )

    def _maybe_enter_symbol(self, symbol: str, account: AccountSnapshot) -> None:
        candidate = self._entry_candidate(symbol, account, self._classify_global_market_state())
        if candidate is None:
            return
        quantity, reason = self._size_order(symbol, candidate.candle.close, candidate.signal, account)
        if float(quantity) <= 0:
            self.log(f"{symbol}: 跳过开仓 ({reason})")
            return
        self.log(f"{symbol}: 市场模式={candidate.state.mode} 信号={candidate.signal.reason} ({candidate.state.reason}; {candidate.edge_reason})")
        self.log(f"{symbol}: 仓位计算 {reason}")
        self._enter_position(symbol, candidate.signal, candidate.candle, quantity)

    def _entry_candidate(self, symbol: str, account: AccountSnapshot, global_state: MarketState) -> EntryCandidate | None:
        symbol_allowed, symbol_reason = self._symbol_allows_trading(symbol)
        if not symbol_allowed:
            self.log(f"{symbol}: 暂停该币种开仓 ({symbol_reason})")
            return None

        if len(account.positions) >= self.config.trading.max_open_positions:
            self.log(f"{symbol}: 已达到最大同时持仓数量")
            return None

        candles = self._closed_candles(symbol)
        if len(candles) < VolatilityBreakoutScalper(self.config.strategy).warmup_bars:
            self.log(f"{symbol}: K 线不足，等待")
            return None

        state = self._classify_market_state(symbol, global_state=global_state)
        signal = self._dual_mode_signal(candles, state)
        if signal.direction == Direction.FLAT:
            self.log(f"{symbol}: 不开仓 ({state.reason}; {signal.reason})")
            return None

        frequency_ok, frequency_reason = self._symbol_entry_frequency_allows(symbol, candles[-1].timestamp)
        if not frequency_ok:
            self.log(f"{symbol}: 交易频率保护 ({frequency_reason})")
            return None

        allowed, reason = self._direction_capacity_allows(signal.direction, [position.direction for position in account.positions.values()])
        if not allowed:
            self.log(f"{symbol}: 同方向组合限制 ({reason})")
            return None

        btc_direction_ok = self._btc_direction_allows(symbol, signal.direction, global_state)
        self._record_condition("btc_direction", btc_direction_ok)
        if not btc_direction_ok and self._condition_enabled("btc_direction"):
            self.log(f"{symbol}: BTC 大盘过滤拒绝 ({global_state.reason}; signal={signal.direction.name})")
            return None

        has_edge, edge_reason = self._signal_has_enough_edge(signal)
        if not has_edge:
            self.log(f"{symbol}: 跳过低边际信号 ({edge_reason})")
            return None

        score = self._signal_score(signal) + self._symbol_priority_bonus(symbol, global_state)
        return EntryCandidate(symbol, state, signal, candles[-1], score, edge_reason)

    @staticmethod
    def _flat_signal(reason: str) -> Signal:
        return Signal(Direction.FLAT, 0.0, reason, 0.0, 0.0)

    @staticmethod
    def _is_unknown_symbol_error(exc: BinanceApiError) -> bool:
        code = None
        if isinstance(exc.payload, dict):
            try:
                code = int(exc.payload.get("code", 0))
            except (TypeError, ValueError):
                code = None
        text = f"{exc} {exc.payload}".lower()
        return code == -1121 or "invalid symbol" in text or "symbol not found" in text

    def _prune_entry_timestamps(self, now: float | None = None) -> None:
        window = max(0, self.config.trading.entry_frequency_window_seconds)
        if window <= 0:
            return
        now = time.time() if now is None else now
        cutoff = now - window
        self._entry_timestamps = [timestamp for timestamp in self._entry_timestamps if timestamp >= cutoff]
        for symbol, timestamps in list(self._symbol_entry_timestamps.items()):
            recent = [timestamp for timestamp in timestamps if timestamp >= cutoff]
            if recent:
                self._symbol_entry_timestamps[symbol] = recent
            else:
                self._symbol_entry_timestamps.pop(symbol, None)

    def _global_entry_frequency_allows(self) -> bool:
        limit = max(0, self.config.trading.max_entries_per_window)
        if limit <= 0:
            return True
        now = time.time()
        self._prune_entry_timestamps(now)
        if len(self._entry_timestamps) < limit:
            return True
        if now - self._last_frequency_log_ts >= 60.0:
            self._last_frequency_log_ts = now
            window_minutes = self.config.trading.entry_frequency_window_seconds / 60.0
            self.log(f"交易频率保护: 最近 {window_minutes:.0f} 分钟已开仓 {len(self._entry_timestamps)}/{limit}，暂停新开仓")
        return False

    def _symbol_entry_frequency_allows(self, symbol: str, candle_time: datetime) -> tuple[bool, str]:
        last_candle_time = self._last_entry_candle_time.get(symbol)
        if last_candle_time is not None and candle_time <= last_candle_time:
            return False, f"same_candle last={last_candle_time}"

        now = time.time()
        last_exit = self._last_symbol_exit_ts.get(symbol, 0.0)
        min_reentry_seconds = max(0, self.config.trading.min_symbol_reentry_seconds)
        if min_reentry_seconds > 0 and last_exit > 0 and now - last_exit < min_reentry_seconds:
            remaining = int(min_reentry_seconds - (now - last_exit))
            return False, f"symbol_reentry_wait {remaining}s"

        limit = max(0, self.config.trading.max_symbol_entries_per_window)
        if limit > 0:
            self._prune_entry_timestamps(now)
            count = len(self._symbol_entry_timestamps.get(symbol, []))
            if count >= limit:
                window_minutes = self.config.trading.entry_frequency_window_seconds / 60.0
                return False, f"symbol_entry_limit {count}/{limit} in {window_minutes:.0f}m"
        return True, "ok"

    def _record_entry(self, symbol: str, candle_time: datetime) -> None:
        if self.config.trading.max_entries_per_window > 0 or self.config.trading.max_symbol_entries_per_window > 0:
            now = time.time()
            self._prune_entry_timestamps(now)
            self._entry_timestamps.append(now)
            self._symbol_entry_timestamps.setdefault(symbol, []).append(now)
        self._last_entry_candle_time[symbol] = candle_time

    def _classify_global_market_state(self) -> MarketState:
        return self._classify_btc_market_state()

    def _classify_btc_market_state(self) -> MarketState:
        config = self.config.filters
        symbol = "BTCUSDT"
        trend_candles = self._closed_candles_for_timeframe(symbol, config.trend_timeframe, config.kline_limit)
        if len(trend_candles) >= 205:
            trend_closes = [candle.close for candle in trend_candles]
            trend_close = trend_closes[-1]
            trend_ema50 = ema(trend_closes, 50)[-1]
            trend_ema200 = ema(trend_closes, 200)[-1]
            trend_adx = adx(trend_candles, config.adx_period)[-1]
            if trend_adx >= config.trend_adx_threshold and trend_close > trend_ema200 and trend_ema50 > trend_ema200:
                return MarketState(
                    "trend",
                    Direction.LONG,
                    f"BTC_{config.trend_timeframe}_trend_long adx={trend_adx:.1f} ema50>ema200",
                )
            if trend_adx >= config.trend_adx_threshold and trend_close < trend_ema200 and trend_ema50 < trend_ema200:
                return MarketState(
                    "trend",
                    Direction.SHORT,
                    f"BTC_{config.trend_timeframe}_trend_short adx={trend_adx:.1f} ema50<ema200",
                )

        range_candles = self._closed_candles_for_timeframe(symbol, config.range_timeframe, config.kline_limit)
        if len(range_candles) < 205:
            return MarketState("no_trade", Direction.FLAT, "btc_market_state_candles_insufficient")

        range_closes = [candle.close for candle in range_candles]
        close = range_closes[-1]
        ema50_value = ema(range_closes, 50)[-1]
        ema200_value = ema(range_closes, 200)[-1]
        ema_distance = abs(ema50_value - ema200_value) / max(close, 1e-12)
        range_adx = adx(range_candles, config.adx_period)[-1]
        _, _, _, bb_widths = bollinger_bands(
            range_closes,
            self.config.strategy.bollinger_period,
            self.config.strategy.bollinger_stddev,
        )
        bb_width_rank = percentile_rank(bb_widths, config.percentile_lookback)
        if range_adx <= config.range_adx_threshold and ema_distance <= config.range_ema_distance_pct:
            return MarketState(
                "range",
                Direction.FLAT,
                f"BTC_{config.range_timeframe}_range adx={range_adx:.1f} ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
            )
        if range_adx >= config.btc_opportunity_adx_threshold:
            if close >= ema50_value:
                return MarketState(
                    "opportunity",
                    Direction.LONG,
                    f"BTC_opportunity_long adx={range_adx:.1f} close>=ema50 ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
                )
            return MarketState(
                "opportunity",
                Direction.SHORT,
                f"BTC_opportunity_short adx={range_adx:.1f} close<ema50 ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
            )

        return MarketState(
            "no_trade",
            Direction.FLAT,
            f"BTC_no_trade adx={range_adx:.1f} ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
        )

    def _classify_market_state(self, symbol: str, global_state: MarketState | None = None) -> MarketState:
        global_state = global_state or self._classify_global_market_state()
        if not self._condition_enabled("btc_market_state"):
            symbol_trend = self._classify_standalone_symbol_trend_state(symbol)
            if symbol_trend.mode == "trend":
                return symbol_trend
            symbol_range = self._classify_symbol_range_state(symbol)
            if symbol_range.mode == "range" or not self._condition_enabled("symbol_range_state"):
                return MarketState("range", Direction.FLAT, f"{symbol_range.reason}; btc_market_state_filter_off")
            return MarketState("no_trade", Direction.FLAT, f"{symbol_trend.reason}; {symbol_range.reason}; btc_market_state_filter_off")
        if global_state.mode == "range":
            symbol_range = self._classify_symbol_range_state(symbol)
            if symbol_range.mode == "range":
                return symbol_range
            if not self._condition_enabled("symbol_range_state"):
                return MarketState("range", Direction.FLAT, f"{symbol_range.reason}; symbol_range_filter_off; {global_state.reason}")
            return MarketState("no_trade", Direction.FLAT, f"{symbol_range.reason}; {global_state.reason}")
        if global_state.mode not in {"trend", "opportunity"} or global_state.direction == Direction.FLAT:
            return global_state

        config = self.config.filters
        trend_candles = self._closed_candles_for_timeframe(symbol, config.trend_timeframe, config.kline_limit)
        if len(trend_candles) < 205:
            return MarketState("no_trade", Direction.FLAT, f"{symbol}_trend_candles_insufficient; {global_state.reason}")

        closes = [candle.close for candle in trend_candles]
        close = closes[-1]
        ema50_value = ema(closes, 50)[-1]
        ema200_value = ema(closes, 200)[-1]
        symbol_adx = adx(trend_candles, config.adx_period)[-1]
        if global_state.direction == Direction.LONG:
            trend_aligned = close > ema200_value and ema50_value > ema200_value
            self._record_condition("symbol_trend_state", trend_aligned)
            if trend_aligned or not self._condition_enabled("symbol_trend_state"):
                return MarketState("trend", Direction.LONG, f"{symbol}_{config.trend_timeframe}_align_long adx={symbol_adx:.1f}; {global_state.reason}")
            if global_state.mode == "opportunity" and close >= ema50_value:
                return MarketState("trend", Direction.LONG, f"{symbol}_{config.range_timeframe}_opportunity_long adx={symbol_adx:.1f}; {global_state.reason}")
            if symbol_adx >= config.trend_adx_threshold and close < ema200_value and ema50_value < ema200_value:
                return MarketState("no_trade", Direction.FLAT, f"{symbol}_opposite_short adx={symbol_adx:.1f}; {global_state.reason}")
            return MarketState("trend", Direction.LONG, f"{symbol}_follow_btc_long adx={symbol_adx:.1f}; {global_state.reason}")

        trend_aligned = close < ema200_value and ema50_value < ema200_value
        self._record_condition("symbol_trend_state", trend_aligned)
        if trend_aligned or not self._condition_enabled("symbol_trend_state"):
            return MarketState("trend", Direction.SHORT, f"{symbol}_{config.trend_timeframe}_align_short adx={symbol_adx:.1f}; {global_state.reason}")
        if global_state.mode == "opportunity" and close <= ema50_value:
            return MarketState("trend", Direction.SHORT, f"{symbol}_{config.range_timeframe}_opportunity_short adx={symbol_adx:.1f}; {global_state.reason}")
        if symbol_adx >= config.trend_adx_threshold and close > ema200_value and ema50_value > ema200_value:
            return MarketState("no_trade", Direction.FLAT, f"{symbol}_opposite_long adx={symbol_adx:.1f}; {global_state.reason}")
        return MarketState("trend", Direction.SHORT, f"{symbol}_follow_btc_short adx={symbol_adx:.1f}; {global_state.reason}")

    def _classify_standalone_symbol_trend_state(self, symbol: str) -> MarketState:
        config = self.config.filters
        trend_candles = self._closed_candles_for_timeframe(symbol, config.trend_timeframe, config.kline_limit)
        if len(trend_candles) < 205:
            self._record_condition("symbol_trend_state", False)
            return MarketState("no_trade", Direction.FLAT, f"{symbol}_trend_candles_insufficient")

        closes = [candle.close for candle in trend_candles]
        close = closes[-1]
        ema50_value = ema(closes, 50)[-1]
        ema200_value = ema(closes, 200)[-1]
        symbol_adx = adx(trend_candles, config.adx_period)[-1]
        long_aligned = symbol_adx >= config.trend_adx_threshold and close > ema200_value and ema50_value > ema200_value
        short_aligned = symbol_adx >= config.trend_adx_threshold and close < ema200_value and ema50_value < ema200_value
        self._record_condition("symbol_trend_state", long_aligned or short_aligned)
        if long_aligned:
            return MarketState("trend", Direction.LONG, f"{symbol}_{config.trend_timeframe}_standalone_long adx={symbol_adx:.1f}")
        if short_aligned:
            return MarketState("trend", Direction.SHORT, f"{symbol}_{config.trend_timeframe}_standalone_short adx={symbol_adx:.1f}")
        return MarketState("no_trade", Direction.FLAT, f"{symbol}_standalone_not_trend adx={symbol_adx:.1f}")

    def _classify_symbol_range_state(self, symbol: str) -> MarketState:
        config = self.config.filters
        range_candles = self._closed_candles_for_timeframe(symbol, config.range_timeframe, config.kline_limit)
        if len(range_candles) < 205:
            self._record_condition("symbol_range_state", False)
            return MarketState("no_trade", Direction.FLAT, f"{symbol}_range_candles_insufficient")

        closes = [candle.close for candle in range_candles]
        close = closes[-1]
        ema50_value = ema(closes, 50)[-1]
        ema200_value = ema(closes, 200)[-1]
        ema_distance = abs(ema50_value - ema200_value) / max(close, 1e-12)
        range_adx = adx(range_candles, config.adx_period)[-1]
        _, _, _, bb_widths = bollinger_bands(closes, self.config.strategy.bollinger_period, self.config.strategy.bollinger_stddev)
        bb_width_rank = percentile_rank(bb_widths, config.percentile_lookback)
        range_ok = (
            range_adx <= config.range_adx_threshold
            and ema_distance <= config.range_ema_distance_pct
            and bb_width_rank <= config.range_bb_width_percentile_max
        )
        self._record_condition("symbol_range_state", range_ok)
        if range_ok:
            return MarketState(
                "range",
                Direction.FLAT,
                f"{symbol}_{config.range_timeframe}_range adx={range_adx:.1f} ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
            )
        return MarketState(
            "no_trade",
            Direction.FLAT,
            f"{symbol}_not_range adx={range_adx:.1f} ema_dist={ema_distance * 100:.2f}% bb_rank={bb_width_rank:.1f}",
        )

    def _dual_mode_signal(self, candles: list[Candle], state: MarketState) -> Signal:
        if state.mode == "trend" and state.direction != Direction.FLAT:
            return self._trend_breakout_signal(candles, state.direction)
        if state.mode == "range":
            return self._bollinger_reversion_signal(candles)
        return self._flat_signal("market_state_no_trade")

    def _trend_breakout_signal(self, candles: list[Candle], direction: Direction) -> Signal:
        strategy = self.config.strategy
        config = self.config.filters
        minimum = max(
            strategy.channel_period + 2,
            strategy.atr_period + 2,
            strategy.volume_period + 2,
            strategy.bollinger_period + 2,
        )
        if len(candles) < minimum:
            return self._flat_signal("trend_warmup")

        candle = candles[-1]
        closes = [candidate.close for candidate in candles]
        highs = [candidate.high for candidate in candles]
        lows = [candidate.low for candidate in candles]
        volumes = [candidate.volume for candidate in candles]
        atr_values = atr(candles, strategy.atr_period)
        atr_value = atr_values[-1]
        if atr_value <= 0 or candle.close <= 0:
            return self._flat_signal("trend_zero_atr")

        atr_pct = atr_value / candle.close
        atr_pct_ok = atr_pct >= strategy.min_atr_pct and (strategy.max_atr_pct <= 0 or atr_pct <= strategy.max_atr_pct)
        self._record_condition("trend_atr_pct", atr_pct_ok)
        if atr_pct < strategy.min_atr_pct and self._condition_enabled("trend_atr_pct"):
            return self._flat_signal(f"trend_atr_too_low {atr_pct * 100:.3f}%")
        if strategy.max_atr_pct > 0 and atr_pct > strategy.max_atr_pct and self._condition_enabled("trend_atr_pct"):
            return self._flat_signal(f"trend_atr_too_high {atr_pct * 100:.3f}%")

        atr_rank = percentile_rank(atr_values, config.percentile_lookback)
        bb_mid_values, _, _, bb_widths = bollinger_bands(closes, strategy.bollinger_period, strategy.bollinger_stddev)
        bb_mid = bb_mid_values[-1]
        bb_rank = percentile_rank(bb_widths, config.percentile_lookback)
        average_volume = sum(volumes[-strategy.volume_period - 1 : -1]) / strategy.volume_period
        volume_ratio = candle.volume / max(average_volume, 1e-12)
        previous = candles[-2]
        breakout_buffer = atr_value * strategy.breakout_buffer_atr
        upper_channel = rolling_high(highs, len(candles) - 1, strategy.channel_period)
        lower_channel = rolling_low(lows, len(candles) - 1, strategy.channel_period)
        stop_pct = max(atr_pct * strategy.stop_loss_atr, self._estimated_round_trip_cost_pct() * 1.5)
        take_profit_pct = max(atr_pct * strategy.take_profit_atr, stop_pct * self.config.trading.min_reward_risk_ratio)
        ema20 = ema(closes, 20)[-1]
        ema50 = ema(closes, 50)[-1] if len(closes) >= 50 else ema(closes, strategy.slow_ema)[-1]
        ema200 = ema(closes, 200)[-1] if len(closes) >= 200 else ema(closes, strategy.slow_ema)[-1]
        rsi_values = rsi(closes, config.rsi_period)
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        entry_adx = adx(candles, config.adx_period)[-1] if len(candles) >= config.adx_period + 2 else 0.0
        pullback_volume_floor = max(0.75, strategy.min_volume_ratio * 0.75)
        pullback_atr_floor = max(10.0, config.trend_atr_percentile_min * 0.6)
        pullback_stop_pct = max(atr_pct * max(1.2, strategy.stop_loss_atr * 0.75), self._estimated_round_trip_cost_pct() * 1.5)
        pullback_take_profit_pct = max(
            atr_pct * max(2.4, strategy.take_profit_atr * 0.55),
            pullback_stop_pct * self.config.trading.min_reward_risk_ratio,
        )
        momentum_volume_floor = max(1.20, strategy.min_volume_ratio * 1.10)
        momentum_stop_pct = max(atr_pct * max(1.35, strategy.stop_loss_atr * 0.80), self._estimated_round_trip_cost_pct() * 1.8)
        momentum_take_profit_pct = max(
            atr_pct * max(3.4, strategy.take_profit_atr * 0.65),
            momentum_stop_pct * self.config.trading.min_reward_risk_ratio,
        )
        continuation_volume_floor = max(0.55, strategy.min_volume_ratio * 0.55)
        continuation_atr_floor = max(20.0, config.trend_atr_percentile_min * 0.8)
        continuation_stop_pct = max(atr_pct * max(1.2, strategy.stop_loss_atr * 0.65), self._estimated_round_trip_cost_pct() * 1.5)
        continuation_take_profit_pct = max(
            atr_pct * max(2.8, strategy.take_profit_atr * 0.45),
            continuation_stop_pct * self.config.trading.min_reward_risk_ratio,
        )

        score = 1
        reasons = ["btc_direction"]

        def add_score_condition(name: str, passed: bool, label: str) -> None:
            nonlocal score
            self._record_condition(name, passed)
            if passed or not self._condition_enabled(name):
                score += 1
                reasons.append(label if passed else f"{label}_off")

        add_score_condition("trend_adx", entry_adx >= config.trend_adx_threshold, f"adx={entry_adx:.1f}")
        add_score_condition("trend_volume", strategy.min_volume_ratio <= 0 or volume_ratio >= strategy.min_volume_ratio, f"vol={volume_ratio:.2f}")
        add_score_condition("trend_atr_rank", atr_rank >= config.trend_atr_percentile_min, f"atr_rank={atr_rank:.1f}")
        self._record_condition("trend_bb_width", bb_rank >= config.trend_bb_width_percentile_min)
        if bb_rank >= config.trend_bb_width_percentile_min:
            reasons.append(f"bb_rank={bb_rank:.1f}")
        if direction == Direction.LONG:
            ema_aligned = candle.close > ema200 and ema50 > ema200
            self._record_condition("trend_ema", ema_aligned)
            if ema_aligned or not self._condition_enabled("trend_ema"):
                score += 1
                reasons.append("ema_align" if ema_aligned else "ema_align_off")
            breakout = (
                candle.close > upper_channel + breakout_buffer
                and current_rsi <= 72.0
                and (candle.close - candle.open) <= 2.0 * atr_value
            )
            pullback_touched = (
                min(candle.low, previous.low) <= ema20 + 0.35 * atr_value
                or min(candle.low, previous.low) <= bb_mid + 0.35 * atr_value
            )
            pullback_holds = candle.close > ema20 and candle.close > ema50
            momentum_resume = candle.close > previous.high or (candle.close > candle.open and current_rsi >= previous_rsi)
            rsi_reset = 45.0 <= current_rsi <= 65.0 and current_rsi >= previous_rsi
            pullback = (
                atr_rank >= pullback_atr_floor
                and volume_ratio >= pullback_volume_floor
                and pullback_touched
                and pullback_holds
                and momentum_resume
                and rsi_reset
            )
            recent_higher_closes = len(closes) >= 4 and closes[-1] > closes[-2] > closes[-3]
            momentum = (
                atr_rank >= pullback_atr_floor
                and bb_rank >= config.trend_bb_width_percentile_min
                and volume_ratio >= momentum_volume_floor
                and candle.close > ema20
                and ema20 > ema50
                and candle.close > previous.close
                and recent_higher_closes
                and 45.0 <= current_rsi <= 68.0
                and candle.close <= ema20 + 1.4 * atr_value
                and (candle.close - candle.open) >= 0.06 * atr_value
            )
            continuation = (
                self.config.trading.trend_continuation_entry_enabled
                and atr_rank >= continuation_atr_floor
                and bb_rank >= max(15.0, config.trend_bb_width_percentile_min * 0.6)
                and volume_ratio >= continuation_volume_floor
                and candle.close >= ema20
                and ema20 >= ema50
                and 44.0 <= current_rsi <= 76.0
                and (candle.close > candle.open or candle.close >= previous.close)
                and candle.close <= ema20 + 2.5 * atr_value
            )
            self._record_condition("trend_continuation", continuation)
            self._record_condition("trend_momentum", momentum)
            setup_ok = breakout or pullback or momentum or continuation
            self._record_condition("trend_setup", setup_ok)
            if not setup_ok and self._condition_enabled("trend_setup"):
                return self._flat_signal(
                    f"trend_long_no_setup score={score} atr_rank={atr_rank:.1f} bb_rank={bb_rank:.1f} vol={volume_ratio:.2f} rsi={current_rsi:.1f}"
                )
            setup = "breakout" if breakout else "momentum" if momentum else "pullback" if pullback else "continuation" if continuation else "no_setup_filter_off"
            signal_score = score + 1
            score_ok = signal_score >= config.trend_score_entry
            if momentum and signal_score < config.trend_score_strong:
                score_ok = False
            self._record_condition("trend_score", score_ok)
            if not score_ok and self._condition_enabled("trend_score"):
                return self._flat_signal(f"trend_long_score_low score={signal_score} setup={setup} {' '.join(reasons)}")
            risk_multiplier = 0.55
            if signal_score >= config.trend_score_normal:
                risk_multiplier = 0.75
            if signal_score >= config.trend_score_strong:
                risk_multiplier = 0.95
            if continuation:
                risk_multiplier = min(risk_multiplier, 0.55)
            if momentum:
                risk_multiplier = min(0.80, max(risk_multiplier, 0.70))
            confidence = min(1.0, 0.50 + signal_score * 0.08 + max(0.0, volume_ratio - 1.0) / 20.0)
            prefix = "trend_breakout_v2" if breakout else "trend_momentum_v2" if momentum else "trend_ema_pullback_v1" if pullback else "trend_continuation_v1"
            return Signal(
                Direction.LONG,
                confidence,
                f"{prefix} score={signal_score} setup={setup} {' '.join(reasons)}",
                stop_pct if breakout else momentum_stop_pct if momentum else pullback_stop_pct if pullback else continuation_stop_pct,
                take_profit_pct if breakout else momentum_take_profit_pct if momentum else pullback_take_profit_pct if pullback else continuation_take_profit_pct,
                risk_multiplier=risk_multiplier,
                max_holding_bars=self.config.trading.trend_continuation_max_holding_bars if continuation else strategy.max_holding_bars,
            )

        ema_aligned = candle.close < ema200 and ema50 < ema200
        self._record_condition("trend_ema", ema_aligned)
        if ema_aligned or not self._condition_enabled("trend_ema"):
            score += 1
            reasons.append("ema_align" if ema_aligned else "ema_align_off")
        breakdown = (
            candle.close < lower_channel - breakout_buffer
            and current_rsi >= 28.0
            and (candle.open - candle.close) <= 2.0 * atr_value
        )
        pullback_touched = (
            max(candle.high, previous.high) >= ema20 - 0.35 * atr_value
            or max(candle.high, previous.high) >= bb_mid - 0.35 * atr_value
        )
        pullback_holds = candle.close < ema20 and candle.close < ema50
        momentum_resume = candle.close < previous.low or (candle.close < candle.open and current_rsi <= previous_rsi)
        rsi_reset = 35.0 <= current_rsi <= 55.0 and current_rsi <= previous_rsi
        pullback = (
            atr_rank >= pullback_atr_floor
            and volume_ratio >= pullback_volume_floor
            and pullback_touched
            and pullback_holds
            and momentum_resume
            and rsi_reset
        )
        recent_lower_closes = len(closes) >= 4 and closes[-1] < closes[-2] < closes[-3]
        momentum = (
            atr_rank >= pullback_atr_floor
            and bb_rank >= config.trend_bb_width_percentile_min
            and volume_ratio >= momentum_volume_floor
            and candle.close < ema20
            and ema20 < ema50
            and candle.close < previous.close
            and recent_lower_closes
                and 32.0 <= current_rsi <= 55.0
                and candle.close >= ema20 - 1.4 * atr_value
                and (candle.open - candle.close) >= 0.06 * atr_value
        )
        continuation = (
            self.config.trading.trend_continuation_entry_enabled
            and atr_rank >= continuation_atr_floor
            and bb_rank >= max(15.0, config.trend_bb_width_percentile_min * 0.6)
            and volume_ratio >= continuation_volume_floor
            and candle.close <= ema20
            and ema20 <= ema50
            and 24.0 <= current_rsi <= 56.0
            and (candle.close < candle.open or candle.close <= previous.close)
            and candle.close >= ema20 - 2.5 * atr_value
        )
        self._record_condition("trend_continuation", continuation)
        self._record_condition("trend_momentum", momentum)
        setup_ok = breakdown or pullback or momentum or continuation
        self._record_condition("trend_setup", setup_ok)
        if not setup_ok and self._condition_enabled("trend_setup"):
            return self._flat_signal(
                f"trend_short_no_setup score={score} atr_rank={atr_rank:.1f} bb_rank={bb_rank:.1f} vol={volume_ratio:.2f} rsi={current_rsi:.1f}"
            )
        setup = "breakout" if breakdown else "momentum" if momentum else "pullback" if pullback else "continuation" if continuation else "no_setup_filter_off"
        signal_score = score + 1
        score_ok = signal_score >= config.trend_score_entry
        if momentum and signal_score < config.trend_score_strong:
            score_ok = False
        self._record_condition("trend_score", score_ok)
        if not score_ok and self._condition_enabled("trend_score"):
            return self._flat_signal(f"trend_short_score_low score={signal_score} setup={setup} {' '.join(reasons)}")
        risk_multiplier = 0.55
        if signal_score >= config.trend_score_normal:
            risk_multiplier = 0.75
        if signal_score >= config.trend_score_strong:
            risk_multiplier = 0.95
        if continuation:
            risk_multiplier = min(risk_multiplier, 0.55)
        if momentum:
            risk_multiplier = min(0.80, max(risk_multiplier, 0.70))
        confidence = min(1.0, 0.50 + signal_score * 0.08 + max(0.0, volume_ratio - 1.0) / 20.0)
        prefix = "trend_breakout_v2" if breakdown else "trend_momentum_v2" if momentum else "trend_ema_pullback_v1" if pullback else "trend_continuation_v1"
        return Signal(
            Direction.SHORT,
            confidence,
            f"{prefix} score={signal_score} setup={setup} {' '.join(reasons)}",
            stop_pct if breakdown else momentum_stop_pct if momentum else pullback_stop_pct if pullback else continuation_stop_pct,
            take_profit_pct if breakdown else momentum_take_profit_pct if momentum else pullback_take_profit_pct if pullback else continuation_take_profit_pct,
            risk_multiplier=risk_multiplier,
            max_holding_bars=self.config.trading.trend_continuation_max_holding_bars if continuation else strategy.max_holding_bars,
        )

    def _bollinger_reversion_signal(self, candles: list[Candle]) -> Signal:
        strategy = self.config.strategy
        config = self.config.filters
        minimum = max(strategy.bollinger_period + 2, strategy.atr_period + 2, config.vwap_period + 2, 22)
        if len(candles) < minimum:
            return self._flat_signal("mean_reversion_warmup")

        candle = candles[-1]
        previous = candles[-2]
        closes = [candidate.close for candidate in candles]
        lows = [candidate.low for candidate in candles]
        highs = [candidate.high for candidate in candles]
        mid, upper, lower, _ = bollinger_bands(closes, strategy.bollinger_period, strategy.bollinger_stddev)
        rsi_values = rsi(closes, config.rsi_period)
        atr_values = atr(candles, strategy.atr_period)
        atr_value = atr_values[-1]
        if atr_value <= 0 or candle.close <= 0:
            return self._flat_signal("mean_reversion_zero_atr")

        ema20 = ema(closes, 20)[-1]
        vwap_value = vwap(candles, config.vwap_period)[-1]
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        stop_floor = self._estimated_round_trip_cost_pct() * 1.5
        bb_long_setup = (
            candle.low < lower[-1]
            and candle.close > lower[-1]
            and previous_rsi <= config.rsi_oversold
            and current_rsi > previous_rsi
            and current_rsi >= config.rsi_long_floor
            and (candle.close > candle.open or candle.close > previous.high)
        )
        self._record_condition("bb_reclaim", bb_long_setup)
        if bb_long_setup and self._condition_enabled("bb_reclaim"):
            targets = [target for target in (mid[-1], vwap_value, ema20) if target > candle.close]
            if not targets:
                return self._flat_signal("mean_reversion_long_no_target")
            target_price = min(targets)
            swing_stop = min(lows[-6:]) - 0.2 * atr_value
            atr_stop = candle.close - strategy.mean_reversion_stop_atr * atr_value
            stop_price = min(swing_stop, atr_stop)
            stop_pct = max((candle.close - stop_price) / candle.close, stop_floor)
            take_profit_pct = (target_price - candle.close) / candle.close
            return Signal(
                Direction.LONG,
                0.72,
                f"bb_reclaim_v2 target={target_price:.6g} rsi={current_rsi:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=0.6,
                max_holding_bars=self.config.trading.mean_reversion_time_stop_bars,
            )

        bb_short_setup = (
            candle.high > upper[-1]
            and candle.close < upper[-1]
            and previous_rsi >= config.rsi_overbought
            and current_rsi < previous_rsi
            and current_rsi <= config.rsi_short_ceiling
            and (candle.close < candle.open or candle.close < previous.low)
        )
        self._record_condition("bb_reclaim", bb_short_setup)
        if bb_short_setup and self._condition_enabled("bb_reclaim"):
            targets = [target for target in (mid[-1], vwap_value, ema20) if target < candle.close]
            if not targets:
                return self._flat_signal("mean_reversion_short_no_target")
            target_price = max(targets)
            swing_stop = max(highs[-6:]) + 0.2 * atr_value
            atr_stop = candle.close + strategy.mean_reversion_stop_atr * atr_value
            stop_price = max(swing_stop, atr_stop)
            stop_pct = max((stop_price - candle.close) / candle.close, stop_floor)
            take_profit_pct = (candle.close - target_price) / candle.close
            return Signal(
                Direction.SHORT,
                0.72,
                f"bb_reclaim_v2 target={target_price:.6g} rsi={current_rsi:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=0.6,
                max_holding_bars=self.config.trading.mean_reversion_time_stop_bars,
            )

        recent_rsi = rsi_values[-4:]
        rsi_long_setup = (
            min(recent_rsi) <= config.rsi_oversold
            and current_rsi > previous_rsi
            and (candle.close > candle.open or candle.close > previous.close)
            and candle.close <= max(mid[-1], vwap_value + 0.5 * atr_value)
        )
        self._record_condition("rsi_extreme", rsi_long_setup)
        if rsi_long_setup and self._condition_enabled("rsi_extreme"):
            targets = [target for target in (mid[-1], vwap_value, ema20) if target > candle.close]
            if not targets:
                return self._flat_signal("rsi_extreme_long_no_target")
            target_price = min(targets)
            swing_stop = min(lows[-5:]) - 0.15 * atr_value
            atr_stop = candle.close - strategy.mean_reversion_stop_atr * atr_value
            stop_price = min(swing_stop, atr_stop)
            stop_pct = max((candle.close - stop_price) / candle.close, stop_floor)
            take_profit_pct = (target_price - candle.close) / candle.close
            confidence = 0.64 if candle.close > previous.high else 0.61
            return Signal(
                Direction.LONG,
                confidence,
                f"rsi_extreme_reversal_v1 target={target_price:.6g} rsi={current_rsi:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=0.45,
                max_holding_bars=self.config.trading.mean_reversion_time_stop_bars,
            )

        rsi_short_setup = (
            max(recent_rsi) >= config.rsi_overbought
            and current_rsi < previous_rsi
            and (candle.close < candle.open or candle.close < previous.close)
            and candle.close >= min(mid[-1], vwap_value - 0.5 * atr_value)
        )
        self._record_condition("rsi_extreme", rsi_short_setup)
        if rsi_short_setup and self._condition_enabled("rsi_extreme"):
            targets = [target for target in (mid[-1], vwap_value, ema20) if target < candle.close]
            if not targets:
                return self._flat_signal("rsi_extreme_short_no_target")
            target_price = max(targets)
            swing_stop = max(highs[-5:]) + 0.15 * atr_value
            atr_stop = candle.close + strategy.mean_reversion_stop_atr * atr_value
            stop_price = max(swing_stop, atr_stop)
            stop_pct = max((stop_price - candle.close) / candle.close, stop_floor)
            take_profit_pct = (candle.close - target_price) / candle.close
            confidence = 0.64 if candle.close < previous.low else 0.61
            return Signal(
                Direction.SHORT,
                confidence,
                f"rsi_extreme_reversal_v1 target={target_price:.6g} rsi={current_rsi:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=0.45,
                max_holding_bars=self.config.trading.mean_reversion_time_stop_bars,
            )

        return self._flat_signal(f"mean_reversion_no_reversal rsi={current_rsi:.1f}")

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
            return Signal(
                Direction.LONG,
                0.7,
                f"indicator_long_macd_golden_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.confirmed_cross_risk_multiplier,
                max_holding_bars=self.config.strategy.max_holding_bars,
            )
        if short_cross:
            return Signal(
                Direction.SHORT,
                0.7,
                f"indicator_short_macd_dead_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.confirmed_cross_risk_multiplier,
                max_holding_bars=self.config.strategy.max_holding_bars,
            )
        if long_pre_cross:
            return Signal(
                Direction.LONG,
                0.45,
                f"indicator_long_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.pre_cross_risk_multiplier,
                max_holding_bars=max(6, self.config.strategy.max_holding_bars // 2),
            )
        if short_pre_cross:
            return Signal(
                Direction.SHORT,
                0.45,
                f"indicator_short_pre_cross rsi={current_rsi:.1f} kdj={current_k:.1f}/{current_d:.1f}",
                stop_pct,
                take_profit_pct,
                risk_multiplier=config.pre_cross_risk_multiplier,
                max_holding_bars=max(6, self.config.strategy.max_holding_bars // 2),
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
        state = self._classify_market_state(symbol)
        signal = self._dual_mode_signal(candles, state)
        dynamic_exit = self._dynamic_exit_reason(symbol, position, candles, state)
        if dynamic_exit:
            self.log(f"{symbol}: 动态退出触发，准备平仓 ({dynamic_exit})")
            self._exit_position(symbol, position, dynamic_exit)
            return
        profit_exit_reason = self._profit_exit_reason(position, candles, state=self._profit_state_for(symbol, position))
        if profit_exit_reason:
            self.log(f"{symbol}: 盈利保护触发，准备平仓 ({profit_exit_reason})")
            self._exit_position(symbol, position, profit_exit_reason)
            return
        if signal.direction != Direction.FLAT and signal.direction != position.direction:
            self.log(f"{symbol}: 出现反向信号，准备市价平仓")
            self._exit_position(symbol, position, "reverse_signal")
            return
        if signal.direction == position.direction and self._signal_mode(signal) == "trend":
            self._maybe_scale_in_position(symbol, position, account, signal, candles)

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
        if profit_pct >= trading.scale_in_min_profit_pct:
            entry_fraction = _scale_fraction(trading.scale_in_entry_fraction, scale_count)
            scale_label = "顺势浮盈"
        elif trading.allow_loss_scale_in and profit_pct <= -trading.loss_scale_in_trigger_pct:
            entry_fraction = trading.loss_scale_in_entry_fraction
            scale_label = "受限亏损"
        else:
            self.log(
                f"{symbol}: 同向信号但暂不补仓 "
                f"(浮动{profit_pct * 100:+.3f}%，未达到浮盈或亏损补仓阈值)"
            )
            return

        has_edge, edge_reason = self._signal_has_enough_edge(signal)
        if not has_edge:
            self.log(f"{symbol}: 同向信号但边际不足，不补仓 ({edge_reason})")
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

    def _sim_execution_price(self, price: float, direction: Direction, is_entry: bool) -> float:
        slippage = max(0.0, self.config.risk.slippage_bps) / 10_000.0
        if slippage <= 0:
            return price
        if direction == Direction.LONG:
            return price * (1.0 + slippage if is_entry else 1.0 - slippage)
        return price * (1.0 - slippage if is_entry else 1.0 + slippage)

    def _sim_fee(self, notional: float) -> float:
        return abs(notional) * max(0.0, self.config.risk.fee_bps) / 10_000.0

    def _condition_enabled(self, name: str) -> bool:
        trading = self.config.trading
        mapping = {
            "btc_market_state": trading.use_btc_market_state_filter,
            "symbol_trend_state": trading.use_symbol_trend_filter,
            "symbol_range_state": trading.use_symbol_range_filter,
            "btc_direction": trading.use_btc_direction_filter,
            "edge_confidence": trading.use_confidence_filter,
            "edge_cost": trading.use_cost_edge_filter,
            "edge_rr": trading.use_reward_risk_filter,
            "trend_atr_pct": trading.use_trend_atr_filter,
            "trend_atr_rank": trading.use_trend_atr_filter,
            "trend_adx": trading.use_trend_adx_filter,
            "trend_volume": trading.use_trend_volume_filter,
            "trend_ema": trading.use_trend_ema_filter,
            "trend_bb_width": True,
            "trend_continuation": trading.trend_continuation_entry_enabled,
            "trend_momentum": True,
            "trend_setup": trading.use_trend_setup_filter,
            "trend_score": trading.use_trend_score_filter,
            "bb_reclaim": trading.use_bollinger_reclaim_entry,
            "rsi_extreme": trading.use_rsi_extreme_entry,
        }
        return mapping.get(name, True)

    def _record_condition(self, name: str, passed: bool) -> bool:
        if not self.config.trading.condition_stats_enabled:
            return passed
        counter = self._condition_stats.setdefault(name, ConditionCounter())
        counter.checked += 1
        if passed:
            counter.passed += 1
        return passed

    def _condition_passes(self, name: str, passed: bool) -> bool:
        self._record_condition(name, passed)
        return passed or not self._condition_enabled(name)

    def _log_condition_stats(self, force: bool = False) -> None:
        if not self.config.trading.condition_stats_enabled or not self._condition_stats:
            return
        now = time.time()
        interval = max(10, self.config.trading.condition_stats_log_interval_seconds)
        if not force and now - self._last_condition_stats_log_ts < interval:
            return
        self._last_condition_stats_log_ts = now
        parts = []
        for name in sorted(self._condition_stats):
            counter = self._condition_stats[name]
            if counter.checked <= 0:
                continue
            rate = counter.passed / counter.checked * 100.0
            suffix = "" if self._condition_enabled(name) else "/OFF"
            parts.append(f"{name}{suffix}={counter.passed}/{counter.checked} {rate:.1f}%")
        if parts:
            self.log("条件触发率: " + "; ".join(parts))

    def _estimated_round_trip_cost_pct(self) -> float:
        one_side_cost = max(0.0, self.config.risk.fee_bps) + max(0.0, self.config.risk.slippage_bps)
        return 2.0 * one_side_cost / 10_000.0

    def _minimum_profit_exit_pct(self) -> float:
        return max(self._estimated_round_trip_cost_pct() * 3.0, self.config.trading.breakeven_lock_pct)

    def _cost_aware_profit_lock_pct(self) -> float:
        return max(self.config.trading.breakeven_lock_pct, self._minimum_profit_exit_pct())

    def _signal_has_enough_edge(self, signal: Signal) -> tuple[bool, str]:
        trading = self.config.trading
        confidence_ok = signal.confidence >= trading.min_signal_confidence
        self._record_condition("edge_confidence", confidence_ok)
        if not confidence_ok and self._condition_enabled("edge_confidence"):
            return False, f"confidence={signal.confidence:.2f}<{trading.min_signal_confidence:.2f}"
        if signal.stop_loss_pct <= 0 or signal.take_profit_pct <= 0:
            return False, "missing_stop_or_take_profit"

        estimated_cost = self._estimated_round_trip_cost_pct()
        minimum_take_profit = estimated_cost * max(1.0, trading.min_take_profit_cost_ratio)
        cost_ok = signal.take_profit_pct >= minimum_take_profit
        self._record_condition("edge_cost", cost_ok)
        if not cost_ok and self._condition_enabled("edge_cost"):
            return (
                False,
                f"tp={signal.take_profit_pct * 100:.3f}%<cost_floor={minimum_take_profit * 100:.3f}%",
            )

        reward_risk = signal.take_profit_pct / signal.stop_loss_pct
        rr_ok = reward_risk >= trading.min_reward_risk_ratio
        self._record_condition("edge_rr", rr_ok)
        if not rr_ok and self._condition_enabled("edge_rr"):
            return False, f"rr={reward_risk:.2f}<{trading.min_reward_risk_ratio:.2f}"
        return True, f"edge_ok cost={estimated_cost * 100:.3f}% min_exit={self._minimum_profit_exit_pct() * 100:.3f}% rr={reward_risk:.2f}"

    def _direction_capacity_allows(self, direction: Direction, directions: list[Direction]) -> tuple[bool, str]:
        if direction == Direction.LONG:
            count = sum(1 for candidate in directions if candidate == Direction.LONG)
            limit = max(0, self.config.trading.max_long_positions)
            return count < limit, f"long_count={count}>={limit}"
        if direction == Direction.SHORT:
            count = sum(1 for candidate in directions if candidate == Direction.SHORT)
            limit = max(0, self.config.trading.max_short_positions)
            return count < limit, f"short_count={count}>={limit}"
        return False, "flat_direction"

    def _btc_direction_allows(self, symbol: str, direction: Direction, btc_state: MarketState) -> bool:
        if symbol == "BTCUSDT" or direction == Direction.FLAT:
            return True
        if btc_state.mode != "trend":
            return True
        if btc_state.direction == Direction.SHORT and direction == Direction.LONG:
            return False
        if btc_state.direction == Direction.LONG and direction == Direction.SHORT:
            return False
        return True

    @staticmethod
    def _signal_score(signal: Signal) -> float:
        parsed_score: float | None = None
        for part in signal.reason.split():
            if part.startswith("score="):
                try:
                    parsed_score = float(part.split("=", 1)[1])
                    break
                except ValueError:
                    break
        base = parsed_score if parsed_score is not None else signal.confidence * 10.0
        reward_risk = signal.take_profit_pct / max(signal.stop_loss_pct, 1e-12)
        edge_bonus = min(1.0, reward_risk / 4.0) + min(0.6, signal.take_profit_pct * 20.0)
        return base + signal.confidence + edge_bonus + signal.risk_multiplier * 0.25

    @staticmethod
    def _symbol_priority_bonus(symbol: str, global_state: MarketState) -> float:
        if global_state.mode not in {"trend", "opportunity"}:
            return 0.0
        priority = {
            "BTCUSDT": 0.90,
            "ETHUSDT": 0.65,
            "SOLUSDT": 0.40,
            "BNBUSDT": 0.25,
            "XRPUSDT": 0.18,
            "ADAUSDT": 0.15,
            "AVAXUSDT": 0.14,
            "LINKUSDT": 0.10,
            "LTCUSDT": 0.08,
            "BCHUSDT": 0.08,
            "DOTUSDT": 0.07,
            "TRXUSDT": 0.07,
            "TONUSDT": 0.06,
            "AAVEUSDT": 0.05,
        }
        penalty = {
            "1000PEPEUSDT": -0.45,
            "1000SHIBUSDT": -0.45,
            "1000BONKUSDT": -0.45,
            "WIFUSDT": -0.35,
            "DOGEUSDT": -0.15,
            "CAKEUSDT": -0.15,
        }
        return priority.get(symbol.upper(), penalty.get(symbol.upper(), 0.0))

    @staticmethod
    def _direction_risk_scale(count: int) -> float:
        if count <= 0:
            return 1.0
        if count == 1:
            return 0.70
        return 0.50

    @staticmethod
    def _signal_mode(signal: Signal) -> str:
        if signal.reason.startswith("trend_breakout"):
            return "trend"
        if signal.reason.startswith("trend_ema_pullback"):
            return "trend"
        if signal.reason.startswith("trend_momentum"):
            return "trend"
        if signal.reason.startswith("trend_continuation"):
            return "trend"
        if signal.reason.startswith("bb_reclaim"):
            return "mean_reversion"
        if signal.reason.startswith("rsi_extreme_reversal"):
            return "mean_reversion"
        return "unknown"

    @staticmethod
    def _signal_target_price(signal: Signal, entry_price: float) -> float:
        if signal.direction == Direction.LONG:
            return entry_price * (1.0 + signal.take_profit_pct)
        if signal.direction == Direction.SHORT:
            return entry_price * (1.0 - signal.take_profit_pct)
        return entry_price

    def _manage_sim_positions(self) -> None:
        for symbol, position in list(self._sim_positions.items()):
            candles = self._closed_candles(symbol)
            new_candles = [candle for candle in candles if candle.timestamp > position.last_checked_time]
            if not new_candles:
                continue
            for candle in new_candles:
                position.bars_held += 1
                position.last_checked_time = candle.timestamp
                candles_until_now = _candles_until(candles, candle)
                self._update_sim_dynamic_stop(position, candles_until_now, candle)
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

                if (
                    exit_price is None
                    and position.mode != "trend"
                    and position.max_holding_bars > 0
                    and position.bars_held >= position.max_holding_bars
                ):
                    exit_price = candle.close
                    reason = "time_stop"

                if exit_price is None:
                    time_exit = self._sim_time_exit_reason(position, candle)
                    if time_exit:
                        exit_price = candle.close
                        reason = time_exit

                if exit_price is not None:
                    self._close_sim_position(symbol, exit_price, reason)
                    break

    def _update_sim_dynamic_stop(self, position: SimPosition, candles: list[Candle], candle: Candle) -> None:
        atr_values = atr(candles, self.config.strategy.atr_period)
        atr_value = atr_values[-1] if atr_values else 0.0
        if atr_value <= 0:
            return
        previous_stop = position.stop_price
        if position.direction == Direction.LONG:
            position.best_price = max(position.best_price, candle.close)
            if position.mode == "trend":
                structure_stop = position.entry_price - position.initial_stop_distance
                current_profit = (candle.close - position.entry_price) / max(position.entry_price, 1e-12)
                profit_r = (candle.close - position.entry_price) / max(position.initial_stop_distance, 1e-12)
                staged_stop = structure_stop
                lock_pct = self._cost_aware_profit_lock_pct()
                if profit_r >= 1.0 and current_profit >= lock_pct:
                    staged_stop = max(staged_stop, position.entry_price * (1.0 + lock_pct))
                if profit_r >= 1.5:
                    trail_stop = position.best_price - self._chandelier_atr_multiplier(position.symbol) * atr_value
                    staged_stop = max(staged_stop, trail_stop)
                position.stop_price = max(position.stop_price, staged_stop)
            elif position.mode == "mean_reversion" and position.bars_held >= self.config.trading.mean_reversion_time_stop_bars:
                position.stop_price = max(position.stop_price, candle.close - 0.6 * atr_value)
        else:
            position.best_price = min(position.best_price, candle.close)
            if position.mode == "trend":
                structure_stop = position.entry_price + position.initial_stop_distance
                current_profit = (position.entry_price - candle.close) / max(position.entry_price, 1e-12)
                profit_r = (position.entry_price - candle.close) / max(position.initial_stop_distance, 1e-12)
                staged_stop = structure_stop
                lock_pct = self._cost_aware_profit_lock_pct()
                if profit_r >= 1.0 and current_profit >= lock_pct:
                    staged_stop = min(staged_stop, position.entry_price * (1.0 - lock_pct))
                if profit_r >= 1.5:
                    trail_stop = position.best_price + self._chandelier_atr_multiplier(position.symbol) * atr_value
                    staged_stop = min(staged_stop, trail_stop)
                position.stop_price = min(position.stop_price, staged_stop)
            elif position.mode == "mean_reversion" and position.bars_held >= self.config.trading.mean_reversion_time_stop_bars:
                position.stop_price = min(position.stop_price, candle.close + 0.6 * atr_value)
        if abs(position.stop_price - previous_stop) / max(position.entry_price, 1e-12) >= 0.0001:
            self.log(f"{position.symbol}: 动态止损 {previous_stop:.6g} -> {position.stop_price:.6g}")

    def _sim_time_exit_reason(self, position: SimPosition, candle: Candle) -> str | None:
        max_holding_bars = position.max_holding_bars or self.config.trading.trend_time_stop_bars
        if position.mode == "trend" and position.bars_held >= max_holding_bars:
            initial_risk = max(position.initial_stop_distance, 1e-12)
            profit_r = position.direction.value * (candle.close - position.entry_price) / initial_risk
            if profit_r < self.config.trading.trend_time_stop_min_r:
                return f"trend_time_stop r={profit_r:.2f}"
        return None

    def _enter_position(
        self,
        symbol: str,
        signal: Signal,
        candle: Candle,
        quantity: str,
        scale_in: bool = False,
        scale_label: str = "",
    ) -> None:
        side = "BUY" if signal.direction == Direction.LONG else "SELL"
        qty = float(quantity)
        entry_price = self._sim_execution_price(candle.close, signal.direction, is_entry=True)
        notional = qty * (entry_price if self.config.trading.dry_run else candle.close)
        action = f"补仓({scale_label})" if scale_in and scale_label else "补仓" if scale_in else "开仓"
        self.log(f"{symbol}: {action} {signal.direction.name} qty={quantity} notional≈{notional:.2f}U reason={signal.reason}")
        if self.config.trading.dry_run:
            entry_fee = self._sim_fee(qty * entry_price)
            self.stats.realized_pnl -= entry_fee
            mode = self._signal_mode(signal)
            initial_stop_distance = entry_price * signal.stop_loss_pct
            target_price = self._signal_target_price(signal, entry_price)
            if signal.direction == Direction.LONG:
                stop = entry_price * (1.0 - signal.stop_loss_pct)
                take_profit = entry_price * (1.0 + signal.take_profit_pct)
            else:
                stop = entry_price * (1.0 + signal.stop_loss_pct)
                take_profit = entry_price * (1.0 - signal.take_profit_pct)
            existing = self._sim_positions.get(symbol)
            if scale_in and existing and existing.direction == signal.direction:
                total_qty = existing.quantity + qty
                if total_qty <= 0:
                    return
                avg_entry = (existing.entry_price * existing.quantity + entry_price * qty) / total_qty
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
                existing.entry_fee += entry_fee
                existing.mode = mode if existing.mode == "unknown" else existing.mode
                existing.initial_stop_distance = max(existing.initial_stop_distance, initial_stop_distance)
                existing.target_price = target_price
                existing.scale_ins += 1
                existing.last_checked_time = max(existing.last_checked_time, candle.timestamp)
                if signal.direction == Direction.LONG:
                    existing.best_price = max(existing.best_price, candle.close)
                else:
                    existing.best_price = min(existing.best_price, candle.close)
                self.log(
                    f"{symbol}: dry-run 已合并虚拟补仓 avg={avg_entry:.6g} fee={entry_fee:.4f}U "
                    f"qty={total_qty:.6g} stop={merged_stop:.6g} take_profit={merged_take_profit:.6g}"
                )
                return
            self._sim_positions[symbol] = SimPosition(
                symbol=symbol,
                direction=signal.direction,
                quantity=qty,
                entry_price=entry_price,
                stop_price=stop,
                take_profit_price=take_profit,
                max_holding_bars=signal.max_holding_bars or self.config.strategy.max_holding_bars,
                entry_time=candle.timestamp,
                last_checked_time=candle.timestamp,
                best_price=candle.close,
                entry_fee=entry_fee,
                mode=mode,
                initial_stop_distance=initial_stop_distance,
                target_price=target_price,
            )
            self._record_entry(symbol, candle.timestamp)
            self.log(
                f"{symbol}: dry-run 已记录虚拟仓 entry={entry_price:.6g} fee={entry_fee:.4f}U "
                f"stop={stop:.6g} take_profit={take_profit:.6g}"
            )
            return

        self._prepare_symbol(symbol)
        response = self.client.new_market_order(symbol, side, quantity, reduce_only=False)
        entry_price = self._entry_price_from_response(response, candle.close)
        self._position_modes[symbol] = self._signal_mode(signal)
        self._position_initial_stop_distances[symbol] = entry_price * signal.stop_loss_pct
        self._position_entry_timestamps[symbol] = time.time()
        self._position_max_holding_bars[symbol] = signal.max_holding_bars or self.config.strategy.max_holding_bars
        if self.config.trading.use_protective_orders:
            self._place_protective_orders(symbol, signal, quantity, entry_price)
        if not scale_in:
            self._record_entry(symbol, candle.timestamp)

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
        self._cancel_symbol_orders(symbol)
        self.client.new_market_order(symbol, side, quantity, reduce_only=self.config.trading.reduce_only_exit)
        self._profit_states.pop(symbol, None)
        self._position_modes.pop(symbol, None)
        self._position_initial_stop_distances.pop(symbol, None)
        self._position_entry_timestamps.pop(symbol, None)
        self._position_max_holding_bars.pop(symbol, None)
        self._last_symbol_exit_ts[symbol] = time.time()
        self.log(f"{symbol}: 已发送 reduce-only 市价平仓 reason={reason}")

    def _close_sim_position(self, symbol: str, exit_price: float, reason: str) -> None:
        position = self._sim_positions.pop(symbol, None)
        if not position:
            return
        execution_price = self._sim_execution_price(exit_price, position.direction, is_entry=False)
        exit_fee = self._sim_fee(position.quantity * execution_price)
        gross_pnl = position.direction.value * position.quantity * (execution_price - position.entry_price)
        net_pnl = gross_pnl - position.entry_fee - exit_fee
        self.stats.closed_trades += 1
        self._on_trade_closed(symbol, net_pnl)
        self.stats.realized_pnl += gross_pnl - exit_fee
        self._scale_in_counts.pop(symbol, None)
        self._last_scale_in_ts.pop(symbol, None)
        self._profit_states.pop(symbol, None)
        self._position_modes.pop(symbol, None)
        self._position_initial_stop_distances.pop(symbol, None)
        self._position_entry_timestamps.pop(symbol, None)
        self._position_max_holding_bars.pop(symbol, None)
        self._last_symbol_exit_ts[symbol] = time.time()
        self.log(
            f"{symbol}: dry-run 虚拟平仓 exit={execution_price:.6g} gross={gross_pnl:+.4f}U "
            f"fees={position.entry_fee + exit_fee:.4f}U pnl={net_pnl:+.4f}U reason={reason}"
        )
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
            self._cancel_symbol_orders(symbol)
            try:
                self.client.new_market_order(symbol, exit_side, quantity, reduce_only=True)
            except BinanceApiError as close_exc:
                self.log(f"{symbol}: 保护单失败且市价平仓失败: {close_exc}")
                raise
            self.log(f"{symbol}: 保护单失败，已发送 reduce-only 市价平仓")

    def _cancel_symbol_orders(self, symbol: str) -> None:
        try:
            self.client.cancel_all_open_orders(symbol)
        except BinanceApiError as exc:
            self.log(f"{symbol}: 普通挂单撤销失败: {exc}")
        cancel_algo_orders = getattr(self.client, "cancel_all_algo_open_orders", None)
        if callable(cancel_algo_orders):
            try:
                cancel_algo_orders(symbol)
            except BinanceApiError as exc:
                self.log(f"{symbol}: Algo 条件单撤销失败: {exc}")

    def _profit_state_for(self, symbol: str, position: LivePosition) -> ProfitState:
        state = self._profit_states.get(symbol)
        if state is None or state.direction != position.direction or abs(state.entry_price - position.entry_price) > 1e-12:
            state = ProfitState(position.direction, position.entry_price, position.entry_price)
            self._profit_states[symbol] = state
        return state

    def _dynamic_exit_reason(self, symbol: str, position: LivePosition, candles: list[Candle], market_state: MarketState) -> str | None:
        if not candles:
            return None
        candle = candles[-1]
        atr_values = atr(candles, self.config.strategy.atr_period)
        atr_value = atr_values[-1] if atr_values else 0.0
        if atr_value <= 0:
            return None

        mode = self._position_modes.get(symbol, "unknown")
        state = self._profit_state_for(symbol, position)
        elapsed_bars = 0
        entry_timestamp = self._position_entry_timestamps.get(symbol)
        if entry_timestamp:
            elapsed_bars = int((time.time() - entry_timestamp) / max(1, _timeframe_seconds(self.config.trading.timeframe)))
        max_holding_bars = self._position_max_holding_bars.get(symbol, self.config.trading.trend_time_stop_bars)
        if position.direction == Direction.LONG:
            state.best_price = max(state.best_price, candle.high, candle.close)
            if mode == "trend":
                initial_risk = self._position_initial_stop_distances.get(symbol, position.entry_price * self.config.strategy.stop_loss_atr * atr_value / max(candle.close, 1e-12))
                initial_stop = position.entry_price - initial_risk
                current_profit = (candle.close - position.entry_price) / max(position.entry_price, 1e-12)
                profit_r = (candle.close - position.entry_price) / max(initial_risk, 1e-12)
                final_stop = initial_stop
                lock_pct = self._cost_aware_profit_lock_pct()
                if profit_r >= 1.0 and current_profit >= lock_pct:
                    final_stop = max(final_stop, position.entry_price * (1.0 + lock_pct))
                if profit_r >= 1.5:
                    chandelier_stop = state.best_price - self._chandelier_atr_multiplier(symbol) * atr_value
                    final_stop = max(final_stop, chandelier_stop)
                if candle.close <= final_stop:
                    return f"trend_staged_stop r={profit_r:.2f} stop={final_stop:.6g}"
                if elapsed_bars >= max_holding_bars and profit_r < self.config.trading.trend_time_stop_min_r:
                    return f"trend_time_stop bars={elapsed_bars} r={profit_r:.2f}"
            if mode == "mean_reversion" and market_state.mode == "trend" and market_state.direction == Direction.SHORT:
                return f"mean_reversion_regime_failure {market_state.reason}"
        else:
            state.best_price = min(state.best_price, candle.low, candle.close)
            if mode == "trend":
                initial_risk = self._position_initial_stop_distances.get(symbol, position.entry_price * self.config.strategy.stop_loss_atr * atr_value / max(candle.close, 1e-12))
                initial_stop = position.entry_price + initial_risk
                current_profit = (position.entry_price - candle.close) / max(position.entry_price, 1e-12)
                profit_r = (position.entry_price - candle.close) / max(initial_risk, 1e-12)
                final_stop = initial_stop
                lock_pct = self._cost_aware_profit_lock_pct()
                if profit_r >= 1.0 and current_profit >= lock_pct:
                    final_stop = min(final_stop, position.entry_price * (1.0 - lock_pct))
                if profit_r >= 1.5:
                    chandelier_stop = state.best_price + self._chandelier_atr_multiplier(symbol) * atr_value
                    final_stop = min(final_stop, chandelier_stop)
                if candle.close >= final_stop:
                    return f"trend_staged_stop r={profit_r:.2f} stop={final_stop:.6g}"
                if elapsed_bars >= max_holding_bars and profit_r < self.config.trading.trend_time_stop_min_r:
                    return f"trend_time_stop bars={elapsed_bars} r={profit_r:.2f}"
            if mode == "mean_reversion" and market_state.mode == "trend" and market_state.direction == Direction.LONG:
                return f"mean_reversion_regime_failure {market_state.reason}"
        return None

    @staticmethod
    def _chandelier_atr_multiplier(symbol: str) -> float:
        return 2.5 if symbol.upper() in {"BTCUSDT", "ETHUSDT", "BNBUSDT"} else 3.0

    def _update_sim_profit_protection(self, position: SimPosition, candle: Candle) -> None:
        if not self.config.trading.profit_exit_enabled:
            return
        previous_stop = position.stop_price
        lock_pct = self._cost_aware_profit_lock_pct()
        trigger_pct = max(self.config.trading.breakeven_trigger_pct, lock_pct)
        if position.direction == Direction.LONG:
            position.best_price = max(position.best_price, candle.high)
            peak_profit = _directional_profit_pct(position.direction, position.entry_price, position.best_price)
            if peak_profit >= trigger_pct:
                position.stop_price = max(
                    position.stop_price,
                    position.entry_price * (1.0 + lock_pct),
                )
            if peak_profit >= self.config.trading.trailing_activation_pct:
                position.stop_price = max(
                    position.stop_price,
                    position.best_price * (1.0 - self.config.trading.trailing_pullback_pct),
                )
        else:
            position.best_price = min(position.best_price, candle.low)
            peak_profit = _directional_profit_pct(position.direction, position.entry_price, position.best_price)
            if peak_profit >= trigger_pct:
                position.stop_price = min(
                    position.stop_price,
                    position.entry_price * (1.0 - lock_pct),
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

        mode = getattr(position, "mode", self._position_modes.get(position.symbol, "unknown"))
        bars_held = getattr(position, "bars_held", None)
        if mode == "trend" and isinstance(bars_held, int) and bars_held < 2:
            return None

        minimum_profit = self._minimum_profit_exit_pct()
        lock_pct = self._cost_aware_profit_lock_pct()
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
            and current_profit > lock_pct
            and pullback >= self.config.trading.trailing_pullback_pct
        ):
            return f"profit_pullback peak={peak_profit * 100:.3f}% now={current_profit * 100:.3f}%"

        if (
            peak_profit >= self.config.trading.breakeven_trigger_pct
            and 0.0 < current_profit <= lock_pct
        ):
            return f"profit_lock now={current_profit * 100:.3f}%"

        if current_profit < self.config.trading.momentum_exit_min_profit_pct:
            return None
        momentum_reason = self._momentum_profit_exit_reason(position.direction, _candles_until(candles, candle))
        if momentum_reason:
            return momentum_reason
        return None

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
        required_score = 3 if loss_scale else 2
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
        same_direction_count = sum(1 for position in account.positions.values() if position.direction == signal.direction)
        if existing_position and existing_position.direction == signal.direction:
            same_direction_count = max(0, same_direction_count - 1)
        direction_scale = self._direction_risk_scale(same_direction_count)
        portfolio_cap_scale = 1.0
        if self.config.risk.risk_per_trade_pct > 0:
            portfolio_cap_scale = self.config.risk.max_portfolio_risk_pct / (self.config.risk.risk_per_trade_pct * max(1, same_direction_count + 1))
        effective_risk_multiplier = max(0.0, min(1.0, signal.risk_multiplier, direction_scale, portfolio_cap_scale))
        risk_notional = account.equity * self.config.risk.risk_per_trade_pct * effective_risk_multiplier / signal.stop_loss_pct
        symbol_margin_notional = account.equity * self.config.risk.max_symbol_margin_pct * self.config.trading.leverage
        remaining_margin_notional = remaining_margin * self.config.trading.leverage
        total_cap = min(risk_notional, symbol_margin_notional, self.config.risk.max_position_notional_usdt)
        remaining_risk_notional = risk_notional - existing_notional
        remaining_symbol_notional = symbol_margin_notional - existing_notional
        remaining_policy_notional = self.config.risk.max_position_notional_usdt - existing_notional
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
        if notional < self.config.risk.min_order_notional_usdt:
            return "0", "below_min_notional"

        rules = self.client.symbol_rules(symbol)
        quantity = rules.round_quantity(notional / price)
        rounded_notional = float(quantity) * price
        if DecimalCompat.less_than(quantity, rules.min_quantity):
            return "0", "below_min_quantity"
        if rounded_notional < max(float(rules.min_notional), self.config.risk.min_order_notional_usdt):
            return "0", "below_exchange_min_notional"
        stop_distance_pct = signal.stop_loss_pct
        risk_usdt = account.equity * self.config.risk.risk_per_trade_pct * effective_risk_multiplier
        margin_usdt = rounded_notional / max(self.config.trading.leverage, 1)
        expected_loss_usdt = rounded_notional * stop_distance_pct
        estimated_fee_open = rounded_notional * max(0.0, self.config.risk.fee_bps) / 10_000.0
        estimated_fee_close = estimated_fee_open
        return (
            quantity,
            "ok "
            f"equity={account.equity:.2f}U risk_pct={self.config.risk.risk_per_trade_pct:.4f} "
            f"risk_usdt={risk_usdt:.2f}U stop_distance={stop_distance_pct * 100:.3f}% "
            f"leverage={self.config.trading.leverage}x notional={rounded_notional:.2f}U "
            f"margin={margin_usdt:.2f}U qty={quantity} expected_loss={expected_loss_usdt:.2f}U "
            f"fee_open≈{estimated_fee_open:.4f}U fee_close≈{estimated_fee_close:.4f}U "
            f"caps=risk:{risk_notional:.2f}/symbol:{symbol_margin_notional:.2f}/policy:{self.config.risk.max_position_notional_usdt:.2f}",
        )

    def _global_risk_allows_trading(self, account: AccountSnapshot) -> bool:
        if account.margin_usage_pct >= self.config.risk.max_account_margin_usage_pct:
            self.log("保证金占用已达到上限，暂停开仓")
            return False
        if account.equity <= self._day_start_equity * (1.0 - self.config.risk.max_daily_loss_pct):
            self.log("触发当日最大亏损限制，暂停开仓")
            return False
        if account.equity <= self._peak_equity * (1.0 - self.config.risk.max_drawdown_pct):
            self.log("触发最大回撤限制，暂停开仓")
            return False
        if not self._global_entry_frequency_allows():
            return False
        return True

    def _session_profit_guard_closes_positions(self, account: AccountSnapshot) -> bool:
        trading = self.config.trading
        if not trading.session_profit_guard_enabled or not account.positions:
            return False
        total_pnl = account.equity - self.stats.starting_equity
        self._session_peak_pnl = max(self._session_peak_pnl, total_pnl)
        if self._session_peak_pnl < trading.session_profit_guard_trigger_usdt:
            return False
        pullback = self._session_peak_pnl - total_pnl
        if pullback < trading.session_profit_guard_pullback_usdt:
            return False
        self.log(
            f"会话盈利回撤保护触发 peak={self._session_peak_pnl:+.4f}U now={total_pnl:+.4f}U "
            f"pullback={pullback:.4f}U，平掉当前持仓"
        )
        for symbol, position in list(account.positions.items()):
            self._exit_position(symbol, position, "session_profit_pullback_guard")
        return True

    def _symbol_allows_trading(self, symbol: str) -> tuple[bool, str]:
        return True, "ok"

    def _on_trade_closed(self, symbol: str, net_pnl: float) -> None:
        if net_pnl > 0:
            self.stats.winning_trades += 1
            self._consecutive_losses = 0
            self._symbol_loss_counts[symbol] = 0
            return

        self.stats.losing_trades += 1
        self._consecutive_losses += 1
        self._symbol_loss_counts[symbol] = self._symbol_loss_counts.get(symbol, 0) + 1

    def _update_loss_limits(self, account: AccountSnapshot) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self._day_start_equity = account.equity
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

    def _prepare_symbol(self, symbol: str) -> None:
        if symbol in self._prepared_symbols:
            return
        if self.config.trading.dry_run:
            self._prepared_symbols.add(symbol)
            return
        try:
            self.client.set_margin_type(symbol, self.config.trading.margin_type)
            self.client.set_leverage(symbol, self.config.trading.leverage)
            self._prepared_symbols.add(symbol)
            self.log(f"{symbol}: 已设置 {self.config.trading.margin_type} / {self.config.trading.leverage}x")
        except BinanceApiError as exc:
            self.log(f"{symbol}: 杠杆或保证金模式设置失败: {exc}")
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


def _position_profit_pct(position: LivePosition, mark_price: float) -> float:
    if position.entry_price <= 0:
        return 0.0
    return _directional_profit_pct(position.direction, position.entry_price, mark_price)


def _directional_profit_pct(direction: Direction, entry_price: float, mark_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return direction.value * (mark_price - entry_price) / entry_price


def _scale_fraction(base_fraction: float, scale_count: int) -> float:
    return max(0.0, min(1.0, base_fraction * (1.0 + 0.25 * max(0, scale_count))))


def _candles_until(candles: list[Candle], candle: Candle) -> list[Candle]:
    return [candidate for candidate in candles if candidate.timestamp <= candle.timestamp]


def _timeframe_seconds(timeframe: str) -> int:
    value = timeframe.strip().lower()
    if not value:
        return 60
    try:
        amount = int(value[:-1])
    except ValueError:
        return 60
    unit = value[-1]
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86_400
    return max(1, amount)


def _format_runtime(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
