from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .binance_client import BinanceApiError, BinanceFuturesClient, SymbolRules
from .live_config import DualThrustShadowConfig, LiveAppConfig, load_live_config
from .live_trader import AccountSnapshot, LivePosition
from .models import Candle, Direction
from .risk import (
    BacktestExecutionConfig,
    conservative_quantity,
    execution_config_from_live_config,
    funding_cashflow,
    market_entry_fill,
    market_exit_fill,
)
from .volatility_breakout import (
    DualThrustSignal,
    VolatilityBreakoutConfig,
    build_dual_thrust_signals,
)


DUAL_THRUST_SHADOW_REASON_TOKEN = "dual_thrust_v2_balanced_50_shadow"
SHADOW_STATE_SCHEMA_VERSION = 1
LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class ShadowCandidate:
    signal: DualThrustSignal
    btc_return_4h: float


def dual_thrust_signal_config(config: DualThrustShadowConfig) -> VolatilityBreakoutConfig:
    return VolatilityBreakoutConfig(
        timeframe_minutes=config.timeframe_minutes,
        lookback_days=config.lookback_days,
        long_k=config.long_k,
        short_k=config.short_k,
        allow_long=config.allow_long,
        allow_short=config.allow_short,
        atr_period=config.atr_period,
        trend_ema_period=config.trend_ema_period,
        max_signals_per_symbol_day=config.max_signals_per_symbol_day,
        stop_atr_multiple=config.stop_atr_multiple,
        take_profit_r=config.take_profit_r,
        fail_fast_minutes=config.fail_fast_minutes,
        fail_fast_min_mfe_r=config.fail_fast_min_mfe_r,
        fail_fast_max_current_r=config.fail_fast_max_current_r,
        max_holding_minutes=config.max_holding_minutes,
    )


def dual_thrust_shadow_config_hash(config: LiveAppConfig) -> str:
    risk = config.risk
    payload = {
        "dual_thrust_shadow": asdict(config.dual_thrust_shadow),
        "symbols": list(config.dual_thrust_shadow.enabled_symbols),
        "safety": {
            "environment": config.exchange.environment,
            "dry_run": config.trading.dry_run,
            "leverage": config.trading.leverage,
            "legacy_strategy_flags": {
                "super_volume": config.strategy.super_volume_breakout_enabled,
                "startup_breakout": config.strategy.startup_breakout_enabled,
                "ordinary_breakout": config.strategy.ordinary_breakout_enabled,
                "pullback_reclaim": config.strategy.pullback_reclaim_enabled,
                "fast_breakout": config.strategy.fast_breakout_enabled,
                "low_base_ignition": bool(
                    getattr(config.strategy, "low_base_volume_ignition_enabled", False)
                ),
                "spike_trade": config.strategy.spike_trade_enabled,
                "rsi_reversal": config.strategy.rsi_reversal_enabled,
                "mtf": config.strategy.mtf_4h_rsi_regime_enabled,
                "mtf_reset": config.strategy.mtf_momentum_reset_enabled,
                "oi_flush": config.strategy.oi_flush_reversal_enabled,
            },
            "vbp": config.vbp_strategy.enabled,
            "reversal": config.reversal_alpha.enabled,
            "cmipr": config.cmipr.enabled,
            "mtper": config.mtper.enabled,
            "mtpc": config.mtpc.enabled,
            "macro": config.macro_events.enabled,
        },
        "execution": {
            "mode": risk.backtest_mode,
            "cost_experiment": risk.cost_experiment,
            "market_slippage_bps": risk.market_slippage_bps,
            "stop_slippage_bps": risk.stop_slippage_bps,
            "take_profit_slippage_bps": risk.take_profit_slippage_bps,
            "maker_fee_rate": risk.maker_fee_rate,
            "taker_fee_rate": risk.taker_fee_rate,
            "funding_enabled": risk.funding_enabled,
            "dynamic_slippage_enabled": risk.dynamic_slippage_enabled,
            "impact_coefficient_bps": risk.impact_coefficient_bps,
            "impact_exponent": risk.impact_exponent,
            "max_bar_participation_rate": risk.max_bar_participation_rate,
            "min_partial_fill_ratio": risk.min_partial_fill_ratio,
        },
        "execution_order": [
            "closed_1m_only",
            "gap_stop",
            "fail_fast_or_time_stop_at_open",
            "stop_before_take_profit_on_conflict",
            "next_signal_scan_market_execution",
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def _closed_candles(candles: list[Candle], minutes: int, now: datetime) -> list[Candle]:
    duration = timedelta(minutes=minutes)
    return [candle for candle in candles if candle.timestamp + duration <= now]


def _raw_target_for_full_cost_r(
    direction: Direction,
    entry_price: float,
    entry_fee_per_unit: float,
    unit_risk: float,
    target_r: float,
    execution: BacktestExecutionConfig,
) -> float:
    target_net = max(0.0, target_r) * unit_risk
    slip = execution.take_profit_slippage_bps / 10_000.0
    fee = execution.taker_fee_rate
    if direction == Direction.LONG:
        denominator = max((1.0 - slip) * (1.0 - fee), 1e-12)
        return (target_net + entry_price + entry_fee_per_unit) / denominator
    denominator = max((1.0 + slip) * (1.0 + fee), 1e-12)
    return max(1e-12, (entry_price - entry_fee_per_unit - target_net) / denominator)


class DualThrustShadowTrader:
    """Isolated, full-cost paper trader for the frozen 50-symbol configuration.

    This class never calls an order endpoint. Its state namespace is tied to the
    complete frozen strategy and execution-cost hash so samples from different
    configurations cannot be combined accidentally.
    """

    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: LogCallback | None = None,
        account_callback: Callable[[AccountSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.shadow = config.dual_thrust_shadow
        self.client = client
        self.logger = logger or (lambda message: None)
        self.account_callback = account_callback
        self.execution = execution_config_from_live_config(
            config,
            cost_experiment="full_cost",
            mode="conservative",
        )
        self.signal_config = dual_thrust_signal_config(self.shadow)
        self.config_hash = dual_thrust_shadow_config_hash(config)
        self.state_path = Path(self.shadow.state_path)
        self.event_log_path = Path(self.shadow.event_log_path)
        self.report_path = Path(self.shadow.report_path)
        self.state = self._load_or_create_state()
        self._last_mark_price = 0.0
        self._last_heartbeat = 0.0
        self._unsupported_symbols: set[str] = set()
        self._funding_reconciled_this_run = False
        self._funding_cache: dict[str, tuple[float, float, bool]] = {}
        self._runtime_lock_handle: Any | None = None
        if not self.state.get("sample_initialized_event_logged", False):
            self._append_event(
                "shadow_sample_started",
                started_at=self.state["started_at"],
                initial_equity=self.state["starting_equity"],
                universe_size=len(self.shadow.enabled_symbols),
            )
            self.state["sample_initialized_event_logged"] = True
            self._persist_state()

    def validate_startup(self) -> None:
        if not self.shadow.enabled:
            raise RuntimeError("dual-thrust shadow strategy is disabled")
        if not self.shadow.shadow_only:
            raise RuntimeError("frozen dual-thrust configuration must remain shadow_only")
        if not self.config.trading.dry_run:
            raise RuntimeError("dual-thrust shadow refuses to start when dry_run=false")
        if self.config.risk.cost_experiment != "full_cost":
            raise RuntimeError("dual-thrust shadow requires full_cost execution")
        if self.config.risk.backtest_mode != "conservative":
            raise RuntimeError("dual-thrust shadow requires conservative execution")
        if self.shadow.max_open_positions != 1:
            raise RuntimeError("frozen 50-symbol balanced shadow requires max_open_positions=1")
        partial_enabled = (
            self.shadow.partial_take_profit_r > 0.0
            or self.shadow.partial_take_profit_fraction > 0.0
        )
        if partial_enabled and not (
            0.0 < self.shadow.partial_take_profit_r < self.shadow.take_profit_r
            and 0.0 < self.shadow.partial_take_profit_fraction < 1.0
        ):
            raise RuntimeError(
                "partial take profit requires 0 < partial R < full target R and fraction in (0, 1)"
            )
        if not self.shadow.enabled_symbols:
            raise RuntimeError("dual-thrust shadow symbol universe is empty")
        if self.config.exchange.environment != "mainnet":
            raise RuntimeError("shadow validation must use mainnet public market data")
        self.client.ping()

    def run_forever(self, stop_event: threading.Event) -> None:
        self.validate_startup()
        self._acquire_runtime_lock()
        try:
            self.log(
                "Hybrid v5 50币 balanced runner shadow 已启动；仅使用主网公开行情，"
                "不会发送订单，旧策略全部旁路"
            )
            self.log(
                f"冻结版本={self.shadow.frozen_version} config_hash={self.config_hash[:16]} "
                f"样本起点={self.state['started_at']}"
            )
            while not stop_event.is_set():
                started = time.time()
                try:
                    self.run_once(stop_event=stop_event)
                except BinanceApiError as exc:
                    self.log(f"Binance 公开行情错误: {exc}")
                except Exception as exc:
                    self.log(f"shadow 运行错误: {type(exc).__name__}: {exc}")
                elapsed = time.time() - started
                stop_event.wait(max(1.0, self.config.trading.poll_seconds - elapsed))
            self._write_report(self.snapshot_account(fetch_mark=False))
            self.log("Dual Thrust shadow 已停止，状态与验收报告已落盘")
        finally:
            self._release_runtime_lock()

    def run_once(self, stop_event: threading.Event | None = None) -> None:
        self.validate_startup_settings_only()
        self._funding_reconciled_this_run = False
        self._reconcile_missing_funding()
        now = _utc_now()
        self._manage_position(now)
        if self.state.get("position") is None:
            self._scan_and_maybe_enter(now, stop_event)
        account = self.snapshot_account(fetch_mark=True)
        self._update_drawdown(account.equity)
        self._persist_state()
        self._write_report(account)
        if self.account_callback:
            self.account_callback(account)
        heartbeat = max(30, int(self.shadow.heartbeat_seconds))
        if time.time() - self._last_heartbeat >= heartbeat:
            self._last_heartbeat = time.time()
            self.log(
                f"shadow统计: 权益={account.equity:.2f}U 平仓={len(self.state['trades'])} "
                f"候选={self.state['candidate_count']} 最大回撤={self.state['max_drawdown_pct'] * 100:.2f}%"
            )

    def validate_startup_settings_only(self) -> None:
        if not self.shadow.enabled or not self.shadow.shadow_only or not self.config.trading.dry_run:
            raise RuntimeError("dual-thrust frozen shadow safety settings are not satisfied")

    def snapshot_account(self, fetch_mark: bool = True) -> AccountSnapshot:
        position = self.state.get("position")
        cash = float(self.state["cash"])
        if not position:
            return AccountSnapshot(
                equity=cash,
                wallet_balance=cash,
                available_balance=cash,
                initial_margin=0.0,
                maintenance_margin=0.0,
                total_unrealized_pnl=0.0,
                positions={},
                position_rows=(),
                position_mode="full-cost shadow",
            )

        raw_mark = self._last_mark_price or float(position["raw_entry_price"])
        if fetch_mark:
            raw_mark = self._current_market_price(str(position["symbol"])) or raw_mark
            self._last_mark_price = raw_mark
        rules = self.client.symbol_rules(str(position["symbol"]))
        direction = Direction[str(position["direction"])]
        quantity = float(position["quantity"])
        mark_fill = market_exit_fill(self.execution, rules, direction, quantity, raw_mark, "market")
        execution_gross = direction.value * quantity * (mark_fill.price - float(position["entry_price"]))
        accrued_funding, funding_complete = self._funding_for_position(position, now=_utc_now())
        if not funding_complete:
            accrued_funding = 0.0
        entry_fee = float(position["entry_fee"])
        wallet_balance = cash + entry_fee
        open_pnl_after_entry = execution_gross - entry_fee - mark_fill.fee + accrued_funding
        equity = wallet_balance + open_pnl_after_entry
        notional = quantity * float(position["entry_price"])
        leverage = max(1, int(self.config.trading.leverage))
        initial_margin = notional / leverage
        maintenance = notional * max(0.0, self.config.risk.estimated_maintenance_margin_rate)
        opened_at = _parse_time(str(position["entry_time"]))
        live_position = LivePosition(
            symbol=str(position["symbol"]),
            position_side="BOTH",
            direction=direction,
            quantity=quantity,
            entry_price=float(position["entry_price"]),
            mark_price=raw_mark,
            notional=notional,
            unrealized_pnl=open_pnl_after_entry,
            leverage=leverage,
            margin_type="SHADOW",
            liquidation_price=None,
            entry_reason=str(position["entry_reason"]),
            opened_at=opened_at.replace(tzinfo=timezone.utc) if opened_at else None,
        )
        return AccountSnapshot(
            equity=equity,
            wallet_balance=wallet_balance,
            available_balance=max(0.0, equity - initial_margin),
            initial_margin=initial_margin,
            maintenance_margin=maintenance,
            total_unrealized_pnl=open_pnl_after_entry,
            positions={live_position.symbol: live_position},
            position_rows=(live_position,),
            position_mode="full-cost shadow",
        )

    def acceptance_report(self, account: AccountSnapshot | None = None) -> dict[str, Any]:
        account = account or self.snapshot_account(fetch_mark=False)
        trades = list(self.state.get("trades", []))
        wins = [row for row in trades if float(row["net_pnl"]) > 0.0]
        losses = [row for row in trades if float(row["net_pnl"]) <= 0.0]
        gross_profit = sum(float(row["net_pnl"]) for row in wins)
        gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
        raw_gross_profit = sum(max(0.0, float(row["gross_pnl"])) for row in trades)
        fees = sum(float(row["fee"]) for row in trades)
        slippage = sum(float(row["slippage"]) for row in trades)
        funding = sum(float(row["funding"]) for row in trades)
        full_cost = fees + slippage - funding
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
        started = _parse_time(str(self.state["started_at"])) or _utc_now()
        elapsed_days = max(0.0, (_utc_now() - started).total_seconds() / 86_400.0)
        funding_complete = all(row.get("funding_status") == "complete" for row in trades)
        criteria = {
            "minimum_30_calendar_days": elapsed_days >= 30.0,
            "minimum_30_closed_trades": len(trades) >= 30,
            "full_cost_funding_complete": funding_complete,
            "profit_factor_above_1_10": profit_factor > 1.10,
            "positive_expectancy": sum(float(row["net_pnl"]) for row in trades) > 0.0,
            "drawdown_below_frozen_hard_limit": float(self.state["max_drawdown_pct"])
            < self.shadow.hard_drawdown_stop_pct,
        }
        enough_data = criteria["minimum_30_calendar_days"] and criteria["minimum_30_closed_trades"]
        status = "collecting_new_shadow_data"
        if enough_data:
            status = "eligible_for_review" if all(criteria.values()) else "historical_shadow_rejected"

        report = {
            "strategy_name": self.shadow.strategy_name,
            "frozen_version": self.shadow.frozen_version,
            "config_hash": self.config_hash,
            "status": status,
            "important_note": (
                "Only observations after started_at belong to this acceptance sample. "
                "The optimized 2026-03-06 through 2026-06-06 period is historical research, not holdout data."
            ),
            "started_at": self.state["started_at"],
            "updated_at": _utc_now().isoformat(),
            "elapsed_days": elapsed_days,
            "universe_size": len(self.shadow.enabled_symbols),
            "candidate_count": int(self.state["candidate_count"]),
            "trade_count": len(trades),
            "open_positions": int(self.state.get("position") is not None),
            "initial_equity": float(self.state["starting_equity"]),
            "current_equity": account.equity,
            "closed_net_pnl": sum(float(row["net_pnl"]) for row in trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "profit_factor": profit_factor,
            "expectancy_usdt": statistics.mean(float(row["net_pnl"]) for row in trades) if trades else 0.0,
            "average_win": statistics.mean(float(row["net_pnl"]) for row in wins) if wins else 0.0,
            "average_loss": statistics.mean(float(row["net_pnl"]) for row in losses) if losses else 0.0,
            "max_drawdown_pct": float(self.state["max_drawdown_pct"]),
            "fee": fees,
            "slippage": slippage,
            "funding": funding,
            "full_cost": full_cost,
            "cost_to_raw_gross_profit_ratio": (
                full_cost / raw_gross_profit if raw_gross_profit > 0 else 0.0
            ),
            "funding_complete": funding_complete,
            "rejected": dict(self.state.get("rejected", {})),
            "criteria": criteria,
            "by_side": self._group_trades(trades, "side"),
            "by_symbol": self._group_trades(trades, "symbol"),
            "by_exit_reason": self._group_trades(trades, "exit_reason"),
            "by_month": self._group_trades(
                [dict(row, month=str(row["exit_time"])[:7]) for row in trades],
                "month",
            ),
            "trades": trades,
        }
        return report

    def log(self, message: str) -> None:
        self.logger(message)

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != SHADOW_STATE_SCHEMA_VERSION:
                raise RuntimeError("dual-thrust shadow state schema mismatch")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError(
                    "dual-thrust shadow state belongs to a different frozen config; "
                    "use a new state_path instead of mixing samples"
                )
            return state

        now = _utc_now().isoformat()
        return {
            "schema_version": SHADOW_STATE_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "frozen_version": self.shadow.frozen_version,
            "started_at": now,
            "starting_equity": float(self.config.risk.starting_capital_usdt),
            "cash": float(self.config.risk.starting_capital_usdt),
            "peak_equity": float(self.config.risk.starting_capital_usdt),
            "max_drawdown_pct": 0.0,
            "last_scanned_available_time": None,
            "seen_events": {},
            "daily_entries": {},
            "cooldown_until": {},
            "position": None,
            "trades": [],
            "candidate_count": 0,
            "rejected": {},
            "sample_initialized_event_logged": False,
        }

    def _acquire_runtime_lock(self) -> None:
        if self._runtime_lock_handle is not None:
            return
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "another Dual Thrust shadow process already owns this frozen sample; "
                "stop it before starting GUI trading"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": _utc_now().isoformat(),
                    "config_hash": self.config_hash,
                },
                ensure_ascii=False,
            )
        )
        handle.flush()
        self._runtime_lock_handle = handle

    def _release_runtime_lock(self) -> None:
        handle = self._runtime_lock_handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._runtime_lock_handle = None

    def _scan_and_maybe_enter(self, now: datetime, stop_event: threading.Event | None) -> None:
        latest_available = now.replace(minute=0, second=0, microsecond=0)
        if (now - latest_available).total_seconds() < max(0, self.shadow.scan_grace_seconds):
            return
        last_scanned = _parse_time(self.state.get("last_scanned_available_time"))
        if last_scanned is not None and last_scanned >= latest_available:
            return

        symbols = tuple(self.shadow.enabled_symbols)
        candles_by_symbol: dict[str, list[Candle]] = {}
        for index, symbol in enumerate(symbols):
            if symbol in self._unsupported_symbols:
                continue
            try:
                rows = self.client.klines(symbol, "1h", 500)
                closed = _closed_candles(rows, 60, now)
                if len(closed) >= max(150, self.shadow.lookback_days * 24 + 50):
                    candles_by_symbol[symbol] = closed
            except Exception as exc:
                if symbol not in self._unsupported_symbols:
                    self.log(f"{symbol}: shadow 1h 数据不可用 ({type(exc).__name__}: {exc})")
                if isinstance(exc, BinanceApiError) and "symbol" in str(exc).lower():
                    self._unsupported_symbols.add(symbol)
            if index + 1 < len(symbols) and self.shadow.request_pacing_seconds > 0:
                if stop_event is not None and stop_event.wait(self.shadow.request_pacing_seconds):
                    return
                if stop_event is None:
                    time.sleep(self.shadow.request_pacing_seconds)

        required = max(5, len(symbols) // 2)
        if len(candles_by_symbol) < required or "BTCUSDT" not in candles_by_symbol:
            self._append_event(
                "scan_incomplete",
                available_time=latest_available.isoformat(),
                loaded_symbols=len(candles_by_symbol),
                required_symbols=required,
            )
            self.log(f"shadow 扫描数据不足 {len(candles_by_symbol)}/{required}，本小时稍后重试")
            return

        btc = candles_by_symbol["BTCUSDT"]
        btc_return_4h = btc[-1].close / max(btc[-5].close, 1e-12) - 1.0
        candidates: list[ShadowCandidate] = []
        for symbol, candles in candles_by_symbol.items():
            signals = build_dual_thrust_signals(symbol, candles, self.signal_config)
            if not signals:
                continue
            signal = signals[-1]
            if signal.signal_available_time != latest_available:
                continue
            if signal.event_id in self.state["seen_events"]:
                continue
            candidates.append(ShadowCandidate(signal, btc_return_4h))

        self.state["last_scanned_available_time"] = latest_available.isoformat()
        self.state["candidate_count"] += len(candidates)
        candidates.sort(key=lambda item: (-item.signal.quality_score, item.signal.symbol))
        opened = False
        for candidate in candidates:
            signal = candidate.signal
            reason = self._candidate_reject_reason(candidate, now)
            if reason is None and not opened:
                opened = self._open_candidate(candidate, now)
                reason = "entered" if opened else "sizing_or_execution_rejected"
            elif reason is None:
                reason = "lower_ranked_same_cycle"
            self.state["seen_events"][signal.event_id] = {
                "time": now.isoformat(),
                "status": reason,
            }
            if reason != "entered":
                self._record_reject(reason)
            self._append_event(
                "candidate_decision",
                event_id=signal.event_id,
                symbol=signal.symbol,
                side=signal.direction.name,
                signal_available_time=signal.signal_available_time.isoformat(),
                decision_time=now.isoformat(),
                status=reason,
                quality_score=signal.quality_score,
                btc_return_4h=btc_return_4h,
            )

    def _candidate_reject_reason(self, candidate: ShadowCandidate, now: datetime) -> str | None:
        signal = candidate.signal
        started = _parse_time(str(self.state["started_at"])) or now
        if signal.signal_available_time < started:
            return "pre_shadow_start"
        age_minutes = (now - signal.signal_available_time).total_seconds() / 60.0
        if age_minutes < 0 or age_minutes > self.shadow.max_signal_age_minutes:
            return "stale_signal"
        directional_btc = signal.direction.value * candidate.btc_return_4h
        if directional_btc > self.shadow.max_directional_btc_return_4h:
            return "btc_4h_overextended"
        if self.state.get("position") is not None:
            return "position_limit"
        current_equity = self.snapshot_account(fetch_mark=False).equity
        peak = max(float(self.state["peak_equity"]), current_equity)
        drawdown = (peak - current_equity) / peak if peak > 0 else 1.0
        if drawdown >= self.shadow.hard_drawdown_stop_pct:
            return "hard_drawdown_stop"
        day_key = now.date().isoformat()
        if int(self.state["daily_entries"].get(day_key, 0)) >= self.shadow.max_daily_trades:
            return "daily_trade_limit"
        cooldown = _parse_time(self.state["cooldown_until"].get(signal.symbol))
        if cooldown is not None and cooldown > now:
            return "symbol_cooldown"
        return None

    def _open_candidate(self, candidate: ShadowCandidate, now: datetime) -> bool:
        signal = candidate.signal
        try:
            one_minute = self.client.klines(signal.symbol, "1m", 3)
            if not one_minute:
                return False
            active = one_minute[-1]
            raw_entry = active.close
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: shadow 成交参考不可用 ({exc})")
            return False
        raw_stop = raw_entry - signal.direction.value * signal.atr_value * self.shadow.stop_atr_multiple
        if raw_entry <= 0.0 or raw_stop <= 0.0:
            return False

        unit_entry = market_entry_fill(self.execution, rules, signal.direction, 1.0, raw_entry)
        unit_stop = market_exit_fill(self.execution, rules, signal.direction, 1.0, raw_stop, "stop_market")
        unit_risk = -(
            signal.direction.value * (unit_stop.price - unit_entry.price)
            - unit_entry.fee
            - unit_stop.fee
        )
        if unit_risk <= 0.0:
            return False
        equity = self.snapshot_account(fetch_mark=False).equity
        side_multiplier = (
            self.shadow.long_risk_multiplier
            if signal.direction == Direction.LONG
            else self.shadow.short_risk_multiplier
        )
        risk_pct = min(
            max(0.0, self.shadow.max_trade_risk_pct),
            max(0.0, self.shadow.risk_per_trade_pct) * max(0.0, side_multiplier),
        )
        requested = equity * risk_pct / unit_risk
        requested = min(
            requested,
            equity * max(0.0, self.shadow.max_notional_multiple) / max(unit_entry.price, 1e-12),
        )
        quantity = conservative_quantity(rules, requested)
        if quantity <= 0 or quantity * unit_entry.price < max(5.0, float(rules.min_notional)):
            return False
        entry_fill = market_entry_fill(
            self.execution,
            rules,
            signal.direction,
            quantity,
            raw_entry,
            active.volume * raw_entry,
        )
        stop_fill = market_exit_fill(self.execution, rules, signal.direction, quantity, raw_stop, "stop_market")
        actual_unit_risk = -(
            signal.direction.value * (stop_fill.price - entry_fill.price)
            - entry_fill.fee / quantity
            - stop_fill.fee / quantity
        )
        if actual_unit_risk <= 0.0:
            return False
        target = _raw_target_for_full_cost_r(
            signal.direction,
            entry_fill.price,
            entry_fill.fee / quantity,
            actual_unit_risk,
            self.shadow.take_profit_r,
            self.execution,
        )
        partial_target = 0.0
        if (
            self.shadow.partial_take_profit_r > 0.0
            and 0.0 < self.shadow.partial_take_profit_fraction < 1.0
        ):
            partial_target = _raw_target_for_full_cost_r(
                signal.direction,
                entry_fill.price,
                entry_fill.fee / quantity,
                actual_unit_risk,
                self.shadow.partial_take_profit_r,
                self.execution,
            )
        risk_usdt = actual_unit_risk * quantity
        reason = (
            f"{DUAL_THRUST_SHADOW_REASON_TOKEN}|event={signal.event_id}|"
            f"quality={signal.quality_score:.4f}"
        )
        self.state["cash"] -= entry_fill.fee
        self.state["position"] = {
            "event_id": signal.event_id,
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "entry_reason": reason,
            "signal_bar_time": signal.signal_bar_time.isoformat(),
            "signal_available_time": signal.signal_available_time.isoformat(),
            "entry_time": now.isoformat(),
            "entry_delay_seconds": max(0.0, (now - signal.signal_available_time).total_seconds()),
            "raw_entry_price": raw_entry,
            "entry_price": entry_fill.price,
            "entry_fee": entry_fill.fee,
            "entry_slippage": entry_fill.slippage_cost,
            "quantity": quantity,
            "original_quantity": quantity,
            "raw_stop_price": raw_stop,
            "take_profit_price": target,
            "partial_take_profit_price": partial_target,
            "partial_take_profit_fraction": self.shadow.partial_take_profit_fraction,
            "partial_taken": False,
            "partial_legs": [],
            "unit_risk": actual_unit_risk,
            "risk_usdt": risk_usdt,
            "original_risk_usdt": risk_usdt,
            "best_price": entry_fill.price,
            "worst_price": entry_fill.price,
            "max_mfe_r": 0.0,
            "max_mae_r": 0.0,
            "last_processed_bar_time": active.timestamp.isoformat(),
            "quality_score": signal.quality_score,
            "btc_return_4h": candidate.btc_return_4h,
            "atr_value": signal.atr_value,
            "volume_ratio": signal.volume_ratio,
            "body_atr": signal.body_atr,
            "range_atr": signal.range_atr,
            "breakout_extension_atr": signal.breakout_extension_atr,
        }
        day_key = now.date().isoformat()
        self.state["daily_entries"] = {
            key: value for key, value in self.state["daily_entries"].items() if key == day_key
        }
        self.state["daily_entries"][day_key] = int(self.state["daily_entries"].get(day_key, 0)) + 1
        self._last_mark_price = raw_entry
        self.log(
            f"{signal.symbol}: shadow开仓 {signal.direction.name} qty={quantity:.8g} "
            f"raw={raw_entry:.8g} fill={entry_fill.price:.8g} stop={raw_stop:.8g} "
            f"风险={risk_usdt:.3f}U ({risk_usdt / max(equity, 1e-12) * 100:.2f}%)"
        )
        self._append_event(
            "entry",
            event_id=signal.event_id,
            symbol=signal.symbol,
            side=signal.direction.name,
            entry_time=now.isoformat(),
            raw_entry_price=raw_entry,
            entry_price=entry_fill.price,
            quantity=quantity,
            risk_usdt=risk_usdt,
            entry_fee=entry_fill.fee,
            entry_slippage=entry_fill.slippage_cost,
        )
        return True

    def _manage_position(self, now: datetime) -> None:
        position = self.state.get("position")
        if not position:
            return
        symbol = str(position["symbol"])
        try:
            candles = self.client.klines(symbol, "1m", 1500)
            closed = _closed_candles(candles, 1, now)
            rules = self.client.symbol_rules(symbol)
        except Exception as exc:
            self.log(f"{symbol}: shadow持仓行情不可用，保留仓位等待恢复 ({exc})")
            self._append_event("position_data_error", symbol=symbol, error=str(exc))
            return
        last_processed = _parse_time(str(position["last_processed_bar_time"]))
        for candle in closed:
            if last_processed is not None and candle.timestamp <= last_processed:
                continue
            position["last_processed_bar_time"] = candle.timestamp.isoformat()
            closed_trade = self._process_closed_bar(position, candle, rules)
            if closed_trade:
                return
        if candles:
            self._last_mark_price = candles[-1].close
            direction = Direction[str(position["direction"])]
            stop = float(position["raw_stop_price"])
            stop_crossed = self._last_mark_price <= stop if direction == Direction.LONG else self._last_mark_price >= stop
            target = float(position["take_profit_price"])
            target_crossed = self._last_mark_price >= target if direction == Direction.LONG else self._last_mark_price <= target
            if stop_crossed:
                self._close_position(position, self._last_mark_price, "stop_loss_live_mark", now, rules)
            elif target_crossed:
                self._close_position(position, self._last_mark_price, "take_profit_live_mark", now, rules)
            elif self._partial_target_crossed(position, self._last_mark_price):
                self._take_partial(position, self._last_mark_price, now, rules)

    def _process_closed_bar(self, position: dict[str, Any], candle: Candle, rules: SymbolRules) -> bool:
        direction = Direction[str(position["direction"])]
        stop = float(position["raw_stop_price"])
        target = float(position["take_profit_price"])
        gap_stop = candle.open <= stop if direction == Direction.LONG else candle.open >= stop
        if gap_stop:
            self._close_position(position, candle.open, "stop_loss_gap", candle.timestamp, rules)
            return True
        entry_time = _parse_time(str(position["entry_time"])) or candle.timestamp
        holding_minutes = int((candle.timestamp - entry_time).total_seconds() // 60)
        current_open_r = self._position_current_r(position, candle.open, rules)
        if (
            self.shadow.fail_fast_minutes > 0
            and holding_minutes >= self.shadow.fail_fast_minutes
            and float(position["max_mfe_r"]) < self.shadow.fail_fast_min_mfe_r
            and current_open_r <= self.shadow.fail_fast_max_current_r
        ):
            self._close_position(position, candle.open, "fail_fast", candle.timestamp, rules)
            return True
        if holding_minutes >= self.shadow.max_holding_minutes:
            self._close_position(position, candle.open, "time_stop", candle.timestamp, rules)
            return True
        stop_hit = candle.low <= stop if direction == Direction.LONG else candle.high >= stop
        target_hit = candle.high >= target if direction == Direction.LONG else candle.low <= target
        if stop_hit:
            self._close_position(position, stop, "stop_loss", candle.timestamp, rules)
            return True
        if target_hit:
            self._close_position(position, target, "take_profit", candle.timestamp, rules)
            return True
        partial_target = float(position.get("partial_take_profit_price", 0.0))
        partial_hit = (
            partial_target > 0.0
            and not bool(position.get("partial_taken", False))
            and (candle.high >= partial_target if direction == Direction.LONG else candle.low <= partial_target)
        )
        if partial_hit:
            self._take_partial(position, partial_target, candle.timestamp, rules)
        favorable = candle.high if direction == Direction.LONG else candle.low
        adverse = candle.low if direction == Direction.LONG else candle.high
        if direction == Direction.LONG:
            position["best_price"] = max(float(position["best_price"]), favorable)
            position["worst_price"] = min(float(position["worst_price"]), adverse)
        else:
            position["best_price"] = min(float(position["best_price"]), favorable)
            position["worst_price"] = max(float(position["worst_price"]), adverse)
        position["max_mfe_r"] = max(
            float(position["max_mfe_r"]),
            self._position_current_r(position, favorable, rules),
        )
        position["max_mae_r"] = min(
            float(position["max_mae_r"]),
            self._position_current_r(position, adverse, rules),
        )
        return False

    def _partial_target_crossed(self, position: dict[str, Any], raw_price: float) -> bool:
        target = float(position.get("partial_take_profit_price", 0.0))
        if target <= 0.0 or bool(position.get("partial_taken", False)):
            return False
        direction = Direction[str(position["direction"])]
        return raw_price >= target if direction == Direction.LONG else raw_price <= target

    def _take_partial(
        self,
        position: dict[str, Any],
        raw_exit: float,
        exit_time: datetime,
        rules: SymbolRules,
    ) -> bool:
        current_quantity = float(position["quantity"])
        original_quantity = float(position.get("original_quantity", current_quantity))
        requested = original_quantity * float(position.get("partial_take_profit_fraction", 0.0))
        partial_quantity = conservative_quantity(rules, requested)
        remaining = current_quantity - partial_quantity
        if (
            partial_quantity <= 0.0
            or remaining <= 0.0
            or remaining * float(position["entry_price"]) < max(5.0, float(rules.min_notional))
        ):
            return False

        direction = Direction[str(position["direction"])]
        ratio = partial_quantity / current_quantity
        fill = market_exit_fill(
            self.execution, rules, direction, partial_quantity, raw_exit, "take_profit_market"
        )
        raw_gross = direction.value * partial_quantity * (
            raw_exit - float(position["raw_entry_price"])
        )
        execution_gross = direction.value * partial_quantity * (
            fill.price - float(position["entry_price"])
        )
        entry_fee = float(position["entry_fee"]) * ratio
        entry_slippage = float(position["entry_slippage"]) * ratio
        risk_usdt = float(position["risk_usdt"]) * ratio
        funding_position = dict(position, quantity=partial_quantity)
        funding, funding_complete = self._funding_for_position(
            funding_position, now=exit_time, force=True
        )
        cash_delta = execution_gross - fill.fee + funding
        self.state["cash"] += cash_delta
        net_pnl = execution_gross - entry_fee - fill.fee + funding
        leg = {
            "leg_type": "partial_take_profit",
            "exit_time": exit_time.isoformat(),
            "raw_exit_price": raw_exit,
            "exit_price": fill.price,
            "quantity": partial_quantity,
            "gross_pnl": raw_gross,
            "execution_gross_pnl": execution_gross,
            "fee": entry_fee + fill.fee,
            "slippage": entry_slippage + fill.slippage_cost,
            "funding": funding,
            "funding_status": "complete" if funding_complete else "missing",
            "net_pnl": net_pnl,
            "risk_usdt": risk_usdt,
            "pnl_r": net_pnl / max(risk_usdt, 1e-12),
        }
        position.setdefault("partial_legs", []).append(leg)
        position["quantity"] = remaining
        position["entry_fee"] = float(position["entry_fee"]) - entry_fee
        position["entry_slippage"] = float(position["entry_slippage"]) - entry_slippage
        position["risk_usdt"] = float(position["risk_usdt"]) - risk_usdt
        position["partial_taken"] = True
        self.log(
            f"{position['symbol']}: shadow分批止盈 qty={partial_quantity:.8g} "
            f"raw={raw_exit:.8g} 净盈亏={net_pnl:+.4f}U"
        )
        self._append_event(
            "partial_exit",
            event_id=position["event_id"],
            symbol=position["symbol"],
            side=position["direction"],
            **leg,
        )
        return True

    def _position_current_r(self, position: dict[str, Any], raw_exit: float, rules: SymbolRules) -> float:
        direction = Direction[str(position["direction"])]
        fill = market_exit_fill(self.execution, rules, direction, 1.0, raw_exit, "market")
        net_per_unit = (
            direction.value * (fill.price - float(position["entry_price"]))
            - float(position["entry_fee"]) / float(position["quantity"])
            - fill.fee
        )
        return net_per_unit / max(float(position["unit_risk"]), 1e-12)

    def _close_position(
        self,
        position: dict[str, Any],
        raw_exit: float,
        reason: str,
        exit_time: datetime,
        rules: SymbolRules,
    ) -> None:
        direction = Direction[str(position["direction"])]
        quantity = float(position["quantity"])
        order_type = "market"
        if "stop" in reason:
            order_type = "stop_market"
        elif "take_profit" in reason:
            order_type = "take_profit_market"
        fill = market_exit_fill(self.execution, rules, direction, quantity, raw_exit, order_type)
        raw_gross = direction.value * quantity * (raw_exit - float(position["raw_entry_price"]))
        execution_gross = direction.value * quantity * (fill.price - float(position["entry_price"]))
        funding, funding_complete = self._funding_for_position(position, now=exit_time, force=True)
        cash_delta = execution_gross - fill.fee + funding
        self.state["cash"] += cash_delta
        total_fee = float(position["entry_fee"]) + fill.fee
        total_slippage = float(position["entry_slippage"]) + fill.slippage_cost
        net_pnl = execution_gross - total_fee + funding
        trade = {
            "event_id": position["event_id"],
            "symbol": position["symbol"],
            "side": position["direction"],
            "signal_available_time": position["signal_available_time"],
            "entry_time": position["entry_time"],
            "exit_time": exit_time.isoformat(),
            "entry_delay_seconds": position["entry_delay_seconds"],
            "raw_entry_price": position["raw_entry_price"],
            "entry_price": position["entry_price"],
            "raw_exit_price": raw_exit,
            "exit_price": fill.price,
            "quantity": quantity,
            "notional": quantity * float(position["entry_price"]),
            "raw_stop_price": position["raw_stop_price"],
            "risk_usdt": position["risk_usdt"],
            "gross_pnl": raw_gross,
            "execution_gross_pnl": execution_gross,
            "fee": total_fee,
            "slippage": total_slippage,
            "funding": funding,
            "funding_status": "complete" if funding_complete else "missing",
            "net_pnl": net_pnl,
            "pnl_r": net_pnl / max(float(position["risk_usdt"]), 1e-12),
            "mfe_r": position["max_mfe_r"],
            "mae_r": position["max_mae_r"],
            "exit_reason": reason,
            "holding_minutes": max(
                0,
                int((exit_time - (_parse_time(str(position["entry_time"])) or exit_time)).total_seconds() // 60),
            ),
            "quality_score": position["quality_score"],
            "btc_return_4h": position["btc_return_4h"],
        }
        partial_legs = list(position.get("partial_legs", []))
        if partial_legs:
            for key in ("gross_pnl", "execution_gross_pnl", "fee", "slippage", "funding", "net_pnl"):
                trade[key] = float(trade[key]) + sum(float(leg[key]) for leg in partial_legs)
            trade["quantity"] = float(position.get("original_quantity", quantity))
            trade["notional"] = trade["quantity"] * float(position["entry_price"])
            trade["risk_usdt"] = float(position.get("original_risk_usdt", trade["risk_usdt"]))
            trade["pnl_r"] = float(trade["net_pnl"]) / max(float(trade["risk_usdt"]), 1e-12)
            trade["partial_exit_count"] = len(partial_legs)
            trade["partial_realized_net_pnl"] = sum(float(leg["net_pnl"]) for leg in partial_legs)
            trade["partial_legs"] = partial_legs
            trade["funding_status"] = (
                "complete"
                if trade["funding_status"] == "complete"
                and all(leg["funding_status"] == "complete" for leg in partial_legs)
                else "missing"
            )
        else:
            trade["partial_exit_count"] = 0
            trade["partial_realized_net_pnl"] = 0.0
            trade["partial_legs"] = []
        self.state["trades"].append(trade)
        cooldown_until = exit_time + timedelta(minutes=self.shadow.symbol_cooldown_minutes)
        self.state["cooldown_until"][str(position["symbol"])] = cooldown_until.isoformat()
        self.state["position"] = None
        self._last_mark_price = 0.0
        self.log(
            f"{trade['symbol']}: shadow平仓 {reason} raw={raw_exit:.8g} fill={fill.price:.8g} "
            f"净盈亏={trade['net_pnl']:+.4f}U ({trade['pnl_r']:+.3f}R)"
        )
        self._append_event("exit", **trade)

    def _current_market_price(self, symbol: str) -> float:
        try:
            candles = self.client.klines(symbol, "1m", 2)
        except Exception:
            return 0.0
        return candles[-1].close if candles else 0.0

    def _funding_for_position(
        self,
        position: dict[str, Any],
        now: datetime,
        *,
        force: bool = False,
    ) -> tuple[float, bool]:
        if not self.execution.funding_enabled:
            return 0.0, True
        entry = _parse_time(str(position["entry_time"]))
        if entry is None or now <= entry:
            return 0.0, True
        cache_key = "|".join((
            str(position.get("event_id") or position["symbol"]),
            str(position["entry_time"]),
            f"{float(position['quantity']):.12g}",
        ))
        cached = self._funding_cache.get(cache_key)
        if not force and cached is not None and time.time() - cached[0] < 300.0:
            return cached[1], cached[2]
        try:
            rows = self.client.funding_rate_history(
                str(position["symbol"]),
                start_time=int(entry.replace(tzinfo=timezone.utc).timestamp() * 1000) + 1,
                end_time=int(now.replace(tzinfo=timezone.utc).timestamp() * 1000),
            )
            rates = [float(row["fundingRate"]) for row in rows]
        except Exception as exc:
            self.log(f"{position['symbol']}: funding 暂不可用，交易标记待补全 ({exc})")
            self._funding_cache[cache_key] = (time.time(), 0.0, False)
            return 0.0, False
        direction = Direction[str(position["direction"])]
        notional = float(position["quantity"]) * float(position["entry_price"])
        value = funding_cashflow(direction, notional, rates)
        self._funding_cache[cache_key] = (time.time(), value, True)
        return value, True

    def _reconcile_missing_funding(self) -> None:
        if self._funding_reconciled_this_run or not self.execution.funding_enabled:
            return
        self._funding_reconciled_this_run = True
        for trade in self.state.get("trades", []):
            if trade.get("funding_status") != "missing":
                continue
            position = {
                "symbol": trade["symbol"],
                "direction": trade["side"],
                "entry_time": trade["entry_time"],
                "quantity": trade["quantity"],
                "entry_price": trade["entry_price"],
            }
            exit_time = _parse_time(str(trade["exit_time"]))
            if exit_time is None:
                continue
            funding, complete = self._funding_for_position(position, now=exit_time, force=True)
            if not complete:
                continue
            old_funding = float(trade.get("funding", 0.0))
            adjustment = funding - old_funding
            trade["funding"] = funding
            trade["funding_status"] = "complete"
            trade["net_pnl"] = float(trade["net_pnl"]) + adjustment
            trade["pnl_r"] = float(trade["net_pnl"]) / max(float(trade["risk_usdt"]), 1e-12)
            self.state["cash"] += adjustment
            self._append_event("funding_reconciled", event_id=trade["event_id"], adjustment=adjustment)

    def _record_reject(self, reason: str) -> None:
        rejected = self.state["rejected"]
        rejected[reason] = int(rejected.get(reason, 0)) + 1

    def _append_event(self, event_type: str, **payload: Any) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "logged_at": _utc_now().isoformat(),
            "config_hash": self.config_hash,
            "frozen_version": self.shadow.frozen_version,
            "event_type": event_type,
            **payload,
        }
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.state_path)

    def _write_report(self, account: AccountSnapshot) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        report = self.acceptance_report(account)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.report_path)

    def _update_drawdown(self, equity: float) -> None:
        peak = max(float(self.state["peak_equity"]), equity)
        self.state["peak_equity"] = peak
        drawdown = (peak - equity) / peak if peak > 0 else 1.0
        self.state["max_drawdown_pct"] = max(float(self.state["max_drawdown_pct"]), drawdown)

    @staticmethod
    def _group_trades(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            grouped[str(trade.get(field, "unknown"))].append(trade)
        output: dict[str, dict[str, Any]] = {}
        for key, rows in sorted(grouped.items()):
            wins = [row for row in rows if float(row["net_pnl"]) > 0]
            losses = [row for row in rows if float(row["net_pnl"]) <= 0]
            gross_profit = sum(float(row["net_pnl"]) for row in wins)
            gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
            output[key] = {
                "trade_count": len(rows),
                "net_pnl": sum(float(row["net_pnl"]) for row in rows),
                "win_rate": len(wins) / len(rows),
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0),
            }
        return output


def _logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen Dual Thrust 50-symbol full-cost shadow runner")
    parser.add_argument(
        "--config",
        default="config.volatility-breakout.v2-balanced-50-shadow.json",
    )
    parser.add_argument("--once", action="store_true", help="run one management/scan cycle and exit")
    parser.add_argument("--report-only", action="store_true", help="print the persisted acceptance report")
    args = parser.parse_args()
    config = load_live_config(args.config)
    client = BinanceFuturesClient(
        api_key=None,
        api_secret=None,
        environment=config.exchange.environment,
        recv_window=config.exchange.recv_window,
        timeout_seconds=config.exchange.timeout_seconds,
    )
    trader = DualThrustShadowTrader(config, client, logger=_logger)
    if args.report_only:
        print(json.dumps(trader.acceptance_report(), indent=2, ensure_ascii=False))
        return 0
    if args.once:
        trader.validate_startup()
        trader._acquire_runtime_lock()
        try:
            trader.run_once()
        finally:
            trader._release_runtime_lock()
        print(json.dumps(trader.acceptance_report(), indent=2, ensure_ascii=False))
        return 0
    stop = threading.Event()
    try:
        trader.run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
