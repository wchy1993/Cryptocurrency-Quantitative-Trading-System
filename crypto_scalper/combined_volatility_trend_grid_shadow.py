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
from array import array
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .binance_client import BinanceApiError, BinanceFuturesClient, SymbolRules
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    COMBINED_STRATEGY_NAME,
    GRID_KEY,
    NOTIONAL_HEADROOM_SAFETY,
    CombinedPortfolioConfig,
)
from .live_config import LiveAppConfig, load_live_config
from .live_trader import AccountSnapshot, LivePosition
from .models import Candle, Direction
from .risk import (
    BacktestExecutionConfig,
    FundingRate,
    execution_config_from_live_config,
    market_exit_fill,
)
from .trend_grid import (
    TrendGridConfig,
    TrendGridSignal,
    TrendGridSnapshot,
    build_trend_grid_timeline,
)
from .trend_grid_optimize import (
    GridCampaign,
    GridCandidate,
    GridLevel,
    GridLot,
    GridPortfolioConfig,
    _create_campaign,
    _process_campaign_bar,
)
from .volatility_breakout import DualThrustSignal, VolatilityBreakoutConfig, build_dual_thrust_signals
from .volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    OpenPosition,
    PortfolioSearchConfig,
    _entry_position,
    _process_position_bar,
    minute_datetime,
    minute_token,
)
from .volatility_breakout_shadow import DUAL_THRUST_SHADOW_REASON_TOKEN, _closed_candles, _parse_time, _utc_now


COMBINED_SHADOW_REASON_TOKEN = "combined_breakout_trend_grid_max2_shadow"
TREND_GRID_SHADOW_REASON_TOKEN = "dynamic_trend_following_grid_shadow"
COMBINED_SHADOW_STATE_SCHEMA_VERSION = 1
LogCallback = Callable[[str], None]


def _resolve_config_path(value: str, anchor: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if anchor is not None:
        candidates.append(anchor.parent / path)
    candidates.extend((Path.cwd() / path, Path(__file__).resolve().parents[1] / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_source_bundle(config: LiveAppConfig) -> dict[str, Any]:
    shadow = config.combined_volatility_trend_grid_shadow
    combined_path = _resolve_config_path(shadow.source_combined_config_path)
    combined_payload = json.loads(combined_path.read_text(encoding="utf-8"))
    sources = combined_payload.get("source_configs", {})
    breakout_path = _resolve_config_path(str(sources[BREAKOUT_KEY]["path"]), combined_path)
    grid_path = _resolve_config_path(str(sources[GRID_KEY]["path"]), combined_path)
    return {
        "combined_path": combined_path,
        "combined_payload": combined_payload,
        "breakout_path": breakout_path,
        "breakout_payload": json.loads(breakout_path.read_text(encoding="utf-8")),
        "grid_path": grid_path,
        "grid_payload": json.loads(grid_path.read_text(encoding="utf-8")),
        "hashes": {
            "combined": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
            "breakout": hashlib.sha256(breakout_path.read_bytes()).hexdigest(),
            "grid": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        },
    }


def combined_shadow_config_hash(config: LiveAppConfig) -> str:
    bundle = _load_source_bundle(config)
    risk = config.risk
    payload = {
        "combined_shadow": asdict(config.combined_volatility_trend_grid_shadow),
        "breakout_shadow": asdict(config.dual_thrust_shadow),
        "source_hashes": bundle["hashes"],
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
        },
        "execution_order": [
            "closed_1m_only",
            "both_source_exits_before_new_entries",
            "breakout_priority_before_grid",
            "adverse_stop_first_on_same_bar",
            "grid_take_profit_before_new_grid_fill",
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dual_signal_from_dict(payload: dict[str, Any]) -> DualThrustSignal:
    values = dict(payload)
    values["direction"] = Direction[str(values["direction"])]
    values["signal_bar_time"] = _parse_time(str(values["signal_bar_time"]))
    values["signal_available_time"] = _parse_time(str(values["signal_available_time"]))
    return DualThrustSignal(**values)


def _trend_grid_signal_from_dict(payload: dict[str, Any]) -> TrendGridSignal:
    values = dict(payload)
    values["direction"] = Direction[str(values["direction"])]
    values["signal_bar_time"] = _parse_time(str(values["signal_bar_time"]))
    values["signal_available_time"] = _parse_time(str(values["signal_available_time"]))
    return TrendGridSignal(**values)


def _open_position_to_dict(position: OpenPosition) -> dict[str, Any]:
    return _jsonable(asdict(position))


def _open_position_from_dict(payload: dict[str, Any]) -> OpenPosition:
    values = dict(payload)
    candidate_payload = dict(values.pop("candidate"))
    signal = _dual_signal_from_dict(dict(candidate_payload.pop("signal")))
    candidate = Candidate(signal=signal, **candidate_payload)
    return OpenPosition(candidate=candidate, **values)


def _grid_campaign_to_dict(campaign: GridCampaign) -> dict[str, Any]:
    return _jsonable(asdict(campaign))


def _grid_campaign_from_dict(payload: dict[str, Any]) -> GridCampaign:
    values = dict(payload)
    candidate_payload = dict(values.pop("candidate"))
    signal = _trend_grid_signal_from_dict(dict(candidate_payload.pop("signal")))
    candidate = GridCandidate(signal=signal, **candidate_payload)
    levels = [GridLevel(**dict(item)) for item in values.pop("levels")]
    lots = {
        int(key): GridLot(**dict(item))
        for key, item in dict(values.pop("lots", {})).items()
    }
    return GridCampaign(candidate=candidate, levels=levels, lots=lots, **values)


def _compact_series(candles: list[Candle]) -> CompactSeries:
    return CompactSeries(
        minutes=array("q", (minute_token(item.timestamp) for item in candles)),
        opens=array("d", (item.open for item in candles)),
        highs=array("d", (item.high for item in candles)),
        lows=array("d", (item.low for item in candles)),
        closes=array("d", (item.close for item in candles)),
        volumes=array("d", (item.volume for item in candles)),
    )


class CombinedVolatilityTrendGridShadowTrader:
    """Shared-account, max-two-position paper trader for the frozen combination.

    It consumes public mainnet market data only.  There is intentionally no
    code path that invokes a Binance order endpoint.
    """

    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: LogCallback | None = None,
        account_callback: Callable[[AccountSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.shadow = config.combined_volatility_trend_grid_shadow
        self.breakout_shadow = config.dual_thrust_shadow
        self.client = client
        self.logger = logger or (lambda _message: None)
        self.account_callback = account_callback
        self.execution = execution_config_from_live_config(
            config,
            cost_experiment="full_cost",
            mode="conservative",
        )
        self.source_bundle = _load_source_bundle(config)
        combined_payload = self.source_bundle["combined_payload"]
        breakout_payload = self.source_bundle["breakout_payload"]
        grid_payload = self.source_bundle["grid_payload"]
        self.global_config = CombinedPortfolioConfig.from_dict(combined_payload["portfolio"])
        self.breakout_signal_config = VolatilityBreakoutConfig(**breakout_payload["balanced_signal"])
        self.breakout_portfolio_config = PortfolioSearchConfig(**breakout_payload["balanced_portfolio"])
        self.grid_signal_config = TrendGridConfig(
            **grid_payload.get("validation_selected_signal", grid_payload["signal"])
        )
        self.grid_portfolio_config = GridPortfolioConfig(
            **grid_payload.get("validation_selected_portfolio", grid_payload["portfolio"])
        )
        self.config_hash = combined_shadow_config_hash(config)
        self.state_path = Path(self.shadow.state_path)
        self.event_log_path = Path(self.shadow.event_log_path)
        self.report_path = Path(self.shadow.report_path)
        self.state = self._load_or_create_state()
        self._last_marks: dict[str, float] = {}
        self._unsupported_symbols: set[str] = set()
        self._last_heartbeat = 0.0
        self._runtime_lock_handle: Any | None = None
        if not self.state.get("sample_initialized_event_logged", False):
            self._append_event(
                "shadow_sample_started",
                started_at=self.state["started_at"],
                initial_equity=self.state["starting_equity"],
                universe_size=len(self.shadow.enabled_symbols),
                max_open_positions=self.shadow.max_open_positions,
            )
            self.state["sample_initialized_event_logged"] = True
            self._persist_state()

    def validate_startup(self) -> None:
        self.validate_startup_settings_only()
        if self.config.risk.cost_experiment != "full_cost":
            raise RuntimeError("combined shadow requires full_cost execution")
        if self.config.risk.backtest_mode != "conservative":
            raise RuntimeError("combined shadow requires conservative execution")
        if not self.config.risk.funding_enabled:
            raise RuntimeError("combined shadow requires funding_enabled=true")
        if self.config.exchange.environment != "mainnet":
            raise RuntimeError("combined shadow must use mainnet public market data")
        if self.shadow.max_open_positions != 2 or self.config.trading.max_open_positions != 2:
            raise RuntimeError("frozen combined shadow requires global max_open_positions=2")
        if self.shadow.max_open_positions_per_strategy != 1:
            raise RuntimeError("frozen combined shadow requires one position per source strategy")
        if self.breakout_portfolio_config.max_open_positions != 1:
            raise RuntimeError("frozen breakout source must remain max_open_positions=1")
        if self.grid_portfolio_config.max_open_campaigns != 1:
            raise RuntimeError("frozen grid source must remain max_open_campaigns=1")
        expected_priority = (BREAKOUT_KEY, GRID_KEY)
        if tuple(self.shadow.entry_priority) != expected_priority:
            raise RuntimeError(f"frozen combined entry priority must be {expected_priority}")
        if tuple(self.global_config.entry_priority) != expected_priority:
            raise RuntimeError("source combined config entry priority changed")
        if self.global_config.max_open_positions != self.shadow.max_open_positions:
            raise RuntimeError("combined source/global position limits do not match")
        if not math.isclose(
            self.global_config.max_gross_notional_multiple,
            self.shadow.max_gross_notional_multiple,
        ):
            raise RuntimeError("combined source/global notional limits do not match")
        if not math.isclose(
            self.global_config.hard_drawdown_stop_pct,
            self.shadow.hard_drawdown_stop_pct,
        ):
            raise RuntimeError("combined source/global drawdown limits do not match")
        if self.global_config.allow_same_symbol_across_strategies:
            raise RuntimeError("frozen combined shadow forbids same-symbol overlap")
        if (
            self.shadow.allow_same_symbol_across_strategies
            != self.global_config.allow_same_symbol_across_strategies
        ):
            raise RuntimeError("live/source same-symbol overlap policies do not match")
        symbols = tuple(self.shadow.enabled_symbols)
        if not symbols or len(symbols) != 50:
            raise RuntimeError("combined shadow requires the frozen 50-symbol universe")
        if symbols != tuple(self.breakout_shadow.enabled_symbols):
            raise RuntimeError("combined and breakout symbol universes differ")
        grid_symbols = tuple(self.source_bundle["grid_payload"]["symbols"])
        breakout_symbols = tuple(self.source_bundle["breakout_payload"]["symbols"])
        if symbols != grid_symbols or symbols != breakout_symbols:
            raise RuntimeError("live universe differs from frozen research source")
        if tuple(self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI trading symbols differ from frozen combined universe")
        if tuple(self.config.trading.entry_symbols or self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI entry symbols differ from frozen combined universe")
        if COMBINED_STRATEGY_NAME != self.shadow.strategy_name:
            raise RuntimeError("combined strategy name changed")
        breakout_signal = self.source_bundle["breakout_payload"]["balanced_signal"]
        breakout_portfolio = self.source_bundle["breakout_payload"]["balanced_portfolio"]
        for field in (
            "timeframe_minutes",
            "lookback_days",
            "long_k",
            "short_k",
            "allow_long",
            "allow_short",
            "atr_period",
            "trend_ema_period",
            "max_signals_per_symbol_day",
            "stop_atr_multiple",
            "take_profit_r",
            "fail_fast_minutes",
            "fail_fast_min_mfe_r",
            "fail_fast_max_current_r",
            "max_holding_minutes",
        ):
            if getattr(self.breakout_shadow, field) != breakout_signal[field]:
                raise RuntimeError(f"breakout source parameter changed: {field}")
        for field in (
            "risk_per_trade_pct",
            "max_trade_risk_pct",
            "max_open_positions",
            "max_daily_trades",
            "symbol_cooldown_minutes",
            "max_notional_multiple",
            "hard_drawdown_stop_pct",
            "ranking_mode",
            "long_risk_multiplier",
            "short_risk_multiplier",
            "max_directional_btc_return_4h",
        ):
            if getattr(self.breakout_shadow, field) != breakout_portfolio[field]:
                raise RuntimeError(f"breakout source portfolio parameter changed: {field}")
        legacy_flags = (
            self.config.strategy.super_volume_breakout_enabled,
            self.config.strategy.startup_breakout_enabled,
            self.config.strategy.ordinary_breakout_enabled,
            self.config.strategy.pullback_reclaim_enabled,
            self.config.strategy.fast_breakout_enabled,
            self.config.strategy.spike_trade_enabled,
            self.config.strategy.rsi_reversal_enabled,
            self.config.strategy.mtf_4h_rsi_regime_enabled,
            self.config.strategy.mtf_momentum_reset_enabled,
            self.config.strategy.oi_flush_reversal_enabled,
        )
        if any(legacy_flags):
            raise RuntimeError("legacy strategy flags must remain disabled in combined shadow mode")
        if any(
            (
                self.config.vbp_strategy.enabled,
                self.config.reversal_alpha.enabled,
                self.config.cmipr.enabled,
                self.config.mtper.enabled,
                self.config.mtpc.enabled,
                self.config.macro_events.enabled,
            )
        ):
            raise RuntimeError("legacy strategies must remain disabled in combined shadow mode")
        self.client.ping()

    def validate_startup_settings_only(self) -> None:
        if not self.shadow.enabled or not self.shadow.shadow_only:
            raise RuntimeError("combined shadow safety settings are not satisfied")
        if not self.breakout_shadow.enabled or not self.breakout_shadow.shadow_only:
            raise RuntimeError("breakout source shadow safety settings are not satisfied")
        if not self.config.trading.dry_run:
            raise RuntimeError("combined shadow refuses to start when dry_run=false")

    def run_forever(self, stop_event: threading.Event) -> None:
        self.validate_startup()
        self._acquire_runtime_lock()
        try:
            self.log(
                "Breakout + Dynamic Trend Grid 50币组合 shadow 已启动；共享模拟账户最多两仓，"
                "每策略最多一仓，只使用主网公开行情，不会发送订单"
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
                    self.log(f"组合shadow运行错误: {type(exc).__name__}: {exc}")
                elapsed = time.time() - started
                stop_event.wait(max(1.0, self.config.trading.poll_seconds - elapsed))
            self._write_report(self.snapshot_account(fetch_mark=False))
            self.log("组合 shadow 已停止，独立状态与验收报告已落盘")
        finally:
            self._release_runtime_lock()

    def run_once(self, stop_event: threading.Event | None = None) -> None:
        self.validate_startup_settings_only()
        now = _utc_now()
        self._manage_breakout_position(now)
        self._manage_grid_campaign(now)
        self._scan_and_maybe_enter(now, stop_event)
        account = self.snapshot_account(fetch_mark=True)
        self._update_drawdown(account.equity)
        self.state["max_concurrent_positions"] = max(
            int(self.state.get("max_concurrent_positions", 0)),
            self._open_count(),
        )
        self._persist_state()
        self._write_report(account)
        if self.account_callback:
            self.account_callback(account)
        heartbeat = max(30, int(self.shadow.heartbeat_seconds))
        if time.time() - self._last_heartbeat >= heartbeat:
            self._last_heartbeat = time.time()
            counts = self.state["candidate_count"]
            self.log(
                f"组合shadow统计: 权益={account.equity:.2f}U 持仓={self._open_count()}/2 "
                f"平仓={len(self.state['trades'])} 候选=BO:{counts[BREAKOUT_KEY]}/GRID:{counts[GRID_KEY]} "
                f"最大回撤={self.state['max_drawdown_pct'] * 100:.2f}%"
            )

    def snapshot_account(self, fetch_mark: bool = True) -> AccountSnapshot:
        cash = float(self.state["cash"])
        rows: list[LivePosition] = []
        total_unrealized = 0.0
        initial_margin = 0.0
        maintenance = 0.0
        leverage = max(1, int(self.config.trading.leverage))

        breakout_payload = self.state.get("breakout_position")
        if breakout_payload:
            position = self._decode_breakout_position(breakout_payload)
            symbol = position.candidate.signal.symbol
            mark = self._mark_price(symbol, position.raw_entry_price, fetch_mark)
            rules = self.client.symbol_rules(symbol)
            fill = market_exit_fill(
                self.execution,
                rules,
                position.candidate.signal.direction,
                position.quantity,
                mark,
                "market",
            )
            unrealized = (
                position.candidate.signal.direction.value
                * position.quantity
                * (fill.price - position.entry_price)
                - fill.fee
            )
            notional = abs(position.quantity * mark)
            rows.append(
                LivePosition(
                    symbol=symbol,
                    position_side="SHADOW-BO",
                    direction=position.candidate.signal.direction,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    mark_price=mark,
                    notional=notional,
                    unrealized_pnl=unrealized,
                    leverage=leverage,
                    margin_type="SHADOW",
                    liquidation_price=None,
                    entry_reason=(
                        f"{COMBINED_SHADOW_REASON_TOKEN}|{DUAL_THRUST_SHADOW_REASON_TOKEN}|"
                        f"event={position.candidate.signal.event_id}"
                    ),
                    opened_at=minute_datetime(position.entry_minute).replace(tzinfo=timezone.utc),
                )
            )
            total_unrealized += unrealized
            initial_margin += notional / leverage
            maintenance += notional * max(
                0.0, self.config.risk.estimated_maintenance_margin_rate
            )

        grid_payload = self.state.get("grid_campaign")
        if grid_payload:
            campaign = _grid_campaign_from_dict(grid_payload)
            symbol = campaign.candidate.signal.symbol
            mark = self._mark_price(symbol, campaign.anchor_price, fetch_mark)
            rules = self.client.symbol_rules(symbol)
            quantity = sum(lot.quantity for lot in campaign.lots.values())
            weighted_entry = (
                sum(lot.quantity * lot.entry_price for lot in campaign.lots.values()) / quantity
                if quantity > 0.0
                else campaign.anchor_price
            )
            unrealized = 0.0
            notional = 0.0
            for lot in campaign.lots.values():
                fill = market_exit_fill(
                    self.execution,
                    rules,
                    campaign.candidate.signal.direction,
                    lot.quantity,
                    mark,
                    "market",
                )
                unrealized += (
                    campaign.candidate.signal.direction.value
                    * lot.quantity
                    * (fill.price - lot.entry_price)
                    - fill.fee
                )
                notional += abs(lot.quantity * mark)
            rows.append(
                LivePosition(
                    symbol=symbol,
                    position_side="SHADOW-GRID",
                    direction=campaign.candidate.signal.direction,
                    quantity=quantity,
                    entry_price=weighted_entry,
                    mark_price=mark,
                    notional=notional,
                    unrealized_pnl=unrealized,
                    leverage=leverage,
                    margin_type="SHADOW",
                    liquidation_price=None,
                    entry_reason=(
                        f"{COMBINED_SHADOW_REASON_TOKEN}|{TREND_GRID_SHADOW_REASON_TOKEN}|"
                        f"event={campaign.candidate.signal.event_id}"
                    ),
                    opened_at=minute_datetime(campaign.start_minute).replace(tzinfo=timezone.utc),
                )
            )
            total_unrealized += unrealized
            initial_margin += notional / leverage
            maintenance += notional * max(
                0.0, self.config.risk.estimated_maintenance_margin_rate
            )

        equity = cash + total_unrealized
        positions = {row.symbol: row for row in rows}
        return AccountSnapshot(
            equity=equity,
            wallet_balance=cash,
            available_balance=max(0.0, equity - initial_margin),
            initial_margin=initial_margin,
            maintenance_margin=maintenance,
            total_unrealized_pnl=total_unrealized,
            positions=positions,
            position_rows=tuple(rows),
            position_mode="combined full-cost shadow max2",
        )

    def acceptance_report(self, account: AccountSnapshot | None = None) -> dict[str, Any]:
        account = account or self.snapshot_account(fetch_mark=False)
        trades = list(self.state.get("trades", []))
        wins = [row for row in trades if float(row["net_pnl"]) > 0.0]
        losses = [row for row in trades if float(row["net_pnl"]) <= 0.0]
        gross_profit = sum(float(row["net_pnl"]) for row in wins)
        gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (
            math.inf if gross_profit > 0.0 else 0.0
        )
        started = _parse_time(str(self.state["started_at"])) or _utc_now()
        elapsed_days = max(0.0, (_utc_now() - started).total_seconds() / 86_400.0)
        funding_complete = all(row.get("funding_status") == "complete" for row in trades)
        integrity_errors = list(self.state.get("sample_integrity_errors", []))
        closed_net = sum(float(row["net_pnl"]) for row in trades)
        criteria = {
            "minimum_30_calendar_days": elapsed_days >= 30.0,
            "minimum_30_closed_trades": len(trades) >= 30,
            "full_cost_funding_complete": funding_complete,
            "sample_integrity_clean": not integrity_errors,
            "profit_factor_above_1_10": profit_factor > 1.10,
            "positive_expectancy": closed_net > 0.0,
            "drawdown_below_frozen_hard_limit": (
                float(self.state["max_drawdown_pct"]) < self.shadow.hard_drawdown_stop_pct
            ),
        }
        enough_data = criteria["minimum_30_calendar_days"] and criteria["minimum_30_closed_trades"]
        status = "collecting_new_combined_shadow_data"
        if enough_data:
            status = "eligible_for_review" if all(criteria.values()) else "combined_shadow_rejected"
        fee = sum(float(row.get("fee", 0.0)) for row in trades)
        slippage = sum(float(row.get("slippage", 0.0)) for row in trades)
        funding = sum(float(row.get("funding", 0.0)) for row in trades)
        return {
            "strategy_name": self.shadow.strategy_name,
            "frozen_version": self.shadow.frozen_version,
            "config_hash": self.config_hash,
            "source_config_hashes": self.source_bundle["hashes"],
            "status": status,
            "important_note": (
                "The +12,358.34% / PF 1.610 / 44.10% drawdown result is in-sample "
                "historical research from 2026-03-06 through 2026-06-06. This report "
                "contains only observations after started_at and cannot authorize live trading."
            ),
            "historical_reference": {
                "period": {"start": "2026-03-06T00:00:00", "end": "2026-06-06T00:00:00"},
                "trade_count": 236,
                "net_profit_usdt": 24716.67,
                "return_pct": 12358.34,
                "profit_factor": 1.610,
                "win_rate_pct": 38.56,
                "max_drawdown_pct": 44.10,
            },
            "started_at": self.state["started_at"],
            "updated_at": _utc_now().isoformat(),
            "elapsed_days": elapsed_days,
            "universe_size": len(self.shadow.enabled_symbols),
            "global_max_open_positions": self.shadow.max_open_positions,
            "per_strategy_max_open_positions": self.shadow.max_open_positions_per_strategy,
            "max_observed_concurrent_positions": int(
                self.state.get("max_concurrent_positions", 0)
            ),
            "open_positions": self._open_count(),
            "open_by_strategy": {
                BREAKOUT_KEY: int(self.state.get("breakout_position") is not None),
                GRID_KEY: int(self.state.get("grid_campaign") is not None),
            },
            "candidate_count_by_strategy": dict(self.state["candidate_count"]),
            "trade_count": len(trades),
            "initial_equity": float(self.state["starting_equity"]),
            "current_equity": account.equity,
            "closed_net_pnl": closed_net,
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "profit_factor": profit_factor,
            "expectancy_usdt": statistics.mean(float(row["net_pnl"]) for row in trades)
            if trades
            else 0.0,
            "max_drawdown_pct": float(self.state["max_drawdown_pct"]),
            "fee": fee,
            "slippage": slippage,
            "funding": funding,
            "full_cost": fee + slippage - funding,
            "funding_complete": funding_complete,
            "sample_integrity_errors": integrity_errors,
            "rejected": self.state["rejected"],
            "criteria": criteria,
            "by_strategy": self._group_trades(trades, "strategy"),
            "by_symbol": self._group_trades(trades, "symbol"),
            "by_exit_reason": self._group_trades(trades, "exit_reason"),
            "trades": trades,
        }

    def log(self, message: str) -> None:
        self.logger(message)

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("schema_version") != COMBINED_SHADOW_STATE_SCHEMA_VERSION:
                raise RuntimeError("combined shadow state schema mismatch")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError(
                    "combined shadow state belongs to a different frozen config; "
                    "use a new state_path instead of mixing samples"
                )
            return state
        now = _utc_now().isoformat()
        starting = float(self.config.risk.starting_capital_usdt)
        return {
            "schema_version": COMBINED_SHADOW_STATE_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "frozen_version": self.shadow.frozen_version,
            "started_at": now,
            "starting_equity": starting,
            "cash": starting,
            "peak_equity": starting,
            "max_drawdown_pct": 0.0,
            "max_concurrent_positions": 0,
            "last_scanned_available_time": None,
            "seen_events": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "daily_entries": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "cooldown_until": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "breakout_position": None,
            "breakout_last_processed_minute": None,
            "grid_campaign": None,
            "grid_last_processed_minute": None,
            "trades": [],
            "candidate_count": {BREAKOUT_KEY: 0, GRID_KEY: 0},
            "rejected": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "sample_integrity_errors": [],
            "sample_initialized_event_logged": False,
        }

    def _scan_and_maybe_enter(
        self,
        now: datetime,
        stop_event: threading.Event | None,
    ) -> None:
        latest_available = now.replace(minute=0, second=0, microsecond=0)
        if (now - latest_available).total_seconds() < max(0, self.shadow.scan_grace_seconds):
            return
        last_scanned = _parse_time(self.state.get("last_scanned_available_time"))
        if last_scanned is not None and last_scanned >= latest_available:
            return

        candles_by_symbol: dict[str, list[Candle]] = {}
        symbols = tuple(self.shadow.enabled_symbols)
        for index, symbol in enumerate(symbols):
            if symbol in self._unsupported_symbols:
                continue
            try:
                rows = self.client.klines(symbol, "1h", 500)
                closed = _closed_candles(rows, 60, now)
                if len(closed) >= 150:
                    candles_by_symbol[symbol] = closed
            except Exception as exc:
                self.log(f"{symbol}: 组合shadow 1h数据不可用 ({type(exc).__name__}: {exc})")
                if isinstance(exc, BinanceApiError) and "symbol" in str(exc).lower():
                    self._unsupported_symbols.add(symbol)
            if index + 1 < len(symbols) and self.shadow.request_pacing_seconds > 0.0:
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
            self.log(f"组合shadow扫描数据不足 {len(candles_by_symbol)}/{required}，本小时稍后重试")
            return

        breakout_candidates, grid_candidates, btc_return_4h = self._build_entry_candidates(
            candles_by_symbol, latest_available
        )

        self.state["last_scanned_available_time"] = latest_available.isoformat()
        self.state["candidate_count"][BREAKOUT_KEY] += len(breakout_candidates)
        self.state["candidate_count"][GRID_KEY] += len(grid_candidates)
        for strategy in self.shadow.entry_priority:
            rows: list[Candidate] | list[GridCandidate]
            rows = breakout_candidates if strategy == BREAKOUT_KEY else grid_candidates
            opened_this_source = False
            for candidate in rows:
                reason = self._candidate_reject_reason(strategy, candidate, now)
                if reason is None and not opened_this_source:
                    opened_this_source = (
                        self._open_breakout_candidate(candidate, now)
                        if strategy == BREAKOUT_KEY
                        else self._open_grid_candidate(candidate, now)
                    )
                    reason = "entered" if opened_this_source else "sizing_or_execution_rejected"
                elif reason is None:
                    reason = "lower_ranked_same_cycle"
                signal = candidate.signal
                self.state["seen_events"][strategy][signal.event_id] = {
                    "time": now.isoformat(),
                    "status": reason,
                }
                if reason != "entered":
                    self._record_reject(strategy, reason)
                self._append_event(
                    "candidate_decision",
                    strategy=strategy,
                    event_id=signal.event_id,
                    symbol=signal.symbol,
                    side=signal.direction.name,
                    signal_available_time=signal.signal_available_time.isoformat(),
                    decision_time=now.isoformat(),
                    status=reason,
                    quality_score=signal.quality_score,
                    btc_return_4h=btc_return_4h,
                )

    def _build_entry_candidates(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        latest_available: datetime,
    ) -> tuple[list[Candidate], list[GridCandidate], float]:
        """Build the original v2/Grid-v2 live candidates.

        New frozen combinations can override this hook without changing the
        historical combined-shadow implementation.
        """

        btc = candles_by_symbol["BTCUSDT"]
        btc_return_4h = btc[-1].close / max(btc[-5].close, 1e-12) - 1.0
        breakout_candidates: list[Candidate] = []
        grid_candidates: list[GridCandidate] = []
        for symbol, candles in candles_by_symbol.items():
            breakout_signals = build_dual_thrust_signals(
                symbol, candles, self.breakout_signal_config
            )
            if breakout_signals:
                signal = breakout_signals[-1]
                if (
                    signal.signal_available_time == latest_available
                    and signal.event_id not in self.state["seen_events"][BREAKOUT_KEY]
                ):
                    breakout_candidates.append(
                        Candidate(
                            signal=signal,
                            entry_minute=minute_token(latest_available),
                            btc_return_4h=btc_return_4h,
                        )
                    )
            _, grid_signals = build_trend_grid_timeline(
                symbol, candles, self.grid_signal_config
            )
            if grid_signals:
                signal = grid_signals[-1]
                if (
                    signal.signal_available_time == latest_available
                    and signal.event_id not in self.state["seen_events"][GRID_KEY]
                ):
                    grid_candidates.append(
                        GridCandidate(
                            signal=signal,
                            entry_minute=minute_token(latest_available),
                        )
                    )
        breakout_candidates.sort(
            key=lambda item: (-item.signal.quality_score, item.signal.symbol)
        )
        grid_candidates.sort(
            key=lambda item: (-item.signal.quality_score, item.signal.symbol)
        )
        return breakout_candidates, grid_candidates, btc_return_4h

    def _candidate_reject_reason(
        self,
        strategy: str,
        candidate: Candidate | GridCandidate,
        now: datetime,
    ) -> str | None:
        signal = candidate.signal
        started = _parse_time(str(self.state["started_at"])) or now
        if signal.signal_available_time < started:
            return "pre_shadow_start"
        age = (now - signal.signal_available_time).total_seconds() / 60.0
        if age < 0.0 or age > self.shadow.max_signal_age_minutes:
            return "stale_signal"
        if strategy == BREAKOUT_KEY:
            directional_btc = signal.direction.value * float(
                getattr(candidate, "btc_return_4h", 0.0) or 0.0
            )
            if directional_btc > self.breakout_portfolio_config.max_directional_btc_return_4h:
                return "btc_4h_overextended"
            if self.state.get("breakout_position") is not None:
                return "strategy_position_limit"
            max_daily = self.breakout_portfolio_config.max_daily_trades
        else:
            if self.state.get("grid_campaign") is not None:
                return "strategy_position_limit"
            max_daily = self.grid_portfolio_config.max_daily_campaigns
        if self._open_count() >= self.shadow.max_open_positions:
            return "global_position_limit"
        if not self.shadow.allow_same_symbol_across_strategies and self._symbol_is_open(signal.symbol):
            return "global_symbol_already_open"
        equity = self.snapshot_account(fetch_mark=False).equity
        peak = max(float(self.state["peak_equity"]), equity)
        drawdown = (peak - equity) / peak if peak > 0.0 else 1.0
        if drawdown >= self.shadow.hard_drawdown_stop_pct or equity <= 0.0:
            return "hard_drawdown_stop"
        day_key = now.date().isoformat()
        if int(self.state["daily_entries"][strategy].get(day_key, 0)) >= max_daily:
            return "daily_entry_limit"
        cooldown = _parse_time(self.state["cooldown_until"][strategy].get(signal.symbol))
        if cooldown is not None and cooldown > now:
            return "symbol_cooldown"
        if self._available_notional_multiple(equity) <= 0.0:
            return "global_notional_limit"
        return None

    def _open_breakout_candidate(self, candidate: Candidate, now: datetime) -> bool:
        signal = candidate.signal
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: Breakout shadow成交参考不可用 ({exc})")
            return False
        raw = active.close
        synthetic = Candle(active.timestamp, raw, raw, raw, raw, active.volume)
        live_candidate = replace(candidate, entry_minute=minute_token(active.timestamp))
        series = _compact_series([synthetic])
        equity = self.snapshot_account(fetch_mark=False).equity
        available = self._available_notional_multiple(equity)
        portfolio = replace(
            self.breakout_portfolio_config,
            max_notional_multiple=min(
                self.breakout_portfolio_config.max_notional_multiple,
                available * NOTIONAL_HEADROOM_SAFETY,
            ),
        )
        opened = _entry_position(
            live_candidate,
            series,
            rules,
            self.breakout_signal_config,
            portfolio,
            self.execution,
            equity if self.breakout_portfolio_config.compound else float(self.state["starting_equity"]),
        )
        if opened is None:
            return False
        position, entry_fee = opened
        self.state["cash"] -= entry_fee
        self.state["breakout_position"] = _open_position_to_dict(position)
        self.state["breakout_last_processed_minute"] = position.entry_minute - 1
        self._record_entry(BREAKOUT_KEY, signal.symbol, now)
        self._last_marks[signal.symbol] = raw
        self.log(
            f"{signal.symbol}: 组合Breakout开仓 {signal.direction.name} qty={position.quantity:.8g} "
            f"fill={position.entry_price:.8g} stop={position.stop_price:.8g} "
            f"风险={position.risk_budget:.3f}U"
        )
        self._append_event(
            "entry",
            strategy=BREAKOUT_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            side=signal.direction.name,
            entry_time=minute_datetime(position.entry_minute).isoformat(),
            raw_entry_price=position.raw_entry_price,
            entry_price=position.entry_price,
            quantity=position.quantity,
            risk_usdt=position.risk_budget,
            entry_fee=entry_fee,
        )
        return True

    def _open_grid_candidate(self, candidate: GridCandidate, now: datetime) -> bool:
        signal = candidate.signal
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: Grid shadow成交参考不可用 ({exc})")
            return False
        raw = active.close
        synthetic = Candle(active.timestamp, raw, raw, raw, raw, active.volume)
        live_candidate = replace(candidate, entry_minute=minute_token(active.timestamp))
        series = _compact_series([synthetic])
        equity = self.snapshot_account(fetch_mark=False).equity
        available = self._available_notional_multiple(equity)
        portfolio = replace(
            self.grid_portfolio_config,
            max_notional_multiple=min(
                self.grid_portfolio_config.max_notional_multiple,
                available * NOTIONAL_HEADROOM_SAFETY,
            ),
        )
        opened = _create_campaign(
            live_candidate,
            series,
            rules,
            self.grid_signal_config,
            portfolio,
            self.execution,
            equity if self.grid_portfolio_config.compound else float(self.state["starting_equity"]),
        )
        if opened is None:
            return False
        campaign, initial_fee = opened
        self.state["cash"] -= initial_fee
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self.state["grid_last_processed_minute"] = campaign.start_minute - 1
        self._record_entry(GRID_KEY, signal.symbol, now)
        self._last_marks[signal.symbol] = raw
        committed = sum(level.quantity * level.raw_price for level in campaign.levels)
        self.log(
            f"{signal.symbol}: 组合Grid开仓 {signal.direction.name} levels={len(campaign.levels)} "
            f"首层qty={campaign.levels[0].quantity:.8g} hard_stop={campaign.hard_stop:.8g} "
            f"预留名义={committed:.2f}U 风险={campaign.risk_budget:.3f}U"
        )
        self._append_event(
            "entry",
            strategy=GRID_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            side=signal.direction.name,
            entry_time=minute_datetime(campaign.start_minute).isoformat(),
            anchor_price=campaign.anchor_price,
            hard_stop=campaign.hard_stop,
            risk_usdt=campaign.risk_budget,
            committed_notional=committed,
            entry_fee=initial_fee,
        )
        return True

    def _manage_breakout_position(self, now: datetime) -> None:
        payload = self.state.get("breakout_position")
        if not payload:
            return
        position = _open_position_from_dict(payload)
        symbol = position.candidate.signal.symbol
        try:
            candles = self.client.klines(symbol, "1m", 1500)
            closed = _closed_candles(candles, 1, now)
            rules = self.client.symbol_rules(symbol)
            execution = self._execution_with_funding(
                symbol, minute_datetime(position.entry_minute), now
            )
        except Exception as exc:
            self.log(f"{symbol}: Breakout持仓数据/资金费不可用，保留仓位等待恢复 ({exc})")
            self._append_event("position_data_error", strategy=BREAKOUT_KEY, symbol=symbol, error=str(exc))
            return
        last = int(self.state.get("breakout_last_processed_minute") or position.entry_minute - 1)
        if not self._history_is_contiguous(BREAKOUT_KEY, symbol, closed, last):
            return
        series = _compact_series(closed)
        for index, minute in enumerate(series.minutes):
            minute = int(minute)
            if minute <= last:
                continue
            result = _process_position_bar(
                position,
                minute,
                series,
                index,
                self.breakout_signal_config,
                execution,
                rules,
            )
            last = minute
            self.state["breakout_last_processed_minute"] = minute
            if result is None:
                continue
            trade, cash_delta = result
            trade = {"strategy": BREAKOUT_KEY, "funding_status": "complete", **trade}
            self.state["cash"] += cash_delta
            self.state["trades"].append(trade)
            self.state["breakout_position"] = None
            self._set_cooldown(BREAKOUT_KEY, symbol, minute, self.breakout_portfolio_config.symbol_cooldown_minutes)
            self.log(
                f"{symbol}: 组合Breakout平仓 {trade['exit_reason']} "
                f"净盈亏={trade['net_pnl']:+.4f}U ({trade['pnl_r']:+.3f}R)"
            )
            self._append_event("exit", **trade)
            return
        self.state["breakout_position"] = _open_position_to_dict(position)
        if candles:
            self._last_marks[symbol] = candles[-1].close

    def _manage_grid_campaign(self, now: datetime) -> None:
        payload = self.state.get("grid_campaign")
        if not payload:
            return
        campaign = _grid_campaign_from_dict(payload)
        symbol = campaign.candidate.signal.symbol
        try:
            candles = self.client.klines(symbol, "1m", 1500)
            closed = _closed_candles(candles, 1, now)
            rules = self.client.symbol_rules(symbol)
            hourly = _closed_candles(self.client.klines(symbol, "1h", 500), 60, now)
            snapshots, _ = build_trend_grid_timeline(symbol, hourly, self.grid_signal_config)
            snapshot_by_minute = {minute_token(item.available_time): item for item in snapshots}
            execution = self._execution_with_funding(
                symbol, minute_datetime(campaign.start_minute), now
            )
        except Exception as exc:
            self.log(f"{symbol}: Grid持仓数据/资金费不可用，保留campaign等待恢复 ({exc})")
            self._append_event("position_data_error", strategy=GRID_KEY, symbol=symbol, error=str(exc))
            return
        last = int(self.state.get("grid_last_processed_minute") or campaign.start_minute - 1)
        if not self._history_is_contiguous(GRID_KEY, symbol, closed, last):
            return
        series = _compact_series(closed)
        for index, minute in enumerate(series.minutes):
            minute = int(minute)
            if minute <= last:
                continue
            cash_delta, close_reason = _process_campaign_bar(
                campaign,
                minute,
                series,
                index,
                snapshot_by_minute.get(minute),
                self.grid_signal_config,
                execution,
                rules,
            )
            self.state["cash"] += cash_delta
            last = minute
            self.state["grid_last_processed_minute"] = minute
            if close_reason is None:
                continue
            report = getattr(campaign, "_pending_report", None)
            self.state["grid_campaign"] = None
            self._set_cooldown(GRID_KEY, symbol, minute, self.grid_portfolio_config.symbol_cooldown_minutes)
            if report is not None:
                trade = {"strategy": GRID_KEY, "funding_status": "complete", **report}
                self.state["trades"].append(trade)
                self.log(
                    f"{symbol}: 组合Grid平仓 {close_reason} 净盈亏={trade['net_pnl']:+.4f}U "
                    f"({trade['pnl_r']:+.3f}R) entries={trade['entry_count']}"
                )
                self._append_event("exit", **trade)
            else:
                self._record_reject(GRID_KEY, "campaign_without_fill")
            return
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        if candles:
            self._last_marks[symbol] = candles[-1].close

    def _execution_with_funding(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> BacktestExecutionConfig:
        if not self.execution.funding_enabled or end <= start:
            return self.execution
        rows = self.client.funding_rate_history(
            symbol,
            start_time=int(start.replace(tzinfo=timezone.utc).timestamp() * 1000) + 1,
            end_time=int(end.replace(tzinfo=timezone.utc).timestamp() * 1000),
        )
        rates = tuple(
            FundingRate(
                timestamp=datetime.fromtimestamp(
                    int(row["fundingTime"]) / 1000.0, tz=timezone.utc
                ).replace(tzinfo=None),
                rate=float(row["fundingRate"]),
            )
            for row in rows
        )
        return replace(self.execution, funding_rates_by_symbol={symbol: rates})

    def _history_is_contiguous(
        self,
        strategy: str,
        symbol: str,
        candles: list[Candle],
        last_processed: int,
    ) -> bool:
        pending = [minute_token(item.timestamp) for item in candles if minute_token(item.timestamp) > last_processed]
        if not pending:
            return True
        if pending[0] <= last_processed + 1:
            return True
        error = {
            "strategy": strategy,
            "symbol": symbol,
            "detected_at": _utc_now().isoformat(),
            "last_processed_minute": last_processed,
            "first_available_minute": pending[0],
            "reason": "closed_1m_history_gap",
        }
        already_recorded = any(
            item.get("strategy") == strategy
            and item.get("symbol") == symbol
            and item.get("last_processed_minute") == last_processed
            and item.get("first_available_minute") == pending[0]
            for item in self.state["sample_integrity_errors"]
        )
        if not already_recorded:
            self.state["sample_integrity_errors"].append(error)
        self.log(
            f"{symbol}: 检测到1m历史缺口，暂停{strategy}模拟撮合；样本已标记不完整"
        )
        if not already_recorded:
            self._append_event("sample_integrity_error", **error)
        return False

    def _record_entry(self, strategy: str, symbol: str, now: datetime) -> None:
        day_key = now.date().isoformat()
        entries = self.state["daily_entries"][strategy]
        self.state["daily_entries"][strategy] = {
            key: value for key, value in entries.items() if key == day_key
        }
        entries = self.state["daily_entries"][strategy]
        entries[day_key] = int(entries.get(day_key, 0)) + 1

    def _set_cooldown(self, strategy: str, symbol: str, minute: int, duration: int) -> None:
        self.state["cooldown_until"][strategy][symbol] = minute_datetime(
            minute + duration
        ).isoformat()

    def _open_count(self) -> int:
        return int(self.state.get("breakout_position") is not None) + int(
            self.state.get("grid_campaign") is not None
        )

    def _decode_breakout_position(self, payload: dict[str, Any]) -> OpenPosition:
        """Decode the legacy raw Breakout position state.

        Protected Breakout variants override this hook while the old frozen
        combined sample remains byte-for-byte compatible with its state shape.
        """

        return _open_position_from_dict(payload)

    def _symbol_is_open(self, symbol: str) -> bool:
        breakout = self.state.get("breakout_position")
        if (
            breakout
            and self._decode_breakout_position(breakout).candidate.signal.symbol == symbol
        ):
            return True
        grid = self.state.get("grid_campaign")
        return bool(grid and grid["candidate"]["signal"]["symbol"] == symbol)

    def _committed_notional(self) -> float:
        total = 0.0
        breakout = self.state.get("breakout_position")
        if breakout:
            position = self._decode_breakout_position(breakout)
            total += abs(position.quantity * position.entry_price)
        grid = self.state.get("grid_campaign")
        if grid:
            campaign = _grid_campaign_from_dict(grid)
            total += sum(abs(level.quantity * level.raw_price) for level in campaign.levels)
        return total

    def _available_notional_multiple(self, equity: float) -> float:
        if equity <= 0.0:
            return 0.0
        headroom = equity * self.shadow.max_gross_notional_multiple - self._committed_notional()
        return max(0.0, headroom / equity)

    def _mark_price(self, symbol: str, fallback: float, fetch_mark: bool) -> float:
        mark = self._last_marks.get(symbol, fallback)
        if fetch_mark:
            try:
                rows = self.client.klines(symbol, "1m", 2)
                if rows:
                    mark = rows[-1].close
            except Exception:
                pass
        self._last_marks[symbol] = mark
        return mark

    def _record_reject(self, strategy: str, reason: str) -> None:
        rejected = self.state["rejected"][strategy]
        rejected[reason] = int(rejected.get(reason, 0)) + 1

    def _update_drawdown(self, equity: float) -> None:
        peak = max(float(self.state["peak_equity"]), equity)
        self.state["peak_equity"] = peak
        drawdown = (peak - equity) / peak if peak > 0.0 else 1.0
        self.state["max_drawdown_pct"] = max(float(self.state["max_drawdown_pct"]), drawdown)

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
            handle.write(json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False) + "\n")

    def _persist_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(self.state), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.state_path)

    def _write_report(self, account: AccountSnapshot) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_jsonable(self.acceptance_report(account)), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.report_path)

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
                "another combined shadow process already owns this sample; stop it before starting GUI"
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

    @staticmethod
    def _group_trades(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            grouped[str(trade.get(field, "unknown"))].append(trade)
        output: dict[str, dict[str, Any]] = {}
        for key, rows in sorted(grouped.items()):
            wins = [row for row in rows if float(row["net_pnl"]) > 0.0]
            losses = [row for row in rows if float(row["net_pnl"]) <= 0.0]
            gross_profit = sum(float(row["net_pnl"]) for row in wins)
            gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
            output[key] = {
                "trade_count": len(rows),
                "net_pnl": sum(float(row["net_pnl"]) for row in rows),
                "win_rate": len(wins) / len(rows),
                "profit_factor": gross_profit / gross_loss if gross_loss > 0.0 else (
                    math.inf if gross_profit > 0.0 else 0.0
                ),
            }
        return output


def _logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined Breakout + Trend Grid max-two shadow runner")
    parser.add_argument(
        "--config",
        default="config.gui.combined-volatility-trend-grid-max2-shadow.json",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = load_live_config(args.config)
    client = BinanceFuturesClient(
        api_key=None,
        api_secret=None,
        environment=config.exchange.environment,
        recv_window=config.exchange.recv_window,
        timeout_seconds=config.exchange.timeout_seconds,
    )
    trader = CombinedVolatilityTrendGridShadowTrader(config, client, logger=_logger)
    if args.report_only:
        print(json.dumps(_jsonable(trader.acceptance_report()), indent=2, ensure_ascii=False))
        return 0
    if args.once:
        trader.validate_startup()
        trader._acquire_runtime_lock()
        try:
            trader.run_once()
        finally:
            trader._release_runtime_lock()
        print(json.dumps(_jsonable(trader.acceptance_report()), indent=2, ensure_ascii=False))
        return 0
    stop = threading.Event()
    try:
        trader.run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
