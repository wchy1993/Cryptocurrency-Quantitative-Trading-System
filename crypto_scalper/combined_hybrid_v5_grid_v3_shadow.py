from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import threading
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .binance_client import BinanceFuturesClient
from .combined_hybrid_v5_grid_v3_backtest import (
    COMBINED_V5_GRID_V3_NAME,
    _daily_cap_candidates,
    build_frozen_configs,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    NOTIONAL_HEADROOM_SAFETY,
)
from .combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
    _closed_candles,
    _compact_series,
    _jsonable,
    _open_position_from_dict,
    _resolve_config_path,
    _utc_now,
)
from .indicators import atr, ema
from .live_config import LiveAppConfig, load_live_config
from .live_trader import AccountSnapshot
from .models import Candle
from .trend_grid import build_trend_grid_timeline
from .trend_grid_optimize import GridCandidate
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .volatility_breakout import build_dual_thrust_signals
from .volatility_breakout_exit_protection import (
    ProtectedPosition,
    _process_protected_bar,
    _protected_position,
)
from .volatility_breakout_optimize import (
    Candidate,
    OpenPosition,
    UNIVERSE_50,
    _candidate_sort_key,
    _entry_position,
    minute_datetime,
    minute_token,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    enrich_candidates_v4,
    filter_candidates_v4,
)


HYBRID_V5_GRID_V3_SHADOW_VERSION = "hybrid_v5_grid_v3_max2_shadow_20260721"


def _source_value_path(value: Any, anchor: Path) -> Path:
    raw = value.get("path") if isinstance(value, dict) else value
    if not raw:
        raise RuntimeError("combined source config path is missing")
    return _resolve_config_path(str(raw), anchor)


def _load_hybrid_source_bundle(config: LiveAppConfig) -> dict[str, Any]:
    shadow = config.combined_volatility_trend_grid_shadow
    combined_path = _resolve_config_path(shadow.source_combined_config_path)
    combined_payload = json.loads(combined_path.read_text(encoding="utf-8"))
    sources = combined_payload.get("source_configs", {})
    breakout_path = _source_value_path(sources.get(BREAKOUT_KEY), combined_path)
    grid_path = _source_value_path(sources.get(GRID_KEY), combined_path)
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


def hybrid_v5_grid_v3_shadow_config_hash(config: LiveAppConfig) -> str:
    bundle = _load_hybrid_source_bundle(config)
    risk = config.risk
    payload = {
        "strategy_name": COMBINED_V5_GRID_V3_NAME,
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
            "hybrid_v5_partial_8r_fraction_0_05",
            "grid_v3_point_in_time_market_overlay",
            "grid_take_profit_before_new_grid_fill",
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _efficiency_ratio(closes: list[float], index: int, lookback: int) -> float:
    displacement = closes[index] - closes[index - lookback]
    path = sum(
        abs(closes[row] - closes[row - 1])
        for row in range(index - lookback + 1, index + 1)
    )
    return displacement / max(path, 1e-12)


def build_hourly_market_context(
    universe: tuple[str, ...],
    hourly_by_symbol: dict[str, list[Candle]],
) -> dict[int, dict[str, V4MarketSnapshot]]:
    """Reproduce the backtest's point-in-time V4 context from closed 1h bars."""

    local: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for symbol in universe:
        candles = hourly_by_symbol.get(symbol, ())
        if len(candles) < 56:
            continue
        closes = [candle.close for candle in candles]
        ema21 = ema(closes, 21)
        ema55 = ema(closes, 55)
        atr14 = atr(list(candles), 14)
        for index in range(55, len(candles)):
            available = minute_token(candles[index].timestamp + timedelta(minutes=60))
            local[available][symbol] = {
                "return_4h": closes[index] / max(closes[index - 4], 1e-12) - 1.0,
                "efficiency_12h": _efficiency_ratio(closes, index, 12),
                "above_ema21": float(closes[index] > ema21[index]),
                "ema55_atr": (closes[index] - ema55[index])
                / max(atr14[index], 1e-12),
            }

    breadth_by_minute: dict[int, float] = {}
    efficiency_by_minute: dict[int, float] = {}
    required = max(5, len(universe) // 2)
    for minute, rows in local.items():
        if len(rows) < required:
            continue
        breadth_by_minute[minute] = statistics.mean(
            row["above_ema21"] for row in rows.values()
        )
        efficiency_by_minute[minute] = statistics.mean(
            abs(row["efficiency_12h"]) for row in rows.values()
        )

    output: dict[int, dict[str, V4MarketSnapshot]] = {}
    for minute, rows in local.items():
        breadth = breadth_by_minute.get(minute)
        market_efficiency = efficiency_by_minute.get(minute)
        btc = rows.get("BTCUSDT")
        eth = rows.get("ETHUSDT")
        if breadth is None or market_efficiency is None or btc is None or eth is None:
            continue
        breadth_change = breadth - breadth_by_minute.get(minute - 240, breadth)
        output[minute] = {
            symbol: V4MarketSnapshot(
                available_minute=minute,
                symbol=symbol,
                btc_return_4h=btc["return_4h"],
                eth_return_4h=eth["return_4h"],
                breadth_above_ema21=breadth,
                breadth_change_4h=breadth_change,
                symbol_return_4h=row["return_4h"],
                symbol_efficiency_12h=row["efficiency_12h"],
                market_efficiency_12h=market_efficiency,
                symbol_ema55_atr=row["ema55_atr"],
            )
            for symbol, row in rows.items()
        }
    return output


def _protected_position_to_dict(protected: ProtectedPosition) -> dict[str, Any]:
    return _jsonable(asdict(protected))


def _protected_position_from_dict(payload: dict[str, Any]) -> ProtectedPosition:
    values = dict(payload)
    position = _open_position_from_dict(dict(values.pop("position")))
    values["realized_legs"] = [dict(row) for row in values.get("realized_legs", ())]
    return ProtectedPosition(position=position, **values)


class CombinedHybridV5GridV3ShadowTrader(CombinedVolatilityTrendGridShadowTrader):
    """Dry-run-only shared account matching the frozen Hybrid-v5/Grid-v3 backtest."""

    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: Callable[[str], None] | None = None,
        account_callback: Callable[[AccountSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.shadow = config.combined_volatility_trend_grid_shadow
        self.breakout_shadow = config.dual_thrust_shadow
        self.client = client
        self.logger = logger or (lambda _message: None)
        self.account_callback = account_callback
        from .risk import execution_config_from_live_config

        self.execution = execution_config_from_live_config(
            config, cost_experiment="full_cost", mode="conservative"
        )
        self.source_bundle = _load_hybrid_source_bundle(config)
        configs = build_frozen_configs(
            self.source_bundle["breakout_path"], self.source_bundle["grid_path"]
        )
        self.global_config = configs["combined"]
        self.breakout_build_signal_config = configs["breakout_build_signal"]
        self.breakout_signal_config = configs["breakout_signal"]
        self.breakout_regime_config = configs["breakout_regime"]
        self.breakout_portfolio_config = configs["breakout_portfolio"]
        self.breakout_exit_config = configs["breakout_exit"]
        self.grid_signal_config = configs["grid_signal"]
        self.grid_market_overlay: GridMarketOverlay = configs["grid_overlay"]
        self.grid_portfolio_config = configs["grid_portfolio"]
        self.config_hash = hybrid_v5_grid_v3_shadow_config_hash(config)
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
                source_strategy=COMBINED_V5_GRID_V3_NAME,
            )
            self.state["sample_initialized_event_logged"] = True
            self._persist_state()

    def validate_startup(self) -> None:
        self.validate_startup_settings_only()
        if self.config.risk.cost_experiment != "full_cost":
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires full_cost execution")
        if self.config.risk.backtest_mode != "conservative":
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires conservative execution")
        if not self.config.risk.funding_enabled:
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires funding_enabled=true")
        if self.config.exchange.environment != "mainnet":
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow must use mainnet public market data")
        if self.config.risk.starting_capital_usdt != 200.0:
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow must start from the backtest's 200U")
        if self.shadow.max_open_positions != 2 or self.config.trading.max_open_positions != 2:
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires global max_open_positions=2")
        if self.shadow.max_open_positions_per_strategy != 1:
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires one position per strategy")
        if self.breakout_portfolio_config.max_open_positions != 1:
            raise RuntimeError("frozen Hybrid v5 sleeve must remain max one position")
        if self.grid_portfolio_config.max_open_campaigns != 1:
            raise RuntimeError("frozen Grid v3 sleeve must remain max one campaign")
        expected_priority = (BREAKOUT_KEY, GRID_KEY)
        if tuple(self.shadow.entry_priority) != expected_priority:
            raise RuntimeError(f"entry priority must remain {expected_priority}")
        if tuple(self.global_config.entry_priority) != expected_priority:
            raise RuntimeError("combined research entry priority changed")
        if self.global_config.max_open_positions != self.shadow.max_open_positions:
            raise RuntimeError("combined source/live position limits differ")
        if not math.isclose(
            self.global_config.max_gross_notional_multiple,
            self.shadow.max_gross_notional_multiple,
        ):
            raise RuntimeError("combined source/live notional limits differ")
        if not math.isclose(
            self.global_config.hard_drawdown_stop_pct,
            self.shadow.hard_drawdown_stop_pct,
        ):
            raise RuntimeError("combined source/live drawdown limits differ")
        if self.global_config.allow_same_symbol_across_strategies:
            raise RuntimeError("frozen combined strategy forbids same-symbol overlap")
        if self.shadow.allow_same_symbol_across_strategies:
            raise RuntimeError("live combined strategy forbids same-symbol overlap")

        combined_payload = self.source_bundle["combined_payload"]
        if combined_payload.get("strategy_name") != COMBINED_V5_GRID_V3_NAME:
            raise RuntimeError("combined research strategy name changed")
        if self.shadow.strategy_name != COMBINED_V5_GRID_V3_NAME:
            raise RuntimeError("GUI combined strategy name changed")
        if self.shadow.frozen_version != HYBRID_V5_GRID_V3_SHADOW_VERSION:
            raise RuntimeError("GUI combined frozen version changed")

        symbols = tuple(self.shadow.enabled_symbols)
        if symbols != tuple(UNIVERSE_50):
            raise RuntimeError("Hybrid-v5/Grid-v3 shadow requires the frozen 50-symbol universe")
        if symbols != tuple(self.breakout_shadow.enabled_symbols):
            raise RuntimeError("combined and Breakout symbol universes differ")
        if symbols != tuple(self.source_bundle["grid_payload"]["symbols"]):
            raise RuntimeError("combined and Grid-v3 symbol universes differ")
        if tuple(self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI trading symbols differ from frozen research universe")
        if tuple(self.config.trading.entry_symbols or self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI entry symbols differ from frozen research universe")

        signal_fields = (
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
        )
        for field in signal_fields:
            if getattr(self.breakout_shadow, field) != getattr(
                self.breakout_signal_config, field
            ):
                raise RuntimeError(f"Hybrid v5 live/source signal parameter differs: {field}")
        portfolio_fields = (
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
        )
        for field in portfolio_fields:
            if getattr(self.breakout_shadow, field) != getattr(
                self.breakout_portfolio_config, field
            ):
                raise RuntimeError(f"Hybrid v5 live/source portfolio parameter differs: {field}")
        if not math.isclose(
            self.breakout_shadow.max_directional_btc_return_4h,
            self.breakout_regime_config.max_directional_btc_return_4h,
        ):
            raise RuntimeError("Hybrid v5 live/source BTC regime limit differs")
        if not math.isclose(
            self.breakout_shadow.partial_take_profit_r,
            self.breakout_exit_config.partial_take_profit_r,
        ) or not math.isclose(
            self.breakout_shadow.partial_take_profit_fraction,
            self.breakout_exit_config.partial_take_profit_fraction,
        ):
            raise RuntimeError("Hybrid v5 live/source partial take-profit differs")

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
            self.config.vbp_strategy.enabled,
            self.config.reversal_alpha.enabled,
            self.config.cmipr.enabled,
            self.config.mtper.enabled,
            self.config.mtpc.enabled,
            self.config.macro_events.enabled,
        )
        if any(legacy_flags):
            raise RuntimeError("all legacy strategies must remain disabled")
        self.breakout_exit_config.validate()
        self.grid_market_overlay.validate()
        self.client.ping()

    def _build_entry_candidates(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        latest_available: datetime,
    ) -> tuple[list[Candidate], list[GridCandidate], float]:
        symbols = tuple(self.shadow.enabled_symbols)
        context = build_hourly_market_context(symbols, candles_by_symbol)
        raw_breakout: dict[int, list[Candidate]] = defaultdict(list)
        raw_grid: dict[int, list[GridCandidate]] = defaultdict(list)
        for symbol, candles in candles_by_symbol.items():
            for signal in build_dual_thrust_signals(
                symbol, candles, self.breakout_build_signal_config
            ):
                entry_minute = minute_token(signal.signal_available_time)
                raw_breakout[entry_minute].append(
                    Candidate(signal=signal, entry_minute=entry_minute)
                )
            _, grid_signals = build_trend_grid_timeline(
                symbol, candles, self.grid_signal_config
            )
            for signal in grid_signals:
                entry_minute = minute_token(signal.signal_available_time)
                raw_grid[entry_minute].append(
                    GridCandidate(signal=signal, entry_minute=entry_minute)
                )

        enriched = enrich_candidates_v4(dict(raw_breakout), context)
        filtered = filter_candidates_v4(
            enriched,
            self.breakout_build_signal_config,
            self.breakout_regime_config,
            context,
        )
        filtered = _daily_cap_candidates(filtered, 2)
        grid_filtered = apply_market_overlay(
            dict(raw_grid), context, self.grid_market_overlay
        )
        current_minute = minute_token(latest_available)
        breakout_candidates = [
            candidate
            for candidate in filtered.get(current_minute, ())
            if candidate.signal.event_id
            not in self.state["seen_events"][BREAKOUT_KEY]
        ]
        grid_candidates = [
            candidate
            for candidate in grid_filtered.get(current_minute, ())
            if candidate.signal.event_id not in self.state["seen_events"][GRID_KEY]
        ]
        breakout_candidates.sort(
            key=lambda candidate: _candidate_sort_key(
                candidate, self.breakout_portfolio_config.ranking_mode
            )
        )
        latest_context = context.get(current_minute, {})
        btc_snapshot = latest_context.get("BTCUSDT")
        btc_return_4h = btc_snapshot.btc_return_4h if btc_snapshot else 0.0
        return breakout_candidates, grid_candidates, btc_return_4h

    def _open_breakout_candidate(self, candidate: Candidate, now: datetime) -> bool:
        signal = candidate.signal
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: Hybrid v5 shadow成交参考不可用 ({exc})")
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
            equity
            if self.breakout_portfolio_config.compound
            else float(self.state["starting_equity"]),
        )
        if opened is None:
            return False
        position, entry_fee = opened
        protected = _protected_position(
            position, self.breakout_exit_config, self.execution
        )
        self.state["cash"] -= entry_fee
        self.state["breakout_position"] = _protected_position_to_dict(protected)
        self.state["breakout_last_processed_minute"] = position.entry_minute - 1
        self._record_entry(BREAKOUT_KEY, signal.symbol, now)
        self._last_marks[signal.symbol] = raw
        self.log(
            f"{signal.symbol}: Hybrid v5开仓 {signal.direction.name} "
            f"qty={position.quantity:.8g} fill={position.entry_price:.8g} "
            f"stop={position.stop_price:.8g} 风险={position.risk_budget:.3f}U"
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
            partial_take_profit_r=self.breakout_exit_config.partial_take_profit_r,
            partial_take_profit_fraction=self.breakout_exit_config.partial_take_profit_fraction,
        )
        return True

    def _decode_breakout_position(self, payload: dict[str, Any]) -> OpenPosition:
        return _protected_position_from_dict(payload).position

    def _manage_breakout_position(self, now: datetime) -> None:
        payload = self.state.get("breakout_position")
        if not payload:
            return
        protected = _protected_position_from_dict(payload)
        position = protected.position
        symbol = position.candidate.signal.symbol
        try:
            candles = self.client.klines(symbol, "1m", 1500)
            closed = _closed_candles(candles, 1, now)
            rules = self.client.symbol_rules(symbol)
            execution = self._execution_with_funding(
                symbol, minute_datetime(position.entry_minute), now
            )
        except Exception as exc:
            self.log(f"{symbol}: Hybrid v5持仓数据/资金费不可用，保留仓位 ({exc})")
            self._append_event(
                "position_data_error", strategy=BREAKOUT_KEY, symbol=symbol, error=str(exc)
            )
            return
        last = int(
            self.state.get("breakout_last_processed_minute")
            or position.entry_minute - 1
        )
        if not self._history_is_contiguous(BREAKOUT_KEY, symbol, closed, last):
            return
        series = _compact_series(closed)
        for index, minute_value in enumerate(series.minutes):
            minute = int(minute_value)
            if minute <= last:
                continue
            partial_before = len(protected.realized_legs)
            trade, cash_delta = _process_protected_bar(
                protected,
                minute,
                series,
                index,
                self.breakout_signal_config,
                self.breakout_exit_config,
                execution,
                rules,
            )
            self.state["cash"] += cash_delta
            last = minute
            self.state["breakout_last_processed_minute"] = minute
            if len(protected.realized_legs) > partial_before:
                leg = protected.realized_legs[-1]
                self.log(
                    f"{symbol}: Hybrid v5 8R分批止盈 "
                    f"qty={leg['quantity']:.8g} 净盈亏={leg['net_pnl']:+.4f}U"
                )
                self._append_event("partial_exit", strategy=BREAKOUT_KEY, **leg)
            if trade is None:
                continue
            tagged = {"strategy": BREAKOUT_KEY, "funding_status": "complete", **trade}
            self.state["trades"].append(tagged)
            self.state["breakout_position"] = None
            self._set_cooldown(
                BREAKOUT_KEY,
                symbol,
                minute,
                self.breakout_portfolio_config.symbol_cooldown_minutes,
            )
            self.log(
                f"{symbol}: Hybrid v5平仓 {tagged['exit_reason']} "
                f"净盈亏={tagged['net_pnl']:+.4f}U ({tagged['pnl_r']:+.3f}R)"
            )
            self._append_event("exit", **tagged)
            return
        self.state["breakout_position"] = _protected_position_to_dict(protected)
        if candles:
            self._last_marks[symbol] = candles[-1].close

    def acceptance_report(
        self, account: AccountSnapshot | None = None
    ) -> dict[str, Any]:
        report = super().acceptance_report(account)
        report.update(
            {
                "important_note": (
                    "Historical 3m/6m results are research references only. This report "
                    "contains only observations collected after this new dry-run sample began."
                ),
                "historical_reference": {
                    "three_month": {
                        "period": {
                            "start": "2026-04-19T00:00:00",
                            "end": "2026-07-19T00:00:00",
                        },
                        "trade_count": 172,
                        "net_profit_usdt": 2873.6194113148113,
                        "return_pct": 1436.8097056574056,
                        "profit_factor": 1.7501308782593792,
                        "win_rate_pct": 38.372093023255815,
                        "max_drawdown_pct": 33.3468640108728,
                    },
                    "six_month": {
                        "period": {
                            "start": "2026-01-19T00:00:00",
                            "end": "2026-07-19T00:00:00",
                        },
                        "trade_count": 322,
                        "net_profit_usdt": 10624.64905272653,
                        "return_pct": 5312.324526363265,
                        "profit_factor": 1.7425537985390949,
                        "win_rate_pct": 36.95652173913043,
                        "max_drawdown_pct": 36.48012239781913,
                    },
                },
                "breakout_partial_exit_count": sum(
                    int(row.get("partial_exit_count", 0))
                    for row in report.get("trades", ())
                    if row.get("strategy") == BREAKOUT_KEY
                ),
                "grid_v3_market_overlay": asdict(self.grid_market_overlay),
            }
        )
        return report


def _logger(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid v5 Breakout plus Grid v3 max-two dry-run shadow"
    )
    parser.add_argument(
        "--config",
        default="config.gui.combined-hybrid-v5-grid-v3-max2-shadow.json",
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
    trader = CombinedHybridV5GridV3ShadowTrader(config, client, logger=_logger)
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
