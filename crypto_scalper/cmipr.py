from __future__ import annotations

import bisect
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .data import interval_to_milliseconds
from .indicators import atr, ema, macd
from .live_trader import EntryCandidate
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import funding_at, oi_change_at


CMIPR_REASON_TOKEN = "cross_sectional_momentum_ignition_pyramid"


class CmiprMarketRegime(str, Enum):
    BULL_EXPANSION = "BULL_EXPANSION"
    BEAR_EXPANSION = "BEAR_EXPANSION"
    NEUTRAL = "NEUTRAL"
    OVERHEATED_BULL = "OVERHEATED_BULL"
    OVEREXTENDED_BEAR = "OVEREXTENDED_BEAR"
    CHAOS_NO_TRADE = "CHAOS_NO_TRADE"


class CmiprState(str, Enum):
    IDLE = "IDLE"
    COMPRESSION_WATCH = "COMPRESSION_WATCH"
    LONG_IGNITION_PENDING = "LONG_IGNITION_PENDING"
    SHORT_IGNITION_PENDING = "SHORT_IGNITION_PENDING"
    ENTRY_CONFIRMING = "ENTRY_CONFIRMING"
    INITIAL_ENTRY_READY = "INITIAL_ENTRY_READY"
    ORDER_PENDING = "ORDER_PENDING"
    PARTIAL_FILL = "PARTIAL_FILL"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    ADDON_1_ARMED = "ADDON_1_ARMED"
    ADDON_1_POSITION = "ADDON_1_POSITION"
    ADDON_2_ARMED = "ADDON_2_ARMED"
    FULL_POSITION = "FULL_POSITION"
    RUNNER = "RUNNER"
    CANCEL_PENDING = "CANCEL_PENDING"
    EXITING = "EXITING"
    COOLDOWN = "COOLDOWN"
    RECOVERY_AFTER_RESTART = "RECOVERY_AFTER_RESTART"


@dataclass(frozen=True)
class CmiprRegimeSnapshot:
    state: CmiprMarketRegime
    btc_1h_return: float
    btc_4h_return: float
    eth_1h_return: float
    breadth_positive_1h: float
    breadth_above_ema21: float
    raw_state: CmiprMarketRegime
    confirmed_bars: int


@dataclass(frozen=True)
class CmiprCompressionSnapshot:
    atr_value: float
    atr_percentile: float
    atr_to_average: float
    channel_width_atr: float
    volume_contraction: float
    prior_move_atr: float
    failed_breakouts: int
    range_high: float
    range_low: float


@dataclass(frozen=True)
class CmiprIgnitionSnapshot:
    direction: Direction
    candle: Candle
    breakout_level: float
    breakout_distance_atr: float
    body_atr: float
    close_position: float
    wick_ratio: float
    volume_ratio: float
    ranking_score: float
    ranking_percentile: float
    quality_score: float
    stop_price: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass
class CmiprSymbolRuntime:
    state: CmiprState = CmiprState.IDLE
    event_id: str | None = None
    direction: Direction = Direction.FLAT
    ignition: CmiprIgnitionSnapshot | None = None
    pending_time: datetime | None = None
    expire_time: datetime | None = None
    last_processed_5m: datetime | None = None
    consumed_event_ids: set[str] = field(default_factory=set)
    cooldown_until: datetime | None = None


@dataclass(frozen=True)
class CmiprDerivativeCoverage:
    model_variant: str
    requested_start: str | None
    requested_end: str | None
    oi_symbols: int
    taker_symbols: int
    funding_symbols: int
    basis_symbols: int
    oi_time_range: tuple[str | None, str | None]
    taker_time_range: tuple[str | None, str | None]
    funding_time_range: tuple[str | None, str | None]
    basis_time_range: tuple[str | None, str | None]
    eligible: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_variant": self.model_variant,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "oi_symbols": self.oi_symbols,
            "taker_symbols": self.taker_symbols,
            "funding_symbols": self.funding_symbols,
            "basis_symbols": self.basis_symbols,
            "oi_time_range": list(self.oi_time_range),
            "taker_time_range": list(self.taker_time_range),
            "funding_time_range": list(self.funding_time_range),
            "basis_time_range": list(self.basis_time_range),
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass
class CmiprOrderLifecycle:
    symbol: str
    state: CmiprState = CmiprState.IDLE
    requested_quantity: float = 0.0
    filled_quantity: float = 0.0
    protection_attempts: int = 0
    last_error: str = ""

    def submit_entry(self, requested_quantity: float) -> None:
        if requested_quantity <= 0:
            raise ValueError("requested_quantity must be positive")
        self.requested_quantity = requested_quantity
        self.filled_quantity = 0.0
        self.state = CmiprState.ORDER_PENDING

    def record_fill(self, filled_quantity: float) -> None:
        if self.state not in (CmiprState.ORDER_PENDING, CmiprState.PARTIAL_FILL):
            raise RuntimeError(f"fill not allowed from {self.state.value}")
        self.filled_quantity += max(0.0, filled_quantity)
        if self.filled_quantity <= 0:
            return
        self.state = (
            CmiprState.PARTIAL_FILL
            if self.filled_quantity + 1e-12 < self.requested_quantity
            else CmiprState.PROTECTION_PENDING
        )

    def request_protection(self) -> None:
        if self.filled_quantity <= 0:
            raise RuntimeError("cannot protect an unfilled order")
        self.protection_attempts += 1
        self.state = CmiprState.PROTECTION_PENDING

    def protection_result(self, success: bool, error: str = "") -> str:
        if success:
            self.last_error = ""
            self.state = CmiprState.PROTECTED
            return "protected"
        self.last_error = error or "protective_stop_failed"
        self.state = CmiprState.EXITING
        return "emergency_reduce_or_close"

    def request_cancel(self) -> None:
        self.state = CmiprState.CANCEL_PENDING

    def recover_after_restart(self, open_quantity: float, protective_stop_present: bool) -> str:
        self.state = CmiprState.RECOVERY_AFTER_RESTART
        self.filled_quantity = max(0.0, open_quantity)
        if self.filled_quantity <= 0:
            self.state = CmiprState.IDLE
            return "no_position"
        if protective_stop_present:
            self.state = CmiprState.PROTECTED
            return "recovered_protected"
        self.state = CmiprState.PROTECTION_PENDING
        return "replace_protection_or_emergency_flatten"


def audit_derivative_coverage(
    config: Any,
    symbols: tuple[str, ...],
    requested_start: datetime | None,
    requested_end: datetime | None,
) -> CmiprDerivativeCoverage:
    cmipr = config.cmipr
    model_variant = str(cmipr.research.model_variant).strip().lower()
    oi_root = Path(cmipr.auxiliary_oi_data_dir)
    funding_root = Path(cmipr.auxiliary_funding_data_dir)
    basis_root = Path(getattr(cmipr, "auxiliary_basis_data_dir", "")) if getattr(cmipr, "auxiliary_basis_data_dir", "") else None
    oi_files = _files_for_symbols(oi_root, symbols, "*_oi_*.csv")
    taker_files = _files_for_symbols(oi_root, symbols, "*_taker_*.csv")
    funding_files = _files_for_symbols(funding_root, symbols, "*_funding_*.csv")
    basis_files = _files_for_symbols(basis_root, symbols, "*_basis_*.csv") if basis_root else []
    oi_range = _csv_time_range(oi_files)
    taker_range = _csv_time_range(taker_files)
    funding_range = _csv_time_range(funding_files)
    basis_range = _csv_time_range(basis_files)
    if model_variant == "core":
        eligible = True
        reason = "core_model_does_not_require_derivatives_history"
    elif model_variant != "derivatives_enhanced":
        eligible = False
        reason = f"unknown_model_variant:{model_variant}"
    else:
        required_sets = [("oi", oi_files, oi_range), ("taker", taker_files, taker_range), ("basis", basis_files, basis_range)]
        missing = [name for name, files, _ in required_sets if len({path.name.split("_", 1)[0] for path in files}) < len(symbols)]
        insufficient = [name for name, _, time_range in required_sets if not _range_covers(time_range, requested_start, requested_end)]
        eligible = not missing and not insufficient
        reason = "ok" if eligible else f"derivatives_history_unavailable missing={','.join(missing) or 'none'} coverage={','.join(insufficient) or 'none'}"
    return CmiprDerivativeCoverage(
        model_variant=model_variant,
        requested_start=requested_start.isoformat() if requested_start else None,
        requested_end=requested_end.isoformat() if requested_end else None,
        oi_symbols=len({path.name.split("_", 1)[0] for path in oi_files}),
        taker_symbols=len({path.name.split("_", 1)[0] for path in taker_files}),
        funding_symbols=len({path.name.split("_", 1)[0] for path in funding_files}),
        basis_symbols=len({path.name.split("_", 1)[0] for path in basis_files}),
        oi_time_range=_format_range(oi_range),
        taker_time_range=_format_range(taker_range),
        funding_time_range=_format_range(funding_range),
        basis_time_range=_format_range(basis_range),
        eligible=eligible,
        reason=reason,
    )


class CmiprEngine:
    def __init__(
        self,
        config: Any,
        candles_by_timeframe: dict[str, dict[str, list[Candle]]],
        auxiliary_features: dict[str, dict[str, Any]] | None = None,
        shared_feature_cache: dict[tuple[Any, ...], Any] | None = None,
    ) -> None:
        self.config = config
        self.cmipr = config.cmipr
        self.candles = candles_by_timeframe
        self.timestamps = {
            timeframe: {
                symbol: [candle.timestamp for candle in candles]
                for symbol, candles in by_symbol.items()
            }
            for timeframe, by_symbol in candles_by_timeframe.items()
        }
        configured = tuple(self.cmipr.enabled_symbols or config.trading.entry_symbols or config.trading.symbols)
        self.symbols = tuple(symbol for symbol in configured if all(symbol in self.candles.get(tf, {}) for tf in ("5m", "15m", "30m", "1h", "4h")))
        self.auxiliary_features = auxiliary_features or {}
        self.shared_feature_cache = shared_feature_cache if shared_feature_cache is not None else {}
        self.runtime = {symbol: CmiprSymbolRuntime() for symbol in self.symbols}
        self.stats: Counter[str] = Counter()
        self.regime_history: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self._active_regime = CmiprMarketRegime.NEUTRAL
        self._raw_regime = CmiprMarketRegime.NEUTRAL
        self._raw_regime_count = 0
        self._exit_count = 0
        self._last_regime_bar: datetime | None = None
        self._active_since_bar: datetime | None = None
        self._last_ignition_bar: datetime | None = None
        self._last_entry_time: datetime | None = None

    def scan(
        self,
        decision_time: datetime,
        occupied_symbols: set[str],
        allowed_symbols: set[str] | None = None,
    ) -> list[EntryCandidate]:
        min_interval = max(0, int(self.cmipr.risk_control.global_min_entry_interval_seconds))
        if self._last_entry_time is not None and (decision_time - self._last_entry_time).total_seconds() < min_interval:
            self.reject_reasons["global_entry_interval"] += 1
            return []
        regime = self.regime(decision_time)
        self.regime_history[regime.state.value] += 1
        rankings = self.rankings(decision_time)
        allowed_symbols = allowed_symbols or set(self.symbols)
        candidates: list[EntryCandidate] = []
        for symbol in self.symbols:
            runtime = self.runtime[symbol]
            if symbol in occupied_symbols or symbol not in allowed_symbols:
                continue
            if runtime.cooldown_until is not None and decision_time < runtime.cooldown_until:
                continue
            pullback_candidate = self._pending_candidate(symbol, decision_time, regime)
            if pullback_candidate is not None:
                candidates.append(pullback_candidate)
                continue
            if runtime.state not in (CmiprState.IDLE, CmiprState.COMPRESSION_WATCH, CmiprState.COOLDOWN):
                continue
            rank = rankings.get(symbol)
            if rank is None:
                continue
            ignition, reject = self._ignition(symbol, decision_time, regime, rank)
            if ignition is None:
                if reject:
                    self.reject_reasons[reject] += 1
                continue
            self.stats["ignition_count"] += 1
            event_id = f"{symbol}:{ignition.direction.name}:{ignition.candle.timestamp.isoformat()}"
            if event_id in runtime.consumed_event_ids:
                continue
            runtime.event_id = event_id
            runtime.direction = ignition.direction
            runtime.ignition = ignition
            runtime.pending_time = decision_time
            runtime.expire_time = decision_time + timedelta(minutes=max(1, int(self.cmipr.entry.pending_expiry_minutes)))
            runtime.state = CmiprState.LONG_IGNITION_PENDING if ignition.direction == Direction.LONG else CmiprState.SHORT_IGNITION_PENDING
            if str(self.cmipr.entry.mode).lower() == "confirmation_open":
                candidate = self._candidate_from_ignition(symbol, ignition, event_id, "confirmation_open")
                runtime.state = CmiprState.INITIAL_ENTRY_READY
                runtime.consumed_event_ids.add(event_id)
                candidates.append(candidate)

        candidates.sort(key=lambda item: (-item.rank_score, item.symbol))
        limit = max(1, int(self.cmipr.ranking.max_candidates_per_scan))
        selected = candidates[:limit]
        if selected:
            self._last_entry_time = decision_time
        return selected

    def regime(self, decision_time: datetime) -> CmiprRegimeSnapshot:
        snapshot = self._raw_regime_snapshot(decision_time)
        one_h = self._closed("1h", self.cmipr.regime.btc_symbol, decision_time, 3)
        latest_bar = one_h[-1].timestamp if one_h else None
        if latest_bar is not None and latest_bar != self._last_regime_bar:
            self._last_regime_bar = latest_bar
            if snapshot.raw_state == self._raw_regime:
                self._raw_regime_count += 1
            else:
                self._raw_regime = snapshot.raw_state
                self._raw_regime_count = 1
            held = _bar_distance(self._active_since_bar, latest_bar, "1h")
            min_hold = max(0, int(self.cmipr.regime.min_state_hold_bars_1h))
            if self._active_regime in (CmiprMarketRegime.BULL_EXPANSION, CmiprMarketRegime.BEAR_EXPANSION):
                maintain = self._maintain_active_regime(snapshot)
                self._exit_count = 0 if maintain else self._exit_count + 1
                if self._exit_count >= max(1, int(self.cmipr.regime.exit_confirmation_bars_1h)) and held >= min_hold:
                    self._active_regime = CmiprMarketRegime.NEUTRAL
                    self._active_since_bar = latest_bar
                    self._exit_count = 0
            elif self._raw_regime_count >= max(1, int(self.cmipr.regime.enter_confirmation_bars_1h)) and held >= min_hold:
                self._active_regime = snapshot.raw_state
                self._active_since_bar = latest_bar
        return CmiprRegimeSnapshot(
            state=self._active_regime,
            btc_1h_return=snapshot.btc_1h_return,
            btc_4h_return=snapshot.btc_4h_return,
            eth_1h_return=snapshot.eth_1h_return,
            breadth_positive_1h=snapshot.breadth_positive_1h,
            breadth_above_ema21=snapshot.breadth_above_ema21,
            raw_state=snapshot.raw_state,
            confirmed_bars=self._raw_regime_count,
        )

    def rankings(self, decision_time: datetime) -> dict[str, tuple[float, float, float]]:
        cache_key = ("rankings", decision_time)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        btc_1h = _period_return(self._closed("1h", self.cmipr.regime.btc_symbol, decision_time, 6), 1)
        btc_4h = _period_return(self._closed("4h", self.cmipr.regime.btc_symbol, decision_time, 4), 1)
        raw: dict[str, float] = {}
        extension: dict[str, float] = {}
        cfg = self.cmipr.ranking
        for symbol in self.symbols:
            c30 = self._closed("30m", symbol, decision_time, 40)
            c1h = self._closed("1h", symbol, decision_time, 30)
            c4h = self._closed("4h", symbol, decision_time, 30)
            if min(len(c30), len(c1h), len(c4h)) < 3:
                continue
            r30 = _period_return(c30, 1)
            r1h = _period_return(c1h, 1)
            r4h = _period_return(c4h, 1)
            volume_ratio = _volume_ratio(c30, 12)
            closes = [item.close for item in c1h]
            aligned = 1.0 if closes[-1] > ema(closes, 9)[-1] > ema(closes, 21)[-1] else -1.0
            score = (
                cfg.weight_return_30m * r30
                + cfg.weight_return_1h * r1h
                + cfg.weight_return_4h * r4h
                + cfg.weight_relative_btc_1h * (r1h - btc_1h)
                + cfg.weight_relative_btc_4h * (r4h - btc_4h)
                + cfg.weight_volume_trend * math.log(max(volume_ratio, 1e-6)) / 10.0
                + cfg.weight_trend_alignment * aligned / 100.0
            )
            atr_values = atr(c30, 14)
            extension[symbol] = abs(c30[-1].close - ema([item.close for item in c30], 21)[-1]) / max(atr_values[-1], 1e-12)
            raw[symbol] = score
        ordered = sorted(raw, key=lambda symbol: (raw[symbol], symbol))
        count = max(1, len(ordered) - 1)
        result = {
            symbol: (raw[symbol], index / count, extension[symbol])
            for index, symbol in enumerate(ordered)
        }
        self.shared_feature_cache[cache_key] = result
        return result

    def mark_order_pending(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.ORDER_PENDING

    def mark_partial_fill(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.PARTIAL_FILL

    def mark_protection_pending(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.PROTECTION_PENDING

    def mark_protected(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.PROTECTED

    def mark_cancel_pending(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.CANCEL_PENDING

    def mark_recovery_after_restart(self, symbol: str) -> None:
        if symbol in self.runtime:
            self.runtime[symbol].state = CmiprState.RECOVERY_AFTER_RESTART

    def mark_closed(self, symbol: str, timestamp: datetime) -> None:
        runtime = self.runtime.get(symbol)
        if runtime is None:
            return
        runtime.state = CmiprState.COOLDOWN
        runtime.cooldown_until = timestamp + timedelta(minutes=max(0, int(self.cmipr.risk_control.symbol_cooldown_minutes)))
        runtime.ignition = None
        runtime.event_id = None

    def report(self) -> dict[str, Any]:
        return {
            "strategy_name": CMIPR_REASON_TOKEN,
            "model_variant": self.cmipr.research.model_variant,
            "stats": dict(sorted(self.stats.items())),
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "regime_observations": dict(sorted(self.regime_history.items())),
            "states": dict(sorted(Counter(runtime.state.value for runtime in self.runtime.values()).items())),
        }

    def _raw_regime_snapshot(self, decision_time: datetime) -> CmiprRegimeSnapshot:
        cache_key = ("raw_regime", decision_time)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        cfg = self.cmipr.regime
        btc_1h = self._closed("1h", cfg.btc_symbol, decision_time, 30)
        btc_4h = self._closed("4h", cfg.btc_symbol, decision_time, 12)
        eth_1h = self._closed("1h", cfg.eth_symbol, decision_time, 30)
        if len(btc_1h) < cfg.ema_period + cfg.ema_slope_lookback or len(btc_4h) < 3:
            raw = CmiprMarketRegime.NEUTRAL
            result = CmiprRegimeSnapshot(raw, 0.0, 0.0, 0.0, 0.0, 0.0, raw, 0)
            self.shared_feature_cache[cache_key] = result
            return result
        btc_1h_return = _period_return(btc_1h, 1)
        btc_4h_return = _period_return(btc_4h, 1)
        eth_1h_return = _period_return(eth_1h, 1)
        positives = 0
        above = 0
        eligible = 0
        for symbol in self.symbols:
            candles = self._closed("1h", symbol, decision_time, cfg.ema_period + 3)
            if len(candles) < cfg.ema_period + 1:
                continue
            eligible += 1
            positives += int(_period_return(candles, 1) > 0)
            above += int(candles[-1].close > ema([item.close for item in candles], cfg.ema_period)[-1])
        breadth_positive = positives / max(eligible, 1)
        breadth_above = above / max(eligible, 1)
        btc_closes = [item.close for item in btc_1h]
        btc_ema = ema(btc_closes, cfg.ema_period)
        slope = btc_ema[-1] / max(btc_ema[-1 - cfg.ema_slope_lookback], 1e-12) - 1.0
        conflict = abs(breadth_positive - breadth_above)
        if abs(btc_1h_return) >= cfg.btc_shock_1h_pct or conflict >= cfg.max_direction_conflict:
            raw = CmiprMarketRegime.CHAOS_NO_TRADE
        elif breadth_above >= cfg.max_breadth_above_ema21_overheated and btc_1h_return > 0.02:
            raw = CmiprMarketRegime.OVERHEATED_BULL
        elif breadth_above <= 1.0 - cfg.max_breadth_above_ema21_overheated and btc_1h_return < -0.02:
            raw = CmiprMarketRegime.OVEREXTENDED_BEAR
        elif btc_closes[-1] > btc_ema[-1] and slope >= cfg.enter_ema_slope_pct and breadth_above >= cfg.enter_breadth_above_ema21 and breadth_positive >= cfg.min_breadth_positive_1h:
            raw = CmiprMarketRegime.BULL_EXPANSION
        elif btc_closes[-1] < btc_ema[-1] and slope <= -cfg.enter_ema_slope_pct and breadth_above <= 1.0 - cfg.enter_breadth_above_ema21 and breadth_positive <= 1.0 - cfg.min_breadth_positive_1h:
            raw = CmiprMarketRegime.BEAR_EXPANSION
        else:
            raw = CmiprMarketRegime.NEUTRAL
        result = CmiprRegimeSnapshot(raw, btc_1h_return, btc_4h_return, eth_1h_return, breadth_positive, breadth_above, raw, 0)
        self.shared_feature_cache[cache_key] = result
        return result

    def _maintain_active_regime(self, snapshot: CmiprRegimeSnapshot) -> bool:
        cfg = self.cmipr.regime
        btc = self._closed("1h", cfg.btc_symbol, self._last_regime_bar + timedelta(hours=1), cfg.ema_period + cfg.ema_slope_lookback + 2) if self._last_regime_bar else []
        if len(btc) < cfg.ema_period + cfg.ema_slope_lookback:
            return False
        values = ema([item.close for item in btc], cfg.ema_period)
        slope = values[-1] / max(values[-1 - cfg.ema_slope_lookback], 1e-12) - 1.0
        if self._active_regime == CmiprMarketRegime.BULL_EXPANSION:
            return slope >= cfg.exit_ema_slope_pct and snapshot.breadth_above_ema21 >= cfg.exit_breadth_above_ema21 and snapshot.raw_state != CmiprMarketRegime.CHAOS_NO_TRADE
        return slope <= -cfg.exit_ema_slope_pct and snapshot.breadth_above_ema21 <= 1.0 - cfg.exit_breadth_above_ema21 and snapshot.raw_state != CmiprMarketRegime.CHAOS_NO_TRADE

    def _ignition(
        self,
        symbol: str,
        decision_time: datetime,
        regime: CmiprRegimeSnapshot,
        rank: tuple[float, float, float],
    ) -> tuple[CmiprIgnitionSnapshot | None, str]:
        cfg = self.cmipr
        if regime.state not in (CmiprMarketRegime.BULL_EXPANSION, CmiprMarketRegime.BEAR_EXPANSION):
            return None, "regime_not_expansion"
        direction = Direction.LONG if regime.state == CmiprMarketRegime.BULL_EXPANSION else Direction.SHORT
        if direction == Direction.LONG and not cfg.allow_long:
            return None, "long_disabled"
        if direction == Direction.SHORT and not cfg.allow_short:
            return None, "short_disabled"
        score, percentile, extension = rank
        if direction == Direction.LONG and percentile < 1.0 - cfg.ranking.long_top_fraction:
            return None, "ranking_not_top"
        if direction == Direction.SHORT and percentile > cfg.ranking.short_bottom_fraction:
            return None, "ranking_not_bottom"
        if extension > cfg.ranking.max_extension_atr:
            return None, "ranking_overextended"
        compression, reason = self._compression(symbol, decision_time)
        if compression is None:
            return None, reason
        candles = self._closed("15m", symbol, decision_time, max(50, cfg.ignition.breakout_lookback_15m + 5))
        if len(candles) < cfg.ignition.breakout_lookback_15m + 2:
            return None, "ignition_warmup"
        latest = candles[-1]
        previous = candles[-cfg.ignition.breakout_lookback_15m - 1:-1]
        breakout_level = max(item.high for item in previous) if direction == Direction.LONG else min(item.low for item in previous)
        distance = direction.value * (latest.close - breakout_level) / max(compression.atr_value, 1e-12)
        candle_range = max(latest.high - latest.low, 1e-12)
        body_atr = abs(latest.close - latest.open) / max(compression.atr_value, 1e-12)
        close_position = (latest.close - latest.low) / candle_range if direction == Direction.LONG else (latest.high - latest.close) / candle_range
        wick = latest.high - max(latest.open, latest.close) if direction == Direction.LONG else min(latest.open, latest.close) - latest.low
        wick_ratio = max(0.0, wick / candle_range)
        volume_ratio = _volume_ratio(candles, 20)
        if not cfg.ignition.min_breakout_distance_atr <= distance <= cfg.ignition.max_breakout_distance_atr:
            return None, "ignition_breakout_distance"
        if body_atr < cfg.ignition.min_body_atr:
            return None, "ignition_body"
        if close_position < cfg.ignition.min_close_position:
            return None, "ignition_close_position"
        if wick_ratio > cfg.ignition.max_wick_ratio:
            return None, "ignition_wick"
        if volume_ratio < cfg.ignition.min_volume_ratio:
            return None, "ignition_volume"
        _, _, histogram = macd([item.close for item in candles])
        bars = max(1, int(cfg.ignition.macd_hist_expanding_bars))
        recent_hist = histogram[-bars - 1:]
        hist_ok = all(recent_hist[index] >= recent_hist[index - 1] for index in range(1, len(recent_hist))) if direction == Direction.LONG else all(recent_hist[index] <= recent_hist[index - 1] for index in range(1, len(recent_hist)))
        if not hist_ok:
            return None, "ignition_macd"
        derivative_reason = self._derivative_guard(symbol, decision_time, direction)
        if derivative_reason:
            return None, derivative_reason
        recent_5m = self._closed("5m", symbol, decision_time, 8)
        if len(recent_5m) < 3:
            return None, "stop_structure_warmup"
        structure = min(item.low for item in recent_5m[-3:]) if direction == Direction.LONG else max(item.high for item in recent_5m[-3:])
        stop = structure - cfg.entry.stop_atr_buffer * compression.atr_value if direction == Direction.LONG else structure + cfg.entry.stop_atr_buffer * compression.atr_value
        stop_atr = direction.value * (latest.close - stop) / max(compression.atr_value, 1e-12)
        if stop_atr < cfg.entry.min_stop_atr:
            return None, "stop_too_tight"
        if stop_atr > cfg.entry.max_stop_atr:
            return None, "stop_too_wide"
        stop_pct = abs(latest.close - stop) / max(latest.close, 1e-12)
        cost_pct = _configured_full_cost_pct(self.config)
        target_pct = stop_pct * max(1.2, float(cfg.exit.runner_activation_r))
        if target_pct / max(cost_pct, 1e-12) < cfg.entry.min_target_to_cost_ratio:
            return None, "target_to_cost"
        quality = min(1.0, 0.35 + 0.20 * close_position + 0.15 * min(volume_ratio / 3.0, 1.0) + 0.15 * (percentile if direction == Direction.LONG else 1.0 - percentile) + 0.15 * max(0.0, 1.0 - compression.atr_percentile))
        return CmiprIgnitionSnapshot(
            direction=direction,
            candle=latest,
            breakout_level=breakout_level,
            breakout_distance_atr=distance,
            body_atr=body_atr,
            close_position=close_position,
            wick_ratio=wick_ratio,
            volume_ratio=volume_ratio,
            ranking_score=score,
            ranking_percentile=percentile,
            quality_score=quality,
            stop_price=stop,
            stop_loss_pct=stop_pct,
            take_profit_pct=max(stop_pct * 20.0, 0.25),
        ), ""

    def _compression(self, symbol: str, decision_time: datetime) -> tuple[CmiprCompressionSnapshot | None, str]:
        cache_key = ("compression", symbol, decision_time)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        cfg = self.cmipr.compression
        candles = self._closed("30m", symbol, decision_time, max(cfg.atr_percentile_lookback + cfg.atr_period + 5, 120))
        if len(candles) < max(cfg.atr_percentile_lookback, cfg.lookback_bars_30m * 2, cfg.atr_period + 5):
            result = (None, "compression_warmup")
            self.shared_feature_cache[cache_key] = result
            return result
        atr_values = atr(candles, cfg.atr_period)
        current_atr = atr_values[-1]
        historical = atr_values[-cfg.atr_percentile_lookback:]
        percentile = sum(value <= current_atr for value in historical) / max(len(historical), 1)
        average_atr = sum(historical) / max(len(historical), 1)
        window = candles[-cfg.lookback_bars_30m:]
        channel_high = max(item.high for item in window)
        channel_low = min(item.low for item in window)
        channel_width = (channel_high - channel_low) / max(current_atr, 1e-12)
        half = max(2, len(window) // 2)
        recent_volume = sum(item.volume for item in window[-half:]) / half
        prior_volume = sum(item.volume for item in window[:-half]) / max(len(window) - half, 1)
        contraction = recent_volume / max(prior_volume, 1e-12)
        prior_move = abs(window[-1].close - window[0].close) / max(current_atr, 1e-12)
        failures = _failed_breakout_count(candles[-cfg.lookback_bars_30m * 2:], cfg.lookback_bars_30m)
        snapshot = CmiprCompressionSnapshot(current_atr, percentile, current_atr / max(average_atr, 1e-12), channel_width, contraction, prior_move, failures, channel_high, channel_low)
        if percentile > cfg.max_atr_percentile:
            result = (None, "compression_atr_percentile")
            self.shared_feature_cache[cache_key] = result
            return result
        if snapshot.atr_to_average > cfg.max_atr_to_average:
            result = (None, "compression_atr_ratio")
            self.shared_feature_cache[cache_key] = result
            return result
        if channel_width > cfg.max_channel_width_atr:
            result = (None, "compression_channel")
            self.shared_feature_cache[cache_key] = result
            return result
        if contraction > cfg.max_volume_contraction:
            result = (None, "compression_volume")
            self.shared_feature_cache[cache_key] = result
            return result
        if prior_move > cfg.max_prior_move_atr:
            result = (None, "compression_prior_move")
            self.shared_feature_cache[cache_key] = result
            return result
        if failures > cfg.max_failed_breakouts:
            result = (None, "compression_failed_breakouts")
            self.shared_feature_cache[cache_key] = result
            return result
        result = (snapshot, "")
        self.shared_feature_cache[cache_key] = result
        return result

    def _pending_candidate(self, symbol: str, decision_time: datetime, regime: CmiprRegimeSnapshot) -> EntryCandidate | None:
        runtime = self.runtime[symbol]
        if runtime.state not in (CmiprState.LONG_IGNITION_PENDING, CmiprState.SHORT_IGNITION_PENDING, CmiprState.ENTRY_CONFIRMING):
            return None
        if runtime.expire_time is not None and decision_time > runtime.expire_time:
            runtime.state = CmiprState.IDLE
            runtime.ignition = None
            self.stats["pending_expired"] += 1
            return None
        ignition = runtime.ignition
        if ignition is None or runtime.event_id is None:
            runtime.state = CmiprState.IDLE
            return None
        expected_regime = CmiprMarketRegime.BULL_EXPANSION if ignition.direction == Direction.LONG else CmiprMarketRegime.BEAR_EXPANSION
        if regime.state != expected_regime:
            runtime.state = CmiprState.IDLE
            runtime.ignition = None
            self.stats["pending_regime_invalidated"] += 1
            return None
        candles = self._closed(self.cmipr.entry.pullback_timeframe, symbol, decision_time, 30)
        after = [item for item in candles if item.timestamp > ignition.candle.timestamp]
        if not after:
            runtime.state = CmiprState.ENTRY_CONFIRMING
            return None
        latest = after[-1]
        if runtime.last_processed_5m == latest.timestamp:
            return None
        runtime.last_processed_5m = latest.timestamp
        atr_value = self._compression(symbol, decision_time)[0]
        if atr_value is None:
            return None
        atr_price = atr_value.atr_value
        if ignition.direction == Direction.LONG:
            extreme = min(item.low for item in after)
            depth = (ignition.candle.close - extreme) / max(atr_price, 1e-12)
            close_position = (latest.close - latest.low) / max(latest.high - latest.low, 1e-12)
            confirmed = latest.close > latest.open and latest.close >= ignition.breakout_level and latest.close >= ema([item.close for item in candles], 9)[-1]
            stop = extreme - self.cmipr.entry.stop_atr_buffer * atr_price
            chase = (latest.close - ignition.breakout_level) / max(atr_price, 1e-12)
        else:
            extreme = max(item.high for item in after)
            depth = (extreme - ignition.candle.close) / max(atr_price, 1e-12)
            close_position = (latest.high - latest.close) / max(latest.high - latest.low, 1e-12)
            confirmed = latest.close < latest.open and latest.close <= ignition.breakout_level and latest.close <= ema([item.close for item in candles], 9)[-1]
            stop = extreme + self.cmipr.entry.stop_atr_buffer * atr_price
            chase = (ignition.breakout_level - latest.close) / max(atr_price, 1e-12)
        normalized_volume = latest.volume * (interval_to_milliseconds("15m") / interval_to_milliseconds(self.cmipr.entry.pullback_timeframe)) / max(ignition.candle.volume, 1e-12)
        if not self.cmipr.entry.pullback_min_depth_atr <= depth <= self.cmipr.entry.pullback_max_depth_atr:
            return None
        if normalized_volume > self.cmipr.entry.pullback_max_volume_ratio:
            return None
        if close_position < self.cmipr.entry.confirmation_min_close_position or not confirmed:
            return None
        if chase > self.cmipr.entry.max_chase_distance_atr:
            self.reject_reasons["entry_chase"] += 1
            runtime.state = CmiprState.IDLE
            return None
        stop_atr = ignition.direction.value * (latest.close - stop) / max(atr_price, 1e-12)
        if not self.cmipr.entry.min_stop_atr <= stop_atr <= self.cmipr.entry.max_stop_atr:
            self.reject_reasons["pullback_stop_invalid"] += 1
            return None
        adjusted = CmiprIgnitionSnapshot(
            direction=ignition.direction,
            candle=latest,
            breakout_level=ignition.breakout_level,
            breakout_distance_atr=ignition.breakout_distance_atr,
            body_atr=ignition.body_atr,
            close_position=close_position,
            wick_ratio=ignition.wick_ratio,
            volume_ratio=ignition.volume_ratio,
            ranking_score=ignition.ranking_score,
            ranking_percentile=ignition.ranking_percentile,
            quality_score=ignition.quality_score,
            stop_price=stop,
            stop_loss_pct=abs(latest.close - stop) / max(latest.close, 1e-12),
            take_profit_pct=ignition.take_profit_pct,
        )
        candidate = self._candidate_from_ignition(symbol, adjusted, runtime.event_id, "first_pullback")
        runtime.state = CmiprState.INITIAL_ENTRY_READY
        runtime.consumed_event_ids.add(runtime.event_id)
        self.stats["pullback_confirmed"] += 1
        return candidate

    def _candidate_from_ignition(self, symbol: str, ignition: CmiprIgnitionSnapshot, event_id: str, mode: str) -> EntryCandidate:
        short_multiplier = self.cmipr.ignition.short_risk_multiplier if ignition.direction == Direction.SHORT else 1.0
        fraction = self.cmipr.entry.initial_risk_fraction * short_multiplier
        reason = (
            f"{CMIPR_REASON_TOKEN} event_id={event_id} entry_mode={mode} "
            f"trigger_tf={'15m' if mode == 'confirmation_open' else self.cmipr.entry.pullback_timeframe} "
            f"rank_pct={ignition.ranking_percentile:.4f} quality={ignition.quality_score:.4f} "
            f"breakout_atr={ignition.breakout_distance_atr:.4f} volume_ratio={ignition.volume_ratio:.4f} "
            f"initial_fraction={fraction:.4f} breakout_level={ignition.breakout_level:.10g}"
        )
        signal = Signal(
            direction=ignition.direction,
            confidence=0.55,
            reason=reason,
            stop_loss_pct=ignition.stop_loss_pct,
            take_profit_pct=ignition.take_profit_pct,
            risk_multiplier=fraction,
            max_holding_bars=max(1, int(self.cmipr.exit.max_holding_minutes / max(interval_to_milliseconds(self.config.trading.timeframe) / 60_000, 1))),
        )
        self.stats["entry_ready_count"] += 1
        return EntryCandidate(symbol, signal, ignition.candle, ignition.quality_score, abs(ignition.ranking_score), ignition.volume_ratio, "cmipr_ok")

    def _derivative_guard(self, symbol: str, decision_time: datetime, direction: Direction) -> str:
        if str(self.cmipr.research.model_variant).lower() != "derivatives_enhanced":
            return ""
        features = self.auxiliary_features.get(symbol)
        oi_change = oi_change_at(features, decision_time)
        if oi_change is None:
            return "derivatives_oi_missing"
        if not self.cmipr.ignition.min_oi_change_30m <= oi_change <= self.cmipr.ignition.max_oi_change_30m:
            return "derivatives_oi_guard"
        funding = funding_at(features, decision_time, default=float("nan"))
        if math.isnan(funding):
            return "derivatives_funding_missing"
        if direction == Direction.LONG and funding > self.cmipr.ignition.max_long_funding_rate:
            return "derivatives_funding_hot"
        if direction == Direction.SHORT and funding < self.cmipr.ignition.min_short_funding_rate:
            return "derivatives_funding_crowded_short"
        if self.cmipr.ignition.require_taker_flow and not features.get("taker"):
            return "derivatives_taker_missing"
        if self.cmipr.ignition.require_basis and not features.get("basis"):
            return "derivatives_basis_missing"
        return ""

    def _closed(self, timeframe: str, symbol: str, decision_time: datetime, limit: int) -> list[Candle]:
        candles = self.candles.get(timeframe, {}).get(symbol, [])
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        available_before = decision_time - timedelta(milliseconds=interval_to_milliseconds(timeframe))
        end = bisect.bisect_right(timestamps, available_before)
        return candles[max(0, end - limit):end]


def _configured_full_cost_pct(config: Any) -> float:
    risk = config.risk
    return (
        2.0 * max(0.0, float(risk.taker_fee_rate))
        + (max(0.0, float(risk.market_slippage_bps)) + max(0.0, float(risk.stop_slippage_bps))) / 10_000.0
    )


def _period_return(candles: list[Candle], bars: int) -> float:
    if len(candles) <= bars or candles[-1 - bars].close <= 0:
        return 0.0
    return candles[-1].close / candles[-1 - bars].close - 1.0


def _volume_ratio(candles: list[Candle], period: int) -> float:
    if len(candles) < period + 1:
        return 0.0
    average = sum(item.volume for item in candles[-period - 1:-1]) / period
    return candles[-1].volume / max(average, 1e-12)


def _failed_breakout_count(candles: list[Candle], lookback: int) -> int:
    failures = 0
    for index in range(max(lookback, 2), len(candles) - 1):
        prior = candles[index - lookback:index]
        high = max(item.high for item in prior)
        low = min(item.low for item in prior)
        if candles[index].high > high and candles[index + 1].close < high:
            failures += 1
        if candles[index].low < low and candles[index + 1].close > low:
            failures += 1
    return failures


def _bar_distance(start: datetime | None, end: datetime | None, timeframe: str) -> int:
    if start is None or end is None:
        return 10**9
    seconds = interval_to_milliseconds(timeframe) / 1000.0
    return max(0, int((end - start).total_seconds() // seconds))


def _files_for_symbols(root: Path | None, symbols: tuple[str, ...], pattern: str) -> list[Path]:
    if root is None or not root.exists():
        return []
    output: list[Path] = []
    for symbol in symbols:
        output.extend(sorted(root.glob(f"{symbol}{pattern[1:]}")))
    return output


def _csv_time_range(paths: list[Path]) -> tuple[datetime | None, datetime | None]:
    minimum: datetime | None = None
    maximum: datetime | None = None
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw = row.get("timestamp") or row.get("time") or row.get("fundingTime")
                    timestamp = _parse_timestamp(raw)
                    if timestamp is None:
                        continue
                    minimum = timestamp if minimum is None else min(minimum, timestamp)
                    maximum = timestamp if maximum is None else max(maximum, timestamp)
        except (OSError, csv.Error):
            continue
    return minimum, maximum


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value) / 1000.0)
        except (TypeError, ValueError, OSError):
            return None


def _range_covers(time_range: tuple[datetime | None, datetime | None], start: datetime | None, end: datetime | None) -> bool:
    minimum, maximum = time_range
    if minimum is None or maximum is None:
        return False
    return (start is None or minimum <= start) and (end is None or maximum >= end)


def _format_range(time_range: tuple[datetime | None, datetime | None]) -> tuple[str | None, str | None]:
    return tuple(value.isoformat() if value is not None else None for value in time_range)  # type: ignore[return-value]
