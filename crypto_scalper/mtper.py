from __future__ import annotations

import bisect
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .data import interval_to_milliseconds
from .indicators import atr, ema, macd, rsi
from .live_trader import EntryCandidate
from .models import Candle, Direction, Signal


MTPER_REASON_TOKEN = "multi_timeframe_pre_cross_exhaustion_reversal"
MTPER_FEATURE_VERSION = "mtper_features_v1"
MTPER_EVENT_VERSION = "mtper_event_v1"


class MtperState(str, Enum):
    IDLE = "IDLE"
    LONG_PRE_CROSS_SETUP = "LONG_PRE_CROSS_SETUP"
    SHORT_PRE_CROSS_SETUP = "SHORT_PRE_CROSS_SETUP"
    HIGHER_TIMEFRAME_CONFIRMING = "HIGHER_TIMEFRAME_CONFIRMING"
    LOWER_TIMEFRAME_CONFIRMING = "LOWER_TIMEFRAME_CONFIRMING"
    INITIAL_ENTRY_READY = "INITIAL_ENTRY_READY"
    ORDER_PENDING = "ORDER_PENDING"
    PARTIAL_FILL = "PARTIAL_FILL"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    INITIAL_POSITION = "INITIAL_POSITION"
    SECOND_ENTRY_OBSERVATION = "SECOND_ENTRY_OBSERVATION"
    SECOND_ENTRY_READY = "SECOND_ENTRY_READY"
    FULL_CAMPAIGN_POSITION = "FULL_CAMPAIGN_POSITION"
    MEAN_TARGET_1 = "MEAN_TARGET_1"
    MEAN_TARGET_2 = "MEAN_TARGET_2"
    TREND_CONVERSION_PENDING = "TREND_CONVERSION_PENDING"
    TREND_CONVERSION_RUNNER = "TREND_CONVERSION_RUNNER"
    STRUCTURAL_FAIL_FAST = "STRUCTURAL_FAIL_FAST"
    HARD_STOP = "HARD_STOP"
    EXITING = "EXITING"
    CANCEL_PENDING = "CANCEL_PENDING"
    RECOVERY_AFTER_RESTART = "RECOVERY_AFTER_RESTART"
    INVALIDATED = "INVALIDATED"
    COOLDOWN = "COOLDOWN"


class MtperRegime(str, Enum):
    MEAN_REVERSION_LONG_ALLOWED = "MEAN_REVERSION_LONG_ALLOWED"
    MEAN_REVERSION_SHORT_ALLOWED = "MEAN_REVERSION_SHORT_ALLOWED"
    TREND_EXPANSION_LONG_BLOCK = "TREND_EXPANSION_LONG_BLOCK"
    TREND_EXPANSION_SHORT_BLOCK = "TREND_EXPANSION_SHORT_BLOCK"
    CHAOS_NO_TRADE = "CHAOS_NO_TRADE"
    NEUTRAL_WAIT = "NEUTRAL_WAIT"


@dataclass(frozen=True)
class MtperPreCrossSnapshot:
    direction: Direction
    candle: Candle
    ema_fast: float
    ema_slow: float
    ema_gap_atr: float
    ema_gap_abs_atr: float
    gap_changes: tuple[float, ...]
    gap_contracting_bars: int
    fast_ema_slope_atr: float
    slow_ema_slope_atr: float
    macd_histogram: float
    macd_histogram_slopes: tuple[float, ...]
    formal_cross: bool
    extreme_score: float
    extreme_components: dict[str, float]
    atr_value: float
    hard_stop: float


@dataclass(frozen=True)
class MtperConfirmationSnapshot:
    candle: Candle
    confirmation_id: str
    trigger_type: str
    close_position: float
    wick_ratio: float
    volume_ratio: float
    atr_value: float
    structural_level: float
    score: float


@dataclass
class MtperSymbolRuntime:
    state: MtperState = MtperState.IDLE
    setup_id: str | None = None
    campaign_id: str | None = None
    confirmation_id: str | None = None
    direction: Direction = Direction.FLAT
    setup: MtperPreCrossSnapshot | None = None
    setup_time: datetime | None = None
    expire_time: datetime | None = None
    cooldown_until: datetime | None = None
    last_processed_4h: datetime | None = None
    last_processed_15m: datetime | None = None
    consumed_confirmation_ids: set[str] = field(default_factory=set)


@dataclass
class MtperOrderLifecycle:
    symbol: str
    state: MtperState = MtperState.IDLE
    requested_quantity: float = 0.0
    filled_quantity: float = 0.0
    protection_attempts: int = 0
    last_error: str = ""

    def submit_entry(self, requested_quantity: float = 1.0) -> None:
        if requested_quantity <= 0:
            raise ValueError("requested_quantity must be positive")
        self.requested_quantity = requested_quantity
        self.filled_quantity = 0.0
        self.state = MtperState.ORDER_PENDING

    def record_fill(self, filled_quantity: float) -> None:
        if self.state not in (MtperState.ORDER_PENDING, MtperState.PARTIAL_FILL):
            raise RuntimeError(f"fill not allowed from {self.state.value}")
        self.filled_quantity += max(0.0, filled_quantity)
        if self.filled_quantity <= 0:
            return
        self.state = (
            MtperState.PARTIAL_FILL
            if self.filled_quantity + 1e-12 < self.requested_quantity
            else MtperState.PROTECTION_PENDING
        )

    def protection_result(self, success: bool, error: str = "") -> str:
        self.protection_attempts += 1
        if success:
            self.last_error = ""
            self.state = MtperState.PROTECTED
            return "protected"
        self.last_error = error or "protective_stop_failed"
        self.state = MtperState.EXITING
        return "emergency_reduce_or_close"

    def request_cancel(self) -> None:
        self.state = MtperState.CANCEL_PENDING

    def recover_after_restart(self, open_quantity: float, protective_stop_present: bool) -> str:
        self.state = MtperState.RECOVERY_AFTER_RESTART
        self.filled_quantity = max(0.0, open_quantity)
        if self.filled_quantity <= 0:
            self.state = MtperState.IDLE
            return "no_position"
        if protective_stop_present:
            self.state = MtperState.PROTECTED
            return "recovered_protected"
        self.state = MtperState.PROTECTION_PENDING
        return "replace_protection_or_emergency_flatten"


class MtperEngine:
    """Point-in-time multi-timeframe pre-cross reversal signal engine."""

    REQUIRED_TIMEFRAMES = ("15m", "30m", "1h", "2h", "4h")

    def __init__(
        self,
        config: Any,
        candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    ) -> None:
        self.config = config
        self.mtper = config.mtper
        self.candles = candles_by_timeframe
        self.timestamps = {
            timeframe: {
                symbol: [candle.timestamp for candle in candles]
                for symbol, candles in by_symbol.items()
            }
            for timeframe, by_symbol in candles_by_timeframe.items()
        }
        configured = tuple(self.mtper.enabled_symbols or config.trading.entry_symbols or config.trading.symbols)
        self.symbols = tuple(
            symbol
            for symbol in configured
            if all(symbol in self.candles.get(timeframe, {}) for timeframe in self.REQUIRED_TIMEFRAMES)
        )
        self.runtime = {symbol: MtperSymbolRuntime() for symbol in self.symbols}
        self.order_lifecycle = {symbol: MtperOrderLifecycle(symbol) for symbol in self.symbols}
        self.stats: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self.regime_counts: Counter[str] = Counter()
        self.trigger_counts: Counter[str] = Counter()
        self.state_transitions: list[dict[str, Any]] = []
        self.setup_rows: list[dict[str, Any]] = []
        self.pre_cross_candidate_rows: list[dict[str, Any]] = []
        self.strategy_config_hash = mtper_strategy_config_hash(config)
        self.cache_namespace = f"mtper:{self.strategy_config_hash}"
        self._last_market_scan_bar: datetime | None = None

    def scan(
        self,
        decision_time: datetime,
        occupied_symbols: set[str],
        allowed_symbols: set[str] | None = None,
    ) -> list[EntryCandidate]:
        market_15m = self._closed("15m", self.mtper.regime.btc_symbol, decision_time, 1)
        if not market_15m:
            return []
        latest_market_bar = market_15m[-1].timestamp
        if latest_market_bar == self._last_market_scan_bar:
            return []
        self._last_market_scan_bar = latest_market_bar
        allowed = allowed_symbols if allowed_symbols is not None else set(self.symbols)
        candidates: list[EntryCandidate] = []
        for symbol in self.symbols:
            runtime = self.runtime[symbol]
            if runtime.cooldown_until is not None and decision_time < runtime.cooldown_until:
                continue
            if runtime.state == MtperState.COOLDOWN:
                self._transition(symbol, MtperState.IDLE, decision_time, "cooldown_complete")
            if symbol in occupied_symbols or symbol not in allowed:
                continue
            candidate = self._scan_symbol(symbol, decision_time)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-item.rank_score, item.symbol))
        return candidates

    def mark_order_pending(self, symbol: str) -> None:
        lifecycle = self.order_lifecycle[symbol]
        lifecycle.submit_entry()
        self.runtime[symbol].state = MtperState.ORDER_PENDING
        self.stats["order_pending_count"] += 1

    def mark_filled(self, symbol: str, quantity: float, requested_quantity: float | None = None) -> None:
        lifecycle = self.order_lifecycle[symbol]
        if requested_quantity is not None and requested_quantity > 0:
            lifecycle.requested_quantity = requested_quantity
        lifecycle.record_fill(quantity)
        self.runtime[symbol].state = lifecycle.state
        if lifecycle.state == MtperState.PARTIAL_FILL:
            self.stats["partial_fill_count"] += 1

    def mark_protected(self, symbol: str) -> None:
        lifecycle = self.order_lifecycle[symbol]
        lifecycle.protection_result(True)
        self.runtime[symbol].state = MtperState.INITIAL_POSITION
        self.stats["protected_count"] += 1

    def mark_protection_failed(self, symbol: str, error: str = "") -> str:
        action = self.order_lifecycle[symbol].protection_result(False, error)
        self.runtime[symbol].state = MtperState.EXITING
        self.stats["protection_failure_count"] += 1
        return action

    def mark_cancelled(self, symbol: str, timestamp: datetime, reason: str) -> None:
        lifecycle = self.order_lifecycle[symbol]
        lifecycle.request_cancel()
        self.stats["cancel_pending_count"] += 1
        self.reject_reasons[f"execution_{reason}"] += 1
        self.mark_closed(symbol, timestamp, f"entry_cancelled:{reason}")

    def mark_closed(self, symbol: str, timestamp: datetime, reason: str = "closed") -> None:
        runtime = self.runtime[symbol]
        cooldown_hours = max(0, int(self.mtper.risk_control.symbol_cooldown_hours))
        runtime.cooldown_until = timestamp + timedelta(hours=cooldown_hours)
        runtime.setup = None
        runtime.setup_id = None
        runtime.confirmation_id = None
        runtime.direction = Direction.FLAT
        self.order_lifecycle[symbol] = MtperOrderLifecycle(symbol)
        self._transition(symbol, MtperState.COOLDOWN, timestamp, reason)

    def report(self) -> dict[str, Any]:
        return {
            "strategy_name": MTPER_REASON_TOKEN,
            "feature_version": MTPER_FEATURE_VERSION,
            "event_definition_version": MTPER_EVENT_VERSION,
            "strategy_config_hash": self.strategy_config_hash,
            "cache_namespace": self.cache_namespace,
            "stage_variant": self.mtper.research.stage_variant,
            "model_variant": self.mtper.research.model_variant,
            "symbols": list(self.symbols),
            "stats": dict(sorted(self.stats.items())),
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "regime_counts": dict(sorted(self.regime_counts.items())),
            "trigger_counts": dict(sorted(self.trigger_counts.items())),
            "setup_count": len(self.setup_rows),
            "setups": self.setup_rows,
            "pre_cross_candidate_count": len(self.pre_cross_candidate_rows),
            "pre_cross_candidates": self.pre_cross_candidate_rows,
            "state_transitions": self.state_transitions,
            "historical_test_policy": {
                "historical_test_start": self.mtper.research.historical_test_start,
                "historical_test_end": self.mtper.research.historical_test_end,
                "is_untouched_final_holdout": False,
                "final_acceptance_source": self.mtper.research.final_acceptance_source,
            },
        }

    def _scan_symbol(self, symbol: str, decision_time: datetime) -> EntryCandidate | None:
        runtime = self.runtime[symbol]
        stage = _stage_variant(self.mtper.research.stage_variant)
        c4h = self._closed("4h", symbol, decision_time, 140)
        if len(c4h) < max(70, int(self.mtper.pre_cross.ema_slow_period) + 10):
            self.reject_reasons["insufficient_4h_history"] += 1
            return None
        latest_4h = c4h[-1].timestamp
        if runtime.setup is not None and runtime.expire_time is not None and decision_time >= runtime.expire_time:
            self._invalidate(symbol, decision_time, "setup_expired")
            runtime = self.runtime[symbol]
        if runtime.setup is not None and self._setup_hard_invalidated(runtime.setup.direction, c4h):
            self._invalidate(symbol, decision_time, "setup_hard_invalidated_before_entry")
            runtime = self.runtime[symbol]

        if runtime.setup is None and runtime.last_processed_4h != latest_4h:
            runtime.last_processed_4h = latest_4h
            setup, reject = self._four_hour_setup(symbol, decision_time, formal_only=stage == "formal_cross")
            if setup is None:
                if reject:
                    self.reject_reasons[reject] += 1
                return None
            setup_id = f"{symbol}:{setup.direction.name}:{setup.candle.timestamp.isoformat()}:{stage}"
            runtime.setup_id = setup_id
            runtime.campaign_id = f"campaign:{setup_id}"
            runtime.direction = setup.direction
            runtime.setup = setup
            runtime.setup_time = decision_time
            runtime.expire_time = decision_time + timedelta(hours=4 * max(1, int(self.mtper.pre_cross.setup_expiry_4h_bars)))
            setup_state = MtperState.LONG_PRE_CROSS_SETUP if setup.direction == Direction.LONG else MtperState.SHORT_PRE_CROSS_SETUP
            self._transition(symbol, setup_state, decision_time, "four_hour_setup_created")
            self.stats["setup_count"] += 1
            self.setup_rows.append(self._setup_row(symbol, setup_id, decision_time, setup))

        setup = runtime.setup
        if setup is None:
            return None
        regime = self._regime(symbol, setup.direction, decision_time)
        self.regime_counts[regime.value] += 1
        allowed_regime = (
            MtperRegime.MEAN_REVERSION_LONG_ALLOWED
            if setup.direction == Direction.LONG
            else MtperRegime.MEAN_REVERSION_SHORT_ALLOWED
        )
        if regime != allowed_regime:
            self.reject_reasons[f"regime_{regime.value.lower()}"] += 1
            return None
        if stage == "formal_cross":
            confirmation = self._direct_confirmation(setup, "formal_cross")
        elif stage == "pre_cross":
            confirmation = self._direct_confirmation(setup, "pre_cross")
        else:
            exhaustion_ok, exhaustion_checks = self._two_hour_exhaustion(symbol, setup.direction, decision_time)
            permission_ok, permission_checks = self._one_hour_permission(symbol, setup.direction, decision_time)
            self._annotate_setup(runtime.setup_id, two_hour_checks=exhaustion_checks, one_hour_checks=permission_checks)
            if not exhaustion_ok:
                self._transition(symbol, MtperState.HIGHER_TIMEFRAME_CONFIRMING, decision_time, "waiting_2h_exhaustion")
                self.reject_reasons["two_hour_exhaustion_not_confirmed"] += 1
                return None
            if not permission_ok:
                self._transition(symbol, MtperState.HIGHER_TIMEFRAME_CONFIRMING, decision_time, "waiting_1h_permission")
                self.reject_reasons["one_hour_permission_not_confirmed"] += 1
                return None
            if stage == "pre_cross_htf":
                confirmation = self._higher_timeframe_confirmation(symbol, setup.direction, decision_time)
            else:
                self._transition(symbol, MtperState.LOWER_TIMEFRAME_CONFIRMING, decision_time, "higher_timeframes_confirmed")
                confirmation = self._fifteen_minute_trigger(symbol, setup.direction, decision_time)
                if confirmation is None:
                    self.reject_reasons["fifteen_minute_trigger_not_confirmed"] += 1
                    return None
        if confirmation.confirmation_id in runtime.consumed_confirmation_ids:
            return None
        candidate = self._candidate(symbol, setup, confirmation, regime, decision_time)
        if candidate is None:
            return None
        runtime.confirmation_id = confirmation.confirmation_id
        runtime.consumed_confirmation_ids.add(confirmation.confirmation_id)
        self._transition(symbol, MtperState.INITIAL_ENTRY_READY, decision_time, confirmation.trigger_type)
        self.stats["entry_ready_count"] += 1
        self.trigger_counts[confirmation.trigger_type] += 1
        self._annotate_setup(
            runtime.setup_id,
            confirmation_id=confirmation.confirmation_id,
            confirmation_time=decision_time.isoformat(),
            trigger_type=confirmation.trigger_type,
            confirmation_score=confirmation.score,
        )
        return candidate

    def _four_hour_setup(
        self,
        symbol: str,
        decision_time: datetime,
        formal_only: bool,
    ) -> tuple[MtperPreCrossSnapshot | None, str]:
        candles = self._closed("4h", symbol, decision_time, 160)
        cfg = self.mtper.pre_cross
        closes = [candle.close for candle in candles]
        fast = ema(closes, int(cfg.ema_fast_period))
        slow = ema(closes, int(cfg.ema_slow_period))
        atr_values = atr(candles, int(cfg.atr_period))
        _, _, hist = macd(closes)
        atr_value = max(atr_values[-1], 1e-12)
        gaps = [(left - right) / max(a, 1e-12) for left, right, a in zip(fast, slow, atr_values)]
        gap = gaps[-1]
        lookback = max(1, int(cfg.ema_slope_lookback_bars))
        fast_slope = (fast[-1] - fast[-1 - lookback]) / atr_value
        slow_slope = (slow[-1] - slow[-1 - lookback]) / atr_value
        long_cross = fast[-2] <= slow[-2] and fast[-1] > slow[-1]
        short_cross = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
        directions: list[tuple[Direction, bool]] = []
        if bool(self.mtper.allow_long):
            directions.append((Direction.LONG, long_cross))
        if bool(self.mtper.allow_short):
            directions.append((Direction.SHORT, short_cross))
        best: MtperPreCrossSnapshot | None = None
        best_score = -1.0
        for direction, formal_cross in directions:
            side = direction.value
            if formal_only:
                if not formal_cross:
                    self.reject_reasons[f"{direction.name.lower()}_formal_cross_absent"] += 1
                    continue
            else:
                if side * gap >= 0:
                    self.reject_reasons[f"{direction.name.lower()}_already_crossed"] += 1
                    continue
                if not float(cfg.min_gap_abs_atr) <= abs(gap) <= float(cfg.max_gap_abs_atr):
                    self.reject_reasons[f"{direction.name.lower()}_gap_outside_pre_cross_range"] += 1
                    continue
                contract_bars = max(1, int(cfg.gap_contracting_bars))
                absolute = [abs(value) for value in gaps[-contract_bars - 1:]]
                if len(absolute) < contract_bars + 1 or any(absolute[index] >= absolute[index - 1] for index in range(1, len(absolute))):
                    self.reject_reasons[f"{direction.name.lower()}_gap_not_contracting"] += 1
                    continue
                if side * fast_slope < float(cfg.min_fast_slope_atr):
                    self.reject_reasons[f"{direction.name.lower()}_fast_ema_slope_not_turning"] += 1
                    continue
                improve_bars = max(1, int(cfg.macd_improvement_bars))
                if not _directionally_improving(hist, side, improve_bars):
                    self.reject_reasons[f"{direction.name.lower()}_macd_not_converging"] += 1
                    continue
            extreme_score, components = self._extreme_score(candles, fast, slow, atr_values, direction)
            self.pre_cross_candidate_rows.append(
                {
                    "symbol": symbol,
                    "side": direction.name,
                    "four_hour_candle_time": candles[-1].timestamp.isoformat(),
                    "formal_cross": formal_cross,
                    "ema_pair": [int(cfg.ema_fast_period), int(cfg.ema_slow_period)],
                    "ema_gap_atr": gap,
                    "ema_gap_abs_atr": abs(gap),
                    "gap_contracting_bars": _contracting_tail([abs(value) for value in gaps]),
                    "fast_ema_slope_atr": fast_slope,
                    "slow_ema_slope_atr": slow_slope,
                    "macd_histogram": hist[-1],
                    "extreme_score": extreme_score,
                    "extreme_components": components,
                    "qualified": extreme_score >= float(self.mtper.extreme.score_threshold),
                }
            )
            if extreme_score < float(self.mtper.extreme.score_threshold):
                self.reject_reasons[f"{direction.name.lower()}_extreme_score_below_threshold"] += 1
                continue
            hard_stop = self._hard_stop(candles, direction, atr_value)
            gap_changes = tuple(gaps[index] - gaps[index - 1] for index in range(max(1, len(gaps) - 3), len(gaps)))
            hist_slopes = tuple(hist[index] - hist[index - 1] for index in range(max(1, len(hist) - 3), len(hist)))
            score = extreme_score + max(0.0, 1.0 - abs(gap) / max(float(cfg.max_gap_abs_atr), 1e-12))
            if score <= best_score:
                continue
            best_score = score
            best = MtperPreCrossSnapshot(
                direction=direction,
                candle=candles[-1],
                ema_fast=fast[-1],
                ema_slow=slow[-1],
                ema_gap_atr=gap,
                ema_gap_abs_atr=abs(gap),
                gap_changes=gap_changes,
                gap_contracting_bars=max(0, _contracting_tail([abs(value) for value in gaps])),
                fast_ema_slope_atr=fast_slope,
                slow_ema_slope_atr=slow_slope,
                macd_histogram=hist[-1],
                macd_histogram_slopes=hist_slopes,
                formal_cross=formal_cross,
                extreme_score=extreme_score,
                extreme_components=components,
                atr_value=atr_value,
                hard_stop=hard_stop,
            )
        return (best, "") if best is not None else (None, "four_hour_setup_not_qualified")

    def _extreme_score(
        self,
        candles: list[Candle],
        fast: list[float],
        slow: list[float],
        atr_values: list[float],
        direction: Direction,
    ) -> tuple[float, dict[str, float]]:
        cfg = self.mtper.extreme
        side = direction.value
        closes = [candle.close for candle in candles]
        current = candles[-1]
        atr_value = max(atr_values[-1], 1e-12)
        distance_fast = side * (fast[-1] - current.close) / atr_value
        distance_slow = side * (slow[-1] - current.close) / atr_value
        z_lookback = max(10, int(cfg.zscore_lookback_4h))
        window = closes[-z_lookback:]
        mean = statistics.fmean(window)
        stdev = statistics.pstdev(window) if len(window) > 1 else 0.0
        raw_z = 0.0 if stdev <= 0 else (current.close - mean) / stdev
        directional_z = -side * raw_z
        rsi_values = rsi(closes, int(cfg.rsi_period))
        current_rsi = rsi_values[-1]
        rsi_window = rsi_values[-max(10, int(cfg.rsi_percentile_lookback)):]
        rsi_pct = _percentile_rank(current_rsi, rsi_window)
        if direction == Direction.LONG:
            rsi_level = _clamp((float(cfg.long_rsi_ceiling) - current_rsi) / 20.0 + 0.5)
            rsi_percentile_score = _clamp((float(cfg.long_rsi_percentile_max) - rsi_pct) / max(float(cfg.long_rsi_percentile_max), 1e-12) + 0.25)
        else:
            rsi_level = _clamp((current_rsi - float(cfg.short_rsi_floor)) / 20.0 + 0.5)
            rsi_percentile_score = _clamp((rsi_pct - float(cfg.short_rsi_percentile_min)) / max(1.0 - float(cfg.short_rsi_percentile_min), 1e-12) + 0.25)
        range_scores = []
        for raw_lookback in cfg.range_lookbacks:
            lookback = max(2, int(raw_lookback))
            sample = candles[-lookback:]
            low = min(item.low for item in sample)
            high = max(item.high for item in sample)
            position = (current.close - low) / max(high - low, 1e-12)
            range_scores.append(1.0 - position if direction == Direction.LONG else position)
        streak = 0
        for index in range(len(candles) - 1, 0, -1):
            move = candles[index].close - candles[index - 1].close
            if side * move < 0:
                streak += 1
            else:
                break
        atr_window = atr_values[-max(20, int(cfg.atr_percentile_lookback)):]
        atr_pct = _percentile_rank(atr_value, atr_window)
        candle_range = max(current.high - current.low, 1e-12)
        wick = (min(current.open, current.close) - current.low) if direction == Direction.LONG else (current.high - max(current.open, current.close))
        wick_ratio = wick / candle_range
        volume_window = [item.volume for item in candles[-21:-1]]
        volume_ratio = current.volume / max(statistics.fmean(volume_window), 1e-12) if volume_window else 0.0
        components = {
            "ema_fast_distance": _clamp(distance_fast / max(float(cfg.ema21_distance_atr), 1e-12)),
            "ema_slow_distance": _clamp(distance_slow / max(float(cfg.ema55_distance_atr), 1e-12)),
            "price_zscore": _clamp(directional_z / max(float(cfg.price_zscore_threshold), 1e-12)),
            "rsi_level": rsi_level,
            "rsi_percentile": rsi_percentile_score,
            "range_position": statistics.fmean(range_scores) if range_scores else 0.0,
            "directional_streak": _clamp(streak / max(1, int(cfg.streak_bars))),
            "atr_state": 0.0 if atr_pct > float(cfg.max_atr_percentile) else _clamp(atr_pct / 0.75),
            "wick_exhaustion": _clamp(wick_ratio / max(float(cfg.wick_exhaustion_min_ratio), 1e-12)),
            "volume_climax": _clamp(volume_ratio / max(float(cfg.volume_climax_ratio), 1e-12)),
        }
        weights = {
            "ema_fast_distance": 0.12,
            "ema_slow_distance": 0.10,
            "price_zscore": 0.12,
            "rsi_level": 0.12,
            "rsi_percentile": 0.10,
            "range_position": 0.14,
            "directional_streak": 0.08,
            "atr_state": 0.06,
            "wick_exhaustion": 0.08,
            "volume_climax": 0.08,
        }
        return sum(components[key] * weight for key, weight in weights.items()), components

    def _regime(self, symbol: str, direction: Direction, decision_time: datetime) -> MtperRegime:
        cfg = self.mtper.regime
        btc = self._closed("1h", cfg.btc_symbol, decision_time, 30)
        eth = self._closed("1h", cfg.eth_symbol, decision_time, 30)
        if len(btc) < 20 or len(eth) < 20:
            return MtperRegime.NEUTRAL_WAIT
        btc_return = btc[-1].close / max(btc[-2].close, 1e-12) - 1.0
        eth_return = eth[-1].close / max(eth[-2].close, 1e-12) - 1.0
        btc_atr = atr(btc, 14)
        atr_pct = _percentile_rank(btc_atr[-1], btc_atr[-100:])
        if abs(btc_return) >= float(cfg.btc_shock_1h_pct) or abs(eth_return) >= float(cfg.eth_shock_1h_pct) or atr_pct >= float(cfg.chaos_atr_percentile):
            return MtperRegime.CHAOS_NO_TRADE
        c2h = self._closed("2h", symbol, decision_time, 80)
        c1h = self._closed("1h", symbol, decision_time, 100)
        if len(c2h) < 30 or len(c1h) < 30:
            return MtperRegime.NEUTRAL_WAIT
        continuation = self._continuation_checks(c2h, direction) + self._continuation_checks(c1h, direction)
        breadth_positive = 0
        breadth_total = 0
        for candidate_symbol in self.symbols:
            rows = self._closed("1h", candidate_symbol, decision_time, 2)
            if len(rows) < 2:
                continue
            breadth_total += 1
            breadth_positive += rows[-1].close > rows[-2].close
        breadth = breadth_positive / breadth_total if breadth_total else 0.5
        adverse_breadth = breadth if direction == Direction.SHORT else 1.0 - breadth
        if continuation >= int(cfg.trend_expansion_min_checks) or adverse_breadth > 1.0 - float(cfg.min_breadth_not_adverse):
            return MtperRegime.TREND_EXPANSION_LONG_BLOCK if direction == Direction.LONG else MtperRegime.TREND_EXPANSION_SHORT_BLOCK
        return MtperRegime.MEAN_REVERSION_LONG_ALLOWED if direction == Direction.LONG else MtperRegime.MEAN_REVERSION_SHORT_ALLOWED

    def _continuation_checks(self, candles: list[Candle], reversal_direction: Direction) -> int:
        closes = [item.close for item in candles]
        _, _, hist = macd(closes)
        ema21 = ema(closes, 21)
        side = reversal_direction.value
        adverse = -side
        checks = 0
        checks += adverse * (closes[-1] - closes[-2]) > 0
        checks += adverse * (hist[-1] - hist[-2]) > 0
        checks += adverse * (ema21[-1] - ema21[-3]) > 0
        checks += (candles[-1].low < min(item.low for item in candles[-4:-1])) if reversal_direction == Direction.LONG else (candles[-1].high > max(item.high for item in candles[-4:-1]))
        return checks

    def _two_hour_exhaustion(self, symbol: str, direction: Direction, decision_time: datetime) -> tuple[bool, int]:
        cfg = self.mtper.higher_timeframe
        candles = self._closed("2h", symbol, decision_time, 100)
        checks = self._exhaustion_checks(
            candles,
            direction,
            max(1, int(cfg.no_new_extreme_bars_2h)),
            max(1, int(cfg.macd_improvement_bars_2h)),
            float(cfg.max_adverse_ema21_slope_atr_2h),
            float(cfg.volume_exhaustion_ratio),
        )
        return checks >= int(cfg.min_2h_exhaustion_checks), checks

    def _one_hour_permission(self, symbol: str, direction: Direction, decision_time: datetime) -> tuple[bool, int]:
        cfg = self.mtper.higher_timeframe
        candles = self._closed("1h", symbol, decision_time, 120)
        checks = self._exhaustion_checks(
            candles,
            direction,
            max(1, int(cfg.no_new_extreme_bars_1h)),
            max(1, int(cfg.macd_improvement_bars_1h)),
            float(cfg.max_adverse_ema21_slope_atr_1h),
            float(cfg.volume_exhaustion_ratio),
        )
        return checks >= int(cfg.min_1h_permission_checks), checks

    @staticmethod
    def _exhaustion_checks(
        candles: list[Candle],
        direction: Direction,
        no_extreme_bars: int,
        improvement_bars: int,
        max_adverse_slope: float,
        volume_ratio: float,
    ) -> int:
        if len(candles) < max(30, no_extreme_bars + 5, improvement_bars + 5):
            return 0
        side = direction.value
        closes = [item.close for item in candles]
        atr_value = max(atr(candles, 14)[-1], 1e-12)
        ema21 = ema(closes, 21)
        _, _, hist = macd(closes)
        rsi_values = rsi(closes, 14)
        previous = candles[-no_extreme_bars - 1:-1]
        no_new = candles[-1].low >= min(item.low for item in previous) if direction == Direction.LONG else candles[-1].high <= max(item.high for item in previous)
        checks = int(no_new)
        checks += int(_directionally_improving(hist, side, improvement_bars))
        checks += int(side * (rsi_values[-1] - rsi_values[-2]) > 0)
        checks += int(side * (ema21[-1] - ema21[-3]) / atr_value >= -max_adverse_slope)
        recent_body = statistics.fmean(abs(item.close - item.open) for item in candles[-2:])
        prior_body = statistics.fmean(abs(item.close - item.open) for item in candles[-5:-2])
        checks += int(recent_body <= prior_body)
        recent_volume = statistics.fmean(item.volume for item in candles[-2:])
        prior_volume = statistics.fmean(item.volume for item in candles[-8:-2])
        checks += int(recent_volume <= prior_volume * volume_ratio)
        checks += int(side * (closes[-1] - closes[-2]) > 0 or side * (closes[-1] - ema21[-1]) > side * (closes[-2] - ema21[-2]))
        return checks

    def _fifteen_minute_trigger(
        self,
        symbol: str,
        direction: Direction,
        decision_time: datetime,
    ) -> MtperConfirmationSnapshot | None:
        cfg = self.mtper.entry
        candles = self._closed("15m", symbol, decision_time, 120)
        if len(candles) < 35:
            return None
        current = candles[-1]
        prior = candles[-2]
        closes = [item.close for item in candles]
        ema_reclaim = ema(closes, max(2, int(cfg.reclaim_ema_period)))[-1]
        _, _, hist = macd(closes)
        rsi_values = rsi(closes, 14)
        atr_value = max(atr(candles, 14)[-1], 1e-12)
        side = direction.value
        candle_range = max(current.high - current.low, 1e-12)
        close_position = (current.close - current.low) / candle_range
        directional_close_position = close_position if direction == Direction.LONG else 1.0 - close_position
        adverse_wick = current.high - max(current.open, current.close) if direction == Direction.LONG else min(current.open, current.close) - current.low
        wick_ratio = adverse_wick / candle_range
        average_volume = statistics.fmean(item.volume for item in candles[-21:-1])
        volume_ratio = current.volume / max(average_volume, 1e-12)
        quality_ok = (
            directional_close_position >= float(cfg.confirmation_min_close_position)
            and wick_ratio <= float(cfg.confirmation_max_wick_ratio)
            and volume_ratio >= float(cfg.confirmation_min_volume_ratio)
            and side * (hist[-1] - hist[-2]) > 0
            and side * (rsi_values[-1] - rsi_values[-2]) > 0
        )
        if not quality_ok:
            return None
        lookback = max(2, int(cfg.higher_low_lookback_15m))
        prior_extreme = min(item.low for item in candles[-lookback - 1:-1]) if direction == Direction.LONG else max(item.high for item in candles[-lookback - 1:-1])
        higher_low = (
            current.low > prior_extreme and current.close > ema_reclaim and current.close > current.open
            if direction == Direction.LONG
            else current.high < prior_extreme and current.close < ema_reclaim and current.close < current.open
        )
        sweep_window = candles[-lookback - 2:-2]
        sweep_level = min(item.low for item in sweep_window) if direction == Direction.LONG else max(item.high for item in sweep_window)
        false_breakdown = (
            prior.low < sweep_level and prior.close > sweep_level and current.low >= prior.low and current.close > sweep_level
            if direction == Direction.LONG
            else prior.high > sweep_level and prior.close < sweep_level and current.high <= prior.high and current.close < sweep_level
        )
        breakout_lookback = max(2, int(cfg.structure_breakout_lookback_15m))
        structure_level = max(item.high for item in candles[-breakout_lookback - 1:-1]) if direction == Direction.LONG else min(item.low for item in candles[-breakout_lookback - 1:-1])
        structure_breakout = side * (current.close - structure_level) > 0 and volume_ratio >= 1.0
        modes = {
            "higher_low_reclaim": higher_low,
            "false_breakdown_reclaim": false_breakdown,
            "local_structure_breakout": structure_breakout,
        }
        configured_mode = str(cfg.trigger_mode).strip().lower()
        if configured_mode == "combined":
            valid = [name for name, passed in modes.items() if passed]
        else:
            valid = [configured_mode] if modes.get(configured_mode, False) else []
        if not valid:
            return None
        trigger_type = valid[0]
        structural_level = sweep_level if trigger_type == "false_breakdown_reclaim" else structure_level if trigger_type == "local_structure_breakout" else prior_extreme
        confirmation_id = f"{symbol}:{direction.name}:{current.timestamp.isoformat()}:{trigger_type}"
        score = _clamp(0.35 * directional_close_position + 0.25 * min(volume_ratio, 2.0) / 2.0 + 0.20 * (1.0 - wick_ratio) + 0.20)
        return MtperConfirmationSnapshot(
            candle=current,
            confirmation_id=confirmation_id,
            trigger_type=trigger_type,
            close_position=directional_close_position,
            wick_ratio=wick_ratio,
            volume_ratio=volume_ratio,
            atr_value=atr_value,
            structural_level=structural_level,
            score=score,
        )

    def _direct_confirmation(self, setup: MtperPreCrossSnapshot, trigger_type: str) -> MtperConfirmationSnapshot:
        candle = setup.candle
        return MtperConfirmationSnapshot(
            candle=candle,
            confirmation_id=f"direct:{trigger_type}:{candle.timestamp.isoformat()}:{setup.direction.name}",
            trigger_type=trigger_type,
            close_position=0.5,
            wick_ratio=0.0,
            volume_ratio=1.0,
            atr_value=setup.atr_value,
            structural_level=setup.hard_stop,
            score=setup.extreme_score,
        )

    def _higher_timeframe_confirmation(self, symbol: str, direction: Direction, decision_time: datetime) -> MtperConfirmationSnapshot:
        candle = self._closed("1h", symbol, decision_time, 1)[-1]
        atr_value = atr(self._closed("1h", symbol, decision_time, 40), 14)[-1]
        return MtperConfirmationSnapshot(
            candle=candle,
            confirmation_id=f"htf:{symbol}:{direction.name}:{candle.timestamp.isoformat()}",
            trigger_type="two_hour_one_hour_confirmation",
            close_position=0.5,
            wick_ratio=0.0,
            volume_ratio=1.0,
            atr_value=atr_value,
            structural_level=candle.low if direction == Direction.LONG else candle.high,
            score=0.60,
        )

    def _candidate(
        self,
        symbol: str,
        setup: MtperPreCrossSnapshot,
        confirmation: MtperConfirmationSnapshot,
        regime: MtperRegime,
        decision_time: datetime,
    ) -> EntryCandidate | None:
        side = setup.direction.value
        entry_reference = confirmation.candle.close
        stop = setup.hard_stop
        stop_distance = side * (entry_reference - stop)
        if stop_distance <= 0:
            self.reject_reasons["hard_stop_wrong_side"] += 1
            return None
        stop_atr_15m = stop_distance / max(confirmation.atr_value, 1e-12)
        stop_pct = stop_distance / max(entry_reference, 1e-12)
        entry_cfg = self.mtper.entry
        if stop_atr_15m < float(entry_cfg.min_stop_atr_15m) or stop_pct < float(entry_cfg.min_stop_pct):
            self.reject_reasons["stop_too_tight"] += 1
            return None
        if stop_atr_15m > float(entry_cfg.max_stop_atr_15m) or stop_pct > float(entry_cfg.max_stop_pct):
            self.reject_reasons["stop_too_wide"] += 1
            return None
        target_1, target_2, target_3 = self._mean_targets(symbol, setup.direction, decision_time, entry_reference, stop_distance)
        target_distance = side * (target_2 - entry_reference)
        configured_cost = (
            2.0 * float(self.config.risk.taker_fee_rate)
            + (float(self.config.risk.market_slippage_bps) + float(self.config.risk.take_profit_slippage_bps)) / 10_000.0
        )
        target_to_cost = target_distance / max(entry_reference * configured_cost, 1e-12)
        if target_to_cost < float(entry_cfg.min_target_to_cost_ratio):
            self.reject_reasons["target_to_cost_insufficient"] += 1
            return None
        initial_fraction = max(0.0, min(1.0, float(entry_cfg.initial_risk_fraction)))
        short_multiplier = 0.75 if setup.direction == Direction.SHORT else 1.0
        confidence = 0.55
        trigger_tf = "4h" if confirmation.trigger_type in {"formal_cross", "pre_cross"} else "1h" if confirmation.trigger_type == "two_hour_one_hour_confirmation" else "15m"
        reason = (
            f"{MTPER_REASON_TOKEN} side={setup.direction.name} setup_id={self.runtime[symbol].setup_id} "
            f"campaign_id={self.runtime[symbol].campaign_id} confirmation_id={confirmation.confirmation_id} "
            f"trigger={confirmation.trigger_type} trigger_tf={trigger_tf} regime={regime.value}"
        )
        signal = Signal(
            direction=setup.direction,
            confidence=confidence,
            reason=reason,
            stop_loss_pct=stop_pct,
            take_profit_pct=max(side * (target_3 - entry_reference) / max(entry_reference, 1e-12), stop_pct),
            risk_multiplier=initial_fraction * short_multiplier,
            max_holding_bars=0,
        )
        quality = _clamp(0.55 * setup.extreme_score + 0.45 * confirmation.score)
        metadata = {
            "strategy_name": MTPER_REASON_TOKEN,
            "setup_id": self.runtime[symbol].setup_id,
            "campaign_id": self.runtime[symbol].campaign_id,
            "confirmation_id": confirmation.confirmation_id,
            "trigger_type": confirmation.trigger_type,
            "trigger_timeframe": trigger_tf,
            "signal_available_time": decision_time.isoformat(),
            "structural_stop_price": stop,
            "setup_atr_4h": setup.atr_value,
            "trigger_atr_15m": confirmation.atr_value,
            "trigger_close": confirmation.candle.close,
            "target_1_price": target_1,
            "target_2_price": target_2,
            "target_3_price": target_3,
            "target_to_cost_ratio": target_to_cost,
            "stop_atr_15m": stop_atr_15m,
            "initial_risk_fraction": initial_fraction,
            "extreme_score": setup.extreme_score,
            "extreme_components": setup.extreme_components,
            "ema_gap_atr": setup.ema_gap_atr,
            "fast_ema_slope_atr": setup.fast_ema_slope_atr,
            "macd_histogram": setup.macd_histogram,
            "formal_cross": setup.formal_cross,
            "confirmation_score": confirmation.score,
            "second_entry_mode": str(self.mtper.second_entry.mode),
            "hard_stop_fixed_before_entry": True,
        }
        return EntryCandidate(
            symbol=symbol,
            signal=signal,
            candle=confirmation.candle,
            rank_score=quality,
            directional_momentum_pct=side * (confirmation.candle.close / max(setup.candle.close, 1e-12) - 1.0),
            volume_ratio=confirmation.volume_ratio,
            filter_reason="mtper_ok",
            metadata=metadata,
        )

    def _mean_targets(
        self,
        symbol: str,
        direction: Direction,
        decision_time: datetime,
        entry: float,
        risk_distance: float,
    ) -> tuple[float, float, float]:
        side = direction.value
        c1h = self._closed("1h", symbol, decision_time, 80)
        c2h = self._closed("2h", symbol, decision_time, 80)
        c4h = self._closed("4h", symbol, decision_time, 100)
        ema1 = ema([item.close for item in c1h], 21)[-1]
        ema2 = ema([item.close for item in c2h], 21)[-1]
        ema4_fast = ema([item.close for item in c4h], int(self.mtper.pre_cross.ema_fast_period))[-1]
        ema4_slow = ema([item.close for item in c4h], int(self.mtper.pre_cross.ema_slow_period))[-1]
        raw = [ema1, ema2, ema4_fast, ema4_slow]
        profitable = sorted((value for value in raw if side * (value - entry) > 0), reverse=direction == Direction.SHORT)
        defaults = [entry + side * risk_distance * multiple for multiple in (0.5, 1.0, 1.5)]
        targets: list[float] = []
        for index in range(3):
            value = profitable[index] if index < len(profitable) else defaults[index]
            minimum = entry + side * risk_distance * (0.5 + 0.5 * index)
            if side * (value - minimum) < 0:
                value = minimum
            targets.append(value)
        if direction == Direction.LONG:
            targets = sorted(targets)
        else:
            targets = sorted(targets, reverse=True)
        return targets[0], targets[1], targets[2]

    def _hard_stop(self, candles: list[Candle], direction: Direction, atr_value: float) -> float:
        lookback = max(2, int(self.mtper.entry.hard_stop_lookback_4h))
        buffer = atr_value * max(0.0, float(self.mtper.entry.hard_stop_atr_buffer))
        window = candles[-lookback:]
        if direction == Direction.LONG:
            return min(item.low for item in window) - buffer
        return max(item.high for item in window) + buffer

    def _setup_hard_invalidated(self, direction: Direction, candles: list[Candle]) -> bool:
        if len(candles) < 5:
            return False
        closes = [item.close for item in candles]
        fast = ema(closes, int(self.mtper.pre_cross.ema_fast_period))
        slow = ema(closes, int(self.mtper.pre_cross.ema_slow_period))
        atr_values = atr(candles, int(self.mtper.pre_cross.atr_period))
        gaps = [(left - right) / max(value, 1e-12) for left, right, value in zip(fast, slow, atr_values)]
        side = direction.value
        gap_worsening = side * (gaps[-1] - gaps[-2]) < 0 and side * (gaps[-2] - gaps[-3]) < 0
        fast_worsening = side * (fast[-1] - fast[-3]) < -float(self.mtper.pre_cross.min_fast_slope_atr) * max(atr_values[-1], 1e-12)
        return gap_worsening and fast_worsening

    def _invalidate(self, symbol: str, timestamp: datetime, reason: str) -> None:
        runtime = self.runtime[symbol]
        self._transition(symbol, MtperState.INVALIDATED, timestamp, reason)
        runtime.setup = None
        runtime.setup_id = None
        runtime.confirmation_id = None
        runtime.direction = Direction.FLAT
        runtime.expire_time = None
        runtime.cooldown_until = timestamp + timedelta(hours=max(1, int(self.mtper.risk_control.symbol_cooldown_hours)))
        self._transition(symbol, MtperState.COOLDOWN, timestamp, reason)
        self.stats["setup_invalidated_count"] += 1

    def _transition(self, symbol: str, state: MtperState, timestamp: datetime, reason: str) -> None:
        runtime = self.runtime[symbol]
        if runtime.state == state:
            return
        self.state_transitions.append(
            {
                "symbol": symbol,
                "setup_id": runtime.setup_id,
                "campaign_id": runtime.campaign_id,
                "confirmation_id": runtime.confirmation_id,
                "from_state": runtime.state.value,
                "to_state": state.value,
                "time": timestamp.isoformat(),
                "reason": reason,
            }
        )
        runtime.state = state

    def _setup_row(
        self,
        symbol: str,
        setup_id: str,
        decision_time: datetime,
        setup: MtperPreCrossSnapshot,
    ) -> dict[str, Any]:
        return {
            "setup_id": setup_id,
            "symbol": symbol,
            "side": setup.direction.name,
            "setup_time": decision_time.isoformat(),
            "four_hour_candle_time": setup.candle.timestamp.isoformat(),
            "ema_gap_atr": setup.ema_gap_atr,
            "ema_gap_abs_atr": setup.ema_gap_abs_atr,
            "gap_changes": list(setup.gap_changes),
            "gap_contracting_bars": setup.gap_contracting_bars,
            "fast_ema_slope_atr": setup.fast_ema_slope_atr,
            "slow_ema_slope_atr": setup.slow_ema_slope_atr,
            "macd_histogram": setup.macd_histogram,
            "macd_histogram_slopes": list(setup.macd_histogram_slopes),
            "formal_cross": setup.formal_cross,
            "extreme_score": setup.extreme_score,
            "extreme_components": setup.extreme_components,
            "hard_stop": setup.hard_stop,
        }

    def _annotate_setup(self, setup_id: str | None, **values: Any) -> None:
        if not setup_id:
            return
        for row in reversed(self.setup_rows):
            if row.get("setup_id") == setup_id:
                row.update(values)
                return

    def _closed(self, timeframe: str, symbol: str, decision_time: datetime, limit: int) -> list[Candle]:
        candles = self.candles.get(timeframe, {}).get(symbol, [])
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        available_before = decision_time - timedelta(milliseconds=interval_to_milliseconds(timeframe))
        end = bisect.bisect_right(timestamps, available_before)
        return candles[max(0, end - limit):end]


def mtper_strategy_config_hash(config: Any) -> str:
    payload = asdict(config.mtper)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_variant(value: Any) -> str:
    normalized = str(value or "pre_cross_htf_15m").strip().lower()
    allowed = {"formal_cross", "pre_cross", "pre_cross_htf", "pre_cross_htf_15m"}
    if normalized not in allowed:
        raise ValueError(f"unsupported MTPER stage_variant: {value}")
    return normalized


def _directionally_improving(values: list[float], side: int, bars: int) -> bool:
    bars = max(1, int(bars))
    if len(values) < bars + 1:
        return False
    return all(side * (values[index] - values[index - 1]) > 0 for index in range(len(values) - bars, len(values)))


def _percentile_rank(value: float, values: list[float]) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return 0.5
    below = sum(item < value for item in finite)
    equal = sum(item == value for item in finite)
    return (below + 0.5 * equal) / len(finite)


def _contracting_tail(values: list[float]) -> int:
    count = 0
    for index in range(len(values) - 1, 0, -1):
        if values[index] < values[index - 1]:
            count += 1
        else:
            break
    return count


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
