from __future__ import annotations

import bisect
import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .data import interval_to_milliseconds
from .indicators import atr, ema, macd
from .live_trader import EntryCandidate
from .models import Candle, Direction, Signal


MTPC_REASON_TOKEN = "multi_timeframe_trend_pullback_continuation"
MTPC_FEATURE_VERSION = "mtpc_features_v1"
MTPC_EVENT_VERSION = "mtpc_bidirectional_trend_pullback_v3"


class MtpcState(str, Enum):
    IDLE = "IDLE"
    TREND_SETUP = "TREND_SETUP"
    FIRST_PULLBACK_PENDING = "FIRST_PULLBACK_PENDING"
    LOWER_TIMEFRAME_CONFIRMING = "LOWER_TIMEFRAME_CONFIRMING"
    ENTRY_READY = "ENTRY_READY"
    ORDER_PENDING = "ORDER_PENDING"
    PARTIAL_FILL = "PARTIAL_FILL"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    POSITION_OPEN = "POSITION_OPEN"
    CANCEL_PENDING = "CANCEL_PENDING"
    EXITING = "EXITING"
    COOLDOWN = "COOLDOWN"
    RECOVERY_AFTER_RESTART = "RECOVERY_AFTER_RESTART"


class MtpcRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    NEUTRAL = "NEUTRAL"
    CHAOS_NO_TRADE = "CHAOS_NO_TRADE"


@dataclass(frozen=True)
class MtpcImpulseSnapshot:
    candle: Candle
    breakout_level: float
    impulse_high: float
    impulse_low: float
    atr_value: float
    breakout_distance_atr: float
    body_atr: float
    close_position: float
    upper_wick_ratio: float
    volume_ratio: float
    prior_move_atr: float
    ema21_extension_atr: float
    quality_score: float
    direction: Direction = Direction.LONG


@dataclass(frozen=True)
class MtpcPullbackSnapshot:
    candle: Candle
    low: float
    depth_atr: float
    retrace_fraction: float
    volume_to_impulse: float
    close_below_breakout_atr: float
    ema_distance_atr: float
    bars_after_impulse: int
    quality_score: float
    direction: Direction = Direction.LONG


@dataclass
class MtpcSymbolRuntime:
    state: MtpcState = MtpcState.IDLE
    setup_id: str | None = None
    event_id: str | None = None
    confirmation_id: str | None = None
    impulse: MtpcImpulseSnapshot | None = None
    pullback: MtpcPullbackSnapshot | None = None
    setup_time: datetime | None = None
    expire_time: datetime | None = None
    cooldown_until: datetime | None = None
    last_processed_15m: datetime | None = None
    consumed_confirmation_ids: set[str] | None = None
    direction: Direction | None = None

    def __post_init__(self) -> None:
        if self.consumed_confirmation_ids is None:
            self.consumed_confirmation_ids = set()


class MtpcEngine:
    """Point-in-time trend impulse, first-pullback, and continuation engine."""

    REQUIRED_TIMEFRAMES = ("5m", "15m", "1h", "4h")

    def __init__(
        self,
        config: Any,
        candles_by_timeframe: dict[str, dict[str, list[Candle]]],
    ) -> None:
        self.config = config
        self.mtpc = config.mtpc
        self.candles = candles_by_timeframe
        self.timestamps = {
            timeframe: {
                symbol: [candle.timestamp for candle in candles]
                for symbol, candles in by_symbol.items()
            }
            for timeframe, by_symbol in candles_by_timeframe.items()
        }
        configured = tuple(config.trading.symbols)
        self.symbols = tuple(
            symbol
            for symbol in configured
            if all(symbol in self.candles.get(timeframe, {}) for timeframe in self.REQUIRED_TIMEFRAMES)
        )
        requested_trade_symbols = tuple(
            self.mtpc.enabled_symbols or config.trading.entry_symbols or config.trading.symbols
        )
        observation_set = set(self.symbols)
        self.trade_symbols = tuple(symbol for symbol in requested_trade_symbols if symbol in observation_set)
        self.runtime = {symbol: MtpcSymbolRuntime() for symbol in self.trade_symbols}
        self.stats: Counter[str] = Counter()
        self.reject_reasons: Counter[str] = Counter()
        self.regime_counts: Counter[str] = Counter()
        self.state_transitions: list[dict[str, Any]] = []
        self.event_rows: list[dict[str, Any]] = []
        self.strategy_config_hash = mtpc_strategy_config_hash(config)
        self.cache_namespace = f"mtpc:{self.strategy_config_hash}"
        self._last_market_scan_bar: datetime | None = None
        self._last_regime_bar: datetime | None = None
        self._raw_regime: MtpcRegime = MtpcRegime.NEUTRAL
        self._raw_regime_count = 0
        self._regime: MtpcRegime = MtpcRegime.NEUTRAL
        self._regime_enter_time: datetime | None = None
        self._ranking_bar: datetime | None = None
        self._ranking_cache: dict[str, dict[str, float]] = {}
        self._symbol_trend_cache: dict[tuple[str, Direction], tuple[datetime, bool, dict[str, float]]] = {}

    def scan(
        self,
        decision_time: datetime,
        occupied_symbols: set[str],
        allowed_symbols: set[str] | None = None,
    ) -> list[EntryCandidate]:
        market = self._closed("5m", self.mtpc.regime.btc_symbol, decision_time, 1)
        if not market:
            return []
        latest_market_bar = market[-1].timestamp
        if latest_market_bar == self._last_market_scan_bar:
            return []
        self._last_market_scan_bar = latest_market_bar
        regime = self._update_regime(decision_time)
        self.regime_counts[regime.value] += 1
        if regime == MtpcRegime.BULL_TREND and bool(self.mtpc.allow_long):
            directions = (Direction.LONG,)
        elif regime == MtpcRegime.BEAR_TREND and bool(self.mtpc.allow_short):
            directions = (Direction.SHORT,)
        elif regime == MtpcRegime.NEUTRAL:
            directions = tuple(
                direction
                for direction, enabled in (
                    (
                        Direction.LONG,
                        bool(self.mtpc.allow_long)
                        and bool(self.mtpc.regime.allow_neutral_symbol_long),
                    ),
                    (
                        Direction.SHORT,
                        bool(self.mtpc.allow_short)
                        and bool(self.mtpc.regime.allow_neutral_symbol_short),
                    ),
                )
                if enabled
            )
            if not directions:
                self.reject_reasons["regime_neutral"] += 1
                return []
        else:
            self.reject_reasons[f"regime_{regime.value.lower()}"] += 1
            return []
        rankings = self._rankings(decision_time)
        allowed = allowed_symbols if allowed_symbols is not None else set(self.trade_symbols)
        candidates: list[EntryCandidate] = []
        for symbol in self.trade_symbols:
            runtime = self.runtime[symbol]
            if runtime.cooldown_until is not None and decision_time < runtime.cooldown_until:
                continue
            if runtime.state == MtpcState.COOLDOWN:
                self._transition(symbol, MtpcState.IDLE, decision_time, "cooldown_complete")
            if symbol in occupied_symbols or symbol not in allowed:
                continue
            rank = rankings.get(symbol)
            if rank is None:
                self.reject_reasons["ranking_missing"] += 1
                continue
            runtime_direction = runtime.direction if runtime.impulse is not None else None
            if runtime_direction in directions:
                symbol_directions = (runtime_direction,)
            elif len(directions) == 2:
                preferred = (
                    Direction.LONG
                    if float(rank.get("percentile", 0.5)) >= 0.5
                    else Direction.SHORT
                )
                symbol_directions = (preferred,)
            else:
                symbol_directions = directions
            for direction in symbol_directions:
                candidate = self._scan_symbol(symbol, decision_time, rank, direction)
                if candidate is not None:
                    candidates.append(candidate)
                    break
        candidates.sort(key=lambda item: (-item.rank_score, item.symbol))
        return candidates[: max(1, int(self.mtpc.ranking.max_candidates_per_scan))]

    def mark_order_pending(self, symbol: str, timestamp: datetime | None = None) -> None:
        self._transition(symbol, MtpcState.ORDER_PENDING, timestamp or datetime.utcnow(), "entry_order_submitted")
        self.stats["order_pending_count"] += 1

    def mark_filled(self, symbol: str, partial: bool = False, timestamp: datetime | None = None) -> None:
        state = MtpcState.PARTIAL_FILL if partial else MtpcState.PROTECTION_PENDING
        self._transition(symbol, state, timestamp or datetime.utcnow(), "entry_fill")
        self.stats["partial_fill_count" if partial else "filled_count"] += 1

    def mark_protected(self, symbol: str, timestamp: datetime | None = None) -> None:
        self._transition(symbol, MtpcState.POSITION_OPEN, timestamp or datetime.utcnow(), "protection_active")
        self.stats["protected_count"] += 1

    def mark_cancelled(self, symbol: str, timestamp: datetime, reason: str) -> None:
        self._transition(symbol, MtpcState.CANCEL_PENDING, timestamp, reason)
        self.stats["cancel_pending_count"] += 1
        self.reject_reasons[f"execution_{reason}"] += 1
        self._reset_with_cooldown(
            symbol,
            timestamp,
            f"entry_cancelled:{reason}",
            max(0, int(self.mtpc.risk_control.entry_cancel_cooldown_minutes)),
        )

    def mark_closed(self, symbol: str, timestamp: datetime, reason: str = "closed") -> None:
        self._reset_with_cooldown(
            symbol,
            timestamp,
            reason,
            max(0, int(self.mtpc.risk_control.symbol_cooldown_hours)) * 60,
        )

    def _reset_with_cooldown(
        self,
        symbol: str,
        timestamp: datetime,
        reason: str,
        cooldown_minutes: int,
    ) -> None:
        runtime = self.runtime[symbol]
        runtime.cooldown_until = timestamp + timedelta(minutes=max(0, cooldown_minutes))
        runtime.setup_id = None
        runtime.event_id = None
        runtime.confirmation_id = None
        runtime.impulse = None
        runtime.pullback = None
        runtime.setup_time = None
        runtime.expire_time = None
        runtime.direction = None
        next_state = MtpcState.COOLDOWN if cooldown_minutes > 0 else MtpcState.IDLE
        self._transition(symbol, next_state, timestamp, reason)

    def report(self) -> dict[str, Any]:
        return {
            "strategy_name": MTPC_REASON_TOKEN,
            "feature_version": MTPC_FEATURE_VERSION,
            "event_definition_version": MTPC_EVENT_VERSION,
            "strategy_config_hash": self.strategy_config_hash,
            "cache_namespace": self.cache_namespace,
            "stage_variant": self.mtpc.research.stage_variant,
            "symbols": list(self.trade_symbols),
            "observation_symbols": list(self.symbols),
            "trade_symbols": list(self.trade_symbols),
            "stats": dict(sorted(self.stats.items())),
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "regime_counts": dict(sorted(self.regime_counts.items())),
            "event_count": len(self.event_rows),
            "events": self.event_rows,
            "state_transitions": self.state_transitions,
            "historical_test_policy": {
                "historical_test_start": self.mtpc.research.historical_test_start,
                "historical_test_end": self.mtpc.research.historical_test_end,
                "is_untouched_final_holdout": False,
                "final_acceptance_source": self.mtpc.research.final_acceptance_source,
            },
        }

    def _scan_symbol(
        self,
        symbol: str,
        decision_time: datetime,
        rank: dict[str, float],
        direction: Direction = Direction.LONG,
    ) -> EntryCandidate | None:
        runtime = self.runtime[symbol]
        stage = _stage_variant(self.mtpc.research.stage_variant)
        if runtime.direction is not None and runtime.direction != direction and runtime.impulse is not None:
            self._invalidate(symbol, decision_time, "regime_direction_changed")
            runtime = self.runtime[symbol]
        directional_rank = self._directional_rank(rank, direction)
        trend_ok, trend_features = self._symbol_trend(symbol, decision_time, direction)
        if not trend_ok:
            if runtime.impulse is not None:
                self._invalidate(symbol, decision_time, "symbol_trend_lost")
            self.reject_reasons["symbol_trend_not_confirmed"] += 1
            return None
        if bool(self.mtpc.ranking.enabled):
            percentile = directional_rank["percentile"]
            minimum_percentile = (
                float(self.mtpc.ranking.min_percentile)
                if direction == Direction.LONG
                else float(self.mtpc.ranking.short_min_percentile)
            )
            maximum_percentile = (
                float(self.mtpc.ranking.max_percentile)
                if direction == Direction.LONG
                else float(self.mtpc.ranking.short_max_percentile)
            )
            if percentile < minimum_percentile:
                self.reject_reasons["ranking_below_minimum"] += 1
                return None
            if percentile > maximum_percentile:
                self.reject_reasons["ranking_overextended_tail"] += 1
                return None
            if directional_rank["extension_atr"] > float(self.mtpc.ranking.max_extension_atr_1h):
                self.reject_reasons["ranking_price_overextended"] += 1
                return None

        impulse_timeframe = str(self.mtpc.impulse.timeframe)
        impulse_candles = self._closed(impulse_timeframe, symbol, decision_time, 80)
        if len(impulse_candles) < 60:
            self.reject_reasons["insufficient_15m_history"] += 1
            return None
        latest_impulse_bar = impulse_candles[-1].timestamp
        if runtime.impulse is not None:
            if runtime.expire_time is not None and decision_time >= runtime.expire_time:
                self._invalidate(symbol, decision_time, "impulse_expired")
                runtime = self.runtime[symbol]
            elif (
                runtime.direction == Direction.LONG
                and impulse_candles[-1].close < runtime.impulse.impulse_low
            ) or (
                runtime.direction == Direction.SHORT
                and impulse_candles[-1].close > runtime.impulse.impulse_high
            ):
                self._invalidate(symbol, decision_time, "impulse_structure_invalidated")
                runtime = self.runtime[symbol]

        if (
            stage == "trend_ema_pullback"
            and runtime.impulse is None
            and runtime.last_processed_15m != latest_impulse_bar
        ):
            runtime.last_processed_15m = latest_impulse_bar
            setup = self._detect_trend_ema_pullback(impulse_candles, direction)
            if setup is None:
                return None
            impulse, pullback = setup
            setup_id = f"{symbol}:{direction.name}:ema:{impulse.candle.timestamp.isoformat()}"
            runtime.setup_id = setup_id
            runtime.event_id = f"event:{setup_id}"
            runtime.impulse = impulse
            runtime.pullback = pullback
            runtime.direction = direction
            runtime.setup_time = decision_time
            runtime.expire_time = decision_time + timedelta(
                minutes=max(5, int(self.mtpc.impulse.pending_expiry_minutes))
            )
            self._transition(symbol, MtpcState.TREND_SETUP, decision_time, "trend_ema_pullback_created")
            self.stats["impulse_count"] += 1
            self.stats["pullback_count"] += 1
            self.event_rows.append(
                self._event_row(symbol, runtime, impulse, directional_rank, trend_features, decision_time)
            )
            self._annotate_event(
                runtime.event_id,
                setup_type="trend_ema_pullback",
                pullback_time=decision_time.isoformat(),
                pullback_candle_time=pullback.candle.timestamp.isoformat(),
                pullback_extreme=pullback.low,
                pullback_depth_atr=pullback.depth_atr,
                pullback_volume_to_impulse=pullback.volume_to_impulse,
                pullback_quality=pullback.quality_score,
                trend_ema_reclaim_extension_atr=impulse.ema21_extension_atr,
            )
            self._transition(
                symbol,
                MtpcState.LOWER_TIMEFRAME_CONFIRMING,
                decision_time,
                "waiting_trend_ema_confirmation",
            )

        if runtime.impulse is None and runtime.last_processed_15m != latest_impulse_bar:
            runtime.last_processed_15m = latest_impulse_bar
            impulse = self._detect_impulse(impulse_candles, direction)
            if impulse is None:
                return None
            setup_id = f"{symbol}:{direction.name}:{impulse.candle.timestamp.isoformat()}"
            runtime.setup_id = setup_id
            runtime.event_id = f"event:{setup_id}"
            runtime.impulse = impulse
            runtime.direction = direction
            runtime.setup_time = decision_time
            runtime.expire_time = decision_time + timedelta(
                minutes=max(5, int(self.mtpc.impulse.pending_expiry_minutes))
            )
            self._transition(symbol, MtpcState.TREND_SETUP, decision_time, "trend_impulse_created")
            self.stats["impulse_count"] += 1
            self.event_rows.append(
                self._event_row(symbol, runtime, impulse, directional_rank, trend_features, decision_time)
            )
            if stage == "trend_impulse":
                structural_stop = (
                    impulse.impulse_low - impulse.atr_value * float(self.mtpc.pullback.stop_atr_buffer)
                    if direction == Direction.LONG
                    else impulse.impulse_high + impulse.atr_value * float(self.mtpc.pullback.stop_atr_buffer)
                )
                return self._candidate(
                    symbol,
                    runtime,
                    impulse.candle,
                    structural_stop,
                    directional_rank,
                    trend_features,
                    confirmation_quality=impulse.quality_score,
                    trigger_type="trend_impulse",
                )
            self._transition(symbol, MtpcState.FIRST_PULLBACK_PENDING, decision_time, "waiting_first_pullback")

        impulse = runtime.impulse
        if impulse is None:
            return None
        if runtime.pullback is None:
            pullback_timeframe = str(self.mtpc.pullback.pullback_timeframe)
            pullback_candles = (
                impulse_candles
                if pullback_timeframe == impulse_timeframe
                else self._closed(pullback_timeframe, symbol, decision_time, 160)
            )
            pullback, pullback_reason = self._detect_pullback(pullback_candles, impulse, direction)
            if pullback_reason:
                if pullback_reason.startswith("pullback_wait_"):
                    self.reject_reasons[pullback_reason] += 1
                    self._annotate_event(runtime.event_id, latest_pullback_wait_reason=pullback_reason)
                else:
                    self._invalidate(symbol, decision_time, pullback_reason)
                return None
            if pullback is None:
                self.reject_reasons["first_pullback_not_confirmed"] += 1
                return None
            runtime.pullback = pullback
            self.stats["pullback_count"] += 1
            self._transition(symbol, MtpcState.LOWER_TIMEFRAME_CONFIRMING, decision_time, "first_pullback_confirmed")
            self._annotate_event(
                runtime.event_id,
                pullback_time=decision_time.isoformat(),
                pullback_candle_time=pullback.candle.timestamp.isoformat(),
                pullback_extreme=pullback.low,
                pullback_depth_atr=pullback.depth_atr,
                pullback_retrace_fraction=pullback.retrace_fraction,
                pullback_volume_to_impulse=pullback.volume_to_impulse,
                pullback_quality=pullback.quality_score,
            )

        pullback = runtime.pullback
        assert pullback is not None
        confirmation = self._detect_confirmation(symbol, decision_time, pullback, direction)
        if confirmation is None:
            self.reject_reasons["five_minute_confirmation_absent"] += 1
            return None
        candle, quality, structural_extreme = confirmation
        confirmation_label = "5m_reclaim" if direction == Direction.LONG else "5m_reject"
        confirmation_id = f"{runtime.event_id}:{candle.timestamp.isoformat()}:{confirmation_label}"
        assert runtime.consumed_confirmation_ids is not None
        if confirmation_id in runtime.consumed_confirmation_ids:
            return None
        runtime.consumed_confirmation_ids.add(confirmation_id)
        runtime.confirmation_id = confirmation_id
        if direction == Direction.LONG:
            stop = min(pullback.low, structural_extreme) - impulse.atr_value * float(
                self.mtpc.pullback.stop_atr_buffer
            )
        else:
            stop = max(pullback.low, structural_extreme) + impulse.atr_value * float(
                self.mtpc.pullback.stop_atr_buffer
            )
        candidate = self._candidate(
            symbol,
            runtime,
            candle,
            stop,
            directional_rank,
            trend_features,
            confirmation_quality=quality,
            trigger_type=(
                f"trend_ema_pullback_{confirmation_label}"
                if stage == "trend_ema_pullback"
                else f"first_pullback_{confirmation_label}"
            ),
        )
        if candidate is not None:
            self._transition(symbol, MtpcState.ENTRY_READY, decision_time, "five_minute_reclaim_confirmed")
            self.stats["entry_ready_count"] += 1
            self._annotate_event(
                runtime.event_id,
                confirmation_id=confirmation_id,
                confirmation_time=decision_time.isoformat(),
                confirmation_candle_time=candle.timestamp.isoformat(),
                confirmation_quality=quality,
            )
        return candidate

    def _detect_trend_ema_pullback(
        self,
        candles: list[Candle],
        direction: Direction = Direction.LONG,
    ) -> tuple[MtpcImpulseSnapshot, MtpcPullbackSnapshot] | None:
        cfg = self.mtpc.pullback
        lookback = max(4, int(cfg.trend_ema_lookback_bars))
        if len(candles) < max(40, lookback + 25):
            return None
        current = candles[-1]
        previous = candles[-lookback - 1:-1]
        closes = [item.close for item in candles]
        ema9_values = ema(closes, 9)
        ema21_values = ema(closes, 21)
        ema9_value = ema9_values[-1]
        ema21_value = ema21_values[-1]
        atr_value = max(atr(candles, 14)[-1], 1e-12)
        side = direction.value
        alignment_atr = side * (ema9_value - ema21_value) / atr_value
        if alignment_atr < float(cfg.trend_ema_min_alignment_atr):
            self.reject_reasons["trend_ema_alignment_rejected"] += 1
            return None
        if direction == Direction.LONG:
            prior_extreme = max(item.high for item in previous)
            pullback_extreme = current.low
            impulse_high = prior_extreme
            impulse_low = current.low
        else:
            prior_extreme = min(item.low for item in previous)
            pullback_extreme = current.high
            impulse_high = current.high
            impulse_low = prior_extreme
        prior_extension = side * (prior_extreme - ema21_value) / atr_value
        depth_atr = side * (prior_extreme - pullback_extreme) / atr_value
        ema_distance_atr = abs(pullback_extreme - ema21_value) / atr_value
        adverse_close_atr = side * (ema21_value - current.close) / atr_value
        reclaim_extension_atr = side * (current.close - ema21_value) / atr_value
        average_volume = statistics.fmean(item.volume for item in candles[-21:-1])
        volume_ratio = current.volume / max(average_volume, 1e-12)
        valid = (
            depth_atr >= float(cfg.trend_ema_min_depth_atr)
            and depth_atr <= float(cfg.trend_ema_max_depth_atr)
            and prior_extension >= float(cfg.trend_ema_min_prior_extension_atr)
            and volume_ratio <= float(cfg.trend_ema_max_volume_ratio)
            and ema_distance_atr <= float(cfg.trend_ema_proximity_atr)
            and adverse_close_atr <= float(cfg.trend_ema_max_adverse_close_atr)
            and reclaim_extension_atr >= float(cfg.trend_ema_min_reclaim_extension_atr)
            and reclaim_extension_atr <= float(cfg.trend_ema_max_reclaim_extension_atr)
        )
        if not valid:
            self.reject_reasons["trend_ema_pullback_quality_rejected"] += 1
            return None
        quality = _clamp(
            0.25 * (1.0 - min(ema_distance_atr / max(float(cfg.trend_ema_proximity_atr), 1e-12), 1.0))
            + 0.20 * min(prior_extension / max(float(cfg.trend_ema_min_prior_extension_atr), 1e-12), 2.0) / 2.0
            + 0.20 * (1.0 - min(volume_ratio / max(float(cfg.trend_ema_max_volume_ratio), 1e-12), 1.0))
            + 0.20 * min(alignment_atr + 0.5, 1.0)
            + 0.15 * (1.0 - min(depth_atr / max(float(cfg.trend_ema_max_depth_atr), 1e-12), 1.0))
        )
        candle_range = max(current.high - current.low, 1e-12)
        close_position = (
            (current.close - current.low) / candle_range
            if direction == Direction.LONG
            else (current.high - current.close) / candle_range
        )
        adverse_wick_ratio = (
            (current.high - max(current.open, current.close)) / candle_range
            if direction == Direction.LONG
            else (min(current.open, current.close) - current.low) / candle_range
        )
        impulse = MtpcImpulseSnapshot(
            candle=current,
            breakout_level=ema21_value,
            impulse_high=impulse_high,
            impulse_low=impulse_low,
            atr_value=atr_value,
            breakout_distance_atr=reclaim_extension_atr,
            body_atr=side * (current.close - current.open) / atr_value,
            close_position=close_position,
            upper_wick_ratio=adverse_wick_ratio,
            volume_ratio=volume_ratio,
            prior_move_atr=prior_extension,
            ema21_extension_atr=reclaim_extension_atr,
            quality_score=quality,
            direction=direction,
        )
        pullback = MtpcPullbackSnapshot(
            candle=current,
            low=pullback_extreme,
            depth_atr=depth_atr,
            retrace_fraction=0.0,
            volume_to_impulse=volume_ratio,
            close_below_breakout_atr=adverse_close_atr,
            ema_distance_atr=ema_distance_atr,
            bars_after_impulse=1,
            quality_score=quality,
            direction=direction,
        )
        return impulse, pullback

    def _detect_impulse(
        self,
        candles: list[Candle],
        direction: Direction = Direction.LONG,
    ) -> MtpcImpulseSnapshot | None:
        cfg = self.mtpc.impulse
        lookback = max(4, int(cfg.breakout_lookback_bars))
        if len(candles) < lookback + 30:
            return None
        current = candles[-1]
        previous = candles[-lookback - 1:-1]
        atr_value = max(atr(candles, 14)[-1], 1e-12)
        side = direction.value
        breakout_level = (
            max(item.high for item in previous)
            if direction == Direction.LONG
            else min(item.low for item in previous)
        )
        breakout_distance = side * (current.close - breakout_level) / atr_value
        body_atr = side * (current.close - current.open) / atr_value
        candle_range = max(current.high - current.low, 1e-12)
        close_position = (
            (current.close - current.low) / candle_range
            if direction == Direction.LONG
            else (current.high - current.close) / candle_range
        )
        adverse_wick_ratio = (
            (current.high - max(current.open, current.close)) / candle_range
            if direction == Direction.LONG
            else (min(current.open, current.close) - current.low) / candle_range
        )
        average_volume = statistics.fmean(item.volume for item in candles[-21:-1])
        volume_ratio = current.volume / max(average_volume, 1e-12)
        prior_move = side * (current.open - candles[-5].close) / atr_value
        ema21_value = ema([item.close for item in candles], 21)[-1]
        extension = side * (current.close - ema21_value) / atr_value
        valid = (
            float(cfg.min_breakout_distance_atr) <= breakout_distance <= float(cfg.max_breakout_distance_atr)
            and body_atr >= float(cfg.min_body_atr)
            and close_position >= float(cfg.min_close_position)
            and adverse_wick_ratio <= float(cfg.max_upper_wick_ratio)
            and volume_ratio >= float(cfg.min_volume_ratio)
            and prior_move >= float(cfg.min_prior_move_atr)
            and prior_move <= float(cfg.max_prior_move_atr)
            and extension <= float(cfg.max_ema21_extension_atr)
        )
        if not valid:
            self.reject_reasons["impulse_quality_rejected"] += 1
            return None
        quality = _clamp(
            0.20 * min(breakout_distance / max(float(cfg.min_breakout_distance_atr), 1e-12), 2.0) / 2.0
            + 0.20 * min(body_atr / max(float(cfg.min_body_atr), 1e-12), 2.0) / 2.0
            + 0.20 * close_position
            + 0.20 * min(volume_ratio, 2.0) / 2.0
            + 0.20 * (1.0 - adverse_wick_ratio)
        )
        return MtpcImpulseSnapshot(
            candle=current,
            breakout_level=breakout_level,
            impulse_high=current.high,
            impulse_low=current.low,
            atr_value=atr_value,
            breakout_distance_atr=breakout_distance,
            body_atr=body_atr,
            close_position=close_position,
            upper_wick_ratio=adverse_wick_ratio,
            volume_ratio=volume_ratio,
            prior_move_atr=prior_move,
            ema21_extension_atr=extension,
            quality_score=quality,
            direction=direction,
        )

    def _detect_pullback(
        self,
        candles: list[Candle],
        impulse: MtpcImpulseSnapshot,
        direction: Direction = Direction.LONG,
    ) -> tuple[MtpcPullbackSnapshot | None, str]:
        cfg = self.mtpc.pullback
        impulse_available_time = impulse.candle.timestamp + timedelta(
            milliseconds=interval_to_milliseconds(self.mtpc.impulse.timeframe)
        )
        after = [item for item in candles if item.timestamp >= impulse_available_time]
        bars = len(after)
        if bars < max(1, int(cfg.min_bars_after_impulse)):
            return None, ""
        if bars > max(1, int(cfg.max_bars_after_impulse)):
            return None, "pullback_arrived_too_late"
        atr_value = max(impulse.atr_value, 1e-12)
        side = direction.value
        structure_broken = any(
            (
                item.low <= impulse.impulse_low
                if direction == Direction.LONG
                else item.high >= impulse.impulse_high
            )
            or side * (impulse.breakout_level - item.close) / atr_value
            > float(cfg.max_close_below_breakout_atr)
            for item in after
        )
        if structure_broken:
            return None, "pullback_structure_broken"

        # The first post-impulse candle is often the pullback trough. Looking
        # only at the latest candle silently discarded that common path once a
        # recovery candle printed a higher low.
        if direction == Direction.LONG:
            extreme_index, extreme_candle = min(enumerate(after), key=lambda item: item[1].low)
            extreme_price = extreme_candle.low
            depth = (impulse.impulse_high - extreme_price) / atr_value
        else:
            extreme_index, extreme_candle = max(enumerate(after), key=lambda item: item[1].high)
            extreme_price = extreme_candle.high
            depth = (extreme_price - impulse.impulse_low) / atr_value
        impulse_range = max(impulse.impulse_high - impulse.impulse_low, 1e-12)
        retrace = depth * atr_value / impulse_range
        close_below = side * (impulse.breakout_level - extreme_candle.close) / atr_value
        if depth > float(cfg.max_depth_atr) or retrace > float(cfg.max_retrace_fraction):
            return None, "pullback_too_deep"
        if depth < float(cfg.min_depth_atr) or retrace < float(cfg.min_retrace_fraction):
            return None, "pullback_wait_too_shallow"
        volume_ratio = extreme_candle.volume / max(impulse.candle.volume, 1e-12)
        if volume_ratio > float(cfg.max_volume_to_impulse):
            return None, "pullback_wait_volume_not_contracted"
        extreme_position = next(
            index for index, item in enumerate(candles) if item.timestamp == extreme_candle.timestamp
        )
        closes = [item.close for item in candles[: extreme_position + 1]]
        ema9 = ema(closes, 9)[-1]
        ema21 = ema(closes, 21)[-1]
        ema_distance = min(abs(extreme_price - ema9), abs(extreme_price - ema21)) / atr_value
        if ema_distance > float(cfg.ema_proximity_atr):
            return None, "pullback_wait_ema_too_far"
        pullback_shape = (
            extreme_candle.low < impulse.candle.close and extreme_candle.close <= impulse.impulse_high
            if direction == Direction.LONG
            else extreme_candle.high > impulse.candle.close and extreme_candle.close >= impulse.impulse_low
        )
        if not pullback_shape:
            return None, "pullback_wait_shape_invalid"
        quality = _clamp(
            0.30 * (1.0 - min(retrace / max(float(cfg.max_retrace_fraction), 1e-12), 1.0))
            + 0.30 * (1.0 - min(volume_ratio / max(float(cfg.max_volume_to_impulse), 1e-12), 1.0))
            + 0.20 * (1.0 - min(ema_distance / max(float(cfg.ema_proximity_atr), 1e-12), 1.0))
            + 0.20 * (1.0 - min(max(close_below, 0.0) / max(float(cfg.max_close_below_breakout_atr), 1e-12), 1.0))
        )
        return (
            MtpcPullbackSnapshot(
                candle=extreme_candle,
                low=extreme_price,
                depth_atr=depth,
                retrace_fraction=retrace,
                volume_to_impulse=volume_ratio,
                close_below_breakout_atr=close_below,
                ema_distance_atr=ema_distance,
                bars_after_impulse=extreme_index + 1,
                quality_score=quality,
                direction=direction,
            ),
            "",
        )

    def _detect_confirmation(
        self,
        symbol: str,
        decision_time: datetime,
        pullback: MtpcPullbackSnapshot,
        direction: Direction = Direction.LONG,
    ) -> tuple[Candle, float, float] | None:
        cfg = self.mtpc.pullback
        candles = self._closed(cfg.confirmation_timeframe, symbol, decision_time, 80)
        if len(candles) < 35:
            return None
        current = candles[-1]
        pullback_available = pullback.candle.timestamp + timedelta(
            milliseconds=interval_to_milliseconds(cfg.pullback_timeframe)
        )
        if current.timestamp < pullback_available:
            return None
        closes = [item.close for item in candles]
        ema_value = ema(closes, max(2, int(cfg.confirmation_ema_period)))[-1]
        _, _, hist = macd(closes)
        candle_range = max(current.high - current.low, 1e-12)
        close_position = (
            (current.close - current.low) / candle_range
            if direction == Direction.LONG
            else (current.high - current.close) / candle_range
        )
        adverse_wick = (
            (current.high - max(current.open, current.close)) / candle_range
            if direction == Direction.LONG
            else (min(current.open, current.close) - current.low) / candle_range
        )
        volume_average = statistics.fmean(item.volume for item in candles[-21:-1])
        volume_ratio = current.volume / max(volume_average, 1e-12)
        higher_low_bars = max(2, int(cfg.confirmation_higher_low_bars))
        previous = candles[-higher_low_bars - 1:-1]
        if direction == Direction.LONG:
            structural_extreme = min(item.low for item in previous)
            structure_ok = current.low > structural_extreme
            confirmation_level = (
                candles[-2].high if bool(cfg.confirmation_break_previous_high) else candles[-2].close
            )
            price_ok = current.close > current.open and current.close > ema_value and current.close > confirmation_level
            macd_improving = hist[-1] > hist[-2]
        else:
            structural_extreme = max(item.high for item in previous)
            structure_ok = current.high < structural_extreme
            confirmation_level = (
                candles[-2].low if bool(cfg.confirmation_break_previous_high) else candles[-2].close
            )
            price_ok = current.close < current.open and current.close < ema_value and current.close < confirmation_level
            macd_improving = hist[-1] < hist[-2]
        macd_ok = not bool(cfg.require_confirmation_macd_improvement) or macd_improving
        if not (
            structure_ok
            and price_ok
            and macd_ok
            and close_position >= float(cfg.confirmation_min_close_position)
            and adverse_wick <= float(cfg.confirmation_max_upper_wick_ratio)
            and volume_ratio >= float(cfg.confirmation_min_volume_ratio)
        ):
            return None
        quality = _clamp(
            0.30 * close_position
            + 0.25 * (1.0 - adverse_wick)
            + 0.25 * min(volume_ratio, 2.0) / 2.0
            + 0.20
        )
        if direction == Direction.LONG:
            return current, quality, min(structural_extreme, current.low)
        return current, quality, max(structural_extreme, current.high)

    def _candidate(
        self,
        symbol: str,
        runtime: MtpcSymbolRuntime,
        candle: Candle,
        structural_stop: float,
        rank: dict[str, float],
        trend_features: dict[str, float],
        confirmation_quality: float,
        trigger_type: str,
    ) -> EntryCandidate | None:
        impulse = runtime.impulse
        if impulse is None:
            return None
        direction = runtime.direction or impulse.direction
        risk_distance = direction.value * (candle.close - structural_stop)
        if risk_distance <= 0:
            self.reject_reasons["structural_stop_wrong_side"] += 1
            return None
        stop_pct = risk_distance / max(candle.close, 1e-12)
        take_profit_r = max(0.1, float(self.mtpc.exit.take_profit_1_r))
        quality = _clamp(
            0.30 * rank["percentile"]
            + 0.25 * impulse.quality_score
            + 0.20 * (runtime.pullback.quality_score if runtime.pullback is not None else impulse.quality_score)
            + 0.25 * confirmation_quality
        )
        trigger_timeframe = (
            str(self.mtpc.impulse.timeframe)
            if trigger_type == "trend_impulse"
            else str(self.mtpc.pullback.confirmation_timeframe)
        )
        reason = (
            f"{MTPC_REASON_TOKEN} side={direction.name} trigger={trigger_type} "
            f"trigger_tf={trigger_timeframe} event_id={runtime.event_id}"
        )
        signal = Signal(
            direction=direction,
            confidence=quality,
            reason=reason,
            stop_loss_pct=stop_pct,
            take_profit_pct=stop_pct * take_profit_r,
            risk_multiplier=1.0,
            max_holding_bars=max(1, int(self.mtpc.exit.max_holding_minutes / 30)),
        )
        metadata = {
            "setup_id": runtime.setup_id,
            "event_id": runtime.event_id,
            "confirmation_id": runtime.confirmation_id,
            "side": direction.name,
            "trigger_type": trigger_type,
            "trigger_timeframe": trigger_timeframe,
            "trigger_close": candle.close,
            "trigger_atr_15m": impulse.atr_value,
            "structural_stop_price": structural_stop,
            "breakout_level": impulse.breakout_level,
            "impulse_time": impulse.candle.timestamp.isoformat(),
            "impulse_high": impulse.impulse_high,
            "impulse_low": impulse.impulse_low,
            "impulse_quality": impulse.quality_score,
            "pullback_time": runtime.pullback.candle.timestamp.isoformat() if runtime.pullback else None,
            "pullback_extreme": runtime.pullback.low if runtime.pullback else None,
            "pullback_low": (
                runtime.pullback.low if runtime.pullback and direction == Direction.LONG else None
            ),
            "pullback_high": (
                runtime.pullback.low if runtime.pullback and direction == Direction.SHORT else None
            ),
            "pullback_quality": runtime.pullback.quality_score if runtime.pullback else None,
            "rank_percentile": rank["percentile"],
            "rank_score": rank["score"],
            "extension_atr_1h": rank["extension_atr"],
            "confirmation_quality": confirmation_quality,
            "take_profit_1_r": float(self.mtpc.exit.take_profit_1_r),
            "take_profit_2_r": float(self.mtpc.exit.take_profit_2_r),
            **trend_features,
        }
        return EntryCandidate(
            symbol=symbol,
            signal=signal,
            candle=candle,
            rank_score=quality,
            directional_momentum_pct=rank["return_4h_pct"],
            volume_ratio=impulse.volume_ratio,
            filter_reason="mtpc_ok",
            metadata=metadata,
        )

    def _symbol_trend(
        self,
        symbol: str,
        decision_time: datetime,
        direction: Direction = Direction.LONG,
    ) -> tuple[bool, dict[str, float]]:
        cfg = self.mtpc.regime
        c1 = self._closed("1h", symbol, decision_time, 120)
        cache_key = (symbol, direction)
        if c1:
            cached = self._symbol_trend_cache.get(cache_key)
            if cached is not None and cached[0] == c1[-1].timestamp:
                return cached[1], dict(cached[2])
        c4 = self._closed("4h", symbol, decision_time, 100)
        if len(c4) < 70 or len(c1) < 70:
            return False, {}
        close4 = [item.close for item in c4]
        close1 = [item.close for item in c1]
        e4_fast = ema(close4, int(cfg.ema_fast_period))
        e4_slow = ema(close4, int(cfg.ema_slow_period))
        e1_fast = ema(close1, int(cfg.ema_fast_period))
        e1_slow = ema(close1, int(cfg.ema_slow_period))
        a4 = max(atr(c4, 14)[-1], 1e-12)
        a1 = max(atr(c1, 14)[-1], 1e-12)
        lookback = max(1, int(cfg.ema_slope_lookback_bars))
        slope4 = (e4_fast[-1] - e4_fast[-1 - lookback]) / a4
        slope1 = (e1_fast[-1] - e1_fast[-1 - lookback]) / a1
        gap1 = (e1_fast[-1] - e1_slow[-1]) / a1
        _, _, hist1 = macd(close1)
        if direction == Direction.LONG:
            valid = (
                close4[-1] > e4_fast[-1] > e4_slow[-1]
                and float(cfg.min_4h_fast_slope_atr) <= slope4 <= float(cfg.max_4h_fast_slope_atr)
                and close1[-1] > e1_fast[-1] > e1_slow[-1]
                and float(cfg.min_1h_fast_slope_atr) <= slope1 <= float(cfg.max_1h_fast_slope_atr)
                and float(cfg.min_1h_ema_gap_atr) <= gap1 <= float(cfg.max_1h_ema_gap_atr)
                and hist1[-1] >= 0
            )
        else:
            valid = (
                close4[-1] < e4_fast[-1] < e4_slow[-1]
                and -float(cfg.max_4h_fast_slope_atr) <= slope4 <= -float(cfg.min_4h_fast_slope_atr)
                and close1[-1] < e1_fast[-1] < e1_slow[-1]
                and -float(cfg.max_1h_fast_slope_atr) <= slope1 <= -float(cfg.min_1h_fast_slope_atr)
                and -float(cfg.max_1h_ema_gap_atr) <= gap1 <= -float(cfg.min_1h_ema_gap_atr)
                and hist1[-1] <= 0
            )
        features = {
            "four_hour_ema21_slope_atr": slope4,
            "one_hour_ema21_slope_atr": slope1,
            "one_hour_ema_gap_atr": gap1,
            "one_hour_macd_hist_atr": hist1[-1] / a1,
            "trend_direction": direction.name,
        }
        self._symbol_trend_cache[cache_key] = (c1[-1].timestamp, valid, dict(features))
        return valid, features

    def _update_regime(self, decision_time: datetime) -> MtpcRegime:
        btc = self._closed("1h", self.mtpc.regime.btc_symbol, decision_time, 1)
        if not btc:
            return self._regime
        bar = btc[-1].timestamp
        if bar == self._last_regime_bar:
            return self._regime
        self._last_regime_bar = bar
        raw = self._raw_market_regime(decision_time)
        if raw == MtpcRegime.CHAOS_NO_TRADE:
            self._raw_regime = raw
            self._raw_regime_count = 1
            self._regime = raw
            self._regime_enter_time = decision_time
            return self._regime
        if raw == self._raw_regime:
            self._raw_regime_count += 1
        else:
            self._raw_regime = raw
            self._raw_regime_count = 1
        if raw == self._regime:
            return self._regime
        held_bars = 10**9
        if self._regime_enter_time is not None:
            held_bars = int((decision_time - self._regime_enter_time).total_seconds() // 3600)
        minimum_hold = max(0, int(self.mtpc.regime.min_state_hold_bars_1h))
        if self._regime in {MtpcRegime.BULL_TREND, MtpcRegime.BEAR_TREND}:
            required = max(1, int(self.mtpc.regime.exit_confirmation_bars_1h))
        else:
            required = max(1, int(self.mtpc.regime.enter_confirmation_bars_1h))
        if self._raw_regime_count >= required and held_bars >= minimum_hold:
            self._regime = raw
            self._regime_enter_time = decision_time
        return self._regime

    def _raw_market_regime(self, decision_time: datetime) -> MtpcRegime:
        cfg = self.mtpc.regime
        btc1 = self._closed("1h", cfg.btc_symbol, decision_time, 140)
        btc4 = self._closed("4h", cfg.btc_symbol, decision_time, 100)
        eth1 = self._closed("1h", cfg.eth_symbol, decision_time, 100)
        eth4 = self._closed("4h", cfg.eth_symbol, decision_time, 80)
        if min(len(btc1), len(eth1)) < 70 or min(len(btc4), len(eth4)) < 60:
            return MtpcRegime.NEUTRAL
        btc_return = btc1[-1].close / max(btc1[-2].close, 1e-12) - 1.0
        btc_atr_values = atr(btc1, 14)
        atr_percentile = _percentile_rank(btc_atr_values[-1], btc_atr_values[-100:])
        if abs(btc_return) >= float(cfg.btc_shock_1h_pct) or atr_percentile >= float(cfg.chaos_atr_percentile):
            return MtpcRegime.CHAOS_NO_TRADE
        btc1_close = [item.close for item in btc1]
        btc4_close = [item.close for item in btc4]
        eth1_close = [item.close for item in eth1]
        eth4_close = [item.close for item in eth4]
        b1_fast = ema(btc1_close, int(cfg.ema_fast_period))
        b1_slow = ema(btc1_close, int(cfg.ema_slow_period))
        b4_fast = ema(btc4_close, int(cfg.ema_fast_period))
        e1_fast = ema(eth1_close, int(cfg.ema_fast_period))
        e1_slow = ema(eth1_close, int(cfg.ema_slow_period))
        e4_fast = ema(eth4_close, int(cfg.ema_fast_period))
        lookback = max(1, int(cfg.ema_slope_lookback_bars))
        btc4_slope = (b4_fast[-1] - b4_fast[-1 - lookback]) / max(atr(btc4, 14)[-1], 1e-12)
        btc_aligned = (
            btc1_close[-1] > b1_fast[-1] > b1_slow[-1]
            and btc4_close[-1] > b4_fast[-1]
            and btc4_slope >= float(cfg.min_4h_fast_slope_atr)
        )
        eth_aligned = eth1_close[-1] > e1_fast[-1] > e1_slow[-1] and eth4_close[-1] > e4_fast[-1]
        btc_bear_aligned = (
            btc1_close[-1] < b1_fast[-1] < b1_slow[-1]
            and btc4_close[-1] < b4_fast[-1]
            and btc4_slope <= -float(cfg.min_4h_fast_slope_atr)
        )
        eth_bear_aligned = (
            eth1_close[-1] < e1_fast[-1] < e1_slow[-1]
            and eth4_close[-1] < e4_fast[-1]
        )
        above = 0
        positive = 0
        total = 0
        for symbol in self.symbols:
            rows = self._closed("1h", symbol, decision_time, 60)
            if len(rows) < 30:
                continue
            closes = [item.close for item in rows]
            total += 1
            above += closes[-1] > ema(closes, int(cfg.ema_fast_period))[-1]
            positive += closes[-1] > closes[-2]
        breadth_above = above / total if total else 0.0
        breadth_positive = positive / total if total else 0.0
        aligned = btc_aligned and (eth_aligned or not bool(cfg.require_btc_eth_alignment))
        if (
            aligned
            and breadth_above >= float(cfg.min_breadth_above_ema21)
            and breadth_positive >= float(cfg.min_breadth_positive_1h)
            and breadth_above <= float(cfg.max_breadth_overheated)
        ):
            return MtpcRegime.BULL_TREND
        bear_aligned = btc_bear_aligned and (
            eth_bear_aligned or not bool(cfg.require_btc_eth_alignment)
        )
        if (
            bear_aligned
            and breadth_above <= 1.0 - float(cfg.min_breadth_above_ema21)
            and breadth_positive <= 1.0 - float(cfg.min_breadth_positive_1h)
            and breadth_above >= 1.0 - float(cfg.max_breadth_overheated)
        ):
            return MtpcRegime.BEAR_TREND
        return MtpcRegime.NEUTRAL

    def _rankings(self, decision_time: datetime) -> dict[str, dict[str, float]]:
        btc = self._closed("1h", self.mtpc.regime.btc_symbol, decision_time, 80)
        if not btc:
            return {}
        bar = btc[-1].timestamp
        if bar == self._ranking_bar:
            return self._ranking_cache
        self._ranking_bar = bar
        cfg = self.mtpc.ranking
        btc_return_4h = btc[-1].close / max(btc[-5].close, 1e-12) - 1.0 if len(btc) >= 5 else 0.0
        raw: dict[str, dict[str, float]] = {}
        for symbol in self.symbols:
            candles = self._closed("1h", symbol, decision_time, 100)
            if len(candles) < 70:
                continue
            closes = [item.close for item in candles]
            atr_value = max(atr(candles, 14)[-1], 1e-12)
            ret4_pct = closes[-1] / max(closes[-5], 1e-12) - 1.0
            ret12_pct = closes[-1] / max(closes[-13], 1e-12) - 1.0
            ret4 = (closes[-1] - closes[-5]) / atr_value
            ret12 = (closes[-1] - closes[-13]) / atr_value
            relative = (ret4_pct - btc_return_4h) / max(atr_value / closes[-1], 1e-12)
            e21 = ema(closes, 21)
            e55 = ema(closes, 55)
            gap = (e21[-1] - e55[-1]) / atr_value
            _, _, hist = macd(closes)
            hist_score = hist[-1] / atr_value
            score = (
                float(cfg.return_4h_weight) * ret4
                + float(cfg.return_12h_weight) * ret12
                + float(cfg.relative_btc_4h_weight) * relative
                + float(cfg.ema_alignment_weight) * gap
                + float(cfg.macd_weight) * hist_score
            )
            raw[symbol] = {
                "score": score,
                "return_4h_pct": ret4_pct,
                "return_12h_pct": ret12_pct,
                "relative_btc_4h": ret4_pct - btc_return_4h,
                "extension_atr": (closes[-1] - e21[-1]) / atr_value,
            }
        ordered = sorted((values["score"], symbol) for symbol, values in raw.items())
        denominator = max(1, len(ordered) - 1)
        for index, (_, symbol) in enumerate(ordered):
            raw[symbol]["percentile"] = index / denominator
        self._ranking_cache = raw
        return raw

    @staticmethod
    def _directional_rank(rank: dict[str, float], direction: Direction) -> dict[str, float]:
        if direction == Direction.LONG:
            return dict(rank)
        directional = dict(rank)
        directional["score"] = -float(rank["score"])
        directional["percentile"] = 1.0 - float(rank["percentile"])
        directional["return_4h_pct"] = -float(rank["return_4h_pct"])
        directional["return_12h_pct"] = -float(rank["return_12h_pct"])
        directional["relative_btc_4h"] = -float(rank["relative_btc_4h"])
        directional["extension_atr"] = -float(rank["extension_atr"])
        return directional

    def _invalidate(self, symbol: str, timestamp: datetime, reason: str) -> None:
        self.reject_reasons[reason] += 1
        self._reset_with_cooldown(
            symbol,
            timestamp,
            reason,
            max(0, int(self.mtpc.risk_control.setup_invalidation_cooldown_minutes)),
        )

    def _transition(self, symbol: str, state: MtpcState, timestamp: datetime, reason: str) -> None:
        runtime = self.runtime[symbol]
        if runtime.state == state:
            return
        self.state_transitions.append(
            {
                "symbol": symbol,
                "setup_id": runtime.setup_id,
                "event_id": runtime.event_id,
                "confirmation_id": runtime.confirmation_id,
                "from_state": runtime.state.value,
                "to_state": state.value,
                "time": timestamp.isoformat(),
                "reason": reason,
            }
        )
        runtime.state = state

    def _event_row(
        self,
        symbol: str,
        runtime: MtpcSymbolRuntime,
        impulse: MtpcImpulseSnapshot,
        rank: dict[str, float],
        trend: dict[str, float],
        decision_time: datetime,
    ) -> dict[str, Any]:
        return {
            "setup_id": runtime.setup_id,
            "event_id": runtime.event_id,
            "symbol": symbol,
            "side": (runtime.direction or impulse.direction).name,
            "setup_time": decision_time.isoformat(),
            "impulse_candle_time": impulse.candle.timestamp.isoformat(),
            "breakout_level": impulse.breakout_level,
            "breakout_distance_atr": impulse.breakout_distance_atr,
            "body_atr": impulse.body_atr,
            "close_position": impulse.close_position,
            "upper_wick_ratio": impulse.upper_wick_ratio,
            "adverse_wick_ratio": impulse.upper_wick_ratio,
            "volume_ratio": impulse.volume_ratio,
            "prior_move_atr": impulse.prior_move_atr,
            "ema21_extension_atr": impulse.ema21_extension_atr,
            "impulse_quality": impulse.quality_score,
            "rank_percentile": rank["percentile"],
            "rank_score": rank["score"],
            **trend,
        }

    def _annotate_event(self, event_id: str | None, **values: Any) -> None:
        if not event_id:
            return
        for row in reversed(self.event_rows):
            if row.get("event_id") == event_id:
                row.update(values)
                return

    def _closed(self, timeframe: str, symbol: str, decision_time: datetime, limit: int) -> list[Candle]:
        candles = self.candles.get(timeframe, {}).get(symbol, [])
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        available_before = decision_time - timedelta(milliseconds=interval_to_milliseconds(timeframe))
        end = bisect.bisect_right(timestamps, available_before)
        return candles[max(0, end - limit):end]


def mtpc_strategy_config_hash(config: Any) -> str:
    encoded = json.dumps(
        asdict(config.mtpc),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_variant(value: Any) -> str:
    normalized = str(value or "first_pullback").strip().lower()
    if normalized not in {"trend_impulse", "first_pullback", "trend_ema_pullback"}:
        raise ValueError(f"unsupported MTPC stage_variant: {value}")
    return normalized


def _percentile_rank(value: float, values: list[float]) -> float:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return 0.5
    below = sum(item < value for item in finite)
    equal = sum(item == value for item in finite)
    return (below + 0.5 * equal) / len(finite)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
