from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .cmipr_campaign_diagnostics import CmiprCampaignDiagnostics
from .data import interval_to_milliseconds
from .indicators import atr, ema, macd
from .live_trader import EntryCandidate
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import funding_at, oi_change_at


CMIPR_REASON_TOKEN = "cross_sectional_momentum_ignition_pyramid"
CMIPR_FEATURE_VERSION = "cmipr_features_v2_5"
CMIPR_EVENT_DEFINITION_VERSION = "cmipr_first_pullback_event_v2_5"


def cmipr_strategy_config_hash(config: Any) -> str:
    payload = {
        "cmipr": asdict(config.cmipr),
        "risk": asdict(config.risk),
        "trading_symbols": list(config.trading.symbols),
        "entry_symbols": list(config.trading.entry_symbols),
        "trading_timeframe": config.trading.timeframe,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CmiprMarketRegime(str, Enum):
    EARLY_BULL_EXPANSION = "EARLY_BULL_EXPANSION"
    MATURE_BULL_EXPANSION = "MATURE_BULL_EXPANSION"
    BULL_EXPANSION = "BULL_EXPANSION"
    BEAR_EXPANSION = "BEAR_EXPANSION"
    NEUTRAL = "NEUTRAL"
    OVERHEATED_BULL = "OVERHEATED_BULL"
    OVEREXTENDED_BEAR = "OVEREXTENDED_BEAR"
    CHAOS_NO_TRADE = "CHAOS_NO_TRADE"


_BULL_REGIMES = frozenset(
    {
        CmiprMarketRegime.EARLY_BULL_EXPANSION,
        CmiprMarketRegime.MATURE_BULL_EXPANSION,
        CmiprMarketRegime.BULL_EXPANSION,
    }
)
_BEAR_REGIMES = frozenset({CmiprMarketRegime.BEAR_EXPANSION})


class CmiprState(str, Enum):
    IDLE = "IDLE"
    COMPRESSION_WATCH = "COMPRESSION_WATCH"
    LONG_IGNITION_PENDING = "LONG_IGNITION_PENDING"
    SHORT_IGNITION_PENDING = "SHORT_IGNITION_PENDING"
    ENTRY_CONFIRMING = "ENTRY_CONFIRMING"
    BULL_FLAG_PENDING = "BULL_FLAG_PENDING"
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
    breadth_acceleration_1h: float = 0.0
    market_return_1h: float = 0.0
    market_return_acceleration_1h: float = 0.0


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
    setup_atr_value: float
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
    event_type: str = "IGNITION"
    stale_parent_event_id: str | None = None
    stale_at: datetime | None = None
    recompression_generation: int = 0
    ignition_breadth: float = 0.0
    ignition_rank_percentile: float = 0.0
    ignition_relative_strength: float = 0.0


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
        self.strategy_config_hash = cmipr_strategy_config_hash(config)
        self.cache_namespace = f"cmipr:{self.strategy_config_hash}"
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
        self.campaign_diagnostics = CmiprCampaignDiagnostics(
            enabled=bool(self.cmipr.research.event_diagnostics_enabled),
            minimum_bucket_size=max(1, int(self.cmipr.research.event_diagnostic_min_bucket_size)),
            full_cost_pct=_configured_full_cost_pct(config),
            ignition_timeframe=str(self.cmipr.ignition.timeframe),
            pullback_timeframe=str(self.cmipr.entry.pullback_timeframe),
            campaign_risk_budget_usdt=_diagnostic_campaign_risk_budget(config),
            initial_risk_fraction=float(self.cmipr.entry.initial_risk_fraction),
            max_shadow_notional_usdt=_diagnostic_max_notional(config),
            min_order_notional_usdt=float(config.risk.min_order_notional_usdt),
            fixed_take_profit_r=float(self.cmipr.exit.fixed_take_profit_r),
            take_profit_r_basis=str(self.cmipr.exit.take_profit_r_basis),
            max_holding_minutes=int(self.cmipr.exit.max_holding_minutes),
        )

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
            base_event_id = f"{symbol}:{ignition.direction.name}:{ignition.candle.timestamp.isoformat()}"
            event_id = base_event_id
            event_type = "IGNITION"
            parent_event_id: str | None = None
            if runtime.stale_parent_event_id is not None and bool(self.cmipr.entry.delayed_recompression_enabled):
                minimum_new_structure_time = (runtime.stale_at or decision_time) + timedelta(minutes=30)
                ignition_available = ignition.candle.timestamp + timedelta(
                    milliseconds=interval_to_milliseconds(self.cmipr.ignition.timeframe)
                )
                if ignition_available < minimum_new_structure_time:
                    self.reject_reasons["new_event_required"] += 1
                    continue
                runtime.recompression_generation += 1
                parent_event_id = runtime.stale_parent_event_id
                event_id = f"{base_event_id}:RECOMP{runtime.recompression_generation}"
                event_type = "DELAYED_RECOMPRESSION"
            compression, _ = self._compression(symbol, ignition.candle.timestamp)
            if compression is not None:
                previous_time = decision_time - timedelta(hours=1)
                previous_regime = self._raw_regime_snapshot(previous_time)
                previous_rank = self.rankings(previous_time).get(symbol)
                self.campaign_diagnostics.record_ignition(
                    event_id,
                    symbol,
                    decision_time,
                    regime,
                    previous_regime,
                    ignition,
                    compression,
                    previous_rank,
                    self._liquidity_percentile(symbol, decision_time),
                    max(1, int(self.cmipr.compression.lookback_bars_30m)),
                )
                self.campaign_diagnostics.mark(
                    event_id,
                    "ignition_observed",
                    event_type=event_type,
                    parent_event_id=parent_event_id,
                    recompression_uses_new_atr=event_type == "DELAYED_RECOMPRESSION",
                    recompression_uses_new_breakout_level=event_type == "DELAYED_RECOMPRESSION",
                )
            if event_id in runtime.consumed_event_ids:
                continue
            runtime.event_id = event_id
            runtime.direction = ignition.direction
            runtime.ignition = ignition
            runtime.pending_time = decision_time
            expiry_minutes = max(1, int(self.cmipr.entry.pending_expiry_minutes))
            if _event_structure_mode(self.cmipr.entry.event_structure_mode) != "current":
                expiry_minutes = min(expiry_minutes, max(1, int(self.cmipr.entry.immediate_max_age_minutes)))
            runtime.expire_time = decision_time + timedelta(minutes=expiry_minutes)
            runtime.event_type = event_type
            runtime.ignition_breadth = regime.breadth_above_ema21
            runtime.ignition_rank_percentile = rank[1]
            runtime.ignition_relative_strength = rank[0]
            if event_type == "DELAYED_RECOMPRESSION":
                runtime.stale_parent_event_id = None
                runtime.stale_at = None
                self.stats["delayed_recompression_count"] += 1
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
            if self._active_regime in _BULL_REGIMES | _BEAR_REGIMES:
                same_direction = (
                    self._active_regime in _BULL_REGIMES and snapshot.raw_state in _BULL_REGIMES
                ) or (
                    self._active_regime in _BEAR_REGIMES and snapshot.raw_state in _BEAR_REGIMES
                )
                if same_direction:
                    self._exit_count = 0
                    if (
                        snapshot.raw_state != self._active_regime
                        and self._raw_regime_count >= max(1, int(self.cmipr.regime.enter_confirmation_bars_1h))
                        and held >= min_hold
                    ):
                        self._active_regime = snapshot.raw_state
                        self._active_since_bar = latest_bar
                else:
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
            breadth_acceleration_1h=snapshot.breadth_acceleration_1h,
            market_return_1h=snapshot.market_return_1h,
            market_return_acceleration_1h=snapshot.market_return_acceleration_1h,
        )

    def rankings(self, decision_time: datetime) -> dict[str, tuple[float, float, float]]:
        cache_key = (self.cache_namespace, "rankings", self.cmipr.ranking, decision_time)
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
            "strategy_config_hash": self.strategy_config_hash,
            "feature_version": CMIPR_FEATURE_VERSION,
            "event_definition_version": CMIPR_EVENT_DEFINITION_VERSION,
            "cache_namespace": self.cache_namespace,
            "stats": dict(sorted(self.stats.items())),
            "reject_reasons": dict(sorted(self.reject_reasons.items())),
            "regime_observations": dict(sorted(self.regime_history.items())),
            "states": dict(sorted(Counter(runtime.state.value for runtime in self.runtime.values()).items())),
        }

    def finalize_campaign_diagnostics(
        self,
        execution_candles_by_symbol: dict[str, list[Candle]],
        trades: list[dict[str, Any]],
        execution_config: Any = None,
        rules_by_symbol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.campaign_diagnostics.finalize(
            execution_candles_by_symbol,
            self.candles,
            trades,
            execution_config,
            rules_by_symbol,
        )

    def _liquidity_percentile(self, symbol: str, decision_time: datetime) -> float:
        cache_key = (self.cache_namespace, "liquidity_percentile", decision_time)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is None:
            quote_volume: dict[str, float] = {}
            for candidate in self.symbols:
                candles = self._closed("30m", candidate, decision_time, 4)
                if candles:
                    quote_volume[candidate] = sum(item.volume * item.close for item in candles) / len(candles)
            ordered = sorted(quote_volume, key=lambda item: (quote_volume[item], item))
            denominator = max(1, len(ordered) - 1)
            cached = {item: index / denominator for index, item in enumerate(ordered)}
            self.shared_feature_cache[cache_key] = cached
        return float(cached.get(symbol, 0.0))

    def _raw_regime_snapshot(self, decision_time: datetime) -> CmiprRegimeSnapshot:
        cache_key = (self.cache_namespace, "raw_regime", self.cmipr.regime, decision_time)
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
        previous_above = 0
        eligible = 0
        market_returns: list[float] = []
        previous_market_returns: list[float] = []
        for symbol in self.symbols:
            candles = self._closed("1h", symbol, decision_time, cfg.ema_period + 3)
            if len(candles) < cfg.ema_period + 2:
                continue
            eligible += 1
            current_return = _period_return(candles, 1)
            previous_return = _period_return(candles[:-1], 1)
            closes = [item.close for item in candles]
            positives += int(current_return > 0)
            above += int(closes[-1] > ema(closes, cfg.ema_period)[-1])
            previous_above += int(closes[-2] > ema(closes[:-1], cfg.ema_period)[-1])
            market_returns.append(current_return)
            previous_market_returns.append(previous_return)
        breadth_positive = positives / max(eligible, 1)
        breadth_above = above / max(eligible, 1)
        previous_breadth_above = previous_above / max(eligible, 1)
        breadth_acceleration = breadth_above - previous_breadth_above
        market_return = sum(market_returns) / max(len(market_returns), 1)
        previous_market_return = sum(previous_market_returns) / max(len(previous_market_returns), 1)
        market_return_acceleration = market_return - previous_market_return
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
            if not cfg.phase_model_enabled:
                raw = CmiprMarketRegime.BULL_EXPANSION
            elif (
                breadth_acceleration >= cfg.early_min_breadth_acceleration_1h
                and breadth_above <= cfg.early_max_breadth_above_ema21
                and btc_1h_return <= cfg.early_max_btc_return_1h
                and eth_1h_return >= cfg.early_min_eth_return_1h
            ):
                raw = CmiprMarketRegime.EARLY_BULL_EXPANSION
            else:
                raw = CmiprMarketRegime.MATURE_BULL_EXPANSION
        elif btc_closes[-1] < btc_ema[-1] and slope <= -cfg.enter_ema_slope_pct and breadth_above <= 1.0 - cfg.enter_breadth_above_ema21 and breadth_positive <= 1.0 - cfg.min_breadth_positive_1h:
            raw = CmiprMarketRegime.BEAR_EXPANSION
        else:
            raw = CmiprMarketRegime.NEUTRAL
        result = CmiprRegimeSnapshot(
            raw,
            btc_1h_return,
            btc_4h_return,
            eth_1h_return,
            breadth_positive,
            breadth_above,
            raw,
            0,
            breadth_acceleration,
            market_return,
            market_return_acceleration,
        )
        self.shared_feature_cache[cache_key] = result
        return result

    def _maintain_active_regime(self, snapshot: CmiprRegimeSnapshot) -> bool:
        cfg = self.cmipr.regime
        btc = self._closed("1h", cfg.btc_symbol, self._last_regime_bar + timedelta(hours=1), cfg.ema_period + cfg.ema_slope_lookback + 2) if self._last_regime_bar else []
        if len(btc) < cfg.ema_period + cfg.ema_slope_lookback:
            return False
        values = ema([item.close for item in btc], cfg.ema_period)
        slope = values[-1] / max(values[-1 - cfg.ema_slope_lookback], 1e-12) - 1.0
        if self._active_regime in _BULL_REGIMES:
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
        if cfg.regime.phase_model_enabled:
            allowed_regimes = {CmiprMarketRegime.EARLY_BULL_EXPANSION, CmiprMarketRegime.BEAR_EXPANSION}
        else:
            allowed_regimes = {CmiprMarketRegime.BULL_EXPANSION, CmiprMarketRegime.BEAR_EXPANSION}
        if regime.state not in allowed_regimes:
            return None, "regime_not_expansion"
        direction = Direction.LONG if regime.state in _BULL_REGIMES else Direction.SHORT
        if direction == Direction.LONG and not cfg.allow_long:
            return None, "long_disabled"
        if direction == Direction.SHORT and not cfg.allow_short:
            return None, "short_disabled"
        score, percentile, extension = rank
        if direction == Direction.LONG and percentile < 1.0 - cfg.ranking.long_top_fraction:
            return None, "ranking_not_top"
        if direction == Direction.SHORT and percentile > cfg.ranking.short_bottom_fraction:
            return None, "ranking_not_bottom"
        if cfg.ranking.strength_acceleration_enabled:
            lookback = max(1, int(cfg.ranking.strength_acceleration_lookback_hours))
            previous_rank = self.rankings(decision_time - timedelta(hours=lookback)).get(symbol)
            if previous_rank is None:
                return None, "ranking_acceleration_warmup"
            acceleration = percentile - previous_rank[1]
            if direction == Direction.LONG and acceleration < cfg.ranking.min_long_strength_acceleration:
                return None, "ranking_acceleration_too_low"
            if direction == Direction.SHORT and acceleration > cfg.ranking.max_short_strength_acceleration:
                return None, "ranking_acceleration_too_high"
        if extension > cfg.ranking.max_extension_atr:
            return None, "ranking_overextended"
        ignition_timeframe = str(cfg.ignition.timeframe)
        candles = self._closed(ignition_timeframe, symbol, decision_time, max(50, cfg.ignition.breakout_lookback_15m + 5))
        if len(candles) < cfg.ignition.breakout_lookback_15m + 2:
            return None, "ignition_warmup"
        latest = candles[-1]
        # Compression must be known before the ignition candle starts. Using
        # decision_time here would include the 30m bar containing the breakout.
        compression, reason = self._compression(symbol, latest.timestamp)
        if compression is None:
            return None, reason
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
        cost_guard_target_r = float(cfg.entry.cost_guard_target_r)
        if cost_guard_target_r <= 0:
            cost_guard_target_r = max(1.2, float(cfg.exit.runner_activation_r))
        target_pct = stop_pct * cost_guard_target_r
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
            setup_atr_value=compression.atr_value,
            stop_price=stop,
            stop_loss_pct=stop_pct,
            take_profit_pct=max(stop_pct * 20.0, 0.25),
        ), ""

    def _compression(self, symbol: str, decision_time: datetime) -> tuple[CmiprCompressionSnapshot | None, str]:
        cache_key = (self.cache_namespace, "compression", self.cmipr.compression, symbol, decision_time)
        cached = self.shared_feature_cache.get(cache_key)
        if cached is not None:
            return cached
        cfg = self.cmipr.compression
        candles = self._closed(str(cfg.timeframe), symbol, decision_time, max(cfg.atr_percentile_lookback + cfg.atr_period + 5, 120))
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
        if runtime.state not in (
            CmiprState.LONG_IGNITION_PENDING,
            CmiprState.SHORT_IGNITION_PENDING,
            CmiprState.ENTRY_CONFIRMING,
            CmiprState.BULL_FLAG_PENDING,
        ):
            return None
        if runtime.expire_time is not None and decision_time > runtime.expire_time:
            mode = _event_structure_mode(self.cmipr.entry.event_structure_mode)
            if mode == "current":
                self.campaign_diagnostics.mark(runtime.event_id, "pending_expired")
                self.stats["pending_expired"] += 1
                self._clear_pending(runtime)
            else:
                self._expire_stale_pending(runtime, decision_time)
            return None
        ignition = runtime.ignition
        if ignition is None or runtime.event_id is None:
            runtime.state = CmiprState.IDLE
            return None
        regime_direction_valid = (
            ignition.direction == Direction.LONG and regime.state in _BULL_REGIMES
        ) or (
            ignition.direction == Direction.SHORT and regime.state in _BEAR_REGIMES
        )
        if not regime_direction_valid:
            self.campaign_diagnostics.mark(runtime.event_id, "pending_regime_invalidated")
            if _event_structure_mode(self.cmipr.entry.event_structure_mode) != "current" and runtime.event_id is not None:
                runtime.consumed_event_ids.add(runtime.event_id)
            self._clear_pending(runtime)
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
        atr_price = ignition.setup_atr_value
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
        stop_atr = ignition.direction.value * (latest.close - stop) / max(atr_price, 1e-12)
        current_rank = self.rankings(decision_time).get(symbol)
        entry_rank = current_rank[1] if current_rank is not None else None
        entry_extension = current_rank[2] if current_rank is not None else None
        confirmation_body_atr = abs(latest.close - latest.open) / max(atr_price, 1e-12)
        if ignition.direction == Direction.LONG:
            confirmation_wick = latest.high - max(latest.open, latest.close)
        else:
            confirmation_wick = min(latest.open, latest.close) - latest.low
        confirmation_wick_ratio = max(0.0, confirmation_wick / max(latest.high - latest.low, 1e-12))
        stop_loss_pct = abs(latest.close - stop) / max(latest.close, 1e-12)
        cost_target_r = float(self.cmipr.entry.cost_guard_target_r)
        if cost_target_r <= 0.0:
            cost_target_r = max(1.2, float(self.cmipr.exit.runner_activation_r))
        target_to_cost = stop_loss_pct * cost_target_r / max(_configured_full_cost_pct(self.config), 1e-12)
        self.campaign_diagnostics.update_pullback(
            runtime.event_id,
            latest,
            depth,
            normalized_volume,
            close_position,
            chase,
            confirmed,
            decision_time=decision_time,
            pullback_bars=len(after),
            pullback_low=extreme,
            confirmation_body_atr=confirmation_body_atr,
            confirmation_wick_ratio=confirmation_wick_ratio,
            stop_price=stop,
            stop_distance_atr=stop_atr,
            target_to_cost_ratio=target_to_cost,
            entry_regime=regime.state.value,
            entry_breadth=regime.breadth_above_ema21,
            entry_rank=entry_rank,
            entry_quality=ignition.quality_score,
            entry_extension=entry_extension,
        )
        entry_mode = str(self.cmipr.entry.mode).strip().lower()
        bull_flag_enabled = entry_mode in ("bull_flag", "first_pullback_or_bull_flag")
        if bull_flag_enabled:
            bull_flag = self._bull_flag_candidate(symbol, decision_time, runtime, ignition, after)
            if bull_flag is not None:
                return bull_flag
            if entry_mode == "bull_flag":
                runtime.state = CmiprState.BULL_FLAG_PENDING
                return None
        if not self.cmipr.entry.pullback_min_depth_atr <= depth <= self.cmipr.entry.pullback_max_depth_atr:
            return None
        if normalized_volume > self.cmipr.entry.pullback_max_volume_ratio:
            return None
        if close_position < self.cmipr.entry.confirmation_min_close_position or not confirmed:
            return None
        if chase > self.cmipr.entry.max_chase_distance_atr:
            self.reject_reasons["entry_chase"] += 1
            self.campaign_diagnostics.mark(runtime.event_id, "entry_chase_rejected")
            if _event_structure_mode(self.cmipr.entry.event_structure_mode) != "current" and not bull_flag_enabled:
                if runtime.event_id is not None:
                    runtime.consumed_event_ids.add(runtime.event_id)
                self._clear_pending(runtime)
            else:
                runtime.state = CmiprState.BULL_FLAG_PENDING if bull_flag_enabled else CmiprState.IDLE
            return None
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
            setup_atr_value=ignition.setup_atr_value,
            stop_price=stop,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=ignition.take_profit_pct,
        )
        revalidation_reason = self._entry_revalidation_reason(
            runtime,
            regime,
            current_rank,
            latest,
            ignition,
            target_to_cost,
            decision_time,
        )
        if revalidation_reason:
            self.reject_reasons[revalidation_reason] += 1
            self.campaign_diagnostics.mark(
                runtime.event_id,
                "entry_revalidation_rejected",
                reject_reason=revalidation_reason,
            )
            if runtime.event_id is not None:
                runtime.consumed_event_ids.add(runtime.event_id)
            self._clear_pending(runtime)
            return None
        if _event_structure_mode(self.cmipr.entry.event_structure_mode) == "delayed_recompression_only" and runtime.event_type != "DELAYED_RECOMPRESSION":
            self.campaign_diagnostics.mark(
                runtime.event_id,
                "waiting_for_delayed_recompression",
                reject_reason="new_event_required",
            )
            self.reject_reasons["new_event_required"] += 1
            return None
        candidate = self._candidate_from_ignition(symbol, adjusted, runtime.event_id, "first_pullback")
        runtime.state = CmiprState.INITIAL_ENTRY_READY
        runtime.consumed_event_ids.add(runtime.event_id)
        self.stats["pullback_confirmed"] += 1
        return candidate

    def _entry_revalidation_reason(
        self,
        runtime: CmiprSymbolRuntime,
        regime: CmiprRegimeSnapshot,
        current_rank: tuple[float, float, float] | None,
        latest: Candle,
        ignition: CmiprIgnitionSnapshot,
        target_to_cost: float,
        decision_time: datetime,
    ) -> str:
        mode = str(self.cmipr.entry.entry_revalidation_mode or "none").strip().lower()
        if mode == "none":
            return ""
        if mode not in {"basic", "strict"}:
            raise ValueError(f"unsupported CMIPR entry_revalidation_mode: {self.cmipr.entry.entry_revalidation_mode}")
        age_minutes = (
            (decision_time - runtime.pending_time).total_seconds() / 60.0
            if runtime.pending_time is not None else float("inf")
        )
        if age_minutes > max(1, int(self.cmipr.entry.immediate_max_age_minutes)):
            return "stale_ignition_event"
        direction_valid = (
            ignition.direction == Direction.LONG and regime.state in _BULL_REGIMES
        ) or (
            ignition.direction == Direction.SHORT and regime.state in _BEAR_REGIMES
        )
        if not direction_valid:
            return "regime_deteriorated_before_entry"
        structure_valid = (
            latest.close >= ignition.breakout_level
            if ignition.direction == Direction.LONG
            else latest.close <= ignition.breakout_level
        )
        if not structure_valid:
            return "breakout_structure_invalidated"
        if target_to_cost < float(self.cmipr.entry.min_target_to_cost_ratio):
            return "target_to_cost_deteriorated"
        if mode == "basic":
            return ""
        if regime.breadth_above_ema21 < runtime.ignition_breadth - float(self.cmipr.entry.max_breadth_deterioration):
            return "breadth_deteriorated_before_entry"
        if current_rank is None:
            return "rank_deteriorated_before_entry"
        score, percentile, extension = current_rank
        if ignition.direction == Direction.LONG:
            if percentile < float(self.cmipr.entry.min_entry_rank_percentile):
                return "rank_deteriorated_before_entry"
            if runtime.ignition_rank_percentile - percentile > float(self.cmipr.entry.max_rank_deterioration):
                return "rank_deteriorated_before_entry"
            if score < runtime.ignition_relative_strength and percentile < runtime.ignition_rank_percentile:
                return "relative_strength_lost"
        else:
            if percentile > 1.0 - float(self.cmipr.entry.min_entry_rank_percentile):
                return "rank_deteriorated_before_entry"
            if percentile - runtime.ignition_rank_percentile > float(self.cmipr.entry.max_rank_deterioration):
                return "rank_deteriorated_before_entry"
            if score > runtime.ignition_relative_strength and percentile > runtime.ignition_rank_percentile:
                return "relative_strength_lost"
        if extension > float(self.cmipr.ranking.max_extension_atr):
            return "overextended_before_entry"
        return ""

    def _expire_stale_pending(self, runtime: CmiprSymbolRuntime, decision_time: datetime) -> None:
        stale_event_id = runtime.event_id
        if stale_event_id is not None:
            runtime.consumed_event_ids.add(stale_event_id)
            self.campaign_diagnostics.mark(
                stale_event_id,
                "stale_ignition_event",
                reject_reason="stale_ignition_event",
                stale_at=decision_time.isoformat(),
            )
        runtime.stale_parent_event_id = stale_event_id
        runtime.stale_at = decision_time
        self.stats["stale_event_reject_count"] += 1
        self.reject_reasons["stale_ignition_event"] += 1
        self._clear_pending(runtime, preserve_stale_parent=True)

    @staticmethod
    def _clear_pending(runtime: CmiprSymbolRuntime, preserve_stale_parent: bool = False) -> None:
        runtime.state = CmiprState.IDLE
        runtime.event_id = None
        runtime.direction = Direction.FLAT
        runtime.ignition = None
        runtime.pending_time = None
        runtime.expire_time = None
        runtime.last_processed_5m = None
        runtime.event_type = "IGNITION"
        runtime.ignition_breadth = 0.0
        runtime.ignition_rank_percentile = 0.0
        runtime.ignition_relative_strength = 0.0
        if not preserve_stale_parent:
            runtime.stale_parent_event_id = None
            runtime.stale_at = None

    def _bull_flag_candidate(
        self,
        symbol: str,
        decision_time: datetime,
        runtime: CmiprSymbolRuntime,
        ignition: CmiprIgnitionSnapshot,
        after: list[Candle],
    ) -> EntryCandidate | None:
        cfg = self.cmipr.entry
        minimum_bars = max(3, int(cfg.bull_flag_min_bars_5m))
        if len(after) < minimum_bars + 1 or runtime.event_id is None:
            runtime.state = CmiprState.BULL_FLAG_PENDING
            return None

        platform = after[-minimum_bars - 1:-1]
        latest = after[-1]
        atr_price = max(ignition.setup_atr_value, 1e-12)
        platform_high = max(item.high for item in platform)
        platform_low = min(item.low for item in platform)
        platform_range_atr = (platform_high - platform_low) / atr_price
        if platform_range_atr > cfg.bull_flag_max_range_atr:
            return None

        timeframe_ratio = interval_to_milliseconds("15m") / interval_to_milliseconds(cfg.pullback_timeframe)
        platform_volume = sum(item.volume for item in platform) / len(platform)
        volume_to_ignition = platform_volume * timeframe_ratio / max(ignition.candle.volume, 1e-12)
        if volume_to_ignition > cfg.bull_flag_max_volume_to_ignition:
            return None

        latest_range = max(latest.high - latest.low, 1e-12)
        breakout_volume_ratio = latest.volume / max(platform_volume, 1e-12)
        current_rank = self.rankings(decision_time).get(symbol)
        if current_rank is None:
            return None
        score, percentile, extension = current_rank
        if extension > self.cmipr.ranking.max_extension_atr:
            return None

        if ignition.direction == Direction.LONG:
            rank_valid = percentile >= 1.0 - self.cmipr.ranking.long_top_fraction
            structure_valid = min(item.close for item in platform) >= ignition.breakout_level
            local_breakout = latest.close > platform_high and latest.close > latest.open
            close_position = (latest.close - latest.low) / latest_range
            stop = platform_low - cfg.stop_atr_buffer * atr_price
            chase = (latest.close - platform_high) / atr_price
            ema_valid = latest.close >= ema([item.close for item in self._closed(cfg.pullback_timeframe, symbol, decision_time, 30)], 9)[-1]
        else:
            rank_valid = percentile <= self.cmipr.ranking.short_bottom_fraction
            structure_valid = max(item.close for item in platform) <= ignition.breakout_level
            local_breakout = latest.close < platform_low and latest.close < latest.open
            close_position = (latest.high - latest.close) / latest_range
            stop = platform_high + cfg.stop_atr_buffer * atr_price
            chase = (platform_low - latest.close) / atr_price
            ema_valid = latest.close <= ema([item.close for item in self._closed(cfg.pullback_timeframe, symbol, decision_time, 30)], 9)[-1]

        if not rank_valid or not structure_valid or not local_breakout or not ema_valid:
            return None
        if close_position < cfg.bull_flag_min_close_position:
            return None
        if breakout_volume_ratio < cfg.bull_flag_min_breakout_volume_ratio:
            return None
        if chase > cfg.bull_flag_max_chase_atr:
            self.reject_reasons["bull_flag_chase"] += 1
            return None

        stop_atr = ignition.direction.value * (latest.close - stop) / atr_price
        if not cfg.min_stop_atr <= stop_atr <= cfg.max_stop_atr:
            self.reject_reasons["bull_flag_stop_invalid"] += 1
            return None

        adjusted = CmiprIgnitionSnapshot(
            direction=ignition.direction,
            candle=latest,
            breakout_level=platform_high if ignition.direction == Direction.LONG else platform_low,
            breakout_distance_atr=ignition.breakout_distance_atr,
            body_atr=abs(latest.close - latest.open) / atr_price,
            close_position=close_position,
            wick_ratio=ignition.wick_ratio,
            volume_ratio=breakout_volume_ratio,
            ranking_score=score,
            ranking_percentile=percentile,
            quality_score=min(1.0, ignition.quality_score + 0.05),
            setup_atr_value=ignition.setup_atr_value,
            stop_price=stop,
            stop_loss_pct=abs(latest.close - stop) / max(latest.close, 1e-12),
            take_profit_pct=ignition.take_profit_pct,
        )
        candidate = self._candidate_from_ignition(symbol, adjusted, runtime.event_id, "bull_flag")
        runtime.state = CmiprState.INITIAL_ENTRY_READY
        runtime.consumed_event_ids.add(runtime.event_id)
        self.stats["bull_flag_confirmed"] += 1
        self.campaign_diagnostics.mark(
            runtime.event_id,
            "entry_ready",
            entry_mode="bull_flag",
            bull_flag_platform_bars=minimum_bars,
            bull_flag_range_atr=platform_range_atr,
            bull_flag_volume_to_ignition=volume_to_ignition,
            bull_flag_breakout_volume_ratio=breakout_volume_ratio,
        )
        return candidate

    def _candidate_from_ignition(self, symbol: str, ignition: CmiprIgnitionSnapshot, event_id: str, mode: str) -> EntryCandidate:
        short_multiplier = self.cmipr.ignition.short_risk_multiplier if ignition.direction == Direction.SHORT else 1.0
        fraction = self.cmipr.entry.initial_risk_fraction * short_multiplier
        reason = (
            f"{CMIPR_REASON_TOKEN} event_id={event_id} entry_mode={mode} "
            f"trigger_tf={self.cmipr.ignition.timeframe if mode == 'confirmation_open' else self.cmipr.entry.pullback_timeframe} "
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
        self.campaign_diagnostics.mark(
            event_id,
            "entry_ready",
            entry_mode=mode,
            entry_chase_distance_atr=(
                ignition.direction.value * (ignition.candle.close - ignition.breakout_level)
                / max(ignition.setup_atr_value, 1e-12)
            ),
        )
        metadata = {
            "event_id": event_id,
            "entry_mode": mode,
            "breakout_level": ignition.breakout_level,
            "setup_atr": ignition.setup_atr_value,
            "structural_stop_price": ignition.stop_price,
            "ranking_percentile": ignition.ranking_percentile,
            "quality_score": ignition.quality_score,
        }
        return EntryCandidate(
            symbol,
            signal,
            ignition.candle,
            ignition.quality_score,
            abs(ignition.ranking_score),
            ignition.volume_ratio,
            "cmipr_ok",
            metadata,
        )

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


def _event_structure_mode(value: Any) -> str:
    normalized = str(value or "current").strip().lower()
    aliases = {
        "immediate_first_pullback": "immediate",
        "delayed_recompression": "delayed_recompression_only",
        "combined": "immediate_or_delayed_recompression",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "current",
        "immediate",
        "delayed_recompression_only",
        "immediate_or_delayed_recompression",
    }
    if normalized not in allowed:
        raise ValueError(f"unsupported CMIPR event_structure_mode: {value}")
    return normalized


def _diagnostic_campaign_risk_budget(config: Any) -> float:
    mode = str(config.cmipr.research.sizing_mode).strip().lower()
    if mode == "fixed_risk_usdt":
        return max(0.0, float(config.cmipr.research.fixed_trade_risk_usdt))
    equity = (
        float(config.cmipr.research.fixed_equity_usdt)
        if mode == "fixed_equity"
        else float(config.risk.starting_capital_usdt)
    )
    return max(0.0, equity) * max(0.0, float(config.risk.risk_per_trade_pct))


def _diagnostic_max_notional(config: Any) -> float:
    mode = str(config.cmipr.research.sizing_mode).strip().lower()
    equity = (
        float(config.cmipr.research.fixed_trade_risk_usdt) / max(float(config.risk.risk_per_trade_pct), 1e-12)
        if mode == "fixed_risk_usdt"
        else (
            float(config.cmipr.research.fixed_equity_usdt)
            if mode == "fixed_equity"
            else float(config.risk.starting_capital_usdt)
        )
    )
    leverage = max(1.0, float(config.trading.leverage))
    policy = float(config.risk.max_position_notional_usdt)
    policy = policy if policy > 0.0 else float("inf")
    symbol_cap = equity * max(0.0, float(config.risk.max_symbol_margin_pct)) * leverage
    account_cap = equity * max(0.0, float(config.risk.max_account_margin_usage_pct)) * leverage
    return min(policy, symbol_cap, account_cap)


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
