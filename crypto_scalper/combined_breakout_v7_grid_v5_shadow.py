from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from .binance_client import BinanceFuturesClient
from .combined_hybrid_v5_grid_v3_shadow import (
    _protected_position_from_dict,
    _protected_position_to_dict,
    build_hourly_market_context,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    NOTIONAL_HEADROOM_SAFETY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
    _closed_candles,
    _compact_series,
    _grid_campaign_from_dict,
    _grid_campaign_to_dict,
    _jsonable,
    _resolve_config_path,
    _utc_now,
)
from .live_config import LiveAppConfig, load_live_config
from .live_trader import AccountSnapshot
from .models import Candle
from .trend_grid import TrendGridConfig, build_trend_grid_timeline
from .trend_grid_optimize import (
    GridCandidate,
    GridPortfolioConfig,
    _create_campaign,
    _process_campaign_bar,
)
from .trend_grid_v3_optimize import GridMarketOverlay, apply_market_overlay
from .trend_grid_v4 import GridV4EntryGate, filter_grid_v4_candidates
from .trend_grid_v5 import (
    TREND_GRID_V5_NAME,
    GridV5ConfidencePolicy,
    grid_v5_tier,
)
from .trend_grid_v5_engine import GridV5ExecutionProfile
from .volatility_breakout import (
    VolatilityBreakoutConfig,
    build_dual_thrust_signals,
)
from .volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    ProtectedPosition,
    _process_protected_bar,
    _protected_position,
)
from .volatility_breakout_optimize import (
    Candidate,
    OpenPosition,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _candidate_sort_key,
    _entry_position,
    minute_datetime,
    minute_token,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    _regime_score,
    enrich_candidates_v4,
)
from .volatility_breakout_v6 import (
    BreakoutV6EntryConfig,
    BreakoutV6SideGate,
    filter_breakout_v6_candidates,
)
from .volatility_breakout_v6_core_runner_optimize import (
    ManagedLaneProfile,
    tiered_drawdown_risk_multiplier,
)
from .volatility_breakout_v6_engine import BreakoutV6ExecutionProfile
from .volatility_breakout_v7 import (
    VOLATILITY_BREAKOUT_V7_NAME,
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
    breakout_v7_risk_multiplier,
)


COMBINED_V7_GRID_V5_NAME = (
    "dual_thrust_volatility_breakout_v7_confidence_allocated"
    "_plus_dynamic_trend_following_grid_v5_confidence_managed_max2"
)
BREAKOUT_V7_GRID_V5_SHADOW_VERSION = "breakout_v7_grid_v5_max2_shadow_20260723"
BREAKOUT_V7_COMPONENT_VERSION = "breakout_v7_confidence_refined_20260723"


def _source_path(
    payload: dict[str, Any],
    key: str,
    anchor: Path,
) -> tuple[Path, str]:
    source = payload.get("source_configs", {}).get(key)
    if isinstance(source, str):
        return _resolve_config_path(source, anchor), ""
    if not isinstance(source, dict) or not source.get("path"):
        raise RuntimeError(f"combined source config is missing: {key}")
    return (
        _resolve_config_path(str(source["path"]), anchor),
        str(source.get("sha256", "")),
    )


def _load_v7_v5_source_bundle(config: LiveAppConfig) -> dict[str, Any]:
    shadow = config.combined_volatility_trend_grid_shadow
    combined_path = _resolve_config_path(shadow.source_combined_config_path)
    combined_payload = json.loads(combined_path.read_text(encoding="utf-8"))
    breakout_path, expected_breakout = _source_path(
        combined_payload, BREAKOUT_KEY, combined_path
    )
    grid_path, expected_grid = _source_path(
        combined_payload, GRID_KEY, combined_path
    )
    hashes = {
        "combined": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
        "breakout": hashlib.sha256(breakout_path.read_bytes()).hexdigest(),
        "grid": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
    }
    if expected_breakout and hashes["breakout"] != expected_breakout:
        raise RuntimeError("Breakout v7 source hash differs from frozen combined config")
    if expected_grid and hashes["grid"] != expected_grid:
        raise RuntimeError("Grid v5 source hash differs from frozen combined config")
    return {
        "combined_path": combined_path,
        "combined_payload": combined_payload,
        "breakout_path": breakout_path,
        "breakout_payload": json.loads(
            breakout_path.read_text(encoding="utf-8")
        ),
        "grid_path": grid_path,
        "grid_payload": json.loads(grid_path.read_text(encoding="utf-8")),
        "hashes": hashes,
    }


def combined_v7_grid_v5_shadow_config_hash(config: LiveAppConfig) -> str:
    bundle = _load_v7_v5_source_bundle(config)
    risk = config.risk
    payload = {
        "strategy_name": COMBINED_V7_GRID_V5_NAME,
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
            "breakout_v7_priority_before_grid_v5",
            "same_symbol_overlap_forbidden",
            "adverse_stop_first_on_same_bar",
            "entry_time_confidence_profile_frozen_for_position",
            "entry_time_grid_profile_frozen_for_campaign",
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _breakout_entry_from_dict(payload: dict[str, Any]) -> BreakoutV6EntryConfig:
    return BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(**payload["long"]),
        short=BreakoutV6SideGate(**payload["short"]),
        max_signals_per_symbol_day=int(payload["max_signals_per_symbol_day"]),
        require_market_context=bool(payload["require_market_context"]),
    )


def _breakout_profile_to_dict(
    profile: BreakoutV6ExecutionProfile,
) -> dict[str, Any]:
    return {
        "lane": profile.lane,
        "signal": asdict(profile.signal),
        "portfolio": asdict(profile.portfolio),
        "exit_protection": asdict(profile.exit_protection),
    }


def _breakout_profile_from_dict(
    payload: dict[str, Any],
) -> BreakoutV6ExecutionProfile:
    return BreakoutV6ExecutionProfile(
        lane=str(payload["lane"]),
        signal=VolatilityBreakoutConfig(**dict(payload["signal"])),
        portfolio=PortfolioSearchConfig(**dict(payload["portfolio"])),
        exit_protection=ExitProtectionConfig(**dict(payload["exit_protection"])),
    )


def _v7_position_to_dict(
    protected: ProtectedPosition,
    profile: BreakoutV6ExecutionProfile,
) -> dict[str, Any]:
    return {
        "protected": _protected_position_to_dict(protected),
        "profile": _breakout_profile_to_dict(profile),
    }


def _v7_position_from_dict(
    payload: dict[str, Any],
) -> tuple[ProtectedPosition, BreakoutV6ExecutionProfile]:
    return (
        _protected_position_from_dict(dict(payload["protected"])),
        _breakout_profile_from_dict(dict(payload["profile"])),
    )


def _grid_profile_to_dict(profile: GridV5ExecutionProfile) -> dict[str, Any]:
    return {
        "tier": profile.tier,
        "signal": asdict(profile.signal),
        "portfolio": asdict(profile.portfolio),
    }


def _grid_profile_from_dict(payload: dict[str, Any]) -> GridV5ExecutionProfile:
    return GridV5ExecutionProfile(
        tier=str(payload["tier"]),
        signal=TrendGridConfig(**dict(payload["signal"])),
        portfolio=GridPortfolioConfig(**dict(payload["portfolio"])),
    )


class CombinedBreakoutV7GridV5ShadowTrader(
    CombinedVolatilityTrendGridShadowTrader
):
    """Dry-run-only shared account for the frozen Breakout-v7/Grid-v5 sleeves."""

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
        self.source_bundle = _load_v7_v5_source_bundle(config)
        combined = self.source_bundle["combined_payload"]
        breakout = self.source_bundle["breakout_payload"]
        grid = self.source_bundle["grid_payload"]
        self.global_config = CombinedPortfolioConfig.from_dict(combined["portfolio"])

        self.breakout_entry_config = _breakout_entry_from_dict(breakout["entry"])
        self.breakout_confidence_policy = BreakoutV7ConfidencePolicy(
            **breakout["confidence_policy"]
        )
        self.breakout_entry_timing = BreakoutV7EntryTiming(
            **breakout["entry_timing"]
        )
        self.breakout_managed_profile = ManagedLaneProfile(
            **breakout["managed_profile"]
        )
        self.breakout_signal_config = VolatilityBreakoutConfig(
            **breakout["operational_signal"]
        )
        self.breakout_build_signal_config = replace(
            self.breakout_signal_config,
            max_signals_per_symbol_day=24,
        )
        self.breakout_portfolio_config = PortfolioSearchConfig(
            **breakout["operational_portfolio"]
        )

        self.grid_entry_gate = GridV4EntryGate(**grid["entry_gate"])
        self.grid_market_overlay = GridMarketOverlay(**grid["market_overlay"])
        self.grid_signal_config = TrendGridConfig(**grid["signal"])
        self.grid_portfolio_config = GridPortfolioConfig(**grid["portfolio"])
        self.grid_confidence_policy = GridV5ConfidencePolicy(
            **grid["confidence_policy"]
        )

        self.config_hash = combined_v7_grid_v5_shadow_config_hash(config)
        self.state_path = Path(self.shadow.state_path)
        self.event_log_path = Path(self.shadow.event_log_path)
        self.report_path = Path(self.shadow.report_path)
        self.state = self._load_or_create_state()
        self.state.setdefault("grid_execution_profile", None)
        self._market_context: dict[int, dict[str, V4MarketSnapshot]] = {}
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
                source_strategy=COMBINED_V7_GRID_V5_NAME,
            )
            self.state["sample_initialized_event_logged"] = True
            self._persist_state()

    def validate_startup(self) -> None:
        self.validate_startup_settings_only()
        if self.config.exchange.environment != "mainnet":
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires mainnet public data")
        if self.config.risk.cost_experiment != "full_cost":
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires full_cost")
        if self.config.risk.backtest_mode != "conservative":
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires conservative mode")
        if not self.config.risk.funding_enabled:
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires funding")
        if self.config.risk.dynamic_slippage_enabled:
            raise RuntimeError("dynamic slippage differs from the frozen backtest")
        expected_costs = (2.0, 5.0, 2.0, 0.0002, 0.0005)
        actual_costs = (
            self.config.risk.market_slippage_bps,
            self.config.risk.stop_slippage_bps,
            self.config.risk.take_profit_slippage_bps,
            self.config.risk.maker_fee_rate,
            self.config.risk.taker_fee_rate,
        )
        if actual_costs != expected_costs:
            raise RuntimeError("GUI costs differ from the v7/v5 backtest costs")
        if self.config.risk.starting_capital_usdt != 200.0:
            raise RuntimeError("Breakout-v7/Grid-v5 sample must start from 200U")
        if self.config.trading.leverage != 10:
            raise RuntimeError("Breakout-v7/Grid-v5 GUI display leverage must remain 10x")
        if self.shadow.max_open_positions != 2 or self.config.trading.max_open_positions != 2:
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires global max two")
        if self.shadow.max_open_positions_per_strategy != 1:
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires one slot per strategy")
        if self.breakout_portfolio_config.max_open_positions != 1:
            raise RuntimeError("Breakout v7 source must remain max one position")
        if self.grid_portfolio_config.max_open_campaigns != 1:
            raise RuntimeError("Grid v5 source must remain max one campaign")
        expected_priority = (BREAKOUT_KEY, GRID_KEY)
        if tuple(self.shadow.entry_priority) != expected_priority:
            raise RuntimeError(f"entry priority must remain {expected_priority}")
        if tuple(self.global_config.entry_priority) != expected_priority:
            raise RuntimeError("combined source entry priority changed")
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
            raise RuntimeError("combined source must forbid same-symbol overlap")
        if self.shadow.allow_same_symbol_across_strategies:
            raise RuntimeError("GUI shadow must forbid same-symbol overlap")

        combined = self.source_bundle["combined_payload"]
        breakout = self.source_bundle["breakout_payload"]
        grid = self.source_bundle["grid_payload"]
        if combined.get("strategy_name") != COMBINED_V7_GRID_V5_NAME:
            raise RuntimeError("combined v7/v5 strategy name changed")
        if self.shadow.strategy_name != COMBINED_V7_GRID_V5_NAME:
            raise RuntimeError("GUI combined v7/v5 strategy name changed")
        if self.shadow.frozen_version != BREAKOUT_V7_GRID_V5_SHADOW_VERSION:
            raise RuntimeError("GUI combined v7/v5 frozen version changed")
        if breakout.get("strategy_name") != VOLATILITY_BREAKOUT_V7_NAME:
            raise RuntimeError("Breakout v7 source strategy changed")
        if breakout.get("selection_status") != "strict_robust_improvement":
            raise RuntimeError("Breakout v7 source is not the selected robust candidate")
        if grid.get("strategy_name") != TREND_GRID_V5_NAME:
            raise RuntimeError("Grid v5 source strategy changed")
        if grid.get("selection_status") != "strict_robust_improvement":
            raise RuntimeError("Grid v5 source is not the selected robust candidate")
        if self.breakout_shadow.strategy_name != VOLATILITY_BREAKOUT_V7_NAME:
            raise RuntimeError("GUI Breakout component name changed")
        if self.breakout_shadow.frozen_version != BREAKOUT_V7_COMPONENT_VERSION:
            raise RuntimeError("GUI Breakout component version changed")

        symbols = tuple(self.shadow.enabled_symbols)
        if symbols != tuple(UNIVERSE_50):
            raise RuntimeError("Breakout-v7/Grid-v5 shadow requires frozen 50 symbols")
        if symbols != tuple(combined.get("symbols", ())):
            raise RuntimeError("combined source and GUI universes differ")
        if symbols != tuple(self.breakout_shadow.enabled_symbols):
            raise RuntimeError("combined and Breakout v7 universes differ")
        if tuple(self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI trading symbols differ from frozen universe")
        if tuple(self.config.trading.entry_symbols or self.config.trading.symbols) != symbols:
            raise RuntimeError("GUI entry symbols differ from frozen universe")

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
        ):
            if getattr(self.breakout_shadow, field) != getattr(
                self.breakout_signal_config, field
            ):
                raise RuntimeError(f"Breakout v7 GUI/source differs: {field}")
        if self.breakout_shadow.risk_per_trade_pct != (
            self.breakout_managed_profile.core_strong_risk_pct
        ):
            raise RuntimeError("Breakout v7 GUI risk display differs from strong base risk")
        if self.breakout_entry_timing.confirmation_minutes != 0:
            raise RuntimeError("selected Breakout v7 source requires immediate entry")
        if self.breakout_managed_profile.drawdown_scope != "global":
            raise RuntimeError("selected Breakout v7 drawdown scope changed")
        if not self.grid_confidence_policy.reject_weak_tier:
            raise RuntimeError("selected Grid v5 must reject the weak tier")

        self.breakout_entry_config.validate()
        self.breakout_confidence_policy.validate()
        self.breakout_entry_timing.validate()
        self.grid_entry_gate.validate()
        self.grid_market_overlay.validate()
        self.grid_confidence_policy.validate()
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
            raise RuntimeError("all unrelated GUI strategies must remain disabled")
        if self.state.get("grid_campaign") and not self.state.get(
            "grid_execution_profile"
        ):
            raise RuntimeError("open Grid v5 campaign is missing its frozen profile")
        self.client.ping()

    def _snapshot_for(
        self,
        candidate: Candidate | GridCandidate,
    ) -> V4MarketSnapshot | None:
        minute = candidate.entry_minute - candidate.entry_minute % 60
        return self._market_context.get(minute, {}).get(
            candidate.signal.symbol
        )

    def _select_breakout_profile(
        self,
        candidate: Candidate,
        equity: float,
    ) -> Optional[BreakoutV6ExecutionProfile]:
        snapshot = self._snapshot_for(candidate)
        managed = self.breakout_managed_profile
        regime = (
            _regime_score(snapshot, candidate.signal.direction)
            if snapshot is not None
            else 0.0
        )
        alignment = (
            float(candidate.signal.direction.value) * snapshot.symbol_ema55_atr
            if snapshot is not None
            else -999.0
        )
        risk = managed.core_risk_pct
        if (
            regime >= managed.strong_regime_score
            and alignment >= managed.strong_alignment_atr
        ):
            risk = managed.core_strong_risk_pct
        elif regime < managed.weak_regime_score:
            risk *= managed.weak_risk_multiplier
        confidence, score, tier = breakout_v7_risk_multiplier(
            candidate, snapshot, self.breakout_confidence_policy
        )
        if confidence <= 0.0:
            return None
        signal = replace(
            self.breakout_signal_config,
            stop_atr_multiple=managed.core_stop_atr,
            take_profit_r=60.0,
            max_holding_minutes=managed.core_max_holding_minutes,
            fail_fast_minutes=managed.core_fail_fast_minutes,
            fail_fast_min_mfe_r=managed.core_fail_fast_min_mfe_r,
            fail_fast_max_current_r=managed.core_fail_fast_max_current_r,
        )
        exit_config = ExitProtectionConfig(
            breakeven_trigger_r=managed.core_breakeven_trigger_r,
            profit_giveback_activation_r=(
                managed.core_profit_giveback_activation_r
            ),
            profit_giveback_r=managed.core_profit_giveback_r,
            partial_take_profit_r=managed.core_partial_r,
            partial_take_profit_fraction=managed.core_partial_fraction,
            move_stop_to_breakeven_after_partial=(
                managed.core_move_breakeven_after_partial
            ),
        )
        peak = max(float(self.state["peak_equity"]), equity)
        drawdown = (peak - equity) / peak if peak > 0.0 else 1.0
        governor = tiered_drawdown_risk_multiplier(
            drawdown,
            managed.drawdown_reduce_start,
            managed.drawdown_reduce_multiplier,
            managed.drawdown_deep_start,
            managed.drawdown_deep_multiplier,
        )
        portfolio = replace(
            self.breakout_portfolio_config,
            risk_per_trade_pct=risk * confidence * governor,
            long_risk_multiplier=managed.core_long_multiplier,
            short_risk_multiplier=managed.core_short_multiplier,
            ranking_mode=managed.ranking_mode,
        )
        profile = BreakoutV6ExecutionProfile(
            f"v7_{tier}_score_{score}",
            signal,
            portfolio,
            exit_config,
        )
        profile.validate()
        return profile

    def _select_grid_profile(
        self,
        candidate: GridCandidate,
    ) -> Optional[GridV5ExecutionProfile]:
        policy = self.grid_confidence_policy
        tier, score = grid_v5_tier(candidate, self._snapshot_for(candidate), policy)
        if tier == "weak" and policy.reject_weak_tier:
            return None
        if tier == "strong":
            risk_multiplier = policy.strong_risk_multiplier
            target = policy.strong_target_spacing
            loss_limit = policy.strong_loss_limit_r
            duration = policy.strong_max_campaign_minutes
        elif tier == "weak":
            risk_multiplier = policy.weak_risk_multiplier
            target = policy.weak_target_spacing
            loss_limit = policy.weak_loss_limit_r
            duration = policy.weak_max_campaign_minutes
        else:
            risk_multiplier = policy.standard_risk_multiplier
            target = policy.standard_target_spacing
            loss_limit = policy.standard_loss_limit_r
            duration = policy.standard_max_campaign_minutes
        return GridV5ExecutionProfile(
            f"{tier}_score_{score}",
            replace(
                self.grid_signal_config,
                grid_target_spacing=target,
                campaign_loss_limit_r=loss_limit,
                max_campaign_minutes=duration,
            ),
            replace(
                self.grid_portfolio_config,
                risk_per_campaign_pct=(
                    self.grid_portfolio_config.risk_per_campaign_pct
                    * risk_multiplier
                ),
                max_campaign_risk_pct=policy.maximum_campaign_risk_pct,
            ),
        )

    def _build_entry_candidates(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        latest_available: datetime,
    ) -> tuple[list[Candidate], list[GridCandidate], float]:
        symbols = tuple(self.shadow.enabled_symbols)
        context = build_hourly_market_context(symbols, candles_by_symbol)
        self._market_context = context
        raw_breakout: dict[int, list[Candidate]] = defaultdict(list)
        raw_grid: dict[int, list[GridCandidate]] = defaultdict(list)
        for symbol, candles in candles_by_symbol.items():
            for signal in build_dual_thrust_signals(
                symbol, candles, self.breakout_build_signal_config
            ):
                entry_minute = minute_token(signal.signal_available_time)
                raw_breakout[entry_minute].append(Candidate(signal, entry_minute))
            _, grid_signals = build_trend_grid_timeline(
                symbol, candles, self.grid_signal_config
            )
            for signal in grid_signals:
                entry_minute = minute_token(signal.signal_available_time)
                raw_grid[entry_minute].append(GridCandidate(signal, entry_minute))

        breakout = filter_breakout_v6_candidates(
            enrich_candidates_v4(dict(raw_breakout), context),
            context,
            self.breakout_entry_config,
        )
        grid = filter_grid_v4_candidates(
            apply_market_overlay(
                dict(raw_grid), context, self.grid_market_overlay
            ),
            context,
            self.grid_entry_gate,
        )
        current_minute = minute_token(latest_available)
        breakout_rows = [
            row
            for row in breakout.get(current_minute, ())
            if row.signal.event_id not in self.state["seen_events"][BREAKOUT_KEY]
        ]
        grid_rows = [
            row
            for row in grid.get(current_minute, ())
            if row.signal.event_id not in self.state["seen_events"][GRID_KEY]
        ]
        breakout_rows.sort(
            key=lambda row: _candidate_sort_key(
                row, self.breakout_managed_profile.ranking_mode
            )
        )
        grid_rows.sort(
            key=lambda row: (-row.signal.quality_score, row.signal.symbol)
        )
        btc = context.get(current_minute, {}).get("BTCUSDT")
        return breakout_rows, grid_rows, btc.btc_return_4h if btc else 0.0

    def _candidate_reject_reason(
        self,
        strategy: str,
        candidate: Candidate | GridCandidate,
        now: datetime,
    ) -> str | None:
        reason = super()._candidate_reject_reason(strategy, candidate, now)
        if reason is not None:
            return reason
        if strategy == BREAKOUT_KEY:
            equity = self.snapshot_account(fetch_mark=False).equity
            if self._select_breakout_profile(candidate, equity) is None:
                return "v7_confidence_rejected"
        elif self._select_grid_profile(candidate) is None:
            return "grid_v5_weak_tier_rejected"
        return None

    def _open_breakout_candidate(
        self,
        candidate: Candidate,
        now: datetime,
    ) -> bool:
        signal = candidate.signal
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: Breakout v7成交参考不可用 ({exc})")
            return False
        equity = self.snapshot_account(fetch_mark=False).equity
        profile = self._select_breakout_profile(candidate, equity)
        if profile is None:
            return False
        raw = active.close
        synthetic = Candle(active.timestamp, raw, raw, raw, raw, active.volume)
        live_candidate = replace(
            candidate, entry_minute=minute_token(active.timestamp)
        )
        available = self._available_notional_multiple(equity)
        profile = replace(
            profile,
            portfolio=replace(
                profile.portfolio,
                max_notional_multiple=min(
                    profile.portfolio.max_notional_multiple,
                    available * NOTIONAL_HEADROOM_SAFETY,
                ),
            ),
        )
        opened = _entry_position(
            live_candidate,
            _compact_series([synthetic]),
            rules,
            profile.signal,
            profile.portfolio,
            self.execution,
            equity
            if profile.portfolio.compound
            else float(self.state["starting_equity"]),
        )
        if opened is None:
            return False
        position, entry_fee = opened
        protected = _protected_position(
            position, profile.exit_protection, self.execution
        )
        self.state["cash"] -= entry_fee
        self.state["breakout_position"] = _v7_position_to_dict(
            protected, profile
        )
        self.state["breakout_last_processed_minute"] = (
            position.entry_minute - 1
        )
        self._record_entry(BREAKOUT_KEY, signal.symbol, now)
        self._last_marks[signal.symbol] = raw
        self.log(
            f"{signal.symbol}: Breakout v7开仓 {signal.direction.name} "
            f"{profile.lane} qty={position.quantity:.8g} "
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
            v7_lane=profile.lane,
            exit_profile=asdict(profile.exit_protection),
        )
        return True

    def _decode_breakout_position(self, payload: dict[str, Any]) -> OpenPosition:
        return _v7_position_from_dict(payload)[0].position

    def _manage_breakout_position(self, now: datetime) -> None:
        payload = self.state.get("breakout_position")
        if not payload:
            return
        protected, profile = _v7_position_from_dict(payload)
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
            self.log(f"{symbol}: Breakout v7持仓数据不可用，保留仓位 ({exc})")
            self._append_event(
                "position_data_error",
                strategy=BREAKOUT_KEY,
                symbol=symbol,
                error=str(exc),
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
                profile.signal,
                profile.exit_protection,
                execution,
                rules,
            )
            self.state["cash"] += cash_delta
            last = minute
            self.state["breakout_last_processed_minute"] = minute
            if len(protected.realized_legs) > partial_before:
                leg = protected.realized_legs[-1]
                self.log(
                    f"{symbol}: Breakout v7分批退出 "
                    f"qty={leg['quantity']:.8g} 净盈亏={leg['net_pnl']:+.4f}U"
                )
                self._append_event(
                    "partial_exit", strategy=BREAKOUT_KEY, **leg
                )
            if trade is None:
                continue
            tagged = {
                "strategy": BREAKOUT_KEY,
                "funding_status": "complete",
                "v6_lane": profile.lane,
                **trade,
            }
            self.state["trades"].append(tagged)
            self.state["breakout_position"] = None
            self._set_cooldown(
                BREAKOUT_KEY,
                symbol,
                minute,
                self.breakout_portfolio_config.symbol_cooldown_minutes,
            )
            self.log(
                f"{symbol}: Breakout v7平仓 {tagged['exit_reason']} "
                f"净盈亏={tagged['net_pnl']:+.4f}U "
                f"({tagged['pnl_r']:+.3f}R)"
            )
            self._append_event("exit", **tagged)
            return
        self.state["breakout_position"] = _v7_position_to_dict(
            protected, profile
        )
        if candles:
            self._last_marks[symbol] = candles[-1].close

    def _open_grid_candidate(
        self,
        candidate: GridCandidate,
        now: datetime,
    ) -> bool:
        signal = candidate.signal
        profile = self._select_grid_profile(candidate)
        if profile is None:
            return False
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
        except Exception as exc:
            self.log(f"{signal.symbol}: Grid v5成交参考不可用 ({exc})")
            return False
        raw = active.close
        synthetic = Candle(active.timestamp, raw, raw, raw, raw, active.volume)
        live_candidate = replace(
            candidate, entry_minute=minute_token(active.timestamp)
        )
        equity = self.snapshot_account(fetch_mark=False).equity
        available = self._available_notional_multiple(equity)
        profile = replace(
            profile,
            portfolio=replace(
                profile.portfolio,
                max_notional_multiple=min(
                    profile.portfolio.max_notional_multiple,
                    available * NOTIONAL_HEADROOM_SAFETY,
                ),
            ),
        )
        opened = _create_campaign(
            live_candidate,
            _compact_series([synthetic]),
            rules,
            profile.signal,
            profile.portfolio,
            self.execution,
            equity
            if profile.portfolio.compound
            else float(self.state["starting_equity"]),
        )
        if opened is None:
            return False
        campaign, initial_fee = opened
        self.state["cash"] -= initial_fee
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self.state["grid_execution_profile"] = _grid_profile_to_dict(profile)
        self.state["grid_last_processed_minute"] = campaign.start_minute - 1
        self._record_entry(GRID_KEY, signal.symbol, now)
        self._last_marks[signal.symbol] = raw
        committed = sum(
            level.quantity * level.raw_price for level in campaign.levels
        )
        self.log(
            f"{signal.symbol}: Grid v5开仓 {signal.direction.name} "
            f"{profile.tier} levels={len(campaign.levels)} "
            f"hard_stop={campaign.hard_stop:.8g} "
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
            v5_tier=profile.tier,
        )
        return True

    def _manage_grid_campaign(self, now: datetime) -> None:
        payload = self.state.get("grid_campaign")
        if not payload:
            return
        profile_payload = self.state.get("grid_execution_profile")
        if not profile_payload:
            raise RuntimeError("Grid v5 campaign lost its entry-time profile")
        profile = _grid_profile_from_dict(profile_payload)
        campaign = _grid_campaign_from_dict(payload)
        symbol = campaign.candidate.signal.symbol
        try:
            candles = self.client.klines(symbol, "1m", 1500)
            closed = _closed_candles(candles, 1, now)
            rules = self.client.symbol_rules(symbol)
            hourly = _closed_candles(
                self.client.klines(symbol, "1h", 500), 60, now
            )
            snapshots, _ = build_trend_grid_timeline(
                symbol, hourly, profile.signal
            )
            snapshot_by_minute = {
                minute_token(item.available_time): item for item in snapshots
            }
            execution = self._execution_with_funding(
                symbol, minute_datetime(campaign.start_minute), now
            )
        except Exception as exc:
            self.log(f"{symbol}: Grid v5持仓数据不可用，保留campaign ({exc})")
            self._append_event(
                "position_data_error",
                strategy=GRID_KEY,
                symbol=symbol,
                error=str(exc),
            )
            return
        last = int(
            self.state.get("grid_last_processed_minute")
            or campaign.start_minute - 1
        )
        if not self._history_is_contiguous(GRID_KEY, symbol, closed, last):
            return
        series = _compact_series(closed)
        for index, minute_value in enumerate(series.minutes):
            minute = int(minute_value)
            if minute <= last:
                continue
            cash_delta, close_reason = _process_campaign_bar(
                campaign,
                minute,
                series,
                index,
                snapshot_by_minute.get(minute),
                profile.signal,
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
            self.state["grid_execution_profile"] = None
            self._set_cooldown(
                GRID_KEY,
                symbol,
                minute,
                self.grid_portfolio_config.symbol_cooldown_minutes,
            )
            if report is not None:
                trade = {
                    "strategy": GRID_KEY,
                    "funding_status": "complete",
                    "v5_tier": profile.tier,
                    **report,
                }
                self.state["trades"].append(trade)
                self.log(
                    f"{symbol}: Grid v5平仓 {close_reason} "
                    f"净盈亏={trade['net_pnl']:+.4f}U "
                    f"({trade['pnl_r']:+.3f}R) "
                    f"entries={trade['entry_count']}"
                )
                self._append_event("exit", **trade)
            else:
                self._record_reject(GRID_KEY, "campaign_without_fill")
            return
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        if candles:
            self._last_marks[symbol] = candles[-1].close

    def acceptance_report(
        self,
        account: AccountSnapshot | None = None,
    ) -> dict[str, Any]:
        report = super().acceptance_report(account)
        report.update(
            {
                "important_note": (
                    "Breakout v7 and Grid v5 historical results are individual-sleeve "
                    "research references. This shared-account report contains only new "
                    "dry-run observations and makes no combined historical performance claim."
                ),
                "historical_reference": {
                    "breakout_v7": {
                        "six_month": {
                            "trade_count": 84,
                            "net_profit_usdt": 28371.32670673413,
                            "profit_factor": 6.4967218779976506,
                            "win_rate_pct": 39.285714285714285,
                            "max_drawdown_pct": 30.72889300032368,
                        },
                        "three_month": {
                            "trade_count": 46,
                            "net_profit_usdt": 8229.433468039266,
                            "profit_factor": 6.898147189032333,
                            "win_rate_pct": 45.65217391304348,
                            "max_drawdown_pct": 29.06828635044768,
                        },
                    },
                    "grid_v5": {
                        "six_month": {
                            "trade_count": 39,
                            "net_profit_usdt": 401.43742249732304,
                            "profit_factor": 11.615272621282275,
                            "win_rate_pct": 87.17948717948718,
                            "max_drawdown_pct": 9.822006268639703,
                        },
                        "three_month": {
                            "trade_count": 22,
                            "net_profit_usdt": 201.7267483895962,
                            "profit_factor": 66.06082418895987,
                            "win_rate_pct": 90.9090909090909,
                            "max_drawdown_pct": 5.836494862506576,
                        },
                    },
                },
                "source_configs": {
                    "breakout_v7": str(self.source_bundle["breakout_path"]),
                    "grid_v5": str(self.source_bundle["grid_path"]),
                },
                "breakout_v7_policy": self.breakout_confidence_policy.as_dict(),
                "grid_v5_policy": self.grid_confidence_policy.as_dict(),
                "by_breakout_v7_lane": self._group_trades(
                    [
                        row
                        for row in report.get("trades", ())
                        if row.get("strategy") == BREAKOUT_KEY
                    ],
                    "v6_lane",
                ),
                "by_grid_v5_tier": self._group_trades(
                    [
                        row
                        for row in report.get("trades", ())
                        if row.get("strategy") == GRID_KEY
                    ],
                    "v5_tier",
                ),
            }
        )
        return report


def _logger(message: str) -> None:
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Breakout v7 plus Grid v5 max-two dry-run shadow"
    )
    parser.add_argument(
        "--config",
        default="config.gui.breakout-v7-grid-v5-max2-shadow.json",
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
    trader = CombinedBreakoutV7GridV5ShadowTrader(
        config, client, logger=_logger
    )
    if args.report_only:
        print(
            json.dumps(
                _jsonable(trader.acceptance_report()),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.once:
        trader.validate_startup()
        trader._acquire_runtime_lock()
        try:
            trader.run_once()
        finally:
            trader._release_runtime_lock()
        print(
            json.dumps(
                _jsonable(trader.acceptance_report()),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    stop = threading.Event()
    try:
        trader.run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
