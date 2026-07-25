from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .binance_client import BinanceFuturesClient
from .combined_breakout_v7_grid_v5_shadow import (
    CombinedBreakoutV7GridV5ShadowTrader,
    _breakout_entry_from_dict,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
    _jsonable,
    _resolve_config_path,
    _utc_now,
)
from .live_config import LiveAppConfig, load_live_config
from .live_trader import AccountSnapshot
from .risk import execution_config_from_live_config
from .trend_grid import TrendGridConfig
from .trend_grid_optimize import GridCandidate, GridPortfolioConfig
from .trend_grid_v3_optimize import (
    GridMarketOverlay,
    _ranking_value,
)
from .trend_grid_v4 import GridV4EntryGate
from .trend_grid_v5 import GridV5ConfidencePolicy
from .trend_grid_v5_engine import GridV5ExecutionProfile
from .trend_grid_v6 import (
    TREND_GRID_V6_NAME,
    GridV6CampaignPolicy,
    grid_v6_entry_decision,
)
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import ExitProtectionConfig
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
)
from .volatility_breakout_v4_research import (
    V4MarketSnapshot,
    _regime_score,
)
from .volatility_breakout_v6_core_runner_optimize import (
    ManagedLaneProfile,
    tiered_drawdown_risk_multiplier,
)
from .volatility_breakout_v6_engine import BreakoutV6ExecutionProfile
from .volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
)
from .volatility_breakout_v8 import (
    VOLATILITY_BREAKOUT_V8_NAME,
    BreakoutV8ScoreAllocation,
    breakout_v8_risk_multiplier,
)


COMBINED_V8_GRID_V6_NAME = (
    "dual_thrust_volatility_breakout_v8_score_convex_plus_"
    "dynamic_trend_following_grid_v6_profit_protected_max2"
)
BREAKOUT_V8_GRID_V6_SHADOW_VERSION = (
    "breakout_v8_grid_v6_max2_shadow_20260724"
)
BREAKOUT_V8_COMPONENT_VERSION = (
    "breakout_v8_score_convex_refined_20260724"
)


def _load_v8_v6_source_bundle(config: LiveAppConfig) -> dict[str, Any]:
    shadow = config.combined_volatility_trend_grid_shadow
    combined_path = _resolve_config_path(shadow.source_combined_config_path)
    combined_payload = json.loads(
        combined_path.read_text(encoding="utf-8")
    )
    sources = combined_payload.get("source_configs", {})

    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {
        "combined": hashlib.sha256(combined_path.read_bytes()).hexdigest()
    }
    for key in (BREAKOUT_KEY, GRID_KEY):
        source = sources.get(key)
        if isinstance(source, str):
            source_path = _resolve_config_path(source, combined_path)
            expected_hash = str(
                combined_payload.get("source_hashes", {}).get(key, "")
            )
        elif isinstance(source, dict) and source.get("path"):
            source_path = _resolve_config_path(
                str(source["path"]), combined_path
            )
            expected_hash = str(source.get("sha256", ""))
        else:
            raise RuntimeError(f"combined source config is missing: {key}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f"{key} source hash differs from the frozen v8/v6 config"
            )
        resolved[key] = source_path
        hashes[key] = actual_hash

    return {
        "combined_path": combined_path,
        "combined_payload": combined_payload,
        "breakout_path": resolved[BREAKOUT_KEY],
        "breakout_payload": json.loads(
            resolved[BREAKOUT_KEY].read_text(encoding="utf-8")
        ),
        "grid_path": resolved[GRID_KEY],
        "grid_payload": json.loads(
            resolved[GRID_KEY].read_text(encoding="utf-8")
        ),
        "hashes": hashes,
    }


def combined_v8_grid_v6_shadow_config_hash(config: LiveAppConfig) -> str:
    bundle = _load_v8_v6_source_bundle(config)
    risk = config.risk
    payload = {
        "strategy_name": COMBINED_V8_GRID_V6_NAME,
        "combined_shadow": asdict(
            config.combined_volatility_trend_grid_shadow
        ),
        "breakout_shadow": asdict(config.dual_thrust_shadow),
        "source_hashes": bundle["hashes"],
        "safety": {
            "environment": config.exchange.environment,
            "dry_run": config.trading.dry_run,
            "leverage": config.trading.leverage,
            "legacy_strategy_flags": {
                "super_volume": (
                    config.strategy.super_volume_breakout_enabled
                ),
                "startup_breakout": (
                    config.strategy.startup_breakout_enabled
                ),
                "ordinary_breakout": (
                    config.strategy.ordinary_breakout_enabled
                ),
                "pullback_reclaim": (
                    config.strategy.pullback_reclaim_enabled
                ),
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
            "breakout_v8_priority_before_grid_v6",
            "same_symbol_overlap_forbidden",
            "adverse_stop_first_on_same_bar",
            "entry_time_breakout_profile_frozen_for_position",
            "entry_time_grid_profile_frozen_for_campaign",
            "grid_extension_ascending_priority",
        ],
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


class CombinedBreakoutV8GridV6ShadowTrader(
    CombinedBreakoutV7GridV5ShadowTrader
):
    """Dry-run-only shared account for Breakout v8 and Grid v6."""

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
        self.execution = execution_config_from_live_config(
            config, cost_experiment="full_cost", mode="conservative"
        )
        self.source_bundle = _load_v8_v6_source_bundle(config)
        combined = self.source_bundle["combined_payload"]
        breakout = self.source_bundle["breakout_payload"]
        grid = self.source_bundle["grid_payload"]
        self.global_config = CombinedPortfolioConfig.from_dict(
            combined["portfolio"]
        )

        self.breakout_entry_config = _breakout_entry_from_dict(
            breakout["entry"]
        )
        self.breakout_confidence_policy = BreakoutV7ConfidencePolicy(
            **breakout["confidence_policy"]
        )
        self.breakout_score_allocation = BreakoutV8ScoreAllocation(
            **breakout["score_allocation"]
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
        self.grid_market_overlay = GridMarketOverlay(
            **grid["market_overlay"]
        )
        self.grid_signal_config = TrendGridConfig(**grid["signal"])
        self.grid_portfolio_config = GridPortfolioConfig(
            **grid["portfolio"]
        )
        self.grid_confidence_policy = GridV5ConfidencePolicy(
            **grid["confidence_policy"]
        )
        self.grid_campaign_policy = GridV6CampaignPolicy(
            **grid["campaign_policy"]
        )

        self.config_hash = combined_v8_grid_v6_shadow_config_hash(config)
        self.state_path = Path(self.shadow.state_path)
        self.event_log_path = Path(self.shadow.event_log_path)
        self.report_path = Path(self.shadow.report_path)
        self.state = self._load_or_create_state()
        self.state.setdefault("grid_execution_profile", None)
        self._market_context: dict[
            int, dict[str, V4MarketSnapshot]
        ] = {}
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
                source_strategy=COMBINED_V8_GRID_V6_NAME,
            )
            self.state["sample_initialized_event_logged"] = True
            self._persist_state()

    def validate_startup(self) -> None:
        self.validate_startup_settings_only()
        if self.config.exchange.environment != "mainnet":
            raise RuntimeError(
                "Breakout-v8/Grid-v6 shadow requires mainnet public data"
            )
        if self.config.risk.cost_experiment != "full_cost":
            raise RuntimeError(
                "Breakout-v8/Grid-v6 shadow requires full_cost"
            )
        if self.config.risk.backtest_mode != "conservative":
            raise RuntimeError(
                "Breakout-v8/Grid-v6 shadow requires conservative mode"
            )
        if not self.config.risk.funding_enabled:
            raise RuntimeError(
                "Breakout-v8/Grid-v6 shadow requires funding"
            )
        if self.config.risk.dynamic_slippage_enabled:
            raise RuntimeError(
                "dynamic slippage differs from the frozen backtest"
            )
        expected_costs = (2.0, 5.0, 2.0, 0.0002, 0.0005)
        actual_costs = (
            self.config.risk.market_slippage_bps,
            self.config.risk.stop_slippage_bps,
            self.config.risk.take_profit_slippage_bps,
            self.config.risk.maker_fee_rate,
            self.config.risk.taker_fee_rate,
        )
        if actual_costs != expected_costs:
            raise RuntimeError(
                "GUI costs differ from the v8/v6 backtest costs"
            )
        if self.config.risk.starting_capital_usdt != 200.0:
            raise RuntimeError(
                "Breakout-v8/Grid-v6 sample must start from 200U"
            )
        if self.config.trading.leverage != 10:
            raise RuntimeError(
                "Breakout-v8/Grid-v6 GUI display leverage must remain 10x"
            )
        if (
            self.shadow.max_open_positions != 2
            or self.config.trading.max_open_positions != 2
        ):
            raise RuntimeError(
                "Breakout-v8/Grid-v6 shadow requires global max two"
            )
        if self.shadow.max_open_positions_per_strategy != 1:
            raise RuntimeError(
                "Breakout-v8/Grid-v6 requires one slot per strategy"
            )
        if self.breakout_portfolio_config.max_open_positions != 1:
            raise RuntimeError(
                "Breakout v8 source must remain max one position"
            )
        if self.grid_portfolio_config.max_open_campaigns != 1:
            raise RuntimeError(
                "Grid v6 source must remain max one campaign"
            )

        expected_priority = (BREAKOUT_KEY, GRID_KEY)
        if tuple(self.shadow.entry_priority) != expected_priority:
            raise RuntimeError(
                f"entry priority must remain {expected_priority}"
            )
        if tuple(self.global_config.entry_priority) != expected_priority:
            raise RuntimeError("combined source entry priority changed")
        if (
            self.global_config.max_open_positions
            != self.shadow.max_open_positions
        ):
            raise RuntimeError(
                "combined source/live position limits differ"
            )
        if not math.isclose(
            self.global_config.max_gross_notional_multiple,
            self.shadow.max_gross_notional_multiple,
        ):
            raise RuntimeError(
                "combined source/live notional limits differ"
            )
        if not math.isclose(
            self.global_config.hard_drawdown_stop_pct,
            self.shadow.hard_drawdown_stop_pct,
        ):
            raise RuntimeError(
                "combined source/live drawdown limits differ"
            )
        if (
            self.global_config.allow_same_symbol_across_strategies
            or self.shadow.allow_same_symbol_across_strategies
        ):
            raise RuntimeError(
                "Breakout-v8/Grid-v6 must forbid same-symbol overlap"
            )

        combined = self.source_bundle["combined_payload"]
        breakout = self.source_bundle["breakout_payload"]
        grid = self.source_bundle["grid_payload"]
        if combined.get("strategy_name") != COMBINED_V8_GRID_V6_NAME:
            raise RuntimeError("combined v8/v6 strategy name changed")
        if combined.get("status") != "gui_dry_run_frozen":
            raise RuntimeError("combined v8/v6 source is not GUI-frozen")
        if float(combined.get("initial_equity", 0.0)) != 200.0:
            raise RuntimeError("combined v8/v6 source equity changed")
        if self.shadow.strategy_name != COMBINED_V8_GRID_V6_NAME:
            raise RuntimeError("GUI combined v8/v6 strategy name changed")
        if (
            self.shadow.frozen_version
            != BREAKOUT_V8_GRID_V6_SHADOW_VERSION
        ):
            raise RuntimeError("GUI combined v8/v6 version changed")
        if breakout.get("strategy_name") != VOLATILITY_BREAKOUT_V8_NAME:
            raise RuntimeError("Breakout v8 source strategy changed")
        if not str(
            breakout.get("selection_status", "")
        ).startswith("strict_robust_improvement"):
            raise RuntimeError(
                "Breakout v8 source is not the selected robust candidate"
            )
        if grid.get("strategy_name") != TREND_GRID_V6_NAME:
            raise RuntimeError("Grid v6 source strategy changed")
        if not str(
            grid.get("selection_status", "")
        ).startswith("strict_robust_improvement"):
            raise RuntimeError(
                "Grid v6 source is not the selected robust candidate"
            )
        if (
            self.breakout_shadow.strategy_name
            != VOLATILITY_BREAKOUT_V8_NAME
        ):
            raise RuntimeError("GUI Breakout v8 component name changed")
        if (
            self.breakout_shadow.frozen_version
            != BREAKOUT_V8_COMPONENT_VERSION
        ):
            raise RuntimeError("GUI Breakout v8 component version changed")

        symbols = tuple(self.shadow.enabled_symbols)
        if symbols != tuple(UNIVERSE_50):
            raise RuntimeError(
                "Breakout-v8/Grid-v6 requires the frozen 50 symbols"
            )
        if symbols != tuple(combined.get("symbols", ())):
            raise RuntimeError(
                "combined source and GUI universes differ"
            )
        if symbols != tuple(self.breakout_shadow.enabled_symbols):
            raise RuntimeError(
                "combined and Breakout v8 universes differ"
            )
        if tuple(self.config.trading.symbols) != symbols:
            raise RuntimeError(
                "GUI trading symbols differ from frozen universe"
            )
        if tuple(
            self.config.trading.entry_symbols
            or self.config.trading.symbols
        ) != symbols:
            raise RuntimeError(
                "GUI entry symbols differ from frozen universe"
            )

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
                raise RuntimeError(
                    f"Breakout v8 GUI/source differs: {field}"
                )
        managed = self.breakout_managed_profile
        display_values = {
            "stop_atr_multiple": managed.core_stop_atr,
            "take_profit_r": 60.0,
            "fail_fast_minutes": managed.core_fail_fast_minutes,
            "fail_fast_min_mfe_r": managed.core_fail_fast_min_mfe_r,
            "fail_fast_max_current_r": (
                managed.core_fail_fast_max_current_r
            ),
            "max_holding_minutes": managed.core_max_holding_minutes,
            "risk_per_trade_pct": managed.core_strong_risk_pct,
            "max_trade_risk_pct": (
                self.breakout_portfolio_config.max_trade_risk_pct
            ),
            "max_open_positions": 1,
            "max_daily_trades": (
                self.breakout_portfolio_config.max_daily_trades
            ),
            "symbol_cooldown_minutes": (
                self.breakout_portfolio_config.symbol_cooldown_minutes
            ),
            "max_notional_multiple": (
                self.breakout_portfolio_config.max_notional_multiple
            ),
            "ranking_mode": managed.ranking_mode,
            "long_risk_multiplier": managed.core_long_multiplier,
            "short_risk_multiplier": managed.core_short_multiplier,
        }
        for field, expected in display_values.items():
            if getattr(self.breakout_shadow, field) != expected:
                raise RuntimeError(
                    f"Breakout v8 GUI display differs: {field}"
                )
        if self.breakout_entry_timing.confirmation_minutes != 0:
            raise RuntimeError(
                "selected Breakout v8 requires immediate entry"
            )
        if managed.drawdown_scope != "global":
            raise RuntimeError(
                "selected Breakout v8 drawdown scope changed"
            )
        if not self.grid_confidence_policy.reject_weak_tier:
            raise RuntimeError(
                "selected Grid v6 must retain the v5 weak-tier rejection"
            )

        self.breakout_entry_config.validate()
        self.breakout_confidence_policy.validate()
        self.breakout_score_allocation.validate()
        self.breakout_entry_timing.validate()
        self.grid_entry_gate.validate()
        self.grid_market_overlay.validate()
        self.grid_confidence_policy.validate()
        self.grid_campaign_policy.validate()

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
            raise RuntimeError(
                "all unrelated GUI strategies must remain disabled"
            )
        if self.state.get("grid_campaign") and not self.state.get(
            "grid_execution_profile"
        ):
            raise RuntimeError(
                "open Grid v6 campaign is missing its frozen profile"
            )
        self.client.ping()

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
            float(candidate.signal.direction.value)
            * snapshot.symbol_ema55_atr
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
        multiplier, score, lane = breakout_v8_risk_multiplier(
            candidate,
            snapshot,
            self.breakout_confidence_policy,
            self.breakout_score_allocation,
        )
        if multiplier <= 0.0:
            return None
        signal = replace(
            self.breakout_signal_config,
            stop_atr_multiple=managed.core_stop_atr,
            take_profit_r=60.0,
            max_holding_minutes=managed.core_max_holding_minutes,
            fail_fast_minutes=managed.core_fail_fast_minutes,
            fail_fast_min_mfe_r=managed.core_fail_fast_min_mfe_r,
            fail_fast_max_current_r=(
                managed.core_fail_fast_max_current_r
            ),
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
            risk_per_trade_pct=risk * multiplier * governor,
            long_risk_multiplier=managed.core_long_multiplier,
            short_risk_multiplier=managed.core_short_multiplier,
            ranking_mode=managed.ranking_mode,
        )
        profile = BreakoutV6ExecutionProfile(
            f"v8_{lane}",
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
        accepted, tier, score, factor = grid_v6_entry_decision(
            candidate,
            self._snapshot_for(candidate),
            self.grid_confidence_policy,
            self.grid_campaign_policy,
        )
        if not accepted:
            return None
        confidence = self.grid_confidence_policy
        if tier == "strong":
            v5_risk = confidence.strong_risk_multiplier
        elif tier == "weak":
            v5_risk = confidence.weak_risk_multiplier
        else:
            v5_risk = confidence.standard_risk_multiplier
        policy = self.grid_campaign_policy
        profile = GridV5ExecutionProfile(
            tier=f"v6_{tier}_score_{score}",
            signal=replace(
                self.grid_signal_config,
                grid_target_spacing=policy.target_spacing,
                campaign_loss_limit_r=policy.campaign_loss_limit_r,
                max_campaign_minutes=policy.max_campaign_minutes,
                campaign_take_profit_r=policy.campaign_take_profit_r,
                profit_lock_activation_r=(
                    policy.profit_lock_activation_r
                ),
                profit_giveback_r=policy.profit_giveback_r,
            ),
            portfolio=replace(
                self.grid_portfolio_config,
                risk_per_campaign_pct=(
                    self.grid_portfolio_config.risk_per_campaign_pct
                    * v5_risk
                    * factor
                ),
                max_campaign_risk_pct=(
                    policy.maximum_campaign_risk_pct
                ),
            ),
        )
        profile.signal.validate()
        return profile

    def _build_entry_candidates(
        self,
        candles_by_symbol: dict[str, list[Any]],
        latest_available: datetime,
    ) -> tuple[list[Candidate], list[GridCandidate], float]:
        breakout_rows, grid_rows, btc_return = super()._build_entry_candidates(
            candles_by_symbol, latest_available
        )
        if grid_rows:
            minute = grid_rows[0].entry_minute
            snapshots = self._market_context.get(
                minute - minute % 60, {}
            )
            mode = self.grid_market_overlay.ranking_mode
            grid_rows.sort(
                key=lambda row: (
                    -_ranking_value(
                        row,
                        snapshots.get(row.signal.symbol),
                        mode,
                    ),
                    row.signal.symbol,
                )
            )
        return breakout_rows, grid_rows, btc_return

    def _candidate_reject_reason(
        self,
        strategy: str,
        candidate: Candidate | GridCandidate,
        now: datetime,
    ) -> str | None:
        reason = (
            CombinedVolatilityTrendGridShadowTrader
            ._candidate_reject_reason(self, strategy, candidate, now)
        )
        if reason is not None:
            return reason
        if strategy == BREAKOUT_KEY:
            equity = self.snapshot_account(fetch_mark=False).equity
            if self._select_breakout_profile(candidate, equity) is None:
                return "v8_score_allocation_rejected"
        elif self._select_grid_profile(candidate) is None:
            return "grid_v6_campaign_policy_rejected"
        return None

    def log(self, message: str) -> None:
        self.logger(
            message.replace("Breakout v7", "Breakout v8").replace(
                "Grid v5", "Grid v6"
            )
        )

    def _append_event(self, event_type: str, **payload: Any) -> None:
        if "v7_lane" in payload:
            payload["v8_lane"] = payload.pop("v7_lane")
        if "v5_tier" in payload:
            payload["v6_tier"] = payload.pop("v5_tier")
        if (
            payload.get("strategy") == BREAKOUT_KEY
            and "v6_lane" in payload
        ):
            payload["v8_lane"] = payload.pop("v6_lane")
        CombinedVolatilityTrendGridShadowTrader._append_event(
            self, event_type, **payload
        )

    def acceptance_report(
        self,
        account: AccountSnapshot | None = None,
    ) -> dict[str, Any]:
        report = (
            CombinedVolatilityTrendGridShadowTrader.acceptance_report(
                self, account
            )
        )
        trades: list[dict[str, Any]] = []
        for source in report.get("trades", ()):
            row = dict(source)
            if row.get("strategy") == BREAKOUT_KEY:
                lane = row.get("v8_lane", row.get("v6_lane"))
                if lane is not None:
                    row["v8_lane"] = lane
            elif row.get("strategy") == GRID_KEY:
                tier = row.get("v6_tier", row.get("v5_tier"))
                if tier is not None:
                    row["v6_tier"] = tier
            trades.append(row)
        combined = self.source_bundle["combined_payload"]
        report.update(
            {
                "important_note": (
                    "Breakout v8 and Grid v6 historical results are frozen "
                    "research references. This report contains only new "
                    "dry-run observations and cannot authorize live trading."
                ),
                "historical_reference": combined.get(
                    "historical_reference", {}
                ),
                "source_configs": {
                    "breakout_v8": str(
                        self.source_bundle["breakout_path"]
                    ),
                    "grid_v6": str(self.source_bundle["grid_path"]),
                    "shared_account": str(
                        self.source_bundle["combined_path"]
                    ),
                },
                "breakout_v8_policy": {
                    "confidence": (
                        self.breakout_confidence_policy.as_dict()
                    ),
                    "score_allocation": (
                        self.breakout_score_allocation.as_dict()
                    ),
                },
                "grid_v6_policy": {
                    "confidence": (
                        self.grid_confidence_policy.as_dict()
                    ),
                    "campaign": self.grid_campaign_policy.as_dict(),
                },
                "trades": trades,
                "by_breakout_v8_lane": self._group_trades(
                    [
                        row
                        for row in trades
                        if row.get("strategy") == BREAKOUT_KEY
                    ],
                    "v8_lane",
                ),
                "by_grid_v6_tier": self._group_trades(
                    [
                        row
                        for row in trades
                        if row.get("strategy") == GRID_KEY
                    ],
                    "v6_tier",
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
        description="Breakout v8 plus Grid v6 max-two dry-run shadow"
    )
    parser.add_argument(
        "--config",
        default="config.gui.breakout-v8-grid-v6-max2-shadow.json",
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
    trader = CombinedBreakoutV8GridV6ShadowTrader(
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
