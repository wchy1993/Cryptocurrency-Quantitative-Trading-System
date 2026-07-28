from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable

from .binance_client import (
    BinanceFuturesClient,
    BinanceRateLimitError,
    SymbolRules,
)
from .binance_streams import BinanceFuturesStreamCache
from .combined_breakout_v7_grid_v5_shadow import (
    _breakout_entry_from_dict,
    _grid_profile_from_dict,
    _grid_profile_to_dict,
    _v7_position_from_dict,
    _v7_position_to_dict,
)
from .combined_breakout_v8_grid_v6_shadow import (
    BREAKOUT_V8_COMPONENT_VERSION,
    COMBINED_V8_GRID_V6_NAME,
    CombinedBreakoutV8GridV6ShadowTrader,
    _load_v8_v6_source_bundle,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    NOTIONAL_HEADROOM_SAFETY,
    CombinedPortfolioConfig,
)
from .combined_volatility_trend_grid_shadow import (
    _closed_candles,
    _compact_series,
    _grid_campaign_from_dict,
    _grid_campaign_to_dict,
    _jsonable,
    _parse_time,
    _utc_now,
)
from .live_config import (
    LiveAppConfig,
    load_live_config,
)
from .live_trader import AccountSnapshot, LivePosition
from .models import Candle, Direction
from .risk import execution_config_from_live_config
from .trend_grid import TrendGridConfig, build_trend_grid_timeline
from .trend_grid_optimize import (
    GridCandidate,
    GridLot,
    GridPortfolioConfig,
    _create_campaign,
)
from .trend_grid_v3_optimize import GridMarketOverlay
from .trend_grid_v4 import GridV4EntryGate
from .trend_grid_v5 import GridV5ConfidencePolicy
from .trend_grid_v5_engine import GridV5ExecutionProfile
from .trend_grid_v6 import (
    TREND_GRID_V6_NAME,
    GridV6CampaignPolicy,
)
from .volatility_breakout import VolatilityBreakoutConfig
from .volatility_breakout_exit_protection import (
    _process_protected_bar,
    _protected_position,
)
from .volatility_breakout_optimize import (
    Candidate,
    PortfolioSearchConfig,
    UNIVERSE_50,
    _entry_position,
    minute_token,
)
from .volatility_breakout_v4_research import V4MarketSnapshot
from .volatility_breakout_v6_core_runner_optimize import ManagedLaneProfile
from .volatility_breakout_v7 import (
    BreakoutV7ConfidencePolicy,
    BreakoutV7EntryTiming,
)
from .volatility_breakout_v8 import (
    VOLATILITY_BREAKOUT_V8_NAME,
    BreakoutV8ScoreAllocation,
)


BREAKOUT_V8_GRID_V6_LIVE_VERSION = (
    "breakout_v8_grid_v6_max2_live_risk1_20260728"
)
BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION = (
    "binance_usdm_ws_weighted_v1"
)
LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS = 45.0
LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES = 200
LIVE_STATE_SCHEMA_VERSION = 1
MARKET_FILL_CONFIRM_ATTEMPTS = 5
MARKET_FILL_CONFIRM_DELAY_SECONDS = 0.20
FLAT_EMERGENCY_RECOVERY_VERSION = (
    "verified_flat_emergency_round_trip_v1"
)
PROTECTIVE_STOP_INCIDENT_RECOVERY_VERSION = (
    "verified_flat_protective_stop_incident_v1"
)
LIVE_REASON_TOKEN = "combined_breakout_v8_grid_v6_live"
_TRANSPORT_ONLY_CONFIG_KEYS = {
    "transport_version",
    "websocket_enabled",
    "websocket_startup_timeout_seconds",
    "websocket_stale_seconds",
    "listen_key_keepalive_seconds",
    "rest_reconcile_interval_seconds",
    "rest_reconcile_fallback_seconds",
    "request_weight_limit",
    "request_weight_soft_limit_ratio",
    "rate_limit_default_cooldown_seconds",
    "transport_acceptance_required",
    "transport_acceptance_report_path",
}
_TERMINAL_ORDER_STATUSES = {
    "FILLED",
    "CANCELED",
    "CANCELLED",
    "REJECTED",
    "EXPIRED",
    "EXPIRED_IN_MATCH",
    "FINISHED",
}


def _planner_config(config: LiveAppConfig) -> LiveAppConfig:
    live = config.combined_breakout_v8_grid_v6_live
    shadow = replace(
        config.combined_volatility_trend_grid_shadow,
        enabled=False,
        shadow_only=True,
        strategy_name=live.strategy_name,
        enabled_symbols=live.enabled_symbols,
        source_combined_config_path=live.source_combined_config_path,
    )
    return replace(config, combined_volatility_trend_grid_shadow=shadow)


def combined_v8_grid_v6_live_config_hash(
    config: LiveAppConfig,
) -> str:
    bundle = _load_v8_v6_source_bundle(_planner_config(config))
    live_payload = asdict(config.combined_breakout_v8_grid_v6_live)
    # Arming and confirmation are runtime authorization, not strategy
    # identity.  They may be cleared after stopping without orphaning the
    # persistent reconciliation ledger.
    for key in (
        "enabled",
        "armed",
        "live_confirmation_text",
    ):
        live_payload.pop(key, None)
    for key in _TRANSPORT_ONLY_CONFIG_KEYS:
        live_payload.pop(key, None)
    trading_payload = asdict(config.trading)
    trading_payload.pop("mainnet_confirmation_text", None)
    payload = {
        "live": live_payload,
        "source_hashes": bundle["hashes"],
        "environment": config.exchange.environment,
        "trading": trading_payload,
        "risk": asdict(config.risk),
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _round_reduce_only_quantity(
    rules: SymbolRules,
    quantity: float,
) -> str:
    """Round an exchange position without losing a float-artifact step.

    Entry sizing must always round down.  A reduce-only order is different:
    its input should already be an exchange-valid position quantity.  Sums of
    valid float lots can land infinitesimally below a step (for example,
    30.9 * 3 -> 92.69999999999999).  Snap only that microscopic artifact to
    the nearest step; genuinely off-step quantities still round down.
    """

    value = Decimal(str(quantity))
    if value <= 0:
        return "0"
    step = rules.quantity_step
    units = value / step
    nearest = units.to_integral_value(rounding=ROUND_HALF_UP)
    tolerance = max(
        Decimal("1e-9"),
        abs(units) * Decimal("1e-15"),
    )
    if abs(units - nearest) <= tolerance:
        rounded_units = nearest
    else:
        rounded_units = units.to_integral_value(rounding=ROUND_DOWN)
    return _decimal_string(rounded_units * step)


def _exact_quantity_sum(values: Any) -> float:
    return float(
        sum(
            (Decimal(str(value)) for value in values),
            Decimal("0"),
        )
    )


def _order_client_id(order: dict[str, Any]) -> str:
    return str(
        order.get("clientOrderId")
        or order.get("origClientOrderId")
        or ""
    )


def _algo_client_id(order: dict[str, Any]) -> str:
    return str(order.get("clientAlgoId") or "")


def _transport_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "crypto_scalper/binance_rate_limit.py",
        "crypto_scalper/binance_streams.py",
        "crypto_scalper/binance_client.py",
        "crypto_scalper/combined_breakout_v8_grid_v6_live.py",
        (
            "crypto_scalper/"
            "combined_breakout_v8_grid_v6_live_acceptance.py"
        ),
    )
    return {
        relative: hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


class CombinedBreakoutV8GridV6LiveTrader(
    CombinedBreakoutV8GridV6ShadowTrader
):
    """Exchange-backed execution for the frozen v8/v6 signal bundle.

    The class inherits only the point-in-time signal construction and profile
    selection.  Position truth comes from Binance, and every exit submitted by
    this runner is reduce-only.  The checked-in live configuration is locked,
    so constructing this class never authorizes an order.
    """

    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: Callable[[str], None] | None = None,
        account_callback: Callable[[AccountSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.live = config.combined_breakout_v8_grid_v6_live
        # The inherited point-in-time scanner uses ``self.shadow`` as its
        # generic execution envelope.  The live dataclass intentionally
        # exposes the same scan/portfolio fields.
        self.shadow = self.live
        self.breakout_shadow = config.dual_thrust_shadow
        self.client = client
        self.logger = logger or (lambda _message: None)
        self.account_callback = account_callback
        self.execution = execution_config_from_live_config(
            config, cost_experiment="full_cost", mode="conservative"
        )
        self.source_bundle = _load_v8_v6_source_bundle(
            _planner_config(config)
        )
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

        self.config_hash = combined_v8_grid_v6_live_config_hash(config)
        path_values = {
            "environment": self.config.exchange.environment.lower()
        }
        self.state_path = Path(
            self.live.state_path.format(**path_values)
        )
        self.event_log_path = Path(
            self.live.event_log_path.format(**path_values)
        )
        self.report_path = Path(
            self.live.report_path.format(**path_values)
        )
        self.state = self._load_or_create_state()
        self._market_context: dict[
            int, dict[str, V4MarketSnapshot]
        ] = {}
        self._last_marks: dict[str, float] = {}
        self._unsupported_symbols: set[str] = set()
        self._last_heartbeat = 0.0
        self._runtime_lock_handle: Any | None = None
        self._prepared_symbols: set[str] = set()
        self._validated = False
        self.stream_cache: BinanceFuturesStreamCache | None = None
        self._last_account: AccountSnapshot | None = None
        self._last_full_reconcile_monotonic = 0.0
        self._last_user_stream_revision = 0
        self._force_reconcile = True

    # ------------------------------------------------------------------
    # Startup and state
    # ------------------------------------------------------------------
    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            state = json.loads(
                self.state_path.read_text(encoding="utf-8")
            )
            if state.get("schema_version") != LIVE_STATE_SCHEMA_VERSION:
                raise RuntimeError("v8/v6 live state schema mismatch")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError(
                    "v8/v6 live state belongs to another config; "
                    "do not mix live ledgers"
                )
            state.setdefault(
                "transport_version",
                BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION,
            )
            state.setdefault("rate_limit_cooldown_until", None)
            state.setdefault("last_stream_health", {})
            state.setdefault("transport_acceptance", {})
            state.setdefault("recovery_journal", [])
            return state
        now = _utc_now().isoformat()
        return {
            "schema_version": LIVE_STATE_SCHEMA_VERSION,
            "config_hash": self.config_hash,
            "frozen_version": self.live.frozen_version,
            "transport_version": (
                BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION
            ),
            "created_at": now,
            "started_at": now,
            "starting_equity": 0.0,
            "peak_equity": 0.0,
            "session_date": None,
            "session_start_equity": 0.0,
            "last_equity": 0.0,
            "max_drawdown_pct": 0.0,
            "circuit_breaker": False,
            "circuit_reason": "",
            "operational_halt": False,
            "halt_reason": "",
            "consecutive_api_failures": 0,
            "reconcile_failures": 0,
            "last_reconciled_at": None,
            "rate_limit_cooldown_until": None,
            "last_stream_health": {},
            "transport_acceptance": {},
            "last_scanned_available_time": None,
            "seen_events": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "daily_entries": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "cooldown_until": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "breakout_position": None,
            "breakout_exchange": None,
            "breakout_last_processed_minute": None,
            "grid_campaign": None,
            "grid_execution_profile": None,
            "grid_exchange": None,
            "grid_last_processed_minute": None,
            "grid_last_regime_minute": None,
            "protective_orders": {},
            "pending_orders": {},
            "order_journal": [],
            "recovery_journal": [],
            "trades": [],
            "candidate_count": {BREAKOUT_KEY: 0, GRID_KEY: 0},
            "rejected": {BREAKOUT_KEY: {}, GRID_KEY: {}},
            "sample_integrity_errors": [],
        }

    def validate_startup_settings_only(self) -> None:
        live = self.live
        if not live.enabled:
            raise RuntimeError("v8/v6 LIVE is disabled in its independent config")
        if not live.armed:
            raise RuntimeError(
                "v8/v6 LIVE is locked: set armed=true only for an intended run"
            )
        if self.config.trading.dry_run:
            raise RuntimeError(
                "v8/v6 live runner cannot use a dry-run trading section"
            )
        if (
            not self.config.trading.require_one_way_mode
            or not self.config.trading.reduce_only_exit
            or not self.config.trading.use_protective_orders
            or self.config.trading.working_type.upper() != "MARK_PRICE"
        ):
            raise RuntimeError(
                "v8/v6 LIVE requires One-way validation, reduceOnly exits, "
                "protective orders, and MARK_PRICE triggers"
            )
        client_environment = getattr(self.client, "environment", None)
        if (
            client_environment is not None
            and str(client_environment).lower()
            != self.config.exchange.environment.lower()
        ):
            raise RuntimeError("client environment differs from LIVE config")
        if (
            live.live_confirmation_text
            != live.required_live_confirmation_text
        ):
            raise RuntimeError(
                "v8/v6 LIVE confirmation text is missing or incorrect"
            )
        if (
            live.required_live_confirmation_text
            != "CONFIRM_BREAKOUT_V8_GRID_V6_LIVE"
            or live.runtime_confirmation_text != "RUN_LIVE_NOW"
        ):
            raise RuntimeError("v8/v6 LIVE confirmation contract changed")
        if self.config.exchange.environment == "mainnet":
            if not self.config.trading.require_mainnet_confirmation:
                raise RuntimeError(
                    "mainnet confirmation requirement cannot be disabled"
                )
            if (
                self.config.trading.mainnet_confirmation_text
                != "CONFIRM_MAINNET"
            ):
                raise RuntimeError(
                    "mainnet live trading requires CONFIRM_MAINNET"
                )
        if not self.client.api_key or not self.client.api_secret:
            raise RuntimeError("v8/v6 LIVE requires API credentials")
        if self.config.trading.leverage != 10:
            raise RuntimeError("v8/v6 LIVE requires the frozen 10x leverage")
        if self.config.trading.margin_type.upper() != "CROSSED":
            raise RuntimeError("v8/v6 LIVE requires CROSSED margin")
        if (
            live.max_open_positions != 2
            or live.max_open_positions_per_strategy != 1
            or self.config.trading.max_open_positions != 2
        ):
            raise RuntimeError("v8/v6 LIVE requires global max2 / sleeve max1")
        if live.allow_same_symbol_across_strategies:
            raise RuntimeError("v8/v6 LIVE must forbid same-symbol overlap")
        if tuple(live.entry_priority) != (BREAKOUT_KEY, GRID_KEY):
            raise RuntimeError("v8/v6 LIVE entry priority changed")
        if not (0.0 < live.risk_scale <= 1.0):
            raise RuntimeError("v8/v6 LIVE risk_scale must be in (0, 1]")
        if not (0.0 < live.max_gross_notional_multiple <= 9.0):
            raise RuntimeError(
                "v8/v6 LIVE notional envelope must be in (0, 9]"
            )
        if not (
            0.0 < live.max_daily_loss_pct <= 1.0
            and 0.0 < live.max_drawdown_pct <= 1.0
        ):
            raise RuntimeError("v8/v6 LIVE loss circuits must be in (0, 1]")
        if (
            live.max_consecutive_api_failures <= 0
            or live.max_reconcile_failures <= 0
            or live.pending_order_resolution_seconds <= 0
            or live.max_market_data_age_seconds <= 0
        ):
            raise RuntimeError(
                "v8/v6 LIVE failure and freshness thresholds must be positive"
            )
        if (
            live.transport_version
            != BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION
            or not live.websocket_enabled
        ):
            raise RuntimeError(
                "v8/v6 LIVE requires the frozen WebSocket transport"
            )
        if (
            live.websocket_startup_timeout_seconds <= 0
            or live.websocket_stale_seconds <= 0
            or live.listen_key_keepalive_seconds <= 0
            or live.rest_reconcile_interval_seconds <= 0
            or live.rest_reconcile_fallback_seconds <= 0
            or live.rest_reconcile_fallback_seconds
            > live.rest_reconcile_interval_seconds
            or live.request_weight_limit != 2_400
            or not (
                0.10
                <= live.request_weight_soft_limit_ratio
                <= 0.75
            )
            or live.rate_limit_default_cooldown_seconds < 60
        ):
            raise RuntimeError(
                "v8/v6 LIVE transport safety limits are invalid"
            )
        if (
            live.transport_acceptance_required
            and not live.transport_acceptance_report_path
        ):
            raise RuntimeError(
                "v8/v6 LIVE requires a transport acceptance report"
            )
        if not live.require_protective_stop:
            raise RuntimeError("v8/v6 LIVE cannot disable protective stops")
        if (
            live.grid_deeper_entry_mode
            != "software_market_after_touch"
        ):
            raise RuntimeError(
                "v8/v6 LIVE only permits software-triggered deeper grid "
                "entries so a stopped campaign cannot be reopened by a "
                "leftover non-reduceOnly limit order"
            )
        if not live.strict_dedicated_account:
            raise RuntimeError(
                "first v8/v6 LIVE version requires a dedicated account"
            )
        if not live.order_id_prefix or len(live.order_id_prefix) > 8:
            raise RuntimeError("live order_id_prefix must be 1-8 characters")

        symbols = tuple(live.enabled_symbols)
        if symbols != tuple(UNIVERSE_50):
            raise RuntimeError("v8/v6 LIVE requires the frozen 50 symbols")
        if tuple(self.config.trading.symbols) != symbols:
            raise RuntimeError("live trading symbols differ from frozen 50")
        if tuple(
            self.config.trading.entry_symbols
            or self.config.trading.symbols
        ) != symbols:
            raise RuntimeError("live entry symbols differ from frozen 50")

        combined = self.source_bundle["combined_payload"]
        breakout = self.source_bundle["breakout_payload"]
        grid = self.source_bundle["grid_payload"]
        if (
            self.source_bundle["hashes"]["combined"]
            != live.source_combined_config_sha256
        ):
            raise RuntimeError(
                "combined v8/v6 source hash differs from the LIVE pin"
            )
        if live.strategy_name != COMBINED_V8_GRID_V6_NAME:
            raise RuntimeError("v8/v6 LIVE strategy name changed")
        if live.frozen_version != BREAKOUT_V8_GRID_V6_LIVE_VERSION:
            raise RuntimeError("v8/v6 LIVE version changed")
        if combined.get("strategy_name") != COMBINED_V8_GRID_V6_NAME:
            raise RuntimeError("combined source strategy changed")
        if combined.get("status") != "gui_dry_run_frozen":
            raise RuntimeError("combined v8/v6 source is not frozen")
        if float(combined.get("initial_equity", 0.0)) != 200.0:
            raise RuntimeError("combined v8/v6 source equity changed")
        if tuple(combined.get("symbols", ())) != symbols:
            raise RuntimeError("combined source universe changed")
        if breakout.get("strategy_name") != VOLATILITY_BREAKOUT_V8_NAME:
            raise RuntimeError("Breakout v8 source changed")
        if not str(
            breakout.get("selection_status", "")
        ).startswith("strict_robust_improvement"):
            raise RuntimeError("Breakout v8 source is not the frozen winner")
        if grid.get("strategy_name") != TREND_GRID_V6_NAME:
            raise RuntimeError("Grid v6 source changed")
        if not str(grid.get("selection_status", "")).startswith(
            "strict_robust_improvement"
        ):
            raise RuntimeError("Grid v6 source is not the frozen winner")
        if (
            self.global_config.max_open_positions != 2
            or self.global_config.allow_same_symbol_across_strategies
        ):
            raise RuntimeError("combined source portfolio contract changed")
        if (
            self.breakout_shadow.strategy_name
            != VOLATILITY_BREAKOUT_V8_NAME
            or self.breakout_shadow.frozen_version
            != BREAKOUT_V8_COMPONENT_VERSION
        ):
            raise RuntimeError("Breakout v8 display/source envelope changed")

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
            raise RuntimeError("unrelated strategies must remain disabled")

        self.breakout_entry_config.validate()
        self.breakout_confidence_policy.validate()
        self.breakout_score_allocation.validate()
        self.breakout_entry_timing.validate()
        self.grid_entry_gate.validate()
        self.grid_market_overlay.validate()
        self.grid_confidence_policy.validate()
        self.grid_campaign_policy.validate()

    def validate_transport_acceptance(self) -> None:
        if not self.live.transport_acceptance_required:
            return
        path = Path(self.live.transport_acceptance_report_path)
        if not path.exists():
            raise RuntimeError(
                "v8/v6 LIVE transport acceptance report is missing"
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise RuntimeError(
                "v8/v6 LIVE transport acceptance did not pass"
            )
        if (
            report.get("schema_version") != 2
            or report.get("environment") != "mainnet"
            or report.get("mode")
            != "MAINNET_DRY_RUN_NO_ORDER_WITH_USER_STREAM"
            or report.get("user_stream_required") is not True
            or report.get("criteria", {}).get(
                "user_stream_connected"
            )
            is not True
        ):
            raise RuntimeError(
                "v8/v6 LIVE transport acceptance did not verify "
                "the authenticated mainnet user stream"
            )
        if (
            _float(report.get("duration_seconds"))
            < LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS
            or int(report.get("cycles", 0))
            < LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES
        ):
            raise RuntimeError(
                "v8/v6 LIVE transport acceptance was not a full "
                "stress run"
            )
        if (
            report.get("transport_version")
            != BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION
            or report.get("live_config_hash") != self.config_hash
            or report.get("strategy_source_hashes")
            != self.source_bundle["hashes"]
        ):
            raise RuntimeError(
                "v8/v6 LIVE transport acceptance belongs to another build"
            )
        expected_hashes = _transport_code_hashes()
        if report.get("transport_code_hashes") != expected_hashes:
            raise RuntimeError(
                "v8/v6 LIVE transport code changed after acceptance"
            )
        self.state["transport_acceptance"] = {
            "report_path": str(path),
            "completed_at": report.get("completed_at"),
            "transport_code_hashes": expected_hashes,
        }

    def _start_streams(self, stop_event: threading.Event) -> None:
        if not isinstance(self.client, BinanceFuturesClient):
            raise RuntimeError(
                "v8/v6 LIVE WebSocket transport requires Binance client"
            )
        self.stream_cache = BinanceFuturesStreamCache(
            self.client,
            tuple(self.live.enabled_symbols),
            logger=self.log,
            stale_after_seconds=self.live.websocket_stale_seconds,
            listen_key_keepalive_seconds=(
                self.live.listen_key_keepalive_seconds
            ),
        )
        self.client.attach_market_stream_cache(self.stream_cache)
        self.stream_cache.start(stop_event, include_user_stream=True)
        ready = self.stream_cache.wait_until_ready(
            self.live.websocket_startup_timeout_seconds,
            require_user_stream=True,
        )
        health = self.stream_cache.health()
        self.state["last_stream_health"] = health
        if not ready:
            self._stop_streams()
            raise RuntimeError(
                "v8/v6 LIVE WebSocket startup failed: "
                f"market={health['market_connected']} "
                f"user={health['user_connected']} "
                f"market_error={health['last_market_error']} "
                f"user_error={health['last_user_error']}"
            )
        self._last_user_stream_revision = (
            self.stream_cache.user_revision
        )
        self._append_event(
            "websocket_transport_ready",
            market_connected=health["market_connected"],
            user_connected=health["user_connected"],
        )

    def _stop_streams(self) -> None:
        cache = self.stream_cache
        self.stream_cache = None
        if isinstance(self.client, BinanceFuturesClient):
            self.client.attach_market_stream_cache(None)
        if cache is not None:
            cache.stop()

    def validate_startup(self) -> None:
        self.validate_startup_settings_only()
        if self.config.trading.require_one_way_mode:
            if self.client.position_mode():
                raise RuntimeError(
                    "v8/v6 LIVE refuses Hedge Mode; switch to One-way first"
                )
        account = self.snapshot_account()
        if account.equity <= 0.0:
            raise RuntimeError("live account equity is not positive")
        self._initialize_equity_baseline(account.equity)
        account = self.reconcile(account=account, startup=True)
        if self.state["operational_halt"]:
            raise RuntimeError(
                f"startup reconciliation halted: {self.state['halt_reason']}"
            )
        self._preflight_test_order()
        self._last_account = account
        self._last_full_reconcile_monotonic = time.monotonic()
        if self.stream_cache is not None:
            self._last_user_stream_revision = (
                self.stream_cache.user_revision
            )
        self._validated = True
        self._append_event(
            "live_startup_validated",
            environment=self.config.exchange.environment,
            equity=account.equity,
            risk_scale=self.live.risk_scale,
            max_open_positions=self.live.max_open_positions,
        )
        self._persist_state()
        self._write_status(account)

    # ------------------------------------------------------------------
    # Account truth and risk circuit breakers
    # ------------------------------------------------------------------
    def snapshot_account(
        self, fetch_mark: bool = True
    ) -> AccountSnapshot:
        del fetch_mark
        payload = self.client.account()
        equity = _float(
            payload.get(
                "totalMarginBalance",
                payload.get("totalWalletBalance", 0.0),
            )
        )
        wallet = _float(payload.get("totalWalletBalance"), equity)
        available = _float(payload.get("availableBalance"))
        initial_margin = _float(payload.get("totalInitialMargin"))
        maintenance = _float(payload.get("totalMaintMargin"))
        unrealized = _float(payload.get("totalUnrealizedProfit"))
        positions: dict[str, LivePosition] = {}
        rows: list[LivePosition] = []
        for raw in payload.get("positions", ()):
            symbol = str(raw.get("symbol", "")).upper()
            if (
                symbol not in self.live.enabled_symbols
                and not self.live.strict_dedicated_account
            ):
                continue
            amount = _float(raw.get("positionAmt"))
            if abs(amount) <= 0.0:
                continue
            entry = _float(raw.get("entryPrice"))
            mark = _float(
                raw.get("markPrice")
                or raw.get("breakEvenPrice")
                or entry
            )
            reason = self._entry_reason_for_symbol(symbol)
            opened_at = self._opened_at_for_symbol(symbol)
            position = LivePosition(
                symbol=symbol,
                position_side=str(raw.get("positionSide", "BOTH")),
                direction=(
                    Direction.LONG if amount > 0.0 else Direction.SHORT
                ),
                quantity=abs(amount),
                entry_price=entry,
                mark_price=mark,
                notional=abs(
                    _float(raw.get("notional"), amount * mark)
                ),
                unrealized_pnl=_float(raw.get("unrealizedProfit")),
                leverage=int(
                    _float(raw.get("leverage"), self.config.trading.leverage)
                ),
                margin_type=str(
                    raw.get("marginType", self.config.trading.margin_type)
                ),
                liquidation_price=(
                    _float(raw.get("liquidationPrice"))
                    if raw.get("liquidationPrice") not in (None, "", "0")
                    else None
                ),
                entry_reason=reason,
                opened_at=opened_at,
            )
            positions[symbol] = position
            rows.append(position)
        snapshot = AccountSnapshot(
            equity=equity,
            wallet_balance=wallet,
            available_balance=available,
            initial_margin=initial_margin,
            maintenance_margin=maintenance,
            total_unrealized_pnl=unrealized,
            positions=positions,
            position_rows=tuple(rows),
            position_mode="One-way 单向",
        )
        self._last_account = snapshot
        return snapshot

    def _entry_reason_for_symbol(self, symbol: str) -> str:
        breakout = self.state.get("breakout_exchange") or {}
        if breakout.get("symbol") == symbol:
            return f"{LIVE_REASON_TOKEN}|breakout_v8"
        grid = self.state.get("grid_exchange") or {}
        if grid.get("symbol") == symbol:
            return f"{LIVE_REASON_TOKEN}|grid_v6"
        return ""

    def _opened_at_for_symbol(
        self, symbol: str
    ) -> datetime | None:
        for key in ("breakout_exchange", "grid_exchange"):
            row = self.state.get(key) or {}
            if row.get("symbol") != symbol or not row.get("opened_at"):
                continue
            value = _parse_time(str(row["opened_at"]))
            return (
                value.replace(tzinfo=timezone.utc)
                if value is not None
                else None
            )
        return None

    def _initialize_equity_baseline(self, equity: float) -> None:
        today = _utc_now().date().isoformat()
        if _float(self.state.get("starting_equity")) <= 0.0:
            self.state["starting_equity"] = equity
        if _float(self.state.get("peak_equity")) <= 0.0:
            self.state["peak_equity"] = equity
        if self.state.get("session_date") != today:
            self.state["session_date"] = today
            self.state["session_start_equity"] = equity
        self.state["last_equity"] = equity

    def _update_risk_circuits(self, account: AccountSnapshot) -> None:
        self._initialize_equity_baseline(account.equity)
        peak = max(_float(self.state["peak_equity"]), account.equity)
        self.state["peak_equity"] = peak
        drawdown = (
            (peak - account.equity) / peak if peak > 0.0 else 1.0
        )
        self.state["max_drawdown_pct"] = max(
            _float(self.state["max_drawdown_pct"]), drawdown
        )
        session = _float(self.state["session_start_equity"])
        daily_loss = (
            (session - account.equity) / session
            if session > 0.0
            else 1.0
        )
        if daily_loss >= self.live.max_daily_loss_pct:
            self._trip_circuit(
                f"daily_loss={daily_loss:.4%} "
                f">={self.live.max_daily_loss_pct:.4%}"
            )
        if drawdown >= self.live.max_drawdown_pct:
            self._trip_circuit(
                f"drawdown={drawdown:.4%} "
                f">={self.live.max_drawdown_pct:.4%}"
            )
        self.state["last_equity"] = account.equity

    def _trip_circuit(self, reason: str) -> None:
        if not self.state.get("circuit_breaker"):
            self._append_event("circuit_breaker", reason=reason)
            self.log(f"LIVE熔断：{reason}；保留保护单，仅停止新开仓")
        self.state["circuit_breaker"] = True
        self.state["circuit_reason"] = reason

    def _halt(self, reason: str) -> None:
        if not self.state.get("operational_halt"):
            self._append_event("operational_halt", reason=reason)
            self.log(f"LIVE运行锁定：{reason}")
            self.state["halt_reason"] = reason
        else:
            existing = str(self.state.get("halt_reason", ""))
            if reason not in existing:
                self._append_event(
                    "operational_halt_additional_reason", reason=reason
                )
                self.state["halt_reason"] = (
                    f"{existing} | {reason}" if existing else reason
                )[-2000:]
        self.state["operational_halt"] = True

    def clear_rate_limit_halt_after_acceptance(
        self,
        account: AccountSnapshot,
        standard_orders: list[dict[str, Any]],
        algo_orders: list[dict[str, Any]],
    ) -> None:
        """Clear only the known 429 lock after a flat-account acceptance.

        This deliberately cannot clear position, order, protection, direction,
        quantity, margin, or other operational faults.
        """

        self.validate_transport_acceptance()
        if not self.state.get("operational_halt"):
            return
        reason = str(self.state.get("halt_reason", "")).lower()
        rate_tokens = (
            "too many requests",
            "request weight",
            "http 429",
            "rate limit",
        )
        reasons = [
            item.strip() for item in reason.split("|") if item.strip()
        ]
        if not reasons or any(
            not any(token in item for token in rate_tokens)
            for item in reasons
        ):
            raise RuntimeError(
                "refusing to clear a non-rate-limit LIVE halt"
            )
        if (
            account.positions
            or standard_orders
            or algo_orders
            or self.state.get("pending_orders")
            or self.state.get("protective_orders")
            or self._local_positions()
        ):
            raise RuntimeError(
                "rate-limit halt can only be cleared while account and "
                "strategy ledger are completely flat"
            )
        previous = self.state["halt_reason"]
        self.state["operational_halt"] = False
        self.state["halt_reason"] = ""
        self.state["reconcile_failures"] = 0
        self.state["consecutive_api_failures"] = 0
        self.state["rate_limit_cooldown_until"] = None
        self.state["last_reconciled_at"] = _utc_now().isoformat()
        self._last_account = account
        self._last_full_reconcile_monotonic = time.monotonic()
        self._force_reconcile = True
        self._append_event(
            "rate_limit_halt_cleared_after_acceptance",
            previous_reason=previous,
            equity=account.equity,
            positions=0,
            standard_orders=0,
            algo_orders=0,
        )
        self._persist_state()
        self._write_status(account)

    def recover_verified_flat_emergency_round_trip(
        self,
        *,
        expected_entry_client_id: str,
        expected_emergency_client_id: str | None = None,
    ) -> dict[str, Any]:
        """Clear one reviewed entry/emergency-exit incident while flat.

        This recovery is intentionally narrower than a generic "unlock".
        It requires one pending initial-entry intent, deterministic and fully
        filled entry/exit orders, matching exchange trade history, two flat
        account snapshots, and no open orders or unrelated halt reason.  The
        incident remains permanently recorded in both the trade ledger and a
        dedicated recovery journal before the pending intent is removed.
        """

        existing_recovery = next(
            (
                row
                for row in self.state.get("recovery_journal", [])
                if row.get("entry_client_id")
                == expected_entry_client_id
            ),
            None,
        )
        if existing_recovery is not None:
            if (
                expected_entry_client_id
                not in self.state.get("pending_orders", {})
                and not self.state.get("operational_halt")
            ):
                return {
                    "cleared": True,
                    "already_recovered": True,
                    "record": existing_recovery,
                }
            raise RuntimeError(
                "recovery record exists but LIVE ledger is not clean"
            )
        if not self.state.get("operational_halt"):
            raise RuntimeError(
                "flat emergency recovery requires an operational halt"
            )
        if self.state.get("circuit_breaker"):
            raise RuntimeError(
                "flat emergency recovery refuses an active circuit breaker"
            )
        pending_orders = self.state.get("pending_orders", {})
        if set(pending_orders) != {expected_entry_client_id}:
            raise RuntimeError(
                "flat emergency recovery requires exactly the reviewed "
                "pending entry intent"
            )
        pending = pending_orders[expected_entry_client_id]
        if "model" not in pending or pending.get("role") == "grid_level":
            raise RuntimeError(
                "reviewed pending order is not an initial entry intent"
            )
        strategy = str(pending.get("strategy", ""))
        if strategy not in {BREAKOUT_KEY, GRID_KEY}:
            raise RuntimeError("pending entry strategy is unknown")
        symbol = str(pending.get("symbol", "")).upper()
        event_id = str(pending.get("event_id", ""))
        direction_name = str(pending.get("direction", ""))
        try:
            direction = Direction[direction_name]
        except KeyError as exc:
            raise RuntimeError(
                "pending entry direction is invalid"
            ) from exc
        if not symbol or not event_id:
            raise RuntimeError("pending entry identity is incomplete")
        emergency_client_id = self._client_id(
            strategy, event_id, "emerg", 0
        )
        if (
            expected_emergency_client_id is not None
            and expected_emergency_client_id != emergency_client_id
        ):
            raise RuntimeError(
                "emergency client ID differs from deterministic ledger ID"
            )

        reason_rows = [
            row.strip()
            for row in str(self.state.get("halt_reason", "")).split("|")
            if row.strip()
        ]
        allowed_reason_tokens = (
            "entry fill was not final",
            "entry fill confirmation failed",
            "protective stop placement failed",
            "filled entry has no exchange position",
        )
        if not reason_rows or any(
            not row.startswith(symbol)
            or not any(token in row.lower() for token in allowed_reason_tokens)
            for row in reason_rows
        ):
            raise RuntimeError(
                "operational halt contains an unrelated recovery reason"
            )
        if (
            self._local_positions()
            or self.state.get("protective_orders")
        ):
            raise RuntimeError(
                "flat emergency recovery found local position/protection data"
            )

        first_account = self.snapshot_account()
        first_standard = self.client.open_orders()
        first_algo = self.client.open_algo_orders()
        if first_account.positions or first_standard or first_algo:
            raise RuntimeError(
                "flat emergency recovery requires a globally flat account "
                "with no open orders"
            )

        entry_order = self.client.query_order(
            symbol,
            orig_client_order_id=expected_entry_client_id,
        )
        emergency_order = self.client.query_order(
            symbol,
            orig_client_order_id=emergency_client_id,
        )
        entry_quantity, entry_price = self._filled_order(entry_order)
        exit_quantity, exit_price = self._filled_order(emergency_order)
        rules = self.client.symbol_rules(symbol)
        tolerance = max(_float(rules.quantity_step) / 2.0, 1e-12)
        if abs(entry_quantity - exit_quantity) > tolerance:
            raise RuntimeError(
                "entry and emergency exit quantities do not match"
            )

        def exchange_bool(value: Any) -> bool:
            return value is True or str(value).strip().lower() == "true"

        entry_side = "BUY" if direction == Direction.LONG else "SELL"
        exit_side = "SELL" if direction == Direction.LONG else "BUY"
        if (
            str(entry_order.get("symbol", "")).upper() != symbol
            or _order_client_id(entry_order)
            != expected_entry_client_id
            or str(entry_order.get("side", "")).upper() != entry_side
            or exchange_bool(entry_order.get("reduceOnly"))
        ):
            raise RuntimeError("entry order does not match pending intent")
        if (
            str(emergency_order.get("symbol", "")).upper() != symbol
            or _order_client_id(emergency_order)
            != emergency_client_id
            or str(emergency_order.get("side", "")).upper() != exit_side
            or not exchange_bool(emergency_order.get("reduceOnly"))
        ):
            raise RuntimeError(
                "emergency order is not the matching reduce-only exit"
            )
        entry_order_id = str(entry_order.get("orderId", ""))
        emergency_order_id = str(emergency_order.get("orderId", ""))
        if (
            not entry_order_id
            or not emergency_order_id
            or entry_order_id == emergency_order_id
        ):
            raise RuntimeError("exchange order IDs are missing or duplicated")

        update_times = [
            int(_float(row.get("updateTime")))
            for row in (entry_order, emergency_order)
            if _float(row.get("updateTime")) > 0.0
        ]
        start_time = (
            max(0, min(update_times) - 300_000)
            if update_times
            else None
        )
        end_time = (
            max(update_times) + 300_000
            if update_times
            else None
        )
        exchange_trades = self.client.user_trades(
            symbol=symbol,
            limit=1000,
            start_time=start_time,
            end_time=end_time,
        )
        entry_fills = [
            row
            for row in exchange_trades
            if str(row.get("orderId", "")) == entry_order_id
        ]
        exit_fills = [
            row
            for row in exchange_trades
            if str(row.get("orderId", "")) == emergency_order_id
        ]
        if not entry_fills or not exit_fills:
            raise RuntimeError(
                "exchange trade history does not contain both reviewed orders"
            )

        def fill_totals(
            rows: list[dict[str, Any]],
            required_side: str,
        ) -> tuple[float, float]:
            if any(
                str(row.get("side", "")).upper() != required_side
                for row in rows
            ):
                raise RuntimeError("exchange trade side differs from order")
            quantity = sum(_float(row.get("qty")) for row in rows)
            quote = sum(_float(row.get("quoteQty")) for row in rows)
            if quantity <= 0.0 or quote <= 0.0:
                raise RuntimeError(
                    "exchange trade history has invalid quantity/quote"
                )
            return quantity, quote / quantity

        entry_trade_quantity, entry_trade_price = fill_totals(
            entry_fills, entry_side
        )
        exit_trade_quantity, exit_trade_price = fill_totals(
            exit_fills, exit_side
        )
        if (
            abs(entry_trade_quantity - entry_quantity) > tolerance
            or abs(exit_trade_quantity - exit_quantity) > tolerance
        ):
            raise RuntimeError(
                "exchange trade quantities do not reconcile to orders"
            )
        commission_by_asset: dict[str, float] = {}
        reviewed_fills = entry_fills + exit_fills
        for row in reviewed_fills:
            asset = str(row.get("commissionAsset", "UNKNOWN")).upper()
            commission_by_asset[asset] = (
                commission_by_asset.get(asset, 0.0)
                + _float(row.get("commission"))
            )
        realized_pnl = sum(
            _float(row.get("realizedPnl")) for row in reviewed_fills
        )
        usdt_commission = commission_by_asset.get("USDT", 0.0)
        net_pnl_usdt = (
            realized_pnl - usdt_commission
            if set(commission_by_asset).issubset({"USDT"})
            else None
        )

        second_account = self.snapshot_account()
        second_standard = self.client.open_orders()
        second_algo = self.client.open_algo_orders()
        if second_account.positions or second_standard or second_algo:
            raise RuntimeError(
                "account changed during flat emergency recovery review"
            )

        def exchange_datetime(order: dict[str, Any]) -> datetime:
            milliseconds = _float(order.get("updateTime"))
            if milliseconds <= 0.0:
                return _utc_now()
            return datetime.fromtimestamp(
                milliseconds / 1000.0,
                tz=timezone.utc,
            ).replace(tzinfo=None)

        entry_time = exchange_datetime(entry_order)
        exit_time = exchange_datetime(emergency_order)
        recovery_id = hashlib.sha256(
            (
                f"{self.config_hash}|{expected_entry_client_id}|"
                f"{emergency_client_id}|{entry_order_id}|"
                f"{emergency_order_id}"
            ).encode("utf-8")
        ).hexdigest()[:24]
        record = {
            "recovery_version": FLAT_EMERGENCY_RECOVERY_VERSION,
            "recovery_id": recovery_id,
            "recovered_at": _utc_now().isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "event_id": event_id,
            "direction": direction.name,
            "entry_client_id": expected_entry_client_id,
            "entry_order_id": entry_order_id,
            "entry_time": entry_time.isoformat(),
            "entry_quantity": entry_quantity,
            "entry_price": entry_trade_price or entry_price,
            "emergency_client_id": emergency_client_id,
            "emergency_order_id": emergency_order_id,
            "exit_time": exit_time.isoformat(),
            "exit_quantity": exit_quantity,
            "exit_price": exit_trade_price or exit_price,
            "realized_pnl": realized_pnl,
            "commission_by_asset": commission_by_asset,
            "net_pnl_usdt": net_pnl_usdt,
            "position_count": 0,
            "standard_order_count": 0,
            "algo_order_count": 0,
        }
        self.state.setdefault("recovery_journal", []).append(record)
        if len(self.state["recovery_journal"]) > 200:
            del self.state["recovery_journal"][:-200]
        self.state["trades"].append(
            {
                "time": exit_time.isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "event_id": event_id,
                "exit_reason": (
                    "emergency_flatten_after_fill_confirmation_fault"
                ),
                "quantity": entry_quantity,
                "entry_price": record["entry_price"],
                "exit_price": record["exit_price"],
                "realized_pnl": realized_pnl,
                "commission_by_asset": commission_by_asset,
                "net_pnl": net_pnl_usdt,
                "entry_client_id": expected_entry_client_id,
                "exit_client_id": emergency_client_id,
                "pnl_source": "verified_exchange_trade_history",
            }
        )
        self.state["seen_events"][strategy][event_id] = {
            "time": exit_time.isoformat(),
            "status": "emergency_flattened_after_execution_fault",
        }
        self._record_entry(strategy, symbol, entry_time)
        cooldown_minutes = (
            self.breakout_portfolio_config.symbol_cooldown_minutes
            if strategy == BREAKOUT_KEY
            else self.grid_portfolio_config.symbol_cooldown_minutes
        )
        self._set_cooldown(
            strategy,
            symbol,
            minute_token(exit_time),
            cooldown_minutes,
        )
        self.state["pending_orders"].pop(
            expected_entry_client_id, None
        )
        self.state["operational_halt"] = False
        self.state["halt_reason"] = ""
        self.state["reconcile_failures"] = 0
        self.state["consecutive_api_failures"] = 0
        self.state["rate_limit_cooldown_until"] = None
        self.state["last_reconciled_at"] = _utc_now().isoformat()
        self._last_account = second_account
        self._last_full_reconcile_monotonic = time.monotonic()
        self._force_reconcile = True
        self._append_event(
            "flat_emergency_round_trip_recovered",
            **record,
        )
        self._persist_state()
        self._write_status(second_account)
        return {
            "cleared": True,
            "already_recovered": False,
            "record": record,
        }

    def recover_verified_flat_protective_stop_incident(
        self,
        *,
        expected_symbol: str,
        expected_event_id: str,
        expected_entry_client_ids: list[str],
        expected_primary_stop_client_id: str,
        expected_residual_stop_client_id: str,
        expected_take_profit_client_ids: list[str],
    ) -> dict[str, Any]:
        """Clear one fully reviewed stop/residual incident while flat.

        This is deliberately incident-specific.  It never places or cancels
        an order.  It proves the complete entry/stop trade history, income
        ledger, two globally flat account snapshots, and the exact historical
        halt reasons before replacing the generic local trade row with an
        exchange-sourced audit record and clearing only this execution halt.
        """

        symbol = expected_symbol.strip().upper()
        event_id = expected_event_id.strip()
        entry_client_ids = [
            value.strip() for value in expected_entry_client_ids
            if value.strip()
        ]
        take_profit_client_ids = [
            value.strip() for value in expected_take_profit_client_ids
            if value.strip()
        ]
        primary_stop_id = expected_primary_stop_client_id.strip()
        residual_stop_id = expected_residual_stop_client_id.strip()
        if (
            not symbol
            or not event_id
            or not entry_client_ids
            or len(set(entry_client_ids)) != len(entry_client_ids)
            or not primary_stop_id
            or not residual_stop_id
            or primary_stop_id == residual_stop_id
        ):
            raise RuntimeError(
                "protective-stop recovery identity is incomplete"
            )

        existing_recovery = next(
            (
                row
                for row in self.state.get("recovery_journal", [])
                if row.get("recovery_version")
                == PROTECTIVE_STOP_INCIDENT_RECOVERY_VERSION
                and row.get("symbol") == symbol
                and row.get("event_id") == event_id
            ),
            None,
        )
        if existing_recovery is not None:
            if (
                not self.state.get("operational_halt")
                and not self.state.get("circuit_breaker")
                and not self._local_positions()
                and not self.state.get("pending_orders")
                and not self.state.get("protective_orders")
            ):
                return {
                    "cleared": True,
                    "already_recovered": True,
                    "record": existing_recovery,
                }
            raise RuntimeError(
                "protective-stop recovery record exists but ledger is "
                "not clean"
            )
        if not self.state.get("operational_halt"):
            raise RuntimeError(
                "protective-stop recovery requires an operational halt"
            )
        if not self.state.get("circuit_breaker"):
            raise RuntimeError(
                "protective-stop recovery requires the reviewed API circuit"
            )
        if (
            self._local_positions()
            or self.state.get("pending_orders")
            or self.state.get("protective_orders")
        ):
            raise RuntimeError(
                "protective-stop recovery requires an empty local ledger"
            )

        halt_rows = [
            row.strip()
            for row in str(self.state.get("halt_reason", "")).split("|")
            if row.strip()
        ]
        allowed_halt = (
            "quantity differs:",
            "reconciliation failed",
            "order would immediately trigger",
            "unknown order sent",
        )
        if (
            not halt_rows
            or not any(
                row.startswith(f"{symbol} quantity differs:")
                for row in halt_rows
            )
            or any(
                not any(token in row.lower() for token in allowed_halt)
                for row in halt_rows
            )
        ):
            raise RuntimeError(
                "operational halt contains an unrelated recovery reason"
            )
        circuit_reason = str(
            self.state.get("circuit_reason", "")
        ).lower()
        if (
            "api failures=" not in circuit_reason
            or not any(
                token in circuit_reason
                for token in (
                    "order would immediately trigger",
                    "unknown order sent",
                )
            )
        ):
            raise RuntimeError(
                "circuit breaker is unrelated to the reviewed incident"
            )

        generic_trades = [
            row
            for row in self.state.get("trades", [])
            if row.get("strategy") == GRID_KEY
            and str(row.get("symbol", "")).upper() == symbol
            and row.get("exit_reason")
            == "exchange_closed_or_protection_triggered"
        ]
        if len(generic_trades) != 1:
            raise RuntimeError(
                "expected exactly one generic stopped Grid trade"
            )
        journal_rows = [
            row
            for row in self.state.get("order_journal", [])
            if str(row.get("symbol", "")).upper() == symbol
        ]
        journal_client_ids = {
            str(row.get("client_id", "")) for row in journal_rows
        }
        reviewed_client_ids = set(
            entry_client_ids
            + take_profit_client_ids
            + [primary_stop_id, residual_stop_id]
        )
        if not reviewed_client_ids.issubset(journal_client_ids):
            raise RuntimeError(
                "local order journal is missing reviewed incident IDs"
            )
        deterministic_entries = [
            self._client_id(GRID_KEY, event_id, "entry", 0)
        ] + [
            self._client_id(GRID_KEY, event_id, f"ge{index}", 0)
            for index in range(1, len(entry_client_ids))
        ]
        if entry_client_ids != deterministic_entries:
            raise RuntimeError(
                "entry IDs differ from deterministic Grid event IDs"
            )
        deterministic_take_profits = [
            self._client_id(
                GRID_KEY, event_id, f"tp{index}", 0
            )
            for index in range(len(entry_client_ids))
        ]
        if take_profit_client_ids != deterministic_take_profits:
            raise RuntimeError(
                "take-profit IDs differ from deterministic Grid event IDs"
            )
        if (
            primary_stop_id
            != self._client_id(
                GRID_KEY,
                event_id,
                "stop",
                len(entry_client_ids),
            )
            or residual_stop_id
            != self._client_id(
                GRID_KEY,
                event_id,
                "stop",
                len(entry_client_ids) + 1,
            )
        ):
            raise RuntimeError(
                "stop IDs differ from deterministic Grid generations"
            )

        first_account = self.snapshot_account()
        first_standard = self.client.open_orders()
        first_algo = self.client.open_algo_orders()
        if first_account.positions or first_standard or first_algo:
            raise RuntimeError(
                "protective-stop recovery requires a globally flat account "
                "with no open orders"
            )

        def exchange_bool(value: Any) -> bool:
            return value is True or str(value).strip().lower() == "true"

        entry_orders = [
            self.client.query_order(
                symbol, orig_client_order_id=client_id
            )
            for client_id in entry_client_ids
        ]
        entry_order_ids: list[str] = []
        entry_quantities: list[float] = []
        for client_id, order in zip(
            entry_client_ids, entry_orders
        ):
            quantity, _price = self._filled_order(order)
            if (
                str(order.get("symbol", "")).upper() != symbol
                or _order_client_id(order) != client_id
                or str(order.get("side", "")).upper() != "SELL"
                or exchange_bool(order.get("reduceOnly"))
            ):
                raise RuntimeError(
                    "reviewed Grid entry order identity differs"
                )
            order_id = str(order.get("orderId", ""))
            if not order_id:
                raise RuntimeError(
                    "reviewed Grid entry order ID is missing"
                )
            entry_order_ids.append(order_id)
            entry_quantities.append(quantity)
        if len(set(entry_order_ids)) != len(entry_order_ids):
            raise RuntimeError("reviewed Grid entry order IDs repeat")

        take_profit_orders = [
            self.client.query_order(
                symbol, orig_client_order_id=client_id
            )
            for client_id in take_profit_client_ids
        ]
        for client_id, order in zip(
            take_profit_client_ids, take_profit_orders
        ):
            if (
                str(order.get("symbol", "")).upper() != symbol
                or _order_client_id(order) != client_id
                or str(order.get("side", "")).upper() != "BUY"
                or not exchange_bool(order.get("reduceOnly"))
                or str(order.get("status", "")).upper()
                not in _TERMINAL_ORDER_STATUSES
                or _float(order.get("executedQty")) > 0.0
            ):
                raise RuntimeError(
                    "reviewed Grid take-profit order is not terminal/unfilled"
                )

        stop_pairs: list[
            tuple[str, dict[str, Any], dict[str, Any]]
        ] = []
        for stop_client_id in (
            primary_stop_id,
            residual_stop_id,
        ):
            stop_order = self.client.query_algo_order(
                client_algo_id=stop_client_id
            )
            stop_status = str(
                stop_order.get(
                    "algoStatus", stop_order.get("status", "")
                )
            ).upper()
            actual_order_id = stop_order.get("actualOrderId")
            if (
                str(stop_order.get("symbol", "")).upper() != symbol
                or _algo_client_id(stop_order) != stop_client_id
                or str(stop_order.get("side", "")).upper() != "BUY"
                or not exchange_bool(stop_order.get("reduceOnly"))
                or stop_status != "FINISHED"
                or actual_order_id in (None, "")
            ):
                raise RuntimeError(
                    "reviewed protective stop is not a finished BUY exit"
                )
            actual_order = self.client.query_order(
                symbol, order_id=actual_order_id
            )
            actual_quantity, _actual_price = self._filled_order(
                actual_order
            )
            if (
                str(actual_order.get("symbol", "")).upper() != symbol
                or str(actual_order.get("orderId", ""))
                != str(actual_order_id)
                or str(actual_order.get("side", "")).upper() != "BUY"
                or not exchange_bool(actual_order.get("reduceOnly"))
                or not math.isclose(
                    _float(stop_order.get("quantity")),
                    actual_quantity,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    "protective stop child fill does not match algo order"
                )
            stop_pairs.append(
                (stop_client_id, stop_order, actual_order)
            )
        exit_order_ids = [
            str(actual.get("orderId", ""))
            for _client_id_value, _stop, actual in stop_pairs
        ]
        if (
            not all(exit_order_ids)
            or len(set(exit_order_ids)) != len(exit_order_ids)
            or set(exit_order_ids).intersection(entry_order_ids)
        ):
            raise RuntimeError(
                "reviewed protective stop child order IDs are invalid"
            )

        rules = self.client.symbol_rules(symbol)
        tolerance = max(
            _float(rules.quantity_step) / 2.0, 1e-12
        )
        entry_quantity = _exact_quantity_sum(entry_quantities)
        exit_quantities = [
            self._filled_order(actual)[0]
            for _client_id_value, _stop, actual in stop_pairs
        ]
        exit_quantity = _exact_quantity_sum(exit_quantities)
        if abs(entry_quantity - exit_quantity) > tolerance:
            raise RuntimeError(
                "reviewed entries and protective exits do not flatten"
            )

        update_times = [
            int(_float(order.get("updateTime")))
            for order in entry_orders
            + [actual for _client_id_value, _stop, actual in stop_pairs]
            if _float(order.get("updateTime")) > 0.0
        ]
        start_time = (
            max(0, min(update_times) - 300_000)
            if update_times
            else None
        )
        end_time = (
            max(update_times) + 300_000
            if update_times
            else None
        )
        exchange_trades = self.client.user_trades(
            symbol=symbol,
            limit=1000,
            start_time=start_time,
            end_time=end_time,
        )
        expected_order_ids = set(entry_order_ids + exit_order_ids)
        actual_trade_order_ids = {
            str(row.get("orderId", ""))
            for row in exchange_trades
            if _float(row.get("qty")) > 0.0
        }
        if actual_trade_order_ids != expected_order_ids:
            raise RuntimeError(
                "trade history contains missing or unrelated order IDs"
            )

        def trade_totals(
            order_ids: list[str],
            required_side: str,
        ) -> tuple[float, float, float, float]:
            rows = [
                row
                for row in exchange_trades
                if str(row.get("orderId", "")) in order_ids
            ]
            if (
                {str(row.get("orderId", "")) for row in rows}
                != set(order_ids)
                or any(
                    str(row.get("side", "")).upper()
                    != required_side
                    for row in rows
                )
                or any(
                    str(row.get("commissionAsset", "")).upper()
                    != "USDT"
                    for row in rows
                )
            ):
                raise RuntimeError(
                    "trade fills differ from reviewed order identities"
                )
            quantity = _exact_quantity_sum(
                _float(row.get("qty")) for row in rows
            )
            quote = _exact_quantity_sum(
                _float(row.get("quoteQty")) for row in rows
            )
            commission = _exact_quantity_sum(
                _float(row.get("commission")) for row in rows
            )
            realized = _exact_quantity_sum(
                _float(row.get("realizedPnl")) for row in rows
            )
            if quantity <= 0.0 or quote <= 0.0:
                raise RuntimeError(
                    "reviewed trade fills have invalid quantity/quote"
                )
            return quantity, quote, commission, realized

        (
            entry_trade_quantity,
            entry_quote,
            entry_commission,
            entry_realized,
        ) = trade_totals(entry_order_ids, "SELL")
        (
            exit_trade_quantity,
            exit_quote,
            exit_commission,
            realized_pnl,
        ) = trade_totals(exit_order_ids, "BUY")
        if (
            abs(entry_trade_quantity - entry_quantity) > tolerance
            or abs(exit_trade_quantity - exit_quantity) > tolerance
            or abs(entry_realized) > 1e-10
        ):
            raise RuntimeError(
                "reviewed trade quantities/PnL do not match orders"
            )
        commission = entry_commission + exit_commission

        income_rows = self.client.income_history(
            symbol=symbol,
            limit=1000,
            start_time=start_time,
            end_time=end_time,
        )
        allowed_income_types = {
            "COMMISSION",
            "FUNDING_FEE",
            "REALIZED_PNL",
        }
        nonzero_unrelated_income = [
            row
            for row in income_rows
            if str(row.get("incomeType", "")).upper()
            not in allowed_income_types
            and abs(_float(row.get("income"))) > 1e-12
        ]
        if nonzero_unrelated_income:
            raise RuntimeError(
                "income history contains unrelated nonzero rows"
            )
        commission_income = _exact_quantity_sum(
            _float(row.get("income"))
            for row in income_rows
            if str(row.get("incomeType", "")).upper()
            == "COMMISSION"
        )
        realized_income = _exact_quantity_sum(
            _float(row.get("income"))
            for row in income_rows
            if str(row.get("incomeType", "")).upper()
            == "REALIZED_PNL"
        )
        funding_pnl = _exact_quantity_sum(
            _float(row.get("income"))
            for row in income_rows
            if str(row.get("incomeType", "")).upper()
            == "FUNDING_FEE"
        )
        if (
            not math.isclose(
                commission_income,
                -commission,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                realized_income,
                realized_pnl,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise RuntimeError(
                "income ledger does not reconcile to trade fills"
            )

        second_account = self.snapshot_account()
        second_standard = self.client.open_orders()
        second_algo = self.client.open_algo_orders()
        if second_account.positions or second_standard or second_algo:
            raise RuntimeError(
                "account changed during protective-stop recovery review"
            )

        def exchange_datetime(order: dict[str, Any]) -> datetime:
            milliseconds = _float(order.get("updateTime"))
            if milliseconds <= 0.0:
                return _utc_now()
            return datetime.fromtimestamp(
                milliseconds / 1000.0,
                tz=timezone.utc,
            ).replace(tzinfo=None)

        entry_time = min(
            exchange_datetime(order) for order in entry_orders
        )
        exit_time = max(
            exchange_datetime(actual)
            for _client_id_value, _stop, actual in stop_pairs
        )
        entry_price = entry_quote / entry_trade_quantity
        exit_price = exit_quote / exit_trade_quantity
        net_pnl_usdt = (
            realized_pnl - commission + funding_pnl
        )
        recovery_id = hashlib.sha256(
            (
                f"{self.config_hash}|{symbol}|{event_id}|"
                + "|".join(
                    entry_order_ids
                    + exit_order_ids
                    + [primary_stop_id, residual_stop_id]
                )
            ).encode("utf-8")
        ).hexdigest()[:24]
        record = {
            "recovery_version": (
                PROTECTIVE_STOP_INCIDENT_RECOVERY_VERSION
            ),
            "recovery_id": recovery_id,
            "recovered_at": _utc_now().isoformat(),
            "strategy": GRID_KEY,
            "symbol": symbol,
            "event_id": event_id,
            "direction": Direction.SHORT.name,
            "entry_client_ids": entry_client_ids,
            "entry_order_ids": entry_order_ids,
            "entry_time": entry_time.isoformat(),
            "entry_quantity": entry_quantity,
            "entry_price": entry_price,
            "take_profit_client_ids": take_profit_client_ids,
            "primary_stop_client_id": primary_stop_id,
            "residual_stop_client_id": residual_stop_id,
            "exit_order_ids": exit_order_ids,
            "exit_time": exit_time.isoformat(),
            "exit_quantity": exit_quantity,
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "commission_usdt": commission,
            "funding_pnl_usdt": funding_pnl,
            "net_pnl_usdt": net_pnl_usdt,
            "position_count": 0,
            "standard_order_count": 0,
            "algo_order_count": 0,
        }
        generic_trades[0].update(
            {
                "time": exit_time.isoformat(),
                "event_id": event_id,
                "direction": Direction.SHORT.name,
                "exit_reason": (
                    "verified_protective_stop_with_residual_close"
                ),
                "quantity": entry_quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "commission_usdt": commission,
                "funding_pnl_usdt": funding_pnl,
                "net_pnl": net_pnl_usdt,
                "entry_client_ids": entry_client_ids,
                "exit_client_ids": [
                    primary_stop_id,
                    residual_stop_id,
                ],
                "pnl_source": (
                    "verified_exchange_trades_and_income_history"
                ),
            }
        )
        self.state.setdefault("recovery_journal", []).append(record)
        if len(self.state["recovery_journal"]) > 200:
            del self.state["recovery_journal"][:-200]
        self.state["operational_halt"] = False
        self.state["halt_reason"] = ""
        self.state["circuit_breaker"] = False
        self.state["circuit_reason"] = ""
        self.state["reconcile_failures"] = 0
        self.state["consecutive_api_failures"] = 0
        self.state["rate_limit_cooldown_until"] = None
        self.state["last_reconciled_at"] = _utc_now().isoformat()
        self._last_account = second_account
        self._last_full_reconcile_monotonic = time.monotonic()
        self._force_reconcile = True
        self._append_event(
            "flat_protective_stop_incident_recovered",
            **record,
        )
        self._persist_state()
        self._write_status(second_account)
        return {
            "cleared": True,
            "already_recovered": False,
            "record": record,
        }

    def _preflight_test_order(self) -> None:
        """Validate signed trade permission without reaching the matcher."""

        symbol = "BTCUSDT"
        rows = self.client.klines(symbol, "1m", 2)
        if not rows or rows[-1].close <= 0.0:
            raise RuntimeError("cannot size the non-matching preflight order")
        rules = self.client.symbol_rules(symbol)
        minimum = max(
            float(rules.min_quantity),
            float(rules.min_notional) / rows[-1].close,
        )
        quantity = rules.round_quantity(
            minimum + float(rules.quantity_step)
        )
        client_id = (
            f"{self.live.order_id_prefix}-preflight-"
            f"{self.config_hash[:12]}"
        )[:36]
        self.client.test_order(
            symbol,
            "BUY",
            quantity,
            new_client_order_id=client_id,
        )
        self._append_event(
            "non_matching_order_preflight_passed",
            symbol=symbol,
            quantity=quantity,
        )

    def _prepare_symbol_for_entry(self, symbol: str) -> None:
        if symbol in self._prepared_symbols:
            return
        self.client.set_margin_type(
            symbol, self.config.trading.margin_type
        )
        self.client.set_leverage(
            symbol, self.config.trading.leverage
        )
        self._prepared_symbols.add(symbol)
        self._append_event(
            "symbol_execution_prepared",
            symbol=symbol,
            margin_type=self.config.trading.margin_type,
            leverage=self.config.trading.leverage,
        )

    def _entry_account_allows(
        self,
        strategy: str,
        symbol: str,
        account: AccountSnapshot,
    ) -> bool:
        if self.state.get("circuit_breaker") or self.state.get(
            "operational_halt"
        ):
            return False
        if (
            account.available_balance
            < self.config.risk.min_available_balance_usdt
        ):
            self._trip_circuit(
                "available_balance="
                f"{account.available_balance:.4f}<"
                f"{self.config.risk.min_available_balance_usdt:.4f}"
            )
            return False
        if (
            account.initial_margin_usage_pct
            >= self.config.risk.max_account_margin_usage_pct
        ):
            self._trip_circuit(
                "initial_margin_usage="
                f"{account.initial_margin_usage_pct:.4%}>="
                f"{self.config.risk.max_account_margin_usage_pct:.4%}"
            )
            return False
        local = self._local_positions()
        if any(
            row.get("strategy") == strategy
            for row in self.state["pending_orders"].values()
            if "model" in row
        ):
            return False
        if any(
            managed_strategy == strategy
            for managed_strategy, _direction, _quantity in local.values()
        ):
            return False
        if symbol in account.positions or symbol in local:
            return False
        if (
            len(account.positions) >= self.live.max_open_positions
            or len(local) >= self.live.max_open_positions
        ):
            return False
        unmanaged = set(account.positions) - set(local)
        if unmanaged:
            self._halt(
                "entry guard found unmanaged exchange positions: "
                + ",".join(sorted(unmanaged)[:3])
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Deterministic, idempotent exchange operations
    # ------------------------------------------------------------------
    def _client_id(
        self,
        strategy: str,
        event_id: str,
        role: str,
        generation: int = 0,
    ) -> str:
        sleeve = "bo" if strategy == BREAKOUT_KEY else "gr"
        digest = hashlib.sha256(
            f"{strategy}|{event_id}|{role}|{generation}".encode("utf-8")
        ).hexdigest()[:16]
        value = (
            f"{self.live.order_id_prefix}-{sleeve}-"
            f"{role[:5]}-{generation}-{digest}"
        )
        return value[:36]

    def _journal(self, action: str, **payload: Any) -> None:
        rows = self.state["order_journal"]
        rows.append(
            {
                "time": _utc_now().isoformat(),
                "action": action,
                **_jsonable(payload),
            }
        )
        if len(rows) > 2000:
            del rows[:-2000]

    def _query_standard_safely(
        self, symbol: str, client_id: str
    ) -> dict[str, Any] | None:
        try:
            return self.client.query_order(
                symbol, orig_client_order_id=client_id
            )
        except BinanceRateLimitError:
            raise
        except Exception:
            return None

    def _query_standard_by_order_id_safely(
        self,
        symbol: str,
        order_id: int | str,
    ) -> dict[str, Any] | None:
        try:
            return self.client.query_order(
                symbol, order_id=order_id
            )
        except BinanceRateLimitError:
            raise
        except Exception:
            return None

    def _query_algo_safely(
        self, client_id: str
    ) -> dict[str, Any] | None:
        try:
            return self.client.query_algo_order(
                client_algo_id=client_id
            )
        except BinanceRateLimitError:
            raise
        except Exception:
            return None

    def _market_order_idempotent(
        self,
        *,
        symbol: str,
        side: str,
        quantity: str,
        reduce_only: bool,
        client_id: str,
    ) -> dict[str, Any]:
        existing = self._query_standard_safely(symbol, client_id)
        if existing is not None:
            return existing
        try:
            response = self.client.new_market_order(
                symbol,
                side,
                quantity,
                reduce_only=reduce_only,
                new_client_order_id=client_id,
            )
        except Exception:
            existing = self._query_standard_safely(symbol, client_id)
            if existing is None:
                raise
            response = existing
        self._journal(
            "market_order",
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            client_id=client_id,
            status=response.get("status"),
        )
        self._force_reconcile = True
        return response

    def _limit_order_idempotent(
        self,
        *,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        reduce_only: bool,
        client_id: str,
    ) -> dict[str, Any]:
        existing = self._query_standard_safely(symbol, client_id)
        if existing is not None:
            status = str(existing.get("status", "")).upper()
            if status in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
                return existing
            raise RuntimeError(
                f"existing limit order {client_id} is {status or 'UNKNOWN'}"
            )
        try:
            response = self.client.new_limit_order(
                symbol,
                side,
                quantity,
                price,
                reduce_only=reduce_only,
                new_client_order_id=client_id,
            )
        except Exception:
            existing = self._query_standard_safely(symbol, client_id)
            if existing is None:
                raise
            response = existing
        status = str(response.get("status", "")).upper()
        if status not in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
            raise RuntimeError(
                f"limit order {client_id} is {status or 'UNKNOWN'}"
            )
        self._journal(
            "limit_order",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            reduce_only=reduce_only,
            client_id=client_id,
            status=response.get("status"),
        )
        self._force_reconcile = True
        return response

    def _algo_order_idempotent(
        self,
        *,
        kind: str,
        symbol: str,
        side: str,
        quantity: str,
        trigger_price: str,
        client_id: str,
    ) -> dict[str, Any]:
        existing = self._query_algo_safely(client_id)
        if existing is not None:
            status = str(
                existing.get("algoStatus", existing.get("status", ""))
            ).upper()
            if status in {"NEW", "ACCEPTED"}:
                return existing
            raise RuntimeError(
                f"existing algo order {client_id} is "
                f"{status or 'UNKNOWN'}"
            )
        method = (
            self.client.new_stop_market_order
            if kind == "stop"
            else self.client.new_take_profit_market_order
        )
        try:
            response = method(
                symbol,
                side,
                trigger_price,
                quantity,
                reduce_only=True,
                working_type=self.config.trading.working_type,
                new_client_algo_id=client_id,
            )
        except Exception:
            existing = self._query_algo_safely(client_id)
            if existing is None:
                raise
            response = existing
        status = str(
            response.get("algoStatus", response.get("status", ""))
        ).upper()
        if status not in {"NEW", "ACCEPTED"}:
            raise RuntimeError(
                f"algo order {client_id} is {status or 'UNKNOWN'}"
            )
        self._journal(
            f"{kind}_algo_order",
            symbol=symbol,
            side=side,
            quantity=quantity,
            trigger_price=trigger_price,
            reduce_only=True,
            client_id=client_id,
            status=response.get("algoStatus", response.get("status")),
        )
        self._force_reconcile = True
        return response

    @staticmethod
    def _filled_order(
        response: dict[str, Any],
    ) -> tuple[float, float]:
        status = str(response.get("status", "")).upper()
        quantity = _float(response.get("executedQty"))
        price = _float(response.get("avgPrice"))
        if price <= 0.0 and quantity > 0.0:
            cumulative_quote = _float(response.get("cumQuote"))
            if cumulative_quote > 0.0:
                price = cumulative_quote / quantity
        if status != "FILLED":
            raise RuntimeError(
                f"entry market order is not fully filled: {status or 'EMPTY'}"
            )
        if quantity <= 0.0 or price <= 0.0:
            raise RuntimeError("market order returned no executed quantity/price")
        return quantity, price

    def _confirm_market_fill(
        self,
        *,
        symbol: str,
        client_id: str,
        response: dict[str, Any],
        expected_quantity: float,
    ) -> tuple[dict[str, Any], float, float]:
        """Require exchange-final quantity and price for a market order.

        Binance can acknowledge a FILLED futures market order with avgPrice
        still zero even though a subsequent order query contains the final
        fill.  A modeled or ticker fallback is never accepted as execution
        truth.  Only a final order response (or a bounded idempotent query by
        client ID) can release the pending entry intent.
        """

        current = dict(response)
        last_error: Exception | None = None
        for attempt in range(1, MARKET_FILL_CONFIRM_ATTEMPTS + 1):
            try:
                quantity, price = self._filled_order(current)
                tolerance = max(
                    abs(float(expected_quantity)) * 1e-9,
                    1e-12,
                )
                if (
                    expected_quantity > 0.0
                    and abs(quantity - expected_quantity) > tolerance
                ):
                    raise RuntimeError(
                        "market fill quantity differs from submitted "
                        f"quantity: filled={quantity:.12g} "
                        f"submitted={expected_quantity:.12g}"
                    )
                if attempt > 1:
                    self._append_event(
                        "market_fill_confirmed_after_query",
                        symbol=symbol,
                        client_id=client_id,
                        attempts=attempt,
                        quantity=quantity,
                        price=price,
                    )
                return current, quantity, price
            except RuntimeError as exc:
                last_error = exc
            if attempt >= MARKET_FILL_CONFIRM_ATTEMPTS:
                break
            time.sleep(MARKET_FILL_CONFIRM_DELAY_SECONDS)
            queried = self._query_standard_safely(symbol, client_id)
            if queried is not None:
                current = queried
        raise RuntimeError(
            "market fill confirmation failed after "
            f"{MARKET_FILL_CONFIRM_ATTEMPTS} attempts: {last_error}"
        )

    def _handle_unconfirmed_entry_fill(
        self,
        *,
        strategy: str,
        event_id: str,
        symbol: str,
        direction: Direction,
        client_id: str,
        error: Exception,
    ) -> None:
        try:
            self._safe_cancel_standard(symbol, client_id)
        except Exception as cancel_error:
            self._append_event(
                "unconfirmed_entry_cancel_failed",
                strategy=strategy,
                symbol=symbol,
                client_id=client_id,
                error=(
                    f"{type(cancel_error).__name__}: {cancel_error}"
                ),
            )
        self._halt(
            f"{symbol} entry fill was not final: "
            f"{type(error).__name__}: {error}"
        )
        try:
            position = self.snapshot_account().positions.get(symbol)
        except Exception:
            self._persist_state()
            raise
        if position is None:
            self._persist_state()
            return
        self._emergency_close_after_unprotected_entry(
            strategy=strategy,
            event_id=event_id,
            symbol=symbol,
            direction=direction,
            quantity=position.quantity,
            error=error,
            failure_context="entry fill confirmation",
        )
        self._persist_state()

    def _safe_cancel_standard(
        self, symbol: str, client_id: str
    ) -> None:
        existing = self._query_standard_safely(symbol, client_id)
        if existing is not None:
            status = str(existing.get("status", "")).upper()
            if status in _TERMINAL_ORDER_STATUSES:
                self._journal(
                    "terminal_order_cancel_skipped",
                    order_surface="standard",
                    symbol=symbol,
                    client_id=client_id,
                    status=status,
                )
                self._force_reconcile = True
                return
        try:
            self.client.cancel_order(
                symbol, orig_client_order_id=client_id
            )
            self._force_reconcile = True
        except Exception as exc:
            order = self._query_standard_safely(symbol, client_id)
            if order is None:
                raise exc
            if str(order.get("status", "")).upper() not in (
                _TERMINAL_ORDER_STATUSES
            ):
                raise exc
            self._force_reconcile = True

    def _safe_cancel_algo(self, client_id: str) -> None:
        existing = self._query_algo_safely(client_id)
        if existing is not None:
            status = str(
                existing.get("algoStatus", existing.get("status", ""))
            ).upper()
            if status in _TERMINAL_ORDER_STATUSES:
                self._journal(
                    "terminal_order_cancel_skipped",
                    order_surface="algo",
                    client_id=client_id,
                    status=status,
                )
                self._force_reconcile = True
                return
        try:
            self.client.cancel_algo_order(client_algo_id=client_id)
            self._force_reconcile = True
        except Exception as exc:
            order = self._query_algo_safely(client_id)
            if order is None:
                raise exc
            if str(
                order.get("algoStatus", order.get("status", ""))
            ).upper() not in _TERMINAL_ORDER_STATUSES:
                raise exc
            self._force_reconcile = True

    # ------------------------------------------------------------------
    # Protective orders
    # ------------------------------------------------------------------
    def _ensure_protection(
        self,
        *,
        strategy: str,
        event_id: str,
        symbol: str,
        direction: Direction,
        quantity: float,
        stop_price: float,
        take_profit_price: float = 0.0,
        open_algo_ids: set[str] | None = None,
        force_replace: bool = False,
    ) -> None:
        rules = self.client.symbol_rules(symbol)
        rounded_quantity = _round_reduce_only_quantity(
            rules, quantity
        )
        if _float(rounded_quantity) <= 0.0:
            raise RuntimeError("protective order quantity rounded to zero")
        exit_side = "SELL" if direction == Direction.LONG else "BUY"
        protection = self.state["protective_orders"].setdefault(
            strategy, {}
        )
        generation = int(protection.get("generation", 0))
        old_stop_id = str(protection.get("stop_client_id", ""))
        old_stop_price = _float(protection.get("stop_price"))
        old_quantity = _float(protection.get("quantity"))
        stop_is_open = (
            bool(old_stop_id)
            and (
                open_algo_ids is None
                or old_stop_id in open_algo_ids
            )
        )
        needs_stop = (
            force_replace
            or not stop_is_open
            or not math.isclose(old_quantity, quantity, rel_tol=1e-9)
            or not math.isclose(old_stop_price, stop_price, rel_tol=1e-9)
        )
        if needs_stop:
            generation += 1
            stop_id = self._client_id(
                strategy, event_id, "stop", generation
            )
            response = self._algo_order_idempotent(
                kind="stop",
                symbol=symbol,
                side=exit_side,
                quantity=rounded_quantity,
                trigger_price=rules.round_price(stop_price),
                client_id=stop_id,
            )
            protection.update(
                {
                    "generation": generation,
                    "symbol": symbol,
                    "quantity": quantity,
                    "stop_price": stop_price,
                    "stop_client_id": stop_id,
                    "stop_algo_id": response.get("algoId"),
                }
            )
            self._persist_state()
            if old_stop_id and old_stop_id != stop_id:
                self._safe_cancel_algo(old_stop_id)

        old_tp_id = str(protection.get("tp_client_id", ""))
        tp_is_open = (
            bool(old_tp_id)
            and (
                open_algo_ids is None
                or old_tp_id in open_algo_ids
            )
        )
        if take_profit_price > 0.0 and (
            force_replace or not tp_is_open or needs_stop
        ):
            generation = int(protection.get("generation", generation)) + 1
            tp_id = self._client_id(
                strategy, event_id, "takep", generation
            )
            response = self._algo_order_idempotent(
                kind="take_profit",
                symbol=symbol,
                side=exit_side,
                quantity=rounded_quantity,
                trigger_price=rules.round_price(take_profit_price),
                client_id=tp_id,
            )
            protection.update(
                {
                    "generation": generation,
                    "tp_price": take_profit_price,
                    "tp_client_id": tp_id,
                    "tp_algo_id": response.get("algoId"),
                }
            )
            self._persist_state()
            if old_tp_id and old_tp_id != tp_id:
                self._safe_cancel_algo(old_tp_id)

    def _emergency_close_after_unprotected_entry(
        self,
        *,
        strategy: str,
        event_id: str,
        symbol: str,
        direction: Direction,
        quantity: float,
        error: Exception,
        failure_context: str = "protective stop placement",
    ) -> None:
        self._halt(
            f"{symbol} {failure_context} failed: "
            f"{type(error).__name__}: {error}"
        )
        rules = self.client.symbol_rules(symbol)
        client_id = self._client_id(
            strategy, event_id, "emerg", 0
        )
        submitted_quantity = _round_reduce_only_quantity(
            rules, quantity
        )
        response = self._market_order_idempotent(
            symbol=symbol,
            side="SELL" if direction == Direction.LONG else "BUY",
            quantity=submitted_quantity,
            reduce_only=True,
            client_id=client_id,
        )
        try:
            response, filled_quantity, fill_price = (
                self._confirm_market_fill(
                    symbol=symbol,
                    client_id=client_id,
                    response=response,
                    expected_quantity=float(submitted_quantity),
                )
            )
        except Exception as close_error:
            self._halt(
                f"{symbol} emergency reduce-only fill was not final: "
                f"{type(close_error).__name__}: {close_error}"
            )
            self._persist_state()
            raise
        verified = self.snapshot_account()
        residual = verified.positions.get(symbol)
        tolerance = max(_float(rules.quantity_step) / 2.0, 1e-12)
        if residual is not None and residual.quantity > tolerance:
            self._halt(
                f"{symbol} emergency reduce-only exit left residual "
                f"{residual.quantity:.12g}"
            )
            self._persist_state()
            raise RuntimeError(
                f"{symbol} emergency close did not flatten exchange position"
            )
        self._append_event(
            "emergency_flatten_confirmed",
            strategy=strategy,
            event_id=event_id,
            symbol=symbol,
            client_id=client_id,
            order_id=response.get("orderId"),
            quantity=filled_quantity,
            price=fill_price,
        )
        self._persist_state()

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------
    def _open_breakout_candidate(
        self, candidate: Candidate, now: datetime
    ) -> bool:
        signal = candidate.signal
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
            account = self.snapshot_account()
        except BinanceRateLimitError:
            raise
        except Exception as exc:
            self.log(f"{signal.symbol}: Breakout LIVE成交参考不可用 ({exc})")
            return False
        if not self._entry_account_allows(
            BREAKOUT_KEY, signal.symbol, account
        ):
            return False
        profile = self._select_breakout_profile(candidate, account.equity)
        if profile is None:
            return False
        available = self._available_notional_multiple(account.equity)
        profile = replace(
            profile,
            portfolio=replace(
                profile.portfolio,
                risk_per_trade_pct=(
                    profile.portfolio.risk_per_trade_pct
                    * self.live.risk_scale
                ),
                max_trade_risk_pct=(
                    profile.portfolio.max_trade_risk_pct
                    * self.live.risk_scale
                ),
                max_notional_multiple=min(
                    profile.portfolio.max_notional_multiple
                    * self.live.risk_scale,
                    available * NOTIONAL_HEADROOM_SAFETY,
                ),
            ),
        )
        raw = active.close
        live_candidate = replace(
            candidate, entry_minute=minute_token(active.timestamp)
        )
        synthetic = Candle(
            active.timestamp, raw, raw, raw, raw, active.volume
        )
        opened = _entry_position(
            live_candidate,
            _compact_series([synthetic]),
            rules,
            profile.signal,
            profile.portfolio,
            self.execution,
            account.equity,
        )
        if opened is None:
            return False
        position, _entry_fee = opened
        protected = _protected_position(
            position, profile.exit_protection, self.execution
        )
        self._prepare_symbol_for_entry(signal.symbol)
        side = "BUY" if signal.direction == Direction.LONG else "SELL"
        client_id = self._client_id(
            BREAKOUT_KEY, signal.event_id, "entry", 0
        )
        pending = {
            "strategy": BREAKOUT_KEY,
            "event_id": signal.event_id,
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "client_id": client_id,
            "created_at": now.isoformat(),
            "model": _v7_position_to_dict(protected, profile),
        }
        self.state["pending_orders"][client_id] = pending
        self._persist_state()
        submitted_quantity = rules.round_quantity(position.quantity)
        response = self._market_order_idempotent(
            symbol=signal.symbol,
            side=side,
            quantity=submitted_quantity,
            reduce_only=False,
            client_id=client_id,
        )
        try:
            response, quantity, fill_price = self._confirm_market_fill(
                symbol=signal.symbol,
                client_id=client_id,
                response=response,
                expected_quantity=float(submitted_quantity),
            )
        except Exception as exc:
            self._handle_unconfirmed_entry_fill(
                strategy=BREAKOUT_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                client_id=client_id,
                error=exc,
            )
            raise
        delta = fill_price - position.entry_price
        position.quantity = quantity
        position.entry_price = fill_price
        position.take_profit_price += delta
        position.risk_budget = position.unit_risk * quantity
        protected.original_quantity = quantity
        protected.original_risk_budget = position.risk_budget
        self.state["breakout_position"] = _v7_position_to_dict(
            protected, profile
        )
        self.state["breakout_exchange"] = {
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "event_id": signal.event_id,
            "entry_client_id": client_id,
            "entry_order_id": response.get("orderId"),
            "quantity": quantity,
            "entry_price": fill_price,
            "opened_at": now.isoformat(),
        }
        self.state["breakout_last_processed_minute"] = (
            position.entry_minute - 1
        )
        self._record_entry(BREAKOUT_KEY, signal.symbol, now)
        self.state["pending_orders"].pop(client_id, None)
        self._persist_state()
        try:
            self._ensure_protection(
                strategy=BREAKOUT_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=quantity,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
            )
        except Exception as exc:
            self._emergency_close_after_unprotected_entry(
                strategy=BREAKOUT_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=quantity,
                error=exc,
            )
            raise
        self._append_event(
            "live_entry",
            strategy=BREAKOUT_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            side=signal.direction.name,
            quantity=quantity,
            fill_price=fill_price,
            stop_price=position.stop_price,
            take_profit_price=position.take_profit_price,
            v8_lane=profile.lane,
            risk_scale=self.live.risk_scale,
        )
        self.log(
            f"{signal.symbol}: Breakout v8 LIVE开仓 "
            f"{signal.direction.name} qty={quantity:.8g} "
            f"fill={fill_price:.8g}，保护止损已确认"
        )
        return True

    def _open_grid_candidate(
        self, candidate: GridCandidate, now: datetime
    ) -> bool:
        signal = candidate.signal
        profile = self._select_grid_profile(candidate)
        if profile is None:
            return False
        try:
            active = self.client.klines(signal.symbol, "1m", 3)[-1]
            rules = self.client.symbol_rules(signal.symbol)
            account = self.snapshot_account()
        except BinanceRateLimitError:
            raise
        except Exception as exc:
            self.log(f"{signal.symbol}: Grid LIVE成交参考不可用 ({exc})")
            return False
        if not self._entry_account_allows(
            GRID_KEY, signal.symbol, account
        ):
            return False
        available = self._available_notional_multiple(account.equity)
        profile = replace(
            profile,
            portfolio=replace(
                profile.portfolio,
                risk_per_campaign_pct=(
                    profile.portfolio.risk_per_campaign_pct
                    * self.live.risk_scale
                ),
                max_campaign_risk_pct=(
                    profile.portfolio.max_campaign_risk_pct
                    * self.live.risk_scale
                ),
                max_notional_multiple=min(
                    profile.portfolio.max_notional_multiple
                    * self.live.risk_scale,
                    available * NOTIONAL_HEADROOM_SAFETY,
                ),
            ),
        )
        raw = active.close
        live_candidate = replace(
            candidate, entry_minute=minute_token(active.timestamp)
        )
        synthetic = Candle(
            active.timestamp, raw, raw, raw, raw, active.volume
        )
        opened = _create_campaign(
            live_candidate,
            _compact_series([synthetic]),
            rules,
            profile.signal,
            profile.portfolio,
            self.execution,
            account.equity,
        )
        if opened is None:
            return False
        campaign, _initial_fee = opened
        first = campaign.levels[0]
        self._prepare_symbol_for_entry(signal.symbol)
        side = "BUY" if signal.direction == Direction.LONG else "SELL"
        client_id = self._client_id(
            GRID_KEY, signal.event_id, "entry", 0
        )
        self.state["pending_orders"][client_id] = {
            "strategy": GRID_KEY,
            "event_id": signal.event_id,
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "client_id": client_id,
            "created_at": now.isoformat(),
            "model": _grid_campaign_to_dict(campaign),
            "profile": _grid_profile_to_dict(profile),
        }
        self._persist_state()
        submitted_quantity = rules.round_quantity(first.quantity)
        response = self._market_order_idempotent(
            symbol=signal.symbol,
            side=side,
            quantity=submitted_quantity,
            reduce_only=False,
            client_id=client_id,
        )
        try:
            response, quantity, fill_price = self._confirm_market_fill(
                symbol=signal.symbol,
                client_id=client_id,
                response=response,
                expected_quantity=float(submitted_quantity),
            )
        except Exception as exc:
            self._handle_unconfirmed_entry_fill(
                strategy=GRID_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                client_id=client_id,
                error=exc,
            )
            raise
        spacing = signal.atr_value * profile.signal.grid_spacing_atr
        first.quantity = quantity
        campaign.lots[0] = GridLot(
            level_index=0,
            quantity=quantity,
            raw_entry_price=raw,
            entry_price=fill_price,
            entry_fee=0.0,
            entry_slippage=abs(fill_price - raw) * quantity,
            entry_minute=live_candidate.entry_minute,
            target_price=(
                fill_price
                + signal.direction.value
                * spacing
                * profile.signal.grid_target_spacing
            ),
            liquidity="taker",
        )
        campaign.entry_count = max(1, campaign.entry_count)
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self.state["grid_execution_profile"] = _grid_profile_to_dict(
            profile
        )
        self.state["grid_exchange"] = {
            "symbol": signal.symbol,
            "direction": signal.direction.name,
            "event_id": signal.event_id,
            "entry_client_id": client_id,
            "entry_order_id": response.get("orderId"),
            "opened_at": now.isoformat(),
            "orders": {},
        }
        self.state["grid_last_processed_minute"] = (
            campaign.start_minute - 1
        )
        self._record_entry(GRID_KEY, signal.symbol, now)
        self.state["pending_orders"].pop(client_id, None)
        self._persist_state()
        try:
            self._ensure_protection(
                strategy=GRID_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=quantity,
                stop_price=campaign.hard_stop,
            )
            self._place_grid_take_profit(campaign, profile, 0)
            self._ensure_grid_entry_orders(campaign, profile)
        except Exception as exc:
            self._emergency_close_after_unprotected_entry(
                strategy=GRID_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=quantity,
                error=exc,
            )
            raise
        self._append_event(
            "live_entry",
            strategy=GRID_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            side=signal.direction.name,
            quantity=quantity,
            fill_price=fill_price,
            hard_stop=campaign.hard_stop,
            v6_tier=profile.tier,
            risk_scale=self.live.risk_scale,
        )
        self.log(
            f"{signal.symbol}: Grid v6 LIVE campaign启动 "
            f"qty={quantity:.8g} fill={fill_price:.8g}，"
            "硬止损、只减仓止盈与本地加层触发已确认"
        )
        return True

    def _grid_order_book(self) -> dict[str, dict[str, Any]]:
        exchange = self.state.get("grid_exchange")
        if not exchange:
            return {}
        return exchange.setdefault("orders", {})

    def _place_grid_take_profit(
        self,
        campaign: Any,
        profile: GridV5ExecutionProfile,
        level_index: int,
    ) -> None:
        lot = campaign.lots.get(level_index)
        if lot is None:
            return
        signal = campaign.candidate.signal
        rules = self.client.symbol_rules(signal.symbol)
        generation = campaign.levels[level_index].cycles
        client_id = self._client_id(
            GRID_KEY,
            signal.event_id,
            f"tp{level_index}",
            generation,
        )
        response = self._limit_order_idempotent(
            symbol=signal.symbol,
            side="SELL" if signal.direction == Direction.LONG else "BUY",
            quantity=_round_reduce_only_quantity(
                rules, lot.quantity
            ),
            price=rules.round_price(lot.target_price),
            reduce_only=True,
            client_id=client_id,
        )
        self._grid_order_book()[client_id] = {
            "kind": "take_profit",
            "level_index": level_index,
            "cycle": generation,
            "quantity": lot.quantity,
            "handled_executed_quantity": 0.0,
            "price": lot.target_price,
            "status": str(response.get("status", "NEW")).upper(),
            "order_id": response.get("orderId"),
        }
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self.state["grid_execution_profile"] = _grid_profile_to_dict(
            profile
        )
        self._persist_state()

    def _ensure_grid_entry_orders(
        self,
        campaign: Any,
        profile: GridV5ExecutionProfile,
    ) -> None:
        signal = campaign.candidate.signal
        book = self._grid_order_book()
        pending_entries = sum(
            1
            for row in book.values()
            if row.get("kind") == "entry_trigger"
            and row.get("status") == "ARMED"
        )
        remaining = max(
            0,
            profile.signal.max_total_entries
            - campaign.entry_count
            - pending_entries,
        )
        if remaining <= 0:
            return
        for level in campaign.levels:
            if remaining <= 0:
                break
            if level.index in campaign.lots:
                continue
            if level.cycles >= profile.signal.max_cycles_per_level:
                continue
            already_pending = any(
                row.get("kind") == "entry_trigger"
                and int(row.get("level_index", -1)) == level.index
                and row.get("status") == "ARMED"
                for row in book.values()
            )
            if already_pending:
                continue
            client_id = self._client_id(
                GRID_KEY,
                signal.event_id,
                f"ge{level.index}",
                level.cycles,
            )
            book[client_id] = {
                "kind": "entry_trigger",
                "level_index": level.index,
                "cycle": level.cycles,
                "quantity": level.quantity,
                "price": level.raw_price,
                "status": "ARMED",
                "execution": "software_market_after_touch",
            }
            remaining -= 1
        self._persist_state()

    def _trigger_grid_entries(self, now: datetime) -> None:
        if self.state.get("circuit_breaker") or self.state.get(
            "operational_halt"
        ):
            return
        if not self.state.get("grid_campaign"):
            return
        campaign = _grid_campaign_from_dict(
            self.state["grid_campaign"]
        )
        profile = _grid_profile_from_dict(
            self.state["grid_execution_profile"]
        )
        signal = campaign.candidate.signal
        try:
            candle = self.client.klines(signal.symbol, "1m", 3)[-1]
            mark_rows = self.client.premium_index(signal.symbol)
            mark_price = _float(
                mark_rows[-1].get("markPrice")
                if mark_rows
                else None,
                candle.close,
            )
            rules = self.client.symbol_rules(signal.symbol)
        except BinanceRateLimitError:
            raise
        except Exception as exc:
            self.log(
                f"{signal.symbol}: Grid加层触发数据不可用 ({exc})"
            )
            return
        candle_age = max(
            0.0,
            (
                now - (candle.timestamp + timedelta(minutes=1))
            ).total_seconds(),
        )
        if candle_age > self.live.max_market_data_age_seconds:
            self._trip_circuit(
                f"{signal.symbol} market_data_stale={candle_age:.1f}s>"
                f"{self.live.max_market_data_age_seconds}s"
            )
            return
        stop_touched = (
            min(candle.low, mark_price) <= campaign.hard_stop
            if signal.direction == Direction.LONG
            else max(candle.high, mark_price) >= campaign.hard_stop
        )
        if stop_touched:
            # The exchange STOP_MARKET owns this adverse path.  Never add
            # exposure in the same candle, even if a deeper level was touched.
            return
        for client_id, metadata in list(
            self._grid_order_book().items()
        ):
            if (
                metadata.get("kind") != "entry_trigger"
                or metadata.get("status") != "ARMED"
            ):
                continue
            price = _float(metadata["price"])
            touched = (
                candle.close <= price
                if signal.direction == Direction.LONG
                else candle.close >= price
            )
            if not touched:
                continue
            level_index = int(metadata["level_index"])
            if (
                profile.signal.max_total_entries > 0
                and campaign.entry_count
                >= profile.signal.max_total_entries
            ):
                metadata["status"] = "CANCELED_BY_CAMPAIGN_LIMIT"
                continue
            pending = {
                "strategy": GRID_KEY,
                "role": "grid_level",
                "event_id": signal.event_id,
                "symbol": signal.symbol,
                "direction": signal.direction.name,
                "client_id": client_id,
                "level_index": level_index,
                "created_at": now.isoformat(),
            }
            submitted_quantity = rules.round_quantity(
                _float(metadata["quantity"])
            )
            pending["submitted_quantity"] = submitted_quantity
            self.state["pending_orders"][client_id] = pending
            self._persist_state()
            response = self._market_order_idempotent(
                symbol=signal.symbol,
                side=(
                    "BUY"
                    if signal.direction == Direction.LONG
                    else "SELL"
                ),
                quantity=submitted_quantity,
                reduce_only=False,
                client_id=client_id,
            )
            self._finalize_grid_level_fill(client_id, response)
            campaign = _grid_campaign_from_dict(
                self.state["grid_campaign"]
            )

    def _finalize_grid_level_fill(
        self, client_id: str, response: dict[str, Any]
    ) -> None:
        if not self.state.get("grid_campaign"):
            raise RuntimeError(
                "grid level filled without an active local campaign"
            )
        campaign = _grid_campaign_from_dict(
            self.state["grid_campaign"]
        )
        profile = _grid_profile_from_dict(
            self.state["grid_execution_profile"]
        )
        signal = campaign.candidate.signal
        metadata = self._grid_order_book()[client_id]
        level_index = int(metadata["level_index"])
        if level_index in campaign.lots:
            metadata["status"] = "HANDLED"
            self.state["pending_orders"].pop(client_id, None)
            self._persist_state()
            return
        try:
            pending = self.state["pending_orders"].get(
                client_id, {}
            )
            response, quantity, fill_price = self._confirm_market_fill(
                symbol=signal.symbol,
                client_id=client_id,
                response=response,
                expected_quantity=_float(
                    pending.get("submitted_quantity"),
                    _float(
                        response.get("origQty"),
                        _float(metadata["quantity"]),
                    ),
                ),
            )
        except Exception as exc:
            self._handle_unconfirmed_entry_fill(
                strategy=GRID_KEY,
                event_id=signal.event_id,
                symbol=signal.symbol,
                direction=signal.direction,
                client_id=client_id,
                error=exc,
            )
            raise
        spacing = signal.atr_value * profile.signal.grid_spacing_atr
        campaign.lots[level_index] = GridLot(
            level_index=level_index,
            quantity=quantity,
            raw_entry_price=_float(metadata["price"]),
            entry_price=fill_price,
            entry_fee=0.0,
            entry_slippage=abs(
                fill_price - _float(metadata["price"])
            )
            * quantity,
            entry_minute=minute_token(_utc_now()),
            target_price=(
                fill_price
                + signal.direction.value
                * spacing
                * profile.signal.grid_target_spacing
            ),
            liquidity="taker",
        )
        campaign.entry_count += 1
        campaign.max_open_lots = max(
            campaign.max_open_lots, len(campaign.lots)
        )
        metadata["status"] = "HANDLED"
        metadata["order_id"] = response.get("orderId")
        metadata["fill_price"] = fill_price
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self.state["pending_orders"].pop(client_id, None)
        self._persist_state()
        # New exposure is protected before its take-profit is submitted.
        total_quantity = _exact_quantity_sum(
            lot.quantity for lot in campaign.lots.values()
        )
        self._ensure_protection(
            strategy=GRID_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            direction=signal.direction,
            quantity=total_quantity,
            stop_price=campaign.hard_stop,
            force_replace=True,
        )
        self._place_grid_take_profit(
            campaign, profile, level_index
        )
        self._ensure_grid_entry_orders(campaign, profile)
        self._append_event(
            "grid_level_live_fill",
            strategy=GRID_KEY,
            event_id=signal.event_id,
            symbol=signal.symbol,
            level_index=level_index,
            quantity=quantity,
            fill_price=fill_price,
            execution="software_market_after_touch",
        )

    # ------------------------------------------------------------------
    # Reconciliation and restart recovery
    # ------------------------------------------------------------------
    def reconcile(
        self,
        *,
        account: AccountSnapshot | None = None,
        startup: bool = False,
    ) -> AccountSnapshot:
        del startup
        try:
            standard = self.client.open_orders()
            algo = self.client.open_algo_orders()
            expected_algo_before = {
                str(protection.get(key, ""))
                for protection in self.state[
                    "protective_orders"
                ].values()
                for key in ("stop_client_id", "tp_client_id")
                if protection.get(key)
            }
            self._process_pending_initial_orders()
            self._process_pending_grid_level_orders()
            self._process_grid_order_updates(standard)
            if account is None:
                account = self.snapshot_account()
            account = self._reconcile_truth(
                account,
                standard,
                algo,
                expected_algo_before=expected_algo_before,
            )
            self.state["reconcile_failures"] = 0
            self.state["last_reconciled_at"] = _utc_now().isoformat()
            self.state["rate_limit_cooldown_until"] = None
            self._last_account = account
            self._last_full_reconcile_monotonic = time.monotonic()
            self._force_reconcile = False
            if self.stream_cache is not None:
                self._last_user_stream_revision = (
                    self.stream_cache.user_revision
                )
            self._persist_state()
            return account
        except BinanceRateLimitError as exc:
            self.state["rate_limit_cooldown_until"] = (
                _utc_now()
                + timedelta(seconds=exc.retry_after_seconds)
            ).isoformat()
            self._append_event(
                "rate_limit_cooldown",
                status=exc.status,
                proactive=exc.proactive,
                retry_after_seconds=exc.retry_after_seconds,
            )
            self._persist_state()
            raise
        except Exception as exc:
            failures = int(self.state.get("reconcile_failures", 0)) + 1
            self.state["reconcile_failures"] = failures
            self._append_event(
                "reconcile_error",
                failures=failures,
                error=f"{type(exc).__name__}: {exc}",
            )
            if failures >= self.live.max_reconcile_failures:
                self._halt(
                    f"reconciliation failed {failures} times: "
                    f"{type(exc).__name__}: {exc}"
                )
            self._persist_state()
            raise

    def _process_pending_initial_orders(self) -> None:
        now = _utc_now()
        for client_id, pending in list(
            self.state["pending_orders"].items()
        ):
            if "model" not in pending:
                continue
            symbol = str(pending["symbol"])
            response = self._query_standard_safely(
                symbol, client_id
            )
            created = _parse_time(str(pending.get("created_at", "")))
            age = (
                (now - created).total_seconds()
                if created is not None
                else float("inf")
            )
            if response is None:
                if age >= self.live.pending_order_resolution_seconds:
                    self._halt(
                        f"{symbol} initial entry intent unresolved for "
                        f"{age:.1f}s"
                    )
                continue
            status = str(response.get("status", "")).upper()
            pending["exchange_status"] = status
            if status in {
                "CANCELED",
                "CANCELLED",
                "REJECTED",
                "EXPIRED",
                "EXPIRED_IN_MATCH",
            }:
                self.state["pending_orders"].pop(client_id, None)
                self._append_event(
                    "pending_entry_not_filled",
                    strategy=pending["strategy"],
                    symbol=symbol,
                    client_id=client_id,
                    status=status,
                )
            elif status not in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
                self._halt(
                    f"{symbol} initial entry has unknown status {status}"
                )
            elif (
                status == "NEW"
                and age >= self.live.pending_order_resolution_seconds
            ):
                self._halt(
                    f"{symbol} initial market order remained NEW for "
                    f"{age:.1f}s"
                )

    def _process_pending_grid_level_orders(self) -> None:
        for client_id, pending in list(
            self.state["pending_orders"].items()
        ):
            if pending.get("role") != "grid_level":
                continue
            symbol = str(pending["symbol"])
            response = self._query_standard_safely(
                symbol, client_id
            )
            if response is None:
                continue
            status = str(response.get("status", "")).upper()
            if status == "FILLED":
                self._finalize_grid_level_fill(
                    client_id, response
                )
            elif status in _TERMINAL_ORDER_STATUSES:
                self._halt(
                    f"{symbol} grid level order ended as {status}"
                )

    def _close_confirmed_protective_stop_residual(
        self,
        *,
        strategy: str,
        symbol: str,
        direction: Direction,
        modeled_quantity: float,
        exchange_position: LivePosition,
    ) -> AccountSnapshot | None:
        """Flatten only a residual proven to follow our finished stop.

        A stale model/exchange mismatch is not enough authority to trade.
        Recovery requires the owned protective algo to be final, its actual
        reduce-only child order to be fully filled in the expected direction,
        and the child fill plus the current residual to reconcile to the
        modeled position within half a quantity step.
        """

        protection = self.state.get("protective_orders", {}).get(
            strategy, {}
        )
        stop_client_id = str(protection.get("stop_client_id", ""))
        if not stop_client_id:
            return None
        stop_order = self._query_algo_safely(stop_client_id)
        if stop_order is None:
            return None
        stop_status = str(
            stop_order.get("algoStatus", stop_order.get("status", ""))
        ).upper()
        if stop_status not in {"FINISHED", "FILLED"}:
            return None
        actual_order_id = stop_order.get("actualOrderId")
        if actual_order_id in (None, ""):
            return None
        actual_order = self._query_standard_by_order_id_safely(
            symbol, actual_order_id
        )
        if actual_order is None:
            return None

        expected_side = (
            "SELL" if direction == Direction.LONG else "BUY"
        )

        def exchange_bool(value: Any) -> bool:
            return value is True or str(value).strip().lower() == "true"

        if (
            str(stop_order.get("symbol", "")).upper() != symbol
            or str(stop_order.get("side", "")).upper()
            != expected_side
            or not exchange_bool(stop_order.get("reduceOnly"))
            or str(actual_order.get("orderId", ""))
            != str(actual_order_id)
            or str(actual_order.get("symbol", "")).upper() != symbol
            or str(actual_order.get("side", "")).upper()
            != expected_side
            or not exchange_bool(actual_order.get("reduceOnly"))
        ):
            return None
        try:
            stop_fill_quantity, stop_fill_price = self._filled_order(
                actual_order
            )
        except RuntimeError:
            return None
        rules = self.client.symbol_rules(symbol)
        tolerance = max(
            _float(rules.quantity_step) / 2.0, 1e-12
        )
        stop_requested_quantity = _float(
            stop_order.get("quantity")
        )
        expected_residual = max(
            0.0, modeled_quantity - stop_fill_quantity
        )
        if (
            exchange_position.quantity <= tolerance
            or stop_fill_quantity <= tolerance
            or stop_requested_quantity <= tolerance
            or abs(
                stop_requested_quantity - stop_fill_quantity
            )
            > tolerance
            or stop_fill_quantity >= modeled_quantity
            or abs(
                expected_residual - exchange_position.quantity
            )
            > tolerance
        ):
            return None

        event_id, _stop_price, _target = self._protection_model(
            strategy
        )
        if not event_id:
            return None
        generation = int(protection.get("generation", 0))
        residual_client_id = self._client_id(
            strategy, event_id, "resid", generation
        )
        submitted_quantity = _round_reduce_only_quantity(
            rules, exchange_position.quantity
        )
        if _float(submitted_quantity) <= 0.0:
            raise RuntimeError(
                f"{symbol} protective-stop residual rounded to zero"
            )
        self.state["pending_orders"][residual_client_id] = {
            "strategy": strategy,
            "symbol": symbol,
            "event_id": event_id,
            "role": "protective_stop_residual_exit",
            "stop_client_id": stop_client_id,
            "stop_actual_order_id": actual_order_id,
            "created_at": _utc_now().isoformat(),
        }
        self._persist_state()
        response = self._market_order_idempotent(
            symbol=symbol,
            side=expected_side,
            quantity=submitted_quantity,
            reduce_only=True,
            client_id=residual_client_id,
        )
        response, filled_quantity, fill_price = (
            self._confirm_market_fill(
                symbol=symbol,
                client_id=residual_client_id,
                response=response,
                expected_quantity=_float(submitted_quantity),
            )
        )
        verified = self.snapshot_account()
        remaining = verified.positions.get(symbol)
        if remaining is not None and remaining.quantity > tolerance:
            self._halt(
                f"{symbol} confirmed protective-stop residual close "
                f"left {remaining.quantity:.12g}"
            )
            self._persist_state()
            raise RuntimeError(
                f"{symbol} protective-stop residual did not flatten"
            )
        self.state["pending_orders"].pop(
            residual_client_id, None
        )
        self._append_event(
            "protective_stop_residual_closed",
            strategy=strategy,
            symbol=symbol,
            event_id=event_id,
            stop_client_id=stop_client_id,
            stop_actual_order_id=actual_order_id,
            stop_fill_quantity=stop_fill_quantity,
            stop_fill_price=stop_fill_price,
            residual_client_id=residual_client_id,
            residual_order_id=response.get("orderId"),
            residual_quantity=filled_quantity,
            residual_fill_price=fill_price,
        )
        self._persist_state()
        return verified

    def _reconcile_truth(
        self,
        account: AccountSnapshot,
        standard_orders: list[dict[str, Any]],
        algo_orders: list[dict[str, Any]],
        *,
        expected_algo_before: set[str] | None = None,
    ) -> AccountSnapshot:
        prefix = f"{self.live.order_id_prefix}-"
        known_standard_before = set(self._grid_order_book())
        owned_standard = {
            _order_client_id(order): order
            for order in standard_orders
            if _order_client_id(order).startswith(prefix)
        }
        owned_algo = {
            _algo_client_id(order): order
            for order in algo_orders
            if _algo_client_id(order).startswith(prefix)
        }
        if self.live.strict_dedicated_account:
            external_standard = [
                order
                for order in standard_orders
                if not _order_client_id(order).startswith(prefix)
            ]
            external_algo = [
                order
                for order in algo_orders
                if not _algo_client_id(order).startswith(prefix)
            ]
            if external_standard or external_algo:
                self._halt(
                    "dedicated account contains external open orders"
                )
                # Keep reconciling our own positions.  An unrelated order must
                # block new exposure, but it must never prevent a missing
                # protective stop from being repaired.

        local = self._local_positions()
        reconciled_account = account
        exchange_closed_reasons: dict[str, str] = {}
        direction_mismatches: set[str] = set()
        for symbol, position in list(account.positions.items()):
            managed = local.get(symbol)
            if managed is None:
                pending = self._pending_for_symbol(symbol)
                if pending is not None:
                    self._recover_pending_entry(pending, position)
                    local = self._local_positions()
                    managed = local.get(symbol)
                if managed is None:
                    self._halt(
                        f"unmanaged exchange position detected: {symbol}"
                    )
                    continue
            strategy, direction, expected = managed
            rules = self.client.symbol_rules(symbol)
            tolerance = max(_float(rules.quantity_step) / 2.0, 1e-12)
            if direction != position.direction:
                direction_mismatches.add(symbol)
                self._halt(f"{symbol} direction differs from live ledger")
            elif abs(expected - position.quantity) > tolerance:
                verified = (
                    self._close_confirmed_protective_stop_residual(
                        strategy=strategy,
                        symbol=symbol,
                        direction=direction,
                        modeled_quantity=expected,
                        exchange_position=position,
                    )
                )
                if verified is not None:
                    reconciled_account = verified
                    exchange_closed_reasons[symbol] = (
                        "protective_stop_residual_closed"
                    )
                    continue
                self._halt(
                    f"{symbol} quantity differs: "
                    f"ledger={expected:.12g} exchange={position.quantity:.12g}"
                )
            if position.leverage != self.config.trading.leverage:
                self._halt(
                    f"{symbol} leverage differs: "
                    f"required={self.config.trading.leverage}x "
                    f"exchange={position.leverage}x"
                )
            if position.margin_type.strip().upper() not in {
                "CROSS",
                "CROSSED",
            }:
                self._halt(
                    f"{symbol} margin type is not CROSSED: "
                    f"{position.margin_type}"
                )

        account = reconciled_account
        for symbol, (strategy, _direction, _quantity) in local.items():
            if symbol in account.positions:
                continue
            self._handle_exchange_closed(
                strategy,
                symbol,
                exit_reason=exchange_closed_reasons.get(
                    symbol,
                    "exchange_closed_or_protection_triggered",
                ),
            )

        for pending in self.state["pending_orders"].values():
            if "model" not in pending:
                continue
            if (
                pending.get("exchange_status")
                in {"FILLED", "PARTIALLY_FILLED"}
                and pending.get("symbol") not in account.positions
            ):
                self._halt(
                    f"{pending.get('symbol')} filled entry has no "
                    "exchange position; trade-history review required"
                )

        open_algo_ids = set(owned_algo)
        local = self._local_positions()
        for symbol, (strategy, direction, quantity) in local.items():
            if symbol not in account.positions:
                continue
            if symbol in direction_mismatches:
                continue
            event_id, stop, target = self._protection_model(strategy)
            if not event_id or stop <= 0.0:
                self._halt(f"{symbol} live ledger lost protective-stop data")
                continue
            self._ensure_protection(
                strategy=strategy,
                event_id=event_id,
                symbol=symbol,
                direction=direction,
                quantity=account.positions[symbol].quantity,
                stop_price=stop,
                take_profit_price=target,
                open_algo_ids=open_algo_ids,
            )

        pending_ids = set(self.state["pending_orders"])
        unexpected_owned = (
            set(owned_standard)
            - known_standard_before
            - pending_ids
        )
        if unexpected_owned:
            for client_id in sorted(unexpected_owned):
                order = owned_standard[client_id]
                self._safe_cancel_standard(
                    str(order.get("symbol", "")).upper(),
                    client_id,
                )
            self._halt(
                "untracked strategy orders canceled: "
                + ",".join(sorted(unexpected_owned)[:3])
            )

        expected_algo_ids: set[str] = set()
        for protection in self.state["protective_orders"].values():
            for key in ("stop_client_id", "tp_client_id"):
                client_id = str(protection.get(key, ""))
                if client_id:
                    expected_algo_ids.add(client_id)
        # A repair can replace a protection order after this cycle's single
        # exchange snapshot.  IDs that were expected at snapshot time are
        # intentionally retired and are checked again on the next cycle.
        intentionally_replaced = (
            set(expected_algo_before or ()) - expected_algo_ids
        )
        stale_algo_ids = (
            set(owned_algo)
            - expected_algo_ids
            - intentionally_replaced
        )
        if stale_algo_ids:
            for client_id in sorted(stale_algo_ids):
                self._safe_cancel_algo(client_id)
            self._halt(
                "stale strategy algo orders canceled: "
                + ",".join(sorted(stale_algo_ids)[:3])
            )
        return account

    def _pending_for_symbol(
        self, symbol: str
    ) -> dict[str, Any] | None:
        rows = [
            row
            for row in self.state["pending_orders"].values()
            if row.get("symbol") == symbol and "model" in row
        ]
        return rows[0] if len(rows) == 1 else None

    def _recover_pending_entry(
        self,
        pending: dict[str, Any],
        position: LivePosition,
    ) -> None:
        strategy = str(pending["strategy"])
        direction = Direction[str(pending["direction"])]
        if position.direction != direction:
            self._halt(
                f"{position.symbol} pending entry direction mismatch"
            )
            return
        event_id = str(pending["event_id"])
        client_id = str(pending["client_id"])
        if strategy == BREAKOUT_KEY:
            protected, profile = _v7_position_from_dict(
                dict(pending["model"])
            )
            model = protected.position
            model.quantity = position.quantity
            model.entry_price = position.entry_price
            model.risk_budget = model.unit_risk * position.quantity
            protected.original_quantity = position.quantity
            protected.original_risk_budget = model.risk_budget
            self.state["breakout_position"] = _v7_position_to_dict(
                protected, profile
            )
            self.state["breakout_exchange"] = {
                "symbol": position.symbol,
                "direction": direction.name,
                "event_id": event_id,
                "entry_client_id": client_id,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "opened_at": pending["created_at"],
            }
        else:
            campaign = _grid_campaign_from_dict(dict(pending["model"]))
            profile = _grid_profile_from_dict(dict(pending["profile"]))
            lot = campaign.lots[0]
            lot.quantity = position.quantity
            lot.entry_price = position.entry_price
            campaign.levels[0].quantity = position.quantity
            self.state["grid_campaign"] = _grid_campaign_to_dict(
                campaign
            )
            self.state["grid_execution_profile"] = _grid_profile_to_dict(
                profile
            )
            self.state["grid_exchange"] = {
                "symbol": position.symbol,
                "direction": direction.name,
                "event_id": event_id,
                "entry_client_id": client_id,
                "opened_at": pending["created_at"],
                "orders": {},
            }
        self.state["pending_orders"].pop(client_id, None)
        self._append_event(
            "restart_entry_recovered",
            strategy=strategy,
            symbol=position.symbol,
            event_id=event_id,
            quantity=position.quantity,
        )
        self._persist_state()

    def _local_positions(
        self,
    ) -> dict[str, tuple[str, Direction, float]]:
        output: dict[str, tuple[str, Direction, float]] = {}
        breakout = self.state.get("breakout_position")
        if breakout:
            protected, _profile = _v7_position_from_dict(breakout)
            position = protected.position
            output[position.candidate.signal.symbol] = (
                BREAKOUT_KEY,
                position.candidate.signal.direction,
                position.quantity,
            )
        grid = self.state.get("grid_campaign")
        if grid:
            campaign = _grid_campaign_from_dict(grid)
            output[campaign.candidate.signal.symbol] = (
                GRID_KEY,
                campaign.candidate.signal.direction,
                _exact_quantity_sum(
                    lot.quantity for lot in campaign.lots.values()
                ),
            )
        return output

    def _protection_model(
        self, strategy: str
    ) -> tuple[str, float, float]:
        if strategy == BREAKOUT_KEY and self.state.get(
            "breakout_position"
        ):
            protected, _profile = _v7_position_from_dict(
                self.state["breakout_position"]
            )
            position = protected.position
            return (
                position.candidate.signal.event_id,
                position.stop_price,
                position.take_profit_price,
            )
        if strategy == GRID_KEY and self.state.get("grid_campaign"):
            campaign = _grid_campaign_from_dict(
                self.state["grid_campaign"]
            )
            return (
                campaign.candidate.signal.event_id,
                campaign.hard_stop,
                0.0,
            )
        return "", 0.0, 0.0

    def _process_grid_order_updates(
        self, open_orders: list[dict[str, Any]]
    ) -> None:
        if not self.state.get("grid_campaign"):
            return
        campaign = _grid_campaign_from_dict(
            self.state["grid_campaign"]
        )
        profile = _grid_profile_from_dict(
            self.state["grid_execution_profile"]
        )
        symbol = campaign.candidate.signal.symbol
        open_by_id = {
            _order_client_id(order): order for order in open_orders
        }
        changed = False
        for client_id, metadata in list(
            self._grid_order_book().items()
        ):
            if metadata.get("kind") == "entry_trigger":
                continue
            if metadata.get("status") == "HANDLED":
                continue
            order = open_by_id.get(client_id)
            if order is None:
                order = self._query_standard_safely(symbol, client_id)
            if order is None:
                continue
            status = str(order.get("status", "")).upper()
            metadata["status"] = status
            if status not in {"PARTIALLY_FILLED", "FILLED"}:
                continue
            level_index = int(metadata["level_index"])
            lot = campaign.lots.get(level_index)
            executed = _float(
                order.get("executedQty"),
                _float(metadata.get("quantity"))
                if status == "FILLED"
                else 0.0,
            )
            handled = _float(
                metadata.get("handled_executed_quantity")
            )
            delta = max(0.0, executed - handled)
            if lot is not None and delta > 0.0:
                lot.quantity = max(0.0, lot.quantity - delta)
                metadata["handled_executed_quantity"] = executed
                changed = True
            rules = self.client.symbol_rules(symbol)
            tolerance = max(
                _float(rules.quantity_step) / 2.0, 1e-12
            )
            fully_closed = (
                status == "FILLED"
                or lot is not None
                and lot.quantity <= tolerance
            )
            if fully_closed:
                if level_index in campaign.lots:
                    campaign.lots.pop(level_index)
                    campaign.levels[level_index].cycles += 1
                    campaign.grid_take_profit_count += 1
                    changed = True
                metadata["status"] = "HANDLED"
        if changed:
            self.state["grid_campaign"] = _grid_campaign_to_dict(
                campaign
            )
            self._ensure_grid_entry_orders(campaign, profile)
            quantity = _exact_quantity_sum(
                lot.quantity for lot in campaign.lots.values()
            )
            if quantity > 0.0:
                self._ensure_protection(
                    strategy=GRID_KEY,
                    event_id=campaign.candidate.signal.event_id,
                    symbol=symbol,
                    direction=campaign.candidate.signal.direction,
                    quantity=quantity,
                    stop_price=campaign.hard_stop,
                    force_replace=True,
                )
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        self._persist_state()

    def _handle_exchange_closed(
        self,
        strategy: str,
        symbol: str,
        *,
        exit_reason: str = "exchange_closed_or_protection_triggered",
    ) -> None:
        self._cancel_owned_symbol_orders(symbol)
        if strategy == BREAKOUT_KEY:
            self.state["breakout_position"] = None
            self.state["breakout_exchange"] = None
        else:
            self.state["grid_campaign"] = None
            self.state["grid_execution_profile"] = None
            self.state["grid_exchange"] = None
        self.state["protective_orders"].pop(strategy, None)
        self.state["trades"].append(
            {
                "time": _utc_now().isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "exit_reason": exit_reason,
                "pnl_source": "exchange_trade_history",
            }
        )
        self._append_event(
            "exchange_position_closed",
            strategy=strategy,
            symbol=symbol,
            exit_reason=exit_reason,
        )
        self._persist_state()

    def _cancel_owned_symbol_orders(self, symbol: str) -> None:
        prefix = f"{self.live.order_id_prefix}-"
        for order in self.client.open_orders(symbol):
            client_id = _order_client_id(order)
            if client_id.startswith(prefix):
                self._safe_cancel_standard(symbol, client_id)
        for order in self.client.open_algo_orders(symbol):
            client_id = _algo_client_id(order)
            if client_id.startswith(prefix):
                self._safe_cancel_algo(client_id)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def _manage_breakout_position(self, now: datetime) -> None:
        payload = self.state.get("breakout_position")
        if not payload:
            return
        protected, profile = _v7_position_from_dict(payload)
        position = protected.position
        symbol = position.candidate.signal.symbol
        try:
            candles = _closed_candles(
                self.client.klines(symbol, "1m", 1500), 1, now
            )
            rules = self.client.symbol_rules(symbol)
        except BinanceRateLimitError:
            raise
        except Exception as exc:
            self.log(f"{symbol}: Breakout LIVE持仓数据不可用 ({exc})")
            return
        last = int(
            self.state.get("breakout_last_processed_minute")
            or position.entry_minute - 1
        )
        if not self._history_is_contiguous(
            BREAKOUT_KEY, symbol, candles, last
        ):
            self._trip_circuit(f"{symbol} closed-1m history gap")
            return
        series = _compact_series(candles)
        for index, minute_value in enumerate(series.minutes):
            minute = int(minute_value)
            if minute <= last:
                continue
            trial = copy.deepcopy(protected)
            trade, _cash_delta = _process_protected_bar(
                trial,
                minute,
                series,
                index,
                profile.signal,
                profile.exit_protection,
                self.execution,
                rules,
            )
            last = minute
            self.state["breakout_last_processed_minute"] = minute
            if trade is not None:
                self._close_live_position(
                    BREAKOUT_KEY,
                    symbol,
                    position.candidate.signal.direction,
                    str(trade.get("exit_reason", "model_exit")),
                )
                return
            stop_changed = not math.isclose(
                trial.position.stop_price,
                protected.position.stop_price,
                rel_tol=1e-12,
            )
            protected = trial
            if stop_changed:
                self._ensure_protection(
                    strategy=BREAKOUT_KEY,
                    event_id=position.candidate.signal.event_id,
                    symbol=symbol,
                    direction=position.candidate.signal.direction,
                    quantity=protected.position.quantity,
                    stop_price=protected.position.stop_price,
                    take_profit_price=protected.position.take_profit_price,
                    force_replace=True,
                )
        self.state["breakout_position"] = _v7_position_to_dict(
            protected, profile
        )

    def _manage_grid_campaign(
        self, now: datetime, account: AccountSnapshot | None = None
    ) -> None:
        if not self.state.get("grid_campaign"):
            return
        campaign = _grid_campaign_from_dict(
            self.state["grid_campaign"]
        )
        profile = _grid_profile_from_dict(
            self.state["grid_execution_profile"]
        )
        symbol = campaign.candidate.signal.symbol
        account = account or self.snapshot_account()
        position = account.positions.get(symbol)
        if position is None:
            return
        current_pnl = position.unrealized_pnl
        campaign.best_equity_pnl = max(
            campaign.best_equity_pnl, current_pnl
        )
        campaign.worst_equity_pnl = min(
            campaign.worst_equity_pnl, current_pnl
        )
        reason = ""
        if (
            profile.signal.campaign_loss_limit_r > 0.0
            and current_pnl
            <= -campaign.risk_budget
            * profile.signal.campaign_loss_limit_r
        ):
            reason = "campaign_loss_limit"
        elif (
            minute_token(now) - campaign.start_minute
            >= profile.signal.max_campaign_minutes
        ):
            reason = "campaign_time_stop"
        else:
            try:
                hourly = _closed_candles(
                    self.client.klines(symbol, "1h", 500), 60, now
                )
                snapshots, _signals = build_trend_grid_timeline(
                    symbol, hourly, profile.signal
                )
                if snapshots:
                    latest = snapshots[-1]
                    available = minute_token(latest.available_time)
                    last_regime = int(
                        self.state.get("grid_last_regime_minute") or -1
                    )
                    if available > last_regime:
                        self.state["grid_last_regime_minute"] = available
                        if latest.exit_invalid(
                            campaign.candidate.signal.direction,
                            profile.signal.regime_exit_mode,
                        ):
                            campaign.regime_invalid_bars += 1
                        else:
                            campaign.regime_invalid_bars = 0
                    if (
                        campaign.regime_invalid_bars
                        >= profile.signal.regime_exit_confirm_bars
                    ):
                        reason = "ema_regime_exit"
            except BinanceRateLimitError:
                raise
            except Exception as exc:
                self.log(f"{symbol}: Grid LIVE regime数据不可用 ({exc})")
        self.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
        if reason:
            self._close_live_position(
                GRID_KEY,
                symbol,
                campaign.candidate.signal.direction,
                reason,
            )
        else:
            self._trigger_grid_entries(now)

    def _market_data_is_fresh(self, now: datetime) -> bool:
        rows = self.client.klines("BTCUSDT", "1m", 3)
        if not rows:
            raise RuntimeError("BTCUSDT 1m market data is empty")
        latest = max(rows, key=lambda row: row.timestamp)
        close_time = latest.timestamp + timedelta(minutes=1)
        age_seconds = max(0.0, (now - close_time).total_seconds())
        if age_seconds <= self.live.max_market_data_age_seconds:
            return True
        self._trip_circuit(
            "market_data_stale="
            f"{age_seconds:.1f}s>{self.live.max_market_data_age_seconds}s"
        )
        return False

    def _close_live_position(
        self,
        strategy: str,
        symbol: str,
        direction: Direction,
        reason: str,
    ) -> None:
        account = self.snapshot_account()
        position = account.positions.get(symbol)
        if position is None:
            self._handle_exchange_closed(strategy, symbol)
            return
        rules = self.client.symbol_rules(symbol)
        event_id, _stop, _target = self._protection_model(strategy)
        client_id = self._client_id(
            strategy, event_id, f"exit{hashlib.sha1(reason.encode()).hexdigest()[:4]}"
        )
        self.state["pending_orders"][client_id] = {
            "strategy": strategy,
            "symbol": symbol,
            "event_id": event_id,
            "role": "reduce_only_exit",
            "reason": reason,
            "created_at": _utc_now().isoformat(),
        }
        self._persist_state()
        self._market_order_idempotent(
            symbol=symbol,
            side="SELL" if direction == Direction.LONG else "BUY",
            quantity=_round_reduce_only_quantity(
                rules, position.quantity
            ),
            reduce_only=True,
            client_id=client_id,
        )
        verified = self.snapshot_account()
        residual = verified.positions.get(symbol)
        tolerance = max(_float(rules.quantity_step) / 2.0, 1e-12)
        if residual is not None and residual.quantity > tolerance:
            # Keep the local ledger and all protective orders intact.  Clearing
            # either before exchange truth is flat would turn a partial/failed
            # close into an unmanaged position.
            self._halt(
                f"{symbol} reduce-only exit left residual "
                f"{residual.quantity:.12g}; manual reconciliation required"
            )
            self._persist_state()
            return
        self._cancel_owned_symbol_orders(symbol)
        self.state["pending_orders"].pop(client_id, None)
        if strategy == BREAKOUT_KEY:
            self.state["breakout_position"] = None
            self.state["breakout_exchange"] = None
        else:
            self.state["grid_campaign"] = None
            self.state["grid_execution_profile"] = None
            self.state["grid_exchange"] = None
        self.state["protective_orders"].pop(strategy, None)
        self.state["trades"].append(
            {
                "time": _utc_now().isoformat(),
                "strategy": strategy,
                "symbol": symbol,
                "exit_reason": reason,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "pnl_source": "exchange_trade_history",
            }
        )
        self._append_event(
            "live_exit",
            strategy=strategy,
            symbol=symbol,
            reason=reason,
            quantity=position.quantity,
            reduce_only=True,
        )
        self._persist_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _full_reconciliation_due(self) -> bool:
        if self._last_account is None or self._force_reconcile:
            return True
        elapsed = (
            time.monotonic() - self._last_full_reconcile_monotonic
        )
        interval = self.live.rest_reconcile_interval_seconds
        if self.stream_cache is None:
            return elapsed >= self.live.rest_reconcile_fallback_seconds
        health = self.stream_cache.health()
        self.state["last_stream_health"] = health
        revision = self.stream_cache.user_revision
        if revision > self._last_user_stream_revision:
            return True
        if not (
            health.get("market_healthy")
            and health["user_connected"]
        ):
            interval = self.live.rest_reconcile_fallback_seconds
        return elapsed >= interval

    def run_once(
        self, stop_event: threading.Event | None = None
    ) -> None:
        self.validate_startup_settings_only()
        if self._full_reconciliation_due():
            account = self.reconcile()
        elif self._last_account is not None:
            account = self._last_account
        else:  # Defensive; the due check above normally owns this case.
            account = self.reconcile()
        self._update_risk_circuits(account)
        now = _utc_now()
        market_data_fresh = self._market_data_is_fresh(now)
        self._manage_breakout_position(now)
        self._manage_grid_campaign(now, account)
        if (
            market_data_fresh
            and not self.state.get("circuit_breaker")
            and not self.state.get("operational_halt")
        ):
            self._scan_and_maybe_enter(now, stop_event)
        if self._last_account is not None:
            account = self._last_account
        self._persist_state()
        self._write_status(account)
        if self.account_callback:
            self.account_callback(account)
        heartbeat = max(30, int(self.live.heartbeat_seconds))
        if time.time() - self._last_heartbeat >= heartbeat:
            self._last_heartbeat = time.time()
            self.log(
                f"v8/v6 LIVE: 权益={account.equity:.2f}U "
                f"持仓={len(account.positions)}/2 "
                f"熔断={'是' if self.state['circuit_breaker'] else '否'} "
                f"锁定={'是' if self.state['operational_halt'] else '否'}"
            )

    def run_forever(self, stop_event: threading.Event) -> None:
        self._acquire_runtime_lock()
        try:
            self.validate_transport_acceptance()
            self._start_streams(stop_event)
            self.validate_startup()
            self.log(
                "Breakout v8 + Grid v6 LIVE已启动；"
                "交易所成交为唯一仓位真相"
            )
            while not stop_event.is_set():
                started = time.time()
                try:
                    self.run_once(stop_event)
                    self.state["consecutive_api_failures"] = 0
                except BinanceRateLimitError as exc:
                    self.state["rate_limit_cooldown_until"] = (
                        _utc_now()
                        + timedelta(
                            seconds=exc.retry_after_seconds
                        )
                    ).isoformat()
                    self._force_reconcile = True
                    self.log(
                        "Binance REST进入强制冷却 "
                        f"{exc.retry_after_seconds:.1f}s；"
                        "冷却期间停止全部REST轮询与新开仓"
                    )
                    self._persist_state()
                    if self._wait_rate_limit_cooldown(
                        stop_event, exc.retry_after_seconds
                    ):
                        break
                except Exception as exc:
                    failures = int(
                        self.state.get("consecutive_api_failures", 0)
                    ) + 1
                    self.state["consecutive_api_failures"] = failures
                    self.log(
                        f"LIVE循环错误 {failures}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if failures >= self.live.max_consecutive_api_failures:
                        self._trip_circuit(
                            f"API failures={failures}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    self._persist_state()
                elapsed = time.time() - started
                stop_event.wait(
                    max(
                        1.0,
                        self.config.trading.poll_seconds - elapsed,
                    )
                )
            self._append_event("live_runner_stopped")
            self.log(
                "v8/v6 LIVE已停止；现有交易所保护单保持有效"
            )
        finally:
            self._stop_streams()
            self._release_runtime_lock()

    @staticmethod
    def _wait_rate_limit_cooldown(
        stop_event: threading.Event, seconds: float
    ) -> bool:
        remaining = max(0.0, float(seconds))
        while remaining > 0.0:
            started = time.monotonic()
            if stop_event.wait(min(30.0, remaining)):
                return True
            remaining -= max(0.0, time.monotonic() - started)
        return stop_event.is_set()

    def _append_event(self, event_type: str, **payload: Any) -> None:
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "logged_at": _utc_now().isoformat(),
            "config_hash": self.config_hash,
            "frozen_version": self.live.frozen_version,
            "event_type": event_type,
            **payload,
        }
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _jsonable(row),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_status(self, account: AccountSnapshot) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        stream_health = (
            self.stream_cache.health()
            if self.stream_cache is not None
            else self.state.get("last_stream_health", {})
        )
        rate_limit = (
            self.client.rate_limit_status()
            if hasattr(self.client, "rate_limit_status")
            else {}
        )
        payload = {
            "generated_at": _utc_now().isoformat(),
            "mode": "LIVE",
            "armed": self.live.armed,
            "environment": self.config.exchange.environment,
            "strategy_name": self.live.strategy_name,
            "frozen_version": self.live.frozen_version,
            "config_hash": self.config_hash,
            "risk_scale": self.live.risk_scale,
            "equity": account.equity,
            "available_balance": account.available_balance,
            "position_count": len(account.positions),
            "positions": [
                {
                    "symbol": row.symbol,
                    "direction": row.direction.name,
                    "quantity": row.quantity,
                    "entry_price": row.entry_price,
                    "mark_price": row.mark_price,
                    "unrealized_pnl": row.unrealized_pnl,
                }
                for row in account.position_rows
            ],
            "circuit_breaker": self.state["circuit_breaker"],
            "circuit_reason": self.state["circuit_reason"],
            "operational_halt": self.state["operational_halt"],
            "halt_reason": self.state["halt_reason"],
            "last_reconciled_at": self.state["last_reconciled_at"],
            "transport_version": self.live.transport_version,
            "stream_health": stream_health,
            "rate_limit": rate_limit,
            "rate_limit_cooldown_until": self.state.get(
                "rate_limit_cooldown_until"
            ),
            "protective_orders": self.state["protective_orders"],
            "important_note": (
                "This is an execution/status report, not a performance "
                "claim. Realized PnL must be reconciled from exchange trades."
            ),
        }
        temporary = self.report_path.with_suffix(
            self.report_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                _jsonable(payload), indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        temporary.replace(self.report_path)

    def log(self, message: str) -> None:
        self.logger(
            message.replace("组合shadow", "v8/v6 LIVE").replace(
                "shadow", "LIVE"
            )
        )


def _logger(message: str) -> None:
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Breakout v8 + Grid v6 isolated live runner"
    )
    parser.add_argument(
        "--config",
        default="config.gui.breakout-v8-grid-v6-max2-live.json",
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_live_config(args.config)
    client = BinanceFuturesClient(
        api_key=None,
        api_secret=None,
        environment=config.exchange.environment,
        recv_window=config.exchange.recv_window,
        timeout_seconds=config.exchange.timeout_seconds,
    )
    # CLI intentionally does not read secrets.  Use the GUI, which obtains
    # credentials from the configured environment/secret store and presents
    # the second runtime confirmation prompt.
    trader = CombinedBreakoutV8GridV6LiveTrader(
        config, client, logger=_logger
    )
    if args.once:
        trader.validate_startup()
        trader.run_once()
        return 0
    raise RuntimeError(
        "live CLI is credential-locked; start through the GUI confirmation"
    )


if __name__ == "__main__":
    raise SystemExit(main())
