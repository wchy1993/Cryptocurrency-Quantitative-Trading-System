from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .combined_breakout_v8_grid_v6_shadow import (
    CombinedBreakoutV8GridV6ShadowTrader,
)
from .combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)
from .combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
    _resolve_config_path,
)
from .live_config import LiveAppConfig
from .trend_grid_v7 import TREND_GRID_V7_NAME
from .volatility_breakout_v9 import VOLATILITY_BREAKOUT_V9_NAME


COMBINED_V9_GRID_V7_NAME = (
    "dual_thrust_volatility_breakout_v9_profit_ladder_plus_"
    "dynamic_trend_following_grid_v7_cycle_profit_floor_max2"
)
BREAKOUT_V9_GRID_V7_SHADOW_VERSION = (
    "breakout_v9_shared_balanced_grid_v7_max2_shadow_20260728"
)
BREAKOUT_V9_COMPONENT_VERSION = (
    "breakout_v9_shared_balanced_20260728"
)


def _load_v9_v7_source_bundle(
    config: LiveAppConfig,
) -> dict[str, Any]:
    shadow = config.combined_volatility_trend_grid_shadow
    combined_path = _resolve_config_path(
        shadow.source_combined_config_path
    )
    combined_payload = json.loads(
        combined_path.read_text(encoding="utf-8")
    )
    sources = combined_payload.get("source_configs", {})
    resolved: dict[str, Path] = {}
    hashes: dict[str, str] = {
        "combined": hashlib.sha256(
            combined_path.read_bytes()
        ).hexdigest()
    }
    for key in (BREAKOUT_KEY, GRID_KEY):
        source = sources.get(key)
        if isinstance(source, str):
            source_path = _resolve_config_path(
                source, combined_path
            )
            expected_hash = str(
                combined_payload.get("source_hashes", {}).get(
                    key, ""
                )
            )
        elif isinstance(source, dict) and source.get("path"):
            source_path = _resolve_config_path(
                str(source["path"]), combined_path
            )
            expected_hash = str(source.get("sha256", ""))
        else:
            raise RuntimeError(
                f"combined v9/v7 source config is missing: {key}"
            )
        actual_hash = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(
                f"{key} source hash differs from the frozen v9/v7 config"
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


def combined_v9_grid_v7_shadow_config_hash(
    config: LiveAppConfig,
) -> str:
    bundle = _load_v9_v7_source_bundle(config)
    payload = {
        "strategy_name": COMBINED_V9_GRID_V7_NAME,
        "combined_shadow": asdict(
            config.combined_volatility_trend_grid_shadow
        ),
        "breakout_shadow": asdict(config.dual_thrust_shadow),
        "source_hashes": bundle["hashes"],
        "environment": config.exchange.environment,
        "trading": asdict(config.trading),
        "risk": asdict(config.risk),
        "execution_order": [
            "closed_1m_only",
            "old_exits_before_new_entries",
            "breakout_v9_priority_before_grid_v7",
            "same_symbol_overlap_forbidden",
            "adverse_stop_first_on_same_bar",
            "entry_time_breakout_profile_frozen",
            "entry_time_grid_profile_frozen",
        ],
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


class CombinedBreakoutV9GridV7ShadowTrader(
    CombinedBreakoutV8GridV6ShadowTrader
):
    """Mainnet-market dry-run for frozen Breakout v9 + Grid v7."""

    combined_strategy_name = COMBINED_V9_GRID_V7_NAME
    shadow_version = BREAKOUT_V9_GRID_V7_SHADOW_VERSION
    breakout_strategy_name = VOLATILITY_BREAKOUT_V9_NAME
    breakout_component_version = BREAKOUT_V9_COMPONENT_VERSION
    grid_strategy_name = TREND_GRID_V7_NAME
    breakout_profile_prefix = "v9"
    grid_profile_prefix = "v7"

    def _source_bundle_for_config(
        self, config: LiveAppConfig
    ) -> dict[str, Any]:
        return _load_v9_v7_source_bundle(config)

    def _shadow_config_hash_for_config(
        self, config: LiveAppConfig
    ) -> str:
        return combined_v9_grid_v7_shadow_config_hash(config)

    def _candidate_reject_reason(
        self, strategy: str, candidate: Any, now: Any
    ) -> str | None:
        reason = super()._candidate_reject_reason(
            strategy, candidate, now
        )
        return {
            "v8_score_allocation_rejected": (
                "v9_score_allocation_rejected"
            ),
            "grid_v6_campaign_policy_rejected": (
                "grid_v7_campaign_policy_rejected"
            ),
        }.get(reason, reason)

    def log(self, message: str) -> None:
        self.logger(
            message.replace("Breakout v7", "Breakout v9")
            .replace("Breakout v8", "Breakout v9")
            .replace("Grid v5", "Grid v7")
            .replace("Grid v6", "Grid v7")
            .replace("v8/v6", "v9/v7")
        )

    def _append_event(
        self, event_type: str, **payload: Any
    ) -> None:
        for old_key in ("v7_lane", "v8_lane", "v6_lane"):
            if old_key in payload:
                payload["v9_lane"] = payload.pop(old_key)
        for old_key in ("v5_tier", "v6_tier"):
            if old_key in payload:
                payload["v7_tier"] = payload.pop(old_key)
        CombinedVolatilityTrendGridShadowTrader._append_event(
            self, event_type, **payload
        )
