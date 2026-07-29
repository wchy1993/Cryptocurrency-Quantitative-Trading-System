from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .combined_breakout_v8_grid_v6_live import (
    _TRANSPORT_ONLY_CONFIG_KEYS,
    _planner_config,
    CombinedBreakoutV8GridV6LiveTrader,
)
from .combined_breakout_v9_grid_v7_shadow import (
    BREAKOUT_V9_COMPONENT_VERSION,
    BREAKOUT_V9_GRID_V7_SHADOW_VERSION,
    COMBINED_V9_GRID_V7_NAME,
    _load_v9_v7_source_bundle,
)
from .combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
)
from .live_config import LiveAppConfig
from .trend_grid_v7 import TREND_GRID_V7_NAME
from .volatility_breakout_v9 import VOLATILITY_BREAKOUT_V9_NAME


BREAKOUT_V9_GRID_V7_LIVE_VERSION = (
    "breakout_v9_shared_balanced_grid_v7_max2_live_risk1_20260728"
)
BREAKOUT_V9_GRID_V7_TRANSPORT_VERSION = (
    "binance_usdm_ws_weighted_v1"
)
LIVE_V9_GRID_V7_REASON_TOKEN = (
    "combined_breakout_v9_grid_v7_live"
)


def combined_v9_grid_v7_live_config_hash(
    config: LiveAppConfig,
) -> str:
    bundle = _load_v9_v7_source_bundle(
        _planner_config(config)
    )
    live_payload = asdict(
        config.combined_breakout_v8_grid_v6_live
    )
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


def _v9_v7_transport_code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    package = root / "crypto_scalper"
    pending = [
        "combined_breakout_v8_grid_v6_live.py",
        "combined_breakout_v8_grid_v6_live_acceptance.py",
        "combined_breakout_v8_grid_v6_shadow.py",
        "combined_breakout_v9_grid_v7_live.py",
        "combined_breakout_v9_grid_v7_live_acceptance.py",
        "combined_breakout_v9_grid_v7_shadow.py",
    ]
    dependencies: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in dependencies:
            continue
        path = package / relative
        if not path.exists():
            raise RuntimeError(
                f"v9/v7 live dependency is missing: {relative}"
            )
        dependencies.add(relative)
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.level != 1
                or not node.module
            ):
                continue
            candidate = (
                node.module.replace(".", "/") + ".py"
            )
            if (
                (package / candidate).exists()
                and candidate not in dependencies
            ):
                pending.append(candidate)
    relative_paths = tuple(
        f"crypto_scalper/{relative}"
        for relative in sorted(dependencies)
    )
    return {
        relative: hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
        for relative in relative_paths
    }


class CombinedBreakoutV9GridV7LiveTrader(
    CombinedBreakoutV8GridV6LiveTrader
):
    """Exchange-backed v9/v7 Max2 runner with isolated state."""

    combined_strategy_name = COMBINED_V9_GRID_V7_NAME
    shadow_version = BREAKOUT_V9_GRID_V7_SHADOW_VERSION
    breakout_strategy_name = VOLATILITY_BREAKOUT_V9_NAME
    breakout_component_version = BREAKOUT_V9_COMPONENT_VERSION
    grid_strategy_name = TREND_GRID_V7_NAME
    breakout_profile_prefix = "v9"
    grid_profile_prefix = "v7"
    live_version = BREAKOUT_V9_GRID_V7_LIVE_VERSION
    transport_version = BREAKOUT_V9_GRID_V7_TRANSPORT_VERSION
    required_live_confirmation_text = (
        "CONFIRM_BREAKOUT_V9_GRID_V7_LIVE"
    )
    live_reason_token = LIVE_V9_GRID_V7_REASON_TOKEN

    def _source_bundle_for_live_config(
        self, config: LiveAppConfig
    ) -> dict[str, Any]:
        return _load_v9_v7_source_bundle(
            _planner_config(config)
        )

    def _live_config_hash_for_config(
        self, config: LiveAppConfig
    ) -> str:
        return combined_v9_grid_v7_live_config_hash(config)

    def _expected_transport_code_hashes(self) -> dict[str, str]:
        return _v9_v7_transport_code_hashes()

    def log(self, message: str) -> None:
        self.logger(
            message.replace("Breakout v8", "Breakout v9")
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
