from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .combined_breakout_v8_grid_v6_live import (
    LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES,
    LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS,
)
from .combined_breakout_v8_grid_v6_live_acceptance import (
    run_mainnet_dry_run_stress as _run_transport_acceptance,
)
from .combined_breakout_v9_grid_v7_live import (
    BREAKOUT_V9_GRID_V7_TRANSPORT_VERSION,
    CombinedBreakoutV9GridV7LiveTrader,
    _v9_v7_transport_code_hashes,
    combined_v9_grid_v7_live_config_hash,
)
from .combined_breakout_v9_grid_v7_shadow import (
    CombinedBreakoutV9GridV7ShadowTrader,
)


DEFAULT_DRY_RUN_CONFIG = (
    "config.gui.breakout-v9-grid-v7-max2-shadow.json"
)
DEFAULT_LIVE_CONFIG = (
    "config.gui.breakout-v9-grid-v7-max2-live.json"
)


def run_mainnet_dry_run_stress(
    *,
    dry_run_config_path: str | Path = DEFAULT_DRY_RUN_CONFIG,
    live_config_path: str | Path = DEFAULT_LIVE_CONFIG,
    duration_seconds: float = LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS,
    stress_cycles: int = LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES,
    require_user_stream: bool = True,
) -> dict[str, Any]:
    return _run_transport_acceptance(
        dry_run_config_path=dry_run_config_path,
        live_config_path=live_config_path,
        duration_seconds=duration_seconds,
        stress_cycles=stress_cycles,
        require_user_stream=require_user_stream,
        shadow_trader_class=CombinedBreakoutV9GridV7ShadowTrader,
        live_trader_class=CombinedBreakoutV9GridV7LiveTrader,
        live_config_hash_fn=combined_v9_grid_v7_live_config_hash,
        transport_code_hashes_fn=_v9_v7_transport_code_hashes,
        transport_version=BREAKOUT_V9_GRID_V7_TRANSPORT_VERSION,
        temporary_prefix="b9g7-live-transport-acceptance-",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Breakout v9 / Grid v7 mainnet no-order transport acceptance"
        )
    )
    parser.add_argument(
        "--dry-run-config", default=DEFAULT_DRY_RUN_CONFIG
    )
    parser.add_argument(
        "--live-config", default=DEFAULT_LIVE_CONFIG
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS,
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES,
    )
    parser.add_argument("--public-only", action="store_true")
    args = parser.parse_args()
    report = run_mainnet_dry_run_stress(
        dry_run_config_path=args.dry_run_config,
        live_config_path=args.live_config,
        duration_seconds=args.duration_seconds,
        stress_cycles=args.cycles,
        require_user_stream=not args.public_only,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
