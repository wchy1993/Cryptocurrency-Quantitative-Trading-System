from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .binance_client import BinanceFuturesClient
from .binance_rate_limit import RequestWeightBudget
from .binance_streams import BinanceFuturesStreamCache
from .combined_breakout_v8_grid_v6_live import (
    BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION,
    CombinedBreakoutV8GridV6LiveTrader,
    LIVE_TRANSPORT_ACCEPTANCE_MIN_CYCLES,
    LIVE_TRANSPORT_ACCEPTANCE_MIN_SECONDS,
    _transport_code_hashes,
    combined_v8_grid_v6_live_config_hash,
)
from .combined_breakout_v8_grid_v6_shadow import (
    CombinedBreakoutV8GridV6ShadowTrader,
)
from .live_config import LiveAppConfig, load_live_config
from .secrets import read_secret


DEFAULT_DRY_RUN_CONFIG = (
    "config.gui.breakout-v8-grid-v6-max2-shadow.json"
)
DEFAULT_LIVE_CONFIG = (
    "config.gui.breakout-v8-grid-v6-max2-live.json"
)


class DryRunGuardedBinanceClient(BinanceFuturesClient):
    """Binance client that makes order submission impossible."""

    _FORBIDDEN_PATH_TOKENS = (
        "/order",
        "/algoOrder",
        "/allOpenOrders",
        "/algoOpenOrders",
        "/leverage",
        "/marginType",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.blocked_mutation_attempts: dict[str, int] = {}

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        api_key: bool = False,
    ) -> Any:
        if (
            method.upper() in {"POST", "PUT", "DELETE"}
            and path != "/fapi/v1/listenKey"
            and any(
                token in path for token in self._FORBIDDEN_PATH_TOKENS
            )
        ):
            endpoint = f"{method.upper()} {path}"
            self.blocked_mutation_attempts[endpoint] = (
                self.blocked_mutation_attempts.get(endpoint, 0) + 1
            )
            raise RuntimeError(
                f"DRY-RUN acceptance blocked mutation: {method} {path}"
            )
        return super()._request(
            method, path, params, api_key=api_key
        )


def _budget_for(config: LiveAppConfig) -> RequestWeightBudget:
    live = config.combined_breakout_v8_grid_v6_live
    return RequestWeightBudget(
        limit=live.request_weight_limit,
        soft_limit_ratio=live.request_weight_soft_limit_ratio,
        default_cooldown_seconds=(
            live.rate_limit_default_cooldown_seconds
        ),
    )


def _mainnet_client(
    config: LiveAppConfig,
    *,
    guarded: bool,
) -> BinanceFuturesClient:
    client_class = (
        DryRunGuardedBinanceClient
        if guarded
        else BinanceFuturesClient
    )
    return client_class(
        api_key=read_secret(config.exchange.api_key_env),
        api_secret=read_secret(config.exchange.api_secret_env),
        environment="mainnet",
        recv_window=config.exchange.recv_window,
        timeout_seconds=config.exchange.timeout_seconds,
        request_weight_budget=_budget_for(config),
    )


def run_mainnet_dry_run_stress(
    *,
    dry_run_config_path: str | Path = DEFAULT_DRY_RUN_CONFIG,
    live_config_path: str | Path = DEFAULT_LIVE_CONFIG,
    duration_seconds: float = 45.0,
    stress_cycles: int = 200,
    require_user_stream: bool = True,
) -> dict[str, Any]:
    """Exercise the frozen strategy through WebSocket-backed mainnet data.

    The guarded client rejects every order/margin/leverage mutation.  Strategy
    observations are written only to a temporary ledger and are discarded
    after the report has been assembled.
    """

    started_at = datetime.utcnow()
    started_monotonic = time.monotonic()
    dry = load_live_config(dry_run_config_path)
    live_config = load_live_config(live_config_path)
    live = live_config.combined_breakout_v8_grid_v6_live
    client = _mainnet_client(live_config, guarded=True)
    stop_event = threading.Event()
    streams = BinanceFuturesStreamCache(
        client,
        tuple(live.enabled_symbols),
        stale_after_seconds=live.websocket_stale_seconds,
        listen_key_keepalive_seconds=(
            live.listen_key_keepalive_seconds
        ),
    )
    client.attach_market_stream_cache(streams)
    cycles = 0
    strategy_report: dict[str, Any] = {}
    cache_replay: dict[str, Any] = {
        "symbols_requested": 0,
        "symbols_returned": 0,
        "rest_requests_before": 0,
        "rest_requests_after": 0,
        "rest_request_delta": 0,
    }
    reconciliation_probe: dict[str, Any] = {
        "completed": False,
        "position_count": None,
        "operational_halt": None,
        "halt_reason": "",
        "account_call_delta": 0,
        "open_orders_call_delta": 0,
        "open_algo_orders_call_delta": 0,
    }
    final_stream_health: dict[str, Any] = {}
    ready = False
    failure = ""
    try:
        streams.start(
            stop_event,
            include_user_stream=require_user_stream,
        )
        ready = streams.wait_until_ready(
            live.websocket_startup_timeout_seconds,
            require_user_stream=require_user_stream,
        )
        if not ready:
            raise RuntimeError(
                f"WebSocket streams not ready: {streams.health()}"
            )
        with tempfile.TemporaryDirectory(
            prefix="b8g6-live-transport-acceptance-"
        ) as temporary:
            root = Path(temporary)
            combined_shadow = replace(
                dry.combined_volatility_trend_grid_shadow,
                state_path=str(root / "shadow_state.json"),
                event_log_path=str(root / "shadow_events.jsonl"),
                report_path=str(root / "shadow_report.json"),
            )
            test_config = replace(
                dry,
                combined_volatility_trend_grid_shadow=combined_shadow,
            )
            trader = CombinedBreakoutV8GridV6ShadowTrader(
                test_config, client
            )
            trader.validate_startup()
            deadline = (
                started_monotonic + max(1.0, duration_seconds)
            )
            while (
                cycles < max(1, stress_cycles)
                or time.monotonic() < deadline
            ):
                trader.run_once(stop_event=stop_event)
                cycles += 1
                if cycles >= stress_cycles:
                    stop_event.wait(0.05)
            strategy_report = trader.acceptance_report()
            # The first hourly strategy scan intentionally warms each 1h
            # series through REST.  Re-read that exact frozen-strategy input
            # through the normal client interface and prove the second pass is
            # served entirely by the WebSocket-backed cache.
            before_replay = client.rate_limit_status()["request_count"]
            replayed = 0
            for symbol in live.enabled_symbols:
                rows = client.klines(symbol, "1h", 500)
                if rows:
                    replayed += 1
            after_replay = client.rate_limit_status()["request_count"]
            cache_replay = {
                "symbols_requested": len(live.enabled_symbols),
                "symbols_returned": replayed,
                "rest_requests_before": before_replay,
                "rest_requests_after": after_replay,
                "rest_request_delta": after_replay - before_replay,
            }
            before_endpoints = dict(
                client.rate_limit_status()["endpoint_counts"]
            )
            acceptance_live = replace(
                live,
                enabled=False,
                armed=False,
                state_path=str(root / "live_probe_state.json"),
                event_log_path=str(root / "live_probe_events.jsonl"),
                report_path=str(root / "live_probe_report.json"),
                transport_acceptance_required=False,
            )
            live_probe_config = replace(
                live_config,
                combined_breakout_v8_grid_v6_live=acceptance_live,
            )
            live_probe = CombinedBreakoutV8GridV6LiveTrader(
                live_probe_config, client
            )
            live_probe.stream_cache = streams
            account = live_probe.reconcile()
            after_endpoints = dict(
                client.rate_limit_status()["endpoint_counts"]
            )

            def endpoint_delta(endpoint: str) -> int:
                return (
                    after_endpoints.get(endpoint, 0)
                    - before_endpoints.get(endpoint, 0)
                )

            reconciliation_probe = {
                "completed": True,
                "position_count": len(account.positions),
                "operational_halt": bool(
                    live_probe.state["operational_halt"]
                ),
                "halt_reason": str(
                    live_probe.state.get("halt_reason", "")
                ),
                "account_call_delta": endpoint_delta(
                    "GET /fapi/v2/account"
                ),
                "open_orders_call_delta": endpoint_delta(
                    "GET /fapi/v1/openOrders"
                ),
                "open_algo_orders_call_delta": endpoint_delta(
                    "GET /fapi/v1/openAlgoOrders"
                ),
            }
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        final_stream_health = streams.health()
        stop_event.set()
        streams.stop()
        client.attach_market_stream_cache(None)

    stream_health = final_stream_health or streams.health()
    rate_limit = client.rate_limit_status()
    forbidden_requests = dict(client.blocked_mutation_attempts)
    strategy_hashes = (
        strategy_report.get("source_config_hashes", {})
        if strategy_report
        else {}
    )
    expected_strategy_hashes = (
        CombinedBreakoutV8GridV6LiveTrader(
            live_config, client
        ).source_bundle["hashes"]
    )
    criteria = {
        "dry_run_config": dry.trading.dry_run is True,
        "guarded_order_surface": not forbidden_requests,
        "websocket_ready": ready,
        "market_stream_active": (
            stream_health["market_event_count"] > 0
            and stream_health.get(
                "market_subscription_acknowledged"
            )
            is True
        ),
        "user_stream_connected": (
            stream_health["user_connected"]
            if require_user_stream
            else True
        ),
        "broad_symbol_coverage": (
            stream_health["symbols_with_stream_data"]
            >= max(1, len(live.enabled_symbols) - 5)
        ),
        "cache_exercised": stream_health["cache_hits"] > 0,
        "cache_replay_no_rest": (
            cache_replay["symbols_returned"]
            >= max(1, len(live.enabled_symbols) - 5)
            and cache_replay["rest_request_delta"] == 0
        ),
        "stress_cycles_completed": cycles >= max(1, stress_cycles),
        "consolidated_reconciliation": (
            reconciliation_probe["completed"]
            and reconciliation_probe["account_call_delta"] == 1
            and reconciliation_probe["open_orders_call_delta"] == 1
            and reconciliation_probe[
                "open_algo_orders_call_delta"
            ]
            == 1
            and not reconciliation_probe["operational_halt"]
        ),
        "no_exchange_rate_limit": (
            rate_limit["rate_limit_count"] == 0
        ),
        "below_soft_request_weight": (
            rate_limit["effective_used_weight"]
            <= rate_limit["soft_limit"]
        ),
        "strategy_sources_unchanged": (
            strategy_hashes == expected_strategy_hashes
        ),
        "no_sample_integrity_errors": not strategy_report.get(
            "sample_integrity_errors", ["missing-report"]
        ),
    }
    completed_at = datetime.utcnow()
    report = {
        "schema_version": 2,
        "transport_version": (
            BREAKOUT_V8_GRID_V6_TRANSPORT_VERSION
        ),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (
            time.monotonic() - started_monotonic
        ),
        "mode": (
            "MAINNET_DRY_RUN_NO_ORDER_WITH_USER_STREAM"
            if require_user_stream
            else "MAINNET_DRY_RUN_PUBLIC_ONLY"
        ),
        "environment": "mainnet",
        "user_stream_required": require_user_stream,
        "passed": not failure and all(criteria.values()),
        "failure": failure,
        "criteria": criteria,
        "cycles": cycles,
        "live_config_hash": (
            combined_v8_grid_v6_live_config_hash(live_config)
        ),
        "strategy_source_hashes": expected_strategy_hashes,
        "transport_code_hashes": _transport_code_hashes(),
        "stream_health": stream_health,
        "cache_replay": cache_replay,
        "reconciliation_probe": reconciliation_probe,
        "rate_limit": rate_limit,
        "forbidden_requests": forbidden_requests,
        "strategy_observation": {
            "candidate_count_by_strategy": strategy_report.get(
                "candidate_count_by_strategy", {}
            ),
            "open_positions": strategy_report.get(
                "open_positions", 0
            ),
            "trade_count": strategy_report.get("trade_count", 0),
            "sample_integrity_errors": strategy_report.get(
                "sample_integrity_errors", []
            ),
        },
    }
    report_path = Path(live.transport_acceptance_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return report


def clear_verified_rate_limit_halt(
    live_config_path: str | Path = DEFAULT_LIVE_CONFIG,
) -> dict[str, Any]:
    config = load_live_config(live_config_path)
    client = _mainnet_client(config, guarded=False)
    trader = CombinedBreakoutV8GridV6LiveTrader(config, client)
    trader.validate_transport_acceptance()
    account = trader.snapshot_account()
    standard_orders = client.open_orders()
    algo_orders = client.open_algo_orders()
    trader.clear_rate_limit_halt_after_acceptance(
        account, standard_orders, algo_orders
    )
    return {
        "cleared": not trader.state["operational_halt"],
        "equity": account.equity,
        "position_count": len(account.positions),
        "standard_order_count": len(standard_orders),
        "algo_order_count": len(algo_orders),
        "state_path": str(trader.state_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Breakout v8/Grid v6 mainnet DRY-RUN transport acceptance"
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
    parser.add_argument(
        "--clear-rate-limit-halt", action="store_true"
    )
    args = parser.parse_args()
    if args.clear_rate_limit_halt:
        report = clear_verified_rate_limit_halt(args.live_config)
    else:
        report = run_mainnet_dry_run_stress(
            dry_run_config_path=args.dry_run_config,
            live_config_path=args.live_config,
            duration_seconds=args.duration_seconds,
            stress_cycles=args.cycles,
            require_user_stream=not args.public_only,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("passed", report.get("cleared")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
