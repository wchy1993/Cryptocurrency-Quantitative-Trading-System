#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from gui_runtime import (  # noqa: E402
    BASE_CONFIG_PATH,
    DEFAULT_DRY_WALLET,
    FREQTRADE_BIN,
    KEY_NAMES,
    MODE_DRY,
    SECRET_NAMES,
    STRATEGY_CLASS,
    USER_DATA_DIR,
    ApiCredentials,
    FreqtradeApiClient,
    build_runtime_overlay,
    find_free_local_port,
    measure_binance_clock,
    secure_write_json,
    sqlite_url,
    validate_running_config,
    verify_release,
)
from user_data.strategies.BreakoutV10FGridV8DualSideFreqtrade import (  # noqa: E402
    LIVE_CONTEXT_COLUMNS,
)


def verify_latest_live_context(
    api: FreqtradeApiClient,
    pairs: list[str],
) -> tuple[str, int, int]:
    latest_dates: set[str] = set()
    candidate_count = 0
    v16_path_ready_count = 0
    for pair in pairs:
        query = urllib.parse.urlencode(
            {
                "pair": pair,
                "timeframe": "1h",
                "limit": 2,
            }
        )
        payload = api.get(f"/pair_candles?{query}")
        columns = payload.get("columns") or []
        data = payload.get("data") or []
        if not data:
            raise RuntimeError(f"{pair} 没有已分析K线")
        latest = dict(zip(columns, data[-1], strict=False))
        missing = [
            column
            for column in LIVE_CONTEXT_COLUMNS
            if latest.get(column) is None
        ]
        if missing:
            raise RuntimeError(
                f"{pair} 最新K线缺少市场环境字段：{','.join(missing)}"
            )
        latest_dates.add(str(latest.get("date") or ""))
        candidate_count += int(latest.get("enter_long") or 0)
        candidate_count += int(latest.get("enter_short") or 0)
        if int(latest.get("v16_mtf_ready") or 0) != 1:
            raise RuntimeError(f"{pair} 最新K线缺少完整 V16 四段15分钟路径")
        v16_path_ready_count += 1
    if len(latest_dates) != 1:
        raise RuntimeError(
            f"50币最新分析时间不一致：{sorted(latest_dates)}"
        )
    return next(iter(latest_dates)), candidate_count, v16_path_ready_count


def main() -> int:
    report = verify_release()
    if not report.ok:
        print("release-check=FAILED")
        for error in report.errors:
            print(f"error={error}")
        return 1

    with tempfile.TemporaryDirectory(
        prefix="breakout_v16_grid_v15_gui_acceptance_"
    ) as directory:
        temp_dir = Path(directory)
        credentials = ApiCredentials.create()
        port = find_free_local_port()
        overlay_path = temp_dir / "dry-acceptance.json"
        database_path = temp_dir / "dry-acceptance.sqlite"
        logfile_path = temp_dir / "dry-acceptance.log"
        clock_sync = measure_binance_clock()
        secure_write_json(
            overlay_path,
            build_runtime_overlay(
                MODE_DRY,
                port,
                credentials,
                time_difference_ms=clock_sync.time_difference_ms,
            ),
        )
        command = [
            str(FREQTRADE_BIN),
            "trade",
            "--no-color",
            "--config",
            str(BASE_CONFIG_PATH),
            "--config",
            str(overlay_path),
            "--userdir",
            str(USER_DATA_DIR),
            "--strategy",
            STRATEGY_CLASS,
            "--db-url",
            sqlite_url(database_path),
            "--logfile",
            str(logfile_path),
            "--dry-run",
            "--dry-run-wallet",
            f"{DEFAULT_DRY_WALLET:.2f}",
        ]
        environment = dict(os.environ)
        for name in (*KEY_NAMES, *SECRET_NAMES):
            environment.pop(name, None)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["NO_COLOR"] = "1"

        output_path = temp_dir / "stdout.log"
        with output_path.open("w+", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
            )
            api = FreqtradeApiClient(port, credentials)
            try:
                deadline = time.monotonic() + 90.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Freqtrade exited with {process.returncode}"
                        )
                    if api.ping(timeout=1.0):
                        break
                    time.sleep(0.5)
                else:
                    raise RuntimeError("Timed out waiting for local API")

                config = api.get("/show_config")
                errors = validate_running_config(config, MODE_DRY)
                if errors:
                    raise RuntimeError("; ".join(errors))
                balance = api.get("/balance")
                status = api.get("/status")
                if status:
                    raise RuntimeError("Stopped preflight unexpectedly has trades")
                start_result = api.post("/start")
                if "start" not in str(start_result).lower():
                    raise RuntimeError(f"Unexpected start response: {start_result}")
                running_deadline = time.monotonic() + 10.0
                while time.monotonic() < running_deadline:
                    running_config = api.get("/show_config")
                    if str(running_config.get("state")).lower() == "running":
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Bot did not enter running state")

                context_deadline = time.monotonic() + 120.0
                latest_date = ""
                candidate_count = 0
                v16_path_ready_count = 0
                pairs = json.loads(
                    BASE_CONFIG_PATH.read_text(encoding="utf-8")
                )["exchange"]["pair_whitelist"]
                last_context_error: Exception | None = None
                while time.monotonic() < context_deadline:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Freqtrade exited with {process.returncode}"
                        )
                    try:
                        latest_date, candidate_count, v16_path_ready_count = (
                            verify_latest_live_context(api, pairs)
                        )
                        break
                    except Exception as exc:
                        last_context_error = exc
                        time.sleep(2.0)
                else:
                    raise RuntimeError(
                        "最新K线市场环境同步超时："
                        f"{last_context_error}"
                    )

                stop_result = api.post("/stop")
                if "stop" not in str(stop_result).lower():
                    raise RuntimeError(f"Unexpected stop response: {stop_result}")
                print("dry-startup=OK")
                print(f"strategy={config.get('strategy')}")
                print(f"state={config.get('state')}")
                print(f"dry_run={config.get('dry_run')}")
                print(f"max_open_trades={config.get('max_open_trades')}")
                print(f"equity={float(balance.get('total') or 0.0):.2f}")
                print(
                    "clock_offset_ms="
                    f"{clock_sync.local_minus_server_ms:+d}"
                )
                print(f"latest_context_date={latest_date}")
                print(f"latest_context_pairs={len(pairs)}/{len(pairs)}")
                print(
                    "latest_v16_path_pairs="
                    f"{v16_path_ready_count}/{len(pairs)}"
                )
                print(f"latest_candidates={candidate_count}")
                print("latest_context_fields=OK")
                print("api_start_stop=OK")
                print("real_orders_submitted=0")
                return 0
            except Exception as exc:
                output.flush()
                output.seek(0)
                tail = output.read()[-4000:]
                print(f"dry-startup=FAILED: {exc}")
                if tail:
                    print(tail)
                return 1
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
