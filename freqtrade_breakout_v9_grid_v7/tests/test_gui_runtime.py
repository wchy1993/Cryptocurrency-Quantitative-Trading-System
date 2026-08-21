from __future__ import annotations

import json
import os
import plistlib
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from freqtrade_gui import (
    APP_ICON_PATH,
    DRY_BUTTON,
    LIVE_BUTTON,
    ManualReconciliationError,
    REFRESH_BUTTON,
    START_BUTTON,
    STOP_BUTTON,
    TradingConsole,
    discover_external_manual_records,
    fitted_window_geometry,
    format_profit_u,
    read_ledger_realized_profit,
    strategy_profit_breakdown,
    synchronize_manual_exchange_changes,
)
from gui_runtime import (
    BACKTEST_CONFIG_PATH,
    BASE_CONFIG_PATH,
    BREAKOUT_OPEN_LIMIT,
    GRID_OPEN_LIMIT,
    MANIFEST_PATH,
    MODE_DRY,
    MODE_LIVE,
    STRATEGY_CLASS,
    STRATEGY_DEPENDENCIES,
    ApiCredentials,
    EngineOutputReducer,
    SecretBundle,
    acquire_pid_lock,
    active_pid_lock,
    analyze_position_reconciliation,
    append_manual_reconciliation_audit,
    archive_oversized_log,
    build_launch_spec,
    build_runtime_overlay,
    classify_component,
    clock_report_from_samples,
    extract_account_snapshot,
    friendly_runtime_error,
    has_open_nfp_trade,
    load_exchange_secrets,
    load_manual_reconciliation_markers,
    parse_dotenv,
    poll_backoff_seconds,
    redact_sensitive,
    secure_write_json,
    release_pid_lock,
    validate_position_reconciliation,
    validate_running_config,
    verify_release,
)


class LayoutTests(unittest.TestCase):
    @staticmethod
    def _white_contrast_ratio(hex_color: str) -> float:
        channels = []
        for offset in (1, 3, 5):
            channel = int(hex_color[offset : offset + 2], 16) / 255.0
            channels.append(
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
            )
        luminance = (
            0.2126 * channels[0]
            + 0.7152 * channels[1]
            + 0.0722 * channels[2]
        )
        return 1.05 / (luminance + 0.05)

    def test_laptop_window_fits_inside_scaled_display(self) -> None:
        width, height, x, y = fitted_window_geometry(1280, 800)
        self.assertEqual((width, height), (1160, 700))
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1280)
        self.assertLessEqual(y + height, 800)

    def test_position_profit_is_formatted_as_absolute_usdt(self) -> None:
        self.assertEqual(format_profit_u(12.345), "+12.35U")
        self.assertEqual(format_profit_u(-2.5), "-2.50U")
        self.assertEqual(format_profit_u(None), "+0.00U")

    def test_larger_display_does_not_create_an_oversized_console(self) -> None:
        width, height, _x, _y = fitted_window_geometry(2560, 1600)
        self.assertEqual((width, height), (1160, 700))

    def test_primary_controls_have_distinct_accessible_colors(self) -> None:
        colors = (
            DRY_BUTTON,
            LIVE_BUTTON,
            REFRESH_BUTTON,
            START_BUTTON,
            STOP_BUTTON,
        )
        self.assertEqual(len(set(colors)), len(colors))
        for color in colors:
            self.assertGreaterEqual(self._white_contrast_ratio(color), 4.5)

    def test_live_start_has_no_typed_confirmation(self) -> None:
        console = object.__new__(TradingConsole)
        console.root = None
        with patch(
            "freqtrade_gui.load_exchange_secrets",
            return_value=SecretBundle("key", "secret", "test"),
        ):
            self.assertTrue(console._check_live_start_prerequisites())

    def test_macos_app_bundle_has_native_coin_icon(self) -> None:
        project = BASE_CONFIG_PATH.parent
        bundle = project / "V16 Breakout MTF Max2 Trader.app"
        plist_path = bundle / "Contents" / "Info.plist"
        executable = (
            bundle / "Contents" / "MacOS" / "V16BreakoutMtfMax2Trader"
        )
        bundle_icon = bundle / "Contents" / "Resources" / "AppIcon.icns"
        payload = plistlib.loads(plist_path.read_bytes())
        self.assertEqual(
            payload["CFBundleIdentifier"],
            "local.breakout.v16.mtf.max2.trader",
        )
        self.assertEqual(payload["CFBundleIconFile"], "AppIcon.icns")
        self.assertTrue(APP_ICON_PATH.is_file())
        self.assertTrue(bundle_icon.is_file())
        self.assertGreater(bundle_icon.stat().st_size, 100_000)
        self.assertTrue(os.access(executable, os.X_OK))


class DotenvTests(unittest.TestCase):
    def test_parse_dotenv_supports_export_quotes_and_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "\n".join(
                    (
                        "# ignored",
                        "export BINANCE_FUTURES_API_KEY='abc123'",
                        'BINANCE_FUTURES_API_SECRET="xyz789"',
                        "PLAIN=value # comment",
                        "NOT AN ASSIGNMENT",
                    )
                ),
                encoding="utf-8",
            )
            values = parse_dotenv(path)
        self.assertEqual(values["BINANCE_FUTURES_API_KEY"], "abc123")
        self.assertEqual(values["BINANCE_FUTURES_API_SECRET"], "xyz789")
        self.assertEqual(values["PLAIN"], "value")

    def test_redaction_removes_exact_secrets_and_labeled_values(self) -> None:
        bundle = SecretBundle("my-key-value", "my-secret-value", "test")
        value = redact_sensitive(
            "api_key=other my-key-value secret=my-secret-value",
            bundle,
        )
        self.assertNotIn("my-key-value", value)
        self.assertNotIn("my-secret-value", value)
        self.assertNotIn("other", value)
        self.assertIn("[REDACTED]", value)

    def test_redaction_removes_signed_request_signature(self) -> None:
        value = redact_sensitive(
            "https://example.test/account?timestamp=1&signature=abcdef123456"
        )
        self.assertNotIn("abcdef123456", value)
        self.assertIn("signature=[REDACTED]", value)

    def test_original_dotenv_has_priority_over_shell_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "BINANCE_FUTURES_API_KEY=file-key\n"
                "BINANCE_FUTURES_API_SECRET=file-secret\n",
                encoding="utf-8",
            )
            with (
                patch("gui_runtime.dotenv_candidates", return_value=(path,)),
                patch.dict(
                    os.environ,
                    {
                        "BINANCE_FUTURES_API_KEY": "shell-key",
                        "BINANCE_FUTURES_API_SECRET": "shell-secret",
                    },
                ),
            ):
                bundle = load_exchange_secrets()
        self.assertEqual(bundle.key, "file-key")
        self.assertEqual(bundle.secret, "file-secret")
        self.assertEqual(bundle.source, str(path))


class ReleaseTests(unittest.TestCase):
    def test_current_release_manifest_matches_files(self) -> None:
        report = verify_release()
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.details["pair_count"], 50)
        self.assertEqual(report.details["max_open_trades"], 2)
        self.assertEqual(report.details["strategy"], STRATEGY_CLASS)
        self.assertEqual(
            report.details["dependency_sha256"].keys(),
            STRATEGY_DEPENDENCIES.keys(),
        )
        manifest = json.loads(MANIFEST_PATH.read_text())
        self.assertTrue(manifest["grid_enabled"])
        self.assertEqual(
            manifest["breakout_open_limit"],
            BREAKOUT_OPEN_LIMIT,
        )
        self.assertEqual(manifest["grid_open_limit"], GRID_OPEN_LIMIT)
        self.assertEqual(manifest["last_slot_priority"], "breakout")

    def test_gui_and_backtest_use_exact_same_active50_universe(self) -> None:
        gui = json.loads(BASE_CONFIG_PATH.read_text())
        backtest = json.loads(BACKTEST_CONFIG_PATH.read_text())
        gui_pairs = gui["exchange"]["pair_whitelist"]
        backtest_pairs = backtest["exchange"]["pair_whitelist"]
        self.assertEqual(gui_pairs, backtest_pairs)
        self.assertEqual(len(gui_pairs), 50)
        self.assertEqual(len(set(gui_pairs)), 50)
        self.assertNotIn("TON/USDT:USDT", gui_pairs)
        self.assertIn("SEI/USDT:USDT", gui_pairs)

    def test_runtime_overlay_starts_stopped_and_has_no_exchange_secret(self) -> None:
        credentials = ApiCredentials(
            username="local-user",
            password="local-password",
            jwt_secret="local-jwt-secret-key-000000000000000001",
            ws_token="local-ws",
        )
        for mode in (MODE_DRY, MODE_LIVE):
            overlay = build_runtime_overlay(mode, 18888, credentials)
            self.assertEqual(overlay["initial_state"], "stopped")
            self.assertFalse(overlay["force_entry_enable"])
            self.assertEqual(overlay["api_server"]["listen_ip_address"], "127.0.0.1")
            self.assertNotIn("exchange", overlay)
            self.assertEqual(overlay["dry_run"], mode == MODE_DRY)

    def test_runtime_overlay_applies_clock_guard_to_sync_and_async_ccxt(self) -> None:
        credentials = ApiCredentials(
            username="local-user",
            password="local-password",
            jwt_secret="local-jwt-secret-key-000000000000000001",
            ws_token="local-ws",
        )
        overlay = build_runtime_overlay(
            MODE_LIVE,
            18888,
            credentials,
            time_difference_ms=2012,
        )
        exchange = overlay["exchange"]
        self.assertEqual(exchange["name"], "binance")
        self.assertEqual(len(exchange["pair_whitelist"]), 50)
        for section in ("ccxt_config", "ccxt_async_config"):
            options = exchange[section]["options"]
            self.assertEqual(options["timeDifference"], 2012)
            self.assertFalse(options["adjustForTimeDifference"])
            self.assertFalse(
                options["setMarginMode"]["throwMarginModeAlreadySet"]
            )
        self.assertNotIn("key", exchange)
        self.assertNotIn("secret", exchange)

    def test_base_config_treats_existing_margin_mode_as_idempotent(self) -> None:
        config = json.loads(BASE_CONFIG_PATH.read_text())
        exchange = config["exchange"]
        for section in ("ccxt_config", "ccxt_async_config"):
            margin_options = exchange[section]["options"]["setMarginMode"]
            self.assertFalse(margin_options["throwMarginModeAlreadySet"])

    def test_ccxt_returns_binance_4046_as_success_with_release_option(self) -> None:
        import ccxt
        from ccxt.base.errors import MarginModeAlreadySet

        config = json.loads(BASE_CONFIG_PATH.read_text())
        exchange = ccxt.binanceusdm(config["exchange"]["ccxt_config"])
        exchange.markets = {}
        exchange.market = lambda _symbol: {
            "id": "BTCUSDT",
            "linear": True,
            "inverse": False,
        }

        def already_set(_request):
            raise MarginModeAlreadySet(
                "binanceusdm -4046 No need to change margin type."
            )

        exchange.fapiPrivatePostMarginType = already_set
        result = exchange.set_margin_mode("isolated", "BTC/USDT:USDT")
        self.assertEqual(result["code"], -4046)
        self.assertIn("No need", result["msg"])

    def test_nfp_overlay_is_rejected_and_base_remains_combined_max2(self) -> None:
        credentials = ApiCredentials(
            username="local-user",
            password="local-password",
            jwt_secret="local-jwt-secret-key-000000000000000001",
            ws_token="local-ws",
        )
        base = build_runtime_overlay(MODE_DRY, 18888, credentials)
        self.assertEqual(base["strategy"], STRATEGY_CLASS)
        self.assertEqual(base["max_open_trades"], 2)
        with self.assertRaisesRegex(ValueError, "仅允许运行"):
            build_runtime_overlay(
                MODE_DRY,
                18889,
                credentials,
                nfp_enabled=True,
            )

    def test_launch_specs_isolate_databases_and_never_put_keys_in_command(self) -> None:
        bundle = SecretBundle("key-material", "secret-material", "test")
        credentials = ApiCredentials(
            username="u",
            password="p",
            jwt_secret="test-jwt-secret-key-000000000000000001",
            ws_token="w",
        )
        with patch.dict(os.environ, {}, clear=False):
            dry = build_launch_spec(
                MODE_DRY,
                port=18881,
                credentials=credentials,
                bundle=bundle,
                session_token="dry-test",
                write_overlay=False,
            )
            live = build_launch_spec(
                MODE_LIVE,
                port=18882,
                credentials=credentials,
                bundle=bundle,
                session_token="live-test",
                write_overlay=False,
            )
        self.assertIn("--dry-run", dry.command)
        self.assertNotIn("--dry-run", live.command)
        self.assertNotEqual(dry.database_path, live.database_path)
        self.assertIn("breakout_v16_grid_v15_combined", dry.database_path.name)
        self.assertIn("breakout_v16_grid_v15_combined", live.database_path.name)
        self.assertNotIn("key-material", " ".join(live.command))
        self.assertNotIn("secret-material", " ".join(live.command))
        self.assertNotIn("FREQTRADE__EXCHANGE__KEY", dry.environment)
        self.assertEqual(
            live.environment["FREQTRADE__EXCHANGE__KEY"],
            "key-material",
        )
        self.assertFalse(dry.nfp_enabled)
        self.assertEqual(dry.strategy_class, STRATEGY_CLASS)
        self.assertEqual(dry.max_open_trades, 2)

    def test_runtime_overlay_file_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            secure_write_json(path, {"dry_run": True})
            permissions = stat.S_IMODE(path.stat().st_mode)
        self.assertEqual(permissions, 0o600)

    def test_pid_lock_blocks_duplicate_and_cleans_stale_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.lock"
            acquire_pid_lock(
                path,
                pid=os.getpid(),
                kind="test engine",
            )
            self.assertEqual(active_pid_lock(path)["pid"], os.getpid())
            with self.assertRaises(RuntimeError):
                acquire_pid_lock(
                    path,
                    pid=os.getpid(),
                    kind="duplicate engine",
                )
            self.assertTrue(release_pid_lock(path, pid=os.getpid()))
            acquire_pid_lock(
                path,
                pid=999_999_999,
                kind="stale engine",
            )
            self.assertIsNone(active_pid_lock(path))
            self.assertFalse(path.exists())


class RuntimeValidationTests(unittest.TestCase):
    def test_clock_guard_uses_lowest_latency_sample_and_safety_lag(self) -> None:
        report = clock_report_from_samples(
            (
                (380.0, -170.0),
                (33.0, 6.2),
                (35.0, 5.8),
            )
        )
        self.assertEqual(report.local_minus_server_ms, 6)
        self.assertEqual(report.time_difference_ms, 2006)
        self.assertEqual(report.round_trip_ms, 33.0)
        self.assertEqual(report.sample_count, 3)

    def test_poll_backoff_is_bounded(self) -> None:
        self.assertEqual(poll_backoff_seconds(1), 20.0)
        self.assertEqual(poll_backoff_seconds(3), 80.0)
        self.assertEqual(poll_backoff_seconds(20), 300.0)

    def test_friendly_runtime_error_compacts_nonce_failure(self) -> None:
        message = friendly_runtime_error(
            'Freqtrade API 500: InvalidNonce {"code":-1021}'
        )
        self.assertIn("-1021", message)
        self.assertNotIn("Freqtrade API 500", message)

    def test_engine_output_reducer_suppresses_nonce_traceback_storm(self) -> None:
        reducer = EngineOutputReducer(repeat_window_seconds=60)
        first = reducer.reduce(
            '2026-07-30 08:56:57 - WARNING - InvalidNonce {"code":-1021}',
            now=1.0,
        )
        repeated = reducer.reduce(
            'ccxt.base.errors.InvalidNonce {"code":-1021}',
            now=2.0,
        )
        traceback = reducer.reduce(
            "Traceback (most recent call last):",
            now=3.0,
        )
        stack_line = reducer.reduce(
            '  File "exchange.py", line 1, in fetch',
            now=4.0,
        )
        heartbeat = reducer.reduce(
            "2026-07-30 08:58:03,100 - freqtrade.worker - "
            "INFO - Bot heartbeat. state='RUNNING'",
            now=5.0,
        )
        self.assertEqual(first[1], "clock_error")
        self.assertIsNone(repeated)
        self.assertIsNone(traceback)
        self.assertIsNone(stack_line)
        self.assertEqual(heartbeat[1], "engine")

    def test_oversized_gui_log_is_archived_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.log"
            payload = b"diagnostic-history"
            path.write_bytes(payload)
            archived = archive_oversized_log(path, maximum_bytes=4)
            self.assertIsNotNone(archived)
            self.assertFalse(path.exists())
            self.assertEqual(archived.read_bytes(), payload)

    def test_running_config_must_match_mode_and_frozen_limits(self) -> None:
        valid = {
            "strategy": STRATEGY_CLASS,
            "dry_run": True,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "max_open_trades": 2,
            "force_entry_enable": False,
            "state": "stopped",
        }
        self.assertEqual(validate_running_config(valid, MODE_DRY), ())
        invalid = dict(valid, dry_run=False, max_open_trades=3)
        errors = validate_running_config(invalid, MODE_DRY)
        self.assertEqual(len(errors), 2)
        nfp_errors = validate_running_config(
            valid,
            MODE_DRY,
            nfp_enabled=True,
        )
        self.assertTrue(any("仅允许运行" in error for error in nfp_errors))

    def test_account_snapshot_detects_unmanaged_exchange_positions(self) -> None:
        snapshot = extract_account_snapshot(
            {
                "total": 289.52,
                "stake": "USDT",
                "currencies": [
                    {
                        "currency": "USDT",
                        "free": 250.0,
                        "is_position": False,
                    },
                    {
                        "currency": "PENGU/USDT:USDT",
                        "position": 100.0,
                        "is_position": True,
                        "is_bot_managed": False,
                    },
                ],
            }
        )
        self.assertEqual(snapshot["equity"], 289.52)
        self.assertEqual(snapshot["available"], 250.0)
        self.assertEqual(len(snapshot["unmanaged_positions"]), 1)

    def test_strategy_profit_is_closed_history_plus_open_position_pnl(self) -> None:
        historical, position, total = strategy_profit_breakdown(
            {
                "profit_closed_coin": 17.35273614,
                "profit_all_coin": 20.85273614,
            },
            [{"profit_abs": 999.0}],
        )
        self.assertAlmostEqual(historical, 17.35273614)
        self.assertAlmostEqual(position, 3.5)
        self.assertAlmostEqual(total, 20.85273614)

        fallback = strategy_profit_breakdown(
            {"profit_closed_coin": 17.35273614},
            [{"profit_abs": -2.0}, {"profit_abs": 0.75}],
        )
        self.assertEqual(fallback, (17.35273614, -1.25, 16.10273614))

    def test_readonly_ledger_profit_includes_closed_and_partial_realized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TABLE trades ("
                    "is_open BOOLEAN NOT NULL, "
                    "close_profit_abs FLOAT, "
                    "realized_profit FLOAT)"
                )
                connection.executemany(
                    "INSERT INTO trades VALUES (?, ?, ?)",
                    (
                        (0, 10.0, 10.0),
                        (0, -3.0, -3.0),
                        (1, None, 1.25),
                    ),
                )
            before = path.stat().st_mtime_ns
            realized = read_ledger_realized_profit(path)
            after = path.stat().st_mtime_ns
        self.assertEqual(realized, 8.25)
        self.assertEqual(after, before)

    def test_component_labels(self) -> None:
        self.assertEqual(
            classify_component("bo_v9_s4"),
            "Breakout V16 MTF",
        )
        self.assertEqual(classify_component("grid_v8_long_s6"), "Grid V15 PF")
        self.assertEqual(classify_component("grid_v8_short_s4"), "Grid V15 PF")
        self.assertEqual(classify_component("grid_v7_s3"), "Grid v7（旧）")
        self.assertEqual(
            classify_component("nfp_v4_20260807t123000z"),
            "非农 Stability v4（旧）",
        )
        self.assertEqual(classify_component(None), "未知")
        self.assertTrue(
            has_open_nfp_trade(
                [{"enter_tag": "nfp_v4_20260807t123000z"}]
            )
        )
        self.assertFalse(has_open_nfp_trade([{"enter_tag": "bo_v9_s4"}]))

    def test_reconciliation_rejects_stale_or_direction_mismatch(self) -> None:
        stale = validate_position_reconciliation(
            [
                {
                    "pair": "PENGU/USDT:USDT",
                    "is_short": False,
                    "has_open_orders": False,
                }
            ],
            [],
        )
        self.assertEqual(len(stale), 1)
        mismatch = validate_position_reconciliation(
            [
                {
                    "pair": "PENGU/USDT:USDT",
                    "is_short": True,
                    "has_open_orders": False,
                }
            ],
            [
                {
                    "currency": "PENGU/USDT:USDT",
                    "side": "long",
                }
            ],
        )
        self.assertEqual(len(mismatch), 1)
        pending = validate_position_reconciliation(
            [
                {
                    "pair": "PENGU/USDT:USDT",
                    "is_short": False,
                    "has_open_orders": True,
                }
            ],
            [],
        )
        self.assertEqual(pending, ())

    def test_reconciliation_classifies_manual_full_and_partial_closes(self) -> None:
        full = analyze_position_reconciliation(
            [
                {
                    "trade_id": 5,
                    "pair": "INJ/USDT:USDT",
                    "amount": 525.3,
                    "is_short": True,
                    "has_open_orders": False,
                }
            ],
            [],
        )
        self.assertEqual(full.errors, ())
        self.assertEqual(len(full.manual_changes), 1)
        self.assertEqual(full.manual_changes[0].kind, "full_close")

        partial = analyze_position_reconciliation(
            [
                {
                    "trade_id": 6,
                    "pair": "HBAR/USDT:USDT",
                    "amount": 3759.0,
                    "is_short": True,
                    "has_open_orders": False,
                }
            ],
            [
                {
                    "currency": "HBAR/USDT:USDT",
                    "side": "short",
                    "position": 1200.0,
                }
            ],
        )
        self.assertEqual(partial.errors, ())
        self.assertEqual(partial.manual_changes[0].kind, "partial_close")
        self.assertEqual(partial.manual_changes[0].exchange_amount, 1200.0)

    def test_reconciliation_never_auto_adopts_manual_additions(self) -> None:
        report = analyze_position_reconciliation(
            [
                {
                    "trade_id": 8,
                    "pair": "BTC/USDT:USDT",
                    "amount": 0.01,
                    "is_short": False,
                    "has_open_orders": False,
                }
            ],
            [
                {
                    "currency": "BTC/USDT:USDT",
                    "side": "long",
                    "position": 0.02,
                }
            ],
        )
        self.assertEqual(report.manual_changes, ())
        self.assertEqual(len(report.errors), 1)
        self.assertIn("禁止自动接管", report.errors[0])

    def test_manual_reconciliation_audit_is_owner_only_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.jsonl"
            count = append_manual_reconciliation_audit(
                [{"trade_id": 5, "pair": "INJ/USDT:USDT"}],
                path=path,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            mode = stat.S_IMODE(path.stat().st_mode)
        self.assertEqual(count, 1)
        self.assertEqual(payload["trade_id"], 5)
        self.assertEqual(payload["source"], "freqtrade_reload_trade_from_exchange")
        self.assertEqual(mode, 0o600)

    def test_manual_reconciliation_markers_skip_corrupt_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.jsonl"
            path.write_text(
                "not-json\n"
                + json.dumps(
                    {
                        "trade_id": 5,
                        "exchange_orders": [{"order_id": "exit-5"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            markers = load_manual_reconciliation_markers(path=path)
        self.assertEqual(markers.trade_ids, frozenset({5}))
        self.assertEqual(markers.order_ids, frozenset({"exit-5"}))


class ManualExchangeSynchronizationTests(unittest.TestCase):
    class FakeApi:
        def __init__(self, *, runtime: bool = False) -> None:
            self.status = [
                {
                    "trade_id": 5,
                    "pair": "INJ/USDT:USDT",
                    "amount": 525.3,
                    "is_short": True,
                    "has_open_orders": False,
                    "enter_tag": "bo_v9_s5",
                    "orders": [{"order_id": "entry-1"}],
                }
            ]
            self.balance = {
                "total": 300.0,
                "stake": "USDT",
                "currencies": [
                    {
                        "currency": "USDT",
                        "free": 300.0,
                        "is_position": False,
                    }
                ],
            }
            self.state = "running" if runtime else "stopped"
            self.calls: list[tuple[str, str]] = []

        def get(self, path: str):
            self.calls.append(("GET", path))
            if path == "/status":
                return self.status
            if path == "/balance":
                return self.balance
            if path == "/profit":
                return {"closed_trade_count": 1}
            if path == "/show_config":
                return {"state": self.state}
            raise AssertionError(path)

        def post(self, path: str):
            self.calls.append(("POST", path))
            if path == "/pause":
                self.state = "paused"
                return {"status": "paused"}
            if path == "/start":
                self.state = "running"
                return {"status": "running"}
            if path == "/trades/5/reload":
                self.status = []
                return {
                    "trade_id": 5,
                    "pair": "INJ/USDT:USDT",
                    "is_open": False,
                    "amount": 525.3,
                    "close_rate": 4.57,
                    "close_date": "2026-08-07T00:05:41Z",
                    "close_profit_abs": 4.0,
                    "realized_profit": 4.0,
                    "exit_reason": "sold_on_exchange",
                    "orders": [
                        {"order_id": "entry-1"},
                        {
                            "order_id": "manual-exit-1",
                            "ft_order_side": "buy",
                            "status": "closed",
                            "filled": 525.3,
                            "average": 4.57,
                        },
                    ],
                }
            raise AssertionError(path)

    def test_startup_reloads_actual_manual_exit_and_records_it(self) -> None:
        api = self.FakeApi()
        with patch("freqtrade_gui.append_manual_reconciliation_audit") as audit:
            result = synchronize_manual_exchange_changes(
                api,
                api.status,
                api.balance,
                runtime_running=False,
            )
        self.assertEqual(result["status"], [])
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(
            result["records"][0]["exchange_orders"][0]["order_id"],
            "manual-exit-1",
        )
        self.assertIn(("POST", "/trades/5/reload"), api.calls)
        self.assertNotIn(("POST", "/start"), api.calls)
        audit.assert_called_once()

    def test_runtime_pauses_before_reload_and_resumes_only_after_clean_check(self) -> None:
        api = self.FakeApi(runtime=True)
        with patch("freqtrade_gui.append_manual_reconciliation_audit"):
            result = synchronize_manual_exchange_changes(
                api,
                api.status,
                api.balance,
                runtime_running=True,
            )
        pause_index = api.calls.index(("POST", "/pause"))
        reload_index = api.calls.index(("POST", "/trades/5/reload"))
        start_index = api.calls.index(("POST", "/start"))
        self.assertLess(pause_index, reload_index)
        self.assertLess(reload_index, start_index)
        self.assertEqual(api.state, "running")
        self.assertFalse(result["paused"])

    def test_runtime_direction_mismatch_stays_paused_and_is_never_imported(self) -> None:
        api = self.FakeApi(runtime=True)
        api.balance["currencies"].append(
            {
                "currency": "INJ/USDT:USDT",
                "position": 525.3,
                "side": "long",
                "is_position": True,
                "is_bot_managed": True,
            }
        )
        with self.assertRaises(ManualReconciliationError) as context:
            synchronize_manual_exchange_changes(
                api,
                api.status,
                api.balance,
                runtime_running=True,
            )
        self.assertTrue(context.exception.trading_paused)
        self.assertEqual(api.state, "paused")
        self.assertNotIn(("POST", "/trades/5/reload"), api.calls)
        self.assertNotIn(("POST", "/start"), api.calls)

    def test_history_scanner_records_auto_imported_partial_manual_close(self) -> None:
        open_trade = {
            "trade_id": 9,
            "pair": "HBAR/USDT:USDT",
            "is_open": True,
            "is_short": True,
            "amount": 1000.0,
            "realized_profit": 2.5,
            "enter_tag": "grid_v8_short_s5",
            "orders": [
                {
                    "order_id": "entry-9",
                    "ft_order_side": "sell",
                    "ft_order_tag": "grid_v8_short_s5",
                    "filled": 1500.0,
                    "safe_price": 0.07,
                    "status": "closed",
                    "is_open": False,
                },
                {
                    "order_id": "manual-partial-9",
                    "ft_order_side": "buy",
                    "ft_order_tag": None,
                    "filled": 500.0,
                    "safe_price": 0.068,
                    "order_filled_timestamp": 1786061163881,
                    "status": "closed",
                    "is_open": False,
                },
            ],
        }
        records = discover_external_manual_records(
            [open_trade],
            [],
            known_trade_ids=set(),
            known_order_ids=set(),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["change"], "partial_close")
        self.assertEqual(records[0]["managed_amount_before"], 1500.0)
        self.assertEqual(records[0]["managed_amount_after"], 1000.0)
        self.assertEqual(records[0]["close_rate"], 0.068)

    def test_history_scanner_ignores_strategy_exit_and_known_manual_order(self) -> None:
        strategy_trade = {
            "trade_id": 10,
            "pair": "LINK/USDT:USDT",
            "is_open": False,
            "is_short": True,
            "amount": 20.0,
            "exit_reason": "grid_tp_0",
            "orders": [
                {
                    "order_id": "strategy-exit-10",
                    "ft_order_side": "buy",
                    "ft_order_tag": "grid_tp_0",
                    "filled": 20.0,
                    "status": "closed",
                    "is_open": False,
                }
            ],
        }
        self.assertEqual(
            discover_external_manual_records(
                [],
                [strategy_trade],
                known_trade_ids=set(),
                known_order_ids=set(),
            ),
            [],
        )
        manual_trade = dict(
            strategy_trade,
            trade_id=11,
            exit_reason="sold_on_exchange",
            orders=[
                {
                    "order_id": "known-manual-11",
                    "ft_order_side": "buy",
                    "ft_order_tag": None,
                    "filled": 20.0,
                    "status": "closed",
                    "is_open": False,
                }
            ],
        )
        self.assertEqual(
            discover_external_manual_records(
                [],
                [manual_trade],
                known_trade_ids={11},
                known_order_ids={"known-manual-11"},
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
