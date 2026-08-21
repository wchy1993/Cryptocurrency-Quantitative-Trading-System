from __future__ import annotations

import argparse
import math
import os
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from gui_runtime import (
    DEFAULT_DRY_WALLET,
    ENGINE_LOCK_PATH,
    GUI_LOCK_PATH,
    LOG_DIR,
    MAX_OPEN_TRADES,
    MODE_DRY,
    MODE_LIVE,
    PROJECT_DIR,
    RELEASE_LABEL,
    STRATEGY_CLASS,
    EngineOutputReducer,
    FreqtradeApiClient,
    LaunchSpec,
    acquire_pid_lock,
    active_pid_lock,
    analyze_position_reconciliation,
    append_manual_reconciliation_audit,
    archive_oversized_log,
    build_launch_spec,
    classify_component,
    compact_command,
    extract_account_snapshot,
    friendly_runtime_error,
    load_exchange_secrets,
    load_manual_reconciliation_markers,
    measure_binance_clock,
    poll_backoff_seconds,
    redact_sensitive,
    release_pid_lock,
    run_static_check,
    validate_position_reconciliation,
    validate_running_config,
    verify_release,
)


BG = "#070B14"
PANEL = "#0E1626"
CARD = "#131E31"
CARD_ALT = "#223451"
BORDER = "#3B506F"
TEXT = "#F4F7FC"
MUTED = "#C2CEE0"
FAINT = "#91A3BD"
BLUE = "#3478E5"
BLUE_HOVER = "#5593F0"
GREEN = "#22C58B"
GREEN_DARK = "#123D35"
RED = "#FF5D6C"
RED_DARK = "#672A3A"
AMBER = "#F7B955"
AMBER_DARK = "#3B3020"
PURPLE = "#A78BFA"
WHITE = "#FFFFFF"
BUTTON_BG = "#294263"
BUTTON_HOVER = "#36577F"
DISABLED_TEXT = "#B2C0D3"
DRY_BUTTON = "#1C64D1"
DRY_BUTTON_HOVER = "#3478E5"
LIVE_BUTTON = "#C2410C"
LIVE_BUTTON_HOVER = "#EA580C"
REFRESH_BUTTON = "#6741D9"
REFRESH_BUTTON_HOVER = "#7C5CE3"
START_BUTTON = "#087F5B"
START_BUTTON_HOVER = "#0B986D"
STOP_BUTTON = "#C92A3E"
STOP_BUTTON_HOVER = "#E33F55"
MODE_IDLE_BUTTON = "#2B3B52"
APP_ICON_PATH = PROJECT_DIR / "assets" / "coin_app_icon_256.png"
LIVE_LEDGER_PATH = (
    PROJECT_DIR
    / "user_data"
    / "tradesv3.breakout_v16_grid_v15_combined.live.sqlite"
)

FONT_FAMILY = "SF Pro Display"
MONO_FAMILY = "SF Mono"

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class LaunchCancelled(RuntimeError):
    pass


class ManualReconciliationError(RuntimeError):
    """A manual-position difference could not be proven and safely reconciled."""

    def __init__(self, message: str, *, trading_paused: bool = False) -> None:
        super().__init__(message)
        self.trading_paused = bool(trading_paused)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def format_profit_u(value: Any) -> str:
    """Format an absolute position profit in the GUI stake currency."""

    return f"{_finite_float(value):+,.2f}U"


def strategy_profit_breakdown(
    profit: dict[str, Any],
    open_trades: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Return historical, open-position and combined strategy PnL.

    Freqtrade's ``profit_all_coin`` is authoritative because it includes every
    closed trade plus the current total profit of open trades.  The explicit
    fallback keeps the GUI correct with older or reduced API payloads.
    """

    historical = _finite_float(profit.get("profit_closed_coin"))
    if "profit_all_coin" in profit:
        total = _finite_float(profit.get("profit_all_coin"), historical)
        position = total - historical
        return historical, position, total

    position = sum(
        _finite_float(trade.get("profit_abs"))
        for trade in open_trades
    )
    return historical, position, historical + position


def read_ledger_realized_profit(path: Path = LIVE_LEDGER_PATH) -> float:
    """Read all realized strategy PnL without creating or mutating a ledger."""

    if not path.is_file():
        return 0.0
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2.0) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN is_open = 0 THEN COALESCE(close_profit_abs, 0.0)
                    ELSE COALESCE(realized_profit, 0.0)
                END
            ), 0.0)
            FROM trades
            """
        ).fetchone()
    return _finite_float(row[0] if row else 0.0)


def _order_id(order: dict[str, Any]) -> str:
    return str(order.get("order_id") or order.get("id") or "")


def _manual_reconciliation_record(
    change: Any,
    before_trade: dict[str, Any],
    reloaded_trade: dict[str, Any],
) -> dict[str, Any]:
    before_order_ids = {
        _order_id(order)
        for order in before_trade.get("orders") or []
        if _order_id(order)
    }
    new_orders: list[dict[str, Any]] = []
    for order in reloaded_trade.get("orders") or []:
        order_id = _order_id(order)
        if not order_id or order_id in before_order_ids:
            continue
        new_orders.append(
            {
                "order_id": order_id,
                "side": order.get("ft_order_side") or order.get("side"),
                "status": order.get("status"),
                "filled": order.get("filled"),
                "average": order.get("average") or order.get("safe_price"),
                "filled_at": order.get("order_filled_date"),
            }
        )
    after_amount = (
        abs(float(reloaded_trade.get("amount") or 0.0))
        if bool(reloaded_trade.get("is_open"))
        else 0.0
    )
    return {
        "trade_id": change.trade_id,
        "pair": change.pair,
        "component": classify_component(before_trade.get("enter_tag")),
        "change": change.kind,
        "side": change.expected_side,
        "managed_amount_before": change.managed_amount,
        "exchange_amount_detected": change.exchange_amount,
        "managed_amount_after": after_amount,
        "is_open_after": bool(reloaded_trade.get("is_open")),
        "close_rate": reloaded_trade.get("close_rate"),
        "close_date": reloaded_trade.get("close_date"),
        "realized_profit": reloaded_trade.get("realized_profit"),
        "close_profit_abs": reloaded_trade.get("close_profit_abs"),
        "exit_reason": reloaded_trade.get("exit_reason"),
        "exchange_orders": new_orders,
        "observation": "active_reconciliation",
    }


def _manual_reconciliation_message(record: dict[str, Any]) -> str:
    action = "手动全平" if record.get("change") == "full_close" else "手动部分平仓"
    prefix = (
        "扫仓补录"
        if record.get("observation") == "freqtrade_history_scan"
        else "已同步"
    )
    order_ids = ",".join(
        str(order.get("order_id"))
        for order in record.get("exchange_orders") or []
        if order.get("order_id")
    ) or "已由 Freqtrade 核验"
    close_rate = record.get("close_rate")
    price_text = f"{float(close_rate):g}" if close_rate is not None else "分批成交"
    profit = record.get("close_profit_abs")
    profit_text = f"{float(profit):+,.4f}U" if profit is not None else "待最终结算"
    close_date = str(record.get("close_date") or "")
    time_text = f"，成交时间={close_date}" if close_date else ""
    return (
        f"{prefix}{action}：{record.get('pair')} · {record.get('component')}，"
        f"数量 {float(record.get('managed_amount_before') or 0):g} → "
        f"{float(record.get('managed_amount_after') or 0):g}，"
        f"成交价={price_text}，已实现={profit_text}，订单={order_ids}"
        f"{time_text}。"
    )


def discover_external_manual_records(
    managed_trades: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    *,
    known_trade_ids: set[int],
    known_order_ids: set[str],
) -> list[dict[str, Any]]:
    """Find manual fills Freqtrade imported before the GUI's 10-second poll.

    Freqtrade tags strategy-created orders, while an order imported from the
    exchange has no Freqtrade order tag.  For closed trades, the standard
    ``sold_on_exchange`` reason is also required.  This lets the GUI observe
    both full and partial manual closes without increasing Binance polling.
    """

    by_id: dict[int, dict[str, Any]] = {}
    for trade in [*closed_trades, *managed_trades]:
        try:
            trade_id = int(trade.get("trade_id") or trade.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if trade_id > 0:
            by_id[trade_id] = trade

    records: list[dict[str, Any]] = []
    external_reasons = {"sold_on_exchange", "manual_exchange_close"}
    explicit_manual_tags = {"sold_on_exchange", "manual_exchange_close"}
    for trade_id, trade in sorted(by_id.items()):
        is_open = bool(trade.get("is_open"))
        is_short = bool(trade.get("is_short"))
        exit_side = "buy" if is_short else "sell"
        exit_reason = str(trade.get("exit_reason") or "").lower()
        candidates: list[dict[str, Any]] = []
        for order in trade.get("orders") or []:
            order_id = _order_id(order)
            side = str(order.get("ft_order_side") or order.get("side") or "").lower()
            tag = str(order.get("ft_order_tag") or "").lower()
            try:
                filled = float(order.get("filled") or order.get("amount") or 0.0)
            except (TypeError, ValueError):
                continue
            if (
                not order_id
                or order_id in known_order_ids
                or side != exit_side
                or filled <= 0
                or bool(order.get("is_open"))
            ):
                continue
            if tag in explicit_manual_tags or (
                not tag and (is_open or exit_reason in external_reasons)
            ):
                candidates.append(order)

        if not candidates:
            if not (
                not is_open
                and exit_reason in external_reasons
                and trade_id not in known_trade_ids
            ):
                continue
            # Backfill an older standard external close even if its historical
            # order tag was normalized by a previous recovery version.
            candidates = [
                order
                for order in trade.get("orders") or []
                if str(order.get("ft_order_side") or "").lower() == exit_side
            ][-1:]

        exchange_orders: list[dict[str, Any]] = []
        execution_amount = 0.0
        weighted_value = 0.0
        latest_timestamp = 0
        for order in candidates:
            order_id = _order_id(order)
            try:
                filled = float(order.get("filled") or order.get("amount") or 0.0)
                price = float(order.get("safe_price") or order.get("average") or 0.0)
                timestamp = int(
                    order.get("order_filled_timestamp")
                    or order.get("order_timestamp")
                    or 0
                )
            except (TypeError, ValueError):
                continue
            execution_amount += filled
            weighted_value += filled * price
            latest_timestamp = max(latest_timestamp, timestamp)
            exchange_orders.append(
                {
                    "order_id": order_id,
                    "side": exit_side,
                    "status": order.get("status"),
                    "filled": filled,
                    "average": price or None,
                    "filled_at": (
                        datetime.fromtimestamp(
                            timestamp / 1000.0,
                            tz=timezone.utc,
                        ).isoformat()
                        if timestamp > 0
                        else None
                    ),
                }
            )
        if not exchange_orders and trade_id in known_trade_ids:
            continue

        current_amount = abs(float(trade.get("amount") or 0.0))
        after_amount = current_amount if is_open else 0.0
        before_amount = current_amount + execution_amount if is_open else current_amount
        weighted_price = (
            weighted_value / execution_amount
            if execution_amount > 0 and weighted_value > 0
            else None
        )
        close_date = trade.get("close_date")
        if not close_date and latest_timestamp > 0:
            close_date = datetime.fromtimestamp(
                latest_timestamp / 1000.0,
                tz=timezone.utc,
            ).isoformat()
        records.append(
            {
                "trade_id": trade_id,
                "pair": trade.get("pair"),
                "component": classify_component(trade.get("enter_tag")),
                "change": "partial_close" if is_open else "full_close",
                "side": "short" if is_short else "long",
                "managed_amount_before": before_amount,
                "exchange_amount_detected": after_amount,
                "managed_amount_after": after_amount,
                "is_open_after": is_open,
                "close_rate": trade.get("close_rate") or weighted_price,
                "close_date": close_date,
                "realized_profit": trade.get("realized_profit"),
                "close_profit_abs": (
                    trade.get("close_profit_abs")
                    if not is_open
                    else trade.get("realized_profit")
                ),
                "exit_reason": trade.get("exit_reason") or "manual_partial_exchange_close",
                "exchange_orders": exchange_orders,
                "observation": "freqtrade_history_scan",
            }
        )
    return records


def synchronize_manual_exchange_changes(
    api: FreqtradeApiClient,
    status: list[dict[str, Any]],
    balance: dict[str, Any],
    *,
    runtime_running: bool,
) -> dict[str, Any]:
    """Import proven manual exits through Freqtrade's official reload endpoint.

    The function never creates, changes, or cancels an exchange order.  While a
    running bot is being reconciled, entries are paused first and only resumed
    after a second exchange snapshot and a clean post-reload reconciliation.
    """

    account = extract_account_snapshot(balance)
    report = analyze_position_reconciliation(
        status,
        account["exchange_positions"],
    )
    if not report.manual_changes and not report.errors:
        return {
            "status": status,
            "balance": balance,
            "profit": None,
            "records": [],
            "paused": False,
        }

    paused = False
    try:
        if runtime_running:
            api.post("/pause")
            paused = True
            state = api.get("/show_config")
            if str(state.get("state") or "").lower() != "paused":
                raise RuntimeError("Freqtrade 未能进入 PAUSED 状态")

        # Confirm with a fresh wallet snapshot after pausing.  This prevents a
        # just-filled bot order or a transient wallet refresh from being labeled
        # as a manual close.
        fresh_status = api.get("/status")
        fresh_balance = api.get("/balance")
        if not isinstance(fresh_status, list) or not isinstance(fresh_balance, dict):
            raise RuntimeError("人工平仓复核响应格式错误")
        fresh_account = extract_account_snapshot(fresh_balance)
        fresh_report = analyze_position_reconciliation(
            fresh_status,
            fresh_account["exchange_positions"],
        )
        if fresh_report.errors:
            raise RuntimeError("；".join(fresh_report.errors))

        records: list[dict[str, Any]] = []
        before_by_id = {
            int(trade.get("trade_id") or trade.get("id") or 0): trade
            for trade in fresh_status
        }
        for change in fresh_report.manual_changes:
            before_trade = before_by_id.get(change.trade_id)
            if before_trade is None:
                raise RuntimeError(
                    f"{change.pair} 的 trade_id={change.trade_id} 在复核时消失"
                )
            # This is Freqtrade's supported recovery endpoint.  It fetches the
            # exchange order history and imports actual fills, fees, price and
            # timestamp into the normal Trade/Order ledger.
            reloaded = api.post(f"/trades/{change.trade_id}/reload")
            if not isinstance(reloaded, dict):
                raise RuntimeError(f"{change.pair} 成交回写响应格式错误")
            records.append(
                _manual_reconciliation_record(change, before_trade, reloaded)
            )

        final_status = api.get("/status")
        final_balance = api.get("/balance")
        final_profit = api.get("/profit")
        if not isinstance(final_status, list) or not isinstance(final_balance, dict):
            raise RuntimeError("人工平仓回写后的账户响应格式错误")
        final_account = extract_account_snapshot(final_balance)
        final_errors = validate_position_reconciliation(
            final_status,
            final_account["exchange_positions"],
        )
        if final_errors:
            raise RuntimeError("回写后仍未对齐：" + "；".join(final_errors))
        if records:
            append_manual_reconciliation_audit(records)

        if runtime_running:
            api.post("/start")
            state = api.get("/show_config")
            if str(state.get("state") or "").lower() != "running":
                raise RuntimeError("对账完成，但 Freqtrade 未能恢复 RUNNING 状态")
            paused = False
        return {
            "status": final_status,
            "balance": final_balance,
            "profit": final_profit,
            "records": records,
            "paused": paused,
        }
    except Exception as exc:
        if runtime_running:
            try:
                api.post("/pause")
                paused = True
            except Exception:
                try:
                    api.post("/stop")
                    paused = True
                except Exception:
                    pass
        raise ManualReconciliationError(
            str(exc),
            trading_paused=paused,
        ) from exc


class ColorButton(tk.Label):
    """A platform-neutral button whose face color is never replaced by Aqua."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        command: Any = None,
        activebackground: str | None = None,
        activeforeground: str | None = None,
        disabledbackground: str | None = None,
        disabledforeground: str = DISABLED_TEXT,
        state: str = "normal",
        **kwargs: Any,
    ) -> None:
        background = str(
            kwargs.pop("background", kwargs.pop("bg", BUTTON_BG))
        )
        foreground = str(
            kwargs.pop("foreground", kwargs.pop("fg", WHITE))
        )
        self._command = command
        self._normal_background = background
        self._normal_foreground = foreground
        self._active_background = activebackground or background
        self._active_foreground = activeforeground or foreground
        self._disabled_background_follows_normal = disabledbackground is None
        self._disabled_background = disabledbackground or background
        self._disabled_foreground = disabledforeground
        self._button_state = state
        self._hovered = False
        self._pressed = False

        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("borderwidth", 0)
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("highlightbackground", BORDER)
        kwargs.setdefault("highlightcolor", TEXT)
        kwargs.setdefault("takefocus", 1)
        kwargs["bg"] = background
        kwargs["fg"] = foreground
        super().__init__(parent, **kwargs)

        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<ButtonPress-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self.bind("<Return>", self._on_keyboard_activate, add="+")
        self.bind("<space>", self._on_keyboard_activate, add="+")
        self._render()

    def _render(self, *, active: bool | None = None) -> None:
        if self._button_state == "disabled":
            background = self._disabled_background
            foreground = self._disabled_foreground
            cursor = "arrow"
        else:
            use_active = self._hovered if active is None else active
            background = (
                self._active_background
                if use_active
                else self._normal_background
            )
            foreground = (
                self._active_foreground
                if use_active
                else self._normal_foreground
            )
            cursor = "hand2"
        super().configure(bg=background, fg=foreground, cursor=cursor)

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._hovered = True
        self._render()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._hovered = False
        self._pressed = False
        self._render(active=False)

    def _on_press(self, _event: tk.Event[Any]) -> None:
        if self._button_state == "disabled":
            return
        self._pressed = True
        self._render(active=True)

    def _on_release(self, event: tk.Event[Any]) -> None:
        if self._button_state == "disabled":
            return
        was_pressed = self._pressed
        self._pressed = False
        inside = (
            0 <= int(event.x) < self.winfo_width()
            and 0 <= int(event.y) < self.winfo_height()
        )
        self._render(active=inside)
        if was_pressed and inside and callable(self._command):
            self._command()

    def _on_keyboard_activate(self, _event: tk.Event[Any]) -> str:
        if self._button_state != "disabled" and callable(self._command):
            self._command()
        return "break"

    def configure(
        self,
        cnf: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        options: dict[str, Any] = {}
        if cnf:
            options.update(cnf)
        options.update(kwargs)
        if "command" in options:
            self._command = options.pop("command")
        if "state" in options:
            state = str(options.pop("state"))
            if state not in {"normal", "disabled"}:
                raise tk.TclError(f"bad state {state!r}: must be normal or disabled")
            self._button_state = state
        if "bg" in options or "background" in options:
            self._normal_background = str(
                options.pop("background", options.pop("bg", ""))
            )
            if self._disabled_background_follows_normal:
                self._disabled_background = self._normal_background
        if "fg" in options or "foreground" in options:
            self._normal_foreground = str(
                options.pop("foreground", options.pop("fg", ""))
            )
        if "activebackground" in options:
            self._active_background = str(options.pop("activebackground"))
        if "activeforeground" in options:
            self._active_foreground = str(options.pop("activeforeground"))
        if "disabledbackground" in options:
            self._disabled_background_follows_normal = False
            self._disabled_background = str(options.pop("disabledbackground"))
        if "disabledforeground" in options:
            self._disabled_foreground = str(options.pop("disabledforeground"))
        result = super().configure(**options) if options else None
        self._render()
        return result

    config = configure

    def cget(self, key: str) -> Any:
        normalized = {
            "background": "bg",
            "foreground": "fg",
        }.get(key, key)
        if normalized == "state":
            return self._button_state
        if normalized == "bg":
            return self._normal_background
        if normalized == "fg":
            return self._normal_foreground
        if normalized == "activebackground":
            return self._active_background
        if normalized == "activeforeground":
            return self._active_foreground
        if normalized == "disabledbackground":
            return self._disabled_background
        if normalized == "disabledforeground":
            return self._disabled_foreground
        return super().cget(key)


def fitted_window_geometry(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int, int, int]:
    """Keep the complete console visible on scaled laptop displays."""

    available_width = max(760, int(screen_width) - 80)
    available_height = max(560, int(screen_height) - 100)
    width = min(1160, available_width)
    height = min(700, available_height)
    x = max(0, (int(screen_width) - width) // 2)
    y = max(24, (int(screen_height) - height) // 2 - 8)
    return width, height, x, y


def apply_app_icon(root: tk.Misc) -> tk.PhotoImage | None:
    if not APP_ICON_PATH.is_file():
        return None
    try:
        icon = tk.PhotoImage(file=str(APP_ICON_PATH))
        root.wm_iconphoto(True, icon)
        return icon
    except tk.TclError:
        return None


class TypedConfirmationDialog:
    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        heading: str,
        body: str,
        phrase: str,
        confirm_text: str,
        accent: str = RED,
    ) -> None:
        self.result = False
        self.phrase = phrase
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.configure(bg=PANEL)
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

        width, height = 500, 400
        parent.update_idletasks()
        x = parent.winfo_rootx() + max(20, (parent.winfo_width() - width) // 2)
        y = parent.winfo_rooty() + max(20, (parent.winfo_height() - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

        top_rule = tk.Frame(self.window, bg=accent, height=4)
        top_rule.pack(fill="x")
        content = tk.Frame(self.window, bg=PANEL, padx=24, pady=20)
        content.pack(fill="both", expand=True)

        tk.Label(
            content,
            text=heading,
            bg=PANEL,
            fg=TEXT,
            font=(FONT_FAMILY, 16, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            content,
            text=body,
            bg=PANEL,
            fg=MUTED,
            font=(FONT_FAMILY, 11),
            justify="left",
            anchor="w",
            wraplength=440,
            pady=11,
        ).pack(fill="x")

        phrase_panel = tk.Frame(
            content,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=14,
            pady=11,
        )
        phrase_panel.pack(fill="x", pady=(4, 14))
        tk.Label(
            phrase_panel,
            text="请输入以下确认短语",
            bg=CARD,
            fg=MUTED,
            font=(FONT_FAMILY, 10),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            phrase_panel,
            text=phrase,
            bg=CARD,
            fg=accent,
            font=(MONO_FAMILY, 14, "bold"),
            anchor="w",
            pady=5,
        ).pack(fill="x")
        self.value = tk.StringVar()
        entry = tk.Entry(
            phrase_panel,
            textvariable=self.value,
            bg=BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=(MONO_FAMILY, 13),
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=accent,
        )
        entry.pack(fill="x", ipady=9, pady=(4, 0))

        buttons = tk.Frame(content, bg=PANEL)
        buttons.pack(fill="x", side="bottom")
        ColorButton(
            buttons,
            text="取消",
            command=self._cancel,
            bg=CARD_ALT,
            activebackground=BORDER,
            fg=TEXT,
            activeforeground=TEXT,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 11, "bold"),
            cursor="hand2",
            padx=24,
            pady=9,
        ).pack(side="left")
        self.confirm_button = ColorButton(
            buttons,
            text=confirm_text,
            command=self._confirm,
            bg=accent,
            activebackground=accent,
            fg=WHITE,
            activeforeground=WHITE,
            disabledforeground=DISABLED_TEXT,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 11, "bold"),
            cursor="hand2",
            padx=24,
            pady=9,
            state="disabled",
        )
        self.confirm_button.pack(side="right")

        self.value.trace_add("write", self._validate)
        entry.bind("<Return>", lambda _event: self._confirm())
        entry.bind("<Escape>", lambda _event: self._cancel())
        self.window.grab_set()
        entry.focus_set()
        parent.wait_window(self.window)

    def _validate(self, *_args: Any) -> None:
        state = "normal" if self.value.get().strip() == self.phrase else "disabled"
        self.confirm_button.configure(state=state)

    def _confirm(self) -> None:
        if self.value.get().strip() != self.phrase:
            return
        self.result = True
        self.window.destroy()

    def _cancel(self) -> None:
        self.result = False
        self.window.destroy()


class MetricCard(tk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        value_var: tk.StringVar,
        hint_var: tk.StringVar | None = None,
        accent: str = BLUE,
    ) -> None:
        super().__init__(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=12,
            pady=9,
        )
        tk.Frame(self, bg=accent, width=3).pack(side="left", fill="y", padx=(0, 9))
        body = tk.Frame(self, bg=CARD)
        body.pack(fill="both", expand=True)
        tk.Label(
            body,
            text=title,
            bg=CARD,
            fg=MUTED,
            font=(FONT_FAMILY, 9),
            anchor="w",
        ).pack(fill="x")
        self.value_label = tk.Label(
            body,
            textvariable=value_var,
            bg=CARD,
            fg=TEXT,
            font=(FONT_FAMILY, 15, "bold"),
            anchor="w",
            pady=2,
        )
        self.value_label.pack(fill="x")
        if hint_var is not None:
            tk.Label(
                body,
                textvariable=hint_var,
                bg=CARD,
                fg=FAINT,
                font=(FONT_FAMILY, 9),
                anchor="w",
            ).pack(fill="x")


class TradingConsole:
    POLL_INTERVAL_MS = 10_000

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "Breakout V16 + Grid V15 PF Score 2 Decay Console"
        )
        self.root.configure(bg=BG)
        self.app_icon = apply_app_icon(self.root)
        width, height, x, y = fitted_window_geometry(
            self.root.winfo_screenwidth(),
            self.root.winfo_screenheight(),
        )
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(min(960, width), min(600, height))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.mode = tk.StringVar(value=MODE_DRY)
        self.engine_state = "OFF"
        self.process: subprocess.Popen[str] | None = None
        self.launch_spec: LaunchSpec | None = None
        self.api: FreqtradeApiClient | None = None
        self.process_lock = threading.RLock()
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.poll_in_flight = False
        self.next_poll_at = 0.0
        self.closing_after_stop = False
        self.intentional_stop = False
        self.session_baseline_equity: float | None = DEFAULT_DRY_WALLET
        self.last_snapshot: dict[str, Any] = {}
        self.log_line_count = 0
        self.last_poll_error = ""
        self.last_poll_error_at = 0.0
        self.poll_failure_count = 0
        self.start_cancel_event = threading.Event()
        manual_markers = load_manual_reconciliation_markers()
        self.known_manual_trade_ids = set(manual_markers.trade_ids)
        self.known_manual_order_ids = set(manual_markers.order_ids)

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.gui_log_path = (
            LOG_DIR
            / f"gui_breakout_v16_grid_v15_combined_{datetime.now():%Y%m%d}.log"
        )
        archived_gui_log = archive_oversized_log(self.gui_log_path)

        self._setup_ttk_styles()
        self._build_ui()
        self._refresh_key_indicator()
        self._apply_offline_mode_defaults()
        self._set_engine_state("OFF", "未启动")
        if archived_gui_log is not None:
            self._log(
                f"上一份超大 GUI 日志已无损归档：{archived_gui_log.name}",
                "info",
            )

        report = verify_release()
        active_engine = active_pid_lock(ENGINE_LOCK_PATH)
        if active_engine is not None:
            pid = active_engine.get("pid") or "?"
            self._set_engine_state("LOCKED", f"检测到已有引擎 PID {pid}")
            self._log(
                f"检测到尚在运行的 V16 + Grid V15 引擎 PID {pid}，"
                "为防止重复下单，本 GUI 已锁定启动。请先检查原进程及持仓。",
                "error",
            )
        elif report.ok:
            self._log(
                f"GUI 已就绪：{RELEASE_LABEL}，50币，最多同时 2 仓。",
                "success",
            )
            self._log(
                "组合限制为最多 1 个 Breakout + 1 个 Grid，最后一格保持 Breakout 优先；非农追加仓禁用。",
                "info",
            )
            self._log(
                "DRY-RUN、LIVE 与回测共用同一组合策略类；Breakout 使用 1m 软确认和交易所硬止损。",
                "info",
            )
            self._log(
                "默认 DRY-RUN 使用 200.00U 模拟本金和 Binance 主网公开行情，不发送真实订单。",
                "info",
            )
        else:
            self._set_engine_state("LOCKED", "版本校验失败")
            for error in report.errors:
                self._log(error, "error")

        self.root.after(100, self._drain_events)
        self.root.after(500, self._poll_scheduler)
        self.root.after(80, self._present_window)

    def _present_window(self) -> None:
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.wm_attributes("-topmost", True)
            self.root.after(
                300,
                lambda: self.root.wm_attributes("-topmost", False),
            )
        except tk.TclError:
            pass

    def _setup_ttk_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "V16.Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=TEXT,
            rowheight=29,
            borderwidth=0,
            font=(FONT_FAMILY, 10),
        )
        style.map(
            "V16.Treeview",
            background=[("selected", "#244A7C")],
            foreground=[("selected", WHITE)],
        )
        style.configure(
            "V16.Treeview.Heading",
            background=CARD_ALT,
            foreground=MUTED,
            relief="flat",
            borderwidth=0,
            padding=(8, 7),
            font=(FONT_FAMILY, 9, "bold"),
        )
        style.map(
            "V16.Treeview.Heading",
            background=[("active", CARD_ALT)],
            relief=[("active", "flat")],
        )
        style.configure(
            "Vertical.TScrollbar",
            background=CARD_ALT,
            troughcolor=BG,
            bordercolor=BG,
            arrowcolor=MUTED,
            relief="flat",
        )

    def _build_ui(self) -> None:
        self._build_header()
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        self.sidebar = tk.Frame(shell, bg=PANEL, width=292)
        self.sidebar.pack(side="left", fill="y", padx=(0, 12))
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        self.workspace = tk.Frame(shell, bg=BG)
        self.workspace.pack(side="left", fill="both", expand=True)
        self._build_workspace()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG, height=70)
        header.pack(fill="x", padx=16, pady=(10, 7))
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", fill="y")
        badge = tk.Label(
            brand,
            text="V16 + GRID · MAX2",
            bg=BLUE,
            fg=WHITE,
            font=(FONT_FAMILY, 11, "bold"),
            padx=10,
            pady=5,
        )
        badge.pack(side="left", pady=12)
        title_box = tk.Frame(brand, bg=BG)
        title_box.pack(side="left", padx=(10, 0), pady=7)
        tk.Label(
            title_box,
            text="Trading Console",
            bg=BG,
            fg=TEXT,
            font=(FONT_FAMILY, 20, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_box,
            text="Breakout V16 MTF + Grid V15 PF  ·  Binance USDT-M Futures  ·  Freqtrade 2026.6",
            bg=BG,
            fg=MUTED,
            font=(FONT_FAMILY, 10),
            anchor="w",
        ).pack(fill="x")

        status = tk.Frame(
            header,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=12,
            pady=7,
        )
        status.pack(side="right", pady=9)
        self.status_dot = tk.Label(
            status,
            text="●",
            bg=PANEL,
            fg=FAINT,
            font=(FONT_FAMILY, 13),
        )
        self.status_dot.pack(side="left", padx=(0, 8))
        status_text = tk.Frame(status, bg=PANEL)
        status_text.pack(side="left")
        self.status_title_var = tk.StringVar(value="OFFLINE")
        self.status_detail_var = tk.StringVar(value="未启动")
        tk.Label(
            status_text,
            textvariable=self.status_title_var,
            bg=PANEL,
            fg=TEXT,
            font=(FONT_FAMILY, 10, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            status_text,
            textvariable=self.status_detail_var,
            bg=PANEL,
            fg=MUTED,
            font=(FONT_FAMILY, 9),
            anchor="w",
        ).pack(fill="x")

    def _section_title(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent,
            text=text.upper(),
            bg=PANEL,
            fg=FAINT,
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 5))

    def _build_sidebar(self) -> None:
        self._section_title(self.sidebar, "运行环境")
        mode_panel = tk.Frame(self.sidebar, bg=CARD, padx=5, pady=5)
        mode_panel.pack(fill="x", padx=12)
        self.dry_button = ColorButton(
            mode_panel,
            text="DRY-RUN 模拟",
            command=lambda: self._select_mode(MODE_DRY),
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 10, "bold"),
            cursor="hand2",
            pady=7,
        )
        self.dry_button.pack(side="left", fill="x", expand=True)
        self.live_button = ColorButton(
            mode_panel,
            text="LIVE 实盘",
            command=lambda: self._select_mode(MODE_LIVE),
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 10, "bold"),
            cursor="hand2",
            pady=7,
        )
        self.live_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._paint_mode_buttons()

        self.mode_description_var = tk.StringVar()
        tk.Label(
            self.sidebar,
            textvariable=self.mode_description_var,
            bg=PANEL,
            fg=MUTED,
            font=(FONT_FAMILY, 9),
            justify="left",
            wraplength=258,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(6, 0))

        self._section_title(self.sidebar, "策略配置")
        strategy = tk.Frame(
            self.sidebar,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=11,
            pady=8,
        )
        strategy.pack(fill="x", padx=12)
        tk.Label(
            strategy,
            text=RELEASE_LABEL,
            bg=CARD,
            fg=TEXT,
            font=(FONT_FAMILY, 11, "bold"),
            anchor="w",
        ).pack(fill="x")
        self.strategy_detail_var = tk.StringVar(
            value=(
                "共享 200U / 实盘账户  ·  Breakout 1 + Grid 1\n"
                "50 币静态池  ·  1H信号+15M路径  ·  Grid分层DCA\n"
                "1M软确认  ·  交易所硬止损  ·  风险等额缩仓"
            )
        )
        tk.Label(
            strategy,
            textvariable=self.strategy_detail_var,
            bg=CARD,
            fg=MUTED,
            font=(FONT_FAMILY, 9),
            justify="left",
            anchor="w",
            pady=5,
        ).pack(fill="x")
        frozen = tk.Frame(strategy, bg=GREEN_DARK, padx=8, pady=4)
        frozen.pack(fill="x", pady=(3, 0))
        tk.Label(
            frozen,
            text="✓  策略与配置 SHA-256 冻结校验",
            bg=GREEN_DARK,
            fg=GREEN,
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(fill="x")

        self._section_title(self.sidebar, "连接与保护")
        connection = tk.Frame(
            self.sidebar,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=11,
            pady=8,
        )
        connection.pack(fill="x", padx=12)
        key_line = tk.Frame(connection, bg=CARD)
        key_line.pack(fill="x")
        self.key_dot = tk.Label(
            key_line,
            text="●",
            bg=CARD,
            fg=FAINT,
            font=(FONT_FAMILY, 10),
        )
        self.key_dot.pack(side="left")
        self.key_status_var = tk.StringVar(value="正在检查原 .env")
        tk.Label(
            key_line,
            textvariable=self.key_status_var,
            bg=CARD,
            fg=TEXT,
            font=(FONT_FAMILY, 9, "bold"),
            anchor="w",
        ).pack(side="left", padx=(7, 0))
        tk.Label(
            connection,
            text="密钥只在启动 LIVE 子进程时通过内存注入；\n不会写入配置、GUI 日志或运行清单。",
            bg=CARD,
            fg=MUTED,
            font=(FONT_FAMILY, 8),
            justify="left",
            anchor="w",
            pady=4,
        ).pack(fill="x")

        spacer = tk.Frame(self.sidebar, bg=PANEL)
        spacer.pack(fill="both", expand=True)

        action = tk.Frame(self.sidebar, bg=PANEL, padx=12, pady=10)
        action.pack(fill="x", side="bottom")
        self.refresh_button = ColorButton(
            action,
            text="↻  刷新账户",
            command=self._on_refresh,
            bg=REFRESH_BUTTON,
            activebackground=REFRESH_BUTTON_HOVER,
            fg=WHITE,
            activeforeground=WHITE,
            disabledforeground=DISABLED_TEXT,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 10, "bold"),
            cursor="hand2",
            pady=8,
        )
        self.refresh_button.pack(fill="x", pady=(0, 8))
        primary = tk.Frame(action, bg=PANEL)
        primary.pack(fill="x")
        self.start_button = ColorButton(
            primary,
            text="▶  启动模拟",
            command=self._on_start,
            bg=START_BUTTON,
            activebackground=START_BUTTON_HOVER,
            fg=WHITE,
            activeforeground=WHITE,
            disabledforeground=DISABLED_TEXT,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 11, "bold"),
            cursor="hand2",
            pady=9,
        )
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ColorButton(
            primary,
            text="■  停止引擎",
            command=self._on_stop,
            bg=STOP_BUTTON,
            activebackground=STOP_BUTTON_HOVER,
            fg=WHITE,
            activeforeground=WHITE,
            disabledforeground=DISABLED_TEXT,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 11, "bold"),
            cursor="hand2",
            padx=18,
            pady=9,
            state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        self.baseline_note_var = tk.StringVar(
            value="DRY-RUN 本金固定为 200.00U"
        )
        tk.Label(
            action,
            textvariable=self.baseline_note_var,
            bg=PANEL,
            fg=FAINT,
            font=(FONT_FAMILY, 8),
            anchor="center",
            pady=5,
        ).pack(fill="x")

    def _build_workspace(self) -> None:
        metrics = tk.Frame(self.workspace, bg=BG)
        metrics.pack(fill="x")
        for column in range(6):
            metrics.grid_columnconfigure(column, weight=1, uniform="metric")

        self.equity_var = tk.StringVar(value="200.00 U")
        self.available_var = tk.StringVar(value="200.00 U")
        self.session_pnl_var = tk.StringVar(value="+0.00 U")
        self.position_count_var = tk.StringVar(value="0 / 2")
        self.closed_count_var = tk.StringVar(value="0")
        self.quality_var = tk.StringVar(value="PF —")

        self.equity_hint = tk.StringVar(value="模拟账户权益")
        self.available_hint = tk.StringVar(value="可用保证金")
        self.pnl_hint = tk.StringVar(value="历史 +0.00U · 持仓 +0.00U")
        self.position_hint = tk.StringVar(value="共享账户")
        self.closed_hint = tk.StringVar(value="当前独立账本")
        self.quality_hint = tk.StringVar(value="胜率 —")

        cards = (
            ("账户权益", self.equity_var, self.equity_hint, BLUE),
            ("可用资金", self.available_var, self.available_hint, PURPLE),
            ("累计盈亏", self.session_pnl_var, self.pnl_hint, GREEN),
            ("当前持仓", self.position_count_var, self.position_hint, AMBER),
            ("已平仓", self.closed_count_var, self.closed_hint, BLUE),
            ("运行质量", self.quality_var, self.quality_hint, GREEN),
        )
        self.metric_cards: list[MetricCard] = []
        for index, (title, value, hint, accent) in enumerate(cards):
            card = MetricCard(
                metrics,
                title=title,
                value_var=value,
                hint_var=hint,
                accent=accent,
            )
            card.grid(
                row=0,
                column=index,
                sticky="nsew",
                padx=(0 if index == 0 else 3, 0 if index == 5 else 3),
            )
            self.metric_cards.append(card)

        positions_panel = tk.Frame(
            self.workspace,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        positions_panel.pack(fill="both", expand=True, pady=(9, 8))
        positions_header = tk.Frame(positions_panel, bg=PANEL, padx=13, pady=8)
        positions_header.pack(fill="x")
        tk.Label(
            positions_header,
            text="账户持仓",
            bg=PANEL,
            fg=TEXT,
            font=(FONT_FAMILY, 13, "bold"),
            anchor="w",
        ).pack(side="left")
        self.position_status_var = tk.StringVar(value="暂无持仓")
        tk.Label(
            positions_header,
            textvariable=self.position_status_var,
            bg=PANEL,
            fg=MUTED,
            font=(FONT_FAMILY, 9),
            anchor="e",
        ).pack(side="right")

        tree_shell = tk.Frame(positions_panel, bg=CARD)
        tree_shell.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        columns = (
            "pair",
            "component",
            "side",
            "leverage",
            "stake",
            "entry",
            "current",
            "pnl",
            "duration",
            "custody",
        )
        self.positions = ttk.Treeview(
            tree_shell,
            columns=columns,
            show="headings",
            style="V16.Treeview",
            height=6,
        )
        headings = {
            "pair": "交易对",
            "component": "策略",
            "side": "方向",
            "leverage": "杠杆",
            "stake": "保证金",
            "entry": "开仓价",
            "current": "当前价",
            "pnl": "盈亏(U)",
            "duration": "持仓时间",
            "custody": "托管状态",
        }
        widths = {
            "pair": 100,
            "component": 122,
            "side": 60,
            "leverage": 58,
            "stake": 82,
            "entry": 90,
            "current": 90,
            "pnl": 76,
            "duration": 84,
            "custody": 84,
        }
        for column in columns:
            anchor = "w" if column in {"pair", "component"} else "center"
            self.positions.heading(column, text=headings[column], anchor=anchor)
            self.positions.column(
                column,
                width=widths[column],
                minwidth=50,
                anchor=anchor,
                stretch=column in {"pair", "component"},
            )
        self.positions.tag_configure("profit", foreground=GREEN)
        self.positions.tag_configure("loss", foreground=RED)
        self.positions.tag_configure("neutral", foreground=TEXT)
        scrollbar = ttk.Scrollbar(
            tree_shell,
            orient="vertical",
            command=self.positions.yview,
            style="Vertical.TScrollbar",
        )
        self.positions.configure(yscrollcommand=scrollbar.set)
        self.positions.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        log_panel = tk.Frame(
            self.workspace,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            height=180,
        )
        log_panel.pack(fill="x")
        log_panel.pack_propagate(False)
        log_header = tk.Frame(log_panel, bg=PANEL, padx=13, pady=7)
        log_header.pack(fill="x")
        tk.Label(
            log_header,
            text="运行日志",
            bg=PANEL,
            fg=TEXT,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(side="left")
        tk.Label(
            log_header,
            text="所有行均含秒级时间 · 密钥自动脱敏",
            bg=PANEL,
            fg=FAINT,
            font=(FONT_FAMILY, 8),
        ).pack(side="left", padx=(10, 0))
        ColorButton(
            log_header,
            text="清空界面",
            command=self._clear_visible_log,
            bg=BUTTON_BG,
            activebackground=BUTTON_HOVER,
            fg=WHITE,
            activeforeground=WHITE,
            relief="flat",
            borderwidth=0,
            font=(FONT_FAMILY, 9),
            cursor="hand2",
            padx=9,
            pady=3,
        ).pack(side="right")

        log_shell = tk.Frame(log_panel, bg=BG)
        log_shell.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        self.log_text = tk.Text(
            log_shell,
            bg="#080E1B",
            fg=MUTED,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            wrap="word",
            font=(MONO_FAMILY, 9),
            padx=10,
            pady=7,
            state="disabled",
        )
        self.log_text.tag_configure("info", foreground=MUTED)
        self.log_text.tag_configure("success", foreground=GREEN)
        self.log_text.tag_configure("warning", foreground=AMBER)
        self.log_text.tag_configure("error", foreground=RED)
        self.log_text.tag_configure("engine", foreground="#9CB8E8")
        log_scrollbar = ttk.Scrollbar(
            log_shell,
            orient="vertical",
            command=self.log_text.yview,
            style="Vertical.TScrollbar",
        )
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.pack(side="right", fill="y")

    def _refresh_key_indicator(self) -> None:
        bundle = load_exchange_secrets()
        if bundle.ready:
            self.key_dot.configure(fg=GREEN)
            self.key_status_var.set("原 .env 密钥已就绪")
        else:
            self.key_dot.configure(fg=RED)
            self.key_status_var.set("未找到完整密钥")

    def _paint_mode_buttons(self) -> None:
        selected = self.mode.get()
        for button, mode in (
            (self.dry_button, MODE_DRY),
            (self.live_button, MODE_LIVE),
        ):
            if selected == mode:
                color = DRY_BUTTON if mode == MODE_DRY else LIVE_BUTTON
                hover = (
                    DRY_BUTTON_HOVER
                    if mode == MODE_DRY
                    else LIVE_BUTTON_HOVER
                )
                button.configure(
                    bg=color,
                    activebackground=hover,
                    fg=WHITE,
                    activeforeground=WHITE,
                )
            else:
                button.configure(
                    bg=MODE_IDLE_BUTTON,
                    activebackground=BUTTON_HOVER,
                    fg=WHITE,
                    activeforeground=WHITE,
                )

    def _selected_max_open_trades(self) -> int:
        with self.process_lock:
            spec = self.launch_spec
            active = self.process is not None and self.process.poll() is None
        if active and spec is not None:
            return int(spec.max_open_trades)
        return MAX_OPEN_TRADES

    def _select_mode(self, mode: str) -> None:
        with self.process_lock:
            active = self.process is not None and self.process.poll() is None
        if active or self.engine_state in {"STARTING", "STOPPING"}:
            self._log("运行期间不能切换环境，请先停止当前引擎。", "warning")
            return
        if mode == self.mode.get():
            return
        self.mode.set(mode)
        self._paint_mode_buttons()
        self.session_baseline_equity = (
            DEFAULT_DRY_WALLET if mode == MODE_DRY else None
        )
        self.last_snapshot = {}
        self._apply_offline_mode_defaults()
        self._refresh_key_indicator()
        if mode == MODE_DRY:
            self._log(
                "已切换到 DRY-RUN：使用主网公开行情和 200.00U 模拟资金，不会发送订单。",
                "success",
            )
        else:
            self._log(
                "已切换到 LIVE：当前尚未启动，不会发送订单；"
                "点击“启动实盘”后会直接进入安全预检。",
                "warning",
            )

    def _apply_offline_mode_defaults(self) -> None:
        if self.mode.get() == MODE_DRY:
            self.start_button.configure(text="▶  启动模拟")
            self.mode_description_var.set(
                "主网实时公开行情 · 模拟资金与成交\n不会提交、修改或撤销交易所订单"
            )
            self.baseline_note_var.set("DRY-RUN 本金固定为 200.00U")
            self.equity_var.set(f"{DEFAULT_DRY_WALLET:,.2f} U")
            self.available_var.set(f"{DEFAULT_DRY_WALLET:,.2f} U")
            self.session_pnl_var.set("+0.00 U")
            self.pnl_hint.set("历史 +0.00U · 持仓 +0.00U")
            self.equity_hint.set("模拟账户权益")
        else:
            self.start_button.configure(text="▶  启动实盘")
            self.mode_description_var.set(
                "Binance 主网真实账户 · 真实资金\n启动后会实际开仓、加仓、止盈与止损"
            )
            self.baseline_note_var.set("累计盈亏 = 历史已实现 + 当前持仓盈亏")
            self.equity_var.set("—")
            self.available_var.set("—")
            self.session_pnl_var.set("+0.00 U")
            self.pnl_hint.set("等待盈亏刷新")
            self.equity_hint.set("主网账户权益")
        self.position_count_var.set(f"0 / {self._selected_max_open_trades()}")
        self.closed_count_var.set("0")
        self.quality_var.set("PF —")
        self.quality_hint.set("胜率 —")
        self._render_positions([], [])

    def _set_engine_state(self, state: str, detail: str) -> None:
        self.engine_state = state
        title_map = {
            "OFF": "OFFLINE",
            "STARTING": "STARTING",
            "RUNNING": self.mode.get(),
            "STOPPING": "STOPPING",
            "LOCKED": "LOCKED",
        }
        color_map = {
            "OFF": FAINT,
            "STARTING": AMBER,
            "RUNNING": GREEN if self.mode.get() == MODE_DRY else RED,
            "STOPPING": AMBER,
            "LOCKED": RED,
        }
        self.status_title_var.set(title_map.get(state, state))
        self.status_detail_var.set(detail)
        self.status_dot.configure(fg=color_map.get(state, FAINT))
        busy = state in {"STARTING", "RUNNING", "STOPPING"}
        with self.process_lock:
            process_active = (
                self.process is not None and self.process.poll() is None
            )
        self.dry_button.configure(state="disabled" if busy else "normal")
        self.live_button.configure(state="disabled" if busy else "normal")
        self.start_button.configure(
            state="disabled" if busy or state == "LOCKED" else "normal"
        )
        self.stop_button.configure(
            state=(
                "normal"
                if state in {"STARTING", "RUNNING"}
                or (state == "LOCKED" and process_active)
                else "disabled"
            )
        )
        self.refresh_button.configure(
            state="disabled" if state in {"STARTING", "STOPPING"} else "normal"
        )

    def _check_live_start_prerequisites(self) -> bool:
        bundle = load_exchange_secrets()
        if not bundle.ready:
            messagebox.showerror(
                "LIVE 无法启动",
                "原 .env 中没有找到完整的 Binance Futures API Key/Secret。",
                parent=self.root,
            )
            return False
        return True

    def _on_start(self) -> None:
        if self.engine_state not in {"OFF"}:
            return
        active_engine = active_pid_lock(ENGINE_LOCK_PATH)
        if active_engine is not None:
            pid = active_engine.get("pid") or "?"
            self._set_engine_state("LOCKED", f"已有引擎 PID {pid}")
            self._log(
                f"启动被阻止：检测到已有 V16 + Grid V15 引擎 PID {pid}。",
                "error",
            )
            return
        report = verify_release()
        if not report.ok:
            self._set_engine_state("LOCKED", "版本校验失败")
            self._log("启动被阻止：策略或配置冻结校验未通过。", "error")
            for error in report.errors:
                self._log(error, "error")
            messagebox.showerror(
                "启动被阻止",
                "\n".join(report.errors),
                parent=self.root,
            )
            return
        selected_mode = self.mode.get()
        if (
            selected_mode == MODE_LIVE
            and not self._check_live_start_prerequisites()
        ):
            self._log("LIVE 启动被阻止：未找到完整密钥。", "error")
            return
        self.intentional_stop = False
        self.start_cancel_event.clear()
        self._set_engine_state("STARTING", "正在进行安全预检")
        self._log(
            f"已请求启动 {RELEASE_LABEL}"
            + f" · {selected_mode}；交易引擎先以 STOPPED 状态加载。",
            "warning" if selected_mode == MODE_LIVE else "info",
        )
        threading.Thread(
            target=self._launch_worker,
            args=(selected_mode,),
            name="v16-grid-v15-launch",
            daemon=True,
        ).start()

    def _launch_worker(self, mode: str) -> None:
        spec: LaunchSpec | None = None
        process: subprocess.Popen[str] | None = None
        api: FreqtradeApiClient | None = None
        trading_started = False
        engine_lock_acquired = False
        output_reader_started = False
        startup_manual_records: list[dict[str, Any]] = []
        try:
            if self.start_cancel_event.is_set():
                raise LaunchCancelled("启动已由用户取消")
            try:
                clock_sync = measure_binance_clock()
            except Exception as exc:
                if mode == MODE_LIVE:
                    raise RuntimeError(
                        "LIVE 启动前无法完成 Binance 公共时间校验，"
                        f"已安全阻止启动：{friendly_runtime_error(str(exc))}"
                    ) from exc
                clock_sync = None
                self.events.put(
                    (
                        "log",
                        (
                            "DRY-RUN 时间校验暂时不可用，将继续使用公开行情："
                            f"{friendly_runtime_error(str(exc))}",
                            "warning",
                            None,
                        ),
                    )
                )
            if clock_sync is not None:
                self.events.put(
                    (
                        "log",
                        (
                            "Binance 时间校验完成："
                            f"本机-服务器={clock_sync.local_minus_server_ms:+d}ms，"
                            "签名请求已预留 2000ms 安全余量。",
                            "success",
                            None,
                        ),
                    )
                )
            spec = build_launch_spec(
                mode,
                time_difference_ms=(
                    clock_sync.time_difference_ms
                    if clock_sync is not None
                    else None
                ),
            )
            if self.start_cancel_event.is_set():
                raise LaunchCancelled("启动已由用户取消")
            self.events.put(
                (
                    "log",
                    (
                        f"启动命令：{compact_command(spec.command)}",
                        "engine",
                        spec.secret_bundle,
                    ),
                )
            )
            process = subprocess.Popen(
                list(spec.command),
                cwd=PROJECT_DIR,
                env=spec.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            acquire_pid_lock(
                ENGINE_LOCK_PATH,
                pid=process.pid,
                kind="Breakout V16 + Grid V15 PF Precision Guard engine",
                details={
                    "mode": mode,
                    "strategy": spec.strategy_class,
                    "max_open_trades": spec.max_open_trades,
                    "api_port": spec.api_port,
                    "database_path": str(spec.database_path),
                    "overlay_path": str(spec.overlay_path),
                },
            )
            engine_lock_acquired = True
            api = FreqtradeApiClient(spec.api_port, spec.api_credentials)
            with self.process_lock:
                self.process = process
                self.launch_spec = spec
                self.api = api

            threading.Thread(
                target=self._read_process_output,
                args=(process, spec),
                name="v16-grid-v15-output",
                daemon=True,
            ).start()
            output_reader_started = True

            deadline = time.monotonic() + 90.0
            while time.monotonic() < deadline:
                if self.start_cancel_event.is_set():
                    raise LaunchCancelled("启动已由用户取消")
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Freqtrade 在预检前退出，返回码 {process.returncode}"
                    )
                if api.ping(timeout=1.0):
                    break
                time.sleep(0.4)
            else:
                raise RuntimeError("等待 Freqtrade 本地 API 超时")

            if self.start_cancel_event.is_set():
                raise LaunchCancelled("启动已由用户取消")
            running_config = api.get("/show_config")
            if not isinstance(running_config, dict):
                raise RuntimeError("Freqtrade 运行配置响应格式错误")
            config_errors = validate_running_config(
                running_config,
                mode,
            )
            if config_errors:
                raise RuntimeError("；".join(config_errors))

            balance = api.get("/balance")
            status = api.get("/status")
            profit = api.get("/profit")
            if not isinstance(balance, dict) or not isinstance(status, list):
                raise RuntimeError("账户或持仓预检响应格式错误")
            if mode == MODE_LIVE:
                synchronization = synchronize_manual_exchange_changes(
                    api,
                    status,
                    balance,
                    runtime_running=False,
                )
                status = synchronization["status"]
                balance = synchronization["balance"]
                if synchronization.get("profit") is not None:
                    profit = synchronization["profit"]
                startup_manual_records = list(synchronization["records"])
            account = extract_account_snapshot(balance)
            if mode == MODE_LIVE and float(account["equity"]) <= 0:
                raise RuntimeError("主网账户权益为 0 或无法读取，已阻止 LIVE 启动")
            if mode == MODE_LIVE and account["unmanaged_positions"]:
                symbols = ", ".join(
                    str(row.get("currency") or "?")
                    for row in account["unmanaged_positions"]
                )
                raise RuntimeError(
                    "检测到不属于当前 Freqtrade 账本的主网持仓，"
                    f"为避免重复开仓已阻止启动：{symbols}"
                )
            if mode == MODE_LIVE:
                reconciliation_errors = validate_position_reconciliation(
                    status,
                    account["exchange_positions"],
                )
                if reconciliation_errors:
                    raise RuntimeError(
                        "主网持仓与独立 Freqtrade 账本对账失败："
                        + "；".join(reconciliation_errors)
                    )

            if self.start_cancel_event.is_set():
                raise LaunchCancelled("启动已由用户取消")
            start_response = api.post("/start")
            trading_started = True
            if not isinstance(start_response, dict):
                raise RuntimeError("Freqtrade 启动响应格式错误")
            running_deadline = time.monotonic() + 12.0
            while time.monotonic() < running_deadline:
                if self.start_cancel_event.is_set():
                    try:
                        api.post("/stop")
                    finally:
                        trading_started = False
                    raise LaunchCancelled("启动已由用户取消")
                active_config = api.get("/show_config")
                if str(active_config.get("state") or "").lower() == "running":
                    running_config = active_config
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Freqtrade 未能进入 RUNNING 状态")
            if self.start_cancel_event.is_set():
                try:
                    api.post("/stop")
                finally:
                    trading_started = False
                raise LaunchCancelled("启动已由用户取消")
            self.events.put(
                (
                    "engine_ready",
                    {
                        "pid": process.pid,
                        "mode": mode,
                        "strategy_class": spec.strategy_class,
                        "max_open_trades": spec.max_open_trades,
                        "balance": balance,
                        "status": status,
                        "profit": profit,
                        "show_config": running_config,
                        "start_response": start_response,
                        "manual_records": startup_manual_records,
                    },
                )
            )
        except LaunchCancelled as exc:
            if trading_started and api is not None:
                try:
                    api.post("/stop")
                except Exception:
                    pass
            self.events.put(
                (
                    "startup_cancelled",
                    {
                        "message": str(exc),
                        "had_process": process is not None,
                    },
                )
            )
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    if mode == MODE_DRY or not trading_started:
                        try:
                            process.kill()
                            process.wait(timeout=5)
                        except Exception:
                            pass
            if (
                process is not None
                and not output_reader_started
                and process.poll() is not None
            ):
                if engine_lock_acquired:
                    release_pid_lock(ENGINE_LOCK_PATH, pid=process.pid)
                if spec is not None:
                    try:
                        spec.overlay_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            elif process is None and spec is not None:
                try:
                    spec.overlay_path.unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as exc:
            if trading_started and api is not None:
                try:
                    api.post("/stop")
                except Exception:
                    pass
            safe_message = redact_sensitive(
                str(exc),
                spec.secret_bundle if spec else None,
            )
            self.events.put(("startup_failed", safe_message))
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except Exception:
                    if mode == MODE_DRY or not trading_started:
                        try:
                            process.kill()
                            process.wait(timeout=5)
                        except Exception:
                            pass
            if (
                process is not None
                and not output_reader_started
                and process.poll() is not None
            ):
                if engine_lock_acquired:
                    release_pid_lock(ENGINE_LOCK_PATH, pid=process.pid)
                if spec is not None:
                    try:
                        spec.overlay_path.unlink(missing_ok=True)
                    except OSError:
                        pass
            elif process is None and spec is not None:
                try:
                    spec.overlay_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _read_process_output(
        self,
        process: subprocess.Popen[str],
        spec: LaunchSpec,
    ) -> None:
        reducer = EngineOutputReducer()
        stream = process.stdout
        if stream is not None:
            for line in iter(stream.readline, ""):
                cleaned = ANSI_ESCAPE.sub("", line).rstrip()
                reduced = reducer.reduce(cleaned)
                if reduced is not None:
                    message, level = reduced
                    if level == "clock_error":
                        self.events.put(
                            (
                                "runtime_clock_error",
                                {
                                    "pid": process.pid,
                                    "message": message,
                                },
                            )
                        )
                        continue
                    self.events.put(
                        ("log", (message, level, spec.secret_bundle))
                    )
            stream.close()
        return_code = process.wait()
        release_pid_lock(ENGINE_LOCK_PATH, pid=process.pid)
        self.events.put(
            (
                "process_exit",
                {
                    "pid": process.pid,
                    "returncode": return_code,
                    "overlay_path": spec.overlay_path,
                },
            )
        )

    def _on_refresh(self) -> None:
        with self.process_lock:
            no_local_process = (
                self.process is None or self.process.poll() is not None
            )
        if self.engine_state == "LOCKED" and no_local_process:
            active_engine = active_pid_lock(ENGINE_LOCK_PATH)
            report = verify_release()
            if active_engine is None and report.ok:
                self._set_engine_state("OFF", "安全锁已重新检查")
                self._log(
                    "未再检测到残留引擎，版本校验正常；启动锁已解除。",
                    "success",
                )
        reset_baseline = self.mode.get() == MODE_LIVE
        with self.process_lock:
            process = self.process
            api = self.api
        if process is not None and process.poll() is None and api is not None:
            self.refresh_button.configure(state="disabled")
            self._log("正在刷新 Freqtrade 账户、持仓和交易统计…", "info")
            self._start_poll(reset_baseline=reset_baseline, manual=True)
            return
        if self.mode.get() == MODE_DRY:
            self.session_baseline_equity = DEFAULT_DRY_WALLET
            self._apply_offline_mode_defaults()
            self._log("DRY-RUN 模拟账户已重置显示为 200.00U。", "success")
            return
        bundle = load_exchange_secrets()
        if not bundle.ready:
            self._refresh_key_indicator()
            messagebox.showerror(
                "账户刷新失败",
                "原 .env 中没有找到完整的 Binance Futures API Key/Secret。",
                parent=self.root,
            )
            return
        self.refresh_button.configure(state="disabled")
        self._log(
            "正在执行 Binance 主网只读账户检查；不会发送或修改订单。",
            "info",
        )
        threading.Thread(
            target=self._direct_live_refresh_worker,
            args=(bundle,),
            name="v16-grid-v15-live-readonly",
            daemon=True,
        ).start()

    def _direct_live_refresh_worker(self, bundle: Any) -> None:
        exchange = None
        try:
            import ccxt  # Imported only in the bundled Freqtrade runtime.

            clock_sync = measure_binance_clock()
            exchange = ccxt.binanceusdm(
                {
                    "apiKey": bundle.key,
                    "secret": bundle.secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "future",
                        "adjustForTimeDifference": False,
                        "timeDifference": clock_sync.time_difference_ms,
                    },
                }
            )
            balance = exchange.fetch_balance({"type": "future"})
            info = balance.get("info") or {}
            usdt = balance.get("USDT") or {}
            equity = float(
                info.get("totalMarginBalance")
                or info.get("totalWalletBalance")
                or usdt.get("total")
                or 0.0
            )
            available = float(
                info.get("availableBalance")
                or usdt.get("free")
                or 0.0
            )
            if equity <= 0:
                raise RuntimeError("主网账户权益为 0 或响应格式无法识别")
            positions: list[dict[str, Any]] = []
            for item in exchange.fetch_positions():
                contracts = float(item.get("contracts") or 0.0)
                if abs(contracts) <= 0:
                    continue
                positions.append(
                    {
                        "pair": str(item.get("symbol") or ""),
                        "component": "交易所持仓",
                        "side": str(item.get("side") or "").lower(),
                        "leverage": float(item.get("leverage") or 0.0),
                        "stake_amount": float(item.get("initialMargin") or 0.0),
                        "open_rate": float(item.get("entryPrice") or 0.0),
                        "current_rate": float(item.get("markPrice") or 0.0),
                        "profit_abs": float(item.get("unrealizedPnl") or 0.0),
                        "profit_pct": float(item.get("percentage") or 0.0),
                        "open_date": "",
                        "custody": "外部/待对账",
                    }
                )
            historical_profit = read_ledger_realized_profit()
            position_profit = sum(
                _finite_float(position.get("profit_abs"))
                for position in positions
            )
            self.events.put(
                (
                    "direct_snapshot",
                    {
                        "equity": equity,
                        "available": available,
                        "stake": "USDT",
                        "positions": positions,
                        "historical_profit": historical_profit,
                        "position_profit": position_profit,
                        "total_profit": historical_profit + position_profit,
                        "clock_offset_ms": clock_sync.local_minus_server_ms,
                    },
                )
            )
        except Exception as exc:
            self.events.put(
                (
                    "refresh_failed",
                    friendly_runtime_error(
                        redact_sensitive(str(exc), bundle)
                    ),
                )
            )
        finally:
            if exchange is not None:
                try:
                    exchange.close()
                except Exception:
                    pass

    def _start_poll(
        self,
        *,
        reset_baseline: bool = False,
        manual: bool = False,
    ) -> None:
        if self.poll_in_flight:
            if manual:
                self.refresh_button.configure(state="normal")
            return
        with self.process_lock:
            process = self.process
            api = self.api
            spec = self.launch_spec
        if process is None or process.poll() is not None or api is None:
            if manual:
                self.refresh_button.configure(state="normal")
            return
        self.poll_in_flight = True
        threading.Thread(
            target=self._poll_worker,
            args=(
                process.pid,
                api,
                reset_baseline,
                manual,
                bool(spec is not None and spec.mode == MODE_LIVE),
            ),
            name="v16-grid-v15-poll",
            daemon=True,
        ).start()

    def _poll_worker(
        self,
        pid: int,
        api: FreqtradeApiClient,
        reset_baseline: bool,
        manual: bool,
        live_mode: bool,
    ) -> None:
        try:
            status = api.get("/status")
            balance = api.get("/balance")
            profit = api.get("/profit")
            health = api.get("/health")
            if not isinstance(status, list) or not isinstance(balance, dict):
                raise RuntimeError("账户轮询响应格式错误")
            manual_records: list[dict[str, Any]] = []
            closed_trade_history: list[dict[str, Any]] = []
            if live_mode:
                try:
                    synchronization = synchronize_manual_exchange_changes(
                        api,
                        status,
                        balance,
                        runtime_running=True,
                    )
                except ManualReconciliationError as exc:
                    self.events.put(
                        (
                            "reconciliation_failed",
                            {
                                "pid": pid,
                                "message": str(exc),
                                "trading_paused": exc.trading_paused,
                                "manual": manual,
                            },
                        )
                    )
                    return
                status = synchronization["status"]
                balance = synchronization["balance"]
                if synchronization.get("profit") is not None:
                    profit = synchronization["profit"]
                manual_records = list(synchronization["records"])
                history = api.get("/trades?limit=100&order_by_id=false")
                if not isinstance(history, dict) or not isinstance(
                    history.get("trades"),
                    list,
                ):
                    raise RuntimeError("人工平仓历史扫描响应格式错误")
                closed_trade_history = list(history["trades"])
            self.events.put(
                (
                    "snapshot",
                    {
                        "pid": pid,
                        "status": status,
                        "balance": balance,
                        "profit": profit,
                        "health": health,
                        "reset_baseline": reset_baseline,
                        "manual": manual,
                        "manual_records": manual_records,
                        "closed_trade_history": closed_trade_history,
                    },
                )
            )
        except Exception as exc:
            self.events.put(
                (
                    "poll_failed",
                    {
                        "message": friendly_runtime_error(str(exc)),
                        "manual": manual,
                    },
                )
            )

    def _poll_scheduler(self) -> None:
        if self.root.winfo_exists():
            with self.process_lock:
                active = (
                    self.process is not None
                    and self.process.poll() is None
                    and self.api is not None
                )
            if (
                active
                and self.engine_state == "RUNNING"
                and time.monotonic() >= self.next_poll_at
            ):
                self.next_poll_at = (
                    time.monotonic() + self.POLL_INTERVAL_MS / 1000.0
                )
                self._start_poll()
            self.root.after(500, self._poll_scheduler)

    def _on_stop(self) -> bool:
        if self.engine_state not in {"STARTING", "RUNNING", "LOCKED"}:
            return False
        if self.engine_state == "STARTING":
            self.start_cancel_event.set()
            self.intentional_stop = True
            self._set_engine_state("STOPPING", "正在取消安全预检")
            self._log("已请求取消启动；策略尚未进入 RUNNING。", "warning")
            return True
        with self.process_lock:
            process = self.process
            api = self.api
        if process is None or process.poll() is not None:
            self._finish_stopped_state()
            return True

        positions = self.last_snapshot.get("status") or []
        exchange_positions = (
            self.last_snapshot.get("account", {}).get("exchange_positions")
            or []
        )
        total_positions = max(len(positions), len(exchange_positions))
        if self.mode.get() == MODE_LIVE:
            if total_positions:
                dialog = TypedConfirmationDialog(
                    self.root,
                    title="确认停止 LIVE",
                    heading="当前仍有实盘持仓",
                    body=(
                        f"检测到 {total_positions} 个主网持仓，其中 "
                        f"{len(positions)} 个由当前 Freqtrade 账本管理。\n\n"
                        "交易所侧硬止损会继续存在，但1分钟软确认、盈利保护、"
                        "Grid管理及其他动态退出都会停止。停止进程不会自动平仓，"
                        "请确认你将立即人工接管这些仓位。"
                    ),
                    phrase="STOP LIVE",
                    confirm_text="停止并人工接管",
                    accent=RED,
                )
                if not dialog.result:
                    self._log("已取消停止，LIVE 继续运行。", "info")
                    return False
            elif not messagebox.askyesno(
                "停止 LIVE",
                "确认停止 LIVE 交易引擎？当前未检测到托管持仓。",
                parent=self.root,
                default="no",
            ):
                return False
        self.intentional_stop = True
        self._set_engine_state("STOPPING", "正在安全停止")
        self._log("已请求停止；正在先关闭策略状态，再终止进程。", "warning")
        threading.Thread(
            target=self._stop_worker,
            args=(process, api, self.mode.get()),
            name="v16-grid-v15-stop",
            daemon=True,
        ).start()
        return True

    def _stop_worker(
        self,
        process: subprocess.Popen[str],
        api: FreqtradeApiClient | None,
        mode: str,
    ) -> None:
        try:
            if api is not None:
                try:
                    api.post("/stop")
                except Exception as exc:
                    self.events.put(
                        ("log", (f"停止 API 返回异常：{exc}", "warning", None))
                    )
            time.sleep(1.0)
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if mode == MODE_DRY:
                    process.kill()
                    process.wait(timeout=5)
                else:
                    self.events.put(
                        (
                            "stop_failed",
                            "LIVE 进程未在 15 秒内退出；为避免强杀导致状态损坏，"
                            "未执行 kill，请检查日志。",
                        )
                    )
        except Exception as exc:
            self.events.put(("stop_failed", str(exc)))

    def _on_close(self) -> None:
        with self.process_lock:
            active = self.process is not None and self.process.poll() is None
        if not active and self.engine_state != "STARTING":
            self.root.destroy()
            return
        if not messagebox.askyesno(
            "退出控制台",
            "交易引擎仍在运行。是否先安全停止引擎再退出？",
            parent=self.root,
            default="no",
        ):
            return
        self.closing_after_stop = True
        if not self._on_stop():
            self.closing_after_stop = False

    def _drain_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle_event(event, payload)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self._drain_events)

    def _remember_manual_record(self, record: dict[str, Any]) -> None:
        try:
            trade_id = int(record.get("trade_id") or 0)
        except (TypeError, ValueError):
            trade_id = 0
        if trade_id > 0:
            self.known_manual_trade_ids.add(trade_id)
        for order in record.get("exchange_orders") or []:
            order_id = str(order.get("order_id") or "")
            if order_id:
                self.known_manual_order_ids.add(order_id)

    def _pause_after_audit_failure(
        self,
        api: FreqtradeApiClient,
        pid: int,
        message: str,
    ) -> None:
        paused = False
        try:
            api.post("/pause")
            paused = True
        except Exception:
            try:
                api.post("/stop")
                paused = True
            except Exception:
                pass
        self.events.put(
            (
                "reconciliation_failed",
                {
                    "pid": pid,
                    "message": message,
                    "trading_paused": paused,
                    "manual": False,
                },
            )
        )

    def _record_observed_manual_events(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        if not records:
            return
        try:
            append_manual_reconciliation_audit(records)
        except Exception as exc:
            with self.process_lock:
                process = self.process
                api = self.api
            if process is not None and process.poll() is None and api is not None:
                threading.Thread(
                    target=self._pause_after_audit_failure,
                    args=(
                        api,
                        process.pid,
                        f"人工平仓已回写，但审计日志无法落盘：{exc}",
                    ),
                    name="v16-grid-v15-audit-failsafe",
                    daemon=True,
                ).start()
            else:
                self._log(f"人工平仓审计日志无法落盘：{exc}", "error")
            return
        for record in records:
            self._remember_manual_record(record)
            self._log(_manual_reconciliation_message(record), "success")

    def _handle_event(self, event: str, payload: Any) -> None:
        if event == "log":
            message, level, bundle = payload
            self._log(message, level, bundle=bundle)
            return
        if event == "engine_ready":
            with self.process_lock:
                current_pid = self.process.pid if self.process else None
            if (
                payload.get("pid") != current_pid
                or self.start_cancel_event.is_set()
                or self.engine_state == "STOPPING"
            ):
                return
            self._set_engine_state("RUNNING", "策略运行中 · 10秒刷新")
            self.poll_failure_count = 0
            self.next_poll_at = time.monotonic() + 1.0
            self._apply_api_snapshot(payload, reset_baseline=False)
            for record in payload.get("manual_records") or []:
                self._remember_manual_record(record)
                self._log(_manual_reconciliation_message(record), "success")
            response = payload.get("start_response") or {}
            result_text = response.get("status") or response.get("result") or "running"
            self._log(
                f"{RELEASE_LABEL}"
                + f" {payload['mode']} 已启动：{result_text}",
                "success" if payload["mode"] == MODE_DRY else "warning",
            )
            config = payload.get("show_config") or {}
            if not bool(config.get("stoploss_on_exchange")):
                self._log(
                    "警告：运行配置未确认交易所硬止损；不要在此状态下托管实盘仓位。",
                    "warning",
                )
            else:
                self._log(
                    "交易所 STOP_MARKET 硬止损已启用；1分钟软确认和动态盈利保护仍需进程在线。",
                    "success",
                )
            return
        if event == "startup_failed":
            self._set_engine_state("LOCKED", "安全预检未通过")
            self._log(f"启动被锁定：{payload}", "error")
            messagebox.showerror(
                "启动被锁定",
                str(payload),
                parent=self.root,
            )
            return
        if event == "startup_cancelled":
            self._log(
                str(payload.get("message") or "启动已取消"),
                "success",
            )
            if not payload.get("had_process"):
                self._finish_stopped_state()
                if self.closing_after_stop:
                    self.root.after(150, self.root.destroy)
            return
        if event == "process_exit":
            with self.process_lock:
                current_pid = self.process.pid if self.process else None
            if current_pid is not None and payload.get("pid") != current_pid:
                return
            overlay = Path(payload.get("overlay_path", ""))
            try:
                if overlay.is_file():
                    overlay.unlink()
            except OSError as exc:
                self._log(f"临时运行配置清理失败：{exc}", "warning")
            return_code = payload.get("returncode")
            expected = self.intentional_stop or self.engine_state == "STOPPING"
            with self.process_lock:
                self.process = None
                self.launch_spec = None
                self.api = None
            self.poll_in_flight = False
            self.poll_failure_count = 0
            if expected:
                self._finish_stopped_state()
                self._log(
                    f"交易引擎已停止，进程返回码 {return_code}。",
                    "success",
                )
                if self.closing_after_stop:
                    self.root.after(150, self.root.destroy)
            else:
                self._set_engine_state("LOCKED", f"进程异常退出 · {return_code}")
                self._log(
                    f"Freqtrade 进程意外退出，返回码 {return_code}；已锁定自动重启。",
                    "error",
                )
            return
        if event == "snapshot":
            self.poll_in_flight = False
            self.poll_failure_count = 0
            self.next_poll_at = (
                time.monotonic() + self.POLL_INTERVAL_MS / 1000.0
            )
            self.refresh_button.configure(
                state="normal" if self.engine_state == "RUNNING" else "disabled"
            )
            with self.process_lock:
                current_pid = self.process.pid if self.process else None
            if payload.get("pid") != current_pid:
                return
            self.last_poll_error = ""
            self.status_detail_var.set("策略运行中 · 10秒刷新")
            self.status_dot.configure(
                fg=GREEN if self.mode.get() == MODE_DRY else RED
            )
            self._apply_api_snapshot(
                payload,
                reset_baseline=bool(payload.get("reset_baseline")),
            )
            for record in payload.get("manual_records") or []:
                self._remember_manual_record(record)
                self._log(_manual_reconciliation_message(record), "success")
            observed_records = discover_external_manual_records(
                list(payload.get("status") or []),
                list(payload.get("closed_trade_history") or []),
                known_trade_ids=self.known_manual_trade_ids,
                known_order_ids=self.known_manual_order_ids,
            )
            self._record_observed_manual_events(observed_records)
            if payload.get("manual"):
                self._log("账户、持仓和交易统计已刷新。", "success")
            return
        if event == "reconciliation_failed":
            self.poll_in_flight = False
            self.refresh_button.configure(state="normal")
            with self.process_lock:
                current_pid = self.process.pid if self.process else None
            if payload.get("pid") != current_pid:
                return
            safety_state = (
                "交易已暂停"
                if payload.get("trading_paused")
                else "无法确认交易已暂停"
            )
            self._set_engine_state("LOCKED", f"人工平仓对账失败 · {safety_state}")
            message = str(payload.get("message") or "未知对账错误")
            self._log(
                f"检测到主网仓位与账本变化，但无法用交易所成交记录安全回写："
                f"{message}；{safety_state}。请停止后重新启动进行只读预检，"
                "不要重复下单。",
                "error",
            )
            messagebox.showerror(
                "人工平仓对账已锁定",
                f"{message}\n\n{safety_state}。请先停止引擎，再重新启动完成预检。",
                parent=self.root,
            )
            return
        if event == "poll_failed":
            self.poll_in_flight = False
            if self.engine_state == "RUNNING":
                self.refresh_button.configure(state="normal")
            message = str(payload.get("message") or "未知错误")
            now = time.monotonic()
            self.poll_failure_count += 1
            retry_seconds = poll_backoff_seconds(self.poll_failure_count)
            self.next_poll_at = now + retry_seconds
            self.status_detail_var.set(
                f"账户刷新退避中 · {int(retry_seconds)}秒后重试"
            )
            self.status_dot.configure(fg=AMBER)
            if (
                message != self.last_poll_error
                or now - self.last_poll_error_at >= 60.0
                or payload.get("manual")
            ):
                self._log(
                    f"状态刷新失败：{message}；"
                    f"将在 {int(retry_seconds)} 秒后重试。",
                    "warning",
                )
                self.last_poll_error = message
                self.last_poll_error_at = now
            return
        if event == "runtime_clock_error":
            with self.process_lock:
                current_pid = self.process.pid if self.process else None
            if payload.get("pid") != current_pid:
                return
            self.poll_failure_count = max(self.poll_failure_count, 3)
            retry_seconds = poll_backoff_seconds(self.poll_failure_count)
            self.next_poll_at = time.monotonic() + retry_seconds
            self.status_detail_var.set(
                f"交易所时间异常 · {int(retry_seconds)}秒冷却"
            )
            self.status_dot.configure(fg=AMBER)
            self._log(str(payload.get("message") or "Binance 时间校验异常"), "error")
            return
        if event == "direct_snapshot":
            self.refresh_button.configure(state="normal")
            equity = float(payload.get("equity") or 0.0)
            self.equity_var.set(f"{equity:,.2f} U")
            self.available_var.set(
                f"{float(payload.get('available') or 0.0):,.2f} U"
            )
            historical_profit = _finite_float(
                payload.get("historical_profit")
            )
            position_profit = _finite_float(payload.get("position_profit"))
            total_profit = _finite_float(
                payload.get("total_profit"),
                historical_profit + position_profit,
            )
            self.session_pnl_var.set(f"{total_profit:+,.2f} U")
            self.pnl_hint.set(
                f"历史 {historical_profit:+,.2f}U · "
                f"持仓 {position_profit:+,.2f}U"
            )
            self.metric_cards[2].value_label.configure(
                fg=GREEN if total_profit >= 0 else RED
            )
            positions = payload.get("positions") or []
            max_open_trades = self._selected_max_open_trades()
            self.position_count_var.set(
                f"{len(positions)} / {max_open_trades}"
            )
            self.position_status_var.set(
                "主网只读检查 · 尚未由本 GUI 托管"
                if positions
                else "主网只读检查 · 当前空仓"
            )
            self._render_positions([], positions)
            clock_offset = int(payload.get("clock_offset_ms") or 0)
            self._log(
                f"LIVE 账户只读刷新完成：权益={equity:,.2f}U，"
                f"持仓={len(positions)}/{max_open_trades}，"
                f"累计盈亏={total_profit:+,.2f}U"
                f"（历史={historical_profit:+,.2f}U，"
                f"持仓={position_profit:+,.2f}U）；"
                f"时间偏差={clock_offset:+d}ms。",
                "success",
            )
            return
        if event == "refresh_failed":
            self.refresh_button.configure(state="normal")
            self._log(f"LIVE 账户只读刷新失败：{payload}", "error")
            messagebox.showerror(
                "账户刷新失败",
                str(payload),
                parent=self.root,
            )
            return
        if event == "stop_failed":
            self._set_engine_state("LOCKED", "停止流程需要人工检查")
            self._log(str(payload), "error")
            messagebox.showerror(
                "停止流程异常",
                str(payload),
                parent=self.root,
            )

    def _finish_stopped_state(self) -> None:
        self.intentional_stop = False
        self._set_engine_state("OFF", "已安全停止")
        if self.mode.get() == MODE_DRY:
            self.session_baseline_equity = DEFAULT_DRY_WALLET
        self.poll_in_flight = False
        self.poll_failure_count = 0

    def _apply_api_snapshot(
        self,
        payload: dict[str, Any],
        *,
        reset_baseline: bool,
    ) -> None:
        balance = payload.get("balance") or {}
        status = payload.get("status") or []
        profit = payload.get("profit") or {}
        account = extract_account_snapshot(balance)
        equity = float(account["equity"])
        historical_profit, position_profit, total_profit = (
            strategy_profit_breakdown(profit, status)
        )

        self.equity_var.set(f"{equity:,.2f} U")
        self.available_var.set(f"{float(account['available']):,.2f} U")
        self.session_pnl_var.set(f"{total_profit:+,.2f} U")
        self.pnl_hint.set(
            f"历史 {historical_profit:+,.2f}U · "
            f"持仓 {position_profit:+,.2f}U"
        )
        pnl_color = GREEN if total_profit >= 0 else RED
        self.metric_cards[2].value_label.configure(fg=pnl_color)

        exchange_positions = account.get("exchange_positions") or []
        position_count = max(len(status), len(exchange_positions))
        self.position_count_var.set(
            f"{position_count} / {self._selected_max_open_trades()}"
        )
        self.position_status_var.set(
            "当前空仓"
            if position_count == 0
            else f"Freqtrade 托管 {len(status)} · 交易所识别 {len(exchange_positions)}"
        )
        self.closed_count_var.set(str(int(profit.get("closed_trade_count") or 0)))
        profit_factor = profit.get("profit_factor")
        winrate = profit.get("winrate")
        if profit_factor is None:
            self.quality_var.set("PF —")
        else:
            self.quality_var.set(f"PF {float(profit_factor):.3f}")
        if winrate is None:
            self.quality_hint.set("胜率 —")
        else:
            winrate_value = float(winrate)
            if winrate_value <= 1.0:
                winrate_value *= 100.0
            self.quality_hint.set(f"胜率 {winrate_value:.2f}%")

        self.last_snapshot = {
            "status": status,
            "balance": balance,
            "profit": profit,
            "account": account,
        }
        self._render_positions(status, exchange_positions)
        if reset_baseline and self.mode.get() == MODE_LIVE:
            self._log(
                "LIVE 手动刷新：累计盈亏已按“历史已实现 + 当前持仓”更新，"
                "不会因充值、提现或刷新而重置。",
                "success",
            )

    @staticmethod
    def _pair_key(value: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", value.upper()).replace("USDTUSDT", "USDT")

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if number == 0:
            return "—"
        if abs(number) >= 1000:
            return f"{number:,.2f}"
        if abs(number) >= 1:
            return f"{number:,.4f}"
        return f"{number:.7f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_duration(open_date: str | None) -> str:
        if not open_date:
            return "—"
        try:
            value = str(open_date).replace("Z", "+00:00")
            started = datetime.fromisoformat(value)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            seconds = max(
                0,
                int((datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()),
            )
            hours, remainder = divmod(seconds, 3600)
            minutes = remainder // 60
            if hours >= 24:
                days, hours = divmod(hours, 24)
                return f"{days}天 {hours}时"
            return f"{hours}时 {minutes}分"
        except (ValueError, TypeError):
            return "—"

    def _render_positions(
        self,
        managed: list[dict[str, Any]],
        exchange_positions: list[dict[str, Any]],
    ) -> None:
        for item in self.positions.get_children():
            self.positions.delete(item)

        seen: set[str] = set()
        for trade in managed:
            pair = str(trade.get("pair") or "")
            seen.add(self._pair_key(pair))
            pnl_abs = _finite_float(trade.get("profit_abs"))
            values = (
                pair.replace("/USDT:USDT", ""),
                classify_component(trade.get("enter_tag")),
                "空" if trade.get("is_short") else "多",
                f"{float(trade.get('leverage') or 1.0):g}x",
                f"{float(trade.get('stake_amount') or 0.0):,.2f}U",
                self._format_price(trade.get("open_rate")),
                self._format_price(trade.get("current_rate")),
                format_profit_u(pnl_abs),
                self._format_duration(trade.get("open_date")),
                "Freqtrade",
            )
            tag = "profit" if pnl_abs > 0 else "loss" if pnl_abs < 0 else "neutral"
            self.positions.insert("", "end", values=values, tags=(tag,))

        for position in exchange_positions:
            pair = str(position.get("currency") or position.get("pair") or "")
            key = self._pair_key(pair)
            if key in seen:
                continue
            if "pair" in position:
                pnl_abs = _finite_float(position.get("profit_abs"))
                side = str(position.get("side") or "")
                values = (
                    pair.replace("/USDT:USDT", ""),
                    position.get("component") or "交易所持仓",
                    "空" if side == "short" else "多",
                    f"{float(position.get('leverage') or 0.0):g}x",
                    f"{float(position.get('stake_amount') or 0.0):,.2f}U",
                    self._format_price(position.get("open_rate")),
                    self._format_price(position.get("current_rate")),
                    format_profit_u(pnl_abs),
                    self._format_duration(position.get("open_date")),
                    position.get("custody") or "待对账",
                )
            else:
                side = str(position.get("side") or "")
                values = (
                    pair.replace("/USDT:USDT", ""),
                    "交易所持仓",
                    "空" if side == "short" else "多",
                    "—",
                    f"{float(position.get('est_stake') or 0.0):,.2f}U",
                    "—",
                    "—",
                    "—",
                    "—",
                    "已托管" if position.get("is_bot_managed") else "未托管",
                )
                pnl_abs = 0.0
            tag = "profit" if pnl_abs > 0 else "loss" if pnl_abs < 0 else "neutral"
            self.positions.insert("", "end", values=values, tags=(tag,))

    def _log(
        self,
        message: str,
        level: str = "info",
        *,
        bundle: Any = None,
    ) -> None:
        cleaned = ANSI_ESCAPE.sub("", redact_sensitive(str(message), bundle)).strip()
        if not cleaned:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {cleaned}"
        try:
            with self.gui_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n", level if level in {
            "info", "success", "warning", "error", "engine"
        } else "info")
        self.log_line_count += 1
        if self.log_line_count > 5_500:
            self.log_text.delete("1.0", "501.0")
            self.log_line_count -= 500
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _clear_visible_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log_line_count = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Breakout V16 + Grid V15 PF Precision Guard GUI"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅执行本地静态验收，不创建窗口、不连接交易所",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.check:
        return run_static_check()
    current_pid = os.getpid()
    try:
        acquire_pid_lock(
            GUI_LOCK_PATH,
            pid=current_pid,
            kind="Breakout V16 + Grid V15 PF Precision Guard GUI",
            details={"strategy": STRATEGY_CLASS},
        )
    except RuntimeError as exc:
        message = str(exc)
        print(message, file=sys.stderr)
        try:
            duplicate_root = tk.Tk()
            duplicate_root.withdraw()
            messagebox.showerror(
                "GUI 已在运行",
                message + "\n\n为防止重复下单，不会打开第二个控制台。",
                parent=duplicate_root,
            )
            duplicate_root.destroy()
        except tk.TclError:
            pass
        return 2

    try:
        root = tk.Tk()
        TradingConsole(root)
        root.mainloop()
        return 0
    finally:
        release_pid_lock(GUI_LOCK_PATH, pid=current_pid)


if __name__ == "__main__":
    sys.exit(main())
