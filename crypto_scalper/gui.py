from __future__ import annotations

import argparse
import queue
import re
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .binance_client import BinanceFuturesClient
from .binance_rate_limit import RequestWeightBudget
from .combined_volatility_trend_grid_shadow import (
    TREND_GRID_SHADOW_REASON_TOKEN,
    CombinedVolatilityTrendGridShadowTrader,
)
from .combined_hybrid_v5_grid_v3_backtest import COMBINED_V5_GRID_V3_NAME
from .combined_hybrid_v5_grid_v3_shadow import CombinedHybridV5GridV3ShadowTrader
from .combined_breakout_v7_grid_v5_shadow import (
    COMBINED_V7_GRID_V5_NAME,
    CombinedBreakoutV7GridV5ShadowTrader,
)
from .combined_breakout_v8_grid_v6_shadow import (
    COMBINED_V8_GRID_V6_NAME,
    CombinedBreakoutV8GridV6ShadowTrader,
)
from .combined_breakout_v8_grid_v6_live import (
    CombinedBreakoutV8GridV6LiveTrader,
)
from .combined_breakout_v9_grid_v7_shadow import (
    COMBINED_V9_GRID_V7_NAME,
    CombinedBreakoutV9GridV7ShadowTrader,
)
from .combined_breakout_v9_grid_v7_live import (
    CombinedBreakoutV9GridV7LiveTrader,
)
from .live_config import (
    DEFAULT_SYMBOLS,
    LiveAppConfig,
    default_live_config,
    load_live_config,
    write_live_config,
)
from .live_trader import AccountSnapshot, BinanceAutoTrader
from .mtf_4h_rsi_regime import MTF_REASON_TOKEN
from .mtf_momentum_reset import MTF_MOMENTUM_RESET_SETUP_TOKEN
from .secrets import mask_secret, read_secret
from .volatility_breakout_shadow import (
    DUAL_THRUST_SHADOW_REASON_TOKEN,
    DualThrustShadowTrader,
)


DEFAULT_CONFIG_PATH = "config.gui.mtf-momentum-reset-stage21.json"
ACTIVE_GUI_CONFIG_PATH = "config.gui.breakout-v9-grid-v7-max2-shadow.json"
LIVE_GUI_CONFIG_PATH = "config.gui.breakout-v9-grid-v7-max2-live.json"
FALLBACK_CONFIG_PATH = "config.live.example.json"
STRATEGY_MODE_INDICATOR = "指标反转稳定版"
STRATEGY_MODE_SUPER_VOLUME = "强放量突破"
STRATEGY_MODE_MTF = "MTF多周期"
STRATEGY_MODE_MTF_RESET = "MTF动量重置"
STRATEGY_MODE_OI_FLUSH = "OI去杠杆反弹"
STRATEGY_MODE_DUAL_THRUST_SHADOW = "Hybrid v5 50币 Shadow"
STRATEGY_MODE_COMBINED_SHADOW = (
    "Breakout v9 + Grid v7 50币 Max2 DRY-RUN"
)
STRATEGY_MODE_COMBINED_LIVE = (
    "Breakout v9 + Grid v7 50币 Max2 LIVE"
)
EXECUTION_MODE_DRY_RUN = "DRY-RUN"
EXECUTION_MODE_LIVE = "LIVE"
EXECUTION_MODE_VALUES = (
    EXECUTION_MODE_DRY_RUN,
    EXECUTION_MODE_LIVE,
)
V9_V7_DRY_RUN_DISPLAY_BASELINE_USDT = 200.0
STRATEGY_MODE_MANUAL = "手动配置"
STRATEGY_MODE_VALUES = (
    STRATEGY_MODE_COMBINED_SHADOW,
    STRATEGY_MODE_COMBINED_LIVE,
    STRATEGY_MODE_DUAL_THRUST_SHADOW,
    STRATEGY_MODE_INDICATOR,
    STRATEGY_MODE_SUPER_VOLUME,
    STRATEGY_MODE_MTF_RESET,
    STRATEGY_MODE_MTF,
    STRATEGY_MODE_OI_FLUSH,
    STRATEGY_MODE_MANUAL,
)
STRATEGY_MODE_SUMMARIES = {
    STRATEGY_MODE_COMBINED_SHADOW: "50币共享模拟账户：Breakout v9 共享平衡版 + Grid v7，各最多1仓、全局最多2仓；实时主网数据，资金与成交均为本地模拟。",
    STRATEGY_MODE_COMBINED_LIVE: "独立实盘执行：交易所成交为仓位真相，严格对账、幂等订单、reduceOnly退出、保护止损与熔断；使用与冻结回测一致的100%风险缩放。",
    STRATEGY_MODE_DUAL_THRUST_SHADOW: "Hybrid v5 Balanced Expansion Runner：50币、60m 多空、单仓、风险2.5%、8R止盈5%、60R主目标、full-cost；强制 dry-run。",
    STRATEGY_MODE_INDICATOR: "当前启用：indicator_reversal 多空分离版。20x/持仓4，risk 0.065，多头0.282/空头0.34，其它策略关闭。",
    STRATEGY_MODE_SUPER_VOLUME: "启用强放量突破策略；适合捕捉高量能趋势启动，旧突破/回踩/反转策略保持关闭。",
    STRATEGY_MODE_MTF_RESET: "MTF动量重置 Stage 2.1：1h 动量重置 + 30m 收缩释放，只做多、单仓、固定 2R、禁止加仓。",
    STRATEGY_MODE_MTF: "MTF Core V3 Rank 3.5：多空双向、单仓、固定 2R、禁止加仓；其余策略全部关闭。",
    STRATEGY_MODE_OI_FLUSH: "启用 OI 去杠杆反弹策略；只做多，旧策略关闭。",
    STRATEGY_MODE_MANUAL: "保留当前配置文件中的隐藏策略开关，只保存界面上可编辑的参数。",
}
THEME = {
    "root": "#0b1017",
    "header": "#111827",
    "panel": "#121a26",
    "card": "#172233",
    "card_alt": "#202c3f",
    "field": "#0a111c",
    "log": "#070b12",
    "border": "#2b3a52",
    "text": "#edf2f7",
    "muted": "#94a3b8",
    "soft": "#cbd5e1",
    "title": "#f8fafc",
    "accent": "#d97706",
    "accent_active": "#b45309",
    "accent_soft": "#2b2112",
    "accent_text": "#fde68a",
    "button": "#263244",
    "button_active": "#334155",
    "danger": "#dc2626",
    "danger_active": "#b91c1c",
    "live_soft": "#3a171d",
    "live_text": "#fecaca",
    "success_soft": "#102d28",
    "success_text": "#a7f3d0",
    "profit": "#22c55e",
    "loss": "#f43f5e",
    "info": "#38bdf8",
}


def _combined_shadow_trader_class(config: LiveAppConfig):
    if (
        config.combined_volatility_trend_grid_shadow.strategy_name
        == COMBINED_V9_GRID_V7_NAME
    ):
        return CombinedBreakoutV9GridV7ShadowTrader
    if (
        config.combined_volatility_trend_grid_shadow.strategy_name
        == COMBINED_V8_GRID_V6_NAME
    ):
        return CombinedBreakoutV8GridV6ShadowTrader
    if (
        config.combined_volatility_trend_grid_shadow.strategy_name
        == COMBINED_V7_GRID_V5_NAME
    ):
        return CombinedBreakoutV7GridV5ShadowTrader
    if (
        config.combined_volatility_trend_grid_shadow.strategy_name
        == COMBINED_V5_GRID_V3_NAME
    ):
        return CombinedHybridV5GridV3ShadowTrader
    return CombinedVolatilityTrendGridShadowTrader


def _combined_live_trader_class(config: LiveAppConfig):
    if (
        config.combined_breakout_v8_grid_v6_live.strategy_name
        == COMBINED_V9_GRID_V7_NAME
    ):
        return CombinedBreakoutV9GridV7LiveTrader
    return CombinedBreakoutV8GridV6LiveTrader


class TradingApp(tk.Tk):
    def __init__(self, initial_config_path: str = ACTIVE_GUI_CONFIG_PATH) -> None:
        super().__init__()
        self.title("V9 / V7 Trading Console")
        self.geometry("1280x700")
        self.minsize(1080, 620)
        self.configure(bg=THEME["root"])
        self._window_icon: tk.PhotoImage | None = None
        self._apply_window_icon()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.account_queue: queue.Queue[
            tuple[AccountSnapshot, bool, bool]
        ] = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.config_path = tk.StringVar(value=initial_config_path)
        self.summary_vars: dict[str, tk.StringVar] = {}
        self.summary_labels: dict[str, ttk.Label] = {}
        self.symbols_text: tk.Text | None = None
        self._last_config: LiveAppConfig | None = None
        self._live_display_equity_baseline: float | None = None
        self._log_file_path = Path("logs") / f"gui_{datetime.now().date().isoformat()}.log"

        self._build_vars()
        self._build_style()
        self._build_ui()
        self._load_initial_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._drain_queues)

    def _apply_window_icon(self) -> None:
        try:
            self._window_icon = _build_coin_icon()
            self.iconphoto(True, self._window_icon)
        except tk.TclError:
            self._window_icon = None

    def _build_vars(self) -> None:
        self.execution_mode = tk.StringVar(
            value=EXECUTION_MODE_DRY_RUN
        )
        self.execution_mode_summary = tk.StringVar()
        self.safety_note = tk.StringVar()
        self.environment = tk.StringVar()
        self.dry_run = tk.BooleanVar()
        self.api_key_env = tk.StringVar()
        self.api_secret_env = tk.StringVar()
        self.symbols = tk.StringVar()
        self.entry_symbols = tk.StringVar()
        self.timeframe = tk.StringVar()
        self.poll_seconds = tk.StringVar()
        self.entry_scan_seconds = tk.StringVar()
        self.symbol_reentry_cooldown_seconds = tk.StringVar()
        self.leverage = tk.StringVar()
        self.max_open_positions = tk.StringVar()
        self.max_new_entries_per_cycle = tk.StringVar()
        self.starting_capital = tk.StringVar()
        self.margin_usage = tk.StringVar()
        self.symbol_margin = tk.StringVar()
        self.min_symbol_margin = tk.StringVar()
        self.max_notional = tk.StringVar()
        self.risk_per_trade = tk.StringVar()
        self.daily_loss = tk.StringVar()
        self.max_drawdown = tk.StringVar()
        self.soft_drawdown_reduce = tk.StringVar()
        self.soft_drawdown_stop = tk.StringVar()
        self.soft_drawdown_min_size = tk.StringVar()
        self.estimated_fee_bps = tk.StringVar()
        self.estimated_slippage_bps = tk.StringVar()
        self.min_profit_after_cost = tk.StringVar()
        self.min_available = tk.StringVar()
        self.mainnet_confirmation = tk.StringVar()
        self.live_armed = tk.BooleanVar(value=False)
        self.live_confirmation = tk.StringVar()
        self.strategy_mode = tk.StringVar(value=STRATEGY_MODE_MTF_RESET)
        self.strategy_mode_summary = tk.StringVar(value=STRATEGY_MODE_SUMMARIES[STRATEGY_MODE_MTF_RESET])
        self.strategy_indicator_enabled = tk.BooleanVar(value=True)
        self.strategy_vbp_enabled = tk.BooleanVar(value=True)
        self.mtf_allow_long = tk.BooleanVar(value=True)
        self.mtf_allow_short = tk.BooleanVar(value=False)
        self.mtf_core_parameters = tk.StringVar()
        self.fast_ema = tk.StringVar()
        self.slow_ema = tk.StringVar()
        self.atr_period = tk.StringVar()
        self.channel_period = tk.StringVar()
        self.volume_period = tk.StringVar()
        self.min_atr_pct = tk.StringVar()
        self.max_atr_pct = tk.StringVar()
        self.min_volume_ratio = tk.StringVar()
        self.breakout_buffer = tk.StringVar()
        self.ema_gap = tk.StringVar()
        self.stop_loss_atr = tk.StringVar()
        self.take_profit_atr = tk.StringVar()
        self.breakeven_atr = tk.StringVar()
        self.trailing_activation_atr = tk.StringVar()
        self.trailing_stop_atr = tk.StringVar()
        self.max_holding_bars = tk.StringVar()
        self.allow_short = tk.BooleanVar()
        self.long_score_threshold = tk.StringVar()
        self.short_score_threshold = tk.StringVar()
        self.long_risk_bias = tk.StringVar()
        self.short_risk_bias = tk.StringVar()
        self.regime_filter_enabled = tk.BooleanVar()
        self.regime_lookback = tk.StringVar()
        self.long_min_slow_slope_atr = tk.StringVar()
        self.short_max_slow_slope_atr = tk.StringVar()
        self.spike_guard_enabled = tk.BooleanVar()
        self.spike_trade_enabled = tk.BooleanVar()
        self.rsi_reversal_enabled = tk.BooleanVar()
        self.spike_min_range_atr = tk.StringVar()
        self.spike_min_wick_atr = tk.StringVar()
        self.spike_min_wick_ratio = tk.StringVar()
        self.spike_min_volume_ratio = tk.StringVar()
        self.spike_block_bars = tk.StringVar()
        self.spike_recovery_ratio = tk.StringVar()
        self.spike_stop_atr = tk.StringVar()
        self.spike_take_profit_atr = tk.StringVar()
        self.spike_risk_multiplier = tk.StringVar()
        self.spike_max_holding_bars = tk.StringVar()
        self.initial_entry_fraction = tk.StringVar()
        self.scale_in_entry_fraction = tk.StringVar()
        self.max_scale_ins_per_symbol = tk.StringVar()
        self.scale_in_min_profit_pct = tk.StringVar()
        self.scale_in_cooldown_seconds = tk.StringVar()
        self.allow_loss_scale_in = tk.BooleanVar()
        self.loss_scale_in_trigger_pct = tk.StringVar()
        self.loss_scale_in_entry_fraction = tk.StringVar()
        self.macro_enabled = tk.BooleanVar()
        self.macro_events_path = tk.StringVar()
        self.macro_symbols = tk.StringVar()
        self.macro_primary_symbol = tk.StringVar()
        self.macro_event_types = tk.StringVar()
        self.macro_leverage = tk.StringVar()
        self.macro_pre_flatten_seconds = tk.StringVar()
        self.macro_post_lockout_seconds = tk.StringVar()
        self.macro_entry_delay_seconds = tk.StringVar()
        self.macro_margin_pct = tk.StringVar()
        self.macro_stop_loss_pct = tk.StringVar()
        self.macro_take_profit_pct = tk.StringVar()
        self.macro_max_holding_seconds = tk.StringVar()
        self.macro_nfp_min_surprise_k = tk.StringVar()
        self.macro_cpi_min_surprise_pct = tk.StringVar()

    def _build_style(self) -> None:
        c = THEME
        style = ttk.Style(self)
        style.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", c["field"])
        self.option_add("*TCombobox*Listbox.foreground", c["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", c["accent_active"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure(".", font=("Microsoft YaHei UI", 10), background=c["root"], foreground=c["text"])
        style.configure("Root.TFrame", background=c["root"])
        style.configure("Header.TFrame", background=c["header"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Card.TFrame", background=c["card"], borderwidth=1, relief=tk.SOLID)
        style.configure("Toolbar.TFrame", background=c["root"])
        style.configure("TLabel", background=c["panel"], foreground=c["soft"])
        style.configure("ToolbarLabel.TLabel", background=c["root"], foreground=c["soft"])
        style.configure("HeaderLabel.TLabel", background=c["header"], foreground=c["soft"])
        style.configure("Title.TLabel", background=c["root"], foreground=c["title"], font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("HeaderTitle.TLabel", background=c["header"], foreground=c["title"], font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("HeaderSubtitle.TLabel", background=c["header"], foreground=c["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("SectionTitle.TLabel", background=c["panel"], foreground=c["title"], font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Muted.TLabel", background=c["panel"], foreground=c["muted"])
        style.configure("CardTitle.TLabel", background=c["card"], foreground=c["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("CardSection.TLabel", background=c["card"], foreground=c["title"], font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("CardBody.TLabel", background=c["card"], foreground=c["muted"])
        style.configure("CardValue.TLabel", background=c["card"], foreground=c["title"], font=("Consolas", 17, "bold"))
        style.configure("Profit.CardValue.TLabel", background=c["card"], foreground=c["profit"], font=("Consolas", 17, "bold"))
        style.configure("Loss.CardValue.TLabel", background=c["card"], foreground=c["loss"], font=("Consolas", 17, "bold"))
        style.configure("Info.CardValue.TLabel", background=c["card"], foreground=c["info"], font=("Consolas", 17, "bold"))
        style.configure("Status.TLabel", background=c["accent_soft"], foreground=c["accent_text"], padding=(14, 8), font=("Microsoft YaHei UI", 10, "bold"))
        style.configure(
            "Dry.Status.TLabel",
            background=c["success_soft"],
            foreground=c["success_text"],
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Live.Status.TLabel",
            background=c["live_soft"],
            foreground=c["live_text"],
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "ModeValue.TLabel",
            background=c["card"],
            foreground=c["title"],
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "Pill.TLabel",
            background=c["accent_soft"],
            foreground=c["accent_text"],
            padding=(9, 4),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.configure(
            "DisabledPill.TLabel",
            background=c["card_alt"],
            foreground=c["muted"],
            padding=(9, 4),
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.configure("TButton", padding=(13, 7), background=c["button"], foreground=c["text"], bordercolor=c["border"], focusthickness=0)
        style.map("TButton", background=[("active", c["button_active"]), ("disabled", c["panel"])], foreground=[("disabled", c["muted"])])
        style.configure("Accent.TButton", background=c["accent"], foreground="#ffffff", bordercolor=c["accent"])
        style.map("Accent.TButton", background=[("active", c["accent_active"]), ("disabled", c["panel"])], foreground=[("disabled", c["muted"])])
        style.configure("Danger.TButton", background=c["danger"], foreground="#ffffff", bordercolor=c["danger"])
        style.map("Danger.TButton", background=[("active", c["danger_active"]), ("disabled", c["panel"])], foreground=[("disabled", c["muted"])])
        style.configure(
            "TEntry",
            fieldbackground=c["field"],
            foreground=c["text"],
            insertcolor=c["text"],
            bordercolor=c["border"],
            lightcolor=c["border"],
            darkcolor=c["border"],
            padding=(6, 4),
        )
        style.configure(
            "TCombobox",
            fieldbackground=c["field"],
            background=c["field"],
            foreground=c["text"],
            arrowcolor=c["muted"],
            bordercolor=c["border"],
            padding=(6, 4),
        )
        style.map("TCombobox", fieldbackground=[("readonly", c["field"])], foreground=[("readonly", c["text"])])
        style.configure("TCheckbutton", background=c["panel"], foreground=c["soft"])
        style.map("TCheckbutton", background=[("active", c["panel"])], foreground=[("active", c["title"])])
        style.configure("TNotebook", background=c["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["card_alt"], foreground=c["muted"], padding=(16, 8), bordercolor=c["border"])
        style.map("TNotebook.Tab", background=[("selected", c["accent"])], foreground=[("selected", "#ffffff")])
        style.configure("Treeview", background=c["field"], fieldbackground=c["field"], foreground=c["text"], rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background=c["card_alt"], foreground=c["title"], font=("Microsoft YaHei UI", 9, "bold"), bordercolor=c["border"])
        style.map("Treeview", background=[("selected", c["accent_active"])], foreground=[("selected", "#ffffff")])
        style.configure("Vertical.TScrollbar", background=c["button"], troughcolor=c["field"], bordercolor=c["border"], arrowcolor=c["muted"])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, style="Root.TFrame", padding=16)
        outer.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(
            outer, style="Header.TFrame", padding=(18, 14)
        )
        toolbar.pack(fill=tk.X)
        title_block = ttk.Frame(toolbar, style="Header.TFrame")
        title_block.pack(side=tk.LEFT)
        ttk.Label(
            title_block,
            text="V9 / V7 Trading Console",
            style="HeaderTitle.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            title_block,
            text="Breakout v9 共享平衡版 + Grid v7  ·  Binance Futures Mainnet",
            style="HeaderSubtitle.TLabel",
        ).pack(anchor=tk.W, pady=(3, 0))

        ttk.Button(
            toolbar,
            text="刷新账户",
            command=self.refresh_account,
        ).pack(side=tk.RIGHT)

        main = ttk.Frame(outer, style="Root.TFrame")
        main.pack(fill=tk.BOTH, expand=True, pady=(16, 0))
        main.columnconfigure(0, weight=0, minsize=320)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Panel.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)
        left.rowconfigure(2, weight=0)
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(main, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        settings_area = ttk.Frame(left, style="Panel.TFrame")
        settings_area.grid(row=0, column=0, sticky="nsew")
        self._build_simple_settings(settings_area)
        self._build_controls(left)
        self._build_dashboard(right)
        self._build_positions(right)
        self._build_log(right)

    def _build_simple_settings(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        ttk.Label(
            parent,
            text="运行设置",
            style="SectionTitle.TLabel",
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        mode_card = ttk.Frame(
            parent, style="Card.TFrame", padding=(14, 13)
        )
        mode_card.grid(row=1, column=0, sticky="ew")
        mode_card.columnconfigure(0, weight=1)
        ttk.Label(
            mode_card,
            text="环境 / 模式",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky=tk.W)
        mode_combo = ttk.Combobox(
            mode_card,
            textvariable=self.execution_mode,
            values=EXECUTION_MODE_VALUES,
            state="readonly",
            width=18,
        )
        mode_combo.grid(
            row=1, column=0, sticky="ew", pady=(8, 8)
        )
        mode_combo.bind(
            "<<ComboboxSelected>>",
            self._on_execution_mode_selected,
        )
        ttk.Label(
            mode_card,
            textvariable=self.execution_mode_summary,
            style="CardBody.TLabel",
            wraplength=270,
            justify=tk.LEFT,
        ).grid(row=2, column=0, sticky="ew")

        strategy_card = ttk.Frame(
            parent, style="Card.TFrame", padding=(14, 13)
        )
        strategy_card.grid(
            row=2, column=0, sticky="ew", pady=(12, 0)
        )
        strategy_card.columnconfigure(0, weight=1)
        ttk.Label(
            strategy_card,
            text="当前冻结策略",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            strategy_card,
            text="Breakout v9 共享平衡版 + Grid v7",
            style="ModeValue.TLabel",
        ).grid(row=1, column=0, sticky=tk.W, pady=(7, 9))
        pill_row = ttk.Frame(strategy_card, style="Card.TFrame")
        pill_row.grid(row=2, column=0, sticky=tk.W)
        ttk.Label(
            pill_row, text="50 币", style="Pill.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            pill_row, text="最多 2 仓", style="Pill.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(
            pill_row, text="10x", style="Pill.TLabel"
        ).pack(side=tk.LEFT)

        macro_card = ttk.Frame(
            parent, style="Card.TFrame", padding=(14, 13)
        )
        macro_card.grid(
            row=3, column=0, sticky="ew", pady=(12, 0)
        )
        macro_card.columnconfigure(0, weight=1)
        ttk.Label(
            macro_card,
            text="附加策略",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky=tk.W)
        macro_row = ttk.Frame(macro_card, style="Card.TFrame")
        macro_row.grid(row=1, column=0, sticky="ew", pady=(7, 4))
        macro_row.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            macro_row,
            text="非农策略",
            variable=self.macro_enabled,
            state=tk.DISABLED,
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            macro_row,
            text="待调试",
            style="DisabledPill.TLabel",
        ).grid(row=0, column=1, sticky=tk.E)
        ttk.Label(
            macro_card,
            text="当前不会启用，也不会参与 DRY-RUN 或 LIVE。",
            style="CardBody.TLabel",
            wraplength=270,
            justify=tk.LEFT,
        ).grid(row=2, column=0, sticky="ew")

        ttk.Label(
            parent,
            textvariable=self.safety_note,
            style="Muted.TLabel",
            wraplength=280,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky="ew", pady=(14, 0))

    def _build_scrollable_settings(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            parent,
            bg=THEME["panel"],
            borderwidth=0,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, style="Panel.TFrame")
        window_id = canvas.create_window((0, 0), window=content, anchor=tk.NW)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def sync_scroll_region(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def sync_content_width(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def scroll_with_wheel(event: tk.Event) -> None:
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            else:
                delta = int(-1 * (event.delta / 120))
                if delta:
                    canvas.yview_scroll(delta, "units")

        def bind_wheel(_event: tk.Event) -> None:
            canvas.bind_all("<MouseWheel>", scroll_with_wheel)
            canvas.bind_all("<Button-4>", scroll_with_wheel)
            canvas.bind_all("<Button-5>", scroll_with_wheel)

        def unbind_wheel(_event: tk.Event) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        content.bind("<Configure>", sync_scroll_region)
        canvas.bind("<Configure>", sync_content_width)
        canvas.bind("<Enter>", bind_wheel)
        content.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)
        content.bind("<Leave>", unbind_wheel)

        self._build_settings(content)

    def _build_settings(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        execution = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        risk = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        strategy = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        macro = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        advanced = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        notebook.add(execution, text="执行")
        notebook.add(risk, text="风控")
        notebook.add(strategy, text="策略")
        notebook.add(macro, text="非农")
        notebook.add(advanced, text="高级")

        self._combo(execution, "环境", self.environment, ("mainnet",), 0)
        ttk.Checkbutton(execution, text="Dry-run 不真实下单", variable=self.dry_run).grid(row=1, column=1, sticky=tk.W, pady=(4, 8))
        self._entry(execution, "周期", self.timeframe, 2)
        self._entry(execution, "持仓监控秒", self.poll_seconds, 3)
        self._entry(execution, "开仓扫描秒", self.entry_scan_seconds, 4)
        self._entry(execution, "单币冷却秒", self.symbol_reentry_cooldown_seconds, 5)
        self._entry(execution, "杠杆上限", self.leverage, 6)
        self._entry(execution, "最大持仓数", self.max_open_positions, 7)
        self._entry(execution, "每轮最多开仓", self.max_new_entries_per_cycle, 8)
        ttk.Label(execution, text="交易币种").grid(row=9, column=0, sticky=tk.NW, pady=5, padx=(0, 8))
        self.symbols_text = tk.Text(
            execution,
            height=4,
            width=28,
            wrap=tk.WORD,
            bg=THEME["field"],
            fg=THEME["text"],
            insertbackground=THEME["text"],
            selectbackground=THEME["accent_active"],
            selectforeground="#ffffff",
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            highlightthickness=1,
            relief=tk.FLAT,
            borderwidth=1,
            font=("Consolas", 10),
        )
        self.symbols_text.grid(row=9, column=1, sticky="ew", pady=5)
        ttk.Button(execution, text="填入主流币预设", command=self._apply_symbol_preset).grid(row=10, column=1, sticky="ew", pady=(2, 0))
        self._entry(execution, "开仓白名单", self.entry_symbols, 11)

        self._entry(risk, "本金U", self.starting_capital, 0)
        self._entry(risk, "总保证金上限", self.margin_usage, 1)
        self._entry(risk, "单币保证金上限", self.symbol_margin, 2)
        self._entry(risk, "单仓最低保证金", self.min_symbol_margin, 3)
        self._entry(risk, "单仓名义上限U", self.max_notional, 4)
        self._entry(risk, "单笔风险比例", self.risk_per_trade, 5)
        self._entry(risk, "日亏损上限", self.daily_loss, 6)
        self._entry(risk, "最大回撤", self.max_drawdown, 7)
        self._entry(risk, "软降仓回撤", self.soft_drawdown_reduce, 8)
        self._entry(risk, "强降仓回撤", self.soft_drawdown_stop, 9)
        self._entry(risk, "最低仓位系数", self.soft_drawdown_min_size, 10)
        self._entry(risk, "手续费bps", self.estimated_fee_bps, 11)
        self._entry(risk, "滑点bps", self.estimated_slippage_bps, 12)
        self._entry(risk, "最低净利", self.min_profit_after_cost, 13)
        self._entry(risk, "最低保留U", self.min_available, 14)

        self._build_strategy_selector(strategy)
        strategy_fields = ttk.Frame(strategy, style="Card.TFrame", padding=(12, 10))
        strategy_fields.grid(row=2, column=0, columnspan=4, sticky="ew")
        strategy_fields.columnconfigure(0, weight=1)
        ttk.Label(strategy_fields, text="当前策略参数", style="CardSection.TLabel").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            strategy_fields,
            textvariable=self.mtf_core_parameters,
            style="CardBody.TLabel",
            justify=tk.LEFT,
            wraplength=360,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ttk.Checkbutton(macro, text="启用非农策略", variable=self.macro_enabled).grid(row=0, column=1, sticky=tk.W, pady=(0, 8))
        self._entry(macro, "事件文件", self.macro_events_path, 1)
        self._entry(macro, "交易币列表", self.macro_symbols, 2)
        self._entry(macro, "主交易币", self.macro_primary_symbol, 3)
        self._entry(macro, "事件类型", self.macro_event_types, 4)
        self._entry(macro, "非农杠杆", self.macro_leverage, 5)
        self._entry(macro, "保证金比例", self.macro_margin_pct, 6)
        self._entry(macro, "发布前清仓秒", self.macro_pre_flatten_seconds, 7)
        self._entry(macro, "发布后锁定秒", self.macro_post_lockout_seconds, 8)
        self._entry(macro, "入场延迟秒", self.macro_entry_delay_seconds, 9)
        self._entry(macro, "止损比例", self.macro_stop_loss_pct, 10)
        self._entry(macro, "止盈比例", self.macro_take_profit_pct, 11)
        self._entry(macro, "最长持仓秒", self.macro_max_holding_seconds, 12)
        self._entry(macro, "NFP最小差值K", self.macro_nfp_min_surprise_k, 13)
        self._entry(macro, "CPI最小差值", self.macro_cpi_min_surprise_pct, 14)
        ttk.Button(macro, text="查看策略摘要", command=self.show_macro_strategy).grid(row=15, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(
            macro,
            text="启用后，非农窗口会先清理持仓，发布后只允许宏观事件临时仓，并在锁定时间内暂停普通开仓。",
            style="Muted.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=16, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self._entry(advanced, "API Key变量", self.api_key_env, 0)
        self._entry(advanced, "API Secret变量", self.api_secret_env, 1)
        self._entry(advanced, "主网确认文本", self.mainnet_confirmation, 2)
        ttk.Checkbutton(
            advanced,
            text="ARM v9/v7 LIVE（仅实盘配置）",
            variable=self.live_armed,
        ).grid(row=3, column=1, sticky=tk.W, pady=5)
        self._entry(
            advanced,
            "v9/v7实盘确认",
            self.live_confirmation,
            4,
        )
        self._entry(advanced, "首单比例", self.initial_entry_fraction, 5)
        self._entry(advanced, "盈利补仓比例", self.scale_in_entry_fraction, 6)
        self._entry(advanced, "盈利触发", self.scale_in_min_profit_pct, 7)
        self._entry(advanced, "最多补仓", self.max_scale_ins_per_symbol, 8)
        self._entry(advanced, "补仓冷却秒", self.scale_in_cooldown_seconds, 9)
        ttk.Checkbutton(advanced, text="允许亏损补仓", variable=self.allow_loss_scale_in).grid(row=10, column=1, sticky=tk.W, pady=5)
        self._entry(advanced, "亏损触发", self.loss_scale_in_trigger_pct, 11)
        self._entry(advanced, "亏损补仓比例", self.loss_scale_in_entry_fraction, 12)
        ttk.Label(
            advanced,
            text=(
                "v9/v7 实盘必须加载独立 LIVE 配置、使用 One-way 专用账户，"
                "同时 ARM，并填写 CONFIRM_MAINNET 与 "
                "CONFIRM_BREAKOUT_V9_GRID_V7_LIVE；点击启动后还会再次要求 RUN_LIVE_NOW。"
            ),
            style="Muted.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=13, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        self.start_button = ttk.Button(
            frame,
            text="启动 DRY-RUN",
            command=self.start_trader,
            style="Accent.TButton",
        )
        self.start_button.pack(fill=tk.X, pady=(0, 8))
        self.stop_button = ttk.Button(
            frame,
            text="停止运行",
            command=self.stop_trader,
            style="Danger.TButton",
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X)

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Root.TFrame")
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        panel.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="正在载入配置")
        self.status_label = ttk.Label(
            panel,
            textvariable=self.status_var,
            style="Dry.Status.TLabel",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

        cards = ttk.Frame(panel, style="Root.TFrame")
        cards.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for index in range(6):
            cards.columnconfigure(index, weight=1, uniform="summary")

        self._summary_card(cards, "账户权益", "equity", "0.00 U", 0)
        self._summary_card(cards, "可用余额", "available", "0.00 U", 1)
        self._summary_card(cards, "未实现盈亏", "unrealized", "0.00 U", 2)
        self._summary_card(cards, "相对本金盈亏", "capital_pnl", "0.00 U", 3)
        self._summary_card(cards, "仓位占用", "initial_margin_usage", "0.00%", 4)
        self._summary_card(cards, "持仓数量", "position_count", "0", 5)

    def _summary_card(self, parent: ttk.Frame, title: str, key: str, default: str, column: int) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 11))
        frame.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0 if column == 0 else 5, 0 if column == 5 else 5),
        )
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
        var = tk.StringVar(value=default)
        label = ttk.Label(frame, textvariable=var, style="CardValue.TLabel")
        label.pack(anchor=tk.W, pady=(7, 0))
        self.summary_vars[key] = var
        self.summary_labels[key] = label

    def _build_positions(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="持仓与币种盈亏", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        columns = ("symbol", "side", "strategy", "size_usdt", "leverage", "entry", "mark", "margin", "pnl", "roe")
        self.positions = ttk.Treeview(
            panel, columns=columns, show="headings", height=6
        )
        headings = {
            "symbol": "币种",
            "side": "方向",
            "strategy": "策略",
            "size_usdt": "仓位U",
            "leverage": "倍率",
            "entry": "开仓价",
            "mark": "标记价",
            "margin": "保证金U",
            "pnl": "盈亏U",
            "roe": "ROE",
        }
        widths = {
            "symbol": 92,
            "side": 78,
            "strategy": 64,
            "size_usdt": 92,
            "leverage": 64,
            "entry": 86,
            "mark": 86,
            "margin": 86,
            "pnl": 86,
            "roe": 72,
        }
        for column in columns:
            self.positions.heading(column, text=headings[column])
            self.positions.column(column, width=widths[column], anchor=tk.E if column not in {"symbol", "side", "strategy"} else tk.W)
        self.positions.tag_configure("profit", foreground=THEME["profit"])
        self.positions.tag_configure("loss", foreground=THEME["loss"])
        self.positions.tag_configure("flat", foreground=THEME["muted"])
        self.positions.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.positions.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.positions.configure(yscrollcommand=scrollbar.set)

    def _build_log(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=2, column=0, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="运行日志", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.log_text = tk.Text(
            panel,
            height=6,
            wrap=tk.WORD,
            bg=THEME["log"],
            fg=THEME["soft"],
            insertbackground=THEME["text"],
            selectbackground=THEME["accent_active"],
            selectforeground="#ffffff",
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            highlightthickness=1,
            relief=tk.FLAT,
            font=("Consolas", 10),
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int = 0) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, pady=5, padx=(0, 8))
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=column + 1, sticky=tk.EW, pady=5, padx=(0, 8))
        parent.columnconfigure(column + 1, weight=1)

    def _combo(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...], row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5, padx=(0, 8))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(row=row, column=1, sticky=tk.EW, pady=5)
        parent.columnconfigure(1, weight=1)

    def _on_execution_mode_selected(
        self, _event: tk.Event | None = None
    ) -> None:
        current_mode = _execution_mode_for_config(self._last_config)
        if self.worker and self.worker.is_alive():
            self.execution_mode.set(current_mode)
            messagebox.showwarning(
                "正在运行",
                "请先停止当前运行，再切换 DRY-RUN / LIVE。",
            )
            return

        selected = self.execution_mode.get().strip().upper()
        path = Path(
            LIVE_GUI_CONFIG_PATH
            if selected == EXECUTION_MODE_LIVE
            else ACTIVE_GUI_CONFIG_PATH
        )
        try:
            config = load_live_config(path)
        except Exception as exc:
            self.execution_mode.set(current_mode)
            messagebox.showerror("模式切换失败", str(exc))
            return

        self.config_path.set(str(path))
        self._apply_config(config)
        self._render_empty_positions(config)
        if selected == EXECUTION_MODE_LIVE:
            self.log(
                "已切换到 LIVE；当前尚未启动，不会发送订单。"
            )
        else:
            self.log(
                "已切换到 DRY-RUN；读取主网实时行情，仅本地模拟。"
            )

    def _update_execution_mode_ui(self) -> None:
        mode = self.execution_mode.get().strip().upper()
        if mode == EXECUTION_MODE_LIVE:
            self.execution_mode_summary.set(
                "主网实时账户与真实订单。点击启动后只再确认一次。"
            )
            self.safety_note.set(
                "LIVE 使用专用 One-way 合约账户。停止程序不会自动平仓，"
                "已有交易所保护单会继续保留。"
            )
            self.start_button.configure(
                text="启动 LIVE 实盘",
                style="Danger.TButton",
            )
            self.status_label.configure(style="Live.Status.TLabel")
            if not (self.worker and self.worker.is_alive()):
                self.status_var.set("MAINNET  /  LIVE 已选择  /  等待启动确认")
            return

        self.execution_mode_summary.set(
            "主网实时行情，本地模拟资金与成交，不发送真实订单。"
        )
        self.safety_note.set(
            "DRY-RUN 与实盘使用相同的 v9/v7 信号源，"
            "但资金、仓位和成交只保存在本地模拟账本。"
        )
        self.start_button.configure(
            text="启动 DRY-RUN",
            style="Accent.TButton",
        )
        self.status_label.configure(style="Dry.Status.TLabel")
        if not (self.worker and self.worker.is_alive()):
            self.status_var.set("MAINNET  /  DRY-RUN  /  等待启动")

    def _build_strategy_selector(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="策略选择", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        selector = ttk.Frame(parent, style="Panel.TFrame")
        selector.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="运行策略").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        strategy_combo = ttk.Combobox(
            selector,
            textvariable=self.strategy_mode,
            values=STRATEGY_MODE_VALUES,
            state="readonly",
            width=31,
        )
        strategy_combo.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=5)
        strategy_combo.bind("<<ComboboxSelected>>", self._on_strategy_mode_selected)
        ttk.Button(
            selector,
            text="应用策略参数",
            command=self._apply_selected_strategy_to_form,
        ).grid(row=0, column=2, sticky=tk.E, pady=5)
        ttk.Label(
            selector,
            textvariable=self.strategy_mode_summary,
            style="Muted.TLabel",
            wraplength=350,
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 0))

    def _on_strategy_mode_selected(self, _event: tk.Event | None = None) -> None:
        self._update_strategy_mode_summary()

    def _update_strategy_mode_summary(self) -> None:
        mode = self._selected_strategy_mode()
        config = self._last_config
        if config is not None:
            symbol_count = len(config.trading.entry_symbols or config.trading.symbols)
            symbol_text = f"扫描币种={symbol_count}。"
        else:
            symbol_text = ""
        self.strategy_mode_summary.set(
            f"当前选择：{mode}。{symbol_text}{STRATEGY_MODE_SUMMARIES.get(mode, '')}"
        )

    def _selected_strategy_mode(self) -> str:
        mode = self.strategy_mode.get().strip()
        return mode if mode in STRATEGY_MODE_VALUES else STRATEGY_MODE_MANUAL

    def _apply_selected_strategy_to_form(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(
                "策略正在运行",
                "请先停止交易线程，等待状态恢复后再切换策略。",
            )
            return
        mode = self._selected_strategy_mode()
        self._update_strategy_mode_summary()
        if mode == STRATEGY_MODE_COMBINED_LIVE:
            path = Path(LIVE_GUI_CONFIG_PATH)
            if not path.exists():
                messagebox.showerror(
                    "LIVE配置缺失", f"找不到独立配置：{path}"
                )
                return
            self.config_path.set(str(path))
            self._apply_config(load_live_config(path))
            self._render_empty_positions(self._last_config)
            self.log(
                "已切换到独立 v9/v7 LIVE 配置；当前默认未ARM，"
                "不会发送订单。"
            )
            return
        if (
            mode == STRATEGY_MODE_COMBINED_SHADOW
            and self._last_config is not None
            and self._last_config.combined_breakout_v8_grid_v6_live.enabled
        ):
            path = Path(ACTIVE_GUI_CONFIG_PATH)
            self.config_path.set(str(path))
            self._apply_config(load_live_config(path))
            self._render_empty_positions(self._last_config)
            self.log("已切回 v9/v7 DRY-RUN；真实订单路径已关闭。")
            return
        self.initial_entry_fraction.set("1.0")
        self.max_scale_ins_per_symbol.set("0")
        self.allow_loss_scale_in.set(False)
        self.starting_capital.set("200.0")
        if mode == STRATEGY_MODE_COMBINED_SHADOW:
            self.max_open_positions.set("2")
            self.max_new_entries_per_cycle.set("2")
            self.risk_per_trade.set("0.034")
            self.margin_usage.set("0.95")
            self.symbol_margin.set("0.95")
            self.max_drawdown.set("0.60")
            self.dry_run.set(True)
            self.log("已应用 Breakout v9 共享平衡版 + Grid v7 shadow参数：50币、共享账户最多2仓、每策略1仓、强制dry-run。")
        elif mode == STRATEGY_MODE_DUAL_THRUST_SHADOW:
            self.starting_capital.set("2000.0")
            self.max_open_positions.set("1")
            self.max_new_entries_per_cycle.set("1")
            self.risk_per_trade.set("0.025")
            self.max_drawdown.set("0.60")
            self.dry_run.set(True)
            self.log("已应用 Hybrid v5 50币单仓 shadow 参数，强制dry-run。")
        elif mode in {STRATEGY_MODE_MTF_RESET, STRATEGY_MODE_MTF}:
            self.max_open_positions.set("1")
            self.max_new_entries_per_cycle.set("1")
            self.risk_per_trade.set("0.0321839081" if mode == STRATEGY_MODE_MTF_RESET else "0.0275862069")
            self.margin_usage.set("0.30")
            self.symbol_margin.set("0.30")
            self.max_drawdown.set("0.40")
            self.mtf_allow_short.set(False)
            self.log(f"已应用 {mode} 单仓冻结参数。")
        else:
            self.log(f"已选择 {mode}；保存或启动时会应用对应配置。")

    def _load_initial_config(self) -> None:
        path = Path(self.config_path.get())
        if path.exists():
            config = load_live_config(path)
        elif Path(FALLBACK_CONFIG_PATH).exists():
            config = load_live_config(FALLBACK_CONFIG_PATH)
        else:
            config = default_live_config()
        self._apply_config(config)
        self._render_empty_positions(config)

    def load_config(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(
                "策略正在运行",
                "请先停止交易线程，等待状态恢复后再加载配置。",
            )
            return
        try:
            config = load_live_config(self.config_path.get())
            self._apply_config(config)
            self._render_empty_positions(config)
            self.log(f"已加载配置 {self.config_path.get()}")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    def save_config(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(
                "策略正在运行",
                "运行期间不保存配置；请先停止交易线程。",
            )
            return
        try:
            config = _lock_live_authorization_for_persistence(
                self._read_config()
            )
            if config.combined_breakout_v8_grid_v6_live.enabled:
                self.live_armed.set(False)
                self.live_confirmation.set("")
                self.mainnet_confirmation.set("")
                self.log(
                    "LIVE配置已按锁定状态保存；ARM与两项确认文本不会持久化。"
                )
            write_live_config(self.config_path.get(), config)
            self.log(f"已保存配置 {self.config_path.get()}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def check_account(self) -> None:
        self.refresh_account()

    def refresh_account(self) -> None:
        try:
            config = self._read_config()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        threading.Thread(target=self._refresh_account_worker, args=(config,), daemon=True).start()

    def show_trade_history(self) -> None:
        try:
            config = self._read_config()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        threading.Thread(target=self._trade_history_worker, args=(config,), daemon=True).start()

    def _refresh_account_worker(self, config: LiveAppConfig) -> None:
        try:
            client = self._client_for_config(config)
            self.log_from_thread(f"API Key: {mask_secret(client.api_key)}")
            if config.combined_breakout_v8_grid_v6_live.enabled:
                trader = _combined_live_trader_class(config)(
                    config, client, logger=self.log_from_thread
                )
                snapshot = trader.snapshot_account()
                self.account_from_thread(
                    snapshot,
                    reset_live_display_baseline=True,
                )
                live = config.combined_breakout_v8_grid_v6_live
                self.log_from_thread(
                    f"v9/v7 LIVE账户只读检查: 权益={snapshot.equity:.2f}U "
                    f"持仓={len(snapshot.position_rows)}/2 "
                    f"状态={'ARMED' if live.armed else 'LOCKED'} "
                    f"账本={trader.state_path}"
                )
                return
            if config.combined_volatility_trend_grid_shadow.enabled:
                trader = _combined_shadow_trader_class(config)(
                    config, client, logger=self.log_from_thread
                )
                snapshot = trader.snapshot_account()
                self.account_from_thread(snapshot, sync_starting_capital=False)
                self.log_from_thread(
                    f"组合Shadow账户: 权益={snapshot.equity:.2f}U "
                    f"持仓={len(snapshot.position_rows)}/2 "
                    f"样本文件={config.combined_volatility_trend_grid_shadow.state_path}"
                )
                return
            if config.dual_thrust_shadow.enabled:
                trader = DualThrustShadowTrader(config, client, logger=self.log_from_thread)
                snapshot = trader.snapshot_account()
                self.account_from_thread(snapshot, sync_starting_capital=False)
                self.log_from_thread(
                    f"Shadow账户: 权益={snapshot.equity:.2f}U 持仓={len(snapshot.positions)} "
                    f"样本文件={config.dual_thrust_shadow.state_path}"
                )
                return
            if not client.api_key or not client.api_secret:
                client.ping()
                self.log_from_thread("未配置密钥，仅完成公开 ping")
                return
            trader = BinanceAutoTrader(config, client, logger=self.log_from_thread)
            snapshot = trader.snapshot_account()
            self.account_from_thread(snapshot, sync_starting_capital=True)
            self.log_from_thread(
                f"账户检查成功: 权益={snapshot.equity:.2f}U 可用={snapshot.available_balance:.2f}U "
                f"持仓模式={snapshot.position_mode}"
            )
        except Exception as exc:
            self.log_from_thread(f"账户检查失败: {type(exc).__name__}: {exc}")

    def _trade_history_worker(self, config: LiveAppConfig) -> None:
        try:
            client = self._client_for_config(config)
            if config.combined_volatility_trend_grid_shadow.enabled:
                trader = _combined_shadow_trader_class(config)(
                    config, client, logger=self.log_from_thread
                )
                report = trader.acceptance_report()
                self.log_from_thread(
                    "组合Shadow交易统计: "
                    f"平仓={report['trade_count']} 净盈亏={report['closed_net_pnl']:+.4f}U "
                    f"胜率={report['win_rate'] * 100:.2f}% PF={report['profit_factor']:.3f} "
                    f"最大回撤={report['max_drawdown_pct'] * 100:.2f}%"
                )
                for trade in reversed(report["trades"][-20:]):
                    self.log_from_thread(
                        f"{trade['exit_time']} {trade['strategy']} {trade['symbol']} "
                        f"{trade['side']} 净盈亏={trade['net_pnl']:+.4f}U "
                        f"({trade['pnl_r']:+.3f}R) 原因={trade['exit_reason']}"
                    )
                return
            if config.dual_thrust_shadow.enabled:
                trader = DualThrustShadowTrader(config, client, logger=self.log_from_thread)
                report = trader.acceptance_report()
                self.log_from_thread(
                    "Dual Thrust Shadow交易统计: "
                    f"平仓={report['trade_count']} 净盈亏={report['closed_net_pnl']:+.4f}U "
                    f"胜率={report['win_rate'] * 100:.2f}% PF={report['profit_factor']:.3f}"
                )
                return
            if not client.api_key or not client.api_secret:
                self.log_from_thread("未配置 API Key/Secret，无法读取历史成交")
                return

            symbols = config.trading.entry_symbols or config.trading.symbols
            all_trades: list[dict] = []
            failed: list[str] = []
            for symbol in symbols:
                try:
                    all_trades.extend(client.user_trades(symbol=symbol, limit=1000))
                except Exception as exc:
                    failed.append(f"{symbol}({type(exc).__name__})")

            if not all_trades:
                self.log_from_thread("没有读取到历史成交记录")
                if failed:
                    self.log_from_thread(f"读取失败币种: {', '.join(failed[:12])}")
                return

            stats = _summarize_user_trades(all_trades)
            self.log_from_thread(
                "历史成交统计: "
                f"币种={len(symbols)} 成交填单={stats['fills']} 平仓订单={stats['closed_orders']} "
                f"净胜率={stats['win_rate_pct']:.2f}% 毛盈亏={stats['realized_pnl']:+.4f}U "
                f"净盈亏={stats['net_pnl']:+.4f}U 平均净平仓={stats['avg_closed_pnl']:+.4f}U"
            )
            if stats["commission_text"]:
                self.log_from_thread(f"手续费合计: {stats['commission_text']}")
            if failed:
                self.log_from_thread(f"部分币种读取失败: {', '.join(failed[:12])}")

            recent = sorted(all_trades, key=lambda item: int(item.get("time", 0)), reverse=True)[:20]
            self.log_from_thread("最近成交记录:")
            for trade in recent:
                realized = _float_from_trade(trade, "realizedPnl")
                commission = _float_from_trade(trade, "commission")
                commission_asset = str(trade.get("commissionAsset", ""))
                self.log_from_thread(
                    f"{_fmt_trade_time(trade)} {trade.get('symbol', '')} {trade.get('side', '')} "
                    f"qty={trade.get('qty', '')} price={trade.get('price', '')} "
                    f"成交U={trade.get('quoteQty', '')} 盈亏={realized:+.4f}U "
                    f"手续费={commission:.6g}{commission_asset}"
                )
        except Exception as exc:
            self.log_from_thread(f"读取历史成交失败: {type(exc).__name__}: {exc}")

    def show_macro_strategy(self) -> None:
        try:
            config = self._read_config()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        macro = config.macro_events
        status = "启用" if macro.enabled else "关闭"
        self.log(
            "非农策略: "
            f"{status} 主币={macro.primary_symbol} 事件={','.join(macro.event_types)} "
            f"杠杆={macro.leverage}x 保证金={macro.margin_pct * 100:.2f}% "
            f"延迟={macro.post_event_entry_delay_seconds}s 持仓≤{macro.max_holding_seconds}s "
            f"止损={macro.stop_loss_pct * 100:.2f}% 止盈={macro.take_profit_pct * 100:.2f}% "
            f"文件={macro.events_path}"
        )

    def start_trader(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            config = self._read_config()
            live_config = config.combined_breakout_v8_grid_v6_live
            if live_config.enabled:
                if not messagebox.askyesno(
                    "确认启动 LIVE 实盘",
                    "Breakout v9 共享平衡版 + Grid v7 将在 Binance 主网真实下单。\n\n"
                    "50 币、10x 全仓、全局最多 2 仓；停止程序不会自动平仓。\n\n"
                    "仅当这是专用 One-way 合约账户，并且你接受真实资金风险时，"
                    "才选择“是”。",
                    icon="warning",
                ):
                    return
                config = _authorize_live_gui_session(config)
                live_config = (
                    config.combined_breakout_v8_grid_v6_live
                )
                self.mainnet_confirmation.set("CONFIRM_MAINNET")
                self.live_armed.set(True)
                self.live_confirmation.set(
                    live_config.live_confirmation_text
                )
            client = self._client_for_config(config)
            if live_config.enabled:
                trader = _combined_live_trader_class(config)(
                    config,
                    client,
                    logger=self.log_from_thread,
                    account_callback=self.account_from_thread,
                )
                try:
                    trader.validate_startup_settings_only()
                except Exception as exc:
                    self._clear_live_runtime_authorization()
                    messagebox.showerror("LIVE启动检查失败", str(exc))
                    return
            elif config.combined_volatility_trend_grid_shadow.enabled:
                trader = _combined_shadow_trader_class(config)(
                    config,
                    client,
                    logger=self.log_from_thread,
                    account_callback=self.account_from_thread,
                )
            elif config.dual_thrust_shadow.enabled:
                trader = DualThrustShadowTrader(
                    config,
                    client,
                    logger=self.log_from_thread,
                    account_callback=self.account_from_thread,
                )
            else:
                trader = BinanceAutoTrader(
                    config,
                    client,
                    logger=self.log_from_thread,
                    account_callback=self.account_from_thread,
                )
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._run_trader_worker, args=(trader, self.stop_event), daemon=True)
            self.worker.start()
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            if live_config.enabled:
                self.status_var.set("MAINNET  /  LIVE  /  正在启动")
                self.status_label.configure(
                    style="Live.Status.TLabel"
                )
            else:
                self.status_var.set("MAINNET  /  DRY-RUN  /  正在启动")
                self.status_label.configure(
                    style="Dry.Status.TLabel"
                )
            self.log("已请求启动")
        except Exception as exc:
            if self.execution_mode.get() == EXECUTION_MODE_LIVE:
                self._clear_live_runtime_authorization()
            messagebox.showerror("启动失败", str(exc))

    def _run_trader_worker(self, trader: Any, stop_event: threading.Event) -> None:
        try:
            trader.run_forever(stop_event)
        except Exception as exc:
            self.log_from_thread(f"交易循环启动失败: {type(exc).__name__}: {exc}")
        finally:
            self.after(0, self._trader_worker_stopped)

    def _trader_worker_stopped(self) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.execution_mode.get() == EXECUTION_MODE_LIVE:
            self._clear_live_runtime_authorization()
            self.log("LIVE运行授权已从界面清除；再次启动需重新确认。")
        self._update_execution_mode_ui()

    def _clear_live_runtime_authorization(self) -> None:
        self.live_armed.set(False)
        self.live_confirmation.set("")
        self.mainnet_confirmation.set("")

    def stop_trader(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.log("已请求停止")

    def _client_for_config(self, config: LiveAppConfig) -> BinanceFuturesClient:
        api_key = read_secret(config.exchange.api_key_env)
        api_secret = read_secret(config.exchange.api_secret_env)
        live = config.combined_breakout_v8_grid_v6_live
        return BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            environment=config.exchange.environment,
            recv_window=config.exchange.recv_window,
            timeout_seconds=config.exchange.timeout_seconds,
            request_weight_budget=RequestWeightBudget(
                limit=live.request_weight_limit,
                soft_limit_ratio=(
                    live.request_weight_soft_limit_ratio
                ),
                default_cooldown_seconds=(
                    live.rate_limit_default_cooldown_seconds
                ),
            ),
        )

    def _apply_config(self, config: LiveAppConfig) -> None:
        previous_mode = _execution_mode_for_config(self._last_config)
        next_mode = _execution_mode_for_config(config)
        if previous_mode != next_mode:
            self._live_display_equity_baseline = None
        self._last_config = config
        self.execution_mode.set(next_mode)
        self.environment.set(config.exchange.environment)
        self.dry_run.set(config.trading.dry_run)
        self.api_key_env.set(config.exchange.api_key_env)
        self.api_secret_env.set(config.exchange.api_secret_env)
        self._set_symbols_value(config.trading.symbols)
        self.entry_symbols.set(",".join(config.trading.entry_symbols))
        self.timeframe.set(config.trading.timeframe)
        self.poll_seconds.set(str(config.trading.poll_seconds))
        self.entry_scan_seconds.set(str(config.trading.entry_scan_seconds))
        self.symbol_reentry_cooldown_seconds.set(str(config.trading.symbol_reentry_cooldown_seconds))
        self.leverage.set(str(config.trading.leverage))
        self.max_open_positions.set(str(config.trading.max_open_positions))
        self.max_new_entries_per_cycle.set(str(config.trading.max_new_entries_per_cycle))
        self.starting_capital.set(str(config.risk.starting_capital_usdt))
        self.margin_usage.set(str(config.risk.max_account_margin_usage_pct))
        self.symbol_margin.set(str(config.risk.max_symbol_margin_pct))
        self.min_symbol_margin.set(str(config.risk.min_symbol_margin_pct))
        self.max_notional.set(str(config.risk.max_position_notional_usdt))
        self.risk_per_trade.set(str(config.risk.risk_per_trade_pct))
        self.daily_loss.set(str(config.risk.max_daily_loss_pct))
        self.max_drawdown.set(str(config.risk.max_drawdown_pct))
        self.soft_drawdown_reduce.set(str(config.risk.soft_drawdown_reduce_pct))
        self.soft_drawdown_stop.set(str(config.risk.soft_drawdown_stop_pct))
        self.soft_drawdown_min_size.set(str(config.risk.soft_drawdown_min_size_multiplier))
        self.estimated_fee_bps.set(str(config.risk.estimated_fee_bps))
        self.estimated_slippage_bps.set(str(config.risk.estimated_slippage_bps))
        self.min_profit_after_cost.set(str(config.risk.min_profit_after_cost_pct))
        self.min_available.set(str(config.risk.min_available_balance_usdt))
        self.mainnet_confirmation.set(config.trading.mainnet_confirmation_text)
        live_config = config.combined_breakout_v8_grid_v6_live
        self.live_armed.set(live_config.armed)
        self.live_confirmation.set(
            live_config.live_confirmation_text
        )
        self.fast_ema.set(str(config.strategy.fast_ema))
        self.slow_ema.set(str(config.strategy.slow_ema))
        self.atr_period.set(str(config.strategy.atr_period))
        self.channel_period.set(str(config.strategy.channel_period))
        self.volume_period.set(str(config.strategy.volume_period))
        self.min_atr_pct.set(str(config.strategy.min_atr_pct))
        self.max_atr_pct.set(str(config.strategy.max_atr_pct))
        self.min_volume_ratio.set(str(config.strategy.min_volume_ratio))
        self.breakout_buffer.set(str(config.strategy.breakout_buffer_atr))
        self.ema_gap.set(str(config.strategy.ema_gap_atr))
        self.stop_loss_atr.set(str(config.strategy.stop_loss_atr))
        self.take_profit_atr.set(str(config.strategy.take_profit_atr))
        self.breakeven_atr.set(str(config.strategy.breakeven_atr))
        self.trailing_activation_atr.set(str(config.strategy.trailing_activation_atr))
        self.trailing_stop_atr.set(str(config.strategy.trailing_stop_atr))
        self.max_holding_bars.set(str(config.strategy.max_holding_bars))
        self.allow_short.set(config.strategy.allow_short)
        self.long_score_threshold.set(str(config.strategy.long_score_threshold))
        self.short_score_threshold.set(str(config.strategy.short_score_threshold))
        self.long_risk_bias.set(str(config.strategy.long_risk_bias))
        self.short_risk_bias.set(str(config.strategy.short_risk_bias))
        self.regime_filter_enabled.set(config.strategy.regime_filter_enabled)
        self.regime_lookback.set(str(config.strategy.regime_lookback))
        self.long_min_slow_slope_atr.set(str(config.strategy.long_min_slow_slope_atr))
        self.short_max_slow_slope_atr.set(str(config.strategy.short_max_slow_slope_atr))
        self.spike_guard_enabled.set(config.strategy.spike_guard_enabled)
        self.spike_trade_enabled.set(config.strategy.spike_trade_enabled)
        self.rsi_reversal_enabled.set(config.strategy.rsi_reversal_enabled)
        self.spike_min_range_atr.set(str(config.strategy.spike_min_range_atr))
        self.spike_min_wick_atr.set(str(config.strategy.spike_min_wick_atr))
        self.spike_min_wick_ratio.set(str(config.strategy.spike_min_wick_ratio))
        self.spike_min_volume_ratio.set(str(config.strategy.spike_min_volume_ratio))
        self.spike_block_bars.set(str(config.strategy.spike_block_bars))
        self.spike_recovery_ratio.set(str(config.strategy.spike_recovery_ratio))
        self.spike_stop_atr.set(str(config.strategy.spike_stop_atr))
        self.spike_take_profit_atr.set(str(config.strategy.spike_take_profit_atr))
        self.spike_risk_multiplier.set(str(config.strategy.spike_risk_multiplier))
        self.spike_max_holding_bars.set(str(config.strategy.spike_max_holding_bars))
        self.initial_entry_fraction.set(str(config.trading.initial_entry_fraction))
        self.scale_in_entry_fraction.set(str(config.trading.scale_in_entry_fraction))
        self.max_scale_ins_per_symbol.set(str(config.trading.max_scale_ins_per_symbol))
        self.scale_in_min_profit_pct.set(str(config.trading.scale_in_min_profit_pct))
        self.scale_in_cooldown_seconds.set(str(config.trading.scale_in_cooldown_seconds))
        self.allow_loss_scale_in.set(config.trading.allow_loss_scale_in)
        self.loss_scale_in_trigger_pct.set(str(config.trading.loss_scale_in_trigger_pct))
        self.loss_scale_in_entry_fraction.set(str(config.trading.loss_scale_in_entry_fraction))
        self.macro_enabled.set(False)
        self.macro_events_path.set(config.macro_events.events_path)
        self.macro_symbols.set(",".join(config.macro_events.symbols))
        self.macro_primary_symbol.set(config.macro_events.primary_symbol)
        self.macro_event_types.set(",".join(config.macro_events.event_types))
        self.macro_leverage.set(str(config.macro_events.leverage))
        self.macro_pre_flatten_seconds.set(str(config.macro_events.pre_event_flatten_seconds))
        self.macro_post_lockout_seconds.set(str(config.macro_events.post_event_lockout_seconds))
        self.macro_entry_delay_seconds.set(str(config.macro_events.post_event_entry_delay_seconds))
        self.macro_margin_pct.set(str(config.macro_events.margin_pct))
        self.macro_stop_loss_pct.set(str(config.macro_events.stop_loss_pct))
        self.macro_take_profit_pct.set(str(config.macro_events.take_profit_pct))
        self.macro_max_holding_seconds.set(str(config.macro_events.max_holding_seconds))
        self.macro_nfp_min_surprise_k.set(str(config.macro_events.nfp_min_surprise_k))
        self.macro_cpi_min_surprise_pct.set(str(config.macro_events.cpi_min_surprise_pct))
        self.strategy_mode.set(_detect_strategy_mode(config))
        self.strategy_indicator_enabled.set(False)
        self.strategy_vbp_enabled.set(False)
        self.mtf_allow_long.set(bool(getattr(config.strategy, "mtf_allow_long", True)))
        self.mtf_allow_short.set(False)
        if live_config.enabled:
            self.mtf_core_parameters.set(
                "Breakout v9 共享平衡版 + Grid v7 50币  |  真实交易所执行\n"
                "全局最多 2 仓  |  每策略最多 1 仓  |  同币互斥\n"
                f"研究风险缩放 {live_config.risk_scale * 100:.0f}%  |  "
                f"总名义上限 {live_config.max_gross_notional_multiple:.2f}x\n"
                f"日亏损熔断 {live_config.max_daily_loss_pct * 100:.1f}%  |  "
                f"峰值回撤熔断 {live_config.max_drawdown_pct * 100:.1f}%\n"
                "幂等订单 + 启动/循环对账 + reduceOnly退出 + "
                "交易所STOP_MARKET保护\n"
                f"当前：{'ARMED（仍需双重确认）' if live_config.armed else 'LOCKED，不会下单'}"
            )
        elif config.combined_volatility_trend_grid_shadow.enabled:
            combined = config.combined_volatility_trend_grid_shadow
            breakout = config.dual_thrust_shadow
            is_v9_grid_v7 = (
                combined.strategy_name == COMBINED_V9_GRID_V7_NAME
            )
            is_v8_grid_v6 = combined.strategy_name == COMBINED_V8_GRID_V6_NAME
            is_v7_grid_v5 = combined.strategy_name == COMBINED_V7_GRID_V5_NAME
            is_v5_grid_v3 = combined.strategy_name == COMBINED_V5_GRID_V3_NAME
            combined_label = (
                "Breakout v9 共享平衡版 + Grid v7"
                if is_v9_grid_v7
                else (
                    "Breakout v8 + Grid v6"
                    if is_v8_grid_v6
                    else (
                        "Breakout v7 + Grid v5"
                        if is_v7_grid_v5
                        else (
                            "Hybrid v5 + Grid v3"
                            if is_v5_grid_v3
                            else "Breakout + Dynamic Trend Grid"
                        )
                    )
                )
            )
            if is_v9_grid_v7:
                breakout_line = (
                    f"Breakout v9: 已收盘{breakout.timeframe_minutes}m 多空 / "
                    "分数分侧利润保护 + 小比例分批止盈 / "
                    f"{breakout.stop_atr_multiple:.2f}ATR止损"
                )
                grid_line = (
                    "Grid v7: 已收盘60m / 仅做空 / 两层网格 / "
                    "弱信号拒绝 + 循环利润回撤保护"
                )
            elif is_v8_grid_v6:
                breakout_line = (
                    f"Breakout v8: 已收盘{breakout.timeframe_minutes}m 多空 / "
                    f"分数凸性动态风险 / {breakout.stop_atr_multiple:.2f}ATR止损"
                )
                grid_line = (
                    "Grid v6: 已收盘60m / 仅做空 / 两层网格 / "
                    "弱信号拒绝 + campaign风险保护"
                )
            elif is_v7_grid_v5:
                breakout_line = (
                    f"Breakout v7: 已收盘{breakout.timeframe_minutes}m 多空 / "
                    f"置信度动态风险 / {breakout.stop_atr_multiple:.2f}ATR止损"
                )
                grid_line = (
                    "Grid v5: 已收盘60m / 仅做空 / 两层网格 / "
                    "置信度分层并拒绝弱信号"
                )
            else:
                breakout_line = (
                    f"Hybrid v5: 已收盘{breakout.timeframe_minutes}m 多空 / 风险"
                    f"{breakout.risk_per_trade_pct * 100:.1f}% / "
                    f"{breakout.stop_atr_multiple:.2f}ATR止损"
                )
                grid_line = (
                    f"{'Grid v3' if is_v5_grid_v3 else 'Grid'}: "
                    "已收盘60m / 仅做空 / 两层网格 / campaign风险10%"
                )
            self.mtf_core_parameters.set(
                f"{combined_label} {len(combined.enabled_symbols)}币  |  共享模拟账户\n"
                f"全局最多 {combined.max_open_positions} 仓  |  每策略最多 "
                f"{combined.max_open_positions_per_strategy} 仓  |  同币互斥\n"
                f"{breakout_line}\n"
                f"{grid_line}\n"
                f"总名义上限 {combined.max_gross_notional_multiple:.1f}x  |  "
                f"硬回撤停止新开仓 {combined.hard_drawdown_stop_pct * 100:.0f}%\n"
                "主网实时行情 + API连接 + full-cost共享撮合  |  当前强制DRY-RUN"
            )
        elif config.dual_thrust_shadow.enabled:
            shadow = config.dual_thrust_shadow
            self.mtf_core_parameters.set(
                f"Hybrid v5 Balanced Runner {len(shadow.enabled_symbols)}币  |  已收盘{shadow.timeframe_minutes}m信号\n"
                f"Range回看 {shadow.lookback_days}日  |  LONG K={shadow.long_k:.2f} / SHORT K={shadow.short_k:.2f}\n"
                f"结构止损 {shadow.stop_atr_multiple:.2f} ATR  |  {shadow.partial_take_profit_r:.0f}R止盈"
                f"{shadow.partial_take_profit_fraction * 100:.0f}% / 主目标{shadow.take_profit_r:.0f}R  |  时间退出 {shadow.max_holding_minutes}m\n"
                f"条件Fail-fast {shadow.fail_fast_minutes}m / MFE<{shadow.fail_fast_min_mfe_r:.2f}R / "
                f"当前<={shadow.fail_fast_max_current_r:.2f}R\n"
                f"单笔风险 {shadow.risk_per_trade_pct * 100:.2f}%  |  单仓  |  禁止加仓\n"
                "主网公开行情 + full-cost shadow  |  强制DRY-RUN"
            )
        else:
            self.mtf_core_parameters.set(
                "Regime 4h trend_pullback  |  Setup 已收盘1h动量重置\n"
                "Trigger 已收盘30m收缩释放  |  最早随后市场执行\n"
                f"止损上限 {float(getattr(config.strategy, 'mtf_max_stop_pct', 0.015)) * 100:.2f}%  |  "
                f"目标 {float(getattr(config.strategy, 'mtf_take_profit_r', 2.0)):.2f}R  |  "
                f"Fail-fast {int(getattr(config.strategy, 'mtf_fail_fast_minutes', 120))}m / "
                f"{float(getattr(config.strategy, 'mtf_fail_fast_min_r', 0.5)):.2f}R\n"
                f"LONG rank >= {float(getattr(config.strategy, 'mtf_long_min_rank_score', 4.25)):.2f}  |  "
                f"广度 >= {float(getattr(config.strategy, 'mtf_momentum_reset_min_breadth_ema21', 0.55)) * 100:.0f}%  |  "
                f"目标/成本 >= {float(getattr(config.strategy, 'mtf_min_target_to_cost_ratio', 12.0)):.0f}x\n"
                f"单笔风险 {float(config.risk.risk_per_trade_pct) * 100:.2f}%  |  "
                f"最大持仓 {int(getattr(config.strategy, 'mtf_max_open_positions', 1))}\n"
                f"最长持仓 {int(getattr(config.strategy, 'mtf_max_holding_minutes', 720))}m  |  "
                f"Funding {'开启' if getattr(config.strategy, 'mtf_use_funding_filter', True) else '关闭'}  |  "
                f"OI {'开启' if getattr(config.strategy, 'mtf_use_oi_filter', False) else '关闭'}"
            )
        self._update_strategy_mode_summary()
        self._update_execution_mode_ui()

    def _read_config(self) -> LiveAppConfig:
        base = self._last_config or default_live_config()
        read_int = _numeric_reader(int)
        read_float = _numeric_reader(float)
        exchange = replace(
            base.exchange,
            environment=self.environment.get(),
            api_key_env=self.api_key_env.get().strip(),
            api_secret_env=self.api_secret_env.get().strip(),
        )
        trading = replace(
            base.trading,
            symbols=self._read_symbols(),
            entry_symbols=self._read_entry_symbols(),
            timeframe=self.timeframe.get().strip(),
            kline_limit=base.trading.kline_limit,
            poll_seconds=read_int(self.poll_seconds, base.trading.poll_seconds),
            entry_scan_seconds=read_int(self.entry_scan_seconds, base.trading.entry_scan_seconds),
            symbol_reentry_cooldown_seconds=read_int(self.symbol_reentry_cooldown_seconds, base.trading.symbol_reentry_cooldown_seconds),
            dry_run=bool(self.dry_run.get()),
            mainnet_confirmation_text=self.mainnet_confirmation.get().strip(),
            leverage=read_int(self.leverage, base.trading.leverage),
            margin_type="CROSSED",
            max_open_positions=read_int(self.max_open_positions, base.trading.max_open_positions),
            max_new_entries_per_cycle=read_int(self.max_new_entries_per_cycle, base.trading.max_new_entries_per_cycle),
            initial_entry_fraction=read_float(self.initial_entry_fraction, base.trading.initial_entry_fraction),
            scale_in_entry_fraction=read_float(self.scale_in_entry_fraction, base.trading.scale_in_entry_fraction),
            max_scale_ins_per_symbol=read_int(self.max_scale_ins_per_symbol, base.trading.max_scale_ins_per_symbol),
            scale_in_min_profit_pct=read_float(self.scale_in_min_profit_pct, base.trading.scale_in_min_profit_pct),
            scale_in_cooldown_seconds=read_int(self.scale_in_cooldown_seconds, base.trading.scale_in_cooldown_seconds),
            allow_loss_scale_in=bool(self.allow_loss_scale_in.get()),
            loss_scale_in_trigger_pct=read_float(self.loss_scale_in_trigger_pct, base.trading.loss_scale_in_trigger_pct),
            loss_scale_in_entry_fraction=read_float(self.loss_scale_in_entry_fraction, base.trading.loss_scale_in_entry_fraction),
        )
        risk = replace(
            base.risk,
            starting_capital_usdt=read_float(self.starting_capital, base.risk.starting_capital_usdt),
            max_account_margin_usage_pct=read_float(self.margin_usage, base.risk.max_account_margin_usage_pct),
            max_symbol_margin_pct=read_float(self.symbol_margin, base.risk.max_symbol_margin_pct),
            min_symbol_margin_pct=read_float(self.min_symbol_margin, base.risk.min_symbol_margin_pct),
            max_position_notional_usdt=read_float(self.max_notional, base.risk.max_position_notional_usdt),
            risk_per_trade_pct=read_float(self.risk_per_trade, base.risk.risk_per_trade_pct),
            max_daily_loss_pct=read_float(self.daily_loss, base.risk.max_daily_loss_pct),
            max_drawdown_pct=read_float(self.max_drawdown, base.risk.max_drawdown_pct),
            soft_drawdown_reduce_pct=read_float(self.soft_drawdown_reduce, base.risk.soft_drawdown_reduce_pct),
            soft_drawdown_stop_pct=read_float(self.soft_drawdown_stop, base.risk.soft_drawdown_stop_pct),
            soft_drawdown_min_size_multiplier=read_float(self.soft_drawdown_min_size, base.risk.soft_drawdown_min_size_multiplier),
            estimated_fee_bps=read_float(self.estimated_fee_bps, base.risk.estimated_fee_bps),
            estimated_slippage_bps=read_float(self.estimated_slippage_bps, base.risk.estimated_slippage_bps),
            min_profit_after_cost_pct=read_float(self.min_profit_after_cost, base.risk.min_profit_after_cost_pct),
            min_available_balance_usdt=read_float(self.min_available, base.risk.min_available_balance_usdt),
        )
        strategy = replace(
            base.strategy,
            fast_ema=read_int(self.fast_ema, base.strategy.fast_ema),
            slow_ema=read_int(self.slow_ema, base.strategy.slow_ema),
            atr_period=read_int(self.atr_period, base.strategy.atr_period),
            channel_period=read_int(self.channel_period, base.strategy.channel_period),
            volume_period=read_int(self.volume_period, base.strategy.volume_period),
            min_atr_pct=read_float(self.min_atr_pct, base.strategy.min_atr_pct),
            max_atr_pct=read_float(self.max_atr_pct, base.strategy.max_atr_pct),
            min_volume_ratio=read_float(self.min_volume_ratio, base.strategy.min_volume_ratio),
            breakout_buffer_atr=read_float(self.breakout_buffer, base.strategy.breakout_buffer_atr),
            ema_gap_atr=read_float(self.ema_gap, base.strategy.ema_gap_atr),
            stop_loss_atr=read_float(self.stop_loss_atr, base.strategy.stop_loss_atr),
            take_profit_atr=read_float(self.take_profit_atr, base.strategy.take_profit_atr),
            breakeven_atr=read_float(self.breakeven_atr, base.strategy.breakeven_atr),
            trailing_activation_atr=read_float(self.trailing_activation_atr, base.strategy.trailing_activation_atr),
            trailing_stop_atr=read_float(self.trailing_stop_atr, base.strategy.trailing_stop_atr),
            max_holding_bars=read_int(self.max_holding_bars, base.strategy.max_holding_bars),
            allow_short=bool(self.allow_short.get()),
            mtf_allow_long=bool(self.mtf_allow_long.get()),
            mtf_allow_short=bool(self.mtf_allow_short.get()),
            long_score_threshold=read_float(self.long_score_threshold, base.strategy.long_score_threshold),
            short_score_threshold=read_float(self.short_score_threshold, base.strategy.short_score_threshold),
            long_risk_bias=read_float(self.long_risk_bias, base.strategy.long_risk_bias),
            short_risk_bias=read_float(self.short_risk_bias, base.strategy.short_risk_bias),
            regime_filter_enabled=bool(self.regime_filter_enabled.get()),
            regime_lookback=read_int(self.regime_lookback, base.strategy.regime_lookback),
            long_min_slow_slope_atr=read_float(self.long_min_slow_slope_atr, base.strategy.long_min_slow_slope_atr),
            short_max_slow_slope_atr=read_float(self.short_max_slow_slope_atr, base.strategy.short_max_slow_slope_atr),
            spike_guard_enabled=bool(self.spike_guard_enabled.get()),
            spike_trade_enabled=bool(self.spike_trade_enabled.get()),
            rsi_reversal_enabled=bool(self.rsi_reversal_enabled.get()),
            spike_min_range_atr=read_float(self.spike_min_range_atr, base.strategy.spike_min_range_atr),
            spike_min_wick_atr=read_float(self.spike_min_wick_atr, base.strategy.spike_min_wick_atr),
            spike_min_wick_ratio=read_float(self.spike_min_wick_ratio, base.strategy.spike_min_wick_ratio),
            spike_min_volume_ratio=read_float(self.spike_min_volume_ratio, base.strategy.spike_min_volume_ratio),
            spike_block_bars=read_int(self.spike_block_bars, base.strategy.spike_block_bars),
            spike_recovery_ratio=read_float(self.spike_recovery_ratio, base.strategy.spike_recovery_ratio),
            spike_stop_atr=read_float(self.spike_stop_atr, base.strategy.spike_stop_atr),
            spike_take_profit_atr=read_float(self.spike_take_profit_atr, base.strategy.spike_take_profit_atr),
            spike_risk_multiplier=read_float(self.spike_risk_multiplier, base.strategy.spike_risk_multiplier),
            spike_max_holding_bars=read_int(self.spike_max_holding_bars, base.strategy.spike_max_holding_bars),
        )
        macro_events = replace(
            base.macro_events,
            enabled=bool(self.macro_enabled.get()),
            events_path=self.macro_events_path.get().strip() or base.macro_events.events_path,
            symbols=self._read_macro_symbols(),
            primary_symbol=_normalize_symbol_text(self.macro_primary_symbol.get(), base.macro_events.primary_symbol),
            event_types=self._read_macro_event_types(),
            leverage=read_int(self.macro_leverage, base.macro_events.leverage),
            pre_event_flatten_seconds=read_int(self.macro_pre_flatten_seconds, base.macro_events.pre_event_flatten_seconds),
            post_event_lockout_seconds=read_int(self.macro_post_lockout_seconds, base.macro_events.post_event_lockout_seconds),
            post_event_entry_delay_seconds=read_int(self.macro_entry_delay_seconds, base.macro_events.post_event_entry_delay_seconds),
            margin_pct=read_float(self.macro_margin_pct, base.macro_events.margin_pct),
            stop_loss_pct=read_float(self.macro_stop_loss_pct, base.macro_events.stop_loss_pct),
            take_profit_pct=read_float(self.macro_take_profit_pct, base.macro_events.take_profit_pct),
            max_holding_seconds=read_int(self.macro_max_holding_seconds, base.macro_events.max_holding_seconds),
            nfp_min_surprise_k=read_float(self.macro_nfp_min_surprise_k, base.macro_events.nfp_min_surprise_k),
            cpi_min_surprise_pct=read_float(self.macro_cpi_min_surprise_pct, base.macro_events.cpi_min_surprise_pct),
        )
        combined_live = replace(
            base.combined_breakout_v8_grid_v6_live,
            armed=bool(self.live_armed.get()),
            live_confirmation_text=self.live_confirmation.get().strip(),
        )
        config = LiveAppConfig(
            exchange=exchange,
            trading=trading,
            strategy=strategy,
            filters=base.filters,
            risk=risk,
            macro_events=macro_events,
            vbp_strategy=base.vbp_strategy,
            portfolio_control=base.portfolio_control,
            regime_score=base.regime_score,
            reversal_alpha=base.reversal_alpha,
            cmipr=base.cmipr,
            mtper=base.mtper,
            mtpc=base.mtpc,
            dual_thrust_shadow=base.dual_thrust_shadow,
            combined_volatility_trend_grid_shadow=base.combined_volatility_trend_grid_shadow,
            combined_breakout_v8_grid_v6_live=combined_live,
        )
        selected_mode = (
            STRATEGY_MODE_COMBINED_LIVE
            if self.execution_mode.get().strip().upper()
            == EXECUTION_MODE_LIVE
            else STRATEGY_MODE_COMBINED_SHADOW
        )
        self.strategy_mode.set(selected_mode)
        config = _config_with_strategy_mode(config, selected_mode)
        if selected_mode == STRATEGY_MODE_MTF_RESET:
            config = replace(
                config,
                strategy=replace(
                    config.strategy,
                    mtf_allow_long=bool(self.mtf_allow_long.get()),
                    mtf_allow_short=False,
                    allow_short=False,
                ),
            )
        return _config_with_strategy_selection(config, indicator_enabled=False, vbp_enabled=False)

    def _apply_symbol_preset(self) -> None:
        self._set_symbols_value(DEFAULT_SYMBOLS)

    def _set_symbols_value(self, symbols: tuple[str, ...]) -> None:
        value = ",".join(symbols)
        self.symbols.set(value)
        if self.symbols_text:
            self.symbols_text.delete("1.0", tk.END)
            self.symbols_text.insert("1.0", value)

    def _read_symbols(self) -> tuple[str, ...]:
        if self.symbols_text:
            raw = self.symbols_text.get("1.0", tk.END)
        else:
            raw = self.symbols.get()
        symbols: list[str] = []
        for part in raw.replace("，", ",").replace("\n", ",").split(","):
            symbol = part.strip().upper()
            if not symbol:
                continue
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            if symbol not in symbols:
                symbols.append(symbol)
        return tuple(symbols)

    def _read_entry_symbols(self) -> tuple[str, ...]:
        raw = self.entry_symbols.get()
        if not raw.strip():
            return ()
        symbols: list[str] = []
        for part in raw.replace("，", ",").replace("\n", ",").split(","):
            symbol = part.strip().upper()
            if not symbol:
                continue
            if not symbol.endswith("USDT"):
                symbol = f"{symbol}USDT"
            if symbol not in symbols:
                symbols.append(symbol)
        return tuple(symbols)

    def _read_macro_symbols(self) -> tuple[str, ...]:
        raw = self.macro_symbols.get()
        symbols: list[str] = []
        for part in raw.replace("，", ",").replace("\n", ",").split(","):
            symbol = _normalize_symbol_text(part, "")
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return tuple(symbols)

    def _read_macro_event_types(self) -> tuple[str, ...]:
        raw = self.macro_event_types.get()
        event_types: list[str] = []
        for part in raw.replace("，", ",").replace("\n", ",").split(","):
            event_type = part.strip().upper()
            if event_type and event_type not in event_types:
                event_types.append(event_type)
        return tuple(event_types)

    def _render_empty_positions(self, config: LiveAppConfig) -> None:
        self.positions.delete(*self.positions.get_children())
        for symbol in config.trading.symbols:
            self.positions.insert("", tk.END, values=(symbol, "空仓", "-", "0.00", "-", "-", "-", "0.00", "0.00", "0.00%"), tags=("flat",))

    def _render_account(
        self,
        snapshot: AccountSnapshot,
        sync_starting_capital: bool = False,
        reset_live_display_baseline: bool = False,
    ) -> None:
        if sync_starting_capital:
            previous = _safe_float(self.starting_capital.get())
            self.starting_capital.set(f"{snapshot.equity:.2f}")
            if previous is None or abs(previous - snapshot.equity) >= 0.005:
                before = "未设置" if previous is None else f"{previous:.2f}U"
                self.log(f"本金U已同步: {before} -> {snapshot.equity:.2f}U")

        config = self._read_config()
        execution_mode = _execution_mode_for_config(config)
        live_baseline_was_reset = (
            reset_live_display_baseline
            and execution_mode == EXECUTION_MODE_LIVE
        )
        if live_baseline_was_reset:
            self._live_display_equity_baseline = snapshot.equity
            self.log(
                "LIVE界面盈亏基准已重置为"
                f"{snapshot.equity:.2f}U；本次刷新后相对盈亏为0。"
            )
        display_baseline = _account_display_equity_baseline(
            config,
            self._live_display_equity_baseline,
        )
        capital_pnl = snapshot.equity - display_baseline
        self.summary_vars["equity"].set(f"{snapshot.equity:.2f} U")
        self.summary_vars["available"].set(f"{snapshot.available_balance:.2f} U")
        self.summary_vars["unrealized"].set(f"{snapshot.total_unrealized_pnl:+.2f} U")
        self.summary_vars["capital_pnl"].set(f"{capital_pnl:+.2f} U")
        self.summary_vars["initial_margin_usage"].set(f"{snapshot.initial_margin_usage_pct * 100:.2f}%")
        self.summary_vars["position_count"].set(str(len(snapshot.position_rows)))
        self._set_summary_value_style("unrealized", snapshot.total_unrealized_pnl)
        self._set_summary_value_style("capital_pnl", capital_pnl)
        self.summary_labels["equity"].configure(style="Info.CardValue.TLabel")
        self.status_label.configure(
            style=(
                "Live.Status.TLabel"
                if execution_mode == EXECUTION_MODE_LIVE
                else "Dry.Status.TLabel"
            )
        )
        if live_baseline_was_reset:
            account_status = "LIVE盈亏已归零"
        elif sync_starting_capital:
            account_status = "本金已同步"
        else:
            account_status = "账户已更新"
        self.status_var.set(
            f"{config.exchange.environment.upper()}  /  {execution_mode}  /  "
            f"{snapshot.position_mode}  /  {account_status}"
        )

        self.positions.delete(*self.positions.get_children())
        rows_by_symbol: dict[str, list] = {}
        for position in snapshot.position_rows:
            rows_by_symbol.setdefault(position.symbol, []).append(position)

        for symbol in _position_symbols_first(config.trading.symbols, rows_by_symbol):
            rows = rows_by_symbol.get(symbol)
            if not rows:
                self.positions.insert("", tk.END, values=(symbol, "空仓", "-", "0.00", "-", "-", "-", "0.00", "0.00", "0.00%"), tags=("flat",))
                continue
            for position in rows:
                margin = position.notional / position.leverage if position.leverage > 0 else 0.0
                roe = position.unrealized_pnl / margin * 100.0 if margin > 0 else 0.0
                side = "多" if position.direction.value > 0 else "空"
                if position.position_side not in {"BOTH", ""}:
                    side = f"{side}/{position.position_side}"
                tag = "profit" if position.unrealized_pnl > 0 else "loss" if position.unrealized_pnl < 0 else "flat"
                self.positions.insert(
                    "",
                    tk.END,
                    values=(
                        position.symbol,
                        side,
                        _strategy_short_name(getattr(position, "entry_reason", "")),
                        _fmt_float(position.notional, 2),
                        f"{position.leverage}x",
                        _fmt_float(position.entry_price, 4),
                        _fmt_float(position.mark_price, 4),
                        _fmt_float(margin, 2),
                        f"{position.unrealized_pnl:+.2f}",
                        f"{roe:+.2f}%",
                    ),
                    tags=(tag,),
                )

    def _set_summary_value_style(self, key: str, value: float) -> None:
        label = self.summary_labels.get(key)
        if not label:
            return
        if value > 0:
            label.configure(style="Profit.CardValue.TLabel")
        elif value < 0:
            label.configure(style="Loss.CardValue.TLabel")
        else:
            label.configure(style="CardValue.TLabel")

    def log_from_thread(self, message: str) -> None:
        self.log_queue.put(message)

    def account_from_thread(
        self,
        snapshot: AccountSnapshot,
        sync_starting_capital: bool = False,
        reset_live_display_baseline: bool = False,
    ) -> None:
        self.account_queue.put(
            (
                snapshot,
                sync_starting_capital,
                reset_live_display_baseline,
            )
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamped_message = f"[{timestamp}] {message}"
        try:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamped_message}\n")
        except OSError:
            pass
        self.log_text.insert(tk.END, f"{timestamped_message}\n")
        self.log_text.see(tk.END)

    def _drain_queues(self) -> None:
        while True:
            try:
                (
                    snapshot,
                    sync_starting_capital,
                    reset_live_display_baseline,
                ) = self.account_queue.get_nowait()
            except queue.Empty:
                break
            self._render_account(
                snapshot,
                sync_starting_capital,
                reset_live_display_baseline,
            )

        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log(message)

        if self.worker and not self.worker.is_alive() and self.stop_button["state"] == tk.NORMAL:
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
        self.after(200, self._drain_queues)

    def _on_close(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        self.destroy()


def _build_coin_icon(size: int = 64) -> tk.PhotoImage:
    image = tk.PhotoImage(width=size, height=size)
    center = (size - 1) / 2.0
    outer_radius = size * 0.44
    inner_radius = size * 0.34
    ring_radius = size * 0.25
    highlight_x = center - size * 0.14
    highlight_y = center - size * 0.18

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            distance_sq = dx * dx + dy * dy
            if distance_sq > outer_radius * outer_radius:
                continue
            if distance_sq > inner_radius * inner_radius:
                color = "#92400e"
            elif distance_sq < ring_radius * ring_radius and abs(distance_sq - ring_radius * ring_radius) < size * 1.8:
                color = "#fde68a"
            elif (x - highlight_x) ** 2 + (y - highlight_y) ** 2 < size * 3.2:
                color = "#fff7ad"
            elif dx > size * 0.18 and dy > size * 0.12:
                color = "#d97706"
            else:
                color = "#fbbf24"
            image.put(color, (x, y))

    mark_color = "#78350f"
    mark_light = "#fef3c7"
    for y in range(18, 47):
        image.put(mark_color, (31, y))
        image.put(mark_light, (32, y))
    for x in range(23, 42):
        for y in (22, 23, 40, 41):
            image.put(mark_color if y == 23 or y == 41 else mark_light, (x, y))
    for x in range(22, 30):
        image.put(mark_color, (x, 24))
        image.put(mark_color, (x, 25))
    for x in range(33, 42):
        image.put(mark_color, (x, 38))
        image.put(mark_color, (x, 39))
    return image


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value else "0"


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_symbol_text(value: str, fallback: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        return fallback
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return symbol


def _numeric_reader(cast: object) -> object:
    def read(variable: tk.StringVar, fallback: int | float) -> int | float:
        raw = str(variable.get()).strip()
        try:
            return cast(raw)  # type: ignore[misc]
        except (TypeError, ValueError):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", raw.replace(",", ""))
            if match:
                try:
                    value = cast(match.group(0))  # type: ignore[misc]
                    variable.set(str(value))
                    return value
                except (TypeError, ValueError):
                    pass
            variable.set(str(fallback))
            return fallback

    return read


def _float_from_trade(trade: dict, key: str) -> float:
    try:
        return float(trade.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_trade_time(trade: dict) -> str:
    try:
        timestamp = int(trade.get("time", 0)) / 1000.0
    except (TypeError, ValueError):
        timestamp = 0.0
    if timestamp <= 0:
        return "unknown-time"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _summarize_user_trades(trades: list[dict]) -> dict[str, float | int | str]:
    closed_orders: dict[tuple[str, str], float] = {}
    realized_pnl = 0.0
    usdt_commission = 0.0
    commission_by_asset: dict[str, float] = {}
    for trade in trades:
        pnl = _float_from_trade(trade, "realizedPnl")
        realized_pnl += pnl
        commission = _float_from_trade(trade, "commission")
        asset = str(trade.get("commissionAsset", "") or "?")
        commission_by_asset[asset] = commission_by_asset.get(asset, 0.0) + commission
        if asset == "USDT":
            usdt_commission += commission
        if abs(pnl) > 0:
            key = (str(trade.get("symbol", "")), str(trade.get("orderId", trade.get("id", ""))))
            closed_pnl = pnl - commission if asset == "USDT" else pnl
            closed_orders[key] = closed_orders.get(key, 0.0) + closed_pnl

    closed_values = list(closed_orders.values())
    wins = sum(1 for value in closed_values if value > 0)
    losses = sum(1 for value in closed_values if value <= 0)
    closed_count = wins + losses
    win_rate = wins / closed_count * 100.0 if closed_count else 0.0
    avg_closed_pnl = sum(closed_values) / closed_count if closed_count else 0.0
    commission_text = ", ".join(
        f"{amount:.6g}{asset}" for asset, amount in sorted(commission_by_asset.items()) if amount
    )
    return {
        "fills": len(trades),
        "closed_orders": closed_count,
        "win_rate_pct": win_rate,
        "realized_pnl": realized_pnl,
        "net_pnl": realized_pnl - usdt_commission,
        "avg_closed_pnl": avg_closed_pnl,
        "commission_text": commission_text,
    }


def _position_symbols_first(symbols: tuple[str, ...], rows_by_symbol: dict[str, list]) -> list[str]:
    active = [symbol for symbol in symbols if rows_by_symbol.get(symbol)]
    inactive = [symbol for symbol in symbols if not rows_by_symbol.get(symbol)]
    extras = [symbol for symbol in rows_by_symbol if symbol not in symbols]
    return active + extras + inactive


def _indicator_strategy_enabled(config: LiveAppConfig) -> bool:
    strategy = config.strategy
    if (
        (
            getattr(strategy, "mtf_4h_rsi_regime_enabled", False)
            or getattr(strategy, "mtf_momentum_reset_enabled", False)
        )
        and getattr(strategy, "mtf_disable_legacy_strategies", False)
    ):
        return False
    return bool(
        config.filters.extreme_reversal_entry_enabled
        or getattr(strategy, "indicator_confirmed_cross_extreme_required_enabled", False)
        or getattr(strategy, "indicator_reversal_size_multiplier", 0.0) > 0
    )


def _strategy_short_name(entry_reason: str | None) -> str:
    reason = (entry_reason or "").lower()
    if not reason:
        return "未知"
    if TREND_GRID_SHADOW_REASON_TOKEN in reason:
        return "DTG"
    if DUAL_THRUST_SHADOW_REASON_TOKEN in reason:
        return "DT50"
    if MTF_MOMENTUM_RESET_SETUP_TOKEN in reason:
        return "MR30"
    if MTF_REASON_TOKEN in reason:
        return "MTF"
    if reason.startswith("vbp_") or "volume_breakout_pullback" in reason:
        return "VBP"
    if "indicator_" in reason or "macd_" in reason:
        return "IND"
    if "super_volume" in reason or "startup_breakout" in reason:
        return "SV"
    if "macro" in reason:
        return "MACRO"
    if "trend" in reason or "breakout" in reason:
        return "TRD"
    return "其他"


def _config_with_strategy_selection(
    config: LiveAppConfig,
    *,
    indicator_enabled: bool,
    vbp_enabled: bool,
) -> LiveAppConfig:
    if config.combined_breakout_v8_grid_v6_live.enabled:
        return replace(
            config,
            trading=replace(
                config.trading,
                dry_run=False,
                max_open_positions=2,
                max_new_entries_per_cycle=2,
                max_scale_ins_per_symbol=0,
                allow_loss_scale_in=False,
                profit_exit_enabled=False,
            ),
            dual_thrust_shadow=replace(
                config.dual_thrust_shadow, enabled=False
            ),
            combined_volatility_trend_grid_shadow=replace(
                config.combined_volatility_trend_grid_shadow,
                enabled=False,
            ),
            filters=replace(
                config.filters,
                enabled=False,
                extreme_reversal_entry_enabled=False,
                pre_cross_entry_enabled=False,
            ),
            vbp_strategy=replace(config.vbp_strategy, enabled=False),
            portfolio_control=replace(
                config.portfolio_control, enabled=False
            ),
            regime_score=replace(config.regime_score, enabled=False),
            reversal_alpha=replace(
                config.reversal_alpha, enabled=False
            ),
            cmipr=replace(config.cmipr, enabled=False),
            mtper=replace(config.mtper, enabled=False),
            mtpc=replace(config.mtpc, enabled=False),
            macro_events=replace(config.macro_events, enabled=False),
        )
    if config.combined_volatility_trend_grid_shadow.enabled:
        return replace(
            config,
            trading=replace(
                config.trading,
                dry_run=True,
                max_open_positions=2,
                max_new_entries_per_cycle=2,
                max_scale_ins_per_symbol=0,
                allow_loss_scale_in=False,
                profit_exit_enabled=False,
            ),
            filters=replace(
                config.filters,
                enabled=False,
                extreme_reversal_entry_enabled=False,
                pre_cross_entry_enabled=False,
            ),
            vbp_strategy=replace(config.vbp_strategy, enabled=False),
            portfolio_control=replace(config.portfolio_control, enabled=False),
            regime_score=replace(config.regime_score, enabled=False),
            reversal_alpha=replace(config.reversal_alpha, enabled=False),
            cmipr=replace(config.cmipr, enabled=False),
            mtper=replace(config.mtper, enabled=False),
            mtpc=replace(config.mtpc, enabled=False),
            macro_events=replace(config.macro_events, enabled=False),
        )
    if config.dual_thrust_shadow.enabled:
        return replace(
            config,
            trading=replace(
                config.trading,
                dry_run=True,
                max_open_positions=1,
                max_new_entries_per_cycle=1,
                max_scale_ins_per_symbol=0,
                allow_loss_scale_in=False,
                profit_exit_enabled=False,
            ),
            filters=replace(
                config.filters,
                enabled=False,
                extreme_reversal_entry_enabled=False,
                pre_cross_entry_enabled=False,
            ),
            vbp_strategy=replace(config.vbp_strategy, enabled=False),
            portfolio_control=replace(config.portfolio_control, enabled=False),
            regime_score=replace(config.regime_score, enabled=False),
            reversal_alpha=replace(config.reversal_alpha, enabled=False),
            cmipr=replace(config.cmipr, enabled=False),
            mtper=replace(config.mtper, enabled=False),
            mtpc=replace(config.mtpc, enabled=False),
            macro_events=replace(config.macro_events, enabled=False),
        )
    if (
        (
            getattr(config.strategy, "mtf_4h_rsi_regime_enabled", False)
            or getattr(config.strategy, "mtf_momentum_reset_enabled", False)
        )
        and getattr(config.strategy, "mtf_disable_legacy_strategies", False)
    ):
        return replace(
            config,
            strategy=replace(config.strategy, mtf_max_open_positions=1),
            trading=replace(
                config.trading,
                max_open_positions=1,
                max_new_entries_per_cycle=1,
                initial_entry_fraction=1.0,
                super_volume_extra_slot_enabled=False,
                max_scale_ins_per_symbol=0,
                allow_loss_scale_in=False,
                profit_exit_enabled=False,
            ),
            filters=replace(
                config.filters,
                enabled=False,
                extreme_reversal_entry_enabled=False,
                pre_cross_entry_enabled=False,
            ),
            vbp_strategy=replace(config.vbp_strategy, enabled=False),
            portfolio_control=replace(config.portfolio_control, enabled=False),
            regime_score=replace(config.regime_score, enabled=False),
            reversal_alpha=replace(config.reversal_alpha, enabled=False),
            cmipr=replace(config.cmipr, enabled=False),
            mtper=replace(config.mtper, enabled=False),
            mtpc=replace(config.mtpc, enabled=False),
            macro_events=replace(config.macro_events, enabled=False),
        )
    trading = replace(config.trading, max_open_positions=5)
    vbp_strategy = replace(config.vbp_strategy, enabled=vbp_enabled)
    portfolio_control = replace(
        config.portfolio_control,
        enabled=True,
        max_open_positions=5,
        max_indicator_positions=3 if indicator_enabled else 0,
        max_vbp_positions=2 if vbp_enabled else 0,
    )
    filters = config.filters
    if not indicator_enabled:
        filters = replace(filters, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
    return replace(
        config,
        trading=trading,
        filters=filters,
        vbp_strategy=vbp_strategy,
        portfolio_control=portfolio_control,
    )


def _vbp_strategy_symbols(config: LiveAppConfig) -> tuple[str, ...]:
    symbols = tuple(getattr(config.vbp_strategy, "enabled_symbols", ()) or ())
    if symbols:
        return symbols
    return tuple(config.trading.entry_symbols or config.trading.symbols)


def _lock_live_authorization_for_persistence(
    config: LiveAppConfig,
) -> LiveAppConfig:
    if not config.combined_breakout_v8_grid_v6_live.enabled:
        return config
    return replace(
        config,
        trading=replace(
            config.trading, mainnet_confirmation_text=""
        ),
        combined_breakout_v8_grid_v6_live=replace(
            config.combined_breakout_v8_grid_v6_live,
            armed=False,
            live_confirmation_text="",
        ),
    )


def _execution_mode_for_config(
    config: LiveAppConfig | None,
) -> str:
    if (
        config is not None
        and config.combined_breakout_v8_grid_v6_live.enabled
    ):
        return EXECUTION_MODE_LIVE
    return EXECUTION_MODE_DRY_RUN


def _account_display_equity_baseline(
    config: LiveAppConfig,
    live_refresh_baseline: float | None,
) -> float:
    """Return the GUI-only baseline without changing strategy risk capital."""

    if config.combined_breakout_v8_grid_v6_live.enabled:
        if live_refresh_baseline is not None:
            return live_refresh_baseline
        return config.risk.starting_capital_usdt
    shadow = config.combined_volatility_trend_grid_shadow
    if (
        shadow.enabled
        and shadow.strategy_name
        in {COMBINED_V9_GRID_V7_NAME, COMBINED_V8_GRID_V6_NAME}
    ):
        return V9_V7_DRY_RUN_DISPLAY_BASELINE_USDT
    return config.risk.starting_capital_usdt


def _authorize_live_gui_session(
    config: LiveAppConfig,
) -> LiveAppConfig:
    """Apply non-persistent authorization after the single GUI warning."""

    live = config.combined_breakout_v8_grid_v6_live
    return replace(
        config,
        trading=replace(
            config.trading,
            mainnet_confirmation_text="CONFIRM_MAINNET",
        ),
        combined_breakout_v8_grid_v6_live=replace(
            live,
            armed=True,
            live_confirmation_text=(
                live.required_live_confirmation_text
            ),
        ),
    )


def _detect_strategy_mode(config: LiveAppConfig) -> str:
    if config.combined_breakout_v8_grid_v6_live.enabled:
        return STRATEGY_MODE_COMBINED_LIVE
    if config.combined_volatility_trend_grid_shadow.enabled:
        return STRATEGY_MODE_COMBINED_SHADOW
    if config.dual_thrust_shadow.enabled:
        return STRATEGY_MODE_DUAL_THRUST_SHADOW
    strategy = config.strategy
    if getattr(strategy, "mtf_momentum_reset_enabled", False):
        return STRATEGY_MODE_MTF_RESET
    if getattr(strategy, "mtf_4h_rsi_regime_enabled", False):
        return STRATEGY_MODE_MTF
    if getattr(strategy, "oi_flush_reversal_enabled", False):
        return STRATEGY_MODE_OI_FLUSH
    indicator_only = (
        config.filters.extreme_reversal_entry_enabled
        and getattr(strategy, "indicator_confirmed_cross_extreme_required_enabled", False)
        and getattr(strategy, "allow_short", False)
        and not getattr(strategy, "ordinary_breakout_enabled", False)
        and not getattr(strategy, "pullback_reclaim_enabled", False)
        and not getattr(strategy, "fast_breakout_enabled", False)
        and not getattr(strategy, "rsi_reversal_enabled", False)
    )
    if getattr(strategy, "super_volume_breakout_enabled", False) or getattr(strategy, "startup_breakout_enabled", False):
        return STRATEGY_MODE_SUPER_VOLUME
    if indicator_only:
        return STRATEGY_MODE_INDICATOR
    return STRATEGY_MODE_MANUAL


def _config_with_strategy_mode(config: LiveAppConfig, mode: str) -> LiveAppConfig:
    if mode == STRATEGY_MODE_MANUAL:
        return config

    strategy = config.strategy
    filters = config.filters
    exchange = config.exchange
    trading = config.trading
    risk = config.risk
    dual_thrust_shadow = replace(config.dual_thrust_shadow, enabled=False)
    combined_shadow = replace(
        config.combined_volatility_trend_grid_shadow, enabled=False
    )
    combined_live = replace(
        config.combined_breakout_v8_grid_v6_live, enabled=False
    )
    disable_legacy_breakout = {
        "super_volume_breakout_enabled": False,
        "startup_breakout_enabled": False,
        "ordinary_breakout_enabled": False,
        "pullback_reclaim_enabled": False,
        "fast_breakout_enabled": False,
        "spike_trade_enabled": False,
        "rsi_reversal_enabled": False,
        "mtf_momentum_reset_enabled": False,
    }

    if mode == STRATEGY_MODE_COMBINED_LIVE:
        exchange = replace(
            exchange,
            environment="mainnet",
            api_key_env="BINANCE_FUTURES_API_KEY",
            api_secret_env="BINANCE_FUTURES_API_SECRET",
        )
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=True,
            oi_flush_reversal_enabled=False,
            allow_short=False,
        )
        filters = replace(
            filters,
            enabled=False,
            extreme_reversal_entry_enabled=False,
            pre_cross_entry_enabled=False,
        )
        trading = replace(
            trading,
            timeframe="1m",
            poll_seconds=15,
            entry_scan_seconds=60,
            dry_run=False,
            leverage=10,
            margin_type="CROSSED",
            max_open_positions=2,
            max_new_entries_per_cycle=2,
            initial_entry_fraction=1.0,
            scale_in_entry_fraction=0.0,
            super_volume_extra_slot_enabled=False,
            max_scale_ins_per_symbol=0,
            allow_loss_scale_in=False,
            profit_exit_enabled=False,
        )
        combined_live = replace(combined_live, enabled=True)
    elif mode == STRATEGY_MODE_COMBINED_SHADOW:
        exchange = replace(
            exchange,
            environment="mainnet",
            api_key_env="BINANCE_FUTURES_API_KEY",
            api_secret_env="BINANCE_FUTURES_API_SECRET",
        )
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=True,
            oi_flush_reversal_enabled=False,
            allow_short=False,
        )
        filters = replace(
            filters,
            enabled=False,
            extreme_reversal_entry_enabled=False,
            pre_cross_entry_enabled=False,
        )
        trading = replace(
            trading,
            timeframe="1m",
            poll_seconds=15,
            entry_scan_seconds=60,
            dry_run=True,
            max_open_positions=2,
            max_new_entries_per_cycle=2,
            initial_entry_fraction=1.0,
            scale_in_entry_fraction=0.0,
            super_volume_extra_slot_enabled=False,
            max_scale_ins_per_symbol=0,
            allow_loss_scale_in=False,
            profit_exit_enabled=False,
        )
        risk = replace(
            risk,
            starting_capital_usdt=200.0,
            risk_per_trade_pct=dual_thrust_shadow.risk_per_trade_pct,
            max_account_margin_usage_pct=0.95,
            max_symbol_margin_pct=0.95,
            max_position_notional_usdt=1800.0,
            max_drawdown_pct=0.60,
            starting_capital_drawdown_stop_pct=0.60,
            soft_drawdown_reduce_pct=0.60,
            soft_drawdown_stop_pct=0.60,
            soft_drawdown_min_size_multiplier=1.0,
            backtest_mode="conservative",
            cost_experiment="full_cost",
            funding_enabled=True,
        )
        dual_thrust_shadow = replace(
            dual_thrust_shadow, enabled=True, shadow_only=True
        )
        combined_shadow = replace(combined_shadow, enabled=True, shadow_only=True)
    elif mode == STRATEGY_MODE_DUAL_THRUST_SHADOW:
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=True,
            oi_flush_reversal_enabled=False,
            allow_short=False,
        )
        filters = replace(filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(
            trading,
            timeframe="1m",
            poll_seconds=15,
            entry_scan_seconds=60,
            dry_run=True,
            max_open_positions=1,
            max_new_entries_per_cycle=1,
            initial_entry_fraction=1.0,
            scale_in_entry_fraction=0.0,
            super_volume_extra_slot_enabled=False,
            max_scale_ins_per_symbol=0,
            allow_loss_scale_in=False,
            profit_exit_enabled=False,
        )
        risk = replace(
            risk,
            starting_capital_usdt=2000.0,
            risk_per_trade_pct=0.025,
            max_account_margin_usage_pct=0.95,
            max_symbol_margin_pct=0.95,
            max_position_notional_usdt=1800.0,
            max_drawdown_pct=0.60,
            starting_capital_drawdown_stop_pct=0.60,
            soft_drawdown_reduce_pct=0.60,
            soft_drawdown_stop_pct=0.60,
            soft_drawdown_min_size_multiplier=1.0,
            backtest_mode="conservative",
            cost_experiment="full_cost",
            funding_enabled=True,
        )
        dual_thrust_shadow = replace(dual_thrust_shadow, enabled=True, shadow_only=True)
    elif mode == STRATEGY_MODE_INDICATOR:
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=False,
            oi_flush_reversal_enabled=False,
            trend_reference_filter_enabled=False,
            btc_market_filter_enabled=False,
            weak_market_long_filter_enabled=False,
            strong_market_short_filter_enabled=True,
            strong_market_breadth_threshold=0.58,
            strong_market_avg_return_threshold=0.006,
            strong_market_short_min_rank_score=6.20,
            strong_market_short_risk_multiplier=0.45,
            entry_timing_filter_enabled=True,
            entry_execution_filter_enabled=True,
            entry_execution_filter_trend_only=False,
            entry_execution_timeframe="5m",
            allow_short=True,
            short_risk_bias=1.05,
            indicator_reversal_size_multiplier=0.30,
            indicator_long_size_multiplier=0.282,
            indicator_short_size_multiplier=0.34,
            indicator_long_stop_loss_atr=2.60,
            indicator_short_stop_loss_atr=2.50,
            indicator_long_take_profit_atr=1.60,
            indicator_short_take_profit_atr=1.60,
            indicator_long_max_holding_bars=18,
            indicator_short_max_holding_bars=18,
            indicator_short_max_close_position=0.55,
            indicator_short_high_close_risk_multiplier=0.65,
            indicator_long_confirmed_cross_risk_multiplier=0.45,
            indicator_short_confirmed_cross_risk_multiplier=0.45,
            indicator_long_pre_cross_risk_multiplier=0.35,
            indicator_short_pre_cross_risk_multiplier=0.35,
            indicator_max_holding_bars=18,
            indicator_confirmed_cross_extreme_required_enabled=True,
            indicator_confirmed_cross_extreme_guard_enabled=False,
            indicator_trend_guard_enabled=False,
            indicator_reference_guard_enabled=False,
            indicator_reversal_loss_pause_enabled=False,
            indicator_reversal_loss_pause_losses=2,
            indicator_reversal_loss_pause_bars=8,
            indicator_long_reclaim_filter_enabled=False,
            indicator_long_reclaim_ema_period=9,
            indicator_long_reclaim_min_close_position=0.55,
            indicator_long_fail_fast_enabled=False,
            indicator_long_fail_fast_minutes=120,
            indicator_long_fail_fast_min_r=0.25,
            indicator_min_rank_guard_enabled=False,
        )
        filters = replace(
            filters,
            enabled=False,
            extreme_reversal_entry_enabled=True,
            pre_cross_entry_enabled=False,
            reversal_cross_lookback_bars=1,
            confirmed_cross_risk_multiplier=0.45,
        )
        trading = replace(trading, max_open_positions=4, super_volume_extra_slot_enabled=False)
        risk = replace(risk, risk_per_trade_pct=0.065)
    elif mode == STRATEGY_MODE_SUPER_VOLUME:
        super_volume_flags = dict(disable_legacy_breakout)
        super_volume_flags["super_volume_breakout_enabled"] = True
        strategy = replace(
            strategy,
            **super_volume_flags,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=False,
            oi_flush_reversal_enabled=False,
            super_volume_allow_short=False,
            entry_timing_filter_enabled=True,
            entry_execution_filter_enabled=True,
            entry_execution_filter_trend_only=False,
        )
        filters = replace(filters, enabled=True, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(trading, super_volume_extra_slot_enabled=True)
    elif mode == STRATEGY_MODE_MTF_RESET:
        reset_strategy_flags = dict(disable_legacy_breakout)
        reset_strategy_flags["mtf_momentum_reset_enabled"] = True
        strategy = replace(
            strategy,
            **reset_strategy_flags,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=True,
            mtf_allow_long=True,
            mtf_allow_short=False,
            allow_short=False,
            mtf_symbols_mode="top30",
            mtf_regime_timeframe="4h",
            mtf_regime_mode="trend_pullback",
            mtf_min_rank_score=-999.0,
            mtf_long_min_rank_score=4.25,
            mtf_short_min_rank_score=-999.0,
            mtf_btc_1h_long_min_return_pct=-0.006,
            mtf_btc_4h_long_min_return_pct=-0.001,
            mtf_btc_4h_block_strong_opposite=True,
            mtf_use_funding_filter=True,
            mtf_long_min_funding_rate=0.0,
            mtf_long_max_funding_rate=0.0002,
            mtf_use_oi_filter=False,
            mtf_min_stop_pct=0.0,
            mtf_max_stop_pct=0.015,
            mtf_min_target_to_cost_ratio=12.0,
            mtf_take_profit_r=2.0,
            mtf_risk_multiplier=1.0,
            mtf_exit_mode="fixed_tp",
            mtf_profit_protection_enabled=False,
            mtf_exit_on_30m_confirm_lost=False,
            mtf_exit_on_1h_confirm_lost=True,
            mtf_fail_fast_minutes=120,
            mtf_fail_fast_min_r=0.5,
            mtf_max_holding_minutes=720,
            mtf_max_daily_trades=2,
            mtf_max_open_positions=1,
            mtf_symbol_cooldown_hours=12,
            mtf_momentum_reset_max_signal_age_minutes=6,
            mtf_momentum_reset_event_cluster_hours=4,
            mtf_momentum_reset_min_breadth_ema21=0.55,
            mtf_momentum_reset_min_breadth_symbols=20,
            mtf_momentum_reset_breadth_cache_seconds=60,
            mtf_momentum_reset_priority_mode="target_to_cost_then_rank",
            oi_flush_reversal_enabled=False,
        )
        filters = replace(filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(
            trading,
            timeframe="30m",
            poll_seconds=30,
            entry_scan_seconds=60,
            symbol_reentry_cooldown_seconds=43200,
            max_open_positions=1,
            max_new_entries_per_cycle=1,
            initial_entry_fraction=1.0,
            scale_in_entry_fraction=0.0,
            super_volume_extra_slot_enabled=False,
            max_scale_ins_per_symbol=0,
            allow_loss_scale_in=False,
            profit_exit_enabled=False,
        )
        risk = replace(
            risk,
            starting_capital_usdt=200.0,
            risk_per_trade_pct=0.0321839081,
            max_account_margin_usage_pct=0.30,
            max_symbol_margin_pct=0.30,
            max_drawdown_pct=0.40,
            starting_capital_drawdown_stop_pct=0.40,
            weekly_profit_drawdown_stop_pct=0.40,
            soft_drawdown_reduce_pct=0.20,
            soft_drawdown_stop_pct=0.35,
            soft_drawdown_min_size_multiplier=0.50,
        )
    elif mode == STRATEGY_MODE_MTF:
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=True,
            mtf_disable_legacy_strategies=True,
            mtf_max_open_positions=1,
            mtf_min_rank_score=-999.0,
            mtf_long_min_rank_score=3.5,
            mtf_short_min_rank_score=-999.0,
            mtf_btc_4h_long_min_return_pct=-1.0,
            mtf_risk_multiplier=1.0,
            mtf_exit_mode="fixed_tp",
            oi_flush_reversal_enabled=False,
        )
        filters = replace(filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(
            trading,
            max_open_positions=1,
            max_new_entries_per_cycle=1,
            initial_entry_fraction=1.0,
            super_volume_extra_slot_enabled=False,
            max_scale_ins_per_symbol=0,
            allow_loss_scale_in=False,
            profit_exit_enabled=False,
        )
        risk = replace(
            risk,
            starting_capital_usdt=200.0,
            risk_per_trade_pct=0.0275862069,
            max_account_margin_usage_pct=0.30,
            max_symbol_margin_pct=0.30,
            max_drawdown_pct=0.40,
            starting_capital_drawdown_stop_pct=0.40,
            weekly_profit_drawdown_stop_pct=0.40,
            soft_drawdown_reduce_pct=0.20,
            soft_drawdown_stop_pct=0.35,
            soft_drawdown_min_size_multiplier=0.50,
        )
    elif mode == STRATEGY_MODE_OI_FLUSH:
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=False,
            mtf_disable_legacy_strategies=False,
            oi_flush_reversal_enabled=True,
            allow_short=False,
            short_risk_bias=0.0,
        )
        filters = replace(filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(trading, super_volume_extra_slot_enabled=False)

    updated = replace(
        config,
        exchange=exchange,
        trading=trading,
        strategy=strategy,
        filters=filters,
        risk=risk,
        dual_thrust_shadow=dual_thrust_shadow,
        combined_volatility_trend_grid_shadow=combined_shadow,
        combined_breakout_v8_grid_v6_live=combined_live,
    )
    if mode in {
        STRATEGY_MODE_COMBINED_SHADOW,
        STRATEGY_MODE_COMBINED_LIVE,
        STRATEGY_MODE_DUAL_THRUST_SHADOW,
        STRATEGY_MODE_MTF_RESET,
        STRATEGY_MODE_MTF,
    }:
        updated = replace(
            updated,
            vbp_strategy=replace(updated.vbp_strategy, enabled=False),
            portfolio_control=replace(updated.portfolio_control, enabled=False),
            regime_score=replace(updated.regime_score, enabled=False),
            reversal_alpha=replace(updated.reversal_alpha, enabled=False),
            cmipr=replace(updated.cmipr, enabled=False),
            mtper=replace(updated.mtper, enabled=False),
            mtpc=replace(updated.mtpc, enabled=False),
            macro_events=replace(updated.macro_events, enabled=False),
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", default=ACTIVE_GUI_CONFIG_PATH)
    args = parser.parse_args()
    app = TradingApp(initial_config_path=args.config)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
