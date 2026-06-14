from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .binance_client import BinanceFuturesClient
from .live_config import (
    DEFAULT_SYMBOLS,
    ExchangeConfig,
    LiveAppConfig,
    LiveRiskConfig,
    LiveTradingConfig,
    default_live_config,
    load_live_config,
    write_live_config,
)
from .live_trader import AccountSnapshot, BinanceAutoTrader
from .secrets import mask_secret, read_secret


DEFAULT_CONFIG_PATH = "config.live.optimized_super_volume.json"
FALLBACK_CONFIG_PATH = "config.live.example.json"
STRATEGY_MODE_INDICATOR = "指标反转稳定版"
STRATEGY_MODE_SUPER_VOLUME = "强放量突破"
STRATEGY_MODE_MTF = "MTF多周期"
STRATEGY_MODE_OI_FLUSH = "OI去杠杆反弹"
STRATEGY_MODE_MANUAL = "手动配置"
STRATEGY_MODE_VALUES = (
    STRATEGY_MODE_INDICATOR,
    STRATEGY_MODE_SUPER_VOLUME,
    STRATEGY_MODE_MTF,
    STRATEGY_MODE_OI_FLUSH,
    STRATEGY_MODE_MANUAL,
)
STRATEGY_MODE_SUMMARIES = {
    STRATEGY_MODE_INDICATOR: "当前启用：indicator_reversal 多空分离版。20x/持仓4，risk 0.065，多头0.282/空头0.30，其它策略关闭。",
    STRATEGY_MODE_SUPER_VOLUME: "启用强放量突破策略；适合捕捉高量能趋势启动，旧突破/回踩/反转策略保持关闭。",
    STRATEGY_MODE_MTF: "启用 MTF 多周期策略；旧策略关闭，由多周期信号单独筛选入场。",
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
    "profit": "#22c55e",
    "loss": "#f43f5e",
    "info": "#38bdf8",
}


class TradingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Crypto Scalper - Binance Futures")
        self.geometry("1320x760")
        self.minsize(1080, 620)
        self.configure(bg=THEME["root"])
        self._window_icon: tk.PhotoImage | None = None
        self._apply_window_icon()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.account_queue: queue.Queue[tuple[AccountSnapshot, bool]] = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.config_path = tk.StringVar(value=DEFAULT_CONFIG_PATH)
        self.summary_vars: dict[str, tk.StringVar] = {}
        self.summary_labels: dict[str, ttk.Label] = {}
        self.symbols_text: tk.Text | None = None
        self._last_config: LiveAppConfig | None = None
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
        self.strategy_mode = tk.StringVar(value=STRATEGY_MODE_INDICATOR)
        self.strategy_mode_summary = tk.StringVar(value=STRATEGY_MODE_SUMMARIES[STRATEGY_MODE_INDICATOR])
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
        style.configure("CardValue.TLabel", background=c["card"], foreground=c["title"], font=("Consolas", 17, "bold"))
        style.configure("Profit.CardValue.TLabel", background=c["card"], foreground=c["profit"], font=("Consolas", 17, "bold"))
        style.configure("Loss.CardValue.TLabel", background=c["card"], foreground=c["loss"], font=("Consolas", 17, "bold"))
        style.configure("Info.CardValue.TLabel", background=c["card"], foreground=c["info"], font=("Consolas", 17, "bold"))
        style.configure("Status.TLabel", background=c["accent_soft"], foreground=c["accent_text"], padding=(14, 8), font=("Microsoft YaHei UI", 10, "bold"))
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
        outer = ttk.Frame(self, style="Root.TFrame", padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(outer, style="Header.TFrame", padding=(14, 12))
        toolbar.pack(fill=tk.X)
        title_block = ttk.Frame(toolbar, style="Header.TFrame")
        title_block.pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(title_block, text="Crypto Scalper", style="HeaderTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(title_block, text="Binance Futures 30m Console", style="HeaderSubtitle.TLabel").pack(anchor=tk.W, pady=(2, 0))

        ttk.Label(toolbar, text="配置文件", style="HeaderLabel.TLabel").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.config_path, width=42).pack(side=tk.LEFT, padx=(8, 6))
        ttk.Button(toolbar, text="加载", command=self.load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存", command=self.save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="检查账户", command=self.check_account, style="Accent.TButton").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Button(toolbar, text="刷新持仓", command=self.refresh_account).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="成交统计", command=self.show_trade_history).pack(side=tk.LEFT, padx=2)

        main = ttk.Frame(outer, style="Root.TFrame")
        main.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        main.columnconfigure(0, weight=0, minsize=405)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
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
        self._build_scrollable_settings(settings_area)
        self._build_controls(left)
        self._build_dashboard(right)
        self._build_positions(right)
        self._build_log(right)

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

        self._combo(execution, "环境", self.environment, ("testnet", "mainnet"), 0)
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
        strategy_fields = ttk.Frame(strategy, style="Panel.TFrame")
        strategy_fields.grid(row=2, column=0, columnspan=4, sticky="ew")
        strategy = strategy_fields

        self._entry(strategy, "快EMA", self.fast_ema, 0, column=0)
        self._entry(strategy, "慢EMA", self.slow_ema, 1, column=0)
        self._entry(strategy, "突破通道", self.channel_period, 2, column=0)
        self._entry(strategy, "量比", self.min_volume_ratio, 3, column=0)
        self._entry(strategy, "突破缓冲", self.breakout_buffer, 4, column=0)
        self._entry(strategy, "EMA差", self.ema_gap, 5, column=0)
        self._entry(strategy, "止损ATR", self.stop_loss_atr, 0, column=2)
        self._entry(strategy, "止盈ATR", self.take_profit_atr, 1, column=2)
        ttk.Checkbutton(strategy, text="插针保护", variable=self.spike_guard_enabled).grid(row=2, column=3, sticky=tk.W, pady=5)
        ttk.Checkbutton(strategy, text="插针反打", variable=self.spike_trade_enabled).grid(row=3, column=3, sticky=tk.W, pady=5)
        self._entry(strategy, "插针范围", self.spike_min_range_atr, 4, column=2)
        self._entry(strategy, "影线ATR", self.spike_min_wick_atr, 5, column=2)
        self._entry(strategy, "影线占比", self.spike_min_wick_ratio, 6, column=0)
        self._entry(strategy, "插针量比", self.spike_min_volume_ratio, 7, column=0)
        self._entry(strategy, "冷却K数", self.spike_block_bars, 6, column=2)
        self._entry(strategy, "收回比例", self.spike_recovery_ratio, 7, column=2)
        self._entry(strategy, "插针止损", self.spike_stop_atr, 8, column=0)
        self._entry(strategy, "插针止盈", self.spike_take_profit_atr, 8, column=2)
        self._entry(strategy, "风险倍数", self.spike_risk_multiplier, 9, column=0)
        self._entry(strategy, "最长持仓", self.spike_max_holding_bars, 9, column=2)
        self._entry(strategy, "ATR周期", self.atr_period, 10, column=0)
        self._entry(strategy, "量均周期", self.volume_period, 10, column=2)
        self._entry(strategy, "最小ATR", self.min_atr_pct, 11, column=0)
        self._entry(strategy, "最大ATR", self.max_atr_pct, 11, column=2)
        self._entry(strategy, "保本ATR", self.breakeven_atr, 12, column=0)
        self._entry(strategy, "移动启动", self.trailing_activation_atr, 12, column=2)
        self._entry(strategy, "移动止损", self.trailing_stop_atr, 13, column=0)
        self._entry(strategy, "常规最长", self.max_holding_bars, 13, column=2)
        ttk.Checkbutton(strategy, text="允许做空", variable=self.allow_short).grid(row=14, column=1, sticky=tk.W, pady=5)
        ttk.Checkbutton(strategy, text="趋势过滤", variable=self.regime_filter_enabled).grid(row=14, column=3, sticky=tk.W, pady=5)
        self._entry(strategy, "多头分数", self.long_score_threshold, 15, column=0)
        self._entry(strategy, "空头分数", self.short_score_threshold, 15, column=2)
        self._entry(strategy, "多头风险", self.long_risk_bias, 16, column=0)
        self._entry(strategy, "空头风险", self.short_risk_bias, 16, column=2)
        self._entry(strategy, "趋势周期", self.regime_lookback, 17, column=0)
        self._entry(strategy, "多头斜率", self.long_min_slow_slope_atr, 17, column=2)
        self._entry(strategy, "空头斜率", self.short_max_slow_slope_atr, 18, column=0)
        ttk.Checkbutton(strategy, text="RSI反打", variable=self.rsi_reversal_enabled).grid(row=18, column=3, sticky=tk.W, pady=5)

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
        self._entry(advanced, "首单比例", self.initial_entry_fraction, 3)
        self._entry(advanced, "盈利补仓比例", self.scale_in_entry_fraction, 4)
        self._entry(advanced, "盈利触发", self.scale_in_min_profit_pct, 5)
        self._entry(advanced, "最多补仓", self.max_scale_ins_per_symbol, 6)
        self._entry(advanced, "补仓冷却秒", self.scale_in_cooldown_seconds, 7)
        ttk.Checkbutton(advanced, text="允许亏损补仓", variable=self.allow_loss_scale_in).grid(row=8, column=1, sticky=tk.W, pady=5)
        self._entry(advanced, "亏损触发", self.loss_scale_in_trigger_pct, 9)
        self._entry(advanced, "亏损补仓比例", self.loss_scale_in_entry_fraction, 10)
        ttk.Label(
            advanced,
            text="实盘前必须是 One-way 单向持仓；主网真实下单还需要取消 Dry-run 并填写 CONFIRM_MAINNET。",
            style="Muted.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=11, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self.start_button = ttk.Button(frame, text="启动", command=self.start_trader, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.stop_button = ttk.Button(frame, text="停止", command=self.stop_trader, style="Danger.TButton", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(
            parent,
            text="Hedge Mode 双向持仓下会阻止真实下单。先改成 One-way 单向，再考虑取消 Dry-run。",
            style="Muted.TLabel",
            wraplength=340,
            justify=tk.LEFT,
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Root.TFrame")
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        panel.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(panel, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="ew")

        cards = ttk.Frame(panel, style="Root.TFrame")
        cards.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for index in range(7):
            cards.columnconfigure(index, weight=1, uniform="summary")

        self._summary_card(cards, "账户权益", "equity", "0.00 U", 0)
        self._summary_card(cards, "可用余额", "available", "0.00 U", 1)
        self._summary_card(cards, "未实现盈亏", "unrealized", "0.00 U", 2)
        self._summary_card(cards, "相对本金盈亏", "capital_pnl", "0.00 U", 3)
        self._summary_card(cards, "强平保证金率", "maintenance_margin_ratio", "0.00%", 4)
        self._summary_card(cards, "仓位占用", "initial_margin_usage", "0.00%", 5)
        self._summary_card(cards, "持仓数量", "position_count", "0", 6)

    def _summary_card(self, parent: ttk.Frame, title: str, key: str, default: str, column: int) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(12, 11))
        frame.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0 if column == 6 else 5))
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
        columns = ("symbol", "side", "size_usdt", "leverage", "entry", "mark", "margin", "pnl", "roe")
        self.positions = ttk.Treeview(panel, columns=columns, show="headings", height=9)
        headings = {
            "symbol": "币种",
            "side": "方向",
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
            self.positions.column(column, width=widths[column], anchor=tk.E if column not in {"symbol", "side"} else tk.W)
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
            height=10,
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

    def _build_strategy_selector(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(parent, text="策略选择", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        selector = ttk.Frame(parent, style="Panel.TFrame")
        selector.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="当前策略").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        combo = ttk.Combobox(
            selector,
            textvariable=self.strategy_mode,
            values=STRATEGY_MODE_VALUES,
            state="readonly",
            width=22,
        )
        combo.grid(row=0, column=1, sticky="ew", pady=5, padx=(0, 8))
        combo.bind("<<ComboboxSelected>>", self._on_strategy_mode_selected)
        ttk.Button(selector, text="应用到表单", command=self._apply_selected_strategy_to_form).grid(row=0, column=2, sticky=tk.E, pady=5)
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
        self.strategy_mode_summary.set(STRATEGY_MODE_SUMMARIES.get(mode, STRATEGY_MODE_SUMMARIES[STRATEGY_MODE_MANUAL]))

    def _selected_strategy_mode(self) -> str:
        mode = self.strategy_mode.get().strip()
        return mode if mode in STRATEGY_MODE_VALUES else STRATEGY_MODE_MANUAL

    def _apply_selected_strategy_to_form(self) -> None:
        mode = self._selected_strategy_mode()
        self._update_strategy_mode_summary()
        if mode == STRATEGY_MODE_INDICATOR:
            self.allow_short.set(True)
            self.short_risk_bias.set("1.05")
            self.risk_per_trade.set("0.065")
            self.max_open_positions.set("4")
            self.spike_trade_enabled.set(False)
            self.rsi_reversal_enabled.set(False)
            self.entry_scan_seconds.set("300")
            self.leverage.set("20")
        elif mode == STRATEGY_MODE_SUPER_VOLUME:
            self.spike_trade_enabled.set(False)
            self.rsi_reversal_enabled.set(False)
        elif mode == STRATEGY_MODE_OI_FLUSH:
            self.allow_short.set(False)
            self.short_risk_bias.set("0.0")
            self.spike_trade_enabled.set(False)
            self.rsi_reversal_enabled.set(False)
        self.log(f"策略已切换为: {mode}。启动或保存时会写入对应策略开关。")

    def _load_initial_config(self) -> None:
        path = Path(DEFAULT_CONFIG_PATH)
        if path.exists():
            config = load_live_config(path)
        elif Path(FALLBACK_CONFIG_PATH).exists():
            config = load_live_config(FALLBACK_CONFIG_PATH)
        else:
            config = default_live_config()
        self._apply_config(config)
        self._render_empty_positions(config)

    def load_config(self) -> None:
        try:
            config = load_live_config(self.config_path.get())
            self._apply_config(config)
            self._render_empty_positions(config)
            self.log(f"已加载配置 {self.config_path.get()}")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    def save_config(self) -> None:
        try:
            write_live_config(self.config_path.get(), self._read_config())
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
            if config.exchange.environment == "mainnet" and not config.trading.dry_run:
                if config.trading.mainnet_confirmation_text != "CONFIRM_MAINNET":
                    messagebox.showerror("主网确认缺失", "真实主网下单前，主网确认文本必须填写 CONFIRM_MAINNET")
                    return
            client = self._client_for_config(config)
            trader = BinanceAutoTrader(config, client, logger=self.log_from_thread, account_callback=self.account_from_thread)
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._run_trader_worker, args=(trader, self.stop_event), daemon=True)
            self.worker.start()
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.log("已请求启动")
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))

    def _run_trader_worker(self, trader: BinanceAutoTrader, stop_event: threading.Event) -> None:
        try:
            trader.run_forever(stop_event)
        except Exception as exc:
            self.log_from_thread(f"交易循环启动失败: {type(exc).__name__}: {exc}")

    def stop_trader(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.log("已请求停止")

    def _client_for_config(self, config: LiveAppConfig) -> BinanceFuturesClient:
        api_key = read_secret(config.exchange.api_key_env)
        api_secret = read_secret(config.exchange.api_secret_env)
        return BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            environment=config.exchange.environment,
            recv_window=config.exchange.recv_window,
            timeout_seconds=config.exchange.timeout_seconds,
        )

    def _apply_config(self, config: LiveAppConfig) -> None:
        self._last_config = config
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
        self.macro_enabled.set(config.macro_events.enabled)
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
        self._update_strategy_mode_summary()
        self.status_var.set(f"{config.exchange.environment.upper()} / {'DRY-RUN' if config.trading.dry_run else 'LIVE'}")

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
        config = LiveAppConfig(exchange=exchange, trading=trading, strategy=strategy, filters=base.filters, risk=risk, macro_events=macro_events)
        return _config_with_strategy_mode(config, self._selected_strategy_mode())

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
            self.positions.insert("", tk.END, values=(symbol, "空仓", "0.00", "-", "-", "-", "0.00", "0.00", "0.00%"), tags=("flat",))

    def _render_account(self, snapshot: AccountSnapshot, sync_starting_capital: bool = False) -> None:
        if sync_starting_capital:
            previous = _safe_float(self.starting_capital.get())
            self.starting_capital.set(f"{snapshot.equity:.2f}")
            if previous is None or abs(previous - snapshot.equity) >= 0.005:
                before = "未设置" if previous is None else f"{previous:.2f}U"
                self.log(f"本金U已同步: {before} -> {snapshot.equity:.2f}U")

        config = self._read_config()
        capital_pnl = snapshot.equity - config.risk.starting_capital_usdt
        self.summary_vars["equity"].set(f"{snapshot.equity:.2f} U")
        self.summary_vars["available"].set(f"{snapshot.available_balance:.2f} U")
        self.summary_vars["unrealized"].set(f"{snapshot.total_unrealized_pnl:+.2f} U")
        self.summary_vars["capital_pnl"].set(f"{capital_pnl:+.2f} U")
        self.summary_vars["maintenance_margin_ratio"].set(f"{snapshot.maintenance_margin_ratio_pct * 100:.2f}%")
        self.summary_vars["initial_margin_usage"].set(f"{snapshot.initial_margin_usage_pct * 100:.2f}%")
        self.summary_vars["position_count"].set(str(len(snapshot.position_rows)))
        self._set_summary_value_style("unrealized", snapshot.total_unrealized_pnl)
        self._set_summary_value_style("capital_pnl", capital_pnl)
        self.summary_labels["equity"].configure(style="Info.CardValue.TLabel")
        self.status_var.set(
            f"{config.exchange.environment.upper()} / {'DRY-RUN' if config.trading.dry_run else 'LIVE'} / "
            f"{snapshot.position_mode} / {'本金已同步' if sync_starting_capital else '更新完成'}"
        )

        self.positions.delete(*self.positions.get_children())
        rows_by_symbol: dict[str, list] = {}
        for position in snapshot.position_rows:
            rows_by_symbol.setdefault(position.symbol, []).append(position)

        for symbol in _position_symbols_first(config.trading.symbols, rows_by_symbol):
            rows = rows_by_symbol.get(symbol)
            if not rows:
                self.positions.insert("", tk.END, values=(symbol, "空仓", "0.00", "-", "-", "-", "0.00", "0.00", "0.00%"), tags=("flat",))
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

    def account_from_thread(self, snapshot: AccountSnapshot, sync_starting_capital: bool = False) -> None:
        self.account_queue.put((snapshot, sync_starting_capital))

    def log(self, message: str) -> None:
        try:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {message}\n")
        except OSError:
            pass
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def _drain_queues(self) -> None:
        while True:
            try:
                snapshot, sync_starting_capital = self.account_queue.get_nowait()
            except queue.Empty:
                break
            self._render_account(snapshot, sync_starting_capital)

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


def _detect_strategy_mode(config: LiveAppConfig) -> str:
    strategy = config.strategy
    if getattr(strategy, "mtf_4h_rsi_regime_enabled", False):
        return STRATEGY_MODE_MTF
    if getattr(strategy, "oi_flush_reversal_enabled", False):
        return STRATEGY_MODE_OI_FLUSH
    if getattr(strategy, "super_volume_breakout_enabled", False) or getattr(strategy, "startup_breakout_enabled", False):
        return STRATEGY_MODE_SUPER_VOLUME
    indicator_only = (
        config.filters.extreme_reversal_entry_enabled
        and getattr(strategy, "indicator_confirmed_cross_extreme_required_enabled", False)
        and getattr(strategy, "allow_short", False)
        and not getattr(strategy, "super_volume_breakout_enabled", False)
        and not getattr(strategy, "ordinary_breakout_enabled", False)
        and not getattr(strategy, "pullback_reclaim_enabled", False)
        and not getattr(strategy, "fast_breakout_enabled", False)
        and not getattr(strategy, "rsi_reversal_enabled", False)
    )
    return STRATEGY_MODE_INDICATOR if indicator_only else STRATEGY_MODE_MANUAL


def _config_with_strategy_mode(config: LiveAppConfig, mode: str) -> LiveAppConfig:
    if mode == STRATEGY_MODE_MANUAL:
        return config

    strategy = config.strategy
    filters = config.filters
    trading = config.trading
    risk = config.risk
    disable_legacy_breakout = {
        "super_volume_breakout_enabled": False,
        "startup_breakout_enabled": False,
        "ordinary_breakout_enabled": False,
        "pullback_reclaim_enabled": False,
        "fast_breakout_enabled": False,
        "spike_trade_enabled": False,
        "rsi_reversal_enabled": False,
    }

    if mode == STRATEGY_MODE_INDICATOR:
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
            allow_short=True,
            short_risk_bias=1.05,
            indicator_reversal_size_multiplier=0.30,
            indicator_long_size_multiplier=0.282,
            indicator_short_size_multiplier=0.30,
            indicator_long_stop_loss_atr=2.60,
            indicator_short_stop_loss_atr=2.50,
            indicator_long_take_profit_atr=1.60,
            indicator_short_take_profit_atr=1.60,
            indicator_long_max_holding_bars=18,
            indicator_short_max_holding_bars=18,
            indicator_long_confirmed_cross_risk_multiplier=0.45,
            indicator_short_confirmed_cross_risk_multiplier=0.45,
            indicator_long_pre_cross_risk_multiplier=0.35,
            indicator_short_pre_cross_risk_multiplier=0.35,
            indicator_max_holding_bars=18,
            indicator_confirmed_cross_extreme_required_enabled=True,
            indicator_confirmed_cross_extreme_guard_enabled=False,
            indicator_trend_guard_enabled=False,
            indicator_reference_guard_enabled=False,
            indicator_reversal_loss_pause_enabled=True,
            indicator_reversal_loss_pause_losses=2,
            indicator_reversal_loss_pause_bars=8,
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
            entry_timing_filter_enabled=True,
            entry_execution_filter_enabled=True,
            entry_execution_filter_trend_only=False,
        )
        filters = replace(filters, enabled=True, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(trading, super_volume_extra_slot_enabled=True)
    elif mode == STRATEGY_MODE_MTF:
        strategy = replace(
            strategy,
            **disable_legacy_breakout,
            mtf_4h_rsi_regime_enabled=True,
            mtf_disable_legacy_strategies=True,
            oi_flush_reversal_enabled=False,
        )
        filters = replace(filters, enabled=False, extreme_reversal_entry_enabled=False, pre_cross_entry_enabled=False)
        trading = replace(trading, super_volume_extra_slot_enabled=False)
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

    return replace(config, trading=trading, strategy=strategy, filters=filters, risk=risk)


def main() -> int:
    app = TradingApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
