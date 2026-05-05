from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import replace
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


DEFAULT_CONFIG_PATH = "config.live.json"
FALLBACK_CONFIG_PATH = "config.live.example.json"
THEME = {
    "root": "#0a1017",
    "panel": "#111923",
    "card": "#172231",
    "card_alt": "#1d2a3a",
    "field": "#080d13",
    "log": "#070b11",
    "border": "#344255",
    "text": "#e6edf3",
    "muted": "#91a4b8",
    "soft": "#cfdae7",
    "title": "#f8fbff",
    "accent": "#14b8a6",
    "accent_active": "#0f766e",
    "accent_soft": "#0d2d32",
    "accent_text": "#b8fff4",
    "button": "#243245",
    "button_active": "#31455f",
    "danger": "#b42318",
    "danger_active": "#991b1b",
    "profit": "#4ade80",
    "loss": "#fb7185",
}


class TradingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Crypto Scalper - Binance Futures")
        self.geometry("1320x820")
        self.minsize(1180, 720)
        self.configure(bg=THEME["root"])

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.account_queue: queue.Queue[AccountSnapshot] = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.config_path = tk.StringVar(value=DEFAULT_CONFIG_PATH)
        self.summary_vars: dict[str, tk.StringVar] = {}
        self.symbols_text: tk.Text | None = None
        self._last_config: LiveAppConfig | None = None

        self._build_vars()
        self._build_style()
        self._build_ui()
        self._load_initial_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._drain_queues)

    def _build_vars(self) -> None:
        self.environment = tk.StringVar()
        self.dry_run = tk.BooleanVar()
        self.api_key_env = tk.StringVar()
        self.api_secret_env = tk.StringVar()
        self.symbols = tk.StringVar()
        self.timeframe = tk.StringVar()
        self.poll_seconds = tk.StringVar()
        self.leverage = tk.StringVar()
        self.max_open_positions = tk.StringVar()
        self.starting_capital = tk.StringVar()
        self.margin_usage = tk.StringVar()
        self.symbol_margin = tk.StringVar()
        self.max_notional = tk.StringVar()
        self.risk_per_trade = tk.StringVar()
        self.daily_loss = tk.StringVar()
        self.max_drawdown = tk.StringVar()
        self.min_available = tk.StringVar()
        self.mainnet_confirmation = tk.StringVar()
        self.fast_ema = tk.StringVar()
        self.slow_ema = tk.StringVar()
        self.channel_period = tk.StringVar()
        self.min_volume_ratio = tk.StringVar()
        self.breakout_buffer = tk.StringVar()
        self.ema_gap = tk.StringVar()
        self.stop_loss_atr = tk.StringVar()
        self.take_profit_atr = tk.StringVar()
        self.spike_guard_enabled = tk.BooleanVar()
        self.spike_trade_enabled = tk.BooleanVar()
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
        self.condition_stats_enabled = tk.BooleanVar()
        self.condition_stats_interval = tk.StringVar()
        self.use_btc_market_state_filter = tk.BooleanVar()
        self.use_symbol_trend_filter = tk.BooleanVar()
        self.use_symbol_range_filter = tk.BooleanVar()
        self.use_btc_direction_filter = tk.BooleanVar()
        self.use_confidence_filter = tk.BooleanVar()
        self.use_cost_edge_filter = tk.BooleanVar()
        self.use_reward_risk_filter = tk.BooleanVar()
        self.use_trend_atr_filter = tk.BooleanVar()
        self.use_trend_adx_filter = tk.BooleanVar()
        self.use_trend_volume_filter = tk.BooleanVar()
        self.use_trend_ema_filter = tk.BooleanVar()
        self.use_trend_setup_filter = tk.BooleanVar()
        self.use_trend_score_filter = tk.BooleanVar()
        self.trend_continuation_entry_enabled = tk.BooleanVar()
        self.trend_continuation_max_holding_bars = tk.StringVar()
        self.use_bollinger_reclaim_entry = tk.BooleanVar()
        self.use_rsi_extreme_entry = tk.BooleanVar()

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
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("Card.TFrame", background=c["card"], borderwidth=1, relief=tk.SOLID)
        style.configure("Toolbar.TFrame", background=c["root"])
        style.configure("TLabel", background=c["panel"], foreground=c["soft"])
        style.configure("ToolbarLabel.TLabel", background=c["root"], foreground=c["soft"])
        style.configure("Title.TLabel", background=c["root"], foreground=c["title"], font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("SectionTitle.TLabel", background=c["panel"], foreground=c["title"], font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Muted.TLabel", background=c["panel"], foreground=c["muted"])
        style.configure("CardTitle.TLabel", background=c["card"], foreground=c["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("CardValue.TLabel", background=c["card"], foreground=c["title"], font=("Consolas", 20, "bold"))
        style.configure("Status.TLabel", background=c["accent_soft"], foreground=c["accent_text"], padding=(12, 7))
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
        )
        style.configure(
            "TCombobox",
            fieldbackground=c["field"],
            background=c["field"],
            foreground=c["text"],
            arrowcolor=c["muted"],
            bordercolor=c["border"],
        )
        style.map("TCombobox", fieldbackground=[("readonly", c["field"])], foreground=[("readonly", c["text"])])
        style.configure("TCheckbutton", background=c["panel"], foreground=c["soft"])
        style.map("TCheckbutton", background=[("active", c["panel"])], foreground=[("active", c["title"])])
        style.configure("TNotebook", background=c["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["card_alt"], foreground=c["muted"], padding=(14, 8), bordercolor=c["border"])
        style.map("TNotebook.Tab", background=[("selected", c["accent"])], foreground=[("selected", "#ffffff")])
        style.configure("Treeview", background=c["field"], fieldbackground=c["field"], foreground=c["text"], rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background=c["card_alt"], foreground=c["title"], font=("Microsoft YaHei UI", 9, "bold"), bordercolor=c["border"])
        style.map("Treeview", background=[("selected", c["accent_active"])], foreground=[("selected", "#ffffff")])
        style.configure("Vertical.TScrollbar", background=c["button"], troughcolor=c["field"], bordercolor=c["border"], arrowcolor=c["muted"])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, style="Root.TFrame", padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(outer, style="Toolbar.TFrame")
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Crypto Scalper", style="Title.TLabel").pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(toolbar, text="配置文件", style="ToolbarLabel.TLabel").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.config_path, width=40).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="加载", command=self.load_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存", command=self.save_config).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="检查账户", command=self.check_account, style="Accent.TButton").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(toolbar, text="刷新持仓", command=self.refresh_account).pack(side=tk.LEFT, padx=2)

        main = ttk.Frame(outer, style="Root.TFrame")
        main.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        main.columnconfigure(0, weight=0, minsize=385)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.rowconfigure(0, weight=0)
        left.rowconfigure(1, weight=1)
        left.rowconfigure(2, weight=0)

        right = ttk.Frame(main, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_settings(left)
        ttk.Frame(left, style="Panel.TFrame").grid(row=1, column=0, sticky="nsew")
        self._build_controls(left)
        self._build_dashboard(right)
        self._build_positions(right)
        self._build_log(right)

    def _build_settings(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)

        execution = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        risk = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        strategy = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        advanced = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        experiment = ttk.Frame(notebook, style="Panel.TFrame", padding=10)
        notebook.add(execution, text="执行")
        notebook.add(risk, text="风控")
        notebook.add(strategy, text="策略")
        notebook.add(advanced, text="高级")

        self._combo(execution, "环境", self.environment, ("testnet", "mainnet"), 0)
        ttk.Checkbutton(execution, text="Dry-run 不真实下单", variable=self.dry_run).grid(row=1, column=1, sticky=tk.W, pady=(4, 8))
        self._entry(execution, "周期", self.timeframe, 2)
        self._entry(execution, "轮询秒数", self.poll_seconds, 3)
        self._entry(execution, "杠杆上限", self.leverage, 4)
        self._entry(execution, "最大持仓数", self.max_open_positions, 5)
        ttk.Label(execution, text="交易币种").grid(row=6, column=0, sticky=tk.NW, pady=5, padx=(0, 8))
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
        self.symbols_text.grid(row=6, column=1, sticky="ew", pady=5)
        notebook.add(experiment, text="实验")
        ttk.Button(execution, text="填入主流币预设", command=self._apply_symbol_preset).grid(row=7, column=1, sticky="ew", pady=(2, 0))

        self._entry(risk, "总保证金上限", self.margin_usage, 0)
        self._entry(risk, "单币保证金上限", self.symbol_margin, 1)
        self._entry(risk, "单仓名义上限U", self.max_notional, 2)
        self._entry(risk, "单笔风险比例", self.risk_per_trade, 3)
        self._entry(risk, "日亏损上限", self.daily_loss, 4)
        self._entry(risk, "最大回撤", self.max_drawdown, 5)
        self._entry(risk, "最低保留U", self.min_available, 6)
        self._entry(risk, "模拟本金U", self.starting_capital, 7)

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

        ttk.Checkbutton(experiment, text="打印条件触发率", variable=self.condition_stats_enabled).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        self._entry(experiment, "统计间隔秒", self.condition_stats_interval, 1)
        ttk.Label(experiment, text="市场过滤", style="SectionTitle.TLabel").grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Checkbutton(experiment, text="BTC市场状态", variable=self.use_btc_market_state_filter).grid(row=3, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="币种趋势对齐", variable=self.use_symbol_trend_filter).grid(row=3, column=1, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="币种震荡状态", variable=self.use_symbol_range_filter).grid(row=4, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="BTC同向过滤", variable=self.use_btc_direction_filter).grid(row=4, column=1, sticky=tk.W, pady=3)
        ttk.Label(experiment, text="边际过滤", style="SectionTitle.TLabel").grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Checkbutton(experiment, text="信心阈值", variable=self.use_confidence_filter).grid(row=6, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="成本/手续费空间", variable=self.use_cost_edge_filter).grid(row=6, column=1, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="盈亏比RR", variable=self.use_reward_risk_filter).grid(row=7, column=0, sticky=tk.W, pady=3)
        ttk.Label(experiment, text="趋势条件", style="SectionTitle.TLabel").grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Checkbutton(experiment, text="ATR过滤", variable=self.use_trend_atr_filter).grid(row=9, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="ADX过滤", variable=self.use_trend_adx_filter).grid(row=9, column=1, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="成交量过滤", variable=self.use_trend_volume_filter).grid(row=10, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="EMA对齐", variable=self.use_trend_ema_filter).grid(row=10, column=1, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="突破/回踩形态", variable=self.use_trend_setup_filter).grid(row=11, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="趋势分数", variable=self.use_trend_score_filter).grid(row=11, column=1, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="顺势延续入场", variable=self.trend_continuation_entry_enabled).grid(row=12, column=0, sticky=tk.W, pady=3)
        self._entry(experiment, "延续持仓K数", self.trend_continuation_max_holding_bars, 13)
        ttk.Label(experiment, text="震荡入场", style="SectionTitle.TLabel").grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Checkbutton(experiment, text="布林收回", variable=self.use_bollinger_reclaim_entry).grid(row=15, column=0, sticky=tk.W, pady=3)
        ttk.Checkbutton(experiment, text="RSI极值反转", variable=self.use_rsi_extreme_entry).grid(row=15, column=1, sticky=tk.W, pady=3)
        ttk.Label(
            experiment,
            text="关闭某项后，它会继续统计真实触发率，但不再拦截开仓。实盘测试前建议先用 dry-run 单独观察。",
            style="Muted.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).grid(row=16, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
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
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Root.TFrame")
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        panel.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(panel, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="ew")

        cards = ttk.Frame(panel, style="Root.TFrame")
        cards.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for index in range(6):
            cards.columnconfigure(index, weight=1)

        self._summary_card(cards, "账户权益", "equity", "0.00 U", 0)
        self._summary_card(cards, "可用余额", "available", "0.00 U", 1)
        self._summary_card(cards, "未实现盈亏", "unrealized", "0.00 U", 2)
        self._summary_card(cards, "相对本金盈亏", "capital_pnl", "0.00 U", 3)
        self._summary_card(cards, "保证金占用", "margin_usage", "0.00%", 4)
        self._summary_card(cards, "持仓数量", "position_count", "0", 5)

    def _summary_card(self, parent: ttk.Frame, title: str, key: str, default: str, column: int) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        frame.grid(row=0, column=column, sticky="nsew", padx=4)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor=tk.W)
        var = tk.StringVar(value=default)
        ttk.Label(frame, textvariable=var, style="CardValue.TLabel").pack(anchor=tk.W, pady=(6, 0))
        self.summary_vars[key] = var

    def _build_positions(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        panel.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="持仓与币种盈亏", style="SectionTitle.TLabel").grid(row=0, column=0, sticky=tk.W)
        columns = ("symbol", "side", "qty", "entry", "mark", "notional", "margin", "pnl", "roe")
        self.positions = ttk.Treeview(panel, columns=columns, show="headings", height=9)
        headings = {
            "symbol": "币种",
            "side": "方向",
            "qty": "数量",
            "entry": "开仓价",
            "mark": "标记价",
            "notional": "名义U",
            "margin": "保证金U",
            "pnl": "盈亏U",
            "roe": "ROE",
        }
        widths = {
            "symbol": 92,
            "side": 78,
            "qty": 92,
            "entry": 86,
            "mark": 86,
            "notional": 86,
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
            self.account_from_thread(snapshot)
            self.log_from_thread(
                f"账户检查成功: 权益={snapshot.equity:.2f}U 可用={snapshot.available_balance:.2f}U "
                f"持仓模式={snapshot.position_mode}"
            )
        except Exception as exc:
            self.log_from_thread(f"账户检查失败: {type(exc).__name__}: {exc}")

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
        self.timeframe.set(config.trading.timeframe)
        self.poll_seconds.set(str(config.trading.poll_seconds))
        self.leverage.set(str(config.trading.leverage))
        self.max_open_positions.set(str(config.trading.max_open_positions))
        self.starting_capital.set(str(config.risk.starting_capital_usdt))
        self.margin_usage.set(str(config.risk.max_account_margin_usage_pct))
        self.symbol_margin.set(str(config.risk.max_symbol_margin_pct))
        self.max_notional.set(str(config.risk.max_position_notional_usdt))
        self.risk_per_trade.set(str(config.risk.risk_per_trade_pct))
        self.daily_loss.set(str(config.risk.max_daily_loss_pct))
        self.max_drawdown.set(str(config.risk.max_drawdown_pct))
        self.min_available.set(str(config.risk.min_available_balance_usdt))
        self.mainnet_confirmation.set(config.trading.mainnet_confirmation_text)
        self.fast_ema.set(str(config.strategy.fast_ema))
        self.slow_ema.set(str(config.strategy.slow_ema))
        self.channel_period.set(str(config.strategy.channel_period))
        self.min_volume_ratio.set(str(config.strategy.min_volume_ratio))
        self.breakout_buffer.set(str(config.strategy.breakout_buffer_atr))
        self.ema_gap.set(str(config.strategy.ema_gap_atr))
        self.stop_loss_atr.set(str(config.strategy.stop_loss_atr))
        self.take_profit_atr.set(str(config.strategy.take_profit_atr))
        self.spike_guard_enabled.set(config.strategy.spike_guard_enabled)
        self.spike_trade_enabled.set(config.strategy.spike_trade_enabled)
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
        self.condition_stats_enabled.set(config.trading.condition_stats_enabled)
        self.condition_stats_interval.set(str(config.trading.condition_stats_log_interval_seconds))
        self.use_btc_market_state_filter.set(config.trading.use_btc_market_state_filter)
        self.use_symbol_trend_filter.set(config.trading.use_symbol_trend_filter)
        self.use_symbol_range_filter.set(config.trading.use_symbol_range_filter)
        self.use_btc_direction_filter.set(config.trading.use_btc_direction_filter)
        self.use_confidence_filter.set(config.trading.use_confidence_filter)
        self.use_cost_edge_filter.set(config.trading.use_cost_edge_filter)
        self.use_reward_risk_filter.set(config.trading.use_reward_risk_filter)
        self.use_trend_atr_filter.set(config.trading.use_trend_atr_filter)
        self.use_trend_adx_filter.set(config.trading.use_trend_adx_filter)
        self.use_trend_volume_filter.set(config.trading.use_trend_volume_filter)
        self.use_trend_ema_filter.set(config.trading.use_trend_ema_filter)
        self.use_trend_setup_filter.set(config.trading.use_trend_setup_filter)
        self.use_trend_score_filter.set(config.trading.use_trend_score_filter)
        self.trend_continuation_entry_enabled.set(config.trading.trend_continuation_entry_enabled)
        self.trend_continuation_max_holding_bars.set(str(config.trading.trend_continuation_max_holding_bars))
        self.use_bollinger_reclaim_entry.set(config.trading.use_bollinger_reclaim_entry)
        self.use_rsi_extreme_entry.set(config.trading.use_rsi_extreme_entry)
        self.status_var.set(f"{config.exchange.environment.upper()} / {'DRY-RUN' if config.trading.dry_run else 'LIVE'}")

    def _read_config(self) -> LiveAppConfig:
        base = self._last_config or default_live_config()
        exchange = replace(
            base.exchange,
            environment=self.environment.get(),
            api_key_env=self.api_key_env.get().strip(),
            api_secret_env=self.api_secret_env.get().strip(),
        )
        trading = replace(
            base.trading,
            symbols=self._read_symbols(),
            timeframe=self.timeframe.get().strip(),
            kline_limit=200,
            poll_seconds=int(self.poll_seconds.get()),
            dry_run=bool(self.dry_run.get()),
            mainnet_confirmation_text=self.mainnet_confirmation.get().strip(),
            leverage=int(self.leverage.get()),
            margin_type="CROSSED",
            max_open_positions=int(self.max_open_positions.get()),
            initial_entry_fraction=float(self.initial_entry_fraction.get()),
            scale_in_entry_fraction=float(self.scale_in_entry_fraction.get()),
            max_scale_ins_per_symbol=int(self.max_scale_ins_per_symbol.get()),
            scale_in_min_profit_pct=float(self.scale_in_min_profit_pct.get()),
            scale_in_cooldown_seconds=int(self.scale_in_cooldown_seconds.get()),
            allow_loss_scale_in=bool(self.allow_loss_scale_in.get()),
            loss_scale_in_trigger_pct=float(self.loss_scale_in_trigger_pct.get()),
            loss_scale_in_entry_fraction=float(self.loss_scale_in_entry_fraction.get()),
            condition_stats_enabled=bool(self.condition_stats_enabled.get()),
            condition_stats_log_interval_seconds=int(self.condition_stats_interval.get()),
            use_btc_market_state_filter=bool(self.use_btc_market_state_filter.get()),
            use_symbol_trend_filter=bool(self.use_symbol_trend_filter.get()),
            use_symbol_range_filter=bool(self.use_symbol_range_filter.get()),
            use_btc_direction_filter=bool(self.use_btc_direction_filter.get()),
            use_confidence_filter=bool(self.use_confidence_filter.get()),
            use_cost_edge_filter=bool(self.use_cost_edge_filter.get()),
            use_reward_risk_filter=bool(self.use_reward_risk_filter.get()),
            use_trend_atr_filter=bool(self.use_trend_atr_filter.get()),
            use_trend_adx_filter=bool(self.use_trend_adx_filter.get()),
            use_trend_volume_filter=bool(self.use_trend_volume_filter.get()),
            use_trend_ema_filter=bool(self.use_trend_ema_filter.get()),
            use_trend_setup_filter=bool(self.use_trend_setup_filter.get()),
            use_trend_score_filter=bool(self.use_trend_score_filter.get()),
            trend_continuation_entry_enabled=bool(self.trend_continuation_entry_enabled.get()),
            trend_continuation_max_holding_bars=int(self.trend_continuation_max_holding_bars.get()),
            use_bollinger_reclaim_entry=bool(self.use_bollinger_reclaim_entry.get()),
            use_rsi_extreme_entry=bool(self.use_rsi_extreme_entry.get()),
        )
        risk = replace(
            base.risk,
            starting_capital_usdt=float(self.starting_capital.get()),
            max_account_margin_usage_pct=float(self.margin_usage.get()),
            max_symbol_margin_pct=float(self.symbol_margin.get()),
            max_position_notional_usdt=float(self.max_notional.get()),
            risk_per_trade_pct=float(self.risk_per_trade.get()),
            max_daily_loss_pct=float(self.daily_loss.get()),
            max_drawdown_pct=float(self.max_drawdown.get()),
            min_available_balance_usdt=float(self.min_available.get()),
        )
        strategy = replace(
            base.strategy,
            fast_ema=int(self.fast_ema.get()),
            slow_ema=int(self.slow_ema.get()),
            channel_period=int(self.channel_period.get()),
            min_volume_ratio=float(self.min_volume_ratio.get()),
            breakout_buffer_atr=float(self.breakout_buffer.get()),
            ema_gap_atr=float(self.ema_gap.get()),
            stop_loss_atr=float(self.stop_loss_atr.get()),
            take_profit_atr=float(self.take_profit_atr.get()),
            spike_guard_enabled=bool(self.spike_guard_enabled.get()),
            spike_trade_enabled=bool(self.spike_trade_enabled.get()),
            spike_min_range_atr=float(self.spike_min_range_atr.get()),
            spike_min_wick_atr=float(self.spike_min_wick_atr.get()),
            spike_min_wick_ratio=float(self.spike_min_wick_ratio.get()),
            spike_min_volume_ratio=float(self.spike_min_volume_ratio.get()),
            spike_block_bars=int(self.spike_block_bars.get()),
            spike_recovery_ratio=float(self.spike_recovery_ratio.get()),
            spike_stop_atr=float(self.spike_stop_atr.get()),
            spike_take_profit_atr=float(self.spike_take_profit_atr.get()),
            spike_risk_multiplier=float(self.spike_risk_multiplier.get()),
            spike_max_holding_bars=int(self.spike_max_holding_bars.get()),
        )
        return LiveAppConfig(exchange=exchange, trading=trading, strategy=strategy, filters=base.filters, risk=risk)

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

    def _render_empty_positions(self, config: LiveAppConfig) -> None:
        self.positions.delete(*self.positions.get_children())
        for symbol in config.trading.symbols:
            self.positions.insert("", tk.END, values=(symbol, "空仓", "-", "-", "-", "0.00", "0.00", "0.00", "0.00%"), tags=("flat",))

    def _render_account(self, snapshot: AccountSnapshot) -> None:
        config = self._read_config()
        capital_pnl = snapshot.equity - config.risk.starting_capital_usdt
        self.summary_vars["equity"].set(f"{snapshot.equity:.2f} U")
        self.summary_vars["available"].set(f"{snapshot.available_balance:.2f} U")
        self.summary_vars["unrealized"].set(f"{snapshot.total_unrealized_pnl:+.2f} U")
        self.summary_vars["capital_pnl"].set(f"{capital_pnl:+.2f} U")
        self.summary_vars["margin_usage"].set(f"{snapshot.margin_usage_pct * 100:.2f}%")
        self.summary_vars["position_count"].set(str(len(snapshot.position_rows)))
        self.status_var.set(
            f"{config.exchange.environment.upper()} / {'DRY-RUN' if config.trading.dry_run else 'LIVE'} / "
            f"{snapshot.position_mode} / 更新完成"
        )

        self.positions.delete(*self.positions.get_children())
        rows_by_symbol: dict[str, list] = {}
        for position in snapshot.position_rows:
            rows_by_symbol.setdefault(position.symbol, []).append(position)

        for symbol in config.trading.symbols:
            rows = rows_by_symbol.get(symbol)
            if not rows:
                self.positions.insert("", tk.END, values=(symbol, "空仓", "-", "-", "-", "0.00", "0.00", "0.00", "0.00%"), tags=("flat",))
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
                        _fmt_float(position.quantity, 6),
                        _fmt_float(position.entry_price, 4),
                        _fmt_float(position.mark_price, 4),
                        _fmt_float(position.notional, 2),
                        _fmt_float(margin, 2),
                        f"{position.unrealized_pnl:+.2f}",
                        f"{roe:+.2f}%",
                    ),
                    tags=(tag,),
                )

    def log_from_thread(self, message: str) -> None:
        self.log_queue.put(message)

    def account_from_thread(self, snapshot: AccountSnapshot) -> None:
        self.account_queue.put(snapshot)

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

    def _drain_queues(self) -> None:
        while True:
            try:
                snapshot = self.account_queue.get_nowait()
            except queue.Empty:
                break
            self._render_account(snapshot)

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


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value else "0"


def main() -> int:
    app = TradingApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
