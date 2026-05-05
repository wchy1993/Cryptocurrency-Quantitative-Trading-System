from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from .config import StrategyConfig


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "DOTUSDT",
    "TRXUSDT",
    "TONUSDT",
    "NEARUSDT",
    "SUIUSDT",
    "APTUSDT",
    "OPUSDT",
    "ARBUSDT",
    "WLDUSDT",
    "INJUSDT",
    "FILUSDT",
    "ETCUSDT",
    "ATOMUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "ICPUSDT",
    "VETUSDT",
    "ALGOUSDT",
    "FETUSDT",
    "RENDERUSDT",
    "POLUSDT",
    "1000PEPEUSDT",
    "1000SHIBUSDT",
    "SEIUSDT",
    "TIAUSDT",
    "TAOUSDT",
    "ENAUSDT",
    "PENDLEUSDT",
    "JUPUSDT",
    "WIFUSDT",
    "1000BONKUSDT",
    "ORDIUSDT",
    "RUNEUSDT",
    "GALAUSDT",
    "SANDUSDT",
    "MANAUSDT",
    "APEUSDT",
)


@dataclass(frozen=True)
class ExchangeConfig:
    environment: str = "testnet"
    api_key_env: str = "BINANCE_FUTURES_API_KEY"
    api_secret_env: str = "BINANCE_FUTURES_API_SECRET"
    recv_window: int = 5_000
    timeout_seconds: int = 10


@dataclass(frozen=True)
class LiveTradingConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    timeframe: str = "15m"
    kline_limit: int = 240
    poll_seconds: int = 8
    dry_run: bool = True
    require_mainnet_confirmation: bool = True
    mainnet_confirmation_text: str = ""
    leverage: int = 5
    margin_type: str = "CROSSED"
    require_one_way_mode: bool = True
    use_market_orders: bool = True
    reduce_only_exit: bool = True
    use_protective_orders: bool = True
    working_type: str = "MARK_PRICE"
    max_open_positions: int = 3
    max_long_positions: int = 3
    max_short_positions: int = 3
    candidate_batch_size: int = 1
    entry_frequency_window_seconds: int = 3600
    max_entries_per_window: int = 10
    max_symbol_entries_per_window: int = 1
    min_symbol_reentry_seconds: int = 1800
    stats_log_interval_seconds: int = 60
    initial_entry_fraction: float = 0.80
    scale_in_entry_fraction: float = 0.20
    max_scale_ins_per_symbol: int = 1
    scale_in_min_profit_pct: float = 0.0040
    scale_in_cooldown_seconds: int = 120
    allow_loss_scale_in: bool = False
    loss_scale_in_trigger_pct: float = 0.0035
    loss_scale_in_entry_fraction: float = 0.12
    min_signal_confidence: float = 0.70
    min_take_profit_cost_ratio: float = 6.0
    min_reward_risk_ratio: float = 2.3
    profit_exit_enabled: bool = True
    breakeven_trigger_pct: float = 0.0050
    breakeven_lock_pct: float = 0.0042
    trailing_activation_pct: float = 0.0065
    trailing_pullback_pct: float = 0.0035
    momentum_exit_min_profit_pct: float = 0.0060
    quick_take_profit_pct: float = 0.0120
    strong_take_profit_pct: float = 0.0200
    profit_exit_rsi_long: float = 72.0
    profit_exit_rsi_short: float = 28.0
    condition_stats_enabled: bool = True
    condition_stats_log_interval_seconds: int = 60
    use_btc_market_state_filter: bool = True
    use_symbol_trend_filter: bool = True
    use_symbol_range_filter: bool = False
    use_btc_direction_filter: bool = True
    use_confidence_filter: bool = True
    use_cost_edge_filter: bool = True
    use_reward_risk_filter: bool = True
    use_trend_atr_filter: bool = True
    use_trend_adx_filter: bool = True
    use_trend_volume_filter: bool = True
    use_trend_ema_filter: bool = True
    use_trend_setup_filter: bool = True
    use_trend_score_filter: bool = True
    trend_continuation_entry_enabled: bool = False
    trend_continuation_max_holding_bars: int = 8
    use_bollinger_reclaim_entry: bool = True
    use_rsi_extreme_entry: bool = True
    session_profit_guard_enabled: bool = True
    session_profit_guard_trigger_usdt: float = 0.35
    session_profit_guard_pullback_usdt: float = 0.20
    session_profit_guard_cooldown_seconds: int = 600
    trend_time_stop_bars: int = 12
    trend_time_stop_min_r: float = 0.5
    mean_reversion_time_stop_bars: int = 8


@dataclass(frozen=True)
class MultiTimeframeFilterConfig:
    enabled: bool = True
    timeframes: tuple[str, ...] = ("5m", "15m", "1h")
    kline_limit: int = 240
    min_score: int = 6
    trend_timeframe: str = "4h"
    range_timeframe: str = "1h"
    adx_period: int = 14
    trend_adx_threshold: float = 24.0
    range_adx_threshold: float = 18.0
    range_ema_distance_pct: float = 0.020
    range_bb_width_percentile_max: float = 70.0
    trend_atr_percentile_min: float = 25.0
    trend_bb_width_percentile_min: float = 25.0
    percentile_lookback: int = 100
    vwap_period: int = 96
    trend_score_entry: int = 5
    trend_score_normal: int = 5
    trend_score_strong: int = 6
    btc_opportunity_adx_threshold: float = 24.0
    vwap_atr_distance_max: float = 2.5
    rsi_period: int = 7
    rsi_long_floor: float = 26.0
    rsi_long_ceiling: float = 82.0
    rsi_short_floor: float = 18.0
    rsi_short_ceiling: float = 74.0
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 5
    kdj_period: int = 5
    extreme_reversal_entry_enabled: bool = True
    pre_cross_entry_enabled: bool = False
    reversal_cross_lookback_bars: int = 3
    long_extreme_rsi: float = 24.0
    short_extreme_rsi: float = 76.0
    long_extreme_kdj: float = 20.0
    short_extreme_kdj: float = 80.0
    confirmed_cross_risk_multiplier: float = 0.65
    pre_cross_risk_multiplier: float = 0.45


@dataclass(frozen=True)
class LiveRiskConfig:
    starting_capital_usdt: float = 10000.0
    max_account_margin_usage_pct: float = 0.10
    max_symbol_margin_pct: float = 0.040
    max_position_notional_usdt: float = 4000.0
    risk_per_trade_pct: float = 0.004
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_portfolio_risk_pct: float = 0.015
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    min_available_balance_usdt: float = 20.0
    min_order_notional_usdt: float = 5.0
    cooldown_seconds_after_loss: int = 0
    max_consecutive_losses: int = 0
    consecutive_loss_cooldown_seconds: int = 0
    max_symbol_consecutive_losses: int = 0
    symbol_loss_cooldown_seconds: int = 0


@dataclass(frozen=True)
class LiveAppConfig:
    exchange: ExchangeConfig
    trading: LiveTradingConfig
    strategy: StrategyConfig
    filters: MultiTimeframeFilterConfig
    risk: LiveRiskConfig


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {', '.join(unknown)}")
    if cls is LiveTradingConfig and "symbols" in values:
        values = dict(values)
        values["symbols"] = tuple(_normalize_symbols(values["symbols"]))
    if cls is MultiTimeframeFilterConfig and "timeframes" in values:
        values = dict(values)
        values["timeframes"] = tuple(_normalize_timeframes(values["timeframes"]))
    return cls(**values)


def _normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("\n", ",").split(",")
    else:
        parts = list(value)
    symbols: list[str] = []
    for part in parts:
        symbol = str(part).strip().upper()
        if not symbol:
            continue
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _normalize_timeframes(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("，", ",").replace("\n", ",").split(",")
    else:
        parts = list(value)
    timeframes: list[str] = []
    for part in parts:
        timeframe = str(part).strip()
        if timeframe and timeframe not in timeframes:
            timeframes.append(timeframe)
    return timeframes


def load_live_config(path: str | Path) -> LiveAppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return LiveAppConfig(
        exchange=_coerce_dataclass(ExchangeConfig, raw.get("exchange", {})),
        trading=_coerce_dataclass(LiveTradingConfig, raw.get("trading", {})),
        strategy=_coerce_dataclass(StrategyConfig, raw.get("strategy", {})),
        filters=_coerce_dataclass(MultiTimeframeFilterConfig, raw.get("filters", {})),
        risk=_coerce_dataclass(LiveRiskConfig, raw.get("risk", {})),
    )


def write_live_config(path: str | Path, config: LiveAppConfig) -> None:
    payload = {
        "exchange": asdict(config.exchange),
        "trading": asdict(config.trading),
        "strategy": asdict(config.strategy),
        "filters": asdict(config.filters),
        "risk": asdict(config.risk),
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_live_config() -> LiveAppConfig:
    return LiveAppConfig(
        exchange=ExchangeConfig(),
        trading=LiveTradingConfig(),
        strategy=StrategyConfig(
            fast_ema=9,
            slow_ema=21,
            atr_period=14,
            channel_period=32,
            min_atr_pct=0.0012,
            max_atr_pct=0.010,
            breakout_buffer_atr=0.25,
            ema_gap_atr=0.22,
            volume_period=20,
            min_volume_ratio=1.00,
            bollinger_period=20,
            bollinger_stddev=2.2,
            stop_loss_atr=2.0,
            take_profit_atr=6.0,
            mean_reversion_stop_atr=1.2,
            breakeven_atr=0.0,
            trailing_activation_atr=1.5,
            trailing_stop_atr=0.0,
            max_holding_bars=48,
            spike_guard_enabled=True,
            spike_min_range_atr=3.0,
            spike_min_wick_atr=1.4,
            spike_min_wick_ratio=0.55,
            spike_min_volume_ratio=1.2,
            spike_block_bars=5,
            spike_trade_enabled=False,
            spike_recovery_ratio=0.45,
            spike_stop_atr=0.7,
            spike_take_profit_atr=0.9,
            spike_risk_multiplier=0.35,
            spike_max_holding_bars=6,
            allow_short=True,
        ),
        filters=MultiTimeframeFilterConfig(),
        risk=LiveRiskConfig(),
    )
