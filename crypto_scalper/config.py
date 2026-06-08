from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    path: str = "data/sample_btcusdt_1m.csv"


@dataclass(frozen=True)
class StrategyConfig:
    fast_ema: int = 9
    slow_ema: int = 21
    atr_period: int = 14
    channel_period: int = 20
    min_atr_pct: float = 0.00035
    max_atr_pct: float = 0.01
    breakout_buffer_atr: float = 0.0
    ema_gap_atr: float = 0.0
    volume_period: int = 20
    min_volume_ratio: float = 0.0
    stop_loss_atr: float = 1.2
    take_profit_atr: float = 1.8
    breakeven_atr: float = 0.0
    trailing_activation_atr: float = 1.0
    trailing_stop_atr: float = 0.0
    max_holding_bars: int = 0
    spike_guard_enabled: bool = True
    spike_min_range_atr: float = 3.0
    spike_min_wick_atr: float = 1.4
    spike_min_wick_ratio: float = 0.55
    spike_min_volume_ratio: float = 1.2
    spike_block_bars: int = 3
    spike_trade_enabled: bool = True
    spike_recovery_ratio: float = 0.45
    spike_stop_atr: float = 0.7
    spike_take_profit_atr: float = 0.9
    spike_risk_multiplier: float = 0.35
    spike_max_holding_bars: int = 6
    rsi_reversal_enabled: bool = True
    allow_short: bool = True
    long_score_threshold: float = 0.25
    short_score_threshold: float = 0.25
    long_risk_bias: float = 1.0
    short_risk_bias: float = 1.0
    regime_filter_enabled: bool = False
    regime_lookback: int = 24
    long_min_slow_slope_atr: float = -1.0
    short_max_slow_slope_atr: float = 1.0
    super_volume_breakout_enabled: bool = True
    super_volume_min_ratio: float = 3.0
    super_volume_min_breakout_atr: float = 0.75
    super_volume_min_body_atr: float = 0.35
    super_volume_confidence_boost: float = 0.15
    super_volume_risk_multiplier: float = 1.35
    super_volume_take_profit_multiplier: float = 1.35
    super_volume_live_chase_guard_enabled: bool = False
    super_volume_max_entry_chase_pct: float = 0.006
    breakout_rsi_guard_enabled: bool = False
    breakout_rsi_fast_period: int = 6
    breakout_rsi_mid_period: int = 12
    breakout_long_rsi_fast_ceiling: float = 85.0
    breakout_long_rsi_mid_ceiling: float = 74.0
    breakout_short_rsi_fast_floor: float = 15.0
    breakout_short_rsi_mid_floor: float = 26.0
    startup_breakout_enabled: bool = False
    startup_min_volume_ratio: float = 2.0
    startup_min_breakout_atr: float = 0.0
    startup_min_body_atr: float = 0.65
    startup_risk_multiplier: float = 0.85
    startup_take_profit_multiplier: float = 1.2
    trend_risk_control_enabled: bool = False
    trend_risk_min_holding_bars: int = 18
    trend_long_risk_multiplier: float = 1.0
    trend_short_risk_multiplier: float = 1.0
    trend_super_volume_min_risk_multiplier: float = 0.85
    trend_loss_guard_enabled: bool = False
    trend_loss_guard_bars: int = 3
    trend_loss_guard_loss_pct: float = 0.006
    trend_loss_guard_super_volume_bars: int = 5
    trend_loss_guard_super_volume_loss_pct: float = 0.009
    indicator_trend_guard_enabled: bool = False
    indicator_trend_guard_lookback_bars: int = 12
    indicator_trend_guard_buffer_atr: float = 0.10
    indicator_trend_guard_slope_atr: float = 0.0
    indicator_max_holding_bars: int = 8
    indicator_confirmed_cross_extreme_guard_enabled: bool = False
    indicator_confirmed_extreme_lookback_bars: int = 4
    indicator_confirmed_long_max_rsi: float = 42.0
    indicator_confirmed_long_max_kdj: float = 35.0
    indicator_confirmed_short_min_rsi: float = 58.0
    indicator_confirmed_short_min_kdj: float = 65.0
    indicator_confirmed_trend_fallback_enabled: bool = True
    indicator_confirmed_trend_buffer_atr: float = 0.15
    indicator_confirmed_trend_slope_atr: float = 0.05
    indicator_confirmed_cross_extreme_required_enabled: bool = False
    indicator_confirmed_cross_long_max_rsi: float = 45.0
    indicator_confirmed_cross_long_max_kdj: float = 35.0
    indicator_confirmed_cross_short_min_rsi: float = 60.0
    indicator_confirmed_cross_short_min_kdj: float = 70.0
    indicator_reference_guard_enabled: bool = False
    indicator_reference_guard_long_enabled: bool = True
    indicator_reference_guard_short_enabled: bool = True
    indicator_reference_timeframe: str = "1h"
    indicator_reference_lookback_bars: int = 12
    indicator_reference_buffer_atr: float = 0.10
    indicator_reference_slope_atr: float = 0.05
    indicator_reference_extreme_override_enabled: bool = True
    indicator_reference_long_extreme_rsi: float = 38.0
    indicator_reference_long_extreme_kdj: float = 30.0
    indicator_reference_short_extreme_rsi: float = 62.0
    indicator_reference_short_extreme_kdj: float = 70.0
    indicator_reversal_size_multiplier: float = 1.0
    indicator_reversal_loss_pause_enabled: bool = False
    indicator_reversal_loss_pause_losses: int = 2
    indicator_reversal_loss_pause_bars: int = 8
    btc_market_filter_enabled: bool = False
    btc_market_symbol: str = "BTCUSDT"
    btc_market_timeframe: str = "30m"
    btc_market_lookback_bars: int = 12
    btc_market_filter_trend_only: bool = True
    btc_market_filter_min_holding_bars: int = 18
    btc_bull_return_threshold: float = 0.006
    btc_bear_return_threshold: float = -0.006
    btc_market_slope_atr_threshold: float = 0.20
    btc_counter_trend_super_volume_only: bool = True
    btc_counter_trend_min_rank_score: float = 6.2
    btc_counter_trend_min_momentum_pct: float = 0.025
    btc_counter_trend_min_volume_ratio: float = 2.2
    btc_counter_trend_risk_multiplier: float = 0.45
    weak_market_long_filter_enabled: bool = False
    weak_market_lookback_bars: int = 48
    weak_market_breadth_threshold: float = 0.42
    weak_market_avg_return_threshold: float = -0.012
    weak_market_long_super_volume_only: bool = True
    weak_market_long_min_rank_score: float = 4.8
    weak_market_long_risk_multiplier: float = 0.55


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 10_000.0
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    max_leverage: float = 2.0
    risk_per_trade_pct: float = 0.005
    max_position_notional_pct: float = 1.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    maintenance_margin_pct: float = 0.005
    min_order_notional: float = 10.0
    cooldown_bars_after_loss: int = 5
    initial_entry_fraction: float = 0.75
    scale_in_enabled: bool = True
    scale_in_fraction: float = 0.25
    scale_in_profit_trigger_pct: float = 0.004
    scale_in_min_score: float = 0.55
    max_scale_ins: int = 1
    loss_scale_in_enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig


T = TypeVar("T")


def _coerce_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {', '.join(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return AppConfig(
        data=_coerce_dataclass(DataConfig, raw.get("data", {})),
        strategy=_coerce_dataclass(StrategyConfig, raw.get("strategy", {})),
        risk=_coerce_dataclass(RiskConfig, raw.get("risk", {})),
    )
