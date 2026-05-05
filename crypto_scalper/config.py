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
    bollinger_period: int = 20
    bollinger_stddev: float = 2.2
    stop_loss_atr: float = 1.2
    take_profit_atr: float = 1.8
    mean_reversion_stop_atr: float = 1.5
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
    allow_short: bool = True


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
