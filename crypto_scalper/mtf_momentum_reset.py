from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .indicators import atr, ema, macd, rsi
from .models import Candle, Direction, Signal
from .mtf_4h_rsi_regime import MTF_LONG_REASON, MTF_SHORT_REASON


MTF_MOMENTUM_RESET_VERSION = "mtf_1h_reset_30m_release_v1"
MTF_MOMENTUM_RESET_SETUP_TOKEN = "setup=1h_reset_30m_release"


@dataclass(frozen=True)
class MtfMomentumResetConfig:
    reset_lookback_1h: int = 8
    ema_period_1h: int = 21
    ema_slope_lookback_1h: int = 3
    reset_rsi_ceiling: float = 62.0
    min_rsi_recovery: float = 0.5
    max_adverse_ema_distance_atr: float = 0.25
    max_favorable_ema_distance_atr: float = 4.0
    min_directional_ema_slope_atr: float = -0.10
    ema_touch_distance_atr: float = 0.35
    contraction_lookback_30m: int = 6
    baseline_lookback_30m: int = 20
    max_contraction_range_atr: float = 5.0
    max_contraction_tr_ratio: float = 1.20
    max_contraction_volume_ratio: float = 1.40
    min_release_body_atr: float = 0.05
    min_release_close_position: float = 0.52
    min_release_volume_ratio: float = 0.40
    max_release_volume_ratio: float = 4.0
    min_breakout_distance_atr: float = 0.0
    max_breakout_distance_atr: float = 1.25
    max_release_extension_atr: float = 2.5
    stop_atr_buffer: float = 0.25


@dataclass(frozen=True)
class MtfMomentumResetRelease:
    direction: Direction
    candle: Candle
    structural_stop: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MtfSetupExperiment:
    name: str
    side_policy: str = "BOTH"
    min_rank_score: float = -999.0
    min_target_to_cost_ratio: float = 0.0
    min_directional_btc_4h_return: float = -999.0
    minimum_reset_components: int = 1
    require_macd_reset: bool = False
    require_rsi_reset: bool = False
    require_ema_reset: bool = False
    min_rsi_recovery: float = 0.5
    min_directional_ema_slope_atr: float = -0.10
    max_directional_ema_distance_atr: float = 4.0
    max_contraction_range_atr: float = 5.0
    max_contraction_tr_ratio: float = 1.20
    max_contraction_volume_ratio: float = 1.40
    min_release_body_atr: float = 0.05
    min_release_close_position: float = 0.52
    min_release_volume_ratio: float = 0.40
    max_release_volume_ratio: float = 4.0
    max_breakout_distance_atr: float = 1.25
    max_release_extension_atr: float = 2.5

    def accepts(self, row: dict[str, Any]) -> bool:
        side = str(row.get("side", "")).upper()
        if self.side_policy != "BOTH" and side != self.side_policy:
            return False
        side_value = 1.0 if side == "LONG" else -1.0
        if float(row.get("rank_score", -999.0)) < self.min_rank_score:
            return False
        if float(row.get("target_to_cost_ratio", 0.0)) < self.min_target_to_cost_ratio:
            return False
        if side_value * float(row.get("btc_4h_return", 0.0)) < self.min_directional_btc_4h_return:
            return False
        if int(row.get("reset_component_count", 0)) < self.minimum_reset_components:
            return False
        if self.require_macd_reset and not bool(row.get("macd_reset_present")):
            return False
        if self.require_rsi_reset and not bool(row.get("rsi_reset_present")):
            return False
        if self.require_ema_reset and not bool(row.get("ema_reset_present")):
            return False
        checks = (
            float(row.get("rsi_recovery", 0.0)) >= self.min_rsi_recovery,
            float(row.get("directional_ema_slope_atr", -999.0)) >= self.min_directional_ema_slope_atr,
            float(row.get("directional_ema_distance_atr", 999.0)) <= self.max_directional_ema_distance_atr,
            float(row.get("contraction_range_atr", 999.0)) <= self.max_contraction_range_atr,
            float(row.get("contraction_tr_ratio", 999.0)) <= self.max_contraction_tr_ratio,
            float(row.get("contraction_volume_ratio", 999.0)) <= self.max_contraction_volume_ratio,
            float(row.get("release_body_atr", 0.0)) >= self.min_release_body_atr,
            float(row.get("release_directional_close_position", 0.0)) >= self.min_release_close_position,
            float(row.get("release_volume_ratio", 0.0)) >= self.min_release_volume_ratio,
            float(row.get("release_volume_ratio", 999.0)) <= self.max_release_volume_ratio,
            float(row.get("breakout_distance_atr", 999.0)) <= self.max_breakout_distance_atr,
            float(row.get("release_extension_atr", 999.0)) <= self.max_release_extension_atr,
        )
        return all(checks)


def momentum_reset_config_from_strategy(strategy: Any) -> MtfMomentumResetConfig:
    prefix = "mtf_momentum_reset_"
    return MtfMomentumResetConfig(
        reset_lookback_1h=max(4, int(getattr(strategy, f"{prefix}reset_lookback_1h", 8))),
        ema_period_1h=max(2, int(getattr(strategy, f"{prefix}ema_period_1h", 21))),
        ema_slope_lookback_1h=max(1, int(getattr(strategy, f"{prefix}ema_slope_lookback_1h", 3))),
        reset_rsi_ceiling=float(getattr(strategy, f"{prefix}rsi_ceiling", 62.0)),
        min_rsi_recovery=float(getattr(strategy, f"{prefix}min_rsi_recovery", 0.5)),
        max_adverse_ema_distance_atr=float(
            getattr(strategy, f"{prefix}max_adverse_ema_distance_atr", 0.25)
        ),
        max_favorable_ema_distance_atr=float(
            getattr(strategy, f"{prefix}max_favorable_ema_distance_atr", 4.0)
        ),
        min_directional_ema_slope_atr=float(
            getattr(strategy, f"{prefix}min_directional_ema_slope_atr", -0.10)
        ),
        ema_touch_distance_atr=float(getattr(strategy, f"{prefix}ema_touch_distance_atr", 0.35)),
        contraction_lookback_30m=max(
            3,
            int(getattr(strategy, f"{prefix}contraction_lookback_30m", 6)),
        ),
        baseline_lookback_30m=max(
            5,
            int(getattr(strategy, f"{prefix}baseline_lookback_30m", 20)),
        ),
        max_contraction_range_atr=float(
            getattr(strategy, f"{prefix}max_contraction_range_atr", 5.0)
        ),
        max_contraction_tr_ratio=float(
            getattr(strategy, f"{prefix}max_contraction_tr_ratio", 1.20)
        ),
        max_contraction_volume_ratio=float(
            getattr(strategy, f"{prefix}max_contraction_volume_ratio", 1.40)
        ),
        min_release_body_atr=float(getattr(strategy, f"{prefix}min_release_body_atr", 0.05)),
        min_release_close_position=float(
            getattr(strategy, f"{prefix}min_release_close_position", 0.52)
        ),
        min_release_volume_ratio=float(
            getattr(strategy, f"{prefix}min_release_volume_ratio", 0.40)
        ),
        max_release_volume_ratio=float(
            getattr(strategy, f"{prefix}max_release_volume_ratio", 4.0)
        ),
        min_breakout_distance_atr=float(
            getattr(strategy, f"{prefix}min_breakout_distance_atr", 0.0)
        ),
        max_breakout_distance_atr=float(
            getattr(strategy, f"{prefix}max_breakout_distance_atr", 1.25)
        ),
        max_release_extension_atr=float(
            getattr(strategy, f"{prefix}max_release_extension_atr", 2.5)
        ),
        stop_atr_buffer=float(getattr(strategy, f"{prefix}stop_atr_buffer", 0.25)),
    )


STAGE16_EXPERIMENTS: tuple[MtfSetupExperiment, ...] = (
    MtfSetupExperiment(name="s16_a_broad_reset_release"),
    MtfSetupExperiment(
        name="s16_b_simple_balanced",
        minimum_reset_components=2,
        min_rsi_recovery=2.0,
        min_directional_ema_slope_atr=0.0,
        max_directional_ema_distance_atr=2.00,
        max_contraction_range_atr=3.50,
        max_contraction_tr_ratio=0.95,
        max_contraction_volume_ratio=1.10,
        min_release_body_atr=0.15,
        min_release_close_position=0.60,
        min_release_volume_ratio=0.70,
        max_release_volume_ratio=2.50,
        max_breakout_distance_atr=0.80,
        max_release_extension_atr=1.75,
    ),
    MtfSetupExperiment(
        name="s16_c_reset_quality",
        minimum_reset_components=2,
        require_macd_reset=True,
        require_rsi_reset=True,
        min_rsi_recovery=3.0,
        min_directional_ema_slope_atr=0.0,
        max_directional_ema_distance_atr=1.75,
        max_contraction_range_atr=4.00,
        max_contraction_tr_ratio=1.00,
        max_contraction_volume_ratio=1.20,
        min_release_body_atr=0.10,
        min_release_close_position=0.58,
        min_release_volume_ratio=0.60,
        max_release_volume_ratio=3.00,
        max_breakout_distance_atr=1.00,
        max_release_extension_atr=2.00,
    ),
    MtfSetupExperiment(
        name="s16_d_release_quality",
        minimum_reset_components=1,
        require_macd_reset=True,
        min_rsi_recovery=1.0,
        min_directional_ema_slope_atr=-0.05,
        max_directional_ema_distance_atr=1.50,
        max_contraction_range_atr=2.75,
        max_contraction_tr_ratio=0.80,
        max_contraction_volume_ratio=0.95,
        min_release_body_atr=0.25,
        min_release_close_position=0.65,
        min_release_volume_ratio=0.90,
        max_release_volume_ratio=2.20,
        max_breakout_distance_atr=0.65,
        max_release_extension_atr=1.50,
    ),
    MtfSetupExperiment(
        name="s16_e_macd_ema_reset",
        minimum_reset_components=2,
        require_macd_reset=True,
        require_ema_reset=True,
        min_rsi_recovery=1.0,
        min_directional_ema_slope_atr=0.0,
        max_directional_ema_distance_atr=2.00,
        max_contraction_range_atr=3.50,
        max_contraction_tr_ratio=0.95,
        max_contraction_volume_ratio=1.10,
        min_release_body_atr=0.15,
        min_release_close_position=0.60,
        min_release_volume_ratio=0.70,
        max_release_volume_ratio=2.50,
        max_breakout_distance_atr=0.80,
        max_release_extension_atr=1.75,
    ),
    MtfSetupExperiment(
        name="s16_f_macd_rsi_reset",
        minimum_reset_components=2,
        require_macd_reset=True,
        require_rsi_reset=True,
        min_rsi_recovery=2.0,
        min_directional_ema_slope_atr=0.0,
        max_directional_ema_distance_atr=2.00,
        max_contraction_range_atr=3.50,
        max_contraction_tr_ratio=0.95,
        max_contraction_volume_ratio=1.10,
        min_release_body_atr=0.15,
        min_release_close_position=0.60,
        min_release_volume_ratio=0.70,
        max_release_volume_ratio=2.50,
        max_breakout_distance_atr=0.80,
        max_release_extension_atr=1.75,
    ),
)


STAGE18_EXPERIMENTS: tuple[MtfSetupExperiment, ...] = (
    MtfSetupExperiment(
        name="s18_a_rank_cost_both",
        min_rank_score=4.25,
        min_target_to_cost_ratio=12.0,
    ),
    MtfSetupExperiment(
        name="s18_b_long_rank_cost",
        side_policy="LONG",
        min_rank_score=4.25,
        min_target_to_cost_ratio=12.0,
    ),
    MtfSetupExperiment(
        name="s18_c_long_rank_cost_btc4h_guard",
        side_policy="LONG",
        min_rank_score=4.25,
        min_target_to_cost_ratio=12.0,
        min_directional_btc_4h_return=-0.001,
    ),
)


def mtf_momentum_reset_event_id(symbol: str, direction: Direction, release_time: datetime) -> str:
    raw = "|".join(
        (
            MTF_MOMENTUM_RESET_VERSION,
            symbol.upper(),
            direction.name,
            release_time.isoformat(),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def can_be_contraction_release(candles: list[Candle], index: int, lookback: int = 6) -> bool:
    lookback = max(3, int(lookback))
    if index < lookback:
        return False
    latest = candles[index]
    prior = candles[index - lookback:index]
    return latest.close > max(item.high for item in prior) or latest.close < min(item.low for item in prior)


def detect_momentum_reset_release(
    direction: Direction,
    candles_1h: list[Candle],
    candles_30m: list[Candle],
    config: MtfMomentumResetConfig | None = None,
) -> MtfMomentumResetRelease | None:
    cfg = config or MtfMomentumResetConfig()
    reset = momentum_reset_features(direction, candles_1h, cfg)
    release = contraction_release_features(direction, candles_30m, cfg)
    if reset is None or release is None:
        return None
    if int(reset["reset_component_count"]) < 1:
        return None
    latest = candles_30m[-1]
    stop = float(release["contraction_low"]) - float(release["atr_30m"]) * cfg.stop_atr_buffer
    if direction is Direction.SHORT:
        stop = float(release["contraction_high"]) + float(release["atr_30m"]) * cfg.stop_atr_buffer
    return MtfMomentumResetRelease(
        direction=direction,
        candle=latest,
        structural_stop=stop,
        metadata={**reset, **release},
    )


def momentum_reset_features(
    direction: Direction,
    candles: list[Candle],
    config: MtfMomentumResetConfig | None = None,
) -> dict[str, Any] | None:
    cfg = config or MtfMomentumResetConfig()
    lookback = max(4, int(cfg.reset_lookback_1h))
    ema_period = max(2, int(cfg.ema_period_1h))
    slope_lookback = max(1, int(cfg.ema_slope_lookback_1h))
    minimum = max(ema_period + lookback + 3, 40)
    if len(candles) < minimum or direction is Direction.FLAT:
        return None

    side = direction.value
    closes = [item.close for item in candles]
    ema_values = ema(closes, ema_period)
    rsi_values = rsi(closes, 14)
    _, _, histogram = macd(closes)
    atr_value = max(atr(candles, 14)[-1], candles[-1].close * 0.0005, 1e-12)
    directional_rsi = [value if side > 0 else 100.0 - value for value in rsi_values]
    directional_hist = [side * value / atr_value for value in histogram]
    reset_rsi = min(directional_rsi[-lookback - 1:-1])
    rsi_recovery = directional_rsi[-1] - reset_rsi
    histogram_pullback = any(
        directional_hist[index] < directional_hist[index - 1]
        for index in range(len(directional_hist) - lookback, len(directional_hist) - 1)
    )
    histogram_turn = directional_hist[-1] > directional_hist[-2]
    macd_reset_present = histogram_pullback and histogram_turn

    directional_distances = [
        side * (candles[index].close - ema_values[index]) / atr_value
        for index in range(len(candles) - lookback - 1, len(candles))
    ]
    ema_touch_distance = min(abs(value) for value in directional_distances[:-1])
    current_ema_distance = directional_distances[-1]
    ema_reset_present = (
        ema_touch_distance <= cfg.ema_touch_distance_atr
        and current_ema_distance >= 0.0
    )
    rsi_reset_present = reset_rsi <= cfg.reset_rsi_ceiling and rsi_recovery >= cfg.min_rsi_recovery
    reset_components = sum((macd_reset_present, rsi_reset_present, ema_reset_present))
    directional_ema_slope = side * (ema_values[-1] - ema_values[-1 - slope_lookback]) / atr_value
    if current_ema_distance < -cfg.max_adverse_ema_distance_atr:
        return None
    if current_ema_distance > cfg.max_favorable_ema_distance_atr:
        return None
    if directional_ema_slope < cfg.min_directional_ema_slope_atr:
        return None

    return {
        "reset_lookback_1h": lookback,
        "directional_rsi_current": directional_rsi[-1],
        "directional_rsi_reset_low": reset_rsi,
        "rsi_recovery": rsi_recovery,
        "macd_reset_present": macd_reset_present,
        "rsi_reset_present": rsi_reset_present,
        "ema_reset_present": ema_reset_present,
        "reset_component_count": reset_components,
        "directional_macd_hist_atr": directional_hist[-1],
        "directional_macd_hist_change_atr": directional_hist[-1] - directional_hist[-2],
        "directional_ema_distance_atr": current_ema_distance,
        "directional_ema_slope_atr": directional_ema_slope,
        "ema_reset_touch_distance_atr": ema_touch_distance,
        "1h_rsi": rsi_values[-1],
        "atr_1h": atr_value,
    }


def contraction_release_features(
    direction: Direction,
    candles: list[Candle],
    config: MtfMomentumResetConfig | None = None,
) -> dict[str, Any] | None:
    cfg = config or MtfMomentumResetConfig()
    contraction_bars = max(3, int(cfg.contraction_lookback_30m))
    baseline_bars = max(contraction_bars + 2, int(cfg.baseline_lookback_30m))
    minimum = max(40, baseline_bars + contraction_bars + 2)
    if len(candles) < minimum or direction is Direction.FLAT:
        return None

    latest = candles[-1]
    prior = candles[-1 - contraction_bars:-1]
    baseline = candles[-1 - contraction_bars - baseline_bars:-1 - contraction_bars]
    side = direction.value
    contraction_low = min(item.low for item in prior)
    contraction_high = max(item.high for item in prior)
    boundary = contraction_high if side > 0 else contraction_low
    breakout_distance = side * (latest.close - boundary)
    if breakout_distance <= 0:
        return None

    atr_value = max(atr(candles, 14)[-1], latest.close * 0.0005, 1e-12)
    contraction_tr = _average_true_range(prior)
    baseline_tr = max(_average_true_range(baseline), 1e-12)
    contraction_volume = _average_volume(prior)
    baseline_volume = max(_average_volume(baseline), 1e-12)
    release_range = max(latest.high - latest.low, 1e-12)
    close_position = (latest.close - latest.low) / release_range
    directional_close = close_position if side > 0 else 1.0 - close_position
    body_atr = abs(latest.close - latest.open) / atr_value
    release_volume_ratio = max(0.0, latest.volume) / baseline_volume
    contraction_range_atr = (contraction_high - contraction_low) / atr_value
    contraction_tr_ratio = contraction_tr / baseline_tr
    contraction_volume_ratio = contraction_volume / baseline_volume
    breakout_distance_atr = breakout_distance / atr_value
    ema21 = ema([item.close for item in candles], 21)[-1]
    release_extension_atr = max(0.0, side * (latest.close - ema21) / atr_value)

    broad_checks = (
        contraction_range_atr <= cfg.max_contraction_range_atr,
        contraction_tr_ratio <= cfg.max_contraction_tr_ratio,
        contraction_volume_ratio <= cfg.max_contraction_volume_ratio,
        body_atr >= cfg.min_release_body_atr,
        directional_close >= cfg.min_release_close_position,
        release_volume_ratio >= cfg.min_release_volume_ratio,
        release_volume_ratio <= cfg.max_release_volume_ratio,
        breakout_distance_atr >= cfg.min_breakout_distance_atr,
        breakout_distance_atr <= cfg.max_breakout_distance_atr,
        release_extension_atr <= cfg.max_release_extension_atr,
    )
    if not all(broad_checks):
        return None

    return {
        "contraction_lookback_30m": contraction_bars,
        "contraction_low": contraction_low,
        "contraction_high": contraction_high,
        "contraction_range_atr": contraction_range_atr,
        "contraction_tr_ratio": contraction_tr_ratio,
        "contraction_volume_ratio": contraction_volume_ratio,
        "release_body_atr": body_atr,
        "release_directional_close_position": directional_close,
        "release_volume_ratio": release_volume_ratio,
        "breakout_distance_atr": breakout_distance_atr,
        "release_extension_atr": release_extension_atr,
        "release_ema21": ema21,
        "atr_30m": atr_value,
    }


def build_release_signal(config: Any, event: MtfMomentumResetRelease) -> tuple[Signal, dict[str, Any]] | None:
    strategy = config.strategy
    entry_reference = event.candle.close
    side = event.direction.value
    stop_distance = side * (entry_reference - event.structural_stop)
    if stop_distance <= 0:
        return None
    stop_pct = stop_distance / max(entry_reference, 1e-12)
    minimum_stop = max(0.0, float(getattr(strategy, "mtf_min_stop_pct", 0.0)))
    maximum_stop = max(0.0, float(getattr(strategy, "mtf_max_stop_pct", 0.02)))
    if minimum_stop > 0.0 and stop_pct < minimum_stop:
        return None
    if maximum_stop > 0.0 and stop_pct > maximum_stop:
        return None
    take_profit_r = max(0.1, float(getattr(strategy, "mtf_take_profit_r", 2.0)))
    max_holding_minutes = max(1, int(getattr(strategy, "mtf_max_holding_minutes", 720)))
    reason_prefix = MTF_LONG_REASON if event.direction is Direction.LONG else MTF_SHORT_REASON
    signal = Signal(
        event.direction,
        0.62,
        f"{reason_prefix}|{MTF_MOMENTUM_RESET_SETUP_TOKEN}",
        stop_pct,
        stop_pct * take_profit_r,
        risk_multiplier=max(0.05, float(getattr(strategy, "mtf_risk_multiplier", 1.0))),
        max_holding_bars=max(1, (max_holding_minutes + 29) // 30),
    )
    metadata = {
        **event.metadata,
        "setup_feature_version": MTF_MOMENTUM_RESET_VERSION,
        "setup_mode": "1h_momentum_reset_30m_contraction_release",
        "trigger_mode": "30m_contraction_release",
        "trigger_atr": event.metadata["atr_30m"],
        "trigger_close": event.candle.close,
        "trigger_volume_ratio": event.metadata["release_volume_ratio"],
        "trigger_body_atr": event.metadata["release_body_atr"],
        "trigger_close_position": (
            event.metadata["release_directional_close_position"]
            if event.direction is Direction.LONG
            else 1.0 - event.metadata["release_directional_close_position"]
        ),
        "structural_stop_price": event.structural_stop,
        "stop_loss_pct": stop_pct,
        "take_profit_pct": stop_pct * take_profit_r,
        "take_profit_r": take_profit_r,
    }
    return signal, metadata


def _average_true_range(candles: list[Candle]) -> float:
    if not candles:
        return 0.0
    return sum(max(item.high - item.low, abs(item.high - item.open), abs(item.low - item.open)) for item in candles) / len(candles)


def _average_volume(candles: list[Candle]) -> float:
    return sum(max(0.0, item.volume) for item in candles) / max(len(candles), 1)
