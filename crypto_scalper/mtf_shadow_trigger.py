from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .indicators import atr, macd
from .models import Candle, Direction
from .mtf_4h_rsi_regime import MtfSetupResult, MtfTriggerResult


MTF_SHADOW_TRIGGER_VERSION = "mtf_native_shadow_trigger_v1"


@dataclass(frozen=True)
class MtfNativeShadowTriggerConfig:
    local_lookback: int = 4
    frozen_structure_lookback: int = 3
    min_confirm_body_atr: float = 0.15
    max_confirm_body_atr: float = 1.80
    min_directional_close_position: float = 0.60
    min_false_break_wick_ratio: float = 0.20
    min_volume_ratio: float = 0.0
    max_volume_ratio: float = 2.0
    max_distance_from_setup_ema_pct: float = 0.010
    require_macd_improvement: bool = True
    allow_two_bar_false_break: bool = True
    allow_pivot_break: bool = True


def mtf_shadow_trigger_event_id(
    symbol: str,
    direction: Direction | str,
    mode: str,
    trigger_time: datetime,
) -> str:
    side = direction.name if isinstance(direction, Direction) else str(direction).upper()
    raw = "|".join(
        (
            MTF_SHADOW_TRIGGER_VERSION,
            symbol.upper(),
            side,
            str(mode),
            trigger_time.isoformat(),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def can_be_native_shadow_trigger(
    candles: list[Candle],
    index: int,
    local_lookback: int = 4,
) -> bool:
    """Cheap necessary condition using only candles closed by this index."""
    lookback = max(2, int(local_lookback))
    if index < lookback + 2:
        return False
    latest = candles[index]
    probe = candles[index - 1]
    prior = candles[index - 1 - lookback:index - 1]
    long_confirmation = latest.close > latest.open and latest.close > probe.high
    short_confirmation = latest.close < latest.open and latest.close < probe.low
    if not long_confirmation and not short_confirmation:
        return False
    false_break_long = probe.low < min(item.low for item in prior)
    false_break_short = probe.high > max(item.high for item in prior)
    pivot_long = probe.low < candles[index - 2].low and latest.low > probe.low
    pivot_short = probe.high > candles[index - 2].high and latest.high < probe.high
    return (long_confirmation and (false_break_long or pivot_long)) or (
        short_confirmation and (false_break_short or pivot_short)
    )


class MtfNativeShadowTriggerDetector:
    def __init__(self, config: MtfNativeShadowTriggerConfig | None = None) -> None:
        self.config = config or MtfNativeShadowTriggerConfig()

    def __call__(
        self,
        direction: Direction,
        candles: list[Candle],
        setup: MtfSetupResult,
        timeframe: str,
    ) -> tuple[MtfTriggerResult | None, str]:
        cfg = self.config
        lookback = max(2, int(cfg.local_lookback))
        minimum = max(24, lookback + 3)
        if len(candles) < minimum:
            return None, f"{timeframe}_shadow_warmup"

        latest = candles[-1]
        probe = candles[-2]
        prior = candles[-2 - lookback:-2]
        atr_value = max(atr(candles, 14)[-1], latest.close * 0.0008, 1e-12)
        body_atr = abs(latest.close - latest.open) / atr_value
        if body_atr < max(0.0, cfg.min_confirm_body_atr):
            return None, f"{timeframe}_shadow_body_too_small"
        if body_atr > max(cfg.min_confirm_body_atr, cfg.max_confirm_body_atr):
            return None, f"{timeframe}_shadow_body_too_large"

        volume_ratio = _latest_volume_ratio(candles, 20)
        if volume_ratio < max(0.0, cfg.min_volume_ratio):
            return None, f"{timeframe}_shadow_volume_too_low"
        if cfg.max_volume_ratio > 0 and volume_ratio > cfg.max_volume_ratio:
            return None, f"{timeframe}_shadow_volume_too_high"

        ema_distance = abs(latest.close - setup.ema21) / max(latest.close, 1e-12)
        if ema_distance > max(0.0, cfg.max_distance_from_setup_ema_pct):
            return None, f"{timeframe}_shadow_far_from_30m_ema"

        _, _, histogram = macd([item.close for item in candles])
        histogram_improving = (
            histogram[-1] > histogram[-2]
            if direction is Direction.LONG
            else histogram[-1] < histogram[-2]
        )
        if cfg.require_macd_improvement and not histogram_improving:
            return None, f"{timeframe}_shadow_macd_not_improving"

        candle_range = max(latest.high - latest.low, 1e-12)
        close_position = (latest.close - latest.low) / candle_range
        directional_close = close_position if direction is Direction.LONG else 1.0 - close_position
        if directional_close < max(0.0, cfg.min_directional_close_position):
            return None, f"{timeframe}_shadow_close_position_weak"

        local_low = min(item.low for item in prior)
        local_high = max(item.high for item in prior)
        probe_range = max(probe.high - probe.low, 1e-12)
        lower_wick_ratio = (min(probe.open, probe.close) - probe.low) / probe_range
        upper_wick_ratio = (probe.high - max(probe.open, probe.close)) / probe_range
        probe_wick_ratio = lower_wick_ratio if direction is Direction.LONG else upper_wick_ratio

        false_break = self._false_break(
            direction,
            latest,
            probe,
            local_low,
            local_high,
            probe_wick_ratio,
        )
        pivot_break = self._pivot_break(direction, latest, probe, candles[-3], setup)
        if cfg.allow_two_bar_false_break and false_break:
            mode = "two_bar_false_break"
            local_reference = local_low if direction is Direction.LONG else local_high
        elif cfg.allow_pivot_break and pivot_break:
            mode = "higher_low_break" if direction is Direction.LONG else "lower_high_break"
            local_reference = setup.support if direction is Direction.LONG else setup.resistance
        else:
            return None, f"{timeframe}_shadow_no_trigger"

        structure_lookback = max(2, int(cfg.frozen_structure_lookback))
        structure_previous = candles[-1 - structure_lookback:-1]
        same_bar_structure = _is_structure_break(direction, latest, structure_previous)
        metadata = {
            "shadow_trigger_version": MTF_SHADOW_TRIGGER_VERSION,
            "shadow_trigger_mode": mode,
            "shadow_local_reference": local_reference,
            "shadow_probe_wick_ratio": probe_wick_ratio,
            "shadow_probe_range_atr": probe_range / atr_value,
            "shadow_confirm_body_atr": body_atr,
            "shadow_directional_close_position": directional_close,
            "shadow_same_bar_structure_break": same_bar_structure,
            "shadow_setup_ema_distance_pct": ema_distance,
            "shadow_probe_distance_from_setup_atr": (
                (probe.low - setup.support) / atr_value
                if direction is Direction.LONG
                else (setup.resistance - probe.high) / atr_value
            ),
        }
        return (
            MtfTriggerResult(
                direction,
                mode,
                latest,
                atr_value,
                volume_ratio,
                body_atr,
                close_position,
                metadata,
            ),
            f"{timeframe}_{direction.name.lower()}_{mode}",
        )

    def _false_break(
        self,
        direction: Direction,
        latest: Candle,
        probe: Candle,
        local_low: float,
        local_high: float,
        probe_wick_ratio: float,
    ) -> bool:
        if probe_wick_ratio < max(0.0, self.config.min_false_break_wick_ratio):
            return False
        if direction is Direction.LONG:
            return (
                probe.low < local_low
                and probe.close > local_low
                and latest.low >= probe.low
                and latest.close > latest.open
                and latest.close > probe.high
            )
        return (
            probe.high > local_high
            and probe.close < local_high
            and latest.high <= probe.high
            and latest.close < latest.open
            and latest.close < probe.low
        )

    @staticmethod
    def _pivot_break(
        direction: Direction,
        latest: Candle,
        probe: Candle,
        previous: Candle,
        setup: MtfSetupResult,
    ) -> bool:
        if direction is Direction.LONG:
            return (
                probe.low > setup.support
                and probe.low < previous.low
                and latest.low > probe.low
                and latest.close > latest.open
                and latest.close > probe.high
            )
        return (
            probe.high < setup.resistance
            and probe.high > previous.high
            and latest.high < probe.high
            and latest.close < latest.open
            and latest.close < probe.low
        )


def classify_structure_overlap(
    trigger_time: datetime,
    structure_times: Iterable[datetime],
    window_minutes: int = 120,
) -> dict[str, object]:
    deltas = sorted((item - trigger_time).total_seconds() / 60.0 for item in structure_times)
    same_bar = any(abs(delta) < 1e-9 for delta in deltas)
    following = [delta for delta in deltas if 0.0 <= delta <= max(0, window_minutes)]
    nearby = [delta for delta in deltas if abs(delta) <= max(0, window_minutes)]
    return {
        "same_bar_structure_candidate": same_bar,
        "structure_followed_within_window": bool(following),
        "minutes_to_next_structure": min(following) if following else None,
        "independent_from_structure_window": not nearby,
    }


def _is_structure_break(direction: Direction, latest: Candle, previous: list[Candle]) -> bool:
    if not previous:
        return False
    if direction is Direction.LONG:
        return latest.low >= min(item.low for item in previous) and latest.close > max(item.high for item in previous)
    return latest.high <= max(item.high for item in previous) and latest.close < min(item.low for item in previous)


def _latest_volume_ratio(candles: list[Candle], period: int) -> float:
    if len(candles) < 2:
        return 0.0
    previous = candles[max(0, len(candles) - period - 1):-1]
    average = sum(max(0.0, item.volume) for item in previous) / max(len(previous), 1)
    return max(0.0, candles[-1].volume) / max(average, 1e-12)
