from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Callable

from .data import interval_to_milliseconds
from .indicators import atr
from .indicators import ema
from .indicators import macd
from .indicators import rsi
from .live_config import load_live_config
from .models import Candle
from .models import Direction
from .models import Signal


MTF_REASON_TOKEN = "mtf_4h_rsi_regime_pullback"
MTF_LONG_REASON = "long_mtf_4h_rsi_regime_pullback"
MTF_SHORT_REASON = "short_mtf_4h_rsi_regime_pullback"


def mtf_min_rank_score_for_direction(strategy: Any, direction: Direction) -> float:
    global_floor = float(getattr(strategy, "mtf_min_rank_score", -999.0))
    field = "mtf_long_min_rank_score" if direction is Direction.LONG else "mtf_short_min_rank_score"
    return max(global_floor, float(getattr(strategy, field, -999.0)))


@dataclass(frozen=True)
class MtfRegimeResult:
    regime: str
    rsi: float
    reason: str = ""
    ema_slope_atr: float = 0.0
    ema_spread_atr: float = 0.0


@dataclass(frozen=True)
class MtfSetupResult:
    direction: Direction
    support: float
    resistance: float
    ema21: float
    ema34: float
    setup_low: float
    setup_high: float
    rsi: float


@dataclass(frozen=True)
class MtfTriggerResult:
    direction: Direction
    mode: str
    candle: Candle
    atr15m: float
    volume_ratio: float = 1.0
    body_atr: float = 0.0
    close_position: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MtfSignalDecision:
    signal: Signal | None
    candle: Candle | None
    rank_candles: list[Candle]
    metadata: dict[str, Any]
    reject_reason: str = ""


@dataclass(frozen=True)
class OiFeature:
    timestamp: datetime
    available_time: datetime
    oi_value: float
    oi_chg_30m: float | None


@dataclass(frozen=True)
class FundingFeature:
    timestamp: datetime
    funding_rate: float


class Mtf4hRsiRegimePullbackStrategy:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.strategy = config.strategy if hasattr(config, "strategy") else config

    def regime_4h(self, candles: list[Candle]) -> MtfRegimeResult:
        return self.regime(candles, "4h")

    def regime(self, candles: list[Candle], timeframe: str | None = None) -> MtfRegimeResult:
        strategy = self.strategy
        timeframe = _normal_timeframe(timeframe or getattr(strategy, "mtf_regime_timeframe", "4h"), "4h")
        prefix = "mtf_2h" if timeframe == "2h" else "mtf_4h"
        label = timeframe
        period = max(2, int(_tf_get(strategy, prefix, "rsi_period", getattr(strategy, "mtf_4h_rsi_period", 14))))
        lookback = max(2, int(_tf_get(strategy, prefix, "rsi_lookback_bars", getattr(strategy, "mtf_4h_rsi_lookback_bars", 8))))
        ema_fast_period = max(2, int(_tf_get(strategy, prefix, "ema_fast", getattr(strategy, "mtf_4h_ema_fast", 9))))
        ema_mid_period = max(2, int(_tf_get(strategy, prefix, "ema_mid", getattr(strategy, "mtf_4h_ema_mid", 21))))
        minimum = max(period + lookback + 3, ema_mid_period + 3, 36)
        if len(candles) < minimum:
            return MtfRegimeResult("NO_TRADE", 50.0, f"{label}_warmup")

        closes = [item.close for item in candles]
        rsi_values = rsi(closes, period)
        ema_fast_values = ema(closes, ema_fast_period)
        ema_mid_values = ema(closes, ema_mid_period)
        _, _, hist = macd(closes)
        index = len(candles) - 1
        current = candles[-1]
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        recent_rsi = rsi_values[-lookback:]
        regime_mode = str(getattr(strategy, "mtf_regime_mode", "rsi_reversal")).strip().lower()
        if regime_mode == "trend_pullback":
            atr_value = max(atr(candles, 14)[-1], 1e-12)
            slope_lookback = max(1, int(getattr(strategy, "mtf_trend_slope_lookback_bars", 3)))
            slope_start = max(0, index - slope_lookback)
            ema_slope_atr = (ema_mid_values[index] - ema_mid_values[slope_start]) / atr_value
            ema_spread_atr = (ema_fast_values[index] - ema_mid_values[index]) / atr_value
            trend_distance = abs(current.close - ema_mid_values[index]) / max(current.close, 1e-12)
            if trend_distance > max(0.0, float(getattr(strategy, "mtf_trend_max_distance_from_ema21_pct", 0.08))):
                return MtfRegimeResult(
                    "NO_TRADE",
                    current_rsi,
                    f"{label}_trend_far_from_ema21",
                    ema_slope_atr,
                    ema_spread_atr,
                )
            require_trend_macd = bool(getattr(strategy, "mtf_trend_require_macd_support", True))
            long_macd = hist[index] > 0 or hist[index] > hist[index - 1]
            short_macd = hist[index] < 0 or hist[index] < hist[index - 1]
            long_bias = (
                bool(getattr(strategy, "mtf_allow_long", True))
                and current.close > ema_mid_values[index]
                and ema_fast_values[index] > ema_mid_values[index]
                and ema_slope_atr >= float(getattr(strategy, "mtf_trend_long_min_ema_slope_atr", 0.05))
                and float(getattr(strategy, "mtf_trend_long_rsi_min", 48.0)) <= current_rsi <= float(getattr(strategy, "mtf_trend_long_rsi_max", 72.0))
                and (not require_trend_macd or long_macd)
            )
            if long_bias:
                return MtfRegimeResult("LONG_BIAS", current_rsi, f"{label}_trend_up", ema_slope_atr, ema_spread_atr)
            short_bias = (
                bool(getattr(strategy, "mtf_allow_short", True))
                and current.close < ema_mid_values[index]
                and ema_fast_values[index] < ema_mid_values[index]
                and ema_slope_atr <= float(getattr(strategy, "mtf_trend_short_max_ema_slope_atr", -0.08))
                and float(getattr(strategy, "mtf_trend_short_rsi_min", 28.0)) <= current_rsi <= float(getattr(strategy, "mtf_trend_short_rsi_max", 52.0))
                and (not require_trend_macd or short_macd)
            )
            if short_bias:
                return MtfRegimeResult("SHORT_BIAS", current_rsi, f"{label}_trend_down", ema_slope_atr, ema_spread_atr)
            return MtfRegimeResult("NO_TRADE", current_rsi, f"{label}_no_trend", ema_slope_atr, ema_spread_atr)

        distance = abs(current.close - ema_mid_values[-1]) / max(current.close, 1e-12)
        max_distance = max(0.0, float(_tf_get(strategy, prefix, "max_distance_from_ema21_pct", getattr(strategy, "mtf_4h_max_distance_from_ema21_pct", 0.035))))
        if distance > max_distance:
            return MtfRegimeResult("NO_TRADE", current_rsi, f"{label}_far_from_ema21 distance={distance:.4f}")

        require_hist = bool(_tf_get(strategy, prefix, "require_macd_hist_turn", getattr(strategy, "mtf_4h_require_macd_hist_turn", True)))
        hist_improves = len(hist) >= 3 and hist[-1] >= hist[-2] >= hist[-3]
        hist_weakens = len(hist) >= 3 and hist[-1] <= hist[-2] <= hist[-3]
        last_two_no_new_low = candles[-1].low >= candles[-2].low or min(item.low for item in candles[-2:]) >= min(item.low for item in candles[-4:-2])
        last_two_no_new_high = candles[-1].high <= candles[-2].high or max(item.high for item in candles[-2:]) <= max(item.high for item in candles[-4:-2])

        if bool(_tf_get(strategy, prefix, "require_divergence", getattr(strategy, "mtf_4h_require_divergence", False))):
            bullish_divergence = _bullish_divergence(candles, rsi_values, lookback)
            bearish_divergence = _bearish_divergence(candles, rsi_values, lookback)
        else:
            bullish_divergence = bearish_divergence = True

        long_bias = (
            bool(getattr(strategy, "mtf_allow_long", True))
            and min(recent_rsi) <= float(_tf_get(strategy, prefix, "long_rsi_oversold", getattr(strategy, "mtf_4h_long_rsi_oversold", 34.0)))
            and current_rsi >= float(_tf_get(strategy, prefix, "long_rsi_reclaim", getattr(strategy, "mtf_4h_long_rsi_reclaim", 38.0)))
            and current_rsi > previous_rsi
            and (not require_hist or hist_improves or current.close > ema_fast_values[-1])
            and last_two_no_new_low
            and bullish_divergence
        )
        if long_bias:
            return MtfRegimeResult("LONG_BIAS", current_rsi, f"{label}_rsi_reclaim")

        short_bias = (
            bool(getattr(strategy, "mtf_allow_short", True))
            and max(recent_rsi) >= float(_tf_get(strategy, prefix, "short_rsi_overbought", getattr(strategy, "mtf_4h_short_rsi_overbought", 66.0)))
            and current_rsi <= float(_tf_get(strategy, prefix, "short_rsi_reject", getattr(strategy, "mtf_4h_short_rsi_reject", 62.0)))
            and current_rsi < previous_rsi
            and (not require_hist or hist_weakens or current.close < ema_fast_values[-1])
            and last_two_no_new_high
            and bearish_divergence
        )
        if short_bias:
            return MtfRegimeResult("SHORT_BIAS", current_rsi, f"{label}_rsi_reject")

        if min(recent_rsi) <= float(_tf_get(strategy, prefix, "long_rsi_oversold", getattr(strategy, "mtf_4h_long_rsi_oversold", 34.0))):
            return MtfRegimeResult("LONG_WATCH", current_rsi, f"{label}_oversold_watch")
        if max(recent_rsi) >= float(_tf_get(strategy, prefix, "short_rsi_overbought", getattr(strategy, "mtf_4h_short_rsi_overbought", 66.0))):
            return MtfRegimeResult("SHORT_WATCH", current_rsi, f"{label}_overbought_watch")
        return MtfRegimeResult("NO_TRADE", current_rsi, f"{label}_no_regime")

    def confirm_1h(self, direction: Direction, candles: list[Candle], btc_1h_return: float) -> tuple[bool, str, float, float]:
        strategy = self.strategy
        period = max(2, int(getattr(strategy, "mtf_1h_ema_period", 21)))
        swing = max(1, int(getattr(strategy, "mtf_1h_swing_lookback", 3)))
        minimum = max(period + 3, 24, swing + 3)
        if len(candles) < minimum:
            return False, "1h_warmup", 50.0, 0.0

        closes = [item.close for item in candles]
        rsi_values = rsi(closes, 14)
        _, _, hist = macd(closes)
        ema_values = ema(closes, period)
        current = candles[-1]
        if direction == Direction.LONG:
            if current.close <= ema_values[-1]:
                return False, "1h_long_below_ema21", rsi_values[-1], ema_values[-1]
            if rsi_values[-1] <= float(getattr(strategy, "mtf_1h_long_min_rsi", 45.0)):
                return False, "1h_long_rsi_low", rsi_values[-1], ema_values[-1]
            if min(item.low for item in candles[-swing:]) < min(item.low for item in candles[-swing - 1:-1]):
                return False, "1h_long_new_low", rsi_values[-1], ema_values[-1]
            if hist[-1] < hist[-2]:
                return False, "1h_long_macd_hist_down", rsi_values[-1], ema_values[-1]
            if btc_1h_return <= float(getattr(strategy, "mtf_btc_1h_long_min_return_pct", -0.006)):
                return False, "btc_1h_too_weak_for_long", rsi_values[-1], ema_values[-1]
            return True, "1h_long_confirmed", rsi_values[-1], ema_values[-1]

        if direction == Direction.SHORT:
            if current.close >= ema_values[-1]:
                return False, "1h_short_above_ema21", rsi_values[-1], ema_values[-1]
            if rsi_values[-1] >= float(getattr(strategy, "mtf_1h_short_max_rsi", 55.0)):
                return False, "1h_short_rsi_high", rsi_values[-1], ema_values[-1]
            if max(item.high for item in candles[-swing:]) > max(item.high for item in candles[-swing - 1:-1]):
                return False, "1h_short_new_high", rsi_values[-1], ema_values[-1]
            if hist[-1] > hist[-2]:
                return False, "1h_short_macd_hist_up", rsi_values[-1], ema_values[-1]
            if btc_1h_return >= float(getattr(strategy, "mtf_btc_1h_short_max_return_pct", 0.006)):
                return False, "btc_1h_too_strong_for_short", rsi_values[-1], ema_values[-1]
            return True, "1h_short_confirmed", rsi_values[-1], ema_values[-1]

        return False, "flat_direction", 50.0, 0.0

    def setup_30m(self, direction: Direction, candles: list[Candle], one_h_swing_low: float, one_h_swing_high: float) -> tuple[MtfSetupResult | None, str]:
        strategy = self.strategy
        ema_period = max(2, int(getattr(strategy, "mtf_30m_ema_period", 21)))
        alt_period = max(ema_period, int(getattr(strategy, "mtf_30m_alt_ema_period", 34)))
        swing = max(2, int(getattr(strategy, "mtf_30m_swing_lookback", 6)))
        minimum = max(alt_period + 3, swing + 3, 40)
        if len(candles) < minimum:
            return None, "30m_warmup"
        closes = [item.close for item in candles]
        ema21 = ema(closes, ema_period)[-1]
        ema34 = ema(closes, alt_period)[-1]
        current_rsi = rsi(closes, 14)[-1]
        latest = candles[-1]
        rsi_min = float(getattr(strategy, "mtf_30m_rsi_min", 42.0))
        rsi_max = float(getattr(strategy, "mtf_30m_rsi_max", 58.0))
        if not (rsi_min <= current_rsi <= rsi_max):
            return None, "30m_rsi_not_neutral"
        max_distance = max(0.0, float(getattr(strategy, "mtf_30m_max_distance_from_ema_pct", 0.008)))
        near_ema = min(
            abs(latest.close - ema21),
            abs(latest.low - ema21),
            abs(latest.high - ema21),
            abs(latest.close - ema34),
            abs(latest.low - ema34),
            abs(latest.high - ema34),
        ) / max(latest.close, 1e-12)
        if near_ema > max_distance:
            return None, "30m_not_near_ema"
        if abs(latest.close - ema21) / max(latest.close, 1e-12) > max_distance:
            return None, "30m_close_far_from_ema21"

        setup_low = min(item.low for item in candles[-swing:])
        setup_high = max(item.high for item in candles[-swing:])
        if direction == Direction.LONG and latest.low < one_h_swing_low:
            return None, "30m_long_broke_1h_swing_low"
        if direction == Direction.SHORT and latest.high > one_h_swing_high:
            return None, "30m_short_broke_1h_swing_high"
        return MtfSetupResult(direction, setup_low, setup_high, ema21, ema34, setup_low, setup_high, current_rsi), "30m_setup_confirmed"

    def trigger_15m(self, direction: Direction, candles: list[Candle], setup: MtfSetupResult) -> tuple[MtfTriggerResult | None, str]:
        return self.trigger_timeframe(direction, candles, setup, "15m")

    def trigger_timeframe(
        self,
        direction: Direction,
        candles: list[Candle],
        setup: MtfSetupResult,
        timeframe: str | None = None,
    ) -> tuple[MtfTriggerResult | None, str]:
        strategy = self.strategy
        timeframe = _normal_timeframe(timeframe or getattr(strategy, "mtf_trigger_timeframe", "15m"), "15m")
        mode = str(getattr(strategy, "mtf_15m_trigger_mode", "both")).lower()
        if mode not in {"sweep", "structure_break", "both", "ema_reclaim", "structure_or_reclaim"}:
            mode = "both"
        lookback = max(2, int(getattr(strategy, "mtf_15m_structure_break_lookback", 3)))
        minimum = max(lookback + 3, 24)
        if len(candles) < minimum:
            return None, f"{timeframe}_warmup"
        latest = candles[-1]
        previous = candles[-1 - lookback:-1]
        closes = [item.close for item in candles]
        atr_values = atr(candles, 14)
        atr15 = max(atr_values[-1], latest.close * 0.0008)
        body_atr = abs(latest.close - latest.open) / max(atr15, 1e-12)
        if body_atr > float(getattr(strategy, "mtf_15m_max_body_atr", 1.8)):
            return None, f"{timeframe}_body_too_large"
        volume_ratio = _latest_volume_ratio(candles, max(1, int(getattr(strategy, "mtf_trigger_volume_period", 20))))
        max_volume_ratio = max(0.0, float(getattr(strategy, "mtf_trigger_max_volume_ratio", 999.0)))
        if max_volume_ratio > 0 and volume_ratio > max_volume_ratio:
            return None, f"{timeframe}_volume_too_high"
        if bool(getattr(strategy, "mtf_trigger_volume_filter_enabled", False)):
            min_volume_ratio = max(0.0, float(getattr(strategy, "mtf_trigger_min_volume_ratio", 1.20)))
            if volume_ratio < min_volume_ratio:
                return None, f"{timeframe}_volume_too_low"
        max_distance = max(0.0, float(getattr(strategy, "mtf_15m_max_distance_from_30m_ema_pct", 0.010)))
        if abs(latest.close - setup.ema21) / max(latest.close, 1e-12) > max_distance:
            return None, f"{timeframe}_far_from_30m_ema"

        candle_range = max(latest.high - latest.low, 1e-12)
        close_position = (latest.close - latest.low) / candle_range
        reclaim_period = max(2, int(getattr(strategy, "mtf_reclaim_ema_period", 9)))
        reclaim_ema = ema(closes, reclaim_period)
        _, _, histogram = macd(closes)
        reclaim_min_body = max(0.0, float(getattr(strategy, "mtf_reclaim_min_body_atr", 0.25)))
        reclaim_min_volume = max(0.0, float(getattr(strategy, "mtf_reclaim_min_volume_ratio", 0.80)))
        require_reclaim_macd = bool(getattr(strategy, "mtf_reclaim_require_macd_improvement", True))
        if direction == Direction.LONG:
            sweep_ref = min(setup.support, setup.ema21)
            sweep = (
                latest.low < sweep_ref
                and latest.close > sweep_ref
                and latest.close > latest.open
                and close_position >= float(getattr(strategy, "mtf_15m_min_close_position_long", 0.60))
            )
            structure = (
                latest.low >= min(item.low for item in previous)
                and latest.close > max(item.high for item in previous)
                and body_atr >= max(0.0, float(getattr(strategy, "mtf_structure_min_body_atr", 0.0)))
                and close_position >= float(getattr(strategy, "mtf_structure_min_close_position_long", 0.0))
            )
            reclaim = (
                bool(getattr(strategy, "mtf_reclaim_allow_long", True))
                and candles[-2].close <= reclaim_ema[-2]
                and latest.close > reclaim_ema[-1]
                and latest.close > latest.open
                and latest.low >= setup.support
                and body_atr >= reclaim_min_body
                and close_position >= float(getattr(strategy, "mtf_reclaim_min_close_position_long", 0.65))
                and volume_ratio >= reclaim_min_volume
                and (not require_reclaim_macd or histogram[-1] > histogram[-2])
            )
            if mode in {"sweep", "both"} and sweep:
                return MtfTriggerResult(direction, "sweep", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_long_sweep"
            if mode in {"structure_break", "both", "structure_or_reclaim"} and structure:
                return MtfTriggerResult(direction, "structure_break", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_long_structure_break"
            if mode in {"ema_reclaim", "structure_or_reclaim"} and reclaim:
                return MtfTriggerResult(direction, "ema_reclaim", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_long_ema_reclaim"

        if direction == Direction.SHORT:
            sweep_ref = max(setup.resistance, setup.ema21)
            sweep = (
                latest.high > sweep_ref
                and latest.close < sweep_ref
                and latest.close < latest.open
                and close_position <= float(getattr(strategy, "mtf_15m_max_close_position_short", 0.40))
            )
            structure = (
                latest.high <= max(item.high for item in previous)
                and latest.close < min(item.low for item in previous)
                and body_atr >= max(0.0, float(getattr(strategy, "mtf_structure_min_body_atr", 0.0)))
                and close_position <= float(getattr(strategy, "mtf_structure_max_close_position_short", 1.0))
            )
            reclaim = (
                bool(getattr(strategy, "mtf_reclaim_allow_short", True))
                and candles[-2].close >= reclaim_ema[-2]
                and latest.close < reclaim_ema[-1]
                and latest.close < latest.open
                and latest.high <= setup.resistance
                and body_atr >= reclaim_min_body
                and close_position <= float(getattr(strategy, "mtf_reclaim_max_close_position_short", 0.35))
                and volume_ratio >= reclaim_min_volume
                and (not require_reclaim_macd or histogram[-1] < histogram[-2])
            )
            if mode in {"sweep", "both"} and sweep:
                return MtfTriggerResult(direction, "sweep", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_short_sweep"
            if mode in {"structure_break", "both", "structure_or_reclaim"} and structure:
                return MtfTriggerResult(direction, "structure_break", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_short_structure_break"
            if mode in {"ema_reclaim", "structure_or_reclaim"} and reclaim:
                return MtfTriggerResult(direction, "ema_reclaim", latest, atr15, volume_ratio, body_atr, close_position), f"{timeframe}_short_ema_reclaim"
        return None, f"{timeframe}_no_trigger"

    def build_signal(
        self,
        symbol: str,
        candles_15m: list[Candle],
        candles_30m: list[Candle],
        candles_1h: list[Candle],
        candles_4h: list[Candle],
        btc_1h_candles: list[Candle],
        btc_4h_candles: list[Candle],
        oi_chg_30m: float | None,
        funding_rate: float,
        candles_regime: list[Candle] | None = None,
        candles_trigger: list[Candle] | None = None,
        regime_timeframe: str | None = None,
        trigger_timeframe: str | None = None,
        trigger_detector: Callable[
            [Direction, list[Candle], MtfSetupResult, str],
            tuple[MtfTriggerResult | None, str],
        ] | None = None,
    ) -> MtfSignalDecision:
        regime_timeframe = _normal_timeframe(regime_timeframe or getattr(self.strategy, "mtf_regime_timeframe", "4h"), "4h")
        trigger_timeframe = _normal_timeframe(trigger_timeframe or getattr(self.strategy, "mtf_trigger_timeframe", "15m"), "15m")
        regime_candles = candles_regime or candles_4h
        trigger_candles = candles_trigger or candles_15m
        metadata: dict[str, Any] = {
            "symbol": symbol,
            "strategy": MTF_REASON_TOKEN,
            "regime_timeframe": regime_timeframe,
            "trigger_timeframe": trigger_timeframe,
            "funding_rate": funding_rate,
            "oi_chg_30m": oi_chg_30m,
        }
        regime = self.regime(regime_candles, regime_timeframe)
        metadata["4h_regime"] = regime.regime
        metadata["4h_rsi"] = regime.rsi
        metadata["regime_reason"] = regime.reason
        metadata["regime_ema_slope_atr"] = regime.ema_slope_atr
        metadata["regime_ema_spread_atr"] = regime.ema_spread_atr
        if regime.regime == "LONG_BIAS":
            direction = Direction.LONG
        elif regime.regime == "SHORT_BIAS":
            direction = Direction.SHORT
        else:
            return MtfSignalDecision(None, None, trigger_candles[-30:] if trigger_candles else [], metadata, regime.reason)

        btc_1h_return = _period_return(btc_1h_candles, 1)
        btc_4h_return = _period_return(btc_4h_candles, 1)
        btc_state = _btc_state_for_direction(direction, btc_1h_return, btc_4h_return)
        metadata["btc_1h_return"] = btc_1h_return
        metadata["btc_4h_return"] = btc_4h_return
        metadata["btc_state"] = btc_state

        rejected = self._futures_filter_reject_reason(direction, oi_chg_30m, funding_rate, btc_1h_return, btc_4h_return)
        if rejected:
            return MtfSignalDecision(None, None, trigger_candles[-30:] if trigger_candles else [], metadata, rejected)

        one_h_ok, one_h_reason, one_h_rsi, _one_h_ema = self.confirm_1h(direction, candles_1h, btc_1h_return)
        metadata["1h_rsi"] = one_h_rsi
        if not one_h_ok:
            return MtfSignalDecision(None, None, trigger_candles[-30:] if trigger_candles else [], metadata, one_h_reason)

        swing_lookback = max(1, int(getattr(self.strategy, "mtf_1h_swing_lookback", 3)))
        one_h_swing_low = min(item.low for item in candles_1h[-swing_lookback:])
        one_h_swing_high = max(item.high for item in candles_1h[-swing_lookback:])
        setup, setup_reason = self.setup_30m(direction, candles_30m, one_h_swing_low, one_h_swing_high)
        if setup is None:
            return MtfSignalDecision(None, None, trigger_candles[-30:] if trigger_candles else [], metadata, setup_reason)

        if trigger_detector is None:
            trigger, trigger_reason = self.trigger_timeframe(direction, trigger_candles, setup, trigger_timeframe)
        else:
            trigger, trigger_reason = trigger_detector(direction, trigger_candles, setup, trigger_timeframe)
        if trigger is None:
            return MtfSignalDecision(None, None, trigger_candles[-30:] if trigger_candles else [], metadata, trigger_reason)

        entry_ref = trigger.candle.close
        min_entry_price = max(0.0, float(getattr(self.strategy, "mtf_min_entry_price", 0.0)))
        if entry_ref < min_entry_price:
            return MtfSignalDecision(None, None, trigger_candles[-30:], metadata, "mtf_entry_price_too_low")
        if direction == Direction.LONG:
            stop_price = min(trigger.candle.low, setup.setup_low) - trigger.atr15m * float(getattr(self.strategy, "mtf_stop_atr15m_mult", 0.25))
            stop_loss_pct = (entry_ref - stop_price) / max(entry_ref, 1e-12)
        else:
            stop_price = max(trigger.candle.high, setup.setup_high) + trigger.atr15m * float(getattr(self.strategy, "mtf_stop_atr15m_mult", 0.25))
            stop_loss_pct = (stop_price - entry_ref) / max(entry_ref, 1e-12)
        max_stop = max(0.0, float(getattr(self.strategy, "mtf_max_stop_pct", 0.020)))
        if stop_loss_pct <= 0:
            return MtfSignalDecision(None, None, trigger_candles[-30:], metadata, "mtf_bad_stop")
        min_stop = max(0.0, float(getattr(self.strategy, "mtf_min_stop_pct", 0.0)))
        if min_stop > 0 and stop_loss_pct < min_stop:
            return MtfSignalDecision(None, None, trigger_candles[-30:], metadata, "mtf_stop_too_tight")
        if max_stop > 0 and stop_loss_pct > max_stop:
            return MtfSignalDecision(None, None, trigger_candles[-30:], metadata, "mtf_stop_too_wide")

        take_profit_pct = stop_loss_pct * max(0.1, float(getattr(self.strategy, "mtf_take_profit_r", 2.0)))
        max_holding_minutes = max(1, int(getattr(self.strategy, "mtf_max_holding_minutes", 720)))
        max_holding_bars = max(1, math.ceil(max_holding_minutes / 30.0))
        reason = _mtf_reason(
            direction,
            regime,
            trigger.mode,
            funding_rate,
            oi_chg_30m,
            btc_state,
            regime_timeframe,
            trigger_timeframe,
            trigger.volume_ratio,
            str(getattr(self.strategy, "mtf_regime_mode", "rsi_reversal")),
        )
        signal = Signal(
            direction,
            0.62,
            reason,
            stop_loss_pct,
            take_profit_pct,
            risk_multiplier=max(0.05, float(getattr(self.strategy, "mtf_risk_multiplier", 0.55))),
            max_holding_bars=max_holding_bars,
        )
        metadata["trigger_mode"] = trigger.mode
        metadata["trigger_volume_ratio"] = trigger.volume_ratio
        metadata["trigger_body_atr"] = trigger.body_atr
        metadata["trigger_close_position"] = trigger.close_position
        metadata["trigger_atr"] = trigger.atr15m
        metadata["trigger_close"] = trigger.candle.close
        metadata.update(trigger.metadata)
        metadata["structural_stop_price"] = stop_price
        metadata["stop_loss_pct"] = stop_loss_pct
        metadata["take_profit_pct"] = take_profit_pct
        metadata["take_profit_r"] = max(0.1, float(getattr(self.strategy, "mtf_take_profit_r", 2.0)))
        metadata["4h_rsi_bucket"] = _rsi_bucket(regime.rsi)
        metadata["funding_bucket"] = _funding_bucket(funding_rate)
        metadata["oi_change_bucket"] = _oi_bucket(oi_chg_30m)
        return MtfSignalDecision(signal, trigger.candle, trigger_candles[-80:], metadata, "")

    def one_h_confirm_lost(self, direction: Direction, candles_1h: list[Candle]) -> bool:
        period = max(2, int(getattr(self.strategy, "mtf_1h_ema_period", 21)))
        if len(candles_1h) < period + 2:
            return False
        closes = [item.close for item in candles_1h]
        ema_values = ema(closes, period)
        if direction == Direction.LONG:
            return closes[-1] < ema_values[-1]
        if direction == Direction.SHORT:
            return closes[-1] > ema_values[-1]
        return False

    def thirty_minute_confirm_lost(self, direction: Direction, candles_30m: list[Candle]) -> bool:
        period = max(2, int(getattr(self.strategy, "mtf_30m_ema_period", 21)))
        confirm_bars = max(1, int(getattr(self.strategy, "mtf_30m_exit_confirm_bars", 2)))
        if len(candles_30m) < period + confirm_bars + 1:
            return False
        closes = [item.close for item in candles_30m]
        ema_values = ema(closes, period)
        if direction == Direction.LONG:
            structure_lost = all(
                closes[index] < ema_values[index]
                for index in range(-confirm_bars, 0)
            )
        elif direction == Direction.SHORT:
            structure_lost = all(
                closes[index] > ema_values[index]
                for index in range(-confirm_bars, 0)
            )
        else:
            return False
        if not structure_lost:
            return False
        if not bool(getattr(self.strategy, "mtf_30m_exit_require_macd_adverse", True)):
            return True
        _, _, histogram = macd(
            closes,
            self.config.filters.macd_fast,
            self.config.filters.macd_slow,
            self.config.filters.macd_signal,
        )
        if direction == Direction.LONG:
            return histogram[-1] < 0.0 and histogram[-1] <= histogram[-2]
        return histogram[-1] > 0.0 and histogram[-1] >= histogram[-2]

    def _futures_filter_reject_reason(
        self,
        direction: Direction,
        oi_chg_30m: float | None,
        funding_rate: float,
        btc_1h_return: float,
        btc_4h_return: float,
    ) -> str | None:
        strategy = self.strategy
        if direction == Direction.LONG:
            if bool(getattr(strategy, "mtf_use_funding_filter", True)):
                if funding_rate < float(getattr(strategy, "mtf_long_min_funding_rate", -1.0)):
                    return "mtf_rejected_funding"
                if funding_rate > float(getattr(strategy, "mtf_long_max_funding_rate", 0.0002)):
                    return "mtf_rejected_funding"
            if btc_1h_return <= float(getattr(strategy, "mtf_btc_1h_long_min_return_pct", -0.006)):
                return "mtf_rejected_btc"
            if btc_4h_return < float(getattr(strategy, "mtf_btc_4h_long_min_return_pct", -1.0)):
                return "mtf_rejected_btc_4h_long_regime"
            if bool(getattr(strategy, "mtf_btc_4h_block_strong_opposite", True)) and btc_4h_return <= -0.012:
                return "mtf_rejected_btc"
        elif direction == Direction.SHORT:
            if bool(getattr(strategy, "mtf_use_funding_filter", True)):
                if funding_rate < float(getattr(strategy, "mtf_short_min_funding_rate", -0.0002)):
                    return "mtf_rejected_funding"
                if funding_rate > float(getattr(strategy, "mtf_short_max_funding_rate", 1.0)):
                    return "mtf_rejected_funding"
            if btc_1h_return >= float(getattr(strategy, "mtf_btc_1h_short_max_return_pct", 0.006)):
                return "mtf_rejected_btc"
            if bool(getattr(strategy, "mtf_btc_4h_block_strong_opposite", True)) and btc_4h_return >= 0.012:
                return "mtf_rejected_btc"
        if bool(getattr(strategy, "mtf_use_oi_filter", True)):
            if oi_chg_30m is None:
                return "mtf_rejected_oi_missing"
            if oi_chg_30m > float(getattr(strategy, "mtf_oi_30m_max_increase_pct", 0.02)):
                return "mtf_rejected_oi"
        return None


def closed_candles_for_decision(
    candles: list[Candle],
    timestamps: list[datetime],
    decision_time: datetime,
    timeframe: str,
    limit: int,
) -> list[Candle]:
    end = available_candle_end(timestamps, decision_time, timeframe)
    if end <= 0:
        return []
    return candles[max(0, end - limit):end]


def available_candle_end(timestamps: list[datetime], decision_time: datetime, timeframe: str) -> int:
    available_before = decision_time - timedelta(milliseconds=interval_to_milliseconds(timeframe))
    return bisect.bisect_right(timestamps, available_before)


def build_oi_features(rows: list[dict[str, str]]) -> list[OiFeature]:
    points_by_timestamp: dict[datetime, float] = {}
    for row in rows:
        timestamp = _parse_dt(row.get("timestamp") or row.get("time") or "")
        if timestamp is None:
            continue
        try:
            value = float(row.get("sumOpenInterest", row.get("sum_open_interest", "0")) or 0)
            if value <= 0:
                value = float(row.get("sumOpenInterestValue", row.get("sum_open_interest_value", "0")) or 0)
        except ValueError:
            continue
        if value > 0:
            points_by_timestamp[timestamp] = value
    points = sorted(points_by_timestamp.items())
    output: list[OiFeature] = []
    for index, (timestamp, value) in enumerate(points):
        oi_chg_30m = None
        if index >= 6 and points[index - 6][1] > 0:
            oi_chg_30m = value / points[index - 6][1] - 1.0
        output.append(OiFeature(timestamp, timestamp + timedelta(minutes=5), value, oi_chg_30m))
    return output


def build_funding_features(rows: list[dict[str, str]]) -> list[FundingFeature]:
    output_by_timestamp: dict[datetime, FundingFeature] = {}
    for row in rows:
        timestamp = _parse_dt(row.get("timestamp") or row.get("fundingTime") or "")
        if timestamp is None:
            continue
        try:
            rate = float(row.get("funding_rate", row.get("fundingRate", "0")) or 0)
        except ValueError:
            rate = 0.0
        output_by_timestamp[timestamp] = FundingFeature(timestamp, rate)
    return [output_by_timestamp[timestamp] for timestamp in sorted(output_by_timestamp)]


def load_auxiliary_features(symbols: tuple[str, ...], oi_data_dir: str, funding_data_dir: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        oi = _load_oi_for_symbol(symbol, oi_data_dir)
        funding = _load_funding_for_symbol(symbol, funding_data_dir)
        output[symbol] = {
            "oi": oi,
            "oi_times": [item.available_time for item in oi],
            "funding": funding,
            "funding_times": [item.timestamp for item in funding],
        }
    return output


def oi_change_at(
    features: dict[str, Any] | None,
    timestamp: datetime,
    max_age_minutes: int = 15,
) -> float | None:
    if not features:
        return None
    items: list[OiFeature] = features.get("oi", [])
    times: list[datetime] = features.get("oi_times", [])
    index = bisect.bisect_right(times, timestamp) - 1
    if index < 0 or index >= len(items):
        return None
    item = items[index]
    if timestamp - item.available_time > timedelta(minutes=max(0, max_age_minutes)):
        return None
    return item.oi_chg_30m


def funding_at(features: dict[str, Any] | None, timestamp: datetime, default: float = 0.0) -> float:
    if not features:
        return default
    items: list[FundingFeature] = features.get("funding", [])
    times: list[datetime] = features.get("funding_times", [])
    index = bisect.bisect_right(times, timestamp) - 1
    if index < 0 or index >= len(items):
        return default
    return items[index].funding_rate


def mtf_report_from_summary(summary: dict[str, Any], reject_stats: dict[str, int] | None = None) -> dict[str, Any]:
    trades = [trade for trade in summary.get("trades", []) if MTF_REASON_TOKEN in str(trade.get("entry_reason", ""))]
    report = {
        "overall": _metric_row("overall", trades),
        "by_strategy": _group_by_tags(trades, ("strategy_bucket",)),
        "by_side": _group_by_tags(trades, ("side",)),
        "by_symbol": _group_by_tags(trades, ("symbol",)),
        "by_month": _group_by_month(trades),
        "by_4h_regime": _group_by_reason_tag(trades, "regime"),
        "by_4h_rsi_bucket": _group_by_reason_tag(trades, "rsi_bucket"),
        "by_regime_timeframe": _group_by_reason_tag(trades, "regime_tf"),
        "by_trigger_timeframe": _group_by_reason_tag(trades, "trigger_tf"),
        "by_trigger_mode": _group_by_reason_tag(trades, "trigger"),
        "by_trigger_volume_bucket": _group_by_reason_tag(trades, "vol"),
        "by_funding_bucket": _group_by_reason_tag(trades, "funding"),
        "by_oi_change_bucket": _group_by_reason_tag(trades, "oi"),
        "by_btc_state": _group_by_reason_tag(trades, "btc"),
        "fail_fast_count": sum(1 for trade in trades if trade.get("exit_reason") == "mtf_fail_fast"),
        "time_stop_count": sum(1 for trade in trades if trade.get("exit_reason") == "mtf_time_stop"),
        "one_h_confirm_lost_count": sum(1 for trade in trades if trade.get("exit_reason") == "mtf_1h_confirm_lost"),
        "rejected_stop_too_wide_count": int((reject_stats or {}).get("mtf_stop_too_wide", 0)),
        "rejected_funding_count": int((reject_stats or {}).get("mtf_rejected_funding", 0)),
        "rejected_oi_count": int((reject_stats or {}).get("mtf_rejected_oi", 0) + (reject_stats or {}).get("mtf_rejected_oi_missing", 0)),
        "rejected_btc_count": int((reject_stats or {}).get("mtf_rejected_btc", 0)),
        "reject_stats": dict(sorted((reject_stats or {}).items())),
    }
    return report


def run_experiments(
    config_path: str,
    execution_data_dir: str,
    initial_equity: float | None,
    trade_start: datetime | None,
    trade_end: datetime | None,
    experiment_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    from .live_execution_backtest import run_execution_backtest_config
    from .live_execution_backtest import _load_symbol_data
    from .live_execution_backtest import _resample_to_timeframe

    base_config = load_live_config(config_path)
    execution_candles = _load_symbol_data(execution_data_dir, tuple(base_config.trading.symbols), "1m")
    symbols = tuple(symbol for symbol in base_config.trading.symbols if symbol in execution_candles)
    execution_candles = {symbol: execution_candles[symbol] for symbol in symbols}
    signal_candles = {symbol: _resample_to_timeframe(candles, "1m", base_config.trading.timeframe) for symbol, candles in execution_candles.items()}
    output: dict[str, Any] = {"experiments": []}
    selected = {item.strip() for item in (experiment_names or ()) if item.strip()}
    for name, overrides in _experiment_overrides():
        if selected and name not in selected:
            continue
        config = _mtf_experiment_config(base_config, symbols, overrides)
        row: dict[str, Any] = {"name": name}
        mtf_candidate_cache: dict[tuple[Any, ...], Any] = {}
        shared_reject_stats: dict[str, int] | None = None
        for cost_experiment in ("no_cost", "full_cost"):
            summary = run_execution_backtest_config(
                config,
                execution_candles,
                signal_candles,
                initial_equity=initial_equity,
                include_trades=True,
                trade_start=trade_start,
                trade_end=trade_end,
                cost_experiment=cost_experiment,
                backtest_mode="conservative",
                mtf_candidate_cache=mtf_candidate_cache,
            )
            row[cost_experiment] = {
                "initial_equity": summary["initial_equity"],
                "final_equity": summary["final_equity"],
                "net_pnl": summary["net_pnl"],
                "max_drawdown_pct": summary["max_drawdown_pct"],
                "trade_count": summary["total_trades"],
                "win_rate_pct": summary["win_rate_pct"],
                "profit_factor": summary["profit_factor"],
                "gross_pnl": summary["gross_pnl"],
                "fee": summary["fee"],
                "slippage": summary["slippage_cost"],
                "funding": summary["funding"],
                "monthly": summary["monthly"],
                "mtf_report": summary.get("mtf_report", mtf_report_from_summary(summary)),
            }
            mtf_report = row[cost_experiment]["mtf_report"]
            if shared_reject_stats is None:
                shared_reject_stats = dict(mtf_report.get("reject_stats", {}))
            elif shared_reject_stats:
                row[cost_experiment]["mtf_report"] = _with_reject_stats(mtf_report, shared_reject_stats)
        output["experiments"].append(row)
    return output


def _with_reject_stats(report: dict[str, Any], reject_stats: dict[str, int]) -> dict[str, Any]:
    updated = dict(report)
    updated["reject_stats"] = dict(sorted(reject_stats.items()))
    updated["rejected_stop_too_wide_count"] = int(reject_stats.get("mtf_stop_too_wide", 0))
    updated["rejected_funding_count"] = int(reject_stats.get("mtf_rejected_funding", 0))
    updated["rejected_oi_count"] = int(reject_stats.get("mtf_rejected_oi", 0) + reject_stats.get("mtf_rejected_oi_missing", 0))
    updated["rejected_btc_count"] = int(reject_stats.get("mtf_rejected_btc", 0))
    return updated


def _load_oi_for_symbol(symbol: str, data_dir: str) -> list[OiFeature]:
    root = Path(data_dir)
    matches = sorted(root.glob(f"{symbol}_oi_5m_*.csv"))
    rows: list[dict[str, str]] = []
    for path in matches:
        rows.extend(_read_csv_rows(path))
    return build_oi_features(rows)


def _load_funding_for_symbol(symbol: str, data_dir: str) -> list[FundingFeature]:
    roots: list[Path] = []
    for root in (Path(data_dir), Path("data/binance_30m_365d")):
        if root not in roots:
            roots.append(root)
    rows: list[dict[str, str]] = []
    for root in roots:
        matches = sorted(root.glob(f"{symbol}_funding_*.csv"))
        for path in matches:
            rows.extend(_read_csv_rows(path))
    return build_funding_features(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value) / 1000.0)
        except ValueError:
            return None


def _bullish_divergence(candles: list[Candle], rsi_values: list[float], lookback: int) -> bool:
    recent_start = max(0, len(candles) - lookback)
    previous_start = max(0, recent_start - lookback)
    recent_lows = [(index, candles[index].low) for index in range(recent_start, len(candles))]
    previous_lows = [(index, candles[index].low) for index in range(previous_start, recent_start)]
    if not recent_lows or not previous_lows:
        return False
    recent_index, recent_low = min(recent_lows, key=lambda item: item[1])
    previous_index, previous_low = min(previous_lows, key=lambda item: item[1])
    return recent_low < previous_low and rsi_values[recent_index] > rsi_values[previous_index]


def _bearish_divergence(candles: list[Candle], rsi_values: list[float], lookback: int) -> bool:
    recent_start = max(0, len(candles) - lookback)
    previous_start = max(0, recent_start - lookback)
    recent_highs = [(index, candles[index].high) for index in range(recent_start, len(candles))]
    previous_highs = [(index, candles[index].high) for index in range(previous_start, recent_start)]
    if not recent_highs or not previous_highs:
        return False
    recent_index, recent_high = max(recent_highs, key=lambda item: item[1])
    previous_index, previous_high = max(previous_highs, key=lambda item: item[1])
    return recent_high > previous_high and rsi_values[recent_index] < rsi_values[previous_index]


def _period_return(candles: list[Candle], bars: int) -> float:
    if len(candles) <= bars or candles[-1 - bars].close <= 0:
        return 0.0
    return candles[-1].close / candles[-1 - bars].close - 1.0


def _btc_state_for_direction(direction: Direction, ret_1h: float, ret_4h: float) -> str:
    if ret_4h <= -0.012:
        return "btc_4h_strong_down"
    if ret_4h >= 0.012:
        return "btc_4h_strong_up"
    if ret_1h <= -0.006:
        return "btc_1h_down"
    if ret_1h >= 0.006:
        return "btc_1h_up"
    return "btc_neutral"


def _mtf_reason(
    direction: Direction,
    regime: MtfRegimeResult,
    trigger: str,
    funding_rate: float,
    oi_chg_30m: float | None,
    btc_state: str,
    regime_timeframe: str = "4h",
    trigger_timeframe: str = "15m",
    trigger_volume_ratio: float = 1.0,
    regime_mode: str = "rsi_reversal",
) -> str:
    base = MTF_LONG_REASON if direction == Direction.LONG else MTF_SHORT_REASON
    return (
        f"{base} regime={regime.regime} regime_mode={regime_mode} regime_tf={regime_timeframe} rsi_bucket={_rsi_bucket(regime.rsi)} "
        f"trigger={trigger} trigger_tf={trigger_timeframe} vol={_volume_bucket(trigger_volume_ratio)} "
        f"funding={_funding_bucket(funding_rate)} "
        f"oi={_oi_bucket(oi_chg_30m)} btc={btc_state}"
    )


def _rsi_bucket(value: float) -> str:
    if value < 38:
        return "rsi_lt_38"
    if value < 45:
        return "rsi_38_45"
    if value < 55:
        return "rsi_45_55"
    if value < 62:
        return "rsi_55_62"
    return "rsi_gte_62"


def _funding_bucket(value: float) -> str:
    if value <= -0.0002:
        return "funding_lte_-2bps"
    if value < 0:
        return "funding_neg"
    if value <= 0.0002:
        return "funding_0_2bps"
    return "funding_gt_2bps"


def _oi_bucket(value: float | None) -> str:
    if value is None:
        return "oi_missing"
    if value <= -0.02:
        return "oi_drop_gt_2pct"
    if value <= 0:
        return "oi_flat_or_down"
    if value <= 0.02:
        return "oi_up_0_2pct"
    return "oi_up_gt_2pct"


def _volume_bucket(value: float) -> str:
    if value < 1.0:
        return "vol_lt_1x"
    if value < 1.5:
        return "vol_1_1p5x"
    if value < 2.0:
        return "vol_1p5_2x"
    return "vol_gte_2x"


def _latest_volume_ratio(candles: list[Candle], period: int) -> float:
    if len(candles) < period + 1:
        return 1.0
    window = candles[-period - 1:-1]
    average = sum(item.volume for item in window) / max(1, len(window))
    if average <= 0:
        return 1.0
    return candles[-1].volume / average


def _normal_timeframe(value: str, default: str) -> str:
    value = str(value or default).strip().lower()
    try:
        interval_to_milliseconds(value)
    except ValueError:
        return default
    return value


def _tf_get(strategy: Any, prefix: str, name: str, default: Any) -> Any:
    return getattr(strategy, f"{prefix}_{name}", default)


def _reason_tags(reason: str) -> dict[str, str]:
    tags = {}
    for token in str(reason).split():
        if "=" in token:
            key, value = token.split("=", 1)
            tags[key] = value
    return tags


def _group_by_reason_tag(trades: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        key = _reason_tags(str(trade.get("entry_reason", ""))).get(tag, "unknown")
        groups.setdefault(key, []).append(trade)
    return [_metric_row(key, groups[key]) for key in sorted(groups)]


def _group_by_tags(trades: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        key = "|".join(str(trade.get(field, "unknown")) for field in fields)
        groups.setdefault(key, []).append(trade)
    return [_metric_row(key, groups[key]) for key in sorted(groups)]


def _group_by_month(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        value = str(trade.get("entry_time", ""))
        key = value[:7] if len(value) >= 7 else "unknown"
        groups.setdefault(key, []).append(trade)
    return [_metric_row(key, groups[key]) for key in sorted(groups)]


def _metric_row(name: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) > 0]
    losses = [trade for trade in trades if float(trade.get("net_pnl", 0.0)) <= 0]
    gross_profit = sum(float(trade.get("net_pnl", 0.0)) for trade in wins)
    gross_loss = abs(sum(float(trade.get("net_pnl", 0.0)) for trade in losses))
    mfe_values = [float(trade.get("mfe", 0.0)) for trade in trades]
    mae_values = [float(trade.get("mae", 0.0)) for trade in trades]
    hold_values = [float(trade.get("hold_minutes", 0.0)) for trade in trades]
    return {
        "name": name,
        "trade_count": len(trades),
        "win_rate": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "gross_pnl": sum(float(trade.get("gross_pnl", 0.0)) for trade in trades),
        "fee": sum(float(trade.get("fee", trade.get("fees", 0.0))) for trade in trades),
        "slippage": sum(float(trade.get("slippage_cost", 0.0)) for trade in trades),
        "funding": sum(float(trade.get("funding", 0.0)) for trade in trades),
        "net_pnl": sum(float(trade.get("net_pnl", 0.0)) for trade in trades),
        "expectancy": 0.0 if not trades else sum(float(trade.get("net_pnl", 0.0)) for trade in trades) / len(trades),
        "avg_mfe": 0.0 if not mfe_values else sum(mfe_values) / len(mfe_values),
        "avg_mae": 0.0 if not mae_values else sum(mae_values) / len(mae_values),
        "avg_hold_minutes": 0.0 if not hold_values else sum(hold_values) / len(hold_values),
        "median_hold_minutes": 0.0 if not hold_values else statistics.median(hold_values),
    }


def _experiment_overrides() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("A_baseline_long_short", {"mtf_allow_long": True, "mtf_allow_short": True, "mtf_15m_trigger_mode": "both"}),
        ("B_long_only", {"mtf_allow_long": True, "mtf_allow_short": False}),
        ("C_short_only", {"mtf_allow_long": False, "mtf_allow_short": True}),
        ("D_strict_4h_divergence", {"mtf_4h_require_divergence": True}),
        ("E_strict_1h", {"mtf_1h_long_min_rsi": 50.0, "mtf_1h_short_max_rsi": 50.0}),
        ("F_sweep_only", {"mtf_15m_trigger_mode": "sweep"}),
        ("G_structure_break_only", {"mtf_15m_trigger_mode": "structure_break"}),
        ("H_funding_strict", {"mtf_long_max_funding_rate": 0.0, "mtf_short_min_funding_rate": 0.0}),
        ("I_oi_strict", {"mtf_oi_30m_max_increase_pct": 0.0}),
        ("J_high_liquidity_top30", {"mtf_symbols_mode": "top30"}),
        (
            "K_cost_guard_sweep_profit",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_15m_trigger_mode": "sweep",
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.0,
            },
        ),
        (
            "L_2h_5m_volume_sweep",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "2h",
                "mtf_trigger_timeframe": "5m",
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.15,
                "mtf_trigger_volume_period": 18,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.0,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.8,
                "mtf_fail_fast_minutes": 45,
            },
        ),
        (
            "M_2h_5m_volume_loose",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "2h",
                "mtf_trigger_timeframe": "5m",
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.05,
                "mtf_trigger_volume_period": 12,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": -0.00005,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.005,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.6,
                "mtf_fail_fast_minutes": 45,
            },
        ),
        (
            "N_2h_5m_volume_strict",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "2h",
                "mtf_trigger_timeframe": "5m",
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.35,
                "mtf_trigger_volume_period": 20,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.0,
                "mtf_max_stop_pct": 0.015,
                "mtf_take_profit_r": 1.8,
                "mtf_fail_fast_minutes": 30,
            },
        ),
        (
            "O_2h_5m_volume_both",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": True,
                "mtf_regime_timeframe": "2h",
                "mtf_trigger_timeframe": "5m",
                "mtf_15m_trigger_mode": "both",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.20,
                "mtf_trigger_volume_period": 18,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": -0.00005,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_short_min_funding_rate": -0.0002,
                "mtf_short_max_funding_rate": 0.0001,
                "mtf_oi_30m_max_increase_pct": 0.005,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.6,
                "mtf_fail_fast_minutes": 45,
            },
        ),
        (
            "P_2h_5m_volume_profit_guard",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "2h",
                "mtf_trigger_timeframe": "5m",
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.05,
                "mtf_trigger_volume_period": 12,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.005,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.6,
                "mtf_fail_fast_minutes": 45,
                "mtf_risk_multiplier": 1.0,
            },
        ),
        (
            "Q_4h_5m_volume_with_2h_fallback",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "4h",
                "mtf_trigger_timeframe": "5m",
                "mtf_secondary_2h_enabled": True,
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.05,
                "mtf_trigger_volume_period": 12,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.005,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.6,
                "mtf_fail_fast_minutes": 45,
                "mtf_risk_multiplier": 1.0,
            },
        ),
        (
            "R_4h_5m_2h_fallback_looser_30m_sized",
            {
                "mtf_allow_long": True,
                "mtf_allow_short": False,
                "mtf_regime_timeframe": "4h",
                "mtf_trigger_timeframe": "5m",
                "mtf_secondary_2h_enabled": True,
                "mtf_15m_trigger_mode": "sweep",
                "mtf_trigger_volume_filter_enabled": True,
                "mtf_trigger_min_volume_ratio": 1.05,
                "mtf_trigger_volume_period": 12,
                "mtf_min_entry_price": 2.0,
                "mtf_long_min_funding_rate": 0.0,
                "mtf_long_max_funding_rate": 0.0002,
                "mtf_oi_30m_max_increase_pct": 0.005,
                "mtf_30m_rsi_min": 40.0,
                "mtf_30m_rsi_max": 62.0,
                "mtf_30m_max_distance_from_ema_pct": 0.012,
                "mtf_15m_max_distance_from_30m_ema_pct": 0.014,
                "mtf_max_stop_pct": 0.018,
                "mtf_take_profit_r": 1.6,
                "mtf_fail_fast_minutes": 45,
                "mtf_risk_multiplier": 1.0,
                "mtf_symbol_margin_pct": 0.20,
                "mtf_account_margin_usage_pct": 0.20,
            },
        ),
    ]


def _mtf_experiment_config(config: Any, symbols: tuple[str, ...], overrides: dict[str, Any]) -> Any:
    mode = str(overrides.get("mtf_symbols_mode", getattr(config.strategy, "mtf_symbols_mode", "top30")))
    selected_symbols = symbols[:30] if mode == "top30" else symbols[:50] if mode == "top50" else symbols
    strategy_values = {
        "mtf_4h_rsi_regime_enabled": True,
        "mtf_disable_legacy_strategies": True,
        "super_volume_breakout_enabled": False,
        "ordinary_breakout_enabled": False,
        "pullback_reclaim_enabled": False,
        "fast_breakout_enabled": False,
        "startup_breakout_enabled": False,
        "rsi_reversal_enabled": False,
        "oi_flush_reversal_enabled": False,
        "allow_short": True,
    }
    strategy_values.update(overrides)
    strategy = replace(config.strategy, **strategy_values)
    trading = replace(
        config.trading,
        symbols=tuple(selected_symbols),
        entry_symbols=tuple(symbol for symbol in (config.trading.entry_symbols or selected_symbols) if symbol in selected_symbols),
        max_open_positions=max(1, int(getattr(strategy, "mtf_max_open_positions", 1))),
        super_volume_extra_slot_enabled=False,
        max_new_entries_per_cycle=1,
    )
    return replace(config, trading=trading, strategy=strategy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MTF 4H RSI regime pullback experiments")
    parser.add_argument("--config", default="config.live.optimized_super_volume.json")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_3m_top100")
    parser.add_argument("--initial-equity", type=float, default=160.0)
    parser.add_argument("--trade-start", default="2026-05-10T00:00:00")
    parser.add_argument("--trade-end", default="2026-06-10T00:00:00")
    parser.add_argument("--output", default="report_mtf_4h_rsi_regime_1m_30d_160u.json")
    parser.add_argument("--experiments", default="", help="Comma separated experiment names; empty runs all")
    args = parser.parse_args()
    trade_start = _parse_dt(args.trade_start)
    trade_end = _parse_dt(args.trade_end)
    selected = tuple(item.strip() for item in args.experiments.split(",") if item.strip())
    report = run_experiments(args.config, args.execution_data_dir, args.initial_equity, trade_start, trade_end, selected or None)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
