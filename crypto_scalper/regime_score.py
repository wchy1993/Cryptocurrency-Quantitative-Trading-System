from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .data import interval_to_milliseconds
from .indicators import atr, ema, kdj, macd, rsi
from .models import Candle, Direction


@dataclass(frozen=True)
class RegimeScoreSnapshot:
    asof: str
    direction: str
    trend_score: float
    reversal_score: float
    score_gap: float
    shadow_regime: str
    components: dict[str, float]
    features: dict[str, float | bool]


class RegimeScoreEngine:
    """Computes direction-aware shadow scores from closed higher-timeframe bars."""

    def __init__(self, config: Any, candles_by_timeframe: dict[str, dict[str, list[Candle]]]) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))
        self.candles_by_timeframe = candles_by_timeframe
        self.timestamps = {
            timeframe: {
                symbol: [candle.timestamp for candle in candles]
                for symbol, candles in candles_by_symbol.items()
            }
            for timeframe, candles_by_symbol in candles_by_timeframe.items()
        }
        self.breadth_cache: dict[tuple[Any, int], tuple[float, float]] = {}

    def score(self, symbol: str, decision_time: Any, direction: Direction) -> RegimeScoreSnapshot | None:
        if not self.enabled or direction == Direction.FLAT:
            return None
        frames = {
            timeframe: self._closed_window(symbol, timeframe, decision_time, 100)
            for timeframe in ("15m", "30m", "1h")
        }
        minimum = max(30, int(getattr(self.config, "minimum_history_bars", 30)))
        if len(frames["30m"]) < minimum or len(frames["1h"]) < minimum:
            return None

        side = 1.0 if direction == Direction.LONG else -1.0
        closes_15m = [candle.close for candle in frames["15m"]]
        closes_30m = [candle.close for candle in frames["30m"]]
        closes_1h = [candle.close for candle in frames["1h"]]
        atr_30m = atr(frames["30m"], 14)
        atr_1h = atr(frames["1h"], 14)
        atr30 = max(atr_30m[-1], 1e-12)
        atr1h = max(atr_1h[-1], 1e-12)
        ema9_30m = ema(closes_30m, 9)
        ema21_30m = ema(closes_30m, 21)
        ema21_1h = ema(closes_1h, 21)
        _, _, macd_hist_30m = macd(closes_30m)
        _, _, macd_hist_1h = macd(closes_1h)
        rsi_30m = rsi(closes_30m, 14)
        k_30m, d_30m, _ = kdj(frames["30m"], 9)

        slope_bars = max(1, int(getattr(self.config, "ema_slope_lookback_bars", 3)))
        slope_index = max(0, len(ema21_1h) - 1 - slope_bars)
        ema_slope_atr = side * (ema21_1h[-1] - ema21_1h[slope_index]) / atr1h / max(1, slope_bars)
        ema_alignment_atr = side * (ema9_30m[-1] - ema21_30m[-1]) / atr30
        macd_level_atr = side * macd_hist_1h[-1] / atr1h
        macd_delta_atr = side * (macd_hist_1h[-1] - macd_hist_1h[-3]) / atr1h
        reversal_macd_delta_atr = side * (macd_hist_30m[-1] - macd_hist_30m[-3]) / atr30
        atr_reference = statistics.mean(atr_30m[-21:-1]) if len(atr_30m) >= 21 else atr30
        atr_expansion_ratio = atr30 / max(atr_reference, 1e-12)
        price_extension_atr = side * (closes_30m[-1] - ema21_30m[-1]) / atr30

        btc_15m_return, btc_1h_return, btc_alignment_atr = self._btc_features(decision_time, side)
        breadth_return, breadth_ema = self._breadth(decision_time, direction)

        trend_components = {
            "trend_ema_slope": float(getattr(self.config, "trend_ema_slope_weight", 20.0))
            * _scale(ema_slope_atr, 0.0, float(getattr(self.config, "ema_slope_full_score_atr", 0.08))),
            "trend_ema_alignment": float(getattr(self.config, "trend_ema_alignment_weight", 20.0))
            * _scale(ema_alignment_atr, 0.0, float(getattr(self.config, "ema_alignment_full_score_atr", 0.50))),
            "trend_macd": float(getattr(self.config, "trend_macd_weight", 15.0))
            * statistics.mean(
                (
                    _scale(macd_level_atr, 0.0, float(getattr(self.config, "macd_level_full_score_atr", 0.12))),
                    _scale(macd_delta_atr, 0.0, float(getattr(self.config, "macd_delta_full_score_atr", 0.06))),
                )
            ),
            "trend_atr_expansion": float(getattr(self.config, "trend_atr_expansion_weight", 10.0))
            * self._atr_expansion_score(atr_expansion_ratio),
            "trend_btc_alignment": float(getattr(self.config, "trend_btc_weight", 20.0))
            * statistics.mean(
                (
                    _scale(side * btc_15m_return, -0.002, float(getattr(self.config, "btc_15m_full_score_return", 0.006))),
                    _scale(side * btc_1h_return, -0.004, float(getattr(self.config, "btc_1h_full_score_return", 0.015))),
                    _scale(btc_alignment_atr, 0.0, float(getattr(self.config, "btc_alignment_full_score_atr", 0.40))),
                )
            ),
            "trend_market_breadth": float(getattr(self.config, "trend_breadth_weight", 15.0))
            * statistics.mean(
                (
                    _scale(breadth_return, 0.50, float(getattr(self.config, "breadth_full_score", 0.68))),
                    _scale(breadth_ema, 0.50, float(getattr(self.config, "breadth_full_score", 0.68))),
                )
            ),
        }
        trend_score = sum(trend_components.values())
        max_extension = max(0.1, float(getattr(self.config, "trend_max_extension_atr", 3.0)))
        extension_penalty = _scale(price_extension_atr, max_extension, max_extension * 1.75)
        trend_score *= 1.0 - extension_penalty

        recovery_bars = max(2, int(getattr(self.config, "reversal_recovery_lookback_bars", 5)))
        rsi_window = rsi_30m[-recovery_bars:]
        k_window = k_30m[-recovery_bars:]
        if direction == Direction.LONG:
            rsi_extreme = min(rsi_window[:-1])
            kdj_extreme = min(k_window[:-1])
            rsi_recovering = rsi_window[-1] > rsi_window[-2]
            kdj_recovering = k_30m[-1] > k_30m[-2] and k_30m[-1] > d_30m[-1]
            no_new_extreme = frames["30m"][-1].low >= min(candle.low for candle in frames["30m"][-4:-1])
            rsi_extreme_score = _scale(
                float(getattr(self.config, "reversal_rsi_neutral", 50.0)) - rsi_extreme,
                float(getattr(self.config, "reversal_rsi_neutral", 50.0)) - float(getattr(self.config, "reversal_rsi_extreme", 35.0)),
                float(getattr(self.config, "reversal_rsi_neutral", 50.0)) - float(getattr(self.config, "reversal_rsi_full_score", 25.0)),
            )
            kdj_extreme_score = _scale(50.0 - kdj_extreme, 50.0 - float(getattr(self.config, "reversal_kdj_extreme", 30.0)), 35.0)
        else:
            rsi_extreme = max(rsi_window[:-1])
            kdj_extreme = max(k_window[:-1])
            rsi_recovering = rsi_window[-1] < rsi_window[-2]
            kdj_recovering = k_30m[-1] < k_30m[-2] and k_30m[-1] < d_30m[-1]
            no_new_extreme = frames["30m"][-1].high <= max(candle.high for candle in frames["30m"][-4:-1])
            rsi_extreme_score = _scale(
                rsi_extreme - (100.0 - float(getattr(self.config, "reversal_rsi_neutral", 50.0))),
                (100.0 - float(getattr(self.config, "reversal_rsi_extreme", 35.0))) - 50.0,
                (100.0 - float(getattr(self.config, "reversal_rsi_full_score", 25.0))) - 50.0,
            )
            kdj_extreme_score = _scale(kdj_extreme - 50.0, (100.0 - float(getattr(self.config, "reversal_kdj_extreme", 30.0))) - 50.0, 35.0)

        range_now = frames["30m"][-1].high - frames["30m"][-1].low
        previous_ranges = [candle.high - candle.low for candle in frames["30m"][-6:-1]]
        range_ratio = range_now / max(statistics.mean(previous_ranges), 1e-12)
        reversal_components = {
            "reversal_price_extension": float(getattr(self.config, "reversal_extension_weight", 25.0))
            * _scale(
                -price_extension_atr,
                float(getattr(self.config, "reversal_extension_min_atr", 0.50)),
                float(getattr(self.config, "reversal_extension_full_score_atr", 2.0)),
            ),
            "reversal_rsi_recovery": float(getattr(self.config, "reversal_rsi_weight", 15.0))
            * rsi_extreme_score
            * float(rsi_recovering),
            "reversal_kdj_recovery": float(getattr(self.config, "reversal_kdj_weight", 10.0))
            * kdj_extreme_score
            * float(kdj_recovering),
            "reversal_macd_turn": float(getattr(self.config, "reversal_macd_weight", 20.0))
            * _scale(reversal_macd_delta_atr, 0.0, float(getattr(self.config, "reversal_macd_delta_full_score_atr", 0.06))),
            "reversal_no_new_extreme": float(getattr(self.config, "reversal_no_extreme_weight", 15.0))
            * float(no_new_extreme),
            "reversal_btc_not_adverse": float(getattr(self.config, "reversal_btc_weight", 10.0))
            * _scale(
                side * btc_1h_return,
                -float(getattr(self.config, "btc_adverse_return_limit", 0.015)),
                0.0,
            ),
            "reversal_range_exhaustion": float(getattr(self.config, "reversal_exhaustion_weight", 5.0))
            * (1.0 - _scale(range_ratio, 0.70, 1.30)),
        }
        reversal_score = sum(reversal_components.values())
        trend_score = _clamp(trend_score, 0.0, 100.0)
        reversal_score = _clamp(reversal_score, 0.0, 100.0)
        gap = trend_score - reversal_score
        regime = self._shadow_regime(trend_score, reversal_score, gap)
        components = {**trend_components, **reversal_components, "trend_extension_penalty": extension_penalty * 100.0}
        features: dict[str, float | bool] = {
            "ema21_slope_1h_atr": ema_slope_atr,
            "ema9_ema21_alignment_30m_atr": ema_alignment_atr,
            "macd_histogram_1h_atr": macd_level_atr,
            "macd_histogram_delta_1h_atr": macd_delta_atr,
            "macd_histogram_delta_30m_atr": reversal_macd_delta_atr,
            "atr_expansion_30m_ratio": atr_expansion_ratio,
            "price_extension_ema21_30m_atr": price_extension_atr,
            "btc_return_15m": btc_15m_return,
            "btc_return_1h": btc_1h_return,
            "btc_alignment_atr": btc_alignment_atr,
            "directional_breadth_return": breadth_return,
            "directional_breadth_ema21": breadth_ema,
            "rsi_extreme": rsi_extreme,
            "rsi_recovering": rsi_recovering,
            "kdj_extreme": kdj_extreme,
            "kdj_recovering": kdj_recovering,
            "no_new_extreme": no_new_extreme,
            "range_contraction_ratio": range_ratio,
        }
        return RegimeScoreSnapshot(
            asof=decision_time.isoformat(),
            direction=direction.name,
            trend_score=trend_score,
            reversal_score=reversal_score,
            score_gap=gap,
            shadow_regime=regime,
            components=components,
            features=features,
        )

    def _closed_window(self, symbol: str, timeframe: str, decision_time: Any, limit: int) -> list[Candle]:
        candles = self.candles_by_timeframe.get(timeframe, {}).get(symbol, [])
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        if not candles or not timestamps:
            return []
        duration = timedelta(milliseconds=interval_to_milliseconds(timeframe))
        end = bisect.bisect_right(timestamps, decision_time - duration)
        return candles[max(0, end - limit):end]

    def _btc_features(self, decision_time: Any, side: float) -> tuple[float, float, float]:
        btc_15m = self._closed_window("BTCUSDT", "15m", decision_time, 10)
        btc_1h = self._closed_window("BTCUSDT", "1h", decision_time, 30)
        if len(btc_15m) < 2 or len(btc_1h) < 22:
            return 0.0, 0.0, 0.0
        return_15m = btc_15m[-1].close / max(btc_15m[-2].close, 1e-12) - 1.0
        return_1h = btc_1h[-1].close / max(btc_1h[-2].close, 1e-12) - 1.0
        closes = [candle.close for candle in btc_1h]
        atr_value = max(atr(btc_1h, 14)[-1], 1e-12)
        alignment = side * (ema(closes, 9)[-1] - ema(closes, 21)[-1]) / atr_value
        return return_15m, return_1h, alignment

    def _breadth(self, decision_time: Any, direction: Direction) -> tuple[float, float]:
        key = (decision_time, direction.value)
        cached = self.breadth_cache.get(key)
        if cached is not None:
            return cached
        positive_return = 0
        aligned_ema = 0
        total = 0
        for symbol in self.candles_by_timeframe.get("30m", {}):
            candles = self._closed_window(symbol, "30m", decision_time, 22)
            if len(candles) < 22:
                continue
            total += 1
            directional_return = direction.value * (candles[-1].close / max(candles[-2].close, 1e-12) - 1.0)
            if directional_return > 0:
                positive_return += 1
            closes = [candle.close for candle in candles]
            if direction.value * (closes[-1] - ema(closes, 21)[-1]) > 0:
                aligned_ema += 1
        result = (positive_return / max(total, 1), aligned_ema / max(total, 1))
        self.breadth_cache[key] = result
        return result

    def _atr_expansion_score(self, ratio: float) -> float:
        chaos = float(getattr(self.config, "atr_chaos_ratio", 2.0))
        if ratio >= chaos:
            return 0.0
        return _scale(
            ratio,
            float(getattr(self.config, "atr_expansion_min_ratio", 0.90)),
            float(getattr(self.config, "atr_expansion_full_score_ratio", 1.25)),
        )

    def _shadow_regime(self, trend_score: float, reversal_score: float, gap: float) -> str:
        minimum_gap = float(getattr(self.config, "minimum_score_gap", 10.0))
        if trend_score >= float(getattr(self.config, "trend_min_score", 60.0)) and gap >= minimum_gap:
            return "TREND"
        if reversal_score >= float(getattr(self.config, "reversal_min_score", 60.0)) and gap <= -minimum_gap:
            return "REVERSAL"
        return "NO_TRADE"


def snapshot_payload(snapshot: RegimeScoreSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {
            "trend_score": None,
            "reversal_score": None,
            "score_gap": None,
            "shadow_regime": "INSUFFICIENT_DATA",
        }
    return {
        "regime_score_asof": snapshot.asof,
        "regime_score_direction": snapshot.direction,
        "trend_score": snapshot.trend_score,
        "reversal_score": snapshot.reversal_score,
        "score_gap": snapshot.score_gap,
        "shadow_regime": snapshot.shadow_regime,
        "regime_score_components": snapshot.components,
        "regime_score_features": snapshot.features,
    }


def _scale(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    return _clamp((value - low) / (high - low), 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
