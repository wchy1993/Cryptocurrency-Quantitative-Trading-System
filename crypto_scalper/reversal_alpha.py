from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .data import interval_to_milliseconds
from .indicators import atr, ema, kdj, macd, rsi
from .models import Candle, Direction


@dataclass(frozen=True)
class ReversalAlphaSnapshot:
    setup_pass: bool
    trigger_pass: bool
    eligible: bool
    reject_reasons: tuple[str, ...]
    features: dict[str, float | bool]


class ReversalAlphaEngine:
    """Shadow evaluation for a closed-30m setup followed by a closed-5m trigger."""

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

    def evaluate(self, symbol: str, decision_time: Any, direction: Direction) -> ReversalAlphaSnapshot | None:
        if not self.enabled or direction == Direction.FLAT:
            return None
        setup = self._closed_window(symbol, "30m", decision_time, 100)
        trigger = self._closed_window(symbol, "5m", decision_time, 100)
        btc_15m = self._closed_window("BTCUSDT", "15m", decision_time, 4)
        btc_1h = self._closed_window("BTCUSDT", "1h", decision_time, 4)
        if len(setup) < 40 or len(trigger) < 30 or len(btc_15m) < 2 or len(btc_1h) < 2:
            return None

        side = 1.0 if direction == Direction.LONG else -1.0
        setup_closes = [candle.close for candle in setup]
        setup_atr = atr(setup, 14)
        setup_ema21 = ema(setup_closes, 21)
        setup_rsi = rsi(setup_closes, 14)
        setup_k, setup_d, _ = kdj(setup, 9)
        _, _, setup_hist = macd(setup_closes)
        lookback = max(3, int(getattr(self.config, "setup_lookback_bars", 5)))
        extension_values = [
            -side * (setup[index].close - setup_ema21[index]) / max(setup_atr[index], 1e-12)
            for index in range(len(setup) - lookback, len(setup))
        ]
        extension_atr = max(extension_values)
        if direction == Direction.LONG:
            rsi_extreme = min(setup_rsi[-lookback:])
            kdj_extreme = min(min(k, d) for k, d in zip(setup_k[-lookback:], setup_d[-lookback:]))
            rsi_recovering = setup_rsi[-1] > setup_rsi[-2]
            kdj_recovering = setup_k[-1] > setup_k[-2] and setup_k[-1] > setup_d[-1]
            no_new_extreme = setup[-1].low >= min(candle.low for candle in setup[-4:-1])
        else:
            rsi_extreme = max(setup_rsi[-lookback:])
            kdj_extreme = max(max(k, d) for k, d in zip(setup_k[-lookback:], setup_d[-lookback:]))
            rsi_recovering = setup_rsi[-1] < setup_rsi[-2]
            kdj_recovering = setup_k[-1] < setup_k[-2] and setup_k[-1] < setup_d[-1]
            no_new_extreme = setup[-1].high <= max(candle.high for candle in setup[-4:-1])
        macd_improvement_count = _directional_improvement_count(setup_hist, side)

        trigger_closes = [candle.close for candle in trigger]
        trigger_ema = ema(trigger_closes, max(2, int(getattr(self.config, "trigger_ema_period", 9))))
        _, _, trigger_hist = macd(trigger_closes)
        trigger_candle = trigger[-1]
        trigger_range = max(trigger_candle.high - trigger_candle.low, 1e-12)
        trigger_reclaim = side * (trigger_candle.close - trigger_ema[-1]) > 0
        trigger_directional = side * (trigger_candle.close - trigger_candle.open) > 0
        trigger_close_position = (
            (trigger_candle.close - trigger_candle.low) / trigger_range
            if direction == Direction.LONG
            else (trigger_candle.high - trigger_candle.close) / trigger_range
        )
        trigger_macd_improving = side * (trigger_hist[-1] - trigger_hist[-2]) > 0
        btc_return_15m = btc_15m[-1].close / max(btc_15m[-2].close, 1e-12) - 1.0
        btc_return_1h = btc_1h[-1].close / max(btc_1h[-2].close, 1e-12) - 1.0
        btc_not_adverse = (
            side * btc_return_15m >= -float(getattr(self.config, "btc_adverse_15m_return", 0.006))
            and side * btc_return_1h >= -float(getattr(self.config, "btc_adverse_1h_return", 0.012))
        )

        reasons = []
        if extension_atr < float(getattr(self.config, "setup_extension_atr_min", 0.5)):
            reasons.append("setup_extension")
        if direction == Direction.LONG:
            if rsi_extreme > float(getattr(self.config, "setup_long_rsi_extreme_max", 40.0)):
                reasons.append("setup_rsi_extreme")
            if kdj_extreme > float(getattr(self.config, "setup_long_kdj_extreme_max", 35.0)):
                reasons.append("setup_kdj_extreme")
        else:
            if rsi_extreme < float(getattr(self.config, "setup_short_rsi_extreme_min", 60.0)):
                reasons.append("setup_rsi_extreme")
            if kdj_extreme < float(getattr(self.config, "setup_short_kdj_extreme_min", 65.0)):
                reasons.append("setup_kdj_extreme")
        if bool(getattr(self.config, "setup_require_rsi_recovery", True)) and not rsi_recovering:
            reasons.append("setup_rsi_recovery")
        if bool(getattr(self.config, "setup_require_kdj_recovery", True)) and not kdj_recovering:
            reasons.append("setup_kdj_recovery")
        if macd_improvement_count < int(getattr(self.config, "setup_macd_improvement_bars", 2)):
            reasons.append("setup_macd_improvement")
        if bool(getattr(self.config, "setup_require_no_new_extreme", True)) and not no_new_extreme:
            reasons.append("setup_new_extreme")
        setup_reasons = list(reasons)
        if not trigger_reclaim:
            reasons.append("trigger_ema_reclaim")
        if bool(getattr(self.config, "trigger_require_directional_candle", True)) and not trigger_directional:
            reasons.append("trigger_directional_candle")
        if trigger_close_position < float(getattr(self.config, "trigger_close_position_min", 0.55)):
            reasons.append("trigger_close_position")
        if bool(getattr(self.config, "trigger_require_macd_improvement", True)) and not trigger_macd_improving:
            reasons.append("trigger_macd_improvement")
        if not btc_not_adverse:
            reasons.append("btc_adverse")
        setup_pass = not setup_reasons
        trigger_pass = not any(reason.startswith("trigger_") or reason == "btc_adverse" for reason in reasons)
        return ReversalAlphaSnapshot(
            setup_pass=setup_pass,
            trigger_pass=trigger_pass,
            eligible=setup_pass and trigger_pass,
            reject_reasons=tuple(reasons),
            features={
                "setup_extension_atr": extension_atr,
                "setup_rsi_extreme": rsi_extreme,
                "setup_rsi_recovering": rsi_recovering,
                "setup_kdj_extreme": kdj_extreme,
                "setup_kdj_recovering": kdj_recovering,
                "setup_macd_improvement_count": float(macd_improvement_count),
                "setup_no_new_extreme": no_new_extreme,
                "trigger_reclaim_ema": trigger_reclaim,
                "trigger_directional_candle": trigger_directional,
                "trigger_close_position": trigger_close_position,
                "trigger_macd_improving": trigger_macd_improving,
                "btc_return_15m": btc_return_15m,
                "btc_return_1h": btc_return_1h,
                "btc_not_adverse": btc_not_adverse,
            },
        )

    def _closed_window(self, symbol: str, timeframe: str, decision_time: Any, limit: int) -> list[Candle]:
        candles = self.candles_by_timeframe.get(timeframe, {}).get(symbol, [])
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        if not candles:
            return []
        duration = timedelta(milliseconds=interval_to_milliseconds(timeframe))
        end = bisect.bisect_right(timestamps, decision_time - duration)
        return candles[max(0, end - limit):end]


def snapshot_payload(snapshot: ReversalAlphaSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {
            "reversal_v2_setup_pass": None,
            "reversal_v2_trigger_pass": None,
            "reversal_v2_eligible": None,
        }
    return {
        "reversal_v2_setup_pass": snapshot.setup_pass,
        "reversal_v2_trigger_pass": snapshot.trigger_pass,
        "reversal_v2_eligible": snapshot.eligible,
        "reversal_v2_reject_reasons": list(snapshot.reject_reasons),
        "reversal_v2_features": snapshot.features,
    }


def _directional_improvement_count(values: list[float], side: float) -> int:
    count = 0
    for index in range(len(values) - 1, 0, -1):
        if side * (values[index] - values[index - 1]) <= 0:
            break
        count += 1
    return count
