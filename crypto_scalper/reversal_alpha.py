from __future__ import annotations

import bisect
from array import array
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Sequence

from .data import interval_to_milliseconds
from .indicators import atr, ema, kdj, macd, rsi
from .models import Candle, Direction, Signal


REVERSAL_V2_REASON_TOKEN = "indicator_reversal_v2"


@dataclass(frozen=True)
class ReversalAlphaSnapshot:
    setup_pass: bool
    trigger_pass: bool
    eligible: bool
    reject_reasons: tuple[str, ...]
    features: dict[str, float | bool]
    event_id: str = ""
    trigger_candle: Candle | None = None
    structural_stop_price: float = 0.0
    setup_atr: float = 0.0
    quality_score: float = 0.0


@dataclass(frozen=True)
class ReversalAlphaDecision:
    signal: Signal
    candle: Candle
    snapshot: ReversalAlphaSnapshot


@dataclass(frozen=True)
class _SetupIndicators:
    atr: Sequence[float]
    ema21: Sequence[float]
    rsi14: Sequence[float]
    k: Sequence[float]
    d: Sequence[float]
    macd_hist: Sequence[float]


@dataclass(frozen=True)
class _TriggerIndicators:
    atr: Sequence[float]
    ema: Sequence[float]
    macd_line: Sequence[float]
    macd_signal: Sequence[float]
    macd_hist: Sequence[float]


@dataclass(frozen=True)
class _MarketIndicators:
    atr: Sequence[float]
    ema21: Sequence[float]


class ReversalAlphaEngine:
    """Closed-30m exhaustion setup followed by a closed-5m reclaim trigger."""

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
        self._setup_indicators: dict[str, _SetupIndicators] = {}
        self._trigger_indicators: dict[str, _TriggerIndicators] = {}
        self._confirmation_indicators: dict[str, _TriggerIndicators] = {}
        self._btc_1h_indicators: _MarketIndicators | None = None
        if self.enabled:
            self._prepare_indicators()

    @property
    def execution_enabled(self) -> bool:
        return self.enabled and not bool(getattr(self.config, "shadow_mode", True))

    def allowed_directions(self) -> tuple[Direction, ...]:
        directions = []
        if bool(getattr(self.config, "allow_long", True)):
            directions.append(Direction.LONG)
        if bool(getattr(self.config, "allow_short", False)):
            directions.append(Direction.SHORT)
        return tuple(directions)

    def evaluate(self, symbol: str, decision_time: Any, direction: Direction) -> ReversalAlphaSnapshot | None:
        if not self.enabled or direction == Direction.FLAT:
            return None
        setup = self.candles_by_timeframe.get("30m", {}).get(symbol, [])
        trigger = self.candles_by_timeframe.get("5m", {}).get(symbol, [])
        btc_15m = self.candles_by_timeframe.get("15m", {}).get("BTCUSDT", [])
        btc_1h = self.candles_by_timeframe.get("1h", {}).get("BTCUSDT", [])
        setup_end = self._closed_end(symbol, "30m", decision_time)
        trigger_end = self._closed_end(symbol, "5m", decision_time)
        confirmation_15m_end = self._closed_end(symbol, "15m", decision_time)
        btc_15m_end = self._closed_end("BTCUSDT", "15m", decision_time)
        btc_1h_end = self._closed_end("BTCUSDT", "1h", decision_time)
        if setup_end < 40 or trigger_end < 30 or confirmation_15m_end < 30 or btc_15m_end < 2 or btc_1h_end < 4:
            return None

        setup_indicators = self._setup_indicators.get(symbol)
        trigger_indicators = self._trigger_indicators.get(symbol)
        confirmation_indicators = self._confirmation_indicators.get(symbol)
        if setup_indicators is None or trigger_indicators is None or confirmation_indicators is None:
            return None

        side = 1.0 if direction == Direction.LONG else -1.0
        setup_index = setup_end - 1
        trigger_index = trigger_end - 1
        setup_atr = max(float(setup_indicators.atr[setup_index]), 1e-12)
        trigger_atr = max(float(trigger_indicators.atr[trigger_index]), 1e-12)
        lookback = max(3, int(getattr(self.config, "setup_lookback_bars", 5)))
        setup_start = max(0, setup_end - lookback)
        extension_values = [
            -side * (setup[index].close - setup_indicators.ema21[index]) / setup_atr
            for index in range(setup_start, setup_end)
        ]
        extension_atr = max(extension_values)
        event_offset = max(range(len(extension_values)), key=extension_values.__getitem__)
        event_index = setup_start + event_offset

        setup_rsi = setup_indicators.rsi14
        setup_k = setup_indicators.k
        setup_d = setup_indicators.d
        if direction == Direction.LONG:
            rsi_extreme = min(setup_rsi[setup_start:setup_end])
            kdj_extreme = min(
                min(setup_k[index], setup_d[index])
                for index in range(setup_start, setup_end)
            )
            rsi_recovering = setup_rsi[setup_index] > setup_rsi[setup_index - 1]
            kdj_recovering = setup_k[setup_index] > setup_k[setup_index - 1] and setup_k[setup_index] > setup_d[setup_index]
            no_new_extreme = setup[setup_index].low >= min(candle.low for candle in setup[setup_index - 3:setup_index])
        else:
            rsi_extreme = max(setup_rsi[setup_start:setup_end])
            kdj_extreme = max(
                max(setup_k[index], setup_d[index])
                for index in range(setup_start, setup_end)
            )
            rsi_recovering = setup_rsi[setup_index] < setup_rsi[setup_index - 1]
            kdj_recovering = setup_k[setup_index] < setup_k[setup_index - 1] and setup_k[setup_index] < setup_d[setup_index]
            no_new_extreme = setup[setup_index].high <= max(candle.high for candle in setup[setup_index - 3:setup_index])
        macd_improvement_count = _directional_improvement_count_at(
            setup_indicators.macd_hist,
            setup_index,
            side,
        )

        slope_lookback = max(1, int(getattr(self.config, "setup_ema_slope_lookback_bars", 3)))
        slope_start = max(0, setup_index - slope_lookback)
        directional_ema_slope_atr = side * (
            setup_indicators.ema21[setup_index] - setup_indicators.ema21[slope_start]
        ) / setup_atr

        structure_lookback = max(2, int(getattr(self.config, "stop_structure_lookback_bars", 4)))
        structure_start = max(0, min(event_index, setup_end - structure_lookback))
        if direction == Direction.LONG:
            structure_extreme = min(candle.low for candle in setup[structure_start:setup_end])
        else:
            structure_extreme = max(candle.high for candle in setup[structure_start:setup_end])

        trigger_candle = trigger[trigger_index]
        trigger_range = max(trigger_candle.high - trigger_candle.low, 1e-12)
        trigger_reclaim_margin_atr = side * (
            trigger_candle.close - trigger_indicators.ema[trigger_index]
        ) / trigger_atr
        trigger_reclaim = trigger_reclaim_margin_atr > 0
        trigger_directional = side * (trigger_candle.close - trigger_candle.open) > 0
        trigger_close_position = (
            (trigger_candle.close - trigger_candle.low) / trigger_range
            if direction == Direction.LONG
            else (trigger_candle.high - trigger_candle.close) / trigger_range
        )
        trigger_macd_improving = side * (
            trigger_indicators.macd_hist[trigger_index] - trigger_indicators.macd_hist[trigger_index - 1]
        ) > 0
        cross_lookback = max(1, int(getattr(self.config, "trigger_macd_cross_lookback_bars", 3)))
        trigger_confirmed_cross = False
        for cross_index in range(max(1, trigger_index + 1 - cross_lookback), trigger_index + 1):
            if direction == Direction.LONG:
                crossed = (
                    trigger_indicators.macd_line[cross_index - 1] <= trigger_indicators.macd_signal[cross_index - 1]
                    and trigger_indicators.macd_line[cross_index] > trigger_indicators.macd_signal[cross_index]
                    and (
                        trigger_indicators.macd_line[cross_index] < 0
                        or trigger_indicators.macd_signal[cross_index] < 0
                    )
                )
            else:
                crossed = (
                    trigger_indicators.macd_line[cross_index - 1] >= trigger_indicators.macd_signal[cross_index - 1]
                    and trigger_indicators.macd_line[cross_index] < trigger_indicators.macd_signal[cross_index]
                    and (
                        trigger_indicators.macd_line[cross_index] > 0
                        or trigger_indicators.macd_signal[cross_index] > 0
                    )
                )
            if crossed:
                trigger_confirmed_cross = True
                break
        trigger_body_atr = abs(trigger_candle.close - trigger_candle.open) / trigger_atr
        volume_start = max(0, trigger_index - 20)
        prior_volumes = [candle.volume for candle in trigger[volume_start:trigger_index]]
        trigger_volume_ratio = trigger_candle.volume / max(sum(prior_volumes) / max(len(prior_volumes), 1), 1e-12)
        rebound_atr = side * (trigger_candle.close - structure_extreme) / setup_atr

        confirmation_15m = self.candles_by_timeframe.get("15m", {}).get(symbol, [])
        confirmation_index = confirmation_15m_end - 1
        confirmation_candle = confirmation_15m[confirmation_index]
        confirmation_reclaim = side * (
            confirmation_candle.close - confirmation_indicators.ema[confirmation_index]
        ) > 0
        confirmation_macd_improving = side * (
            confirmation_indicators.macd_hist[confirmation_index]
            - confirmation_indicators.macd_hist[confirmation_index - 1]
        ) > 0

        btc_return_15m = btc_15m[btc_15m_end - 1].close / max(btc_15m[btc_15m_end - 2].close, 1e-12) - 1.0
        btc_return_1h = btc_1h[btc_1h_end - 1].close / max(btc_1h[btc_1h_end - 2].close, 1e-12) - 1.0
        btc_not_adverse = (
            side * btc_return_15m >= -float(getattr(self.config, "btc_adverse_15m_return", 0.006))
            and side * btc_return_1h >= -float(getattr(self.config, "btc_adverse_1h_return", 0.012))
        )
        btc_1h_indicators = self._btc_1h_indicators
        if btc_1h_indicators is None:
            return None
        btc_1h_index = btc_1h_end - 1
        btc_1h_atr = max(float(btc_1h_indicators.atr[btc_1h_index]), 1e-12)
        btc_1h_ema_distance_atr = side * (
            btc_1h[btc_1h_index].close - btc_1h_indicators.ema21[btc_1h_index]
        ) / btc_1h_atr
        btc_slope_start = max(0, btc_1h_index - 3)
        btc_1h_ema_slope_atr = side * (
            btc_1h_indicators.ema21[btc_1h_index] - btc_1h_indicators.ema21[btc_slope_start]
        ) / btc_1h_atr

        stop_buffer = max(0.0, float(getattr(self.config, "stop_atr_buffer", 0.15))) * setup_atr
        structural_stop = structure_extreme - side * stop_buffer
        stop_distance = side * (trigger_candle.close - structural_stop)
        stop_distance_atr = stop_distance / setup_atr
        stop_pct = stop_distance / max(trigger_candle.close, 1e-12)

        reasons = []
        if extension_atr < float(getattr(self.config, "setup_extension_atr_min", 0.5)):
            reasons.append("setup_extension")
        if extension_atr > float(getattr(self.config, "setup_extension_atr_max", 999.0)):
            reasons.append("setup_extension_too_far")
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
        if directional_ema_slope_atr < -float(getattr(self.config, "setup_max_adverse_ema_slope_atr", 0.8)):
            reasons.append("setup_adverse_ema_slope")
        if rebound_atr > float(getattr(self.config, "setup_max_rebound_atr", 2.5)):
            reasons.append("setup_rebound_chase")
        setup_reasons = list(reasons)
        if not trigger_reclaim:
            reasons.append("trigger_ema_reclaim")
        if trigger_reclaim_margin_atr < float(getattr(self.config, "trigger_reclaim_margin_atr_min", 0.0)):
            reasons.append("trigger_reclaim_margin")
        if bool(getattr(self.config, "trigger_require_directional_candle", True)) and not trigger_directional:
            reasons.append("trigger_directional_candle")
        if trigger_close_position < float(getattr(self.config, "trigger_close_position_min", 0.55)):
            reasons.append("trigger_close_position")
        if bool(getattr(self.config, "trigger_require_macd_improvement", True)) and not trigger_macd_improving:
            reasons.append("trigger_macd_improvement")
        if bool(getattr(self.config, "trigger_require_confirmed_macd_cross", False)) and not trigger_confirmed_cross:
            reasons.append("trigger_macd_cross")
        if trigger_body_atr < float(getattr(self.config, "trigger_body_atr_min", 0.10)):
            reasons.append("trigger_body")
        if trigger_volume_ratio < float(getattr(self.config, "trigger_volume_ratio_min", 0.80)):
            reasons.append("trigger_volume")
        if bool(getattr(self.config, "confirmation_15m_require_reclaim", False)) and not confirmation_reclaim:
            reasons.append("confirmation_15m_reclaim")
        if bool(getattr(self.config, "confirmation_15m_require_macd_improvement", False)) and not confirmation_macd_improving:
            reasons.append("confirmation_15m_macd")
        if not btc_not_adverse:
            reasons.append("btc_adverse")
        if btc_1h_ema_distance_atr < float(getattr(self.config, "btc_1h_min_ema_distance_atr", -999.0)):
            reasons.append("btc_1h_ema_distance")
        if btc_1h_ema_slope_atr < float(getattr(self.config, "btc_1h_min_ema_slope_atr", -999.0)):
            reasons.append("btc_1h_ema_slope")
        if stop_distance <= 0:
            reasons.append("stop_invalid")
        if stop_distance_atr < float(getattr(self.config, "min_stop_atr", 0.5)):
            reasons.append("stop_too_tight")
        if stop_distance_atr > float(getattr(self.config, "max_stop_atr", 2.5)):
            reasons.append("stop_too_wide")
        if stop_pct < float(getattr(self.config, "min_stop_pct", 0.0025)):
            reasons.append("stop_pct_too_tight")
        if stop_pct > float(getattr(self.config, "max_stop_pct", 0.025)):
            reasons.append("stop_pct_too_wide")

        quality_score = _quality_score(
            direction,
            extension_atr,
            float(rsi_extreme),
            float(kdj_extreme),
            macd_improvement_count,
            no_new_extreme,
            trigger_close_position,
            trigger_body_atr,
            trigger_volume_ratio,
            btc_not_adverse,
            trigger_confirmed_cross,
        )
        if quality_score < float(getattr(self.config, "quality_score_min", 0.50)):
            reasons.append("quality_score")

        setup_pass = not setup_reasons
        trigger_pass = not any(
            reason.startswith("trigger_")
            or reason.startswith("confirmation_")
            or reason.startswith("stop_")
            or reason.startswith("btc_")
            or reason == "quality_score"
            for reason in reasons
        )
        event_time = setup[event_index].timestamp
        event_id = f"{symbol}-{direction.name.lower()}-{event_time:%Y%m%d%H%M}"
        return ReversalAlphaSnapshot(
            setup_pass=setup_pass,
            trigger_pass=trigger_pass,
            eligible=setup_pass and trigger_pass,
            reject_reasons=tuple(reasons),
            features={
                "setup_extension_atr": extension_atr,
                "setup_rsi_extreme": float(rsi_extreme),
                "setup_rsi_recovering": rsi_recovering,
                "setup_kdj_extreme": float(kdj_extreme),
                "setup_kdj_recovering": kdj_recovering,
                "setup_macd_improvement_count": float(macd_improvement_count),
                "setup_no_new_extreme": no_new_extreme,
                "setup_directional_ema_slope_atr": float(directional_ema_slope_atr),
                "setup_rebound_atr": float(rebound_atr),
                "trigger_reclaim_ema": trigger_reclaim,
                "trigger_reclaim_margin_atr": float(trigger_reclaim_margin_atr),
                "trigger_directional_candle": trigger_directional,
                "trigger_close_position": trigger_close_position,
                "trigger_macd_improving": trigger_macd_improving,
                "trigger_confirmed_macd_cross": trigger_confirmed_cross,
                "trigger_body_atr": trigger_body_atr,
                "trigger_volume_ratio": trigger_volume_ratio,
                "confirmation_15m_reclaim": confirmation_reclaim,
                "confirmation_15m_macd_improving": confirmation_macd_improving,
                "btc_return_15m": btc_return_15m,
                "btc_return_1h": btc_return_1h,
                "btc_not_adverse": btc_not_adverse,
                "btc_1h_ema_distance_atr": btc_1h_ema_distance_atr,
                "btc_1h_ema_slope_atr": btc_1h_ema_slope_atr,
                "structural_stop_price": structural_stop,
                "stop_distance_atr": stop_distance_atr,
                "stop_pct": stop_pct,
                "quality_score": quality_score,
            },
            event_id=event_id,
            trigger_candle=trigger_candle,
            structural_stop_price=structural_stop,
            setup_atr=setup_atr,
            quality_score=quality_score,
        )

    def decision_from_snapshot(
        self,
        snapshot: ReversalAlphaSnapshot | None,
        direction: Direction,
    ) -> ReversalAlphaDecision | None:
        if snapshot is None or not snapshot.eligible or snapshot.trigger_candle is None:
            return None
        if direction not in self.allowed_directions():
            return None
        stop_pct = float(snapshot.features.get("stop_pct", 0.0))
        take_profit_r = max(0.1, float(getattr(self.config, "take_profit_r", 1.5)))
        confidence = min(0.85, max(0.55, 0.55 + snapshot.quality_score * 0.30))
        side_name = "long" if direction == Direction.LONG else "short"
        signal = Signal(
            direction=direction,
            confidence=confidence,
            reason=(
                f"{REVERSAL_V2_REASON_TOKEN}_{side_name} "
                f"event_id={snapshot.event_id} trigger_tf=5m quality={snapshot.quality_score:.3f}"
            ),
            stop_loss_pct=stop_pct,
            take_profit_pct=stop_pct * take_profit_r,
            risk_multiplier=max(0.0, float(getattr(self.config, "risk_multiplier", 0.6))),
            max_holding_bars=0,
        )
        return ReversalAlphaDecision(signal, snapshot.trigger_candle, snapshot)

    def build_decision(self, symbol: str, decision_time: Any, direction: Direction) -> ReversalAlphaDecision | None:
        return self.decision_from_snapshot(self.evaluate(symbol, decision_time, direction), direction)

    def _closed_end(self, symbol: str, timeframe: str, decision_time: Any) -> int:
        timestamps = self.timestamps.get(timeframe, {}).get(symbol, [])
        duration = timedelta(milliseconds=interval_to_milliseconds(timeframe))
        return bisect.bisect_right(timestamps, decision_time - duration)

    def _closed_window(self, symbol: str, timeframe: str, decision_time: Any, limit: int) -> list[Candle]:
        candles = self.candles_by_timeframe.get(timeframe, {}).get(symbol, [])
        end = self._closed_end(symbol, timeframe, decision_time)
        return candles[max(0, end - limit):end]

    def _prepare_indicators(self) -> None:
        for symbol, candles in self.candles_by_timeframe.get("30m", {}).items():
            if not candles:
                continue
            closes = [candle.close for candle in candles]
            k_values, d_values, _ = kdj(candles, 9)
            _, _, histogram = macd(closes)
            self._setup_indicators[symbol] = _SetupIndicators(
                _pack(atr(candles, 14)),
                _pack(ema(closes, 21)),
                _pack(rsi(closes, 14)),
                _pack(k_values),
                _pack(d_values),
                _pack(histogram),
            )
        trigger_ema_period = max(2, int(getattr(self.config, "trigger_ema_period", 9)))
        for symbol, candles in self.candles_by_timeframe.get("5m", {}).items():
            if not candles:
                continue
            closes = [candle.close for candle in candles]
            macd_line, macd_signal, histogram = macd(closes)
            self._trigger_indicators[symbol] = _TriggerIndicators(
                _pack(atr(candles, 14)),
                _pack(ema(closes, trigger_ema_period)),
                _pack(macd_line),
                _pack(macd_signal),
                _pack(histogram),
            )
        confirmation_ema_period = max(2, int(getattr(self.config, "confirmation_15m_ema_period", 9)))
        for symbol, candles in self.candles_by_timeframe.get("15m", {}).items():
            if not candles:
                continue
            closes = [candle.close for candle in candles]
            macd_line, macd_signal, histogram = macd(closes)
            self._confirmation_indicators[symbol] = _TriggerIndicators(
                _pack(atr(candles, 14)),
                _pack(ema(closes, confirmation_ema_period)),
                _pack(macd_line),
                _pack(macd_signal),
                _pack(histogram),
            )
        btc_1h = self.candles_by_timeframe.get("1h", {}).get("BTCUSDT", [])
        if btc_1h:
            btc_closes = [candle.close for candle in btc_1h]
            self._btc_1h_indicators = _MarketIndicators(
                _pack(atr(btc_1h, 14)),
                _pack(ema(btc_closes, 21)),
            )


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
        "reversal_v2_event_id": snapshot.event_id,
        "reversal_v2_quality_score": snapshot.quality_score,
    }


def _pack(values: Sequence[float]) -> array:
    return array("d", (float(value) for value in values))


def _directional_improvement_count_at(values: Sequence[float], index: int, side: float) -> int:
    count = 0
    for current in range(index, 0, -1):
        if side * (values[current] - values[current - 1]) <= 0:
            break
        count += 1
    return count


def _directional_improvement_count(values: list[float], side: float) -> int:
    return _directional_improvement_count_at(values, len(values) - 1, side)


def _quality_score(
    direction: Direction,
    extension_atr: float,
    rsi_extreme: float,
    kdj_extreme: float,
    macd_improvement_count: int,
    no_new_extreme: bool,
    trigger_close_position: float,
    trigger_body_atr: float,
    trigger_volume_ratio: float,
    btc_not_adverse: bool,
    trigger_confirmed_cross: bool,
) -> float:
    if direction == Direction.LONG:
        rsi_score = (40.0 - rsi_extreme) / 20.0
        kdj_score = (35.0 - kdj_extreme) / 25.0
    else:
        rsi_score = (rsi_extreme - 60.0) / 20.0
        kdj_score = (kdj_extreme - 65.0) / 25.0
    score = (
        0.20 * _unit(extension_atr / 2.0)
        + 0.12 * _unit(rsi_score)
        + 0.08 * _unit(kdj_score)
        + 0.15 * _unit(macd_improvement_count / 3.0)
        + 0.10 * float(no_new_extreme)
        + 0.12 * _unit((trigger_close_position - 0.50) / 0.40)
        + 0.10 * _unit(trigger_body_atr / 1.0)
        + 0.05 * _unit(trigger_volume_ratio / 1.5)
        + 0.05 * float(btc_not_adverse)
        + 0.03 * float(trigger_confirmed_cross)
    )
    return _unit(score)


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
