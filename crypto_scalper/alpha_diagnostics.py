from __future__ import annotations

import bisect
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from .indicators import atr, ema, kdj, macd, rsi
from .models import Candle, Direction, Signal
from .regime_score import snapshot_payload


_EVENT_TOKEN = re.compile(r"alpha_event_id=([^ ]+)")


@dataclass
class AlphaCandidateDiagnostics:
    enabled: bool = False
    full_round_trip_cost_pct: float = 0.0
    stop_round_trip_cost_pct: float = 0.0
    regime_score_engine: Any = None
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    reversal_keys: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def record_reversal(
        self,
        symbol: str,
        signal: Signal,
        candles: list[Candle],
        index: int,
        decision_time: Any = None,
    ) -> str | None:
        if not self.enabled or signal.direction == Direction.FLAT or index >= len(candles):
            return None
        candle = candles[index]
        key = (symbol, candle.timestamp.isoformat(), signal.direction.name)
        existing = self.reversal_keys.get(key)
        if existing:
            return existing
        event_id = f"reversal-{symbol}-{candle.timestamp.strftime('%Y%m%dT%H%M%S')}-{signal.direction.name.lower()}"
        window = candles[max(0, index - 119):index + 1]
        closes = [row.close for row in window]
        atr_values = atr(window, 14)
        rsi_values = rsi(closes, 14)
        k_values, d_values, _ = kdj(window, 9)
        _, _, histogram = macd(closes)
        ema9 = ema(closes, 9)
        ema21 = ema(closes, 21)
        atr_value = max(atr_values[-1], 1e-12)
        recent_lows = [row.low for row in window[-5:]]
        recent_highs = [row.high for row in window[-5:]]
        regime_snapshot = (
            self.regime_score_engine.score(symbol, decision_time or candle.timestamp, signal.direction)
            if self.regime_score_engine is not None
            else None
        )
        self.rows[event_id] = {
            "event_id": event_id,
            "strategy": "indicator_reversal",
            "timestamp": candle.timestamp.isoformat(),
            "symbol": symbol,
            "side": signal.direction.name,
            "raw_signal_reason": signal.reason,
            "status": "raw_signal",
            "filter_reason": None,
            "signal_price": candle.close,
            "stop_loss_pct": signal.stop_loss_pct,
            "take_profit_pct": signal.take_profit_pct,
            "target_to_full_cost_ratio": signal.take_profit_pct / max(self.full_round_trip_cost_pct, 1e-12),
            "stop_to_full_cost_ratio": signal.stop_loss_pct / max(self.stop_round_trip_cost_pct, 1e-12),
            "rsi14": rsi_values[-1],
            "kdj_k": k_values[-1],
            "kdj_d": d_values[-1],
            "macd_histogram": histogram[-1],
            "macd_histogram_change": histogram[-1] - histogram[-2],
            "ema9": ema9[-1],
            "ema21": ema21[-1],
            "price_extension_ema21_atr": (candle.close - ema21[-1]) / atr_value,
            "atr_pct": atr_value / max(candle.close, 1e-12),
            "reclaim_ema9": candle.close > ema9[-1],
            "reclaim_ema21": candle.close > ema21[-1],
            "no_new_low_3": recent_lows[-1] > min(recent_lows[:-1]),
            "no_new_high_3": recent_highs[-1] < max(recent_highs[:-1]),
            **snapshot_payload(regime_snapshot),
        }
        self.reversal_keys[key] = event_id
        return event_id

    def mark_reversal_accepted(
        self,
        symbol: str,
        timestamp: Any,
        direction: Direction,
        rank_score: float,
        filter_reason: str,
    ) -> None:
        if not self.enabled:
            return
        key = (symbol, timestamp.isoformat(), direction.name)
        event_id = self.reversal_keys.get(key)
        if not event_id:
            return
        self.rows[event_id].update(
            status="accepted_after_filters",
            rank_score=rank_score,
            filter_reason=filter_reason,
        )

    def record_vbp_breakout(
        self,
        symbol: str,
        candles: list[Candle],
        index: int,
        breakout_level: float,
        consolidation_bottom: float,
        breakout_volume_ratio: float,
        decision_time: Any = None,
        compression_metrics: dict[str, float] | None = None,
    ) -> str | None:
        if not self.enabled or index >= len(candles):
            return None
        candle = candles[index]
        event_id = f"vbp-{symbol}-{candle.timestamp.strftime('%Y%m%dT%H%M%S')}"
        if event_id in self.rows:
            return event_id
        window = candles[max(0, index - 119):index + 1]
        atr_values = atr(window, 14)
        atr_value = max(atr_values[-2] if len(atr_values) > 1 else atr_values[-1], 1e-12)
        candle_range = max(candle.high - candle.low, 1e-12)
        previous = window[:-1]
        recent_volume = statistics.mean(row.volume for row in previous[-12:]) if previous else 0.0
        older_volume = statistics.mean(row.volume for row in previous[-48:-12]) if len(previous) > 12 else recent_volume
        atr_history = atr_values[-80:-1]
        atr_percentile = sum(value <= atr_value for value in atr_history) / max(1, len(atr_history))
        failed_breakouts = sum(row.high > breakout_level and row.close <= breakout_level for row in previous[-48:])
        regime_snapshot = (
            self.regime_score_engine.score(symbol, decision_time or candle.timestamp, Direction.LONG)
            if self.regime_score_engine is not None
            else None
        )
        self.rows[event_id] = {
            "event_id": event_id,
            "strategy": "volume_breakout_pullback",
            "timestamp": candle.timestamp.isoformat(),
            "symbol": symbol,
            "side": "LONG",
            "status": "breakout_pending",
            "filter_reason": None,
            "breakout_index": index,
            "anchor_timestamp": candle.timestamp.isoformat(),
            "anchor_price": candle.close,
            "breakout_level": breakout_level,
            "vbp_bottom": consolidation_bottom,
            "breakout_atr": atr_value,
            "breakout_distance_atr": (candle.close - breakout_level) / atr_value,
            "breakout_body_atr": abs(candle.close - candle.open) / atr_value,
            "breakout_range_atr": candle_range / atr_value,
            "breakout_volume_ratio": breakout_volume_ratio,
            "breakout_close_position": (candle.close - candle.low) / candle_range,
            "upper_wick_ratio": (candle.high - max(candle.open, candle.close)) / candle_range,
            "lower_wick_ratio": (min(candle.open, candle.close) - candle.low) / candle_range,
            "pre_breakout_atr_percentile": atr_percentile,
            "pre_breakout_range_compression_atr": (breakout_level - consolidation_bottom) / atr_value,
            "pre_breakout_volume_contraction": recent_volume / max(older_volume, 1e-12),
            "previous_failed_breakout_count": failed_breakouts,
            "pullback_low": candle.close,
            "pullback_bars": 0,
            "pullback_broke_breakout_level": False,
            "pullback_broke_vbp_bottom": False,
            "full_round_trip_cost_pct": self.full_round_trip_cost_pct,
            "stop_round_trip_cost_pct": self.stop_round_trip_cost_pct,
            **snapshot_payload(regime_snapshot),
            **(compression_metrics or {}),
        }
        return event_id

    def update_vbp_pullback(
        self,
        event_id: str | None,
        candle: Candle,
        age: int,
        breakout_close: float,
        breakout_volume: float,
    ) -> None:
        if not self.enabled or not event_id or event_id not in self.rows:
            return
        row = self.rows[event_id]
        row["pullback_bars"] = age
        row["pullback_low"] = min(float(row.get("pullback_low", candle.low)), candle.low)
        atr_value = max(float(row["breakout_atr"]), 1e-12)
        depth = max(0.0, breakout_close - float(row["pullback_low"]))
        distance = max(breakout_close - float(row["breakout_level"]), 1e-12)
        row["pullback_depth_atr"] = depth / atr_value
        row["pullback_depth_to_breakout_distance"] = depth / distance
        row["pullback_volume_to_breakout_volume"] = candle.volume / max(breakout_volume, 1e-12)
        row["pullback_broke_breakout_level"] = bool(row["pullback_broke_breakout_level"] or candle.close < row["breakout_level"])
        row["pullback_broke_vbp_bottom"] = bool(row["pullback_broke_vbp_bottom"] or candle.low < row["vbp_bottom"])

    def mark_vbp(self, event_id: str | None, status: str, reason: str | None = None, **values: Any) -> None:
        if not self.enabled or not event_id or event_id not in self.rows:
            return
        self.rows[event_id].update(status=status, filter_reason=reason, **values)

    def finalize(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        trades: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        trade_map: dict[str, list[dict[str, Any]]] = {}
        reversal_trade_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for trade in trades:
            match = _EVENT_TOKEN.search(str(trade.get("entry_reason", "")))
            if match:
                trade_map.setdefault(match.group(1), []).append(trade)
            if trade.get("strategy_bucket") == "indicator_reversal":
                key = (str(trade.get("symbol")), str(trade.get("signal_time")), str(trade.get("direction", trade.get("side"))))
                reversal_trade_map.setdefault(key, []).append(trade)
        timestamp_cache = {symbol: [candle.timestamp for candle in candles] for symbol, candles in candles_by_symbol.items()}
        for row in self.rows.values():
            event_id = str(row["event_id"])
            matched = trade_map.get(event_id, [])
            if row["strategy"] == "indicator_reversal":
                matched = reversal_trade_map.get((row["symbol"], row["timestamp"], row["side"]), matched)
            if matched:
                row.update(
                    status="traded",
                    entry_time=min(str(trade["entry_time"]) for trade in matched),
                    exit_time=max(str(trade["exit_time"]) for trade in matched),
                    full_cost_net_pnl=sum(float(trade["net_pnl"]) for trade in matched),
                    gross_pnl=sum(float(trade["gross_pnl"]) for trade in matched),
                    fee=sum(float(trade.get("fee", 0.0)) for trade in matched),
                    slippage=sum(float(trade.get("slippage_cost", 0.0)) for trade in matched),
                    funding=sum(float(trade.get("funding", 0.0)) for trade in matched),
                    exit_reason=matched[-1].get("exit_reason"),
                )
                anchor_timestamp = row["entry_time"]
                anchor_price = float(matched[0]["raw_entry_price"])
            else:
                anchor_timestamp = str(row.get("anchor_timestamp", row["timestamp"]))
                anchor_price = float(row.get("anchor_price", row.get("signal_price", 0.0)))
            candles = candles_by_symbol.get(str(row["symbol"]), [])
            timestamps = timestamp_cache.get(str(row["symbol"]), [])
            if not candles or not timestamps or anchor_price <= 0:
                continue
            from datetime import datetime
            anchor = datetime.fromisoformat(anchor_timestamp)
            start = bisect.bisect_left(timestamps, anchor)
            direction = 1.0 if row["side"] == "LONG" else -1.0
            for minutes in (15, 30, 60, 120):
                forward = candles[start:min(len(candles), start + minutes + 1)]
                if direction > 0:
                    favorable = max((candle.high / anchor_price - 1.0 for candle in forward), default=0.0)
                    adverse = min((candle.low / anchor_price - 1.0 for candle in forward), default=0.0)
                else:
                    favorable = max((1.0 - candle.low / anchor_price for candle in forward), default=0.0)
                    adverse = min((1.0 - candle.high / anchor_price for candle in forward), default=0.0)
                row[f"mfe_{minutes}m_pct"] = favorable
                row[f"mae_{minutes}m_pct"] = adverse
        return sorted(self.rows.values(), key=lambda row: (row["timestamp"], row["symbol"], row["strategy"]))
