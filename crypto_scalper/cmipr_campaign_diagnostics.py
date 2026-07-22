from __future__ import annotations

import bisect
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable

from .binance_client import SymbolRules
from .cmipr_r_basis import executable_net_pnl, initial_leg_risk
from .data import interval_to_milliseconds
from .models import Candle, Direction
from .risk import (
    BacktestExecutionConfig,
    capacity_limited_quantity,
    conservative_quantity,
    market_entry_fill,
    validate_order_size,
)


_EVENT_TOKEN = re.compile(r"(?:^| )event_id=([^ ]+)")
_HORIZONS = (15, 30, 60, 120, 240)
_HIT_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass
class CmiprCampaignDiagnostics:
    """Research-only CMIPR event diagnostics.

    Shadow paths never reserve cash, occupy a portfolio slot, mutate the CMIPR
    state machine, or suppress a later candidate.  They intentionally reuse the
    production execution/capacity helpers so their R values are full-cost values.
    """

    enabled: bool = False
    minimum_bucket_size: int = 30
    full_cost_pct: float = 0.0
    ignition_timeframe: str = "15m"
    pullback_timeframe: str = "5m"
    campaign_risk_budget_usdt: float = 0.80
    initial_risk_fraction: float = 0.40
    max_shadow_notional_usdt: float = float("inf")
    min_order_notional_usdt: float = 0.0
    fixed_take_profit_r: float = 1.50
    take_profit_r_basis: str = "campaign"
    max_holding_minutes: int = 2880
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_ignition(
        self,
        event_id: str,
        symbol: str,
        decision_time: datetime,
        regime: Any,
        previous_regime: Any,
        ignition: Any,
        compression: Any,
        previous_rank: tuple[float, float, float] | None,
        liquidity_percentile: float,
        compression_duration_bars: int,
    ) -> None:
        if not self.enabled:
            return
        if event_id in self.rows:
            row = self.rows[event_id]
            row["observation_count"] = int(row.get("observation_count", 1)) + 1
            row["last_observation_time"] = decision_time.isoformat()
            return
        previous_percentile = previous_rank[1] if previous_rank is not None else ignition.ranking_percentile
        signal_available = ignition.candle.timestamp + timedelta(
            milliseconds=interval_to_milliseconds(self.ignition_timeframe)
        )
        breadth_acceleration = regime.breadth_above_ema21 - previous_regime.breadth_above_ema21
        volatility_pct = compression.atr_value / max(ignition.candle.close, 1e-12)
        self.rows[event_id] = {
            "event_id": event_id,
            "campaign_id": event_id,
            "event_confirmation_id": f"{event_id}:ignition",
            "event_type": "IGNITION",
            "symbol": symbol,
            "side": ignition.direction.name,
            "ignition_time": ignition.candle.timestamp.isoformat(),
            "decision_time": decision_time.isoformat(),
            "last_observation_time": decision_time.isoformat(),
            "observation_count": 1,
            "signal_available_time": signal_available.isoformat(),
            "status": "ignition_observed",
            "regime": regime.state.value,
            "raw_regime": regime.raw_state.value,
            "btc_state": _direction_state(regime.btc_1h_return),
            "eth_state": _direction_state(regime.eth_1h_return),
            "btc_1h_return": regime.btc_1h_return,
            "btc_4h_return": regime.btc_4h_return,
            "eth_1h_return": regime.eth_1h_return,
            "btc_eth_alignment": regime.btc_1h_return * regime.eth_1h_return >= 0.0,
            "breadth_above_ema21": regime.breadth_above_ema21,
            "breadth_positive_1h": regime.breadth_positive_1h,
            "breadth_acceleration": breadth_acceleration,
            "ranking_percentile": ignition.ranking_percentile,
            "quality_score": float(getattr(ignition, "quality_score", _breakout_quality(ignition))),
            "relative_strength_score": ignition.ranking_score,
            "strength_acceleration_1h": ignition.ranking_percentile - previous_percentile,
            "liquidity_percentile": liquidity_percentile,
            "compression_quality": _compression_quality(compression),
            "compression_atr_percentile": compression.atr_percentile,
            "compression_atr_to_average": compression.atr_to_average,
            "channel_width_atr": compression.channel_width_atr,
            "compression_volume_contraction": compression.volume_contraction,
            "prior_move_atr": compression.prior_move_atr,
            "failed_breakout_count": compression.failed_breakouts,
            "compression_duration_bars": compression_duration_bars,
            "breakout_quality": _breakout_quality(ignition),
            "breakout_level": ignition.breakout_level,
            "breakout_atr": ignition.breakout_distance_atr,
            "breakout_body_atr": ignition.body_atr,
            "breakout_close_position": ignition.close_position,
            "breakout_wick_ratio": ignition.wick_ratio,
            "volume_ratio": ignition.volume_ratio,
            "ignition_atr": float(getattr(ignition, "setup_atr_value", compression.atr_value)),
            "ignition_price": ignition.candle.close,
            "initial_stop_price": ignition.stop_price,
            "initial_stop_pct": ignition.stop_loss_pct,
            "point_in_time_universe_eligibility": True,
            "execution_capacity_eligibility": None,
            "volatility_pct": volatility_pct,
            "volatility_bucket": _volatility_bucket(volatility_pct),
            "liquidity_bucket": _percentile_bucket(liquidity_percentile),
            "first_pullback_time": None,
            "pullback_confirmation_time": None,
            "entry_mode": None,
            "cost_model": "production_full_cost_shadow_execution",
        }

    def update_pullback(
        self,
        event_id: str | None,
        candle: Candle,
        depth_atr: float,
        volume_ratio: float,
        close_position: float,
        chase_atr: float,
        confirmed: bool,
        *,
        decision_time: datetime | None = None,
        pullback_bars: int | None = None,
        pullback_low: float | None = None,
        confirmation_body_atr: float | None = None,
        confirmation_wick_ratio: float | None = None,
        stop_price: float | None = None,
        stop_distance_atr: float | None = None,
        target_to_cost_ratio: float | None = None,
        entry_regime: str | None = None,
        entry_breadth: float | None = None,
        entry_rank: float | None = None,
        entry_quality: float | None = None,
        entry_extension: float | None = None,
    ) -> None:
        if not self.enabled or not event_id or event_id not in self.rows:
            return
        row = self.rows[event_id]
        quality = _pullback_quality(depth_atr, volume_ratio, close_position, confirmed)
        if row.get("first_pullback_time") is None:
            row.update(
                first_pullback_time=candle.timestamp.isoformat(),
                first_pullback_quality=quality,
                first_pullback_depth_atr=depth_atr,
                first_pullback_volume_ratio=volume_ratio,
                first_pullback_close_position=close_position,
            )
        row.update(
            latest_pullback_time=candle.timestamp.isoformat(),
            latest_pullback_depth_atr=depth_atr,
            latest_pullback_volume_ratio=volume_ratio,
            latest_pullback_close_position=close_position,
            latest_pullback_confirmed=confirmed,
            chase_distance_atr=chase_atr,
        )
        if not confirmed:
            return
        available = candle.timestamp + timedelta(milliseconds=interval_to_milliseconds(self.pullback_timeframe))
        ignition_available = datetime.fromisoformat(str(row["signal_available_time"]))
        row.update(
            event_confirmation_id=f"{event_id}:first_pullback:{candle.timestamp.isoformat()}",
            event_type="IMMEDIATE_FIRST_PULLBACK",
            pullback_confirmation_time=candle.timestamp.isoformat(),
            pullback_signal_available_time=available.isoformat(),
            ignition_to_pullback_minutes=max(0.0, (available - ignition_available).total_seconds() / 60.0),
            pullback_depth_atr=depth_atr,
            pullback_volume_ratio=volume_ratio,
            pullback_bars=pullback_bars,
            pullback_low=pullback_low,
            pullback_close_position=close_position,
            confirmation_body_atr=confirmation_body_atr,
            confirmation_wick_ratio=confirmation_wick_ratio,
            confirmation_quality=quality,
            chase_distance_atr=chase_atr,
            pullback_stop_price=stop_price,
            pullback_stop_distance_atr=stop_distance_atr,
            pullback_stop_loss_pct=(
                abs(candle.close - stop_price) / max(candle.close, 1e-12)
                if stop_price is not None
                else row.get("initial_stop_pct")
            ),
            target_to_cost_ratio=target_to_cost_ratio,
            entry_time_regime=entry_regime,
            entry_time_breadth=entry_breadth,
            entry_time_rank=entry_rank,
            entry_time_quality=entry_quality,
            entry_time_extension=entry_extension,
            pullback_decision_time=(decision_time or available).isoformat(),
        )

    def mark(self, event_id: str | None, status: str, **values: Any) -> None:
        if not self.enabled or not event_id or event_id not in self.rows:
            return
        self.rows[event_id].update(status=status, **values)

    def finalize(
        self,
        execution_candles_by_symbol: dict[str, list[Candle]],
        mtf_candles_by_timeframe: dict[str, dict[str, list[Candle]]],
        trades: list[dict[str, Any]],
        execution_config: BacktestExecutionConfig | None = None,
        rules_by_symbol: dict[str, SymbolRules] | None = None,
    ) -> dict[str, Any]:
        del mtf_candles_by_timeframe  # Diagnostics are intentionally 1m/path based in Stage 2.5.
        if not self.enabled:
            return {}
        execution_config = execution_config or BacktestExecutionConfig()
        rules_by_symbol = rules_by_symbol or {}
        trade_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in trades:
            event_id = _trade_event_id(trade)
            if event_id:
                trade_map[event_id].append(trade)

        for event_id, row in self.rows.items():
            matched = trade_map.get(event_id, [])
            if matched:
                first = min(matched, key=lambda item: str(item.get("entry_time", "")))
                last = max(matched, key=lambda item: str(item.get("exit_time", "")))
                entry_time = datetime.fromisoformat(str(first["entry_time"]))
                ignition_available = datetime.fromisoformat(str(row["signal_available_time"]))
                row.update(
                    status="traded",
                    fill_count=len(matched),
                    actual_entry_time=entry_time.isoformat(),
                    actual_exit_time=str(last.get("exit_time")),
                    ignition_to_entry_minutes=(entry_time - ignition_available).total_seconds() / 60.0,
                    actual_net_pnl=sum(float(item.get("net_pnl", 0.0) or 0.0) for item in matched),
                    actual_exit_reason=str(last.get("exit_reason", "")),
                    actual_fail_fast=str(last.get("exit_reason", "")).startswith("cmipr_fail_fast"),
                    actual_max_executable_initial_leg_r=max(
                        float(item.get("max_executable_initial_leg_r", 0.0) or 0.0) for item in matched
                    ),
                    actual_max_executable_campaign_r=max(
                        float(item.get("max_executable_campaign_r", 0.0) or 0.0) for item in matched
                    ),
                    actual_net_initial_leg_r=sum(float(item.get("net_initial_leg_r", 0.0) or 0.0) for item in matched),
                    actual_net_campaign_r=sum(float(item.get("net_campaign_r", 0.0) or 0.0) for item in matched),
                )
            else:
                row["fill_count"] = 0

            symbol = str(row["symbol"])
            candles = execution_candles_by_symbol.get(symbol, [])
            rules = rules_by_symbol.get(symbol) or _fallback_rules(symbol)
            _simulate_shadow(
                row,
                candles,
                execution_config,
                rules,
                prefix="ignition",
                available_time_key="signal_available_time",
                stop_pct_key="initial_stop_pct",
                campaign_risk_budget_usdt=self.campaign_risk_budget_usdt,
                initial_risk_fraction=self.initial_risk_fraction,
                max_notional_usdt=self.max_shadow_notional_usdt,
                min_order_notional_usdt=self.min_order_notional_usdt,
            )
            if row.get("pullback_signal_available_time") and row.get("status") in {"entry_ready", "traded"}:
                _simulate_shadow(
                    row,
                    candles,
                    execution_config,
                    rules,
                    prefix="pullback",
                    available_time_key="pullback_signal_available_time",
                    stop_pct_key="pullback_stop_loss_pct",
                    campaign_risk_budget_usdt=self.campaign_risk_budget_usdt,
                    initial_risk_fraction=self.initial_risk_fraction,
                    max_notional_usdt=self.max_shadow_notional_usdt,
                    min_order_notional_usdt=self.min_order_notional_usdt,
                )
                _copy_primary_pullback_aliases(row)

        rows = sorted(self.rows.values(), key=lambda item: (item["ignition_time"], item["symbol"]))
        pullback_rows = [row for row in rows if row.get("pullback_shadow_eligible")]
        fail_fast = _fail_fast_counterfactual(
            trades,
            execution_candles_by_symbol,
            execution_config,
            rules_by_symbol,
            self.fixed_take_profit_r,
            self.take_profit_r_basis,
            self.max_holding_minutes,
            self.rows,
        )
        score_report = _score_decile_report(rows, pullback_rows, self.minimum_bucket_size)
        return {
            "research_only": True,
            "does_not_affect_execution": True,
            "shadow_portfolio_isolation": "no_cash_no_slot_no_state_mutation",
            "minimum_bucket_size": self.minimum_bucket_size,
            "ignition_observation_count": sum(int(row.get("observation_count", 1)) for row in rows),
            "unique_ignition_event_count": len(rows),
            "raw_pullback_confirmation_event_count": sum(bool(row.get("pullback_confirmation_time")) for row in rows),
            "pullback_entry_ready_event_count": sum(row.get("status") in {"entry_ready", "traded"} for row in rows),
            "shadow_executable_pullback_count": len(pullback_rows),
            "ignition_events": rows,
            "ignition_summary": _event_summary(rows, "ignition"),
            "pullback_summary": _event_summary(pullback_rows, "pullback"),
            "shadow_target_study": _shadow_target_study(pullback_rows),
            "signal_age_report": _signal_age_report(pullback_rows),
            "score_decile_analysis": score_report,
            "quality_score_not_predictive": score_report["quality_score"]["predictive_status"] != "predictive",
            "rank_pct_not_predictive": score_report["ranking_percentile"]["predictive_status"] != "predictive",
            "bucket_report": _bucket_report(rows, self.minimum_bucket_size),
            "fail_fast_counterfactual": fail_fast,
            "campaign_baseline": _campaign_baseline(trades),
        }


def _simulate_shadow(
    row: dict[str, Any],
    candles: list[Candle],
    execution_config: BacktestExecutionConfig,
    rules: SymbolRules,
    *,
    prefix: str,
    available_time_key: str,
    stop_pct_key: str,
    campaign_risk_budget_usdt: float,
    initial_risk_fraction: float,
    max_notional_usdt: float,
    min_order_notional_usdt: float,
) -> None:
    eligible_key = f"{prefix}_shadow_eligible"
    row[eligible_key] = False
    if not candles or not row.get(available_time_key):
        row[f"{prefix}_shadow_reject_reason"] = "missing_candles_or_signal_time"
        return
    timestamps = [item.timestamp for item in candles]
    available = datetime.fromisoformat(str(row[available_time_key]))
    start = bisect.bisect_right(timestamps, available)
    if start >= len(candles):
        row[f"{prefix}_shadow_reject_reason"] = "no_next_1m_open"
        return
    entry_candle = candles[start]
    direction = Direction[str(row["side"])]
    stop_pct = float(row.get(stop_pct_key, 0.0) or 0.0)
    raw_entry = float(entry_candle.open)
    planned_price_risk = max(0.0, campaign_risk_budget_usdt * initial_risk_fraction)
    if stop_pct <= 0.0 or raw_entry <= 0.0 or planned_price_risk <= 0.0:
        row[f"{prefix}_shadow_reject_reason"] = "invalid_shadow_risk"
        return
    requested_notional = planned_price_risk / stop_pct
    requested_notional = min(requested_notional, max_notional_usdt)
    requested_quantity = conservative_quantity(rules, requested_notional / raw_entry)
    quote_volume = abs(entry_candle.volume * raw_entry)
    quantity, fill_ratio, capacity_reason = capacity_limited_quantity(
        execution_config,
        rules,
        requested_quantity,
        raw_entry,
        quote_volume,
    )
    if quantity <= 0.0:
        row[f"{prefix}_shadow_reject_reason"] = capacity_reason
        return
    entry_fill = market_entry_fill(
        execution_config,
        rules,
        direction,
        quantity,
        raw_entry,
        quote_volume,
    )
    size_reason = validate_order_size(rules, quantity, entry_fill.price, min_order_notional_usdt)
    if size_reason != "ok":
        row[f"{prefix}_shadow_reject_reason"] = size_reason
        return
    stop_price = entry_fill.price * (1.0 - direction.value * stop_pct)
    position = SimpleNamespace(
        symbol=str(row["symbol"]),
        direction=direction,
        quantity=quantity,
        entry_price=entry_fill.price,
        raw_entry_price=entry_fill.raw_price,
        entry_fee=entry_fill.fee,
        entry_slippage_cost=entry_fill.slippage_cost,
        initial_stop_price=stop_price,
        stop_price=stop_price,
        liquidity_reference_quote_volume=quote_volume,
        capacity_fill_ratio=fill_ratio,
    )
    risk = initial_leg_risk(
        position,
        campaign_risk_budget_usdt,
        initial_risk_fraction,
        execution_config,
        rules,
    )
    if risk.initial_leg_full_cost_risk_usdt <= 0.0:
        row[f"{prefix}_shadow_reject_reason"] = "zero_full_cost_risk"
        return
    position.initial_leg_full_cost_risk_usdt = risk.initial_leg_full_cost_risk_usdt
    position.campaign_risk_budget_usdt = risk.campaign_risk_budget_usdt
    row.update({
        eligible_key: True,
        f"{prefix}_shadow_reject_reason": "",
        f"{prefix}_shadow_entry_time": entry_candle.timestamp.isoformat(),
        f"{prefix}_shadow_raw_entry_price": entry_fill.raw_price,
        f"{prefix}_shadow_entry_price": entry_fill.price,
        f"{prefix}_shadow_quantity": quantity,
        f"{prefix}_shadow_capacity_fill_ratio": fill_ratio,
        f"{prefix}_shadow_stop_price": stop_price,
        f"{prefix}_shadow_stop_execution_price": risk.stop_execution_price_estimate,
        f"{prefix}_shadow_campaign_risk_usdt": risk.campaign_risk_budget_usdt,
        f"{prefix}_shadow_initial_leg_risk_usdt": risk.initial_leg_full_cost_risk_usdt,
        f"{prefix}_shadow_actual_risk_fraction": risk.initial_leg_actual_risk_fraction,
    })
    if prefix == "pullback":
        row["execution_capacity_eligibility"] = True
        row["shadow_capacity_fill_ratio"] = fill_ratio
        row["shadow_initial_leg_actual_risk_fraction"] = risk.initial_leg_actual_risk_fraction
        row["ignition_to_shadow_entry_minutes"] = (entry_candle.timestamp - available).total_seconds() / 60.0 + float(row.get("ignition_to_pullback_minutes", 0.0) or 0.0)

    end_time = entry_candle.timestamp + timedelta(minutes=max(_HORIZONS))
    end = bisect.bisect_right(timestamps, end_time)
    path = candles[start:end]
    max_initial_r = float("-inf")
    min_initial_r = float("inf")
    max_campaign_r = float("-inf")
    min_campaign_r = float("inf")
    best_time: datetime | None = None
    first_hits: dict[float, datetime] = {}
    stop_time: datetime | None = None
    observations: list[tuple[datetime, float, float]] = []
    raw_mfe = float("-inf")
    raw_mae = float("inf")
    terminal_net: float | None = None
    terminal_gross: float | None = None
    terminal_time: datetime | None = None
    terminal_reason = "shadow_240m_time_exit"

    for candle in path:
        adverse_raw = candle.low if direction == Direction.LONG else candle.high
        favorable_raw = candle.high if direction == Direction.LONG else candle.low
        raw_mfe = max(raw_mfe, direction.value * quantity * (favorable_raw - raw_entry))
        raw_mae = min(raw_mae, direction.value * quantity * (adverse_raw - raw_entry))
        stop_hit = candle.low <= stop_price if direction == Direction.LONG else candle.high >= stop_price
        if stop_hit:
            stop_net = executable_net_pnl(position, stop_price, execution_config, rules, order_type="stop_market")
            stop_initial_r = stop_net / risk.initial_leg_full_cost_risk_usdt
            stop_campaign_r = stop_net / max(risk.campaign_risk_budget_usdt, 1e-12)
            observations.append((candle.timestamp, stop_initial_r, stop_campaign_r))
            min_initial_r = min(min_initial_r, stop_initial_r)
            min_campaign_r = min(min_campaign_r, stop_campaign_r)
            stop_time = candle.timestamp
            if terminal_net is None:
                terminal_net = stop_net
                terminal_gross = direction.value * quantity * (stop_price - raw_entry)
                terminal_time = candle.timestamp
                terminal_reason = "shadow_original_stop"
            break

        current_net = executable_net_pnl(position, candle.close, execution_config, rules)
        current_initial_r = current_net / risk.initial_leg_full_cost_risk_usdt
        current_campaign_r = current_net / max(risk.campaign_risk_budget_usdt, 1e-12)
        observations.append((candle.timestamp, current_initial_r, current_campaign_r))
        if current_initial_r > max_initial_r:
            max_initial_r = current_initial_r
            best_time = candle.timestamp
        min_initial_r = min(min_initial_r, current_initial_r)
        max_campaign_r = max(max_campaign_r, current_campaign_r)
        min_campaign_r = min(min_campaign_r, current_campaign_r)
        for threshold in _HIT_LEVELS:
            if threshold not in first_hits and current_initial_r >= threshold:
                first_hits[threshold] = candle.timestamp
        if terminal_net is None and current_initial_r >= 1.5:
            terminal_net = current_net
            terminal_gross = direction.value * quantity * (candle.close - raw_entry)
            terminal_time = candle.timestamp
            terminal_reason = "shadow_1p5_initial_leg_target"

    if terminal_net is None:
        terminal_candle = path[-1]
        terminal_net = executable_net_pnl(position, terminal_candle.close, execution_config, rules)
        terminal_gross = direction.value * quantity * (terminal_candle.close - raw_entry)
        terminal_time = terminal_candle.timestamp
    if max_initial_r == float("-inf") and observations:
        max_initial_r = max(item[1] for item in observations)
        max_campaign_r = max(item[2] for item in observations)
    for threshold, hit_time in first_hits.items():
        token = str(threshold).replace(".", "p")
        row[f"{prefix}_first_hit_{token}_initial_leg_r_time"] = hit_time.isoformat()
    row.update({
        f"{prefix}_first_hit_stop_time": stop_time.isoformat() if stop_time else None,
        f"{prefix}_stop_first": stop_time is not None and not first_hits,
        f"{prefix}_maximum_reachable_initial_leg_r": _finite_or_none(max_initial_r),
        f"{prefix}_maximum_reachable_campaign_r": _finite_or_none(max_campaign_r),
        f"{prefix}_minimum_reachable_initial_leg_r": _finite_or_none(min_initial_r),
        f"{prefix}_minimum_reachable_campaign_r": _finite_or_none(min_campaign_r),
        f"{prefix}_raw_mfe_usdt": _finite_or_none(raw_mfe),
        f"{prefix}_raw_mae_usdt": _finite_or_none(raw_mae),
        f"{prefix}_time_to_maximum_mfe_minutes": (
            (best_time - entry_candle.timestamp).total_seconds() / 60.0 if best_time else None
        ),
        f"{prefix}_shadow_terminal_time": terminal_time.isoformat() if terminal_time else None,
        f"{prefix}_shadow_terminal_reason": terminal_reason,
        f"{prefix}_shadow_terminal_net_pnl": terminal_net,
        f"{prefix}_shadow_terminal_initial_leg_r": terminal_net / risk.initial_leg_full_cost_risk_usdt,
        f"{prefix}_shadow_terminal_campaign_r": terminal_net / max(risk.campaign_risk_budget_usdt, 1e-12),
        f"{prefix}_shadow_round_trip_cost_usdt": max(0.0, float(terminal_gross or 0.0) - terminal_net),
        f"{prefix}_shadow_gross_pnl_usdt": terminal_gross,
    })
    for minutes in _HORIZONS:
        cutoff = entry_candle.timestamp + timedelta(minutes=minutes)
        subset = [item for item in observations if item[0] <= cutoff]
        if not subset:
            continue
        row[f"{prefix}_future_mfe_{minutes}m_initial_leg_r"] = max(item[1] for item in subset)
        row[f"{prefix}_future_mae_{minutes}m_initial_leg_r"] = min(item[1] for item in subset)
        row[f"{prefix}_future_mfe_{minutes}m_campaign_r"] = max(item[2] for item in subset)
        row[f"{prefix}_future_mae_{minutes}m_campaign_r"] = min(item[2] for item in subset)


def _copy_primary_pullback_aliases(row: dict[str, Any]) -> None:
    for minutes in _HORIZONS:
        row[f"future_mfe_{minutes}m_r"] = row.get(f"pullback_future_mfe_{minutes}m_initial_leg_r")
        row[f"future_mae_{minutes}m_r"] = row.get(f"pullback_future_mae_{minutes}m_initial_leg_r")
    row["maximum_reachable_r_240m"] = row.get("pullback_maximum_reachable_initial_leg_r")
    row["maximum_reachable_initial_leg_r"] = row.get("pullback_maximum_reachable_initial_leg_r")
    row["maximum_reachable_campaign_r"] = row.get("pullback_maximum_reachable_campaign_r")


def _fail_fast_counterfactual(
    trades: list[dict[str, Any]],
    candles_by_symbol: dict[str, list[Candle]],
    execution_config: BacktestExecutionConfig,
    rules_by_symbol: dict[str, SymbolRules],
    fixed_take_profit_r: float,
    take_profit_r_basis: str,
    max_holding_minutes: int,
    event_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    timestamp_cache = {symbol: [candle.timestamp for candle in candles] for symbol, candles in candles_by_symbol.items()}
    for trade in trades:
        reason = str(trade.get("exit_reason", ""))
        if not reason.startswith("cmipr_fail_fast"):
            continue
        symbol = str(trade["symbol"])
        candles = candles_by_symbol.get(symbol, [])
        timestamps = timestamp_cache.get(symbol, [])
        if not candles:
            continue
        exit_time = datetime.fromisoformat(str(trade["exit_time"]))
        entry_time = datetime.fromisoformat(str(trade["entry_time"]))
        start = bisect.bisect_right(timestamps, exit_time)
        end_time = entry_time + timedelta(minutes=max(1, max_holding_minutes))
        end = bisect.bisect_right(timestamps, end_time)
        if start >= min(end, len(candles)):
            continue
        direction = Direction[str(trade.get("direction", trade.get("side", "LONG")))]
        rules = rules_by_symbol.get(symbol) or _fallback_rules(symbol)
        position = SimpleNamespace(
            symbol=symbol,
            direction=direction,
            quantity=float(trade.get("quantity", trade.get("qty", 0.0)) or 0.0),
            entry_price=float(trade.get("entry_price", 0.0) or 0.0),
            raw_entry_price=float(trade.get("raw_entry_price", trade.get("entry_price", 0.0)) or 0.0),
            entry_fee=float(trade.get("entry_fee", 0.0) or 0.0),
            entry_slippage_cost=abs(
                float(trade.get("entry_price", 0.0) or 0.0)
                - float(trade.get("raw_entry_price", trade.get("entry_price", 0.0)) or 0.0)
            ) * float(trade.get("quantity", trade.get("qty", 0.0)) or 0.0),
            initial_stop_price=float(trade.get("initial_stop_price", 0.0) or 0.0),
            stop_price=float(trade.get("initial_stop_price", 0.0) or 0.0),
            liquidity_reference_quote_volume=(
                abs(candles[max(0, bisect.bisect_left(timestamps, entry_time))].volume * float(trade.get("raw_entry_price", 0.0) or 0.0))
            ),
            initial_leg_full_cost_risk_usdt=float(trade.get("initial_leg_full_cost_risk_usdt", 0.0) or 0.0),
            campaign_risk_budget_usdt=float(trade.get("campaign_risk_budget_usdt", trade.get("risk_budget_usdt", 0.0)) or 0.0),
        )
        if position.quantity <= 0.0 or position.initial_leg_full_cost_risk_usdt <= 0.0:
            continue
        event_id = _trade_event_id(trade)
        event = event_rows.get(event_id or "", {})
        actual_net = float(trade.get("net_pnl", 0.0) or 0.0)
        actual_initial_r = actual_net / position.initial_leg_full_cost_risk_usdt
        actual_campaign_r = actual_net / max(position.campaign_risk_budget_usdt, 1e-12)
        path = candles[start:end]
        max_post_initial_r = float("-inf")
        min_post_initial_r = float("inf")
        reached = {threshold: False for threshold in (0.5, 1.0, 1.5, 2.0)}
        stop_time: datetime | None = None
        tp_time: datetime | None = None
        terminal_net: float | None = None
        terminal_reason = "original_time_stop"
        horizon_values: dict[int, list[float]] = defaultdict(list)
        for candle in path:
            stop_hit = candle.low <= position.stop_price if direction == Direction.LONG else candle.high >= position.stop_price
            if stop_hit:
                terminal_net = executable_net_pnl(position, position.stop_price, execution_config, rules, order_type="stop_market")
                stop_time = candle.timestamp
                terminal_reason = "original_stop"
                for minutes in (15, 30, 60, 120):
                    if candle.timestamp <= exit_time + timedelta(minutes=minutes):
                        horizon_values[minutes].append(terminal_net / position.initial_leg_full_cost_risk_usdt)
                break
            net = executable_net_pnl(position, candle.close, execution_config, rules)
            initial_r = net / position.initial_leg_full_cost_risk_usdt
            campaign_r = net / max(position.campaign_risk_budget_usdt, 1e-12)
            max_post_initial_r = max(max_post_initial_r, initial_r)
            min_post_initial_r = min(min_post_initial_r, initial_r)
            for threshold in reached:
                reached[threshold] = reached[threshold] or initial_r >= threshold
            for minutes in (15, 30, 60, 120):
                if candle.timestamp <= exit_time + timedelta(minutes=minutes):
                    horizon_values[minutes].append(initial_r)
            basis_value = initial_r if take_profit_r_basis == "initial_leg" else campaign_r
            if basis_value >= fixed_take_profit_r:
                terminal_net = net
                tp_time = candle.timestamp
                terminal_reason = "original_tp"
                break
        if terminal_net is None:
            terminal_candle = path[-1]
            terminal_net = executable_net_pnl(position, terminal_candle.close, execution_config, rules)
        terminal_initial_r = terminal_net / position.initial_leg_full_cost_risk_usdt
        improvement_r = terminal_initial_r - actual_initial_r
        if reached[1.0] or tp_time is not None or improvement_r >= 0.25:
            classification = "PREMATURE_EXIT"
        elif stop_time is not None or (max_post_initial_r < 0.25 and improvement_r <= 0.10):
            classification = "CORRECT_EARLY_EXIT"
        else:
            classification = "AMBIGUOUS"
        row = {
            "event_id": event_id,
            "symbol": symbol,
            "actual_fail_fast_exit_time": exit_time.isoformat(),
            "exit_reason": reason,
            "actual_fail_fast_net_pnl": actual_net,
            "actual_fail_fast_initial_leg_r": actual_initial_r,
            "actual_fail_fast_campaign_r": actual_campaign_r,
            "original_stop": position.stop_price,
            "original_tp_r": fixed_take_profit_r,
            "original_tp_r_basis": take_profit_r_basis,
            "later_reached_0p5_initial_leg_r": reached[0.5],
            "later_reached_1p0_initial_leg_r": reached[1.0],
            "later_reached_1p5_initial_leg_r": reached[1.5],
            "later_reached_2p0_initial_leg_r": reached[2.0],
            "later_hit_original_stop": stop_time is not None,
            "later_hit_original_tp": tp_time is not None,
            "original_stop_time": stop_time.isoformat() if stop_time else None,
            "original_tp_time": tp_time.isoformat() if tp_time else None,
            "post_exit_maximum_initial_leg_r": _finite_or_none(max_post_initial_r),
            "post_exit_minimum_initial_leg_r": _finite_or_none(min_post_initial_r),
            "continue_hold_terminal_net_pnl": terminal_net,
            "continue_hold_terminal_initial_leg_r": terminal_initial_r,
            "continue_hold_terminal_reason": terminal_reason,
            "classification": classification,
            "saved_loss_usdt": max(0.0, actual_net - terminal_net),
            "missed_profit_usdt": max(0.0, terminal_net - actual_net),
            "signal_age_minutes": event.get("ignition_to_entry_minutes"),
            "regime": event.get("regime"),
            "ranking_percentile": event.get("ranking_percentile"),
            "pullback_quality": event.get("confirmation_quality"),
        }
        for minutes in (15, 30, 60, 120):
            values = horizon_values.get(minutes, [])
            row[f"post_exit_mfe_{minutes}m_initial_leg_r"] = max(values) if values else None
            row[f"post_exit_mae_{minutes}m_initial_leg_r"] = min(values) if values else None
        rows.append(row)
    classifications = defaultdict(list)
    for row in rows:
        classifications[str(row["classification"])].append(row)
    return {
        "count": len(rows),
        "correct_early_exit_count": len(classifications["CORRECT_EARLY_EXIT"]),
        "premature_exit_count": len(classifications["PREMATURE_EXIT"]),
        "ambiguous_count": len(classifications["AMBIGUOUS"]),
        "false_early_exit_rate": len(classifications["PREMATURE_EXIT"]) / len(rows) if rows else 0.0,
        "later_reached_0p5r_count": sum(bool(row["later_reached_0p5_initial_leg_r"]) for row in rows),
        "later_reached_1r_count": sum(bool(row["later_reached_1p0_initial_leg_r"]) for row in rows),
        "later_reached_2r_count": sum(bool(row["later_reached_2p0_initial_leg_r"]) for row in rows),
        "saved_loss_usdt": sum(float(row["saved_loss_usdt"]) for row in rows),
        "missed_profit_usdt": sum(float(row["missed_profit_usdt"]) for row in rows),
        "post_exit_recovery_distribution": _distribution([
            row.get("post_exit_maximum_initial_leg_r") for row in rows
        ]),
        "by_fail_fast_reason": _group_counterfactual(rows, "exit_reason"),
        "by_signal_age": _group_counterfactual(rows, "signal_age_minutes", _signal_age_bucket),
        "by_regime": _group_counterfactual(rows, "regime"),
        "by_rank": _group_counterfactual(rows, "ranking_percentile", _percentile_bucket),
        "by_pullback_quality": _group_counterfactual(rows, "pullback_quality", _optional_quality_bucket),
        "rows": rows,
    }


def _signal_age_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ages = [row.get("ignition_to_pullback_minutes") for row in rows]
    fill_ages = [row.get("ignition_to_entry_minutes") for row in rows if row.get("fill_count")]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_signal_age_bucket(row.get("ignition_to_pullback_minutes"))].append(row)
    return {
        "candidate_age_distribution_minutes": _distribution(candidate_ages, include_p95=True),
        "fill_age_distribution_minutes": _distribution(fill_ages, include_p95=True),
        "buckets": {key: _shadow_bucket_metrics(group, "pullback") for key, group in grouped.items()},
    }


def _score_decile_report(
    ignition_rows: list[dict[str, Any]],
    pullback_rows: list[dict[str, Any]],
    minimum: int,
) -> dict[str, Any]:
    definitions = {
        "ranking_percentile": (ignition_rows, "ranking_percentile", "ignition"),
        "quality_score": (ignition_rows, "quality_score", "ignition"),
        "breakout_atr": (ignition_rows, "breakout_atr", "ignition"),
        "volume_ratio": (ignition_rows, "volume_ratio", "ignition"),
        "signal_age": (pullback_rows, "ignition_to_pullback_minutes", "pullback"),
        "pullback_depth": (pullback_rows, "pullback_depth_atr", "pullback"),
        "pullback_volume_ratio": (pullback_rows, "pullback_volume_ratio", "pullback"),
        "chase_distance": (pullback_rows, "chase_distance_atr", "pullback"),
        "target_to_cost": (pullback_rows, "target_to_cost_ratio", "pullback"),
        "breadth": (ignition_rows, "breadth_above_ema21", "ignition"),
        "breadth_acceleration": (ignition_rows, "breadth_acceleration", "ignition"),
    }
    return {
        name: _one_decile_report(rows, field, prefix, minimum)
        for name, (rows, field, prefix) in definitions.items()
    }


def _one_decile_report(
    rows: list[dict[str, Any]],
    field: str,
    prefix: str,
    minimum: int,
) -> dict[str, Any]:
    valid = [
        row for row in rows
        if _finite(row.get(field)) and row.get(f"{prefix}_shadow_eligible")
    ]
    valid.sort(key=lambda row: (float(row[field]), str(row.get("event_id", ""))))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(valid):
        decile = min(9, int(index * 10 / max(len(valid), 1))) + 1
        buckets[f"D{decile}"].append(row)
    metrics = {name: _shadow_bucket_metrics(group, prefix) for name, group in sorted(buckets.items())}
    for value in metrics.values():
        value["sample_status"] = "eligible" if value["count"] >= minimum else "insufficient_sample"
        value["eligible_for_parameter_selection"] = value["count"] >= minimum
    averages = [float(metrics[name]["average_initial_leg_r"]) for name in sorted(metrics)]
    monotonic = all(averages[index] >= averages[index - 1] for index in range(1, len(averages))) if len(averages) > 1 else False
    x = [float(row[field]) for row in valid]
    y = [float(row.get(f"{prefix}_shadow_terminal_initial_leg_r", 0.0) or 0.0) for row in valid]
    correlation = _spearman(x, y)
    enough = len(valid) >= minimum * 3
    predictive = enough and monotonic and correlation > 0.10
    return {
        "field": field,
        "evaluated_count": len(valid),
        "minimum_bucket_size": minimum,
        "deciles": metrics,
        "spearman_rank_correlation": correlation,
        "expectancy_monotonic_non_decreasing": monotonic,
        "predictive_status": "predictive" if predictive else ("not_predictive" if enough else "insufficient_sample"),
        "eligible_as_hard_filter": predictive,
    }


def _shadow_bucket_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    terminal = [float(row.get(f"{prefix}_shadow_terminal_initial_leg_r", 0.0) or 0.0) for row in rows]
    campaign = [float(row.get(f"{prefix}_shadow_terminal_campaign_r", 0.0) or 0.0) for row in rows]
    wins = [value for value in terminal if value > 0.0]
    losses = [value for value in terminal if value <= 0.0]
    gross = sum(max(0.0, float(row.get(f"{prefix}_shadow_gross_pnl_usdt", 0.0) or 0.0)) for row in rows)
    cost = sum(float(row.get(f"{prefix}_shadow_round_trip_cost_usdt", 0.0) or 0.0) for row in rows)
    return {
        "count": len(rows),
        "fill_count": sum(int(row.get("fill_count", 0) or 0) > 0 for row in rows),
        "initial_leg_expectancy": statistics.mean(terminal) if terminal else 0.0,
        "campaign_expectancy": statistics.mean(campaign) if campaign else 0.0,
        "average_initial_leg_r": statistics.mean(terminal) if terminal else 0.0,
        "median_initial_leg_r": statistics.median(terminal) if terminal else 0.0,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0.0),
        "0p5r_hit_rate": _hit_rate(rows, prefix, 0.5),
        "1r_hit_rate": _hit_rate(rows, prefix, 1.0),
        "2r_hit_rate": _hit_rate(rows, prefix, 2.0),
        "3r_hit_rate": _hit_rate(rows, prefix, 3.0),
        "stop_first_rate": sum(bool(row.get(f"{prefix}_stop_first")) for row in rows) / len(rows) if rows else 0.0,
        "fail_fast_rate": sum(bool(row.get("actual_fail_fast")) for row in rows) / len(rows) if rows else 0.0,
        "cost_to_gross_ratio": cost / gross if gross > 0.0 else None,
    }


def _event_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get(f"{prefix}_shadow_eligible")]
    values = [float(row.get(f"{prefix}_maximum_reachable_initial_leg_r", 0.0) or 0.0) for row in evaluated]
    return {
        "event_count": len(rows),
        "path_evaluated_count": len(evaluated),
        "reached_0p5r_count": sum(value >= 0.5 for value in values),
        "reached_1r_count": sum(value >= 1.0 for value in values),
        "reached_2r_count": sum(value >= 2.0 for value in values),
        "reached_3r_count": sum(value >= 3.0 for value in values),
        "average_maximum_reachable_initial_leg_r": statistics.mean(values) if values else 0.0,
        "median_maximum_reachable_initial_leg_r": statistics.median(values) if values else 0.0,
    }


def _shadow_target_study(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("pullback_shadow_eligible")]
    initial_targets = (1.0, 1.5, 2.0, 2.5, 3.0)
    return {
        "research_only_no_portfolio_exit_change": True,
        "evaluated_count": len(evaluated),
        "initial_leg_targets": {
            str(target): {
                "hit_count": sum(
                    float(row.get("pullback_maximum_reachable_initial_leg_r", float("-inf")) or float("-inf")) >= target
                    for row in evaluated
                ),
                "hit_rate": _hit_rate(evaluated, "pullback", target),
            }
            for target in initial_targets
        },
        "current_1p5_campaign_r": {
            "hit_count": sum(
                float(row.get("pullback_maximum_reachable_campaign_r", float("-inf")) or float("-inf")) >= 1.5
                for row in evaluated
            ),
            "hit_rate": (
                sum(
                    float(row.get("pullback_maximum_reachable_campaign_r", float("-inf")) or float("-inf")) >= 1.5
                    for row in evaluated
                ) / len(evaluated)
                if evaluated else 0.0
            ),
        },
    }


def _campaign_baseline(trades: list[dict[str, Any]]) -> dict[str, Any]:
    campaigns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        event_id = _trade_event_id(trade)
        if event_id:
            campaigns[event_id].append(trade)
    rows = []
    for campaign_id, grouped in campaigns.items():
        campaign_risk = max(float(item.get("campaign_risk_budget_usdt", item.get("risk_budget_usdt", 0.0)) or 0.0) for item in grouped)
        initial_risk = sum(float(item.get("initial_leg_full_cost_risk_usdt", 0.0) or 0.0) for item in grouped)
        net = sum(float(item.get("net_pnl", 0.0) or 0.0) for item in grouped)
        rows.append({
            "campaign_id": campaign_id,
            "symbol": str(grouped[0].get("symbol", "")),
            "month": str(grouped[0].get("entry_time", ""))[:7],
            "net_pnl": net,
            "net_campaign_r": net / max(campaign_risk, 1e-12),
            "net_initial_leg_r": net / max(initial_risk, 1e-12),
        })
    campaign_values = [float(row["net_campaign_r"]) for row in rows]
    initial_values = [float(row["net_initial_leg_r"]) for row in rows]
    wins = [value for value in campaign_values if value > 0.0]
    losses = [value for value in campaign_values if value <= 0.0]
    return {
        "campaign_count": len(rows),
        "campaign_expectancy_r": statistics.mean(campaign_values) if campaign_values else 0.0,
        "initial_leg_expectancy_r": statistics.mean(initial_values) if initial_values else 0.0,
        "campaign_profit_factor": sum(wins) / abs(sum(losses)) if losses else (float("inf") if wins else 0.0),
        "rows": rows,
    }


def _bucket_report(rows: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    dimensions: dict[str, Callable[[dict[str, Any]], str]] = {
        "regime": lambda row: str(row.get("regime")),
        "breadth_acceleration": lambda row: _signed_bucket(float(row.get("breadth_acceleration", 0.0) or 0.0)),
        "btc_eth_alignment": lambda row: "aligned" if row.get("btc_eth_alignment") else "conflict",
        "rank": lambda row: _percentile_bucket(float(row.get("ranking_percentile", 0.0) or 0.0)),
        "compression_quality": lambda row: _quality_bucket(float(row.get("compression_quality", 0.0) or 0.0)),
        "signal_age": lambda row: _signal_age_bucket(row.get("ignition_to_pullback_minutes")),
        "liquidity": lambda row: str(row.get("liquidity_bucket")),
        "volatility": lambda row: str(row.get("volatility_bucket")),
    }
    output: dict[str, Any] = {}
    for dimension, resolver in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[resolver(row)].append(row)
        output[dimension] = {
            "eligible": {key: _event_summary(group, "ignition") for key, group in grouped.items() if len(group) >= minimum},
            "excluded_small_sample": {key: len(group) for key, group in grouped.items() if len(group) < minimum},
        }
    return output


def _group_counterfactual(
    rows: list[dict[str, Any]],
    field: str,
    resolver: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        key = resolver(value) if resolver else str(value)
        grouped[key].append(row)
    return {
        key: {
            "count": len(group),
            "correct": sum(row["classification"] == "CORRECT_EARLY_EXIT" for row in group),
            "premature": sum(row["classification"] == "PREMATURE_EXIT" for row in group),
            "ambiguous": sum(row["classification"] == "AMBIGUOUS" for row in group),
            "missed_profit_usdt": sum(float(row["missed_profit_usdt"]) for row in group),
            "saved_loss_usdt": sum(float(row["saved_loss_usdt"]) for row in group),
        }
        for key, group in grouped.items()
    }


def _distribution(values: list[Any], include_p95: bool = False) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values if _finite(value))
    fields = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    names = ("min", "p10", "p25", "median", "p75", "p90", "max")
    result: dict[str, Any] = {"count": len(ordered)}
    for name, fraction in zip(names, fields):
        result[name] = _percentile(ordered, fraction) if ordered else None
    if include_p95:
        result["p95"] = _percentile(ordered, 0.95) if ordered else None
    return result


def _percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_rank = _ranks(left)
    right_rank = _ranks(right)
    left_mean = statistics.mean(left_rank)
    right_mean = statistics.mean(right_rank)
    covariance = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_rank)
        * sum((y - right_mean) ** 2 for y in right_rank)
    )
    return covariance / denominator if denominator > 0.0 else 0.0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for offset in range(index, end):
            ranks[order[offset]] = rank
        index = end
    return ranks


def _hit_rate(rows: list[dict[str, Any]], prefix: str, threshold: float) -> float:
    if not rows:
        return 0.0
    return sum(
        float(row.get(f"{prefix}_maximum_reachable_initial_leg_r", float("-inf")) or float("-inf")) >= threshold
        for row in rows
    ) / len(rows)


def _trade_event_id(trade: dict[str, Any]) -> str | None:
    match = _EVENT_TOKEN.search(str(trade.get("entry_reason", "")))
    return match.group(1) if match else None


def _compression_quality(compression: Any) -> float:
    values = (
        1.0 - min(1.0, max(0.0, compression.atr_percentile)),
        1.0 - min(1.0, max(0.0, compression.atr_to_average)),
        1.0 - min(1.0, max(0.0, compression.volume_contraction)),
        1.0 - min(1.0, max(0.0, compression.failed_breakouts / 4.0)),
    )
    return sum(values) / len(values)


def _breakout_quality(ignition: Any) -> float:
    return max(0.0, min(1.0, (
        min(1.0, ignition.breakout_distance_atr / 0.75)
        + min(1.0, ignition.body_atr)
        + ignition.close_position
        + min(1.0, ignition.volume_ratio / 3.0)
        + (1.0 - ignition.wick_ratio)
    ) / 5.0))


def _pullback_quality(depth: float, volume: float, close_position: float, confirmed: bool) -> float:
    depth_score = max(0.0, 1.0 - abs(depth - 0.5))
    volume_score = max(0.0, 1.0 - volume)
    return max(0.0, min(1.0, (depth_score + volume_score + close_position + float(confirmed)) / 4.0))


def _fallback_rules(symbol: str) -> SymbolRules:
    return SymbolRules(symbol, "0.000001", "0.000001", "0.000001", "0")


def _direction_state(value: float) -> str:
    if value > 0.001:
        return "UP"
    if value < -0.001:
        return "DOWN"
    return "FLAT"


def _signal_age_bucket(value: Any) -> str:
    if not _finite(value):
        return "missing"
    minutes = float(value)
    if minutes <= 30:
        return "le_30m"
    if minutes <= 45:
        return "31_45m"
    if minutes <= 60:
        return "46_60m"
    if minutes <= 90:
        return "61_90m"
    return "gt_90m"


def _percentile_bucket(value: Any) -> str:
    if not _finite(value):
        return "missing"
    number = float(value)
    if number >= 0.95:
        return "top_5pct"
    if number >= 0.90:
        return "top_10pct"
    if number >= 0.75:
        return "top_25pct"
    return "lower"


def _signed_bucket(value: float) -> str:
    if value >= 0.05:
        return "strong_rising"
    if value >= 0.01:
        return "rising"
    if value <= -0.01:
        return "falling"
    return "flat"


def _quality_bucket(value: float) -> str:
    if value >= 0.70:
        return "high"
    if value >= 0.50:
        return "medium"
    return "low"


def _optional_quality_bucket(value: Any) -> str:
    return "missing" if not _finite(value) else _quality_bucket(float(value))


def _volatility_bucket(value: float) -> str:
    if value < 0.005:
        return "low"
    if value < 0.015:
        return "medium"
    return "high"


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
