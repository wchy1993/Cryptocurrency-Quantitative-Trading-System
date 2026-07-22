from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .risk import BacktestExecutionConfig, market_exit_fill


CMIPR_R_DEFINITION_VERSION = "cmipr_dual_r_v1"
CAMPAIGN_R_BASIS = "campaign"
INITIAL_LEG_R_BASIS = "initial_leg"
VALID_R_BASES = frozenset({CAMPAIGN_R_BASIS, INITIAL_LEG_R_BASIS})


@dataclass(frozen=True)
class CmiprInitialLegRisk:
    campaign_risk_budget_usdt: float
    initial_leg_price_risk_usdt: float
    initial_leg_full_cost_risk_usdt: float
    initial_leg_actual_risk_fraction: float
    capacity_clipped_initial_risk_fraction: float
    stop_execution_price_estimate: float
    estimated_stop_exit_fee_usdt: float
    estimated_stop_exit_slippage_usdt: float


def normalize_r_basis(value: Any) -> str:
    basis = str(value or CAMPAIGN_R_BASIS).strip().lower()
    if basis not in VALID_R_BASES:
        raise ValueError(f"unsupported CMIPR R basis: {basis}")
    return basis


def initial_leg_risk(
    position: Any,
    campaign_risk_budget_usdt: float,
    planned_initial_risk_fraction: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
) -> CmiprInitialLegRisk:
    quantity = max(0.0, float(position.quantity))
    raw_stop = float(position.initial_stop_price or position.stop_price)
    stop_fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        quantity,
        raw_stop,
        "stop_market",
        position.liquidity_reference_quote_volume or None,
    )
    side = position.direction.value
    price_risk = max(0.0, side * quantity * (float(position.entry_price) - stop_fill.price))
    full_cost_risk = price_risk + max(0.0, float(position.entry_fee)) + max(0.0, stop_fill.fee)
    campaign_budget = max(0.0, float(campaign_risk_budget_usdt))
    actual_fraction = full_cost_risk / campaign_budget if campaign_budget > 0.0 else 0.0
    capacity_fraction = (
        max(0.0, float(planned_initial_risk_fraction))
        * max(0.0, min(1.0, float(position.capacity_fill_ratio)))
    )
    return CmiprInitialLegRisk(
        campaign_risk_budget_usdt=campaign_budget,
        initial_leg_price_risk_usdt=price_risk,
        initial_leg_full_cost_risk_usdt=full_cost_risk,
        initial_leg_actual_risk_fraction=actual_fraction,
        capacity_clipped_initial_risk_fraction=capacity_fraction,
        stop_execution_price_estimate=stop_fill.price,
        estimated_stop_exit_fee_usdt=stop_fill.fee,
        estimated_stop_exit_slippage_usdt=stop_fill.slippage_cost,
    )


def executable_net_pnl(
    position: Any,
    raw_exit_price: float,
    execution_config: BacktestExecutionConfig,
    rules: Any,
    funding: float = 0.0,
    order_type: str = "market",
) -> float:
    fill = market_exit_fill(
        execution_config,
        rules,
        position.direction,
        position.quantity,
        raw_exit_price,
        order_type,
        position.liquidity_reference_quote_volume or None,
    )
    raw_entry = float(position.raw_entry_price or position.entry_price)
    gross = position.direction.value * position.quantity * (raw_exit_price - raw_entry)
    return (
        gross
        - float(position.entry_fee)
        - float(position.entry_slippage_cost)
        - fill.fee
        - fill.slippage_cost
        + float(funding)
    )


def risk_denominator(position: Any, basis: str) -> float:
    normalized = normalize_r_basis(basis)
    if normalized == INITIAL_LEG_R_BASIS:
        value = float(getattr(position, "initial_leg_full_cost_risk_usdt", 0.0) or 0.0)
    else:
        value = float(
            getattr(position, "campaign_risk_budget_usdt", 0.0)
            or getattr(position, "risk_budget_usdt", 0.0)
            or 0.0
        )
    return max(value, 1e-12)


def net_pnl_r(position: Any, net_pnl: float, basis: str) -> float:
    return float(net_pnl) / risk_denominator(position, basis)


def campaign_threshold_in_initial_leg_r(position: Any, campaign_threshold_r: float) -> float:
    fraction = float(getattr(position, "initial_leg_actual_risk_fraction", 0.0) or 0.0)
    return float(campaign_threshold_r) / max(fraction, 1e-12)
