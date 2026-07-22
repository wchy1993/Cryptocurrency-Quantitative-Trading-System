from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.live_config import load_live_config
from crypto_scalper.gui import (
    STRATEGY_MODE_DUAL_THRUST_SHADOW,
    _config_with_strategy_mode,
    _detect_strategy_mode,
)
from crypto_scalper.models import Candle, Direction
from crypto_scalper.volatility_breakout_shadow import (
    DualThrustShadowTrader,
    dual_thrust_shadow_config_hash,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = ROOT / "config.volatility-breakout.v2-balanced-50-shadow.json"
RESEARCH_CONFIG = ROOT / "config.volatility-breakout.hybrid-v5-balanced-expansion-runner-50.json"


class FakeShadowClient:
    def __init__(self) -> None:
        self.rules = SymbolRules("BTCUSDT", "0.001", "0.001", "0.01", "5")

    def ping(self) -> dict:
        return {}

    def symbol_rules(self, _symbol: str) -> SymbolRules:
        return self.rules

    def funding_rate_history(self, *_args, **_kwargs) -> list[dict]:
        return []

    def klines(self, *_args, **_kwargs) -> list[Candle]:
        return []

    def new_market_order(self, *_args, **_kwargs) -> None:
        raise AssertionError("shadow trader must never call an order endpoint")


def _temporary_config(tmp_path: Path):
    config = load_live_config(FROZEN_CONFIG)
    shadow = replace(
        config.dual_thrust_shadow,
        state_path=str(tmp_path / "state.json"),
        event_log_path=str(tmp_path / "events.jsonl"),
        report_path=str(tmp_path / "report.json"),
    )
    return replace(config, dual_thrust_shadow=shadow)


def test_frozen_shadow_config_is_exact_hybrid_v5_balanced_runner_and_dry_run_only() -> None:
    config = load_live_config(FROZEN_CONFIG)
    shadow = config.dual_thrust_shadow

    assert config.exchange.environment == "mainnet"
    assert config.trading.dry_run is True
    assert config.risk.cost_experiment == "full_cost"
    assert config.risk.backtest_mode == "conservative"
    assert config.risk.funding_enabled is True
    assert shadow.enabled is True
    assert shadow.shadow_only is True
    assert len(shadow.enabled_symbols) == 50
    assert shadow.timeframe_minutes == 60
    assert shadow.lookback_days == 5
    assert shadow.long_k == 0.7
    assert shadow.short_k == 0.6
    assert shadow.trend_ema_period == 48
    assert shadow.stop_atr_multiple == 1.0
    assert shadow.take_profit_r == 60.0
    assert shadow.partial_take_profit_r == 8.0
    assert shadow.partial_take_profit_fraction == 0.05
    assert shadow.fail_fast_minutes == 120
    assert shadow.max_holding_minutes == 960
    assert shadow.risk_per_trade_pct == 0.025
    assert shadow.max_open_positions == 1
    assert config.risk.starting_capital_usdt == 2000.0
    assert config.risk.risk_per_trade_pct == 0.025
    assert config.strategy.super_volume_breakout_enabled is False
    assert config.strategy.ordinary_breakout_enabled is False
    assert config.strategy.pullback_reclaim_enabled is False
    assert config.strategy.oi_flush_reversal_enabled is False
    assert config.vbp_strategy.enabled is False
    assert config.reversal_alpha.enabled is False
    assert config.cmipr.enabled is False
    assert config.mtper.enabled is False
    assert config.mtpc.enabled is False


def test_shadow_parameters_match_hybrid_v5_balanced_research_selection() -> None:
    config = load_live_config(FROZEN_CONFIG)
    shadow = config.dual_thrust_shadow
    research = json.loads(RESEARCH_CONFIG.read_text(encoding="utf-8"))
    entry = research["entry"]
    runner = research["runner"]

    assert entry["family_key"] == "tf60_lb5_lk0.7_sk0.6"
    assert shadow.timeframe_minutes == 60
    assert shadow.lookback_days == 5
    assert shadow.long_k == 0.7
    assert shadow.short_k == 0.6
    assert shadow.max_directional_btc_return_4h == entry["max_directional_btc_return_4h"]
    assert shadow.risk_per_trade_pct == runner["risk_per_trade_pct"]
    assert shadow.long_risk_multiplier == runner["long_risk_multiplier"]
    assert shadow.short_risk_multiplier == runner["short_risk_multiplier"]
    assert shadow.stop_atr_multiple == runner["stop_atr_multiple"]
    assert shadow.take_profit_r == runner["take_profit_r"]
    assert shadow.partial_take_profit_r == runner["partial_take_profit_r"]
    assert shadow.partial_take_profit_fraction == runner["partial_take_profit_fraction"]
    assert shadow.max_holding_minutes == runner["max_holding_minutes"]
    assert shadow.fail_fast_minutes == runner["fail_fast_minutes"]
    assert shadow.fail_fast_min_mfe_r == runner["fail_fast_min_mfe_r"]
    assert shadow.fail_fast_max_current_r == runner["fail_fast_max_current_r"]
    assert shadow.ranking_mode == runner["ranking_mode"]
    assert shadow.max_open_positions == runner["max_open_positions"]


def test_gui_mode_preserves_shadow_safety_and_disables_old_strategies() -> None:
    config = load_live_config(FROZEN_CONFIG)
    selected = _config_with_strategy_mode(config, STRATEGY_MODE_DUAL_THRUST_SHADOW)

    assert _detect_strategy_mode(selected) == STRATEGY_MODE_DUAL_THRUST_SHADOW
    assert selected.trading.dry_run is True
    assert selected.trading.max_open_positions == 1
    assert selected.trading.max_scale_ins_per_symbol == 0
    assert selected.risk.starting_capital_usdt == 2000.0
    assert selected.risk.risk_per_trade_pct == 0.025
    assert selected.dual_thrust_shadow.enabled is True
    assert selected.vbp_strategy.enabled is False
    assert selected.reversal_alpha.enabled is False
    assert selected.cmipr.enabled is False
    assert selected.mtper.enabled is False
    assert selected.mtpc.enabled is False


def test_shadow_refuses_live_orders(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    unsafe = replace(config, trading=replace(config.trading, dry_run=False))
    trader = DualThrustShadowTrader(unsafe, FakeShadowClient())

    with pytest.raises(RuntimeError, match="dry_run=false"):
        trader.validate_startup()


def test_shadow_state_hash_prevents_cross_experiment_sample_mixing(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    first = DualThrustShadowTrader(config, FakeShadowClient())
    first._persist_state()

    changed = replace(
        config,
        dual_thrust_shadow=replace(config.dual_thrust_shadow, long_k=0.71),
    )
    assert dual_thrust_shadow_config_hash(changed) != dual_thrust_shadow_config_hash(config)
    with pytest.raises(RuntimeError, match="different frozen config"):
        DualThrustShadowTrader(changed, FakeShadowClient())


def test_shadow_runtime_lock_prevents_two_writers(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    first = DualThrustShadowTrader(config, FakeShadowClient())
    second = DualThrustShadowTrader(config, FakeShadowClient())
    first._acquire_runtime_lock()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second._acquire_runtime_lock()
    finally:
        first._release_runtime_lock()


def test_shadow_same_bar_conflict_uses_adverse_stop_first_and_full_cost(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    trader = DualThrustShadowTrader(config, FakeShadowClient())
    entry_time = datetime(2026, 7, 18, 1, 0)
    position = {
        "event_id": "event-1",
        "symbol": "BTCUSDT",
        "direction": Direction.LONG.name,
        "entry_reason": "dual_thrust_v2_balanced_50_shadow",
        "signal_available_time": entry_time.isoformat(),
        "entry_time": entry_time.isoformat(),
        "entry_delay_seconds": 0.0,
        "raw_entry_price": 100.0,
        "entry_price": 100.02,
        "entry_fee": 0.05,
        "entry_slippage": 0.02,
        "quantity": 1.0,
        "raw_stop_price": 95.0,
        "take_profit_price": 105.0,
        "unit_risk": 5.12,
        "risk_usdt": 5.12,
        "best_price": 100.02,
        "worst_price": 100.02,
        "max_mfe_r": 0.0,
        "max_mae_r": 0.0,
        "last_processed_bar_time": entry_time.isoformat(),
        "quality_score": 1.5,
        "btc_return_4h": 0.0,
    }
    trader.state["cash"] = 199.95
    trader.state["position"] = position
    conflict = Candle(
        entry_time + timedelta(minutes=1),
        100.0,
        110.0,
        90.0,
        100.0,
        1000.0,
    )

    closed = trader._process_closed_bar(position, conflict, FakeShadowClient().rules)

    assert closed is True
    assert trader.state["position"] is None
    trade = trader.state["trades"][0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["fee"] > 0.0
    assert trade["slippage"] > 0.0
    assert trade["net_pnl"] < 0.0


def test_shadow_takes_five_percent_at_8r_and_aggregates_the_runner_trade(
    tmp_path: Path,
) -> None:
    config = _temporary_config(tmp_path)
    trader = DualThrustShadowTrader(config, FakeShadowClient())
    rules = FakeShadowClient().rules
    entry_time = datetime(2026, 7, 18, 1, 0)
    position = {
        "event_id": "v5-partial-event",
        "symbol": "BTCUSDT",
        "direction": Direction.LONG.name,
        "entry_reason": "hybrid_v5_balanced_runner",
        "signal_available_time": entry_time.isoformat(),
        "entry_time": entry_time.isoformat(),
        "entry_delay_seconds": 0.0,
        "raw_entry_price": 100.0,
        "entry_price": 100.02,
        "entry_fee": 0.50,
        "entry_slippage": 0.20,
        "quantity": 10.0,
        "original_quantity": 10.0,
        "raw_stop_price": 95.0,
        "take_profit_price": 160.0,
        "partial_take_profit_price": 140.0,
        "partial_take_profit_fraction": 0.05,
        "partial_taken": False,
        "partial_legs": [],
        "unit_risk": 5.12,
        "risk_usdt": 51.2,
        "original_risk_usdt": 51.2,
        "best_price": 100.02,
        "worst_price": 100.02,
        "max_mfe_r": 0.0,
        "max_mae_r": 0.0,
        "last_processed_bar_time": entry_time.isoformat(),
        "quality_score": 1.5,
        "btc_return_4h": 0.0,
    }
    trader.state["cash"] = 1999.50
    trader.state["position"] = position
    partial_bar = Candle(
        entry_time + timedelta(minutes=1),
        120.0,
        141.0,
        110.0,
        135.0,
        1000.0,
    )

    closed = trader._process_closed_bar(position, partial_bar, rules)

    assert closed is False
    assert position["partial_taken"] is True
    assert position["quantity"] == pytest.approx(9.5)
    assert position["entry_fee"] == pytest.approx(0.475)
    assert position["risk_usdt"] == pytest.approx(48.64)
    assert len(position["partial_legs"]) == 1
    assert position["partial_legs"][0]["quantity"] == pytest.approx(0.5)
    assert position["partial_legs"][0]["net_pnl"] > 0.0

    trader._close_position(
        position,
        raw_exit=130.0,
        reason="time_stop",
        exit_time=entry_time + timedelta(minutes=2),
        rules=rules,
    )

    trade = trader.state["trades"][0]
    assert trader.state["position"] is None
    assert trade["quantity"] == pytest.approx(10.0)
    assert trade["risk_usdt"] == pytest.approx(51.2)
    assert trade["partial_exit_count"] == 1
    assert trade["partial_realized_net_pnl"] > 0.0
    assert trade["net_pnl"] > trade["partial_realized_net_pnl"]
    assert trade["pnl_r"] == pytest.approx(trade["net_pnl"] / 51.2)


def test_new_shadow_report_cannot_be_accepted_without_new_sample(tmp_path: Path) -> None:
    trader = DualThrustShadowTrader(_temporary_config(tmp_path), FakeShadowClient())

    report = trader.acceptance_report()

    assert report["status"] == "collecting_new_shadow_data"
    assert report["trade_count"] == 0
    assert report["criteria"]["minimum_30_calendar_days"] is False
    assert report["criteria"]["minimum_30_closed_trades"] is False
