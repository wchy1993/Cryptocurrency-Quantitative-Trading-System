from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_volatility_trend_grid_backtest import BREAKOUT_KEY, GRID_KEY
from crypto_scalper.combined_volatility_trend_grid_shadow import (
    CombinedVolatilityTrendGridShadowTrader,
    _grid_campaign_from_dict,
    _grid_campaign_to_dict,
    _utc_now,
    combined_shadow_config_hash,
)
from crypto_scalper.gui import (
    STRATEGY_MODE_COMBINED_SHADOW,
    _config_with_strategy_mode,
    _config_with_strategy_selection,
    _detect_strategy_mode,
)
from crypto_scalper.live_config import load_live_config
from crypto_scalper.models import Candle, Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import Candidate, minute_token


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.gui.combined-volatility-trend-grid-max2-shadow.json"


class FakeCombinedClient:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 7, 19, 12, 0)
        self.order_calls = 0

    def ping(self) -> dict:
        return {}

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")

    def klines(self, _symbol: str, interval: str, _limit: int) -> list[Candle]:
        if interval == "1m":
            return [Candle(self.now, 100.0, 100.0, 100.0, 100.0, 100_000.0)]
        return []

    def funding_rate_history(self, *_args, **_kwargs) -> list[dict]:
        return []

    def new_market_order(self, *_args, **_kwargs) -> None:
        self.order_calls += 1
        raise AssertionError("combined shadow must never call an order endpoint")


def _temporary_config(tmp_path: Path):
    config = load_live_config(CONFIG)
    shadow = replace(
        config.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "state.json"),
        event_log_path=str(tmp_path / "events.jsonl"),
        report_path=str(tmp_path / "report.json"),
    )
    return replace(config, combined_volatility_trend_grid_shadow=shadow)


def _breakout_candidate(symbol: str, now: datetime) -> Candidate:
    signal = DualThrustSignal(
        event_id=f"bo-{symbol}",
        symbol=symbol,
        direction=Direction.LONG,
        signal_bar_time=now,
        signal_available_time=now,
        raw_signal_price=100.0,
        session_open=99.0,
        upper_band=99.5,
        lower_band=95.0,
        dual_thrust_range=5.0,
        atr_value=2.0,
        trend_alignment_atr=1.0,
        volume_ratio=1.0,
        body_atr=1.0,
        directional_close_position=0.8,
        range_atr=1.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    return Candidate(signal, minute_token(now), btc_return_4h=0.0)


def _grid_candidate(symbol: str, now: datetime) -> GridCandidate:
    signal = TrendGridSignal(
        event_id=f"grid-{symbol}",
        symbol=symbol,
        direction=Direction.SHORT,
        signal_bar_time=now,
        signal_available_time=now,
        raw_signal_price=100.0,
        atr_value=2.0,
        fast_ema=99.0,
        slow_ema=101.0,
        fast_slope_atr=-0.2,
        slow_slope_atr=-0.1,
        alignment_atr=1.0,
        extension_atr=0.5,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=1.0,
    )
    return GridCandidate(signal, minute_token(now))


def test_legacy_combined_config_remains_exact_50_symbol_dry_run() -> None:
    config = load_live_config(CONFIG)
    combined = config.combined_volatility_trend_grid_shadow

    assert _detect_strategy_mode(config) == STRATEGY_MODE_COMBINED_SHADOW
    assert config.exchange.environment == "mainnet"
    assert config.trading.dry_run is True
    assert len(config.trading.symbols) == 50
    assert tuple(config.trading.symbols) == tuple(combined.enabled_symbols)
    assert config.trading.max_open_positions == 2
    assert combined.max_open_positions == 2
    assert combined.max_open_positions_per_strategy == 1
    assert combined.max_gross_notional_multiple == 9.0
    assert combined.allow_same_symbol_across_strategies is False
    assert tuple(combined.entry_priority) == (BREAKOUT_KEY, GRID_KEY)
    assert config.dual_thrust_shadow.max_open_positions == 1
    assert config.risk.cost_experiment == "full_cost"
    assert config.risk.backtest_mode == "conservative"
    assert config.risk.funding_enabled is True


def test_gui_selection_cannot_disable_dry_run_or_raise_combined_limit() -> None:
    config = load_live_config(CONFIG)
    selected = _config_with_strategy_mode(config, STRATEGY_MODE_COMBINED_SHADOW)
    selected = _config_with_strategy_selection(selected, indicator_enabled=True, vbp_enabled=True)

    assert selected.trading.dry_run is True
    assert selected.trading.max_open_positions == 2
    assert selected.trading.max_new_entries_per_cycle == 2
    assert selected.trading.max_scale_ins_per_symbol == 0
    assert selected.combined_volatility_trend_grid_shadow.enabled is True
    assert selected.dual_thrust_shadow.enabled is True
    assert selected.vbp_strategy.enabled is False
    assert selected.cmipr.enabled is False
    assert selected.mtper.enabled is False
    assert selected.mtpc.enabled is False


def test_combined_shadow_refuses_live_and_hash_isolates_state(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    trader = CombinedVolatilityTrendGridShadowTrader(config, FakeCombinedClient())
    trader.validate_startup()

    unsafe = replace(config, trading=replace(config.trading, dry_run=False))
    unsafe_shadow = replace(
        unsafe.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "unsafe-state.json"),
        event_log_path=str(tmp_path / "unsafe-events.jsonl"),
        report_path=str(tmp_path / "unsafe-report.json"),
    )
    unsafe = replace(unsafe, combined_volatility_trend_grid_shadow=unsafe_shadow)
    unsafe_trader = CombinedVolatilityTrendGridShadowTrader(unsafe, FakeCombinedClient())
    with pytest.raises(RuntimeError, match="dry_run=false"):
        unsafe_trader.validate_startup()
    assert combined_shadow_config_hash(config) != combined_shadow_config_hash(unsafe)


def test_shared_account_opens_one_source_each_and_forbids_same_symbol(tmp_path: Path) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    client = FakeCombinedClient(now)
    trader = CombinedVolatilityTrendGridShadowTrader(_temporary_config(tmp_path), client)
    trader.state["started_at"] = (now - timedelta(minutes=1)).isoformat()

    assert trader._open_breakout_candidate(_breakout_candidate("BTCUSDT", now), now)
    same_symbol_reason = trader._candidate_reject_reason(
        GRID_KEY, _grid_candidate("BTCUSDT", now), now
    )
    assert same_symbol_reason == "global_symbol_already_open"
    assert trader._open_grid_candidate(_grid_candidate("ETHUSDT", now), now)
    assert trader._open_count() == 2
    assert trader._committed_notional() <= (
        trader.snapshot_account(fetch_mark=False).equity
        * trader.shadow.max_gross_notional_multiple
    )
    assert trader._candidate_reject_reason(
        BREAKOUT_KEY, _breakout_candidate("SOLUSDT", now), now
    ) == "strategy_position_limit"
    assert client.order_calls == 0


def test_grid_campaign_persistence_round_trip(tmp_path: Path) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    trader = CombinedVolatilityTrendGridShadowTrader(
        _temporary_config(tmp_path), FakeCombinedClient(now)
    )
    trader.state["started_at"] = (now - timedelta(minutes=1)).isoformat()
    assert trader._open_grid_candidate(_grid_candidate("ETHUSDT", now), now)
    restored = _grid_campaign_from_dict(trader.state["grid_campaign"])
    encoded = _grid_campaign_to_dict(restored)

    assert restored.candidate.signal.direction == Direction.SHORT
    assert len(restored.levels) == 3
    assert len(restored.lots) == 1
    assert encoded == trader.state["grid_campaign"]
