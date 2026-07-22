from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_hybrid_v5_grid_v3_backtest import (
    COMBINED_V5_GRID_V3_NAME,
)
from crypto_scalper.combined_hybrid_v5_grid_v3_shadow import (
    CombinedHybridV5GridV3ShadowTrader,
    HYBRID_V5_GRID_V3_SHADOW_VERSION,
    hybrid_v5_grid_v3_shadow_config_hash,
)
from crypto_scalper.combined_volatility_trend_grid_shadow import _utc_now
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)
from crypto_scalper.gui import (
    ACTIVE_GUI_CONFIG_PATH,
    STRATEGY_MODE_COMBINED_SHADOW,
    _combined_shadow_trader_class,
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
CONFIG = ROOT / ACTIVE_GUI_CONFIG_PATH


class FakeClient:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 7, 21, 12, 0)
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
        raise AssertionError("dry-run shadow must never call an order endpoint")


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


def test_active_gui_is_new_combined_dry_run_matching_backtest_sources(
    tmp_path: Path,
) -> None:
    config = _temporary_config(tmp_path)
    combined = config.combined_volatility_trend_grid_shadow
    trader = CombinedHybridV5GridV3ShadowTrader(config, FakeClient())

    trader.validate_startup()
    assert _detect_strategy_mode(config) == STRATEGY_MODE_COMBINED_SHADOW
    assert _combined_shadow_trader_class(config) is CombinedHybridV5GridV3ShadowTrader
    assert combined.strategy_name == COMBINED_V5_GRID_V3_NAME
    assert combined.frozen_version == HYBRID_V5_GRID_V3_SHADOW_VERSION
    assert config.exchange.environment == "mainnet"
    assert config.trading.dry_run is True
    assert config.risk.starting_capital_usdt == 200.0
    assert len(config.trading.symbols) == 50
    assert config.trading.max_open_positions == 2
    assert combined.max_open_positions_per_strategy == 1
    assert combined.allow_same_symbol_across_strategies is False
    assert tuple(combined.entry_priority) == (BREAKOUT_KEY, GRID_KEY)
    assert trader.breakout_exit_config.partial_take_profit_r == 8.0
    assert trader.breakout_exit_config.partial_take_profit_fraction == 0.05
    assert trader.grid_market_overlay.ranking_mode == "alignment_desc"
    assert trader.grid_signal_config.allow_long is False
    assert trader.grid_signal_config.allow_short is True


def test_gui_selection_keeps_new_combination_dry_run_and_max_two() -> None:
    config = load_live_config(CONFIG)
    selected = _config_with_strategy_mode(config, STRATEGY_MODE_COMBINED_SHADOW)
    selected = _config_with_strategy_selection(
        selected, indicator_enabled=True, vbp_enabled=True
    )

    assert selected.trading.dry_run is True
    assert selected.trading.max_open_positions == 2
    assert selected.risk.starting_capital_usdt == 200.0
    assert selected.risk.risk_per_trade_pct == 0.025
    assert selected.combined_volatility_trend_grid_shadow.enabled is True
    assert selected.dual_thrust_shadow.enabled is True
    assert selected.vbp_strategy.enabled is False
    assert selected.cmipr.enabled is False
    assert selected.mtper.enabled is False
    assert selected.mtpc.enabled is False


def test_new_combined_state_hash_isolated_and_live_mode_refused(tmp_path: Path) -> None:
    config = _temporary_config(tmp_path)
    trader = CombinedHybridV5GridV3ShadowTrader(config, FakeClient())
    trader.validate_startup()

    unsafe = replace(config, trading=replace(config.trading, dry_run=False))
    unsafe_shadow = replace(
        unsafe.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "unsafe-state.json"),
        event_log_path=str(tmp_path / "unsafe-events.jsonl"),
        report_path=str(tmp_path / "unsafe-report.json"),
    )
    unsafe = replace(unsafe, combined_volatility_trend_grid_shadow=unsafe_shadow)
    unsafe_trader = CombinedHybridV5GridV3ShadowTrader(unsafe, FakeClient())
    with pytest.raises(RuntimeError, match="dry_run=false"):
        unsafe_trader.validate_startup()
    assert hybrid_v5_grid_v3_shadow_config_hash(config) != (
        hybrid_v5_grid_v3_shadow_config_hash(unsafe)
    )


def test_protected_breakout_and_grid_share_max_two_account(tmp_path: Path) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    client = FakeClient(now)
    trader = CombinedHybridV5GridV3ShadowTrader(
        _temporary_config(tmp_path), client
    )
    trader.state["started_at"] = (now - timedelta(minutes=1)).isoformat()

    assert trader._open_breakout_candidate(_breakout_candidate("BTCUSDT", now), now)
    protected_payload = trader.state["breakout_position"]
    assert protected_payload["partial_target_price"] > 100.0
    assert protected_payload["partial_taken"] is False
    assert trader._decode_breakout_position(protected_payload).candidate.signal.symbol == "BTCUSDT"
    assert trader._candidate_reject_reason(
        GRID_KEY, _grid_candidate("BTCUSDT", now), now
    ) == "global_symbol_already_open"
    assert trader._open_grid_candidate(_grid_candidate("ETHUSDT", now), now)
    assert trader._open_count() == 2
    assert trader._committed_notional() <= (
        trader.snapshot_account(fetch_mark=False).equity
        * trader.shadow.max_gross_notional_multiple
    )
    assert client.order_calls == 0
