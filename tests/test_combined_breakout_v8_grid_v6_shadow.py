from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_breakout_v8_grid_v6_shadow import (
    BREAKOUT_V8_GRID_V6_SHADOW_VERSION,
    COMBINED_V8_GRID_V6_NAME,
    CombinedBreakoutV8GridV6ShadowTrader,
    combined_v8_grid_v6_shadow_config_hash,
)
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)
from crypto_scalper.combined_volatility_trend_grid_shadow import _utc_now
from crypto_scalper.gui import (
    STRATEGY_MODE_COMBINED_SHADOW,
    TradingApp,
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
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import (
    V4MarketSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.gui.breakout-v8-grid-v6-max2-shadow.json"


class FakeClient:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 7, 24, 12, 0)
        self.order_calls = 0

    def ping(self) -> dict:
        return {}

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")

    def klines(
        self, _symbol: str, interval: str, _limit: int
    ) -> list[Candle]:
        if interval == "1m":
            return [
                Candle(
                    self.now,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100_000.0,
                )
            ]
        return []

    def funding_rate_history(self, *_args, **_kwargs) -> list[dict]:
        return []

    def new_market_order(self, *_args, **_kwargs) -> None:
        self.order_calls += 1
        raise AssertionError(
            "Breakout v8 / Grid v6 dry-run must never send an order"
        )


def _temporary_config(tmp_path: Path):
    config = load_live_config(CONFIG)
    shadow = replace(
        config.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "state.json"),
        event_log_path=str(tmp_path / "events.jsonl"),
        report_path=str(tmp_path / "report.json"),
    )
    return replace(
        config, combined_volatility_trend_grid_shadow=shadow
    )


def _snapshot(symbol: str, now: datetime) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol=symbol,
        btc_return_4h=0.0,
        eth_return_4h=0.0,
        breadth_above_ema21=0.5,
        breadth_change_4h=0.0,
        symbol_return_4h=0.0,
        symbol_efficiency_12h=0.0,
        market_efficiency_12h=0.5,
        symbol_ema55_atr=-1.0,
    )


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
        volume_ratio=2.2,
        body_atr=3.0,
        directional_close_position=0.8,
        range_atr=2.0,
        breakout_extension_atr=1.0,
        quality_score=2.0,
    )
    return Candidate(signal, minute_token(now), btc_return_4h=0.0)


def _grid_candidate(
    symbol: str,
    now: datetime,
    *,
    strong: bool = True,
    extension_atr: float | None = None,
) -> GridCandidate:
    extension = (
        extension_atr
        if extension_atr is not None
        else (0.3 if strong else 0.1)
    )
    signal = TrendGridSignal(
        event_id=f"grid-{symbol}-{strong}-{extension}",
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
        alignment_atr=0.6 if strong else 0.1,
        extension_atr=extension,
        directional_close_position=0.8,
        volume_ratio=1.0 if strong else 2.0,
        quality_score=0.6 if strong else 0.1,
    )
    return GridCandidate(signal, minute_token(now))


def _install_context(
    trader: CombinedBreakoutV8GridV6ShadowTrader,
    now: datetime,
    *symbols: str,
) -> None:
    minute = minute_token(now)
    hour = minute - minute % 60
    trader._market_context = {
        hour: {symbol: _snapshot(symbol, now) for symbol in symbols}
    }


def test_active_gui_routes_exact_v8_v6_dry_run_sources(
    tmp_path: Path,
) -> None:
    config = _temporary_config(tmp_path)
    combined = config.combined_volatility_trend_grid_shadow
    trader = CombinedBreakoutV8GridV6ShadowTrader(
        config, FakeClient()
    )

    trader.validate_startup()
    assert (
        CONFIG.name
        == "config.gui.breakout-v8-grid-v6-max2-shadow.json"
    )
    assert _detect_strategy_mode(config) == STRATEGY_MODE_COMBINED_SHADOW
    assert (
        _combined_shadow_trader_class(config)
        is CombinedBreakoutV8GridV6ShadowTrader
    )
    assert combined.strategy_name == COMBINED_V8_GRID_V6_NAME
    assert (
        combined.frozen_version
        == BREAKOUT_V8_GRID_V6_SHADOW_VERSION
    )
    assert config.exchange.environment == "mainnet"
    assert config.trading.dry_run is True
    assert config.risk.starting_capital_usdt == 200.0
    assert config.trading.leverage == 10
    assert len(config.trading.symbols) == 50
    assert config.trading.max_open_positions == 2
    assert combined.max_open_positions_per_strategy == 1
    assert combined.allow_same_symbol_across_strategies is False
    assert tuple(combined.entry_priority) == (BREAKOUT_KEY, GRID_KEY)
    assert trader.breakout_score_allocation.score_5_short_factor == 2.6
    assert trader.breakout_managed_profile.core_stop_atr == 0.77
    assert (
        trader.grid_campaign_policy.minimum_actual_extension_atr
        == 0.05
    )
    assert trader.grid_signal_config.allow_long is False
    assert trader.grid_signal_config.allow_short is True


def test_gui_selection_preserves_v8_v6_dry_run_envelope() -> None:
    config = load_live_config(CONFIG)
    selected = _config_with_strategy_mode(
        config, STRATEGY_MODE_COMBINED_SHADOW
    )
    selected = _config_with_strategy_selection(
        selected, indicator_enabled=True, vbp_enabled=True
    )

    assert selected.trading.dry_run is True
    assert selected.trading.max_open_positions == 2
    assert selected.risk.starting_capital_usdt == 200.0
    assert selected.combined_volatility_trend_grid_shadow.enabled is True
    assert selected.dual_thrust_shadow.enabled is True
    assert selected.vbp_strategy.enabled is False
    assert selected.cmipr.enabled is False
    assert selected.mtper.enabled is False
    assert selected.mtpc.enabled is False


def test_gui_shadow_client_loads_api_credentials_but_remains_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crypto_scalper.gui as gui_module

    config = load_live_config(CONFIG)
    monkeypatch.setattr(
        gui_module,
        "read_secret",
        lambda name: (
            "test-api-key"
            if "KEY" in name
            else "test-api-secret"
        ),
    )

    client = TradingApp._client_for_config(object(), config)

    assert client.api_key == "test-api-key"
    assert client.api_secret == "test-api-secret"
    assert config.trading.dry_run is True
    assert config.combined_volatility_trend_grid_shadow.shadow_only is True


def test_v8_v6_hash_isolated_and_live_mode_refused(
    tmp_path: Path,
) -> None:
    config = _temporary_config(tmp_path)
    trader = CombinedBreakoutV8GridV6ShadowTrader(
        config, FakeClient()
    )
    trader.validate_startup()

    unsafe = replace(
        config, trading=replace(config.trading, dry_run=False)
    )
    unsafe_shadow = replace(
        unsafe.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "unsafe-state.json"),
        event_log_path=str(tmp_path / "unsafe-events.jsonl"),
        report_path=str(tmp_path / "unsafe-report.json"),
    )
    unsafe = replace(
        unsafe,
        combined_volatility_trend_grid_shadow=unsafe_shadow,
    )
    unsafe_trader = CombinedBreakoutV8GridV6ShadowTrader(
        unsafe, FakeClient()
    )
    with pytest.raises(RuntimeError, match="dry_run=false"):
        unsafe_trader.validate_startup()
    assert combined_v8_grid_v6_shadow_config_hash(config) != (
        combined_v8_grid_v6_shadow_config_hash(unsafe)
    )


def test_v8_and_v6_share_max_two_without_order_calls(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    client = FakeClient(now)
    config = _temporary_config(tmp_path)
    trader = CombinedBreakoutV8GridV6ShadowTrader(config, client)
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT", "ETHUSDT")

    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    breakout_payload = trader.state["breakout_position"]
    assert breakout_payload["profile"]["lane"].startswith("v8_")
    assert (
        trader._candidate_reject_reason(
            GRID_KEY, _grid_candidate("BTCUSDT", now), now
        )
        == "global_symbol_already_open"
    )
    assert trader._open_grid_candidate(
        _grid_candidate("ETHUSDT", now), now
    )
    assert trader.state["grid_execution_profile"]["tier"].startswith(
        "v6_strong"
    )
    assert trader._open_count() == 2
    assert trader._committed_notional() <= (
        trader.snapshot_account(fetch_mark=False).equity
        * trader.shadow.max_gross_notional_multiple
    )
    trader._persist_state()
    restored = CombinedBreakoutV8GridV6ShadowTrader(config, client)
    assert restored._open_count() == 2
    assert restored.state["grid_execution_profile"]["tier"].startswith(
        "v6_strong"
    )
    assert client.order_calls == 0


def test_grid_v6_entry_guard_rejects_too_shallow_extension(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    trader = CombinedBreakoutV8GridV6ShadowTrader(
        _temporary_config(tmp_path), FakeClient(now)
    )
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "ETHUSDT")

    candidate = _grid_candidate(
        "ETHUSDT", now, strong=True, extension_atr=0.01
    )
    assert trader._select_grid_profile(candidate) is None
    assert (
        trader._candidate_reject_reason(GRID_KEY, candidate, now)
        == "grid_v6_campaign_policy_rejected"
    )
