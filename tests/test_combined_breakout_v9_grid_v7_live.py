from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_breakout_v7_grid_v5_shadow import (
    _grid_profile_to_dict,
)
from crypto_scalper.combined_breakout_v9_grid_v7_live import (
    BREAKOUT_V9_GRID_V7_LIVE_VERSION,
    CombinedBreakoutV9GridV7LiveTrader,
)
from crypto_scalper.combined_breakout_v9_grid_v7_shadow import (
    BREAKOUT_V9_GRID_V7_SHADOW_VERSION,
    COMBINED_V9_GRID_V7_NAME,
    CombinedBreakoutV9GridV7ShadowTrader,
)
from crypto_scalper.combined_volatility_trend_grid_shadow import (
    _grid_campaign_to_dict,
)
from crypto_scalper.gui import (
    ACTIVE_GUI_CONFIG_PATH,
    LIVE_GUI_CONFIG_PATH,
    _combined_live_trader_class,
    _combined_shadow_trader_class,
)
from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_trader import (
    AccountSnapshot,
    LivePosition,
)
from crypto_scalper.models import Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import (
    GridCampaign,
    GridCandidate,
    GridLevel,
    GridLot,
)
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import (
    V4MarketSnapshot,
)


ROOT = Path(__file__).resolve().parents[1]


class SettingsClient:
    api_key = "key"
    api_secret = "secret"
    environment = "mainnet"

    def ping(self) -> dict:
        return {}

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")


def _snapshot(
    symbol: str,
    now: datetime,
    *,
    breadth: float = 0.5,
    ema55_atr: float = -1.0,
) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol=symbol,
        btc_return_4h=0.0,
        eth_return_4h=0.0,
        breadth_above_ema21=breadth,
        breadth_change_4h=0.0,
        symbol_return_4h=0.0,
        symbol_efficiency_12h=0.0,
        market_efficiency_12h=0.5,
        symbol_ema55_atr=ema55_atr,
    )


def _breakout_candidate(
    symbol: str,
    now: datetime,
    direction: Direction,
    *,
    quality: float,
    body: float,
    volume: float,
    extension: float,
) -> Candidate:
    signal = DualThrustSignal(
        event_id=f"v9-{symbol}-{direction.name}",
        symbol=symbol,
        direction=direction,
        signal_bar_time=now,
        signal_available_time=now,
        raw_signal_price=100.0,
        session_open=99.0,
        upper_band=99.5,
        lower_band=95.0,
        dual_thrust_range=5.0,
        atr_value=2.0,
        trend_alignment_atr=1.0,
        volume_ratio=volume,
        body_atr=body,
        directional_close_position=0.8,
        range_atr=2.0,
        breakout_extension_atr=extension,
        quality_score=quality,
    )
    return Candidate(signal, minute_token(now), btc_return_4h=0.0)


def _grid_candidate(symbol: str, now: datetime) -> GridCandidate:
    signal = TrendGridSignal(
        event_id=f"v7-{symbol}",
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
        alignment_atr=0.6,
        extension_atr=0.3,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=0.6,
    )
    return GridCandidate(signal, minute_token(now))


def _temporary_shadow_config(tmp_path: Path):
    config = load_live_config(ROOT / ACTIVE_GUI_CONFIG_PATH)
    shadow = replace(
        config.combined_volatility_trend_grid_shadow,
        state_path=str(tmp_path / "shadow.json"),
        event_log_path=str(tmp_path / "events.jsonl"),
        report_path=str(tmp_path / "report.json"),
    )
    return replace(
        config,
        combined_volatility_trend_grid_shadow=shadow,
    )


def _temporary_live_config(tmp_path: Path):
    config = load_live_config(ROOT / LIVE_GUI_CONFIG_PATH)
    live = replace(
        config.combined_breakout_v8_grid_v6_live,
        armed=True,
        live_confirmation_text=(
            "CONFIRM_BREAKOUT_V9_GRID_V7_LIVE"
        ),
        state_path=str(tmp_path / "v9_grid_v7_live.json"),
        event_log_path=str(tmp_path / "events.jsonl"),
        report_path=str(tmp_path / "status.json"),
        transport_acceptance_required=False,
    )
    return replace(
        config,
        trading=replace(
            config.trading,
            mainnet_confirmation_text="CONFIRM_MAINNET",
        ),
        combined_breakout_v8_grid_v6_live=live,
    )


def test_active_gui_routes_v9_v7_with_isolated_ledgers(
    tmp_path: Path,
) -> None:
    dry = _temporary_shadow_config(tmp_path)
    shadow = CombinedBreakoutV9GridV7ShadowTrader(
        dry, SettingsClient()
    )
    shadow.validate_startup()
    assert shadow.combined_strategy_name == COMBINED_V9_GRID_V7_NAME
    assert (
        shadow.shadow.frozen_version
        == BREAKOUT_V9_GRID_V7_SHADOW_VERSION
    )
    assert (
        _combined_shadow_trader_class(dry)
        is CombinedBreakoutV9GridV7ShadowTrader
    )

    live_config = _temporary_live_config(tmp_path)
    live = CombinedBreakoutV9GridV7LiveTrader(
        live_config, SettingsClient()
    )
    live.validate_startup_settings_only()
    assert live.live.frozen_version == BREAKOUT_V9_GRID_V7_LIVE_VERSION
    assert (
        _combined_live_trader_class(live_config)
        is CombinedBreakoutV9GridV7LiveTrader
    )
    assert "v9_grid_v7" in str(live.state_path)
    assert live.live.order_id_prefix == "b9g7r1"


def test_v9_side_score_profit_protection_matches_frozen_candidate(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 28, 12, 0)
    trader = CombinedBreakoutV9GridV7ShadowTrader(
        _temporary_shadow_config(tmp_path), SettingsClient()
    )
    hour = minute_token(now)
    short = _breakout_candidate(
        "BTCUSDT",
        now,
        Direction.SHORT,
        quality=1.1,
        body=0.6,
        volume=1.0,
        extension=0.3,
    )
    long = _breakout_candidate(
        "ETHUSDT",
        now,
        Direction.LONG,
        quality=2.0,
        body=3.0,
        volume=2.2,
        extension=1.0,
    )
    trader._market_context = {
        hour: {
            "BTCUSDT": _snapshot("BTCUSDT", now),
            "ETHUSDT": _snapshot(
                "ETHUSDT", now, breadth=0.9, ema55_atr=1.0
            ),
        }
    }
    short_profile = trader._select_breakout_profile(short, 200.0)
    long_profile = trader._select_breakout_profile(long, 200.0)
    assert short_profile is not None
    assert short_profile.lane == "v9_score_4_short"
    assert short_profile.exit_protection.partial_take_profit_r == 2.5
    assert (
        short_profile.exit_protection.partial_take_profit_fraction
        == 0.1
    )
    assert short_profile.exit_protection.profit_floor_levels == (
        (4.0, 1.0),
    )
    assert long_profile is not None
    assert long_profile.lane == "v9_score_4_long"
    assert not long_profile.exit_protection.partial_enabled
    assert long_profile.exit_protection.profit_floor_levels == (
        (6.5, 0.75),
    )


def test_v7_profile_freezes_cycle_floor_and_notional_cap(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 28, 12, 0)
    trader = CombinedBreakoutV9GridV7ShadowTrader(
        _temporary_shadow_config(tmp_path), SettingsClient()
    )
    candidate = _grid_candidate("SOLUSDT", now)
    hour = minute_token(now)
    trader._market_context = {
        hour: {"SOLUSDT": _snapshot("SOLUSDT", now)}
    }
    profile = trader._select_grid_profile(candidate)
    assert profile is not None
    assert profile.tier.startswith("v7_")
    assert profile.signal.cycle_profit_floor_min_take_profits == 2
    assert profile.signal.cycle_profit_floor_activation_r == 0.2
    assert profile.signal.cycle_profit_floor_r == 0.03
    assert profile.portfolio.max_notional_multiple == 5.0


def test_live_grid_cycle_floor_closes_after_realized_cycles(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 28, 12, 0)
    trader = CombinedBreakoutV9GridV7LiveTrader(
        _temporary_live_config(tmp_path), SettingsClient()
    )
    candidate = _grid_candidate("SOLUSDT", now)
    hour = minute_token(now)
    trader._market_context = {
        hour: {"SOLUSDT": _snapshot("SOLUSDT", now)}
    }
    profile = trader._select_grid_profile(candidate)
    assert profile is not None
    campaign = GridCampaign(
        candidate=candidate,
        start_minute=hour - 60,
        anchor_price=100.0,
        hard_stop=106.0,
        risk_budget=100.0,
        levels=[GridLevel(0, 100.0, 1.0)],
        lots={
            0: GridLot(
                0,
                1.0,
                100.0,
                100.0,
                0.05,
                0.0,
                hour - 60,
                98.0,
                "taker",
            )
        },
        entry_count=3,
        grid_take_profit_count=2,
        best_equity_pnl=20.0,
    )
    trader.state["grid_campaign"] = _grid_campaign_to_dict(campaign)
    trader.state["grid_execution_profile"] = _grid_profile_to_dict(
        profile
    )
    position = LivePosition(
        symbol="SOLUSDT",
        position_side="BOTH",
        direction=Direction.SHORT,
        quantity=1.0,
        entry_price=100.0,
        mark_price=100.0,
        notional=100.0,
        unrealized_pnl=0.0,
        leverage=10,
        margin_type="cross",
        liquidation_price=None,
    )
    account = AccountSnapshot(
        equity=200.0,
        wallet_balance=200.0,
        available_balance=190.0,
        initial_margin=10.0,
        maintenance_margin=1.0,
        total_unrealized_pnl=0.0,
        positions={"SOLUSDT": position},
        position_rows=(position,),
        position_mode="ONE-WAY",
    )
    exits = []
    trader._close_live_position = (  # type: ignore[assignment]
        lambda strategy, symbol, direction, reason: exits.append(
            (strategy, symbol, direction, reason)
        )
    )
    trader._manage_grid_campaign(now, account)
    assert exits
    assert exits[0][3] == "cycle_profit_floor"
