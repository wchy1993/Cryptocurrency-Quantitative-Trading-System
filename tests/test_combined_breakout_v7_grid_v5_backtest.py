from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from crypto_scalper.combined_breakout_v7_grid_v5_backtest import (
    UNIVERSE_100,
    _breakout_profile_selector,
    _eligible_grid_for_combined,
    _grid_profile_selector,
    build_frozen_v7_v5_configs,
)
from crypto_scalper.combined_breakout_v7_grid_v5_shadow import (
    COMBINED_V7_GRID_V5_NAME,
)
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)
from crypto_scalper.models import Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    UNIVERSE_50,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


ROOT = Path(__file__).resolve().parents[1]


def _configs():
    return build_frozen_v7_v5_configs(
        ROOT / "config.volatility-breakout.v7-confidence-refined-50.json",
        ROOT / "config.trend-grid.v5-confidence-final-50.json",
        ROOT / "config.combined-breakout-v7-grid-v5-max2.json",
    )


def _snapshot(symbol: str, now: datetime) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol=symbol,
        btc_return_4h=0.01,
        eth_return_4h=0.01,
        breadth_above_ema21=0.6,
        breadth_change_4h=0.0,
        symbol_return_4h=0.01,
        symbol_efficiency_12h=0.1,
        market_efficiency_12h=0.5,
        symbol_ema55_atr=1.0,
    )


def _breakout_candidate(symbol: str, now: datetime) -> Candidate:
    return Candidate(
        DualThrustSignal(
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
        ),
        minute_token(now),
        btc_return_4h=0.01,
    )


def _grid_candidate(
    symbol: str,
    now: datetime,
    tier: str,
) -> GridCandidate:
    values = {
        "strong": (0.6, 0.6, 0.3, 1.0),
        "standard": (0.4, 0.6, 0.3, 1.0),
        "weak": (0.1, 0.1, 0.1, 2.0),
    }
    quality, alignment, extension, volume = values[tier]
    return GridCandidate(
        TrendGridSignal(
            event_id=f"grid-{symbol}-{tier}",
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
            alignment_atr=alignment,
            extension_atr=extension,
            directional_close_position=0.8,
            volume_ratio=volume,
            quality_score=quality,
        ),
        minute_token(now),
    )


def test_frozen_shared_account_is_exact_v7_v5_max_two() -> None:
    configs = _configs()

    assert (
        configs["combined_payload"]["strategy_name"]
        == COMBINED_V7_GRID_V5_NAME
    )
    assert configs["breakout_managed_signal"].stop_atr_multiple == 0.8
    assert configs["breakout_managed_signal"].max_holding_minutes == 1200
    assert configs["breakout_exit"].breakeven_trigger_r == 5.0
    assert configs["breakout_exit"].profit_giveback_activation_r == 15.0
    assert configs["grid_policy"].reject_weak_tier is True
    assert configs["grid_signal"].allow_long is False
    assert configs["grid_signal"].allow_short is True
    assert configs["combined"].max_open_positions == 2
    assert configs["combined"].entry_priority == (BREAKOUT_KEY, GRID_KEY)


def test_universe_100_is_a_strict_extension_of_frozen_50() -> None:
    assert len(UNIVERSE_100) == 100
    assert len(set(UNIVERSE_100)) == 100
    assert set(UNIVERSE_50) < set(UNIVERSE_100)
    assert UNIVERSE_100[:2] == ("BTCUSDT", "ETHUSDT")


def test_breakout_v7_profile_uses_entry_time_confidence_risk() -> None:
    now = datetime(2026, 7, 1, 12, 0)
    configs = _configs()
    minute = minute_token(now)
    selector = _breakout_profile_selector(
        configs,
        {minute: {"BTCUSDT": _snapshot("BTCUSDT", now)}},
    )

    profile = selector(_breakout_candidate("BTCUSDT", now), minute, 200.0)

    assert profile is not None
    assert profile.lane == "v7_strong_score_5"
    assert math.isclose(
        profile.portfolio.risk_per_trade_pct,
        0.034 * 1.3 * 1.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert profile.signal.stop_atr_multiple == 0.8
    assert profile.exit_protection.profit_giveback_r == 12.0


def test_grid_v5_tiers_collapse_exactly_and_weak_is_rejected() -> None:
    now = datetime(2026, 7, 1, 12, 0)
    minute = minute_token(now)
    configs = _configs()
    symbols = ("ETHUSDT", "SOLUSDT", "XRPUSDT")
    context = {
        minute: {symbol: _snapshot(symbol, now) for symbol in symbols}
    }
    selector = _grid_profile_selector(configs, context)
    candidates = {
        minute: [
            _grid_candidate("ETHUSDT", now, "strong"),
            _grid_candidate("SOLUSDT", now, "standard"),
            _grid_candidate("XRPUSDT", now, "weak"),
        ]
    }

    eligible, signal, portfolio, rejected = _eligible_grid_for_combined(
        candidates, selector
    )

    assert len(eligible[minute]) == 2
    assert rejected == 1
    assert signal.grid_target_spacing == 1.85
    assert signal.campaign_loss_limit_r == 0.7
    assert signal.max_campaign_minutes == 4320
    assert math.isclose(
        portfolio.risk_per_campaign_pct,
        0.11,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert portfolio.short_risk_multiplier == 1.0
