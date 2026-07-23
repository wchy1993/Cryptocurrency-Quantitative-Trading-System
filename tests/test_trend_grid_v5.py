from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.trend_grid import TrendGridConfig, TrendGridSignal
from crypto_scalper.trend_grid_optimize import (
    CompactSeries,
    GridCandidate,
    GridPortfolioConfig,
    minute_token,
    simulate_grid_portfolio,
)
from crypto_scalper.trend_grid_v5 import (
    GridV5ConfidencePolicy,
    grid_v5_confidence_score,
    grid_v5_tier,
)
from crypto_scalper.trend_grid_v5_engine import simulate_grid_v5_portfolio
from crypto_scalper.trend_grid_v5_optimize import _profitable_tier
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


def _candidate(start: datetime) -> GridCandidate:
    signal = TrendGridSignal(
        event_id="grid-v5-test",
        symbol="TESTUSDT",
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(minutes=15),
        signal_available_time=start,
        raw_signal_price=100.0,
        atr_value=1.0,
        fast_ema=99.0,
        slow_ema=98.0,
        fast_slope_atr=0.2,
        slow_slope_atr=0.1,
        alignment_atr=1.0,
        extension_atr=0.3,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=0.8,
    )
    return GridCandidate(signal, minute_token(start))


def _snapshot(start: datetime) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(start),
        symbol="TESTUSDT",
        btc_return_4h=0.0,
        eth_return_4h=0.0,
        breadth_above_ema21=0.5,
        breadth_change_4h=0.0,
        symbol_return_4h=0.0,
        symbol_efficiency_12h=0.0,
        market_efficiency_12h=0.0,
        symbol_ema55_atr=0.0,
    )


def _final_policy() -> GridV5ConfidencePolicy:
    return GridV5ConfidencePolicy(
        min_quality_score=0.45,
        min_alignment_atr=0.52,
        min_extension_atr=0.2,
        max_extension_atr=0.45,
        max_volume_ratio=1.1,
        max_regime_score=0.3,
        strong_score=6,
        weak_score=2,
        reject_weak_tier=True,
    )


def test_grid_v5_scores_only_point_in_time_campaign_features() -> None:
    start = datetime(2026, 1, 1)
    candidate = _candidate(start)
    policy = _final_policy()

    assert grid_v5_confidence_score(candidate, _snapshot(start), policy) == 6
    assert grid_v5_tier(candidate, _snapshot(start), policy) == ("strong", 6)

    weak = replace(
        candidate,
        signal=replace(
            candidate.signal,
            quality_score=0.1,
            alignment_atr=0.1,
            extension_atr=0.1,
            volume_ratio=2.0,
        ),
    )
    assert grid_v5_tier(weak, None, policy) == ("weak", 1)


def test_grid_v5_default_engine_is_path_equivalent_to_grid_v4() -> None:
    start = datetime(2026, 1, 1)
    candidate = _candidate(start)
    minutes = [minute_token(start + timedelta(minutes=i)) for i in range(3)]
    rows = (
        (100.0, 100.2, 99.8, 100.0),
        (100.0, 100.1, 98.8, 99.0),
        (99.0, 99.0, 99.0, 99.0),
    )
    series = CompactSeries(
        minutes=array("q", minutes),
        opens=array("d", [row[0] for row in rows]),
        highs=array("d", [row[1] for row in rows]),
        lows=array("d", [row[2] for row in rows]),
        closes=array("d", [row[3] for row in rows]),
        volumes=array("d", [1_000.0] * len(rows)),
    )
    rules = SymbolRules(
        "TESTUSDT",
        Decimal("0.001"),
        Decimal("0.001"),
        Decimal("0.1"),
        Decimal("5"),
    )
    signal = TrendGridConfig(
        grid_levels=2,
        grid_spacing_atr=0.5,
        hard_stop_atr_multiple=2.0,
        max_total_entries=1,
        campaign_loss_limit_r=0.05,
    )
    portfolio = GridPortfolioConfig(risk_per_campaign_pct=0.05)
    execution = BacktestExecutionConfig(
        market_slippage_bps=0.0,
        stop_slippage_bps=0.0,
        take_profit_slippage_bps=0.0,
        taker_fee_rate=0.0,
    )
    common = (
        {candidate.entry_minute: [candidate]},
        {"TESTUSDT": {}},
        ("TESTUSDT",),
        {"TESTUSDT": series},
        {"TESTUSDT": rules},
        signal,
        portfolio,
        execution,
        start,
        start + timedelta(minutes=3),
        200.0,
    )

    frozen = simulate_grid_portfolio(*common)
    managed = simulate_grid_v5_portfolio(*common)

    assert managed["final_equity"] == frozen["final_equity"]
    assert managed["max_drawdown_pct"] == frozen["max_drawdown_pct"]
    assert managed["trades"][0]["net_pnl"] == frozen["trades"][0]["net_pnl"]
    assert managed["trades"][0]["v5_tier"] == "default"


def test_grid_v5_all_winning_tier_passes_robustness_check() -> None:
    assert _profitable_tier(
        {
            "net_pnl": 10.0,
            "gross_profit": 10.0,
            "gross_loss": 0.0,
            "profit_factor": None,
        }
    )
