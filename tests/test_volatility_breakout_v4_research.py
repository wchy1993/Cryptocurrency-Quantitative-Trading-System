from __future__ import annotations

from array import array
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_volatility_trend_grid_backtest import CombinedPortfolioConfig
from crypto_scalper.combined_volatility_trend_grid_v4_backtest import (
    simulate_combined_v4_portfolio,
)
from crypto_scalper.models import Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.trend_grid import TrendGridConfig, TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate, GridPortfolioConfig
from crypto_scalper.volatility_breakout import DualThrustSignal, VolatilityBreakoutConfig
from crypto_scalper.volatility_breakout_exit_protection import ExitProtectionConfig
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import (
    V4MarketSnapshot,
    V4RegimeConfig,
    filter_candidates_v4,
)


def _rules(symbol: str) -> SymbolRules:
    return SymbolRules(
        symbol,
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("5"),
    )


def _signal(symbol: str, available: datetime, event_id: str) -> DualThrustSignal:
    return DualThrustSignal(
        event_id=event_id,
        symbol=symbol,
        direction=Direction.LONG,
        signal_bar_time=available - timedelta(hours=1),
        signal_available_time=available,
        raw_signal_price=100.0,
        session_open=98.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=5.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )


def _snapshot(minute: int, symbol: str) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute,
        symbol=symbol,
        btc_return_4h=0.01,
        eth_return_4h=0.01,
        breadth_above_ema21=0.70,
        breadth_change_4h=0.10,
        symbol_return_4h=0.02,
        symbol_efficiency_12h=0.40,
        market_efficiency_12h=0.30,
        symbol_ema55_atr=1.0,
    )


def test_v4_filters_regime_and_applies_daily_signal_cap_after_filtering() -> None:
    start = datetime(2026, 1, 1, 1)
    first = minute_token(start)
    second = minute_token(start + timedelta(hours=1))
    candidates = {
        first: [Candidate(_signal("AAAUSDT", start, "first"), first)],
        second: [
            Candidate(
                _signal("AAAUSDT", start + timedelta(hours=1), "second"), second
            )
        ],
    }
    context = {
        first: {"AAAUSDT": _snapshot(first, "AAAUSDT")},
        second: {"AAAUSDT": _snapshot(second, "AAAUSDT")},
    }
    selected = filter_candidates_v4(
        candidates,
        VolatilityBreakoutConfig(
            min_trend_alignment_atr=0.5,
            min_volume_ratio=1.0,
            max_signals_per_symbol_day=1,
        ),
        V4RegimeConfig(
            min_directional_breadth=0.60,
            min_directional_symbol_efficiency_12h=0.20,
            min_market_efficiency_12h=0.20,
        ),
        context,
    )

    assert list(selected) == [first]
    assert selected[first][0].signal.event_id == "first"


def _series(start: datetime, rows: list[tuple[float, float, float, float]]) -> CompactSeries:
    return CompactSeries(
        minutes=array(
            "q", [minute_token(start + timedelta(minutes=index)) for index in range(len(rows))]
        ),
        opens=array("d", [row[0] for row in rows]),
        highs=array("d", [row[1] for row in rows]),
        lows=array("d", [row[2] for row in rows]),
        closes=array("d", [row[3] for row in rows]),
        volumes=array("d", [100_000.0] * len(rows)),
    )


def _grid_candidate(symbol: str, start: datetime) -> GridCandidate:
    signal = TrendGridSignal(
        event_id=f"grid-{symbol}",
        symbol=symbol,
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(hours=1),
        signal_available_time=start,
        raw_signal_price=100.0,
        atr_value=1.0,
        fast_ema=99.0,
        slow_ema=98.0,
        fast_slope_atr=0.2,
        slow_slope_atr=0.1,
        alignment_atr=1.0,
        extension_atr=1.0,
        directional_close_position=0.8,
        volume_ratio=1.0,
        quality_score=1.0,
    )
    return GridCandidate(signal, minute_token(start))


def test_v4_combined_keeps_max_two_and_accounts_for_partial_breakout_exit() -> None:
    start = datetime(2026, 1, 1)
    minute = minute_token(start)
    breakout_symbol = "AAAUSDT"
    grid_symbol = "BBBUSDT"
    rows = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 106.0, 99.0, 105.0),
        (105.0, 105.0, 94.0, 95.0),
        (95.0, 95.0, 95.0, 95.0),
    ]
    result = simulate_combined_v4_portfolio(
        {
            minute: [
                Candidate(
                    _signal(breakout_symbol, start, "breakout"), minute
                )
            ]
        },
        {minute: [_grid_candidate(grid_symbol, start)]},
        {grid_symbol: {}},
        (breakout_symbol, grid_symbol),
        {
            breakout_symbol: _series(start, rows),
            grid_symbol: _series(start, [(100.0, 100.1, 99.9, 100.0)] * 4),
        },
        {breakout_symbol: _rules(breakout_symbol), grid_symbol: _rules(grid_symbol)},
        VolatilityBreakoutConfig(
            stop_atr_multiple=1.0,
            take_profit_r=20.0,
            max_holding_minutes=60,
        ),
        PortfolioSearchConfig(
            risk_per_trade_pct=0.02,
            max_open_positions=1,
            max_notional_multiple=100.0,
        ),
        ExitProtectionConfig(
            partial_take_profit_r=1.0,
            partial_take_profit_fraction=0.5,
        ),
        TrendGridConfig(
            grid_levels=2,
            grid_spacing_atr=0.5,
            hard_stop_atr_multiple=2.0,
        ),
        GridPortfolioConfig(
            risk_per_campaign_pct=0.02,
            max_open_campaigns=1,
            max_notional_multiple=100.0,
        ),
        CombinedPortfolioConfig(
            max_open_positions=2,
            max_gross_notional_multiple=100.0,
        ),
        BacktestExecutionConfig(
            mode="conservative",
            market_slippage_bps=0.0,
            stop_slippage_bps=0.0,
            take_profit_slippage_bps=0.0,
            taker_fee_rate=0.0,
        ),
        start,
        start + timedelta(minutes=4),
        200.0,
    )

    breakout_trade = next(
        trade for trade in result["trades"] if trade["strategy"] == "volatility_breakout"
    )
    assert result["max_concurrent_positions"] == 2
    assert breakout_trade["partial_exit_count"] == 1
    assert result["breakout_partial_exit_count"] == 1
    assert abs(result["pnl_reconciliation_error"]) < 1e-9
