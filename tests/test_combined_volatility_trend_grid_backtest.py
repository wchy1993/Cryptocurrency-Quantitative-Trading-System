from __future__ import annotations

from array import array
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
    CombinedPortfolioConfig,
    simulate_combined_portfolio,
)
from crypto_scalper.models import Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.trend_grid import TrendGridConfig, TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate, GridPortfolioConfig
from crypto_scalper.volatility_breakout import DualThrustSignal, VolatilityBreakoutConfig
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    minute_token,
)


def _rules(symbol: str) -> SymbolRules:
    return SymbolRules(
        symbol,
        Decimal("0.001"),
        Decimal("0.001"),
        Decimal("0.01"),
        Decimal("5"),
    )


def _series(start: datetime, price: float = 100.0) -> CompactSeries:
    return CompactSeries(
        minutes=array("q", [minute_token(start + timedelta(minutes=index)) for index in range(3)]),
        opens=array("d", [price, price, price]),
        highs=array("d", [price + 0.1, price + 0.1, price + 0.1]),
        lows=array("d", [price - 0.1, price - 0.1, price - 0.1]),
        closes=array("d", [price, price, price]),
        volumes=array("d", [100_000.0, 100_000.0, 100_000.0]),
    )


def _breakout_candidate(symbol: str, start: datetime) -> Candidate:
    signal = DualThrustSignal(
        event_id=f"breakout-{symbol}",
        symbol=symbol,
        direction=Direction.LONG,
        signal_bar_time=start - timedelta(hours=1),
        signal_available_time=start,
        raw_signal_price=100.0,
        session_open=100.0,
        upper_band=99.0,
        lower_band=90.0,
        dual_thrust_range=10.0,
        atr_value=2.0,
        trend_alignment_atr=1.0,
        volume_ratio=2.0,
        body_atr=1.0,
        directional_close_position=0.9,
        range_atr=2.0,
        breakout_extension_atr=0.2,
        quality_score=2.0,
    )
    return Candidate(signal, minute_token(start), btc_return_4h=0.0)


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


def _run(
    breakout_symbols: tuple[str, ...],
    grid_symbols: tuple[str, ...],
    *,
    max_positions: int = 2,
    allow_same_symbol: bool = False,
) -> dict:
    start = datetime(2026, 1, 1)
    symbols = tuple(dict.fromkeys((*breakout_symbols, *grid_symbols)))
    execution_data = {symbol: _series(start) for symbol in symbols}
    rules = {symbol: _rules(symbol) for symbol in symbols}
    minute = minute_token(start)
    return simulate_combined_portfolio(
        {minute: [_breakout_candidate(symbol, start) for symbol in breakout_symbols]},
        {minute: [_grid_candidate(symbol, start) for symbol in grid_symbols]},
        {symbol: {} for symbol in symbols},
        symbols,
        execution_data,
        rules,
        VolatilityBreakoutConfig(stop_atr_multiple=1.0, take_profit_r=20.0),
        PortfolioSearchConfig(
            risk_per_trade_pct=0.02,
            max_open_positions=2,
            max_notional_multiple=100.0,
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
            max_open_positions=max_positions,
            max_gross_notional_multiple=100.0,
            allow_same_symbol_across_strategies=allow_same_symbol,
        ),
        BacktestExecutionConfig(mode="conservative", taker_fee_rate=0.0),
        start,
        start + timedelta(minutes=3),
        200.0,
    )


def test_combined_portfolio_never_exceeds_global_two_position_limit() -> None:
    result = _run(("AAAUSDT", "BBBUSDT"), ("CCCUSDT",))

    assert result["max_concurrent_positions"] == 2
    assert result["max_concurrent_positions"] <= result["global_portfolio_config"]["max_open_positions"]
    assert result["rejected"][GRID_KEY]["global_position_limit"] == 1


def test_combined_portfolio_allows_one_position_from_each_strategy() -> None:
    result = _run(("AAAUSDT",), ("BBBUSDT",))

    assert result["max_concurrent_positions"] == 2
    assert result["max_entry_committed_notional_multiple"] <= 100.0
    assert result["by_strategy"][BREAKOUT_KEY]["trade_count"] == 1
    assert result["by_strategy"][GRID_KEY]["trade_count"] == 1


def test_combined_portfolio_blocks_duplicate_symbol_across_strategies() -> None:
    result = _run(("AAAUSDT",), ("AAAUSDT",))

    assert result["max_concurrent_positions"] == 1
    assert result["by_strategy"][BREAKOUT_KEY]["trade_count"] == 1
    assert result["by_strategy"][GRID_KEY]["trade_count"] == 0
    assert result["rejected"][GRID_KEY]["global_symbol_already_open"] == 1


def test_combined_portfolio_honors_stricter_one_position_override() -> None:
    result = _run(("AAAUSDT",), ("BBBUSDT",), max_positions=1)

    assert result["max_concurrent_positions"] == 1
    assert result["rejected"][GRID_KEY]["global_position_limit"] == 1
