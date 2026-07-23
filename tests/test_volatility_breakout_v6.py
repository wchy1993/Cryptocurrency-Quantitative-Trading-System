from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal

from crypto_scalper.binance_client import SymbolRules
from crypto_scalper.models import Direction
from crypto_scalper.risk import BacktestExecutionConfig
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout import VolatilityBreakoutConfig
from crypto_scalper.volatility_breakout_exit_protection import (
    ExitProtectionConfig,
    simulate_exit_protected_portfolio,
)
from crypto_scalper.volatility_breakout_optimize import (
    Candidate,
    CompactSeries,
    PortfolioSearchConfig,
    minute_token,
)
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot
from crypto_scalper.volatility_breakout_v6 import (
    BreakoutV6EntryConfig,
    BreakoutV6SideGate,
    build_breakout_v6_lane_candidates,
    filter_breakout_v6_candidates,
    passes_breakout_v6_entry,
)
from crypto_scalper.volatility_breakout_v6_engine import (
    BreakoutV6ExecutionProfile,
    simulate_v6_managed_portfolio,
)
from crypto_scalper.volatility_breakout_v6_core_runner_optimize import (
    tiered_drawdown_risk_multiplier,
)
from crypto_scalper.volatility_breakout_v6_optimize import _strict_improvement


def _candidate(direction: Direction, symbol: str = "BTCUSDT") -> Candidate:
    from datetime import datetime

    signal = DualThrustSignal(
        event_id=f"event-{symbol}-{direction.name}",
        symbol=symbol,
        direction=direction,
        signal_bar_time=datetime(2026, 1, 1, 9),
        signal_available_time=datetime(2026, 1, 1, 10),
        raw_signal_price=100.0,
        session_open=99.0,
        upper_band=100.0,
        lower_band=98.0,
        dual_thrust_range=5.0,
        atr_value=1.0,
        trend_alignment_atr=3.0,
        volume_ratio=2.0,
        body_atr=1.5,
        directional_close_position=0.8,
        range_atr=5.0,
        breakout_extension_atr=0.6,
        quality_score=1.7,
    )
    return Candidate(signal=signal, entry_minute=29_455_800)


def _snapshot(symbol: str = "BTCUSDT") -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=29_455_800,
        symbol=symbol,
        btc_return_4h=0.005,
        eth_return_4h=0.004,
        breadth_above_ema21=0.75,
        breadth_change_4h=0.05,
        symbol_return_4h=0.01,
        symbol_efficiency_12h=0.2,
        market_efficiency_12h=0.3,
        symbol_ema55_atr=1.0,
    )


def test_v6_uses_direction_specific_quality_gates() -> None:
    long_candidate = _candidate(Direction.LONG)
    short_candidate = _candidate(Direction.SHORT)
    snapshot = _snapshot()
    config = BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(min_body_atr=1.0),
        short=BreakoutV6SideGate(max_body_atr=1.0),
    )

    assert passes_breakout_v6_entry(long_candidate, snapshot, config)
    assert not passes_breakout_v6_entry(short_candidate, snapshot, config)


def test_v6_rejects_missing_point_in_time_context_by_default() -> None:
    candidate = _candidate(Direction.LONG)

    assert not passes_breakout_v6_entry(
        candidate,
        None,
        BreakoutV6EntryConfig(),
    )
    assert passes_breakout_v6_entry(
        candidate,
        None,
        BreakoutV6EntryConfig(require_market_context=False),
    )


def test_v6_applies_daily_cap_after_quality_filter() -> None:
    first = _candidate(Direction.LONG, "BTCUSDT")
    rejected = replace(
        _candidate(Direction.LONG, "BTCUSDT"),
        signal=replace(first.signal, event_id="rejected", body_atr=0.2),
        entry_minute=first.entry_minute + 60,
    )
    accepted = replace(
        _candidate(Direction.LONG, "BTCUSDT"),
        signal=replace(first.signal, event_id="accepted"),
        entry_minute=first.entry_minute + 120,
    )
    context = {
        first.entry_minute: {"BTCUSDT": _snapshot()},
        rejected.entry_minute: {"BTCUSDT": _snapshot()},
        accepted.entry_minute: {"BTCUSDT": _snapshot()},
    }
    config = BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(min_body_atr=1.0),
        max_signals_per_symbol_day=2,
    )

    result = filter_breakout_v6_candidates(
        {
            first.entry_minute: [first],
            rejected.entry_minute: [rejected],
            accepted.entry_minute: [accepted],
        },
        context,
        config,
    )

    event_ids = [
        row.signal.event_id
        for rows in result.values()
        for row in rows
    ]
    assert event_ids == [first.signal.event_id, "accepted"]


def test_v6_strict_improvement_requires_both_periods() -> None:
    baseline = {
        "trade_count": 100,
        "net_profit": 100.0,
        "profit_factor": 1.5,
        "max_drawdown_pct": 0.30,
        "win_rate": 0.30,
    }
    improved = {
        "trade_count": 100,
        "net_profit": 110.0,
        "profit_factor": 1.7,
        "max_drawdown_pct": 0.25,
        "win_rate": 0.35,
    }

    assert _strict_improvement(improved, improved, baseline, baseline)
    assert not _strict_improvement(improved, baseline, baseline, baseline)


def test_v6_lane_union_applies_one_live_daily_cap() -> None:
    first = _candidate(Direction.LONG, "BTCUSDT")
    second = replace(
        first,
        signal=replace(first.signal, event_id="second", body_atr=0.4),
        entry_minute=first.entry_minute + 60,
    )
    third = replace(
        first,
        signal=replace(first.signal, event_id="third"),
        entry_minute=first.entry_minute + 120,
    )
    context = {
        first.entry_minute: {"BTCUSDT": _snapshot()},
        second.entry_minute: {"BTCUSDT": _snapshot()},
        third.entry_minute: {"BTCUSDT": _snapshot()},
    }
    core = BreakoutV6EntryConfig(
        long=BreakoutV6SideGate(min_body_atr=1.0),
        max_signals_per_symbol_day=2,
    )
    runner = BreakoutV6EntryConfig(max_signals_per_symbol_day=2)

    selected, core_ids, runner_ids = build_breakout_v6_lane_candidates(
        {
            first.entry_minute: [first],
            second.entry_minute: [second],
            third.entry_minute: [third],
        },
        context,
        core,
        runner,
    )

    assert [row.signal.event_id for rows in selected.values() for row in rows] == [
        first.signal.event_id,
        "second",
    ]
    assert core_ids == frozenset({first.signal.event_id})
    assert runner_ids == frozenset({"second"})


def _engine_fixture() -> tuple[
    datetime,
    dict[int, list[Candidate]],
    dict[str, CompactSeries],
    dict[str, SymbolRules],
    VolatilityBreakoutConfig,
    PortfolioSearchConfig,
    ExitProtectionConfig,
    BacktestExecutionConfig,
]:
    start = datetime(2026, 1, 1)
    minute = minute_token(start)
    signal = replace(
        _candidate(Direction.LONG, "TESTUSDT").signal,
        event_id="engine-event",
        symbol="TESTUSDT",
        signal_bar_time=start - timedelta(hours=1),
        signal_available_time=start,
        atr_value=5.0,
    )
    rows = (
        (100.0, 103.0, 99.0, 102.0),
        (102.0, 104.0, 94.0, 95.0),
        (95.0, 96.0, 95.0, 95.0),
    )
    series = CompactSeries(
        minutes=array(
            "q", [minute_token(start + timedelta(minutes=i)) for i in range(3)]
        ),
        opens=array("d", [row[0] for row in rows]),
        highs=array("d", [row[1] for row in rows]),
        lows=array("d", [row[2] for row in rows]),
        closes=array("d", [row[3] for row in rows]),
        volumes=array("d", [100_000.0] * 3),
    )
    rules = SymbolRules(
        "TESTUSDT",
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("5"),
    )
    signal_config = VolatilityBreakoutConfig(
        stop_atr_multiple=1.0,
        take_profit_r=20.0,
        max_holding_minutes=60,
    )
    portfolio = PortfolioSearchConfig(risk_per_trade_pct=0.02)
    exit_config = ExitProtectionConfig()
    execution = BacktestExecutionConfig(
        mode="conservative",
        market_slippage_bps=0.0,
        stop_slippage_bps=0.0,
        take_profit_slippage_bps=0.0,
        taker_fee_rate=0.0,
    )
    return (
        start,
        {minute: [Candidate(signal, minute)]},
        {"TESTUSDT": series},
        {"TESTUSDT": rules},
        signal_config,
        portfolio,
        exit_config,
        execution,
    )


def test_v6_managed_engine_default_path_is_invariant() -> None:
    (
        start,
        candidates,
        execution_data,
        rules,
        signal,
        portfolio,
        exit_config,
        execution,
    ) = _engine_fixture()
    common = (
        candidates,
        ("TESTUSDT",),
        execution_data,
        rules,
        signal,
        portfolio,
        exit_config,
        execution,
        start,
        start + timedelta(minutes=3),
        200.0,
    )

    frozen = simulate_exit_protected_portfolio(*common)
    managed = simulate_v6_managed_portfolio(*common)

    assert managed["final_equity"] == frozen["final_equity"]
    assert managed["max_drawdown_pct"] == frozen["max_drawdown_pct"]
    assert managed["trades"][0]["net_pnl"] == frozen["trades"][0]["net_pnl"]
    assert managed["trades"][0]["v6_lane"] == "default"


def test_v6_managed_engine_can_reduce_runner_risk_only() -> None:
    (
        start,
        candidates,
        execution_data,
        rules,
        signal,
        portfolio,
        exit_config,
        execution,
    ) = _engine_fixture()
    common = (
        candidates,
        ("TESTUSDT",),
        execution_data,
        rules,
        signal,
        portfolio,
        exit_config,
        execution,
        start,
        start + timedelta(minutes=3),
        200.0,
    )

    def select(_candidate: Candidate, _minute: int, _equity: float):
        return BreakoutV6ExecutionProfile(
            lane="runner",
            signal=signal,
            portfolio=replace(portfolio, risk_per_trade_pct=0.01),
            exit_protection=exit_config,
        )

    normal = simulate_v6_managed_portfolio(*common)
    reduced = simulate_v6_managed_portfolio(*common, profile_selector=select)

    assert reduced["trades"][0]["v6_lane"] == "runner"
    assert reduced["trades"][0]["risk_usdt"] < normal["trades"][0]["risk_usdt"]
    assert abs(reduced["trades"][0]["net_pnl"]) < abs(
        normal["trades"][0]["net_pnl"]
    )


def test_v6_tiered_drawdown_governor_is_deterministic() -> None:
    values = (
        tiered_drawdown_risk_multiplier(0.05, 0.10, 0.5, 0.20, 0.2),
        tiered_drawdown_risk_multiplier(0.10, 0.10, 0.5, 0.20, 0.2),
        tiered_drawdown_risk_multiplier(0.25, 0.10, 0.5, 0.20, 0.2),
    )

    assert values == (1.0, 0.5, 0.2)
