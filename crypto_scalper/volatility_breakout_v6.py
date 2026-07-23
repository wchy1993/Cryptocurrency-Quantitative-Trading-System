from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .models import Direction
from .volatility_breakout_optimize import Candidate, minute_datetime
from .volatility_breakout_v4_research import V4MarketSnapshot, _regime_score


VOLATILITY_BREAKOUT_V6_NAME = "dual_thrust_volatility_breakout_v6_live_robust"


@dataclass(frozen=True)
class BreakoutV6SideGate:
    """Direction-specific, point-in-time entry quality gate.

    Every field is available at the close of the decision bar or from the
    latest completed market-context bucket. Defaults are deliberately
    permissive so a gate can be introduced one constraint at a time.
    """

    min_quality_score: float = -999.0
    max_quality_score: float = 999.0
    min_body_atr: float = 0.0
    max_body_atr: float = 999.0
    min_breakout_extension_atr: float = -999.0
    max_breakout_extension_atr: float = 999.0
    min_trend_alignment_atr: float = -999.0
    max_trend_alignment_atr: float = 999.0
    min_range_atr: float = 0.0
    max_range_atr: float = 999.0
    min_volume_ratio: float = 0.0
    max_volume_ratio: float = 999.0
    min_directional_breadth: float = 0.0
    max_directional_breadth: float = 1.0
    min_directional_btc_return_4h: float = -999.0
    max_directional_btc_return_4h: float = 0.02
    min_directional_eth_return_4h: float = -999.0
    max_directional_eth_return_4h: float = 999.0
    min_market_efficiency_12h: float = 0.0
    min_directional_symbol_efficiency_12h: float = -999.0
    min_directional_symbol_ema55_atr: float = -999.0
    max_directional_symbol_ema55_atr: float = 999.0
    min_regime_score: float = -999.0
    max_regime_score: float = 999.0

    def validate(self) -> None:
        pairs = (
            (self.min_quality_score, self.max_quality_score),
            (self.min_body_atr, self.max_body_atr),
            (
                self.min_breakout_extension_atr,
                self.max_breakout_extension_atr,
            ),
            (self.min_trend_alignment_atr, self.max_trend_alignment_atr),
            (self.min_range_atr, self.max_range_atr),
            (self.min_volume_ratio, self.max_volume_ratio),
            (self.min_directional_breadth, self.max_directional_breadth),
            (
                self.min_directional_btc_return_4h,
                self.max_directional_btc_return_4h,
            ),
            (
                self.min_directional_eth_return_4h,
                self.max_directional_eth_return_4h,
            ),
            (
                self.min_directional_symbol_ema55_atr,
                self.max_directional_symbol_ema55_atr,
            ),
            (self.min_regime_score, self.max_regime_score),
        )
        if any(low > high for low, high in pairs):
            raise ValueError("Breakout v6 side-gate minimum exceeds maximum")
        if not 0.0 <= self.min_directional_breadth <= 1.0:
            raise ValueError("minimum directional breadth must be in [0, 1]")
        if not 0.0 <= self.max_directional_breadth <= 1.0:
            raise ValueError("maximum directional breadth must be in [0, 1]")
        if not 0.0 <= self.min_market_efficiency_12h <= 1.0:
            raise ValueError("minimum market efficiency must be in [0, 1]")


@dataclass(frozen=True)
class BreakoutV6EntryConfig:
    long: BreakoutV6SideGate = BreakoutV6SideGate()
    short: BreakoutV6SideGate = BreakoutV6SideGate()
    max_signals_per_symbol_day: int = 2
    require_market_context: bool = True

    def validate(self) -> None:
        self.long.validate()
        self.short.validate()
        if self.max_signals_per_symbol_day <= 0:
            raise ValueError("max signals per symbol/day must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _context_minute(minute: int) -> int:
    return minute - minute % 60


def _side_gate(config: BreakoutV6EntryConfig, direction: Direction) -> BreakoutV6SideGate:
    return config.long if direction == Direction.LONG else config.short


def passes_breakout_v6_entry(
    candidate: Candidate,
    snapshot: V4MarketSnapshot | None,
    config: BreakoutV6EntryConfig,
) -> bool:
    """Return whether a candidate is tradable with information known then."""

    signal = candidate.signal
    if signal.direction not in {Direction.LONG, Direction.SHORT}:
        return False
    if snapshot is None:
        return not config.require_market_context

    gate = _side_gate(config, signal.direction)
    side = float(signal.direction.value)
    directional_breadth = (
        snapshot.breadth_above_ema21
        if signal.direction == Direction.LONG
        else 1.0 - snapshot.breadth_above_ema21
    )
    directional_btc = side * snapshot.btc_return_4h
    directional_eth = side * snapshot.eth_return_4h
    directional_symbol_efficiency = side * snapshot.symbol_efficiency_12h
    directional_symbol_ema55 = side * snapshot.symbol_ema55_atr
    regime_score = _regime_score(snapshot, signal.direction)
    return (
        gate.min_quality_score
        <= signal.quality_score
        <= gate.max_quality_score
        and gate.min_body_atr <= signal.body_atr <= gate.max_body_atr
        and gate.min_breakout_extension_atr
        <= signal.breakout_extension_atr
        <= gate.max_breakout_extension_atr
        and gate.min_trend_alignment_atr
        <= signal.trend_alignment_atr
        <= gate.max_trend_alignment_atr
        and gate.min_range_atr <= signal.range_atr <= gate.max_range_atr
        and gate.min_volume_ratio <= signal.volume_ratio <= gate.max_volume_ratio
        and gate.min_directional_breadth
        <= directional_breadth
        <= gate.max_directional_breadth
        and gate.min_directional_btc_return_4h
        <= directional_btc
        <= gate.max_directional_btc_return_4h
        and gate.min_directional_eth_return_4h
        <= directional_eth
        <= gate.max_directional_eth_return_4h
        and snapshot.market_efficiency_12h >= gate.min_market_efficiency_12h
        and directional_symbol_efficiency
        >= gate.min_directional_symbol_efficiency_12h
        and gate.min_directional_symbol_ema55_atr
        <= directional_symbol_ema55
        <= gate.max_directional_symbol_ema55_atr
        and gate.min_regime_score <= regime_score <= gate.max_regime_score
    )


def filter_breakout_v6_candidates(
    candidates: dict[int, list[Candidate]],
    context: dict[int, dict[str, V4MarketSnapshot]],
    config: BreakoutV6EntryConfig,
) -> dict[int, list[Candidate]]:
    """Apply v6 gates before the frozen per-symbol/day signal cap."""

    config.validate()
    daily_counts: dict[tuple[str, str], int] = defaultdict(int)
    output: dict[int, list[Candidate]] = {}
    for minute in sorted(candidates):
        snapshots = context.get(_context_minute(minute), {})
        selected: list[Candidate] = []
        for candidate in candidates[minute]:
            signal = candidate.signal
            snapshot = snapshots.get(signal.symbol)
            if not passes_breakout_v6_entry(candidate, snapshot, config):
                continue
            count_key = (
                signal.symbol,
                minute_datetime(minute).date().isoformat(),
            )
            if daily_counts[count_key] >= config.max_signals_per_symbol_day:
                continue
            daily_counts[count_key] += 1
            selected.append(candidate)
        if selected:
            output[minute] = selected
    return output


def build_breakout_v6_lane_candidates(
    candidates: dict[int, list[Candidate]],
    context: dict[int, dict[str, V4MarketSnapshot]],
    core: BreakoutV6EntryConfig,
    runner: BreakoutV6EntryConfig,
) -> tuple[dict[int, list[Candidate]], frozenset[str], frozenset[str]]:
    """Build a live-order-equivalent union of core and runner signals.

    The per-symbol/day cap is applied once, after lane classification.  This is
    important because applying separate caps and merging them would admit
    signals that a live process could never have accepted.  Core membership is
    deterministic at the decision minute and can therefore safely drive entry
    priority, sizing and exits.
    """

    core.validate()
    runner.validate()
    cap = min(core.max_signals_per_symbol_day, runner.max_signals_per_symbol_day)
    daily_counts: dict[tuple[str, str], int] = defaultdict(int)
    output: dict[int, list[Candidate]] = {}
    core_ids: set[str] = set()
    runner_ids: set[str] = set()
    for minute in sorted(candidates):
        snapshots = context.get(_context_minute(minute), {})
        selected: list[Candidate] = []
        for candidate in candidates[minute]:
            signal = candidate.signal
            snapshot = snapshots.get(signal.symbol)
            is_core = passes_breakout_v6_entry(candidate, snapshot, core)
            is_runner = passes_breakout_v6_entry(candidate, snapshot, runner)
            if not is_core and not is_runner:
                continue
            count_key = (
                signal.symbol,
                minute_datetime(minute).date().isoformat(),
            )
            if daily_counts[count_key] >= cap:
                continue
            daily_counts[count_key] += 1
            selected.append(candidate)
            if is_core:
                core_ids.add(signal.event_id)
            else:
                runner_ids.add(signal.event_id)
        if selected:
            output[minute] = selected
    return output, frozenset(core_ids), frozenset(runner_ids)
