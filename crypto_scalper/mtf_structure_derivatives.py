from __future__ import annotations

import bisect
import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .data import load_candles_csv, parse_timestamp
from .indicators import atr, ema
from .models import Candle, Direction
from .volatility_breakout_optimize import minute_token


STRATEGY_NAME = "mtf_structure_derivatives_trend_following_10"
STRATEGY_VERSION = "closed_4h_1h_15m_structure_derivatives_v1"

UNIVERSE_10: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "LTCUSDT",
)


@dataclass(frozen=True)
class StructureFeatureConfig:
    atr_period: int = 14
    four_h_fast_ema: int = 12
    four_h_slow_ema: int = 36
    one_h_fast_ema: int = 20
    one_h_slow_ema: int = 50
    swing_left: int = 2
    swing_right: int = 2
    efficiency_lookback: int = 12
    slope_lookback: int = 3
    liquidity_lookback_15m: int = 12
    liquidity_min_penetration_atr: float = 0.02
    liquidity_max_penetration_atr: float = 1.50
    liquidity_close_reclaim_atr: float = 0.0
    trigger_break_lookback_15m: int = 3
    broad_max_sweep_age_bars: int = 6
    volume_lookback_15m: int = 20
    cvd_lookback_15m: int = 8

    def validate(self) -> None:
        if self.four_h_slow_ema <= self.four_h_fast_ema:
            raise ValueError("four_h_slow_ema must exceed four_h_fast_ema")
        if self.one_h_slow_ema <= self.one_h_fast_ema:
            raise ValueError("one_h_slow_ema must exceed one_h_fast_ema")
        if min(self.swing_left, self.swing_right, self.atr_period) <= 0:
            raise ValueError("swing and ATR periods must be positive")


@dataclass(frozen=True)
class SignalFilterConfig:
    allow_long: bool = True
    allow_short: bool = True
    allow_bos_trigger: bool = True
    allow_choch_trigger: bool = True
    allow_micro_bos_trigger: bool = True
    max_sweep_age_bars: int = 4
    max_four_h_bos_age: int = 30
    max_one_h_structure_age: int = 16
    min_four_h_spread_atr: float = 0.15
    min_four_h_efficiency: float = 0.18
    min_four_h_slope_atr: float = 0.0
    require_four_h_structure_alignment: bool = False
    min_one_h_spread_atr: float = 0.10
    min_one_h_efficiency: float = 0.10
    min_one_h_slope_atr: float = -0.03
    require_one_h_structure_alignment: bool = True
    min_entry_extension_atr: float = -3.0
    max_entry_extension_atr: float = 1.50
    min_volume_ratio: float = 1.0
    max_volume_ratio: float = 999.0
    min_oi_change_30m: float = -0.01
    max_oi_change_30m: float = 0.05
    min_directional_taker_imbalance: float = 0.0
    min_directional_cvd: float = 0.0
    max_directional_funding_rate: float = 0.0003
    min_quality_score: float = 4.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameFeatures:
    candles: tuple[Candle, ...]
    available_times: tuple[datetime, ...]
    atr_values: tuple[float, ...]
    fast_ema: tuple[float, ...]
    slow_ema: tuple[float, ...]
    slope_atr: tuple[float, ...]
    spread_atr: tuple[float, ...]
    efficiency: tuple[float, ...]
    structure_direction: tuple[int, ...]
    break_direction: tuple[int, ...]
    break_kind: tuple[str, ...]
    bull_bos_age: tuple[int, ...]
    bear_bos_age: tuple[int, ...]
    bull_structure_age: tuple[int, ...]
    bear_structure_age: tuple[int, ...]


@dataclass(frozen=True)
class DerivativesPoint:
    available_time: datetime
    oi_change_30m: float | None
    taker_imbalance: float | None


@dataclass(frozen=True)
class FundingPoint:
    timestamp: datetime
    rate: float


@dataclass(frozen=True)
class RawSetup:
    event_id: str
    symbol: str
    direction: Direction
    signal_bar_time: datetime
    signal_available_time: datetime
    raw_signal_price: float
    atr_15m: float
    sweep_price: float
    sweep_age_bars: int
    trigger_kind: str
    four_h_spread_atr: float
    four_h_efficiency: float
    four_h_slope_atr: float
    four_h_bos_age: int
    four_h_structure_aligned: bool
    one_h_spread_atr: float
    one_h_efficiency: float
    one_h_slope_atr: float
    one_h_structure_age: int
    one_h_structure_aligned: bool
    entry_extension_atr: float
    volume_ratio: float
    oi_change_30m: float
    directional_taker_imbalance: float
    directional_cvd: float
    directional_funding_rate: float
    quality_score: float


@dataclass(frozen=True)
class TrendState:
    available_time: datetime
    fast_invalid_long: bool
    fast_invalid_short: bool
    slow_invalid_long: bool
    slow_invalid_short: bool
    structure_invalid_long: bool
    structure_invalid_short: bool

    def invalid(self, direction: Direction, mode: str) -> bool:
        if direction == Direction.LONG:
            fast, slow, structure = (
                self.fast_invalid_long,
                self.slow_invalid_long,
                self.structure_invalid_long,
            )
        else:
            fast, slow, structure = (
                self.fast_invalid_short,
                self.slow_invalid_short,
                self.structure_invalid_short,
            )
        if mode == "fast":
            return fast
        if mode == "slow":
            return slow
        if mode == "structure":
            return structure
        return slow or structure


@dataclass(frozen=True)
class FeatureBundle:
    symbols: tuple[str, ...]
    candles_15m: dict[str, tuple[Candle, ...]]
    raw_setups: tuple[RawSetup, ...]
    trend_states: dict[str, dict[int, TrendState]]
    funding: dict[str, tuple[FundingPoint, ...]]
    source_files: dict[str, dict[str, str]]
    cvd_definition: str


def resample_candles(candles: Iterable[Candle], timeframe_minutes: int) -> list[Candle]:
    if timeframe_minutes <= 0 or timeframe_minutes % 15:
        raise ValueError("target timeframe must be a positive multiple of 15 minutes")
    factor = timeframe_minutes // 15
    if factor == 1:
        return list(candles)
    output: list[Candle] = []
    bucket: list[Candle] = []
    bucket_key: tuple[int, int] | None = None
    for candle in candles:
        minute_of_day = candle.timestamp.hour * 60 + candle.timestamp.minute
        key = (candle.timestamp.toordinal(), minute_of_day // timeframe_minutes)
        if bucket_key is not None and key != bucket_key:
            if len(bucket) == factor:
                output.append(_merge_candles(bucket))
            bucket = []
        bucket_key = key
        bucket.append(candle)
    if len(bucket) == factor:
        output.append(_merge_candles(bucket))
    return output


def _merge_candles(rows: list[Candle]) -> Candle:
    return Candle(
        timestamp=rows[0].timestamp,
        open=rows[0].open,
        high=max(item.high for item in rows),
        low=min(item.low for item in rows),
        close=rows[-1].close,
        volume=sum(item.volume for item in rows),
    )


def build_frame_features(
    candles: list[Candle],
    timeframe_minutes: int,
    fast_period: int,
    slow_period: int,
    config: StructureFeatureConfig,
) -> FrameFeatures:
    if not candles:
        return FrameFeatures((), (), (), (), (), (), (), (), (), (), (), (), (), (), ())
    closes = [item.close for item in candles]
    fast = ema(closes, fast_period)
    slow = ema(closes, slow_period)
    atr_values = atr(candles, config.atr_period)
    slope_values: list[float] = []
    spread_values: list[float] = []
    efficiency_values: list[float] = []
    for index, candle in enumerate(candles):
        current_atr = max(atr_values[index], 1e-12)
        slope_start = max(0, index - config.slope_lookback)
        slope_values.append((fast[index] - fast[slope_start]) / current_atr)
        spread_values.append((fast[index] - slow[index]) / current_atr)
        efficiency_start = max(0, index - config.efficiency_lookback)
        path = sum(
            abs(closes[row] - closes[row - 1])
            for row in range(efficiency_start + 1, index + 1)
        )
        efficiency_values.append(
            abs(candle.close - closes[efficiency_start]) / path if path > 0.0 else 0.0
        )

    structure_direction: list[int] = []
    break_direction: list[int] = []
    break_kind: list[str] = []
    bull_bos_age: list[int] = []
    bear_bos_age: list[int] = []
    bull_structure_age: list[int] = []
    bear_structure_age: list[int] = []
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    state = 0
    last_bull_bos = -10**9
    last_bear_bos = -10**9
    last_bull_structure = -10**9
    last_bear_structure = -10**9
    for index, candle in enumerate(candles):
        confirmed = index - config.swing_right
        if confirmed >= config.swing_left:
            left = confirmed - config.swing_left
            right = confirmed + config.swing_right + 1
            candidate = candles[confirmed]
            if candidate.high >= max(item.high for item in candles[left:right]):
                last_swing_high = candidate.high
            if candidate.low <= min(item.low for item in candles[left:right]):
                last_swing_low = candidate.low
        previous_close = candles[index - 1].close if index > 0 else candle.close
        bull_break = (
            last_swing_high is not None
            and candle.close > last_swing_high
            and previous_close <= last_swing_high
        )
        bear_break = (
            last_swing_low is not None
            and candle.close < last_swing_low
            and previous_close >= last_swing_low
        )
        direction = 0
        kind = "none"
        if bull_break and not bear_break:
            direction = 1
            kind = "choch" if state < 0 else "bos"
            state = 1
            last_bull_structure = index
            if kind == "bos":
                last_bull_bos = index
        elif bear_break and not bull_break:
            direction = -1
            kind = "choch" if state > 0 else "bos"
            state = -1
            last_bear_structure = index
            if kind == "bos":
                last_bear_bos = index
        structure_direction.append(state)
        break_direction.append(direction)
        break_kind.append(kind)
        bull_bos_age.append(index - last_bull_bos)
        bear_bos_age.append(index - last_bear_bos)
        bull_structure_age.append(index - last_bull_structure)
        bear_structure_age.append(index - last_bear_structure)

    return FrameFeatures(
        candles=tuple(candles),
        available_times=tuple(
            item.timestamp + timedelta(minutes=timeframe_minutes) for item in candles
        ),
        atr_values=tuple(atr_values),
        fast_ema=tuple(fast),
        slow_ema=tuple(slow),
        slope_atr=tuple(slope_values),
        spread_atr=tuple(spread_values),
        efficiency=tuple(efficiency_values),
        structure_direction=tuple(structure_direction),
        break_direction=tuple(break_direction),
        break_kind=tuple(break_kind),
        bull_bos_age=tuple(bull_bos_age),
        bear_bos_age=tuple(bear_bos_age),
        bull_structure_age=tuple(bull_structure_age),
        bear_structure_age=tuple(bear_structure_age),
    )


def load_feature_bundle(
    symbols: tuple[str, ...] = UNIVERSE_10,
    price_data_dir: str = "data/binance_15m_365d_top100",
    derivatives_data_dir: str = "data/binance_derivatives_metrics_5m",
    funding_data_dir: str = "data/binance_funding_365d_top100",
    feature_config: StructureFeatureConfig = StructureFeatureConfig(),
) -> FeatureBundle:
    feature_config.validate()
    price_root = Path(price_data_dir)
    derivatives_root = Path(derivatives_data_dir)
    funding_root = Path(funding_data_dir)
    all_setups: list[RawSetup] = []
    all_states: dict[str, dict[int, TrendState]] = {}
    all_candles: dict[str, tuple[Candle, ...]] = {}
    all_funding: dict[str, tuple[FundingPoint, ...]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for symbol in symbols:
        price_path = _latest_path(price_root, f"{symbol}_15m_*.csv")
        oi_path = _latest_path(derivatives_root, f"{symbol}_oi_5m_*.csv")
        taker_path = _latest_path(derivatives_root, f"{symbol}_taker_ratio_5m_*.csv")
        funding_path = _latest_path(funding_root, f"{symbol}_funding*.csv")
        candles = load_candles_csv(price_path)
        derivatives = _load_derivatives(oi_path, taker_path)
        funding = _load_funding(funding_path)
        setups, states = build_symbol_features(
            symbol, candles, derivatives, funding, feature_config
        )
        all_setups.extend(setups)
        all_states[symbol] = states
        all_candles[symbol] = tuple(candles)
        all_funding[symbol] = tuple(funding)
        source_files[symbol] = {
            "price": str(price_path),
            "oi": str(oi_path),
            "taker_ratio": str(taker_path),
            "funding": str(funding_path),
        }
    all_setups.sort(key=lambda item: (item.signal_available_time, item.symbol, item.event_id))
    return FeatureBundle(
        symbols=symbols,
        candles_15m=all_candles,
        raw_setups=tuple(all_setups),
        trend_states=all_states,
        funding=all_funding,
        source_files=source_files,
        cvd_definition=(
            "Rolling 15m CVD proxy = sum[15m base volume * (taker buy/sell ratio - 1) / "
            "(ratio + 1)] / rolling base volume. Binance Vision long-history metrics expose "
            "the ratio but not raw taker buy/sell volume; no raw CVD is fabricated."
        ),
    )


def load_extended_feature_bundle(
    symbols: tuple[str, ...] = UNIVERSE_10,
    base_price_data_dir: str = "data/binance_15m_365d_top100",
    bridge_1m_data_dir: str = "data/binance_1m_365d_top100",
    holdout_price_data_dir: str = "data/binance_15m_v3_exit_holdout_20260612_20260719",
    base_derivatives_data_dir: str = "data/binance_derivatives_metrics_5m",
    holdout_derivatives_data_dir: str = "data/binance_derivatives_metrics_holdout_20260612_20260718",
    base_funding_data_dir: str = "data/binance_funding_365d_top100",
    holdout_funding_data_dir: str = "data/binance_funding_v3_exit_holdout_20260612_20260719",
    feature_config: StructureFeatureConfig = StructureFeatureConfig(),
) -> FeatureBundle:
    """Load the original research year plus a separately sourced forward holdout.

    The original 15-minute archive ends before the new holdout begins. The small
    warm-up gap is reconstructed from the already stored 1-minute archive, never
    from future bars. Derivatives and funding data are not forward-filled across
    the gap.
    """

    feature_config.validate()
    base_price_root = Path(base_price_data_dir)
    bridge_root = Path(bridge_1m_data_dir)
    holdout_price_root = Path(holdout_price_data_dir)
    base_derivatives_root = Path(base_derivatives_data_dir)
    holdout_derivatives_root = Path(holdout_derivatives_data_dir)
    base_funding_root = Path(base_funding_data_dir)
    holdout_funding_root = Path(holdout_funding_data_dir)
    all_setups: list[RawSetup] = []
    all_states: dict[str, dict[int, TrendState]] = {}
    all_candles: dict[str, tuple[Candle, ...]] = {}
    all_funding: dict[str, tuple[FundingPoint, ...]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for symbol in symbols:
        base_price_path = _latest_path(base_price_root, f"{symbol}_15m_*.csv")
        bridge_path = _latest_path(bridge_root, f"{symbol}_1m_*.csv")
        holdout_price_path = _latest_path(holdout_price_root, f"{symbol}_15m_*.csv")
        base_oi_path = _latest_path(base_derivatives_root, f"{symbol}_oi_5m_*.csv")
        base_taker_path = _latest_path(base_derivatives_root, f"{symbol}_taker_ratio_5m_*.csv")
        holdout_oi_path = _latest_path(holdout_derivatives_root, f"{symbol}_oi_5m_*.csv")
        holdout_taker_path = _latest_path(
            holdout_derivatives_root, f"{symbol}_taker_ratio_5m_*.csv"
        )
        base_funding_path = _latest_path(base_funding_root, f"{symbol}_funding*.csv")
        holdout_funding_path = _latest_path(
            holdout_funding_root, f"{symbol}_funding*.csv"
        )

        base_candles = load_candles_csv(base_price_path)
        holdout_candles = load_candles_csv(holdout_price_path)
        if not base_candles or not holdout_candles:
            raise ValueError(f"empty base or holdout price data for {symbol}")
        bridge_start = base_candles[-1].timestamp + timedelta(minutes=15)
        bridge_end = holdout_candles[0].timestamp
        bridge_minutes = _load_candles_range(bridge_path, bridge_start, bridge_end)
        bridge_candles = _resample_1m_to_15m(bridge_minutes)
        candles = _deduplicate_candles(base_candles + bridge_candles + holdout_candles)

        base_derivatives = _load_derivatives(base_oi_path, base_taker_path)
        holdout_derivatives = _load_derivatives(holdout_oi_path, holdout_taker_path)
        derivatives = _deduplicate_derivatives(base_derivatives + holdout_derivatives)
        funding = _deduplicate_funding(
            _load_funding(base_funding_path) + _load_funding(holdout_funding_path)
        )
        setups, states = build_symbol_features(
            symbol, candles, derivatives, funding, feature_config
        )
        all_setups.extend(setups)
        all_states[symbol] = states
        all_candles[symbol] = tuple(candles)
        all_funding[symbol] = tuple(funding)
        source_files[symbol] = {
            "base_price": str(base_price_path),
            "bridge_1m_price": str(bridge_path),
            "holdout_price": str(holdout_price_path),
            "base_oi": str(base_oi_path),
            "base_taker_ratio": str(base_taker_path),
            "holdout_oi": str(holdout_oi_path),
            "holdout_taker_ratio": str(holdout_taker_path),
            "base_funding": str(base_funding_path),
            "holdout_funding": str(holdout_funding_path),
        }
    all_setups.sort(key=lambda item: (item.signal_available_time, item.symbol, item.event_id))
    return FeatureBundle(
        symbols=symbols,
        candles_15m=all_candles,
        raw_setups=tuple(all_setups),
        trend_states=all_states,
        funding=all_funding,
        source_files=source_files,
        cvd_definition=(
            "Rolling 15m CVD proxy = sum[15m base volume * (taker buy/sell ratio - 1) / "
            "(ratio + 1)] / rolling base volume. Binance Vision long-history metrics expose "
            "the ratio but not raw taker buy/sell volume; no raw CVD is fabricated."
        ),
    )


def build_symbol_features(
    symbol: str,
    candles_15m: list[Candle],
    derivatives: list[DerivativesPoint],
    funding: list[FundingPoint],
    config: StructureFeatureConfig,
) -> tuple[list[RawSetup], dict[int, TrendState]]:
    candles_1h = resample_candles(candles_15m, 60)
    candles_4h = resample_candles(candles_15m, 240)
    frame_15m = build_frame_features(candles_15m, 15, 8, 21, config)
    frame_1h = build_frame_features(
        candles_1h, 60, config.one_h_fast_ema, config.one_h_slow_ema, config
    )
    frame_4h = build_frame_features(
        candles_4h, 240, config.four_h_fast_ema, config.four_h_slow_ema, config
    )
    derivative_times = [item.available_time for item in derivatives]
    funding_times = [item.timestamp for item in funding]
    cvd_proxy = _cvd_proxy_series(candles_15m, derivatives, config.cvd_lookback_15m)
    volume_ratios = _volume_ratios(candles_15m, config.volume_lookback_15m)
    setups: list[RawSetup] = []
    states: dict[int, TrendState] = {}
    last_sweep = {Direction.LONG: -10**9, Direction.SHORT: -10**9}
    last_sweep_price = {Direction.LONG: 0.0, Direction.SHORT: 0.0}
    warmup = max(100, config.liquidity_lookback_15m + config.volume_lookback_15m)
    for index in range(warmup, len(candles_15m)):
        candle = candles_15m[index]
        available = candle.timestamp + timedelta(minutes=15)
        one_index = bisect.bisect_right(frame_1h.available_times, available) - 1
        four_index = bisect.bisect_right(frame_4h.available_times, available) - 1
        if one_index < config.one_h_slow_ema or four_index < config.four_h_slow_ema:
            continue
        current_atr = max(frame_15m.atr_values[index], 1e-12)
        prior = candles_15m[index - config.liquidity_lookback_15m:index]
        prior_low = min(item.low for item in prior)
        prior_high = max(item.high for item in prior)
        long_penetration = (prior_low - candle.low) / current_atr
        short_penetration = (candle.high - prior_high) / current_atr
        long_sweep = (
            config.liquidity_min_penetration_atr
            <= long_penetration
            <= config.liquidity_max_penetration_atr
            and candle.close >= prior_low + config.liquidity_close_reclaim_atr * current_atr
        )
        short_sweep = (
            config.liquidity_min_penetration_atr
            <= short_penetration
            <= config.liquidity_max_penetration_atr
            and candle.close <= prior_high - config.liquidity_close_reclaim_atr * current_atr
        )
        if long_sweep:
            last_sweep[Direction.LONG] = index
            last_sweep_price[Direction.LONG] = candle.low
        if short_sweep:
            last_sweep[Direction.SHORT] = index
            last_sweep_price[Direction.SHORT] = candle.high

        states[minute_token(available)] = _trend_state(
            available, frame_15m, index, frame_1h, one_index, frame_4h, four_index
        )
        derivative_index = bisect.bisect_right(derivative_times, available) - 1
        derivative = derivatives[derivative_index] if derivative_index >= 0 else None
        funding_index = bisect.bisect_right(funding_times, available) - 1
        funding_rate = funding[funding_index].rate if funding_index >= 0 else 0.0
        trigger_window = candles_15m[
            max(0, index - config.trigger_break_lookback_15m):index
        ]
        for direction in (Direction.LONG, Direction.SHORT):
            sweep_age = index - last_sweep[direction]
            if sweep_age < 0 or sweep_age > config.broad_max_sweep_age_bars:
                continue
            if direction == Direction.LONG:
                micro_break = candle.close > max(item.high for item in trigger_window)
            else:
                micro_break = candle.close < min(item.low for item in trigger_window)
            structural_break = frame_15m.break_direction[index] == direction.value
            if not (micro_break or structural_break):
                continue
            four_spread = direction.value * frame_4h.spread_atr[four_index]
            one_spread = direction.value * frame_1h.spread_atr[one_index]
            if four_spread <= 0.0 or one_spread <= -0.10:
                continue
            four_structure_age = _directional_age(
                frame_4h, four_index, direction, bos_only=True
            )
            one_structure_age = _directional_age(
                frame_1h, one_index, direction, bos_only=False
            )
            one_aligned = frame_1h.structure_direction[one_index] == direction.value
            four_aligned = frame_4h.structure_direction[four_index] == direction.value
            extension = direction.value * (
                candle.close - frame_1h.fast_ema[one_index]
            ) / current_atr
            oi_change = (
                derivative.oi_change_30m
                if derivative is not None and derivative.oi_change_30m is not None
                else -999.0
            )
            taker = (
                direction.value * derivative.taker_imbalance
                if derivative is not None and derivative.taker_imbalance is not None
                else -999.0
            )
            directional_cvd = direction.value * cvd_proxy[index]
            directional_funding = direction.value * funding_rate
            quality = _quality_score(
                four_spread,
                frame_4h.efficiency[four_index],
                direction.value * frame_4h.slope_atr[four_index],
                four_structure_age,
                one_spread,
                frame_1h.efficiency[one_index],
                direction.value * frame_1h.slope_atr[one_index],
                one_structure_age,
                volume_ratios[index],
                oi_change,
                taker,
                directional_cvd,
                directional_funding,
                sweep_age,
                four_aligned,
                one_aligned,
            )
            trigger_kind = (
                frame_15m.break_kind[index]
                if structural_break
                else "micro_bos"
            )
            setups.append(
                RawSetup(
                    event_id=_event_id(symbol, direction, candle.timestamp),
                    symbol=symbol,
                    direction=direction,
                    signal_bar_time=candle.timestamp,
                    signal_available_time=available,
                    raw_signal_price=candle.close,
                    atr_15m=current_atr,
                    sweep_price=last_sweep_price[direction],
                    sweep_age_bars=sweep_age,
                    trigger_kind=trigger_kind,
                    four_h_spread_atr=four_spread,
                    four_h_efficiency=frame_4h.efficiency[four_index],
                    four_h_slope_atr=direction.value * frame_4h.slope_atr[four_index],
                    four_h_bos_age=four_structure_age,
                    four_h_structure_aligned=four_aligned,
                    one_h_spread_atr=one_spread,
                    one_h_efficiency=frame_1h.efficiency[one_index],
                    one_h_slope_atr=direction.value * frame_1h.slope_atr[one_index],
                    one_h_structure_age=one_structure_age,
                    one_h_structure_aligned=one_aligned,
                    entry_extension_atr=extension,
                    volume_ratio=volume_ratios[index],
                    oi_change_30m=oi_change,
                    directional_taker_imbalance=taker,
                    directional_cvd=directional_cvd,
                    directional_funding_rate=directional_funding,
                    quality_score=quality,
                )
            )
    return setups, states


def setup_passes(setup: RawSetup, config: SignalFilterConfig) -> bool:
    if setup.direction == Direction.LONG and not config.allow_long:
        return False
    if setup.direction == Direction.SHORT and not config.allow_short:
        return False
    if setup.trigger_kind == "bos" and not config.allow_bos_trigger:
        return False
    if setup.trigger_kind == "choch" and not config.allow_choch_trigger:
        return False
    if setup.trigger_kind == "micro_bos" and not config.allow_micro_bos_trigger:
        return False
    return (
        setup.sweep_age_bars <= config.max_sweep_age_bars
        and setup.four_h_bos_age <= config.max_four_h_bos_age
        and setup.one_h_structure_age <= config.max_one_h_structure_age
        and setup.four_h_spread_atr >= config.min_four_h_spread_atr
        and setup.four_h_efficiency >= config.min_four_h_efficiency
        and setup.four_h_slope_atr >= config.min_four_h_slope_atr
        and (
            setup.four_h_structure_aligned
            or not config.require_four_h_structure_alignment
        )
        and setup.one_h_spread_atr >= config.min_one_h_spread_atr
        and setup.one_h_efficiency >= config.min_one_h_efficiency
        and setup.one_h_slope_atr >= config.min_one_h_slope_atr
        and (
            setup.one_h_structure_aligned
            or not config.require_one_h_structure_alignment
        )
        and config.min_entry_extension_atr
        <= setup.entry_extension_atr
        <= config.max_entry_extension_atr
        and config.min_volume_ratio
        <= setup.volume_ratio
        <= config.max_volume_ratio
        and config.min_oi_change_30m
        <= setup.oi_change_30m
        <= config.max_oi_change_30m
        and setup.directional_taker_imbalance >= config.min_directional_taker_imbalance
        and setup.directional_cvd >= config.min_directional_cvd
        and setup.directional_funding_rate <= config.max_directional_funding_rate
        and setup.quality_score >= config.min_quality_score
    )


def select_setups(
    setups: Iterable[RawSetup],
    config: SignalFilterConfig,
    start: datetime,
    end: datetime,
) -> list[RawSetup]:
    return [
        item
        for item in setups
        if start <= item.signal_available_time < end and setup_passes(item, config)
    ]


def _trend_state(
    available: datetime,
    frame_15m: FrameFeatures,
    index_15m: int,
    frame_1h: FrameFeatures,
    index_1h: int,
    frame_4h: FrameFeatures,
    index_4h: int,
) -> TrendState:
    one = frame_1h.candles[index_1h]
    four = frame_4h.candles[index_4h]
    fifteen_break = frame_15m.break_direction[index_15m]
    return TrendState(
        available_time=available,
        fast_invalid_long=(one.close < frame_1h.fast_ema[index_1h]),
        fast_invalid_short=(one.close > frame_1h.fast_ema[index_1h]),
        slow_invalid_long=(
            one.close < frame_1h.slow_ema[index_1h]
            or four.close < frame_4h.fast_ema[index_4h]
        ),
        slow_invalid_short=(
            one.close > frame_1h.slow_ema[index_1h]
            or four.close > frame_4h.fast_ema[index_4h]
        ),
        structure_invalid_long=(
            frame_1h.structure_direction[index_1h] < 0 or fifteen_break < 0
        ),
        structure_invalid_short=(
            frame_1h.structure_direction[index_1h] > 0 or fifteen_break > 0
        ),
    )


def _directional_age(
    frame: FrameFeatures,
    index: int,
    direction: Direction,
    *,
    bos_only: bool,
) -> int:
    if direction == Direction.LONG:
        return frame.bull_bos_age[index] if bos_only else frame.bull_structure_age[index]
    return frame.bear_bos_age[index] if bos_only else frame.bear_structure_age[index]


def _quality_score(
    four_spread: float,
    four_efficiency: float,
    four_slope: float,
    four_age: int,
    one_spread: float,
    one_efficiency: float,
    one_slope: float,
    one_age: int,
    volume_ratio: float,
    oi_change: float,
    taker: float,
    cvd: float,
    funding: float,
    sweep_age: int,
    four_aligned: bool,
    one_aligned: bool,
) -> float:
    return (
        0.65 * min(2.0, max(0.0, four_spread))
        + 0.65 * min(1.0, max(0.0, four_efficiency) * 2.0)
        + 0.35 * min(1.0, max(0.0, four_slope + 0.10))
        + 0.35 * max(0.0, 1.0 - four_age / 40.0)
        + 0.60 * min(2.0, max(0.0, one_spread))
        + 0.50 * min(1.0, max(0.0, one_efficiency) * 2.0)
        + 0.35 * min(1.0, max(0.0, one_slope + 0.10))
        + 0.40 * max(0.0, 1.0 - one_age / 20.0)
        + 0.45 * min(2.0, max(0.0, volume_ratio - 0.5))
        + 0.40 * min(1.0, max(0.0, (oi_change + 0.01) / 0.03))
        + 0.45 * min(1.0, max(0.0, taker) * 4.0)
        + 0.55 * min(1.0, max(0.0, cvd) * 4.0)
        + 0.25 * min(1.0, max(0.0, (0.0003 - funding) / 0.0003))
        + 0.45 * max(0.0, 1.0 - sweep_age / 6.0)
        + 0.35 * float(four_aligned)
        + 0.45 * float(one_aligned)
    )


def _volume_ratios(candles: list[Candle], lookback: int) -> list[float]:
    output: list[float] = []
    running = 0.0
    for index, candle in enumerate(candles):
        if index <= lookback:
            window = candles[max(0, index - lookback):index]
            average = sum(item.volume for item in window) / max(1, len(window))
        else:
            if index == lookback + 1:
                running = sum(item.volume for item in candles[index - lookback:index])
            else:
                running += candles[index - 1].volume - candles[index - lookback - 1].volume
            average = running / lookback
        output.append(candle.volume / max(average, 1e-12))
    return output


def _cvd_proxy_series(
    candles: list[Candle],
    derivatives: list[DerivativesPoint],
    lookback: int,
) -> list[float]:
    times = [item.available_time for item in derivatives]
    deltas: list[float] = []
    output: list[float] = []
    for candle in candles:
        available = candle.timestamp + timedelta(minutes=15)
        index = bisect.bisect_right(times, available) - 1
        imbalance = derivatives[index].taker_imbalance if index >= 0 else None
        delta = candle.volume * float(imbalance or 0.0)
        deltas.append(delta)
        start = max(0, len(deltas) - lookback)
        total_volume = sum(item.volume for item in candles[start:len(deltas)])
        output.append(sum(deltas[start:]) / max(total_volume, 1e-12))
    return output


def _load_derivatives(oi_path: Path, taker_path: Path) -> list[DerivativesPoint]:
    oi_rows: list[tuple[datetime, float]] = []
    with oi_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = float(row.get("sumOpenInterestValue") or row.get("sumOpenInterest") or 0.0)
            if value > 0.0:
                oi_rows.append((parse_timestamp(row["timestamp"]), value))
    taker_by_time: dict[datetime, float] = {}
    with taker_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ratio = max(1e-6, float(row.get("buySellRatio") or 1.0))
            taker_by_time[parse_timestamp(row["timestamp"])] = (ratio - 1.0) / (ratio + 1.0)
    output: list[DerivativesPoint] = []
    for index, (timestamp, value) in enumerate(oi_rows):
        oi_change = value / oi_rows[index - 6][1] - 1.0 if index >= 6 else None
        window = [
            taker_by_time.get(oi_rows[row][0])
            for row in range(max(0, index - 2), index + 1)
        ]
        valid = [item for item in window if item is not None]
        imbalance = sum(valid) / len(valid) if valid else None
        output.append(
            DerivativesPoint(
                available_time=timestamp + timedelta(minutes=5),
                oi_change_30m=oi_change,
                taker_imbalance=imbalance,
            )
        )
    return output


def _load_funding(path: Path) -> list[FundingPoint]:
    output: list[FundingPoint] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output.append(
                FundingPoint(
                    timestamp=parse_timestamp(row["timestamp"]),
                    rate=float(row.get("rate") or row.get("funding_rate") or 0.0),
                )
            )
    output.sort(key=lambda item: item.timestamp)
    return output


def _load_candles_range(path: Path, start: datetime, end: datetime) -> list[Candle]:
    output: list[Candle] = []
    start_text = start.isoformat(timespec="seconds")
    end_text = end.isoformat(timespec="seconds")
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token = row["timestamp"]
            if token < start_text:
                continue
            if token >= end_text:
                break
            output.append(
                Candle(
                    timestamp=parse_timestamp(token),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return output


def _resample_1m_to_15m(candles: list[Candle]) -> list[Candle]:
    output: list[Candle] = []
    bucket: list[Candle] = []
    bucket_start: datetime | None = None
    for candle in candles:
        current_start = candle.timestamp.replace(
            minute=(candle.timestamp.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        if bucket_start is not None and current_start != bucket_start:
            if len(bucket) == 15 and all(
                right.timestamp - left.timestamp == timedelta(minutes=1)
                for left, right in zip(bucket, bucket[1:])
            ):
                output.append(_merge_candles(bucket))
            bucket = []
        bucket_start = current_start
        bucket.append(candle)
    if len(bucket) == 15 and all(
        right.timestamp - left.timestamp == timedelta(minutes=1)
        for left, right in zip(bucket, bucket[1:])
    ):
        output.append(_merge_candles(bucket))
    return output


def _deduplicate_candles(rows: list[Candle]) -> list[Candle]:
    by_time = {item.timestamp: item for item in rows}
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _deduplicate_derivatives(rows: list[DerivativesPoint]) -> list[DerivativesPoint]:
    by_time = {item.available_time: item for item in rows}
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _deduplicate_funding(rows: list[FundingPoint]) -> list[FundingPoint]:
    by_time = {item.timestamp: item for item in rows}
    return [by_time[timestamp] for timestamp in sorted(by_time)]


def _latest_path(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {root / pattern}")
    return matches[-1]


def _event_id(symbol: str, direction: Direction, timestamp: datetime) -> str:
    raw = f"{STRATEGY_VERSION}|{symbol}|{direction.name}|{timestamp.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
