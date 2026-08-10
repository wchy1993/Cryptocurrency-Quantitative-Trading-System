from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from BreakoutV12MultiRegimeGridV9Freqtrade import (
    V12_CONTEXT_COLUMNS,
    _efficiency_ratio,
    _ema,
    _normalized_trends,
)


class DynamicTopNUniverse:
    """Immutable, causal month-to-membership lookup."""

    def __init__(
        self,
        path: Path,
        top_n: int,
        members_by_month: dict[str, frozenset[str]],
    ) -> None:
        self.path = path
        self.top_n = top_n
        self.members_by_month = members_by_month

    @classmethod
    def load(cls, path: Path) -> "DynamicTopNUniverse":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("causal_selection") is not True:
            raise ValueError(f"dynamic universe is not marked causal: {path}")
        top_n = int(payload.get("top_n", 0))
        if top_n < 2:
            raise ValueError(f"invalid dynamic universe top_n={top_n}: {path}")
        months: dict[str, frozenset[str]] = {}
        for month, detail in payload.get("months", {}).items():
            members = tuple(str(pair) for pair in detail.get("members", ()))
            if len(members) != int(detail.get("member_count", len(members))):
                raise ValueError(f"member count mismatch for {month}: {path}")
            if len(members) > top_n or len(set(members)) != len(members):
                raise ValueError(f"invalid members for {month}: {path}")
            months[str(month)] = frozenset(members)
        if not months:
            raise ValueError(f"dynamic universe has no months: {path}")
        return cls(path=path, top_n=top_n, members_by_month=months)

    @staticmethod
    def _month(value: Any) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.strftime("%Y-%m")

    def contains(self, pair: str, decision_time: Any) -> bool:
        return pair in self.members_by_month.get(self._month(decision_time), ())

    def mask(
        self,
        pair: str,
        candle_dates: Series | pd.DatetimeIndex,
        *,
        decision_delay: pd.Timedelta = pd.Timedelta(hours=1),
    ) -> Series:
        dates = pd.Series(pd.to_datetime(candle_dates, utc=True))
        months = (dates + decision_delay).dt.strftime("%Y-%m")
        values = months.map(
            lambda month: pair in self.members_by_month.get(month, ())
        )
        values.index = getattr(candle_dates, "index", dates.index)
        return values.astype(bool)


def _empty_context() -> DataFrame:
    return DataFrame(columns=("date", *V12_CONTEXT_COLUMNS))


def build_dynamic_market_context(
    snapshots: dict[str, DataFrame],
    pairs: tuple[str, ...],
    universe: DynamicTopNUniverse,
    *,
    require_aligned_latest: bool,
) -> tuple[DataFrame, pd.Timestamp | None, tuple[str, ...]]:
    """Build V12/V13 context from only each decision month's members.

    Price indicators retain each contract's earlier candles for warm-up, then
    the cross-section masks the completed candle by the universe applicable at
    its availability time (candle timestamp + 1h).  No future month's volume
    or membership can enter the current row.
    """

    if not pairs:
        return _empty_context(), None, ()
    normalized: dict[str, DataFrame] = {}
    missing: list[str] = []
    latest_by_pair: dict[str, pd.Timestamp] = {}
    for pair in pairs:
        frame = snapshots.get(pair)
        if frame is None or frame.empty or not {"date", "close"} <= set(frame.columns):
            missing.append(pair)
            continue
        local = frame[["date", "close"]].copy()
        local["date"] = pd.to_datetime(local["date"], utc=True)
        local["close"] = pd.to_numeric(local["close"], errors="coerce")
        local = (
            local.dropna(subset=["date", "close"])
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
        if local.empty:
            missing.append(pair)
            continue
        normalized[pair] = local
        latest_by_pair[pair] = pd.Timestamp(local["date"].iloc[-1])
    if not normalized:
        return _empty_context(), None, tuple(sorted(set(missing)))

    target = max(latest_by_pair.values())
    if require_aligned_latest:
        stale = [pair for pair in pairs if latest_by_pair.get(pair) != target]
        unavailable = tuple(sorted(set(missing + stale)))
        if unavailable:
            return _empty_context(), target, unavailable

    above_fast: dict[str, Series] = {}
    above_slow: dict[str, Series] = {}
    efficiencies: dict[str, Series] = {}
    returns_4h: dict[str, Series] = {}
    returns_24h: dict[str, Series] = {}
    trend_4h: dict[str, Series] = {}
    trend_1d: dict[str, Series] = {}
    active: dict[str, Series] = {}
    full_returns_4h: dict[str, Series] = {}
    full_returns_24h: dict[str, Series] = {}
    full_trend_4h: dict[str, Series] = {}
    full_trend_1d: dict[str, Series] = {}

    for pair, local_frame in normalized.items():
        local = local_frame.loc[local_frame["date"] <= target].set_index("date")
        close = local["close"].astype(float)
        symbol = pair.split("/", maxsplit=1)[0] + "USDT"
        mask = universe.mask(pair, close.index)
        mask.index = close.index
        local_return_4h = close / close.shift(4) - 1.0
        local_return_24h = close / close.shift(24) - 1.0
        local_trend_4h, local_trend_1d = _normalized_trends(close)
        active[symbol] = mask.astype(float)
        above_fast[symbol] = (close > _ema(close, 21)).astype(float).where(mask)
        above_slow[symbol] = (close > _ema(close, 96)).astype(float).where(mask)
        efficiencies[symbol] = _efficiency_ratio(close, 12).abs().where(mask)
        returns_4h[symbol] = local_return_4h.where(mask)
        returns_24h[symbol] = local_return_24h.where(mask)
        trend_4h[symbol] = local_trend_4h.where(mask)
        trend_1d[symbol] = local_trend_1d.where(mask)
        full_returns_4h[symbol] = local_return_4h
        full_returns_24h[symbol] = local_return_24h
        full_trend_4h[symbol] = local_trend_4h
        full_trend_1d[symbol] = local_trend_1d

    above_fast_frame = pd.concat(above_fast, axis=1).sort_index()
    above_slow_frame = pd.concat(above_slow, axis=1).reindex(above_fast_frame.index)
    efficiency_frame = pd.concat(efficiencies, axis=1).reindex(above_fast_frame.index)
    return_24h_frame = pd.concat(returns_24h, axis=1).reindex(above_fast_frame.index)
    active_frame = pd.concat(active, axis=1).reindex(above_fast_frame.index).fillna(0.0)
    active_count = active_frame.sum(axis=1)
    required = np.ceil(active_count / 2.0).clip(lower=5.0)
    valid_fast = above_fast_frame.count(axis=1) >= required
    valid_slow = above_slow_frame.count(axis=1) >= required
    valid_efficiency = efficiency_frame.count(axis=1) >= required
    valid_returns = return_24h_frame.count(axis=1) >= required
    breadth = above_fast_frame.mean(axis=1).where(valid_fast)
    breadth_slow = above_slow_frame.mean(axis=1).where(valid_slow)
    market_efficiency = efficiency_frame.mean(axis=1).where(valid_efficiency)
    market_dispersion_24h = return_24h_frame.std(axis=1).where(valid_returns)
    market_median_return_24h = return_24h_frame.median(axis=1).where(valid_returns)
    market_dispersion_7d = market_dispersion_24h.ewm(
        span=168,
        adjust=False,
        min_periods=72,
    ).mean()
    low_opportunity_fraction_72h = (
        (market_dispersion_7d <= 0.030)
        .astype(float)
        .rolling(72, min_periods=48)
        .mean()
    )

    def direct(values: dict[str, Series], symbol: str) -> Series:
        return values.get(symbol, Series(dtype=float)).reindex(above_fast_frame.index)

    btc_return_4h = direct(full_returns_4h, "BTCUSDT")
    eth_return_4h = direct(full_returns_4h, "ETHUSDT")
    btc_return_24h = direct(full_returns_24h, "BTCUSDT")
    eth_return_24h = direct(full_returns_24h, "ETHUSDT")
    btc_trend_4h = direct(full_trend_4h, "BTCUSDT")
    eth_trend_4h = direct(full_trend_4h, "ETHUSDT")
    btc_trend_1d = direct(full_trend_1d, "BTCUSDT")
    eth_trend_1d = direct(full_trend_1d, "ETHUSDT")
    breadth_change_24h = breadth - breadth.shift(24)
    breadth_travel_24h = breadth.diff().abs().rolling(24, min_periods=12).sum()
    raw_regime = (
        0.22 * ((breadth - 0.50) / 0.25).clip(-1.5, 1.5)
        + 0.12 * ((breadth_slow - 0.50) / 0.25).clip(-1.5, 1.5)
        + 0.08 * (breadth_change_24h / 0.15).clip(-1.5, 1.5)
        + 0.15 * btc_trend_4h.clip(-1.5, 1.5)
        + 0.11 * eth_trend_4h.clip(-1.5, 1.5)
        + 0.17 * btc_trend_1d.clip(-1.5, 1.5)
        + 0.15 * eth_trend_1d.clip(-1.5, 1.5)
    )
    regime_score = raw_regime.ewm(span=6, adjust=False).mean().clip(-1.0, 1.0)
    context = DataFrame(
        {
            "date": above_fast_frame.index,
            "breadth": breadth,
            "breadth_change_4h": breadth - breadth.shift(4),
            "market_efficiency": market_efficiency,
            "btc_return_4h": btc_return_4h,
            "eth_return_4h": eth_return_4h,
            "breadth_slow": breadth_slow,
            "breadth_change_24h": breadth_change_24h,
            "btc_return_24h": btc_return_24h,
            "eth_return_24h": eth_return_24h,
            "btc_trend_4h": btc_trend_4h,
            "eth_trend_4h": eth_trend_4h,
            "btc_trend_1d": btc_trend_1d,
            "eth_trend_1d": eth_trend_1d,
            "market_dispersion_24h": market_dispersion_24h,
            "market_dispersion_7d": market_dispersion_7d,
            "low_opportunity_fraction_72h": low_opportunity_fraction_72h,
            "market_median_return_24h": market_median_return_24h,
            "breadth_travel_24h": breadth_travel_24h,
            "market_regime_score": regime_score,
            "market_bull_strength": ((regime_score - 0.10) / 0.60).clip(0.0, 1.0),
            "market_bear_strength": ((-regime_score - 0.10) / 0.60).clip(0.0, 1.0),
        }
    ).reset_index(drop=True)
    if require_aligned_latest:
        latest = context.loc[context["date"] == target]
        if latest.empty or latest[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None):
            return _empty_context(), target, pairs
    return context, target, ()
