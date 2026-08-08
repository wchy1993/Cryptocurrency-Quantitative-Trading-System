from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


PROJECTS_DIR = Path(__file__).resolve().parents[3]
BREAKOUT_PROJECT = PROJECTS_DIR / "freqtrade_breakout_v9_grid_v7"
GRID_PROJECT = PROJECTS_DIR / "freqtrade_grid_v8_dual_side"
for strategy_dir in (
    BREAKOUT_PROJECT / "user_data" / "strategies",
    GRID_PROJECT / "user_data" / "strategies",
):
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

import BreakoutV9GridV7Freqtrade as breakout_v9  # noqa: E402
import BreakoutV10GridV7Freqtrade as breakout_v10  # noqa: E402
import GridV8DualSideFreqtrade as grid_v8  # noqa: E402


logger = logging.getLogger(__name__)

LIVE_CONTEXT_COLUMNS = (
    "breadth",
    "breadth_change_4h",
    "market_efficiency",
    "btc_return_4h",
    "eth_return_4h",
)


def build_synchronized_market_context(
    snapshots: dict[str, DataFrame],
    pairs: tuple[str, ...],
) -> tuple[DataFrame, pd.Timestamp | None, tuple[str, ...]]:
    """Build the frozen cross-sectional context from one aligned closed bar.

    Live Freqtrade analyzes the whitelist sequentially.  The frame passed to
    ``populate_indicators`` can therefore be one closed candle ahead of frames
    returned by ``DataProvider.get_pair_dataframe``.  Building context lazily
    from the latter leaves the newest row empty and makes every live entry
    expire one hour later.

    This helper is deliberately strict: all active pairs must expose the same
    newest closed timestamp.  It never fills a current row with older context.
    The indicator formulas are identical to the frozen v9 implementation.
    """

    if not pairs:
        return DataFrame(columns=("date", *LIVE_CONTEXT_COLUMNS)), None, ()

    missing = tuple(pair for pair in pairs if pair not in snapshots)
    if missing:
        return (
            DataFrame(columns=("date", *LIVE_CONTEXT_COLUMNS)),
            None,
            missing,
        )

    normalized: dict[str, DataFrame] = {}
    latest_by_pair: dict[str, pd.Timestamp] = {}
    for pair in pairs:
        frame = snapshots[pair]
        if frame is None or frame.empty or not {"date", "close"} <= set(
            frame.columns
        ):
            missing += (pair,)
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
            missing += (pair,)
            continue
        normalized[pair] = local
        latest_by_pair[pair] = pd.Timestamp(local["date"].iloc[-1])

    if missing:
        return (
            DataFrame(columns=("date", *LIVE_CONTEXT_COLUMNS)),
            None,
            tuple(sorted(set(missing))),
        )

    target = max(latest_by_pair.values())
    stale = tuple(
        pair for pair in pairs if latest_by_pair.get(pair) != target
    )
    if stale:
        return (
            DataFrame(columns=("date", *LIVE_CONTEXT_COLUMNS)),
            target,
            stale,
        )

    above_ema: dict[str, Series] = {}
    absolute_efficiency: dict[str, Series] = {}
    returns: dict[str, Series] = {}
    for pair in pairs:
        local = normalized[pair]
        local = local.loc[local["date"] <= target].set_index("date")
        close = local["close"].astype(float)
        symbol = breakout_v9.BreakoutV9GridV7Freqtrade._symbol(pair)
        above_ema[symbol] = (
            close > breakout_v9._ema(close, 21)
        ).astype(float)
        absolute_efficiency[symbol] = breakout_v9._efficiency_ratio(
            close, 12
        ).abs()
        returns[symbol] = close / close.shift(4) - 1.0

    above_frame = pd.concat(above_ema, axis=1).sort_index()
    efficiency_frame = pd.concat(
        absolute_efficiency, axis=1
    ).reindex(above_frame.index)
    minimum_count = max(5, len(pairs) // 2)
    breadth = above_frame.mean(axis=1).where(
        above_frame.count(axis=1) >= minimum_count
    )
    market_efficiency = efficiency_frame.mean(axis=1).where(
        efficiency_frame.count(axis=1) >= minimum_count
    )
    btc_return = returns.get("BTCUSDT", Series(dtype=float)).reindex(
        above_frame.index
    )
    eth_return = returns.get("ETHUSDT", Series(dtype=float)).reindex(
        above_frame.index
    )
    context = DataFrame(
        {
            "date": above_frame.index,
            "breadth": breadth,
            "breadth_change_4h": breadth - breadth.shift(4),
            "market_efficiency": market_efficiency,
            "btc_return_4h": btc_return,
            "eth_return_4h": eth_return,
        }
    ).reset_index(drop=True)
    latest = context.loc[context["date"] == target]
    if latest.empty or latest[list(LIVE_CONTEXT_COLUMNS)].isna().any(axis=None):
        return (
            DataFrame(columns=("date", *LIVE_CONTEXT_COLUMNS)),
            target,
            pairs,
        )
    return context, target, ()


class BreakoutV10FGridV8DualSideFreqtrade(
    breakout_v10.BreakoutV10FGridV7Freqtrade,
    grid_v8.GridV8DualSideSelected,
):
    """Frozen Breakout v10F plus frozen long/short Grid v8.

    This adapter only composes the two selected sleeves.  It does not change
    either sleeve's signals, risk budgets, position management, or exits.
    The shared account preserves the original combined-policy constraints:
    at most one Breakout and one Grid campaign, with Breakout taking priority
    when only one portfolio slot remains.
    """

    MAX_OPEN_TRADES = 2

    def bot_start(self, **kwargs: Any) -> None:
        super().bot_start(**kwargs)
        self._live_pair_snapshots: dict[str, DataFrame] = {}
        self._live_context_applied_at: pd.Timestamp | None = None
        self._live_context_pending_at: pd.Timestamp | None = None
        self._live_context_wait_signature: tuple[
            pd.Timestamp | None, tuple[str, ...]
        ] | None = None

    def _live_context_mode(self) -> bool:
        provider = getattr(self, "dp", None)
        runmode = getattr(provider, "runmode", None)
        value = str(getattr(runmode, "value", runmode) or "").lower()
        return value in {"live", "dry_run", "dry-run"}

    def _capture_live_snapshot(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        if not self._live_context_mode():
            return
        pair = str(metadata.get("pair") or "")
        if not pair or dataframe.empty:
            return
        self._live_pair_snapshots[pair] = dataframe[
            ["date", "close"]
        ].copy()

    def _prepare_live_context(self) -> pd.Timestamp | None:
        pairs = tuple(self.dp.current_whitelist())
        context, target, unavailable = build_synchronized_market_context(
            self._live_pair_snapshots,
            pairs,
        )
        if unavailable:
            signature = (target, unavailable)
            if signature != self._live_context_wait_signature:
                logger.info(
                    "LIVE context waiting: target=%s ready=%d/%d missing_or_stale=%s",
                    target,
                    len(pairs) - len(unavailable),
                    len(pairs),
                    ",".join(unavailable),
                )
                self._live_context_wait_signature = signature
            return None
        if target is None or context.empty:
            return None
        self._market_context_cache = context
        self._market_context_end = target
        self._live_context_pending_at = target
        self._live_context_wait_signature = None
        return target

    def _attach_market_context(self, dataframe: DataFrame) -> DataFrame:
        if not self._live_context_mode():
            return super()._attach_market_context(dataframe)
        context = getattr(self, "_market_context_cache", None)
        if context is None or context.empty:
            output = dataframe.copy()
            for column in LIVE_CONTEXT_COLUMNS:
                output[column] = np.nan
            return output
        return dataframe.merge(
            context,
            on="date",
            how="left",
            validate="m:1",
        )

    def _latest_live_context_failures(
        self,
        pairs: tuple[str, ...],
        target: pd.Timestamp,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        for pair in pairs:
            frame, _updated = self.dp.get_analyzed_dataframe(
                pair, self.timeframe
            )
            if frame is None or frame.empty:
                failures.append(pair)
                continue
            dates = pd.to_datetime(frame["date"], utc=True)
            row = frame.loc[dates == target]
            if (
                row.empty
                or not set(LIVE_CONTEXT_COLUMNS) <= set(row.columns)
                or row[list(LIVE_CONTEXT_COLUMNS)].isna().any(axis=None)
            ):
                failures.append(pair)
        return tuple(failures)

    def analyze(self, pairs: list[str]) -> None:
        if not self._live_context_mode():
            super().analyze(pairs)
            return

        # First pass captures the exact closed frame Freqtrade supplied for
        # every pair.  Most loops skip unchanged candles and remain cheap.
        self.process_only_new_candles = True
        super().analyze(pairs)
        target = self._prepare_live_context()
        if target is None or target == self._live_context_applied_at:
            return

        # Re-run this one timestamp once with synchronized context.  Entry
        # discovery happens only after analyze() returns, so candidates from
        # this pass are still current and can reach confirm_trade_entry.
        self.process_only_new_candles = False
        try:
            super().analyze(pairs)
        finally:
            self.process_only_new_candles = True

        failures = self._latest_live_context_failures(tuple(pairs), target)
        if failures:
            logger.error(
                "LIVE context synchronization failed: target=%s failed=%s",
                target,
                ",".join(failures),
            )
            return

        self._live_context_applied_at = target
        candidates: list[str] = []
        for pair in pairs:
            frame, _updated = self.dp.get_analyzed_dataframe(
                pair, self.timeframe
            )
            if frame is None or frame.empty:
                continue
            dates = pd.to_datetime(frame["date"], utc=True)
            row = frame.loc[dates == target]
            if row.empty:
                continue
            latest = row.iloc[-1]
            if int(latest.get("enter_long") or 0):
                candidates.append(f"{pair}:LONG")
            if int(latest.get("enter_short") or 0):
                candidates.append(f"{pair}:SHORT")
        logger.info(
            "LIVE context synchronized: target=%s pairs=%d candidates=%s",
            target,
            len(pairs),
            ",".join(candidates) if candidates else "none",
        )

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        self._capture_live_snapshot(dataframe, metadata)
        return super().populate_indicators(dataframe, metadata)

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        breakout_long = dataframe["bo_entry_long"].astype(bool)
        breakout_short = dataframe["bo_entry_short"].astype(bool)
        breakout = breakout_long | breakout_short
        grid_long = (
            dataframe["grid_entry_long"].astype(bool) & ~breakout
        )
        grid_short = (
            dataframe["grid_entry_short"].astype(bool) & ~breakout
        )

        dataframe.loc[breakout_long | grid_long, "enter_long"] = 1
        dataframe.loc[breakout_short | grid_short, "enter_short"] = 1

        for index in dataframe.index[breakout]:
            score = int(dataframe.at[index, "bo_score"])
            risk_units = int(
                round(
                    float(dataframe.at[index, "bo_risk_pct"])
                    * 1_000_000
                )
            )
            capture = int(dataframe.at[index, "bo_capture"])
            long_floor = int(dataframe.at[index, "bo_long_floor"])
            dataframe.at[index, "enter_tag"] = (
                f"bo_v9_s{score}_r{risk_units}"
                f"_c{capture}_l{long_floor}"
            )
        for index in dataframe.index[grid_long]:
            score = int(dataframe.at[index, "grid_long_score"])
            dataframe.at[index, "enter_tag"] = (
                f"grid_v8_long_s{score}"
            )
        for index in dataframe.index[grid_short]:
            score = int(dataframe.at[index, "grid_score"])
            dataframe.at[index, "enter_tag"] = (
                f"grid_v8_short_s{score}"
            )

        pair = str(metadata.get("pair") or "")
        if pair:
            ranks: dict[tuple[str, int], float] = {}
            available_times = (
                pd.to_datetime(dataframe["date"], utc=True)
                + pd.Timedelta(self.timeframe)
            )
            for index in dataframe.index[breakout]:
                ranks[
                    (
                        "breakout",
                        int(available_times.at[index].timestamp()),
                    )
                ] = float(dataframe.at[index, "bo_v10_rank"])
            for index in dataframe.index[grid_long | grid_short]:
                ranks[
                    (
                        "grid",
                        int(available_times.at[index].timestamp()),
                    )
                ] = float(dataframe.at[index, "grid_rank"])
            self._candidate_ranks[pair] = ranks
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        component = self._component(entry_tag)
        arguments = (
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
        )
        if component == "breakout":
            return breakout_v9.BreakoutV9GridV7Freqtrade.custom_stake_amount(
                self,
                *arguments,
                **kwargs,
            )
        if component == "grid":
            return grid_v8.GridV8DualSideFreqtrade.custom_stake_amount(
                self,
                *arguments,
                **kwargs,
            )
        return 0.0

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        """Apply state changes once for each unique completed order.

        Binance/Freqtrade may revisit a closed order when fee information is
        updated.  Claim the stable order id at the outer combined adapter so
        both Grid v8's initial-fill handler and the inherited Grid campaign
        counters are protected from duplicate callbacks.
        """

        component = self._component(getattr(trade, "enter_tag", None))
        tag = str(getattr(order, "ft_order_tag", "") or "")
        initial_entry = (
            getattr(order, "ft_order_side", None)
            == getattr(trade, "entry_side", None)
            and int(getattr(trade, "nr_of_successful_entries", 0)) == 1
        )
        stateful_fill = initial_entry or (
            component == "breakout" and tag == "bo_v9_partial"
        ) or (
            component == "grid"
            and (
                tag.startswith("grid_dca_")
                or tag.startswith("grid_tp_")
            )
        )
        if initial_entry and self._latest_signal_row(
            pair,
            component,
            current_time,
        ) is None:
            # Do not claim an initial order until its initialization context
            # is available; a later retry must still be able to initialize it.
            return
        if stateful_fill and not self._claim_fill_state_update(trade, order):
            return

        forwarded = dict(kwargs)
        if stateful_fill:
            forwarded["_fill_state_claimed"] = True
        super().order_filled(
            pair,
            trade,
            order,
            current_time,
            **forwarded,
        )
