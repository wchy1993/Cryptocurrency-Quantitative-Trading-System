from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from BreakoutV10FGridV8DualSideFreqtrade import (
    LIVE_CONTEXT_COLUMNS,
    build_synchronized_market_context,
)
from BreakoutV12MultiRegimeGridV9Freqtrade import (
    V12_CONTEXT_COLUMNS,
    build_v12_market_context,
)
from BreakoutV16GridV15QualityPfCombinedResearchFreqtrade import (
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade,
)


logger = logging.getLogger(__name__)


V12_EXTRA_CONTEXT_COLUMNS = tuple(
    column
    for column in V12_CONTEXT_COLUMNS
    if column not in LIVE_CONTEXT_COLUMNS
)


class BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade(
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
):
    """Production adapter for the selected V16 + Grid V15 shared account.

    Backtest and hyperopt pass directly through to the frozen combined
    research class. DRY-RUN and LIVE synchronize both the frozen five-column
    market context and the complete inherited V12 sidecar context from the
    same closed 1h candle across all 50 pairs before ranking entries. Signals,
    component arbitration, sizing, leverage, Grid DCA, stops and exits remain
    owned by the parent.
    """

    STRATEGY_VERSION = (
        "breakout_v16_grid_v15_quality_pf_combined_live_parity_20260812"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v16_grid_v15_quality_pf_combined_live_parity"
    )
    LIVE_ENTRY_WINDOW_SECONDS = 90

    def bot_start(self, **kwargs: Any) -> None:
        super().bot_start(**kwargs)
        if self._live_context_mode():
            # The exact synchronized-batch check below still fails closed;
            # this window only allows the observed 50-pair analysis latency.
            self.ignore_buying_expired_candle_after = (
                self.LIVE_ENTRY_WINDOW_SECONDS
            )

    def _prepare_live_context(self) -> pd.Timestamp | None:
        """Build one complete, aligned context batch for LIVE/DRY-RUN.

        The first five columns deliberately keep their frozen implementation;
        the remaining inherited V12 columns are merged as a sidecar.  This is
        the same split used by the research/backtest path and prevents a live
        sizing rule from silently receiving a missing-value fallback.
        """

        pairs = tuple(self.dp.current_whitelist())
        frozen_context, frozen_target, frozen_unavailable = (
            build_synchronized_market_context(
                self._live_pair_snapshots,
                pairs,
            )
        )
        expanded, expanded_target, expanded_unavailable = (
            build_v12_market_context(
                self._live_pair_snapshots,
                pairs,
                require_aligned_latest=True,
            )
        )
        target = (
            expanded_target
            if expanded_target is not None
            else frozen_target
        )
        unavailable = tuple(
            sorted(set(frozen_unavailable + expanded_unavailable))
        )
        if frozen_target != expanded_target:
            unavailable = pairs
        if unavailable:
            signature = (target, unavailable)
            if signature != self._live_context_wait_signature:
                logger.info(
                    "V16+Grid LIVE full context waiting: target=%s "
                    "ready=%d/%d missing_or_stale=%s",
                    target,
                    len(pairs) - len(unavailable),
                    len(pairs),
                    ",".join(unavailable),
                )
                self._live_context_wait_signature = signature
            return None
        if target is None or frozen_context.empty or expanded.empty:
            return None

        context = frozen_context.merge(
            expanded[["date", *V12_EXTRA_CONTEXT_COLUMNS]],
            on="date",
            how="left",
            validate="1:1",
        )
        latest = context.loc[
            pd.to_datetime(context["date"], utc=True) == target
        ]
        if (
            latest.empty
            or latest[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None)
        ):
            self._live_context_wait_signature = (target, pairs)
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
            for column in V12_CONTEXT_COLUMNS:
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
                pair,
                self.timeframe,
            )
            if frame is None or frame.empty:
                failures.append(pair)
                continue
            dates = pd.to_datetime(frame["date"], utc=True)
            row = frame.loc[dates == target]
            if (
                row.empty
                or not set(V12_CONTEXT_COLUMNS) <= set(row.columns)
                or row[list(V12_CONTEXT_COLUMNS)].isna().any(axis=None)
            ):
                failures.append(pair)
        return tuple(failures)

    @staticmethod
    def _utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    def _live_entry_batch_gate(
        self,
        current_time: datetime,
    ) -> tuple[bool, str, pd.Timestamp, float]:
        now = self._utc_timestamp(current_time)
        available_at = now.floor(self.timeframe)
        expected_target = available_at - pd.Timedelta(self.timeframe)
        age_seconds = max(0.0, (now - available_at).total_seconds())

        applied = getattr(self, "_live_context_applied_at", None)
        if applied is None:
            return False, "context_not_synchronized", expected_target, age_seconds
        applied_target = self._utc_timestamp(applied)
        if applied_target != expected_target:
            return False, "context_timestamp_mismatch", expected_target, age_seconds
        if age_seconds > float(self.LIVE_ENTRY_WINDOW_SECONDS):
            return False, "entry_window_expired", expected_target, age_seconds
        return True, "synchronized_current_batch", expected_target, age_seconds

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> bool:
        component = self._component(entry_tag)
        if not self._live_context_mode() or component not in {
            "breakout",
            "grid",
        }:
            return super().confirm_trade_entry(
                pair,
                order_type,
                amount,
                rate,
                time_in_force,
                current_time,
                entry_tag,
                side,
                **kwargs,
            )

        gate_ok, reason, target, age_seconds = self._live_entry_batch_gate(
            current_time
        )
        if not gate_ok:
            logger.warning(
                "V16+Grid LIVE parity entry denied: pair=%s component=%s "
                "side=%s target=%s age=%.3fs reason=%s",
                pair,
                component,
                side,
                target,
                age_seconds,
                reason,
            )
            return False

        if component == "breakout":
            signal_row = self._latest_signal_row(
                pair,
                component,
                current_time,
            )
            path_ready = (
                0
                if signal_row is None
                else signal_row.get("v16_mtf_ready", 0)
            )
            if (
                not pd.notna(path_ready)
                or int(path_ready) != 1
            ):
                logger.warning(
                    "V16+Grid LIVE parity entry denied: pair=%s component=%s "
                    "side=%s target=%s age=%.3fs reason=mtf_path_incomplete",
                    pair,
                    component,
                    side,
                    target,
                    age_seconds,
                )
                return False

        ranked = self._ranked_candidates(component, current_time)
        rank_position = next(
            (
                position
                for position, (_rank, candidate_pair) in enumerate(
                    ranked, start=1
                )
                if candidate_pair == pair
            ),
            None,
        )
        accepted = super().confirm_trade_entry(
            pair,
            order_type,
            amount,
            rate,
            time_in_force,
            current_time,
            entry_tag,
            side,
            **kwargs,
        )
        logger.info(
            "V16+Grid LIVE parity entry decision: pair=%s component=%s "
            "side=%s target=%s age=%.3fs rank=%s candidates=%d accepted=%s",
            pair,
            component,
            side,
            target,
            age_seconds,
            rank_position if rank_position is not None else "none",
            len(ranked),
            accepted,
        )
        return accepted
