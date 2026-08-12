from __future__ import annotations

from datetime import datetime
import logging
from typing import Any

import pandas as pd

from BreakoutV16GridV15QualityPfCombinedResearchFreqtrade import (
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade,
)


logger = logging.getLogger(__name__)


class BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade(
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
):
    """Production adapter for the selected V16 + Grid V15 shared account.

    Backtest and hyperopt pass directly through to the frozen combined
    research class. DRY-RUN and LIVE add only a completed-hour batch gate:
    both Breakout and Grid entries must be ranked from the same synchronized
    closed 1h candle across all 50 pairs. Signals, component arbitration,
    sizing, leverage, Grid DCA, stops and exits remain owned by the parent.
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
