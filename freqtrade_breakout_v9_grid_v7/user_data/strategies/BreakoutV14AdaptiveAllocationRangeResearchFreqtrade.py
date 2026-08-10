from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import Order, Trade, stoploss_from_absolute

from BreakoutV9GridV7Freqtrade import _causal_signal_cap
from BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade import (
    BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade,
)
from BreakoutV12RegimeAdaptiveGridV9Freqtrade import (
    _BreakoutV12BullReentryMixin,
)
from BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade import (
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
)
from V14DynamicUniverseSupport import (
    DynamicTopNUniverse,
    build_dynamic_market_context,
)


def _rsi(series: Series, period: int = 14) -> Series:
    """Wilder RSI using only completed values at and before each row."""

    delta = series.astype(float).diff()
    gain = delta.clip(lower=0.0).ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    loss = (-delta.clip(upper=0.0)).ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    relative = gain / loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    return result.where(loss > 0.0, 100.0).where(gain > 0.0, 0.0)


class _V14WalletEquityView:
    """Delegate Wallets while presenting one deterministic virtual equity."""

    def __init__(self, wallets: Any, total_stake_amount: float) -> None:
        self._wallets = wallets
        self._total_stake_amount = float(total_stake_amount)

    def get_total_stake_amount(self) -> float:
        return self._total_stake_amount

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wallets, name)


class _V14GridTruePullbackMixin:
    """Require an actual completed 4h pullback for every Grid long.

    The inherited Grid-v8 long entry can still trigger after four hours of
    positive target-symbol return.  That is momentum continuation, not a
    pullback, and caused the largest 2024 Grid-long losses.  Zero is a
    structural boundary with a stable -0.5%..0% neighborhood, rather than a
    fitted decimal threshold next to one historical trade.
    """

    V14_GRID_LONG_MAX_RETURN_4H = 0.0

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_grid(dataframe)
        false_pullback = (
            dataframe["grid_entry_long"].astype(bool)
            & (
                dataframe["symbol_return_4h"]
                > self.V14_GRID_LONG_MAX_RETURN_4H
            )
        )
        dataframe["grid_v14_false_pullback_rejected"] = (
            false_pullback.astype(int)
        )
        dataframe.loc[false_pullback, "grid_entry_long"] = 0
        dataframe["grid_entry"] = (
            dataframe["grid_entry_long"].astype(bool)
            | dataframe["grid_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class _V14DynamicUniverseMixin:
    """Apply a causal historical Top50 to both context and executable signals."""

    V14_DYNAMIC_UNIVERSE_PATH = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "v14_work"
        / "dynamic_top50_universe.json"
    )

    def _v14_universe_path(self) -> Path:
        configured = self.config.get("v14_dynamic_universe_manifest")
        return Path(configured) if configured else self.V14_DYNAMIC_UNIVERSE_PATH

    def bot_start(self, **kwargs: Any) -> None:
        self._v14_dynamic_universe = DynamicTopNUniverse.load(
            self._v14_universe_path()
        )
        super().bot_start(**kwargs)

    def _build_market_context(self) -> DataFrame:
        if self._live_context_mode():
            # LIVE uses the current month's exact pairlist and the inherited
            # synchronized latest-candle barrier.
            return super()._build_market_context()
        pairs = tuple(self.dp.current_whitelist())
        snapshots: dict[str, DataFrame] = {}
        for pair in pairs:
            raw = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if raw is not None and not raw.empty:
                snapshots[pair] = raw[["date", "close"]].copy()
        context, _target, _unavailable = build_dynamic_market_context(
            snapshots,
            pairs,
            self._v14_dynamic_universe,
            require_aligned_latest=False,
        )
        return context

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        pair = str(metadata.get("pair") or "")
        if not pair:
            return dataframe
        member = self._v14_dynamic_universe.mask(pair, dataframe["date"])
        member.index = dataframe.index
        dataframe["v14_dynamic_universe_member"] = member.astype(int)
        rejected = ~member & (
            dataframe["enter_long"].astype(bool)
            | dataframe["enter_short"].astype(bool)
        )
        dataframe["v14_dynamic_universe_rejected"] = rejected.astype(int)
        dataframe.loc[~member, ["enter_long", "enter_short"]] = 0
        dataframe.loc[~member, "enter_tag"] = None

        # Candidate ranks are built before this final gate.  Remove stale
        # ranks so an ineligible contract cannot consume the shared Max2 slot.
        ranks = dict(self._candidate_ranks.get(pair, {}))
        self._candidate_ranks[pair] = {
            key: value
            for key, value in ranks.items()
            if self._v14_dynamic_universe.contains(
                pair,
                datetime.fromtimestamp(int(key[1]), tz=timezone.utc),
            )
        }
        return dataframe

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
        if not self._v14_dynamic_universe.contains(pair, current_time):
            return False
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


class _V14CrossUniverseSignalQualityMixin:
    """Reject late Breakout-long chases and falling-knife Grid shorts.

    Both boundaries describe market structure rather than a calendar regime:
    a Breakout long is late once the target has already advanced too far in
    four completed hours, while a mean-reversion Grid short must first show an
    actual target-symbol rebound.  The checks use the same completed 1h candle
    in backtest and LIVE and do not depend on the selected universe rank.
    """

    V14_BO_LONG_MAX_RETURN_4H = 0.09
    V14_GRID_SHORT_MIN_RETURN_4H = 0.0

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        exhausted_long = dataframe["bo_entry_long"].astype(bool) & (
            dataframe["symbol_return_4h"]
            > self.V14_BO_LONG_MAX_RETURN_4H
        )
        dataframe["bo_v14_late_chase_rejected"] = exhausted_long.astype(int)
        dataframe.loc[exhausted_long, "bo_entry_long"] = 0
        dataframe["bo_entry"] = (
            dataframe["bo_entry_long"].astype(bool)
            | dataframe["bo_entry_short"].astype(bool)
        ).astype(int)
        return dataframe

    def _populate_grid(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_grid(dataframe)
        falling_short = dataframe["grid_entry_short"].astype(bool) & (
            dataframe["symbol_return_4h"]
            < self.V14_GRID_SHORT_MIN_RETURN_4H
        )
        dataframe["grid_v14_falling_short_rejected"] = falling_short.astype(
            int
        )
        dataframe.loc[falling_short, "grid_entry_short"] = 0
        dataframe["grid_entry"] = (
            dataframe["grid_entry_long"].astype(bool)
            | dataframe["grid_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class _V14LeadershipAwareSignalQualityMixin(
    _V14CrossUniverseSignalQualityMixin,
):
    """Retain a genuine idiosyncratic leader despite large 4h extension.

    The ordinary late-chase gate remains the default.  Its only exception is
    an unusually liquid, high-efficiency impulse with broad participation
    while BTC itself is not already extended.  This captures independent
    leadership rather than indiscriminately reopening late momentum trades.
    """

    V14_LEADER_MIN_VOLUME_RATIO = 20.0
    V14_LEADER_MIN_BODY_ATR = 5.0
    V14_LEADER_MIN_SYMBOL_EFFICIENCY = 0.70
    V14_LEADER_MIN_BREADTH = 0.40
    V14_LEADER_MAX_BTC_TREND_4H = 0.10

    def _populate_breakout(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_breakout(dataframe)
        leadership = (
            dataframe["bo_v14_late_chase_rejected"].astype(bool)
            & (
                dataframe["bo_volume_ratio"]
                >= self.V14_LEADER_MIN_VOLUME_RATIO
            )
            & (dataframe["bo_body_atr"] >= self.V14_LEADER_MIN_BODY_ATR)
            & (
                dataframe["symbol_efficiency_12h"]
                >= self.V14_LEADER_MIN_SYMBOL_EFFICIENCY
            )
            & (dataframe["breadth"] >= self.V14_LEADER_MIN_BREADTH)
            & (
                dataframe["btc_trend_4h"]
                <= self.V14_LEADER_MAX_BTC_TREND_4H
            )
        )
        dataframe["bo_v14_leadership_exception"] = leadership.astype(int)
        dataframe.loc[leadership, "bo_v14_late_chase_rejected"] = 0
        dataframe.loc[leadership, "bo_entry_long"] = 1
        dataframe.loc[leadership, "bo_entry_short"] = 0
        dataframe.loc[leadership, "bo_entry"] = 1
        return dataframe


class _V14ContinuousEntryQualityRiskMixin:
    """Continuously compress structurally late entries without deleting them.

    Signal priority and Max2 occupancy remain unchanged.  Only initial risk is
    interpolated over broad completed-candle zones, which avoids the discrete
    alternate-trade path caused by a hard gate and makes nearby live fills less
    likely to flip the portfolio into a different compounding branch.
    """

    V14_BO_LONG_RISK_START_RETURN_4H = 0.08
    V14_BO_LONG_RISK_FULL_RETURN_4H = 0.14
    V14_BO_LONG_RISK_MIN_SCALE = 0.50
    V14_GRID_SHORT_RISK_START_DECLINE_4H = 0.0
    V14_GRID_SHORT_RISK_FULL_DECLINE_4H = 0.02
    V14_GRID_SHORT_RISK_MIN_SCALE = 0.50
    V14_GRID_LONG_RISK_START_EFFICIENCY = 0.24
    V14_GRID_LONG_RISK_FULL_EFFICIENCY = 0.32
    V14_GRID_LONG_RISK_MIN_SCALE = 0.50

    @staticmethod
    def _v14_continuous_floor_scale(
        value: float,
        start: float,
        full: float,
        minimum: float,
    ) -> float:
        progress = np.clip(
            (float(value) - float(start))
            / max(float(full) - float(start), 1e-12),
            0.0,
            1.0,
        )
        return 1.0 - float(progress) * (1.0 - float(minimum))

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        component = self._component(entry_tag)
        if stake <= 0.0 or component not in {"breakout", "grid"}:
            return stake
        row = self._latest_signal_row(pair, component, current_time)
        if row is None:
            return stake
        scale = 1.0
        if component == "breakout" and side == "long":
            scale = self._v14_continuous_floor_scale(
                float(row.get("symbol_return_4h", 0.0)),
                self.V14_BO_LONG_RISK_START_RETURN_4H,
                self.V14_BO_LONG_RISK_FULL_RETURN_4H,
                self.V14_BO_LONG_RISK_MIN_SCALE,
            )
        elif component == "grid" and side == "short":
            decline = max(0.0, -float(row.get("symbol_return_4h", 0.0)))
            scale = self._v14_continuous_floor_scale(
                decline,
                self.V14_GRID_SHORT_RISK_START_DECLINE_4H,
                self.V14_GRID_SHORT_RISK_FULL_DECLINE_4H,
                self.V14_GRID_SHORT_RISK_MIN_SCALE,
            )
        elif component == "grid" and side == "long":
            scale = self._v14_continuous_floor_scale(
                float(row.get("market_efficiency", 0.0)),
                self.V14_GRID_LONG_RISK_START_EFFICIENCY,
                self.V14_GRID_LONG_RISK_FULL_EFFICIENCY,
                self.V14_GRID_LONG_RISK_MIN_SCALE,
            )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14ChopBreakoutRiskMixin:
    """Continuously de-risk unconfirmed shorts in persistent low opportunity.

    The 72h persistence input acts as a causal time confirmation: one noisy
    candle cannot flip the allocation.  Breakout remains executable and keeps
    its Max2 priority, while risk is reduced only when the market has stayed
    compressed and BTC lacks a confirming daily downtrend.
    """

    V14_CHOP_RISK_START_PERSISTENCE = 0.55
    V14_CHOP_RISK_FULL_PERSISTENCE = 0.85
    V14_CHOP_SHORT_FULL_CONFIRM_TREND_1D = -0.10
    V14_CHOP_SHORT_NO_CONFIRM_TREND_1D = 0.10
    V14_CHOP_SHORT_MIN_SCALE = 0.35

    def _v14_chop_short_scale(
        self,
        persistence: float,
        btc_trend_1d: float,
    ) -> float:
        active = np.clip(
            (
                float(persistence)
                - self.V14_CHOP_RISK_START_PERSISTENCE
            )
            / max(
                self.V14_CHOP_RISK_FULL_PERSISTENCE
                - self.V14_CHOP_RISK_START_PERSISTENCE,
                1e-12,
            ),
            0.0,
            1.0,
        )
        unconfirmed = np.clip(
            (
                float(btc_trend_1d)
                - self.V14_CHOP_SHORT_FULL_CONFIRM_TREND_1D
            )
            / max(
                self.V14_CHOP_SHORT_NO_CONFIRM_TREND_1D
                - self.V14_CHOP_SHORT_FULL_CONFIRM_TREND_1D,
                1e-12,
            ),
            0.0,
            1.0,
        )
        return 1.0 - float(active * unconfirmed) * (
            1.0 - self.V14_CHOP_SHORT_MIN_SCALE
        )

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "breakout"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        scale = self._v14_chop_short_scale(
            float(row.get("low_opportunity_fraction_72h", 0.0)),
            float(row.get("btc_trend_1d", 0.0)),
        )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14ContinuousPortfolioGovernorMixin:
    """Replace the 20% risk cliff with a persistent smooth state transition.

    The inherited recovery evidence still requires eight completed campaigns,
    which supplies hysteresis across noisy fills.  Drawdown activation uses a
    smoothstep ramp, so one unit of quantity rounding or slippage cannot jump
    the whole account directly from full risk to the defensive floor.
    """

    V14_PORTFOLIO_RISK_START_DRAWDOWN = 0.17
    V14_PORTFOLIO_RISK_FULL_DRAWDOWN = 0.20

    @staticmethod
    def _v14_smoothstep(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return bounded * bounded * (3.0 - 2.0 * bounded)

    def _portfolio_risk_scale(
        self,
        equity: float,
        current_time: datetime,
    ) -> float:
        if equity <= 0.0 or self._peak_equity <= 0.0:
            return 0.0
        drawdown = max(0.0, 1.0 - float(equity) / float(self._peak_equity))
        span = max(
            self.V14_PORTFOLIO_RISK_FULL_DRAWDOWN
            - self.V14_PORTFOLIO_RISK_START_DRAWDOWN,
            1e-12,
        )
        activation = self._v14_smoothstep(
            (drawdown - self.V14_PORTFOLIO_RISK_START_DRAWDOWN) / span
        )
        if activation <= 0.0:
            return 1.0

        recent = self._recent_portfolio_returns(current_time)
        target = self.PORTFOLIO_DEFENSIVE_SCALE
        if len(recent) >= self.PORTFOLIO_RECENT_WINDOW:
            wins = sum(value > 0.0 for value in recent)
            aggregate_return = float(sum(recent))
            if (
                wins >= self.PORTFOLIO_FULL_RISK_MIN_WINS
                and aggregate_return >= self.PORTFOLIO_FULL_RISK_MIN_RETURN
            ):
                target = 1.0
            elif (
                wins >= self.PORTFOLIO_RECOVERY_MIN_WINS
                and aggregate_return >= self.PORTFOLIO_RECOVERY_MIN_RETURN
            ):
                target = self.PORTFOLIO_RECOVERY_SCALE
        return 1.0 - activation * (1.0 - float(target))


class _V14NarrowContinuousPortfolioGovernorMixin(
    _V14ContinuousPortfolioGovernorMixin,
):
    """Smooth the frozen 18% boundary without moving its economic center.

    The earlier wide 17%..20% experiment materially changed otherwise healthy
    trend paths.  This version keeps the inherited 18% decision center and
    only replaces the discontinuity with a one-percentage-point smoothstep.
    Recovery still requires the same eight completed campaigns.
    """

    V14_PORTFOLIO_RISK_START_DRAWDOWN = 0.175
    V14_PORTFOLIO_RISK_FULL_DRAWDOWN = 0.185


class _V14BenchmarkConfirmedBreakoutLongRiskMixin:
    """Continuously size Breakout longs by completed BTC 4h confirmation.

    The target symbol may lead a flat benchmark at full risk.  Compression
    starts only when BTC's normalized 4h trend is actually negative, which
    separated losing long attempts from the large 2024 and 2026 winners in
    both dynamic-universe audits.  Signal timing and Max2 priority are kept.
    """

    V14_BO_LONG_BTC_TREND_MIN = -0.25
    V14_BO_LONG_BTC_TREND_FULL = 0.0
    V14_BO_LONG_BTC_TREND_MIN_SCALE = 0.35

    def _v14_benchmark_long_scale(self, btc_trend_4h: float) -> float:
        progress = np.clip(
            (
                float(btc_trend_4h)
                - self.V14_BO_LONG_BTC_TREND_MIN
            )
            / max(
                self.V14_BO_LONG_BTC_TREND_FULL
                - self.V14_BO_LONG_BTC_TREND_MIN,
                1e-12,
            ),
            0.0,
            1.0,
        )
        smooth = float(progress * progress * (3.0 - 2.0 * progress))
        return self.V14_BO_LONG_BTC_TREND_MIN_SCALE + smooth * (
            1.0 - self.V14_BO_LONG_BTC_TREND_MIN_SCALE
        )

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "breakout"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "breakout", current_time)
        if row is None:
            return stake
        scale = self._v14_benchmark_long_scale(
            float(row.get("btc_trend_4h", 0.0))
        )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14CrowdedShallowGridShortRiskMixin:
    """Compress a crowded Grid short before a meaningful rebound develops.

    High participation is useful only after price has pulled far enough from
    the fast trend.  A shallow extension with rising volume was the same
    adverse structure in the dynamic 2024 and 2026 audits.  Two smooth axes
    avoid a new volume or extension cliff while leaving ordinary Grid shorts
    and every signal timestamp unchanged.
    """

    V14_GRID_CROWD_VOLUME_START = 1.40
    V14_GRID_CROWD_VOLUME_FULL = 1.80
    V14_GRID_CROWD_EXTENSION_SAFE = 0.25
    V14_GRID_CROWD_EXTENSION_FULL = 0.10
    V14_GRID_CROWD_MIN_SCALE = 0.25

    @staticmethod
    def _v14_axis_smoothstep(progress: float) -> float:
        bounded = float(np.clip(progress, 0.0, 1.0))
        return bounded * bounded * (3.0 - 2.0 * bounded)

    def _v14_crowded_grid_short_scale(
        self,
        volume_ratio: float,
        extension: float,
    ) -> float:
        volume_activation = self._v14_axis_smoothstep(
            (
                float(volume_ratio)
                - self.V14_GRID_CROWD_VOLUME_START
            )
            / max(
                self.V14_GRID_CROWD_VOLUME_FULL
                - self.V14_GRID_CROWD_VOLUME_START,
                1e-12,
            )
        )
        shallow_activation = self._v14_axis_smoothstep(
            (
                self.V14_GRID_CROWD_EXTENSION_SAFE
                - float(extension)
            )
            / max(
                self.V14_GRID_CROWD_EXTENSION_SAFE
                - self.V14_GRID_CROWD_EXTENSION_FULL,
                1e-12,
            )
        )
        activation = volume_activation * shallow_activation
        return 1.0 - activation * (1.0 - self.V14_GRID_CROWD_MIN_SCALE)

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        scale = self._v14_crowded_grid_short_scale(
            float(row.get("grid_volume_ratio", 0.0)),
            float(row.get("grid_extension", np.inf)),
        )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14GridFalsePullbackContinuousRiskMixin:
    """Continuously reduce Grid-long exposure as the alleged pullback vanishes.

    Keeping the signal preserves Max2 campaign timing and subsequent Breakout
    opportunities.  Position risk falls smoothly from full size at a genuine
    non-positive 4h return to a floor once the symbol has already rebounded by
    half a percent, avoiding both a binary entry gate and a sizing cliff.
    """

    V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H = 0.005
    V14_GRID_PULLBACK_MIN_SCALE = 0.20

    def _v14_grid_pullback_scale(self, return_4h: float) -> float:
        rebound = max(0.0, float(return_4h))
        progress = np.clip(
            rebound
            / max(self.V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H, 1e-12),
            0.0,
            1.0,
        )
        return 1.0 - float(progress) * (
            1.0 - self.V14_GRID_PULLBACK_MIN_SCALE
        )

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "long"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None:
            return stake
        scale = self._v14_grid_pullback_scale(
            float(row.get("symbol_return_4h", 0.0))
        )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14SmoothGridConflictRiskMixin:
    """Replace V13's 3% binary Grid-risk gate with a continuous ramp."""

    # Disable the inherited binary activation.  The feature conflict itself
    # is kept byte-for-byte below; only its drawdown activation is smoothed.
    V13_GRID_SQUEEZE_MIN_DRAWDOWN = 1.0
    V14_GRID_CONFLICT_START_DRAWDOWN = 0.02
    V14_GRID_CONFLICT_FULL_DRAWDOWN = 0.04
    V14_GRID_CONFLICT_MIN_SCALE = 0.20

    def _v14_grid_conflict(self, row: Series) -> bool:
        base_conflict = (
            float(row.get("symbol_return_4h", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_SYMBOL_RETURN_4H
            and float(row.get("market_regime_score", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_MARKET_REGIME
            and float(row.get("breadth", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_BREADTH
        )
        broad_contraction = (
            float(row.get("breadth_change_24h", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_BREADTH_CHANGE_24H
        )
        crowded_rebound = (
            float(row.get("grid_volume_ratio", -np.inf))
            >= self.V13_GRID_SQUEEZE_MIN_VOLUME_RATIO
        )
        low_score_conflict = (
            float(row.get("grid_score", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_SCORE
            and (broad_contraction or crowded_rebound)
        )
        weak_structure_conflict = (
            self.V13_GRID_SQUEEZE_ENABLE_WEAK_STRUCTURE
            and float(row.get("grid_alignment", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_ALIGNMENT
            and float(row.get("grid_extension", np.inf))
            <= self.V13_GRID_SQUEEZE_MAX_EXTENSION
        )
        return bool(
            base_conflict
            and (low_score_conflict or weak_structure_conflict)
        )

    def _v14_grid_conflict_drawdown_scale(self, drawdown: float) -> float:
        span = max(
            self.V14_GRID_CONFLICT_FULL_DRAWDOWN
            - self.V14_GRID_CONFLICT_START_DRAWDOWN,
            1e-12,
        )
        activation = np.clip(
            (float(drawdown) - self.V14_GRID_CONFLICT_START_DRAWDOWN) / span,
            0.0,
            1.0,
        )
        return 1.0 - float(activation) * (
            1.0 - self.V14_GRID_CONFLICT_MIN_SCALE
        )

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if (
            stake <= 0.0
            or self._component(entry_tag) != "grid"
            or side != "short"
        ):
            return stake
        row = self._latest_signal_row(pair, "grid", current_time)
        if row is None or not self._v14_grid_conflict(row):
            return stake
        equity = float(self.wallets.get_total_stake_amount())
        peak = max(float(getattr(self, "_peak_equity", 0.0)), equity)
        drawdown = 0.0 if peak <= 0.0 else max(0.0, 1.0 - equity / peak)
        scale = self._v14_grid_conflict_drawdown_scale(drawdown)
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14NarrowSmoothGridConflictRiskMixin(
    _V14SmoothGridConflictRiskMixin,
):
    """Smooth the selected Grid 3% risk boundary around the same center."""

    V14_GRID_CONFLICT_START_DRAWDOWN = 0.0275
    V14_GRID_CONFLICT_FULL_DRAWDOWN = 0.0325


class _V14GridSleeveHighWaterMixin:
    """Use a Grid-only realized-PnL high-water with a continuous risk curve.

    The ledger is reconstructed from completed Grid campaigns on every sizing
    decision.  It is therefore deterministic in backtest and restart-safe in
    LIVE without another mutable state file.  Breakout profits cannot silently
    re-arm Grid risk, and one Grid loss cannot change Breakout allocation.
    """

    V14_GRID_LEDGER_START_SHARE = 0.50
    V14_GRID_LEDGER_RISK_START_DRAWDOWN = 0.04
    V14_GRID_LEDGER_RISK_MID_DRAWDOWN = 0.14
    V14_GRID_LEDGER_RISK_FULL_DRAWDOWN = 0.25
    V14_GRID_LEDGER_MID_SCALE = 0.45
    V14_GRID_LEDGER_MIN_SCALE = 0.20

    def _v14_grid_ledger_drawdown(self, current_time: datetime) -> float:
        starting = float(self.wallets.get_starting_balance()) * (
            self.V14_GRID_LEDGER_START_SHARE
        )
        if starting <= 0.0:
            return 1.0
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        completed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if self._component(getattr(trade, "enter_tag", None)) != "grid":
                continue
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            if close_time is None or close_time > boundary or profit is None:
                continue
            value = float(profit)
            if np.isfinite(value):
                completed.append((close_time, value))
        completed.sort(key=lambda item: item[0])
        equity = starting
        peak = starting
        for _close_time, profit in completed:
            equity += profit
            peak = max(peak, equity)
        return 1.0 if equity <= 0.0 else max(0.0, 1.0 - equity / peak)

    def _v14_grid_ledger_risk_scale(self, drawdown: float) -> float:
        value = max(0.0, float(drawdown))
        start = self.V14_GRID_LEDGER_RISK_START_DRAWDOWN
        middle = self.V14_GRID_LEDGER_RISK_MID_DRAWDOWN
        full = self.V14_GRID_LEDGER_RISK_FULL_DRAWDOWN
        if value <= start:
            return 1.0
        if value <= middle:
            progress = (value - start) / max(middle - start, 1e-12)
            return 1.0 - progress * (1.0 - self.V14_GRID_LEDGER_MID_SCALE)
        if value <= full:
            progress = (value - middle) / max(full - middle, 1e-12)
            return self.V14_GRID_LEDGER_MID_SCALE - progress * (
                self.V14_GRID_LEDGER_MID_SCALE - self.V14_GRID_LEDGER_MIN_SCALE
            )
        return self.V14_GRID_LEDGER_MIN_SCALE

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
        stake = super().custom_stake_amount(
            pair,
            current_time,
            current_rate,
            proposed_stake,
            min_stake,
            max_stake,
            leverage,
            entry_tag,
            side,
            **kwargs,
        )
        if stake <= 0.0 or self._component(entry_tag) != "grid":
            return stake
        scale = self._v14_grid_ledger_risk_scale(
            self._v14_grid_ledger_drawdown(current_time)
        )
        scaled = min(float(max_stake), max(0.0, float(stake) * scale))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled


class _V14RangePersistenceMixin:
    """Keep range fades out of broad, fast rebounds.

    A statistical band touch is not sufficient when participation is rapidly
    expanding.  These completed-candle checks describe a persistent range:
    neither 4h nor 24h breadth is accelerating upward and the rejecting candle
    has normal participation.  The bounds are coarse and directly reproducible
    by LIVE from the same synchronized 1h market context.
    """

    V14_RANGE_MAX_BREADTH_CHANGE_4H = 0.20
    V14_RANGE_MAX_BREADTH_CHANGE_24H = 0.20
    V14_RANGE_SHORT_MIN_VOLUME_RATIO = 0.85

    def _populate_range(self, dataframe: DataFrame) -> DataFrame:
        dataframe = super()._populate_range(dataframe)
        unstable_short = dataframe["range_entry_short"].astype(bool) & ~(
            (
                dataframe["breadth_change_4h"]
                <= self.V14_RANGE_MAX_BREADTH_CHANGE_4H
            )
            & (
                dataframe["breadth_change_24h"]
                <= self.V14_RANGE_MAX_BREADTH_CHANGE_24H
            )
            & (
                dataframe["grid_volume_ratio"]
                >= self.V14_RANGE_SHORT_MIN_VOLUME_RATIO
            )
        )
        dataframe["range_v14_persistence_rejected"] = unstable_short.astype(int)
        dataframe.loc[unstable_short, "range_entry_short"] = 0
        dataframe["range_entry"] = (
            dataframe["range_entry_long"].astype(bool)
            | dataframe["range_entry_short"].astype(bool)
        ).astype(int)
        return dataframe


class _V14RangePrimaryIsolationMixin:
    """Keep an auxiliary Range fill out of primary risk-state calculations.

    The shared wallet still enforces real available capital and Max2.  Only
    Breakout/Grid's risk numerator, drawdown high-water, and recent-trade
    governor see a virtual primary equity with realized Range PnL removed.
    This prevents a tiny auxiliary fill from changing every later campaign
    through quantity rounding and Grid campaign state.
    """

    @staticmethod
    def _v14_primary_virtual_equity(
        actual_equity: float,
        closed_range_profit: float,
        tradable_balance_ratio: float,
    ) -> float:
        return max(
            0.0,
            float(actual_equity)
            - float(closed_range_profit) * float(tradable_balance_ratio),
        )

    def _v14_closed_range_profit(self, current_time: datetime) -> float:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        total = 0.0
        for trade in Trade.get_trades_proxy(is_open=False):
            if self._component(getattr(trade, "enter_tag", None)) != "range":
                continue
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            if close_time is None or close_time > boundary or profit is None:
                continue
            value = float(profit)
            if np.isfinite(value):
                total += value
        return total

    def _v14_primary_closed_returns(
        self,
        current_time: datetime,
        side: str | None = None,
    ) -> list[tuple[datetime, float]]:
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        target_short = side == "short" if side is not None else None
        closed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if self._component(getattr(trade, "enter_tag", None)) == "range":
                continue
            if (
                target_short is not None
                and bool(getattr(trade, "is_short", False)) != target_short
            ):
                continue
            close_time = self._trade_close_time(trade)
            value = self._closed_trade_return(trade)
            if close_time is None or close_time > boundary or value is None:
                continue
            closed.append((close_time, value))
        closed.sort(key=lambda item: item[0])
        return closed

    def _recent_portfolio_returns(
        self,
        current_time: datetime,
    ) -> tuple[float, ...]:
        closed = self._v14_primary_closed_returns(current_time)
        return tuple(
            value
            for _close_time, value in closed[-self.PORTFOLIO_RECENT_WINDOW :]
        )

    def _recent_side_returns(
        self,
        current_time: datetime,
        side: str,
        limit: int = 3,
    ) -> tuple[float, ...]:
        closed = self._v14_primary_closed_returns(current_time, side)
        return tuple(value for _close_time, value in closed[-limit:])

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
        if self._component(entry_tag) == "range":
            return super().custom_stake_amount(
                pair,
                current_time,
                current_rate,
                proposed_stake,
                min_stake,
                max_stake,
                leverage,
                entry_tag,
                side,
                **kwargs,
            )
        actual_wallets = self.wallets
        actual_equity = float(actual_wallets.get_total_stake_amount())
        virtual_equity = self._v14_primary_virtual_equity(
            actual_equity,
            self._v14_closed_range_profit(current_time),
            float(self.config.get("tradable_balance_ratio", 1.0)),
        )
        self.wallets = _V14WalletEquityView(actual_wallets, virtual_equity)
        try:
            return super().custom_stake_amount(
                pair,
                current_time,
                current_rate,
                proposed_stake,
                min_stake,
                max_stake,
                leverage,
                entry_tag,
                side,
                **kwargs,
            )
        finally:
            self.wallets = actual_wallets


class _V14RangeSleeveMixin:
    """Causal, live-implementable mean-reversion sleeve for choppy regimes.

    The sleeve has no DCA and no partial exits.  It may replace the Grid
    tactical slot only when Grid has no executable candidate.  Breakout and
    Grid always retain priority, so a range signal can use genuine spare
    Max2 capacity but cannot change which primary campaign is accepted.  All
    market inputs are inherited from the synchronized, completed-1h V13
    context.
    """

    RANGE_ONLY = False
    RANGE_VERSION = "range_v14_chop_reversion_research_20260809"

    RANGE_WINDOW = 24
    RANGE_DEVIATION = 1.50
    RANGE_MIN_BAND_ATR = 1.15
    RANGE_MAX_BAND_ATR = 2.50
    RANGE_MAX_ABS_REGIME = 1.00
    RANGE_MIN_MARKET_EFFICIENCY = 0.08
    RANGE_MAX_MARKET_EFFICIENCY = 0.28
    RANGE_MIN_DISPERSION_7D = 0.012
    RANGE_MAX_DISPERSION_7D = 0.044
    RANGE_MIN_BREADTH = 0.10
    RANGE_MAX_BREADTH = 0.85
    RANGE_MAX_ABS_EMA_SLOPE_ATR = 0.25
    RANGE_MIN_VOLUME_RATIO = 0.55
    RANGE_MAX_VOLUME_RATIO = 1.35
    RANGE_LONG_MAX_RSI = 40.0
    RANGE_SHORT_MIN_RSI = 55.0
    RANGE_LONG_MIN_REGIME = 0.00
    RANGE_LONG_MIN_BREADTH = 0.55
    RANGE_LONG_MIN_BTC_TREND_1D = -0.10
    RANGE_SHORT_MAX_REGIME = -0.15
    RANGE_SHORT_MAX_BREADTH = 0.60
    RANGE_SHORT_MAX_BTC_TREND_4H = 0.20
    RANGE_SHORT_MAX_BTC_TREND_1D = 0.20
    # A mean-reversion short needs actual extension above the slow trend.
    # Without this guard, a candle that merely wicks into the statistical
    # band while sitting close to EMA55 is a weak rebound, not a range edge.
    RANGE_SHORT_MIN_SYMBOL_EMA55_ATR = 0.40
    RANGE_SHORT_ORDERLY_EFFICIENCY = 0.18
    RANGE_SHORT_CONFIRMING_VOLUME = 0.90
    RANGE_MIN_CLOSE_POSITION = 0.55
    RANGE_MAX_CLOSE_POSITION = 0.45
    RANGE_DAILY_LIMIT = 3
    RANGE_MINIMUM_BAR_GAP = 6

    RANGE_RISK_PCT = 0.018
    RANGE_MAX_NOTIONAL = 2.50
    RANGE_STOP_ATR = 1.20
    RANGE_MIN_TARGET_ATR = 0.55
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.80
    RANGE_BREAKEVEN_LOCK_ATR = 0.08
    RANGE_MAX_HOLD_HOURS = 18
    RANGE_EARLY_FAILURE_HOURS = 5
    RANGE_FAILURE_REGIME = 0.34
    RANGE_FAILURE_SLOPE_ATR = 0.75
    RANGE_LEDGER_START_SHARE = 0.10
    RANGE_LEDGER_RISK_START_DRAWDOWN = 0.05
    RANGE_LEDGER_RISK_FULL_DRAWDOWN = 0.25
    RANGE_LEDGER_MIN_SCALE = 0.30

    @staticmethod
    def _component(tag: str | None) -> str:
        value = str(tag or "")
        if value.startswith("range_v14"):
            return "range"
        return BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade._component(
            tag
        )

    def _populate_range(self, dataframe: DataFrame) -> DataFrame:
        close = dataframe["close"].astype(float)
        atr = dataframe["atr"].clip(lower=1e-12)
        midpoint = close.rolling(
            self.RANGE_WINDOW,
            min_periods=self.RANGE_WINDOW,
        ).mean()
        deviation = close.rolling(
            self.RANGE_WINDOW,
            min_periods=self.RANGE_WINDOW,
        ).std(ddof=0)
        lower = midpoint - self.RANGE_DEVIATION * deviation
        upper = midpoint + self.RANGE_DEVIATION * deviation
        candle_range = (dataframe["high"] - dataframe["low"]).clip(
            lower=1e-12
        )

        dataframe["range_mid"] = midpoint
        dataframe["range_lower"] = lower
        dataframe["range_upper"] = upper
        dataframe["range_z"] = (close - midpoint) / deviation.clip(
            lower=1e-12
        )
        dataframe["range_band_atr"] = (
            2.0 * self.RANGE_DEVIATION * deviation / atr
        )
        dataframe["range_rsi"] = _rsi(close, 14)
        dataframe["range_close_position"] = (
            close - dataframe["low"]
        ) / candle_range
        dataframe["range_ema_slope_atr"] = (
            dataframe["fast_ema"] - dataframe["fast_ema"].shift(6)
        ) / atr

        ready_columns = (
            "market_regime_score",
            "market_efficiency",
            "market_dispersion_7d",
            "breadth",
            "grid_volume_ratio",
        )
        context_ready = dataframe[list(ready_columns)].notna().all(axis=1)
        range_regime = (
            context_ready
            & dataframe["market_regime_score"].abs().le(
                self.RANGE_MAX_ABS_REGIME
            )
            & dataframe["market_efficiency"].between(
                self.RANGE_MIN_MARKET_EFFICIENCY,
                self.RANGE_MAX_MARKET_EFFICIENCY,
            )
            & dataframe["market_dispersion_7d"].between(
                self.RANGE_MIN_DISPERSION_7D,
                self.RANGE_MAX_DISPERSION_7D,
            )
            & dataframe["breadth"].between(
                self.RANGE_MIN_BREADTH,
                self.RANGE_MAX_BREADTH,
            )
            & dataframe["range_band_atr"].between(
                self.RANGE_MIN_BAND_ATR,
                self.RANGE_MAX_BAND_ATR,
            )
            & dataframe["range_ema_slope_atr"].abs().le(
                self.RANGE_MAX_ABS_EMA_SLOPE_ATR
            )
            & dataframe["grid_volume_ratio"].between(
                self.RANGE_MIN_VOLUME_RATIO,
                self.RANGE_MAX_VOLUME_RATIO,
            )
        )
        dataframe["range_regime"] = range_regime.astype(int)

        long_rejection = (
            (dataframe["low"] <= lower)
            & (close > lower)
            & (dataframe["range_close_position"] >= self.RANGE_MIN_CLOSE_POSITION)
            & (close > dataframe["open"])
            & (dataframe["range_rsi"] <= self.RANGE_LONG_MAX_RSI)
            & (dataframe["range_rsi"] > dataframe["range_rsi"].shift(1))
            & (
                dataframe["market_regime_score"]
                >= self.RANGE_LONG_MIN_REGIME
            )
            & (dataframe["breadth"] >= self.RANGE_LONG_MIN_BREADTH)
            & (
                dataframe["btc_trend_1d"]
                >= self.RANGE_LONG_MIN_BTC_TREND_1D
            )
        )
        short_rejection = (
            (dataframe["high"] >= upper)
            & (close < upper)
            & (dataframe["range_close_position"] <= self.RANGE_MAX_CLOSE_POSITION)
            & (close < dataframe["open"])
            & (dataframe["range_rsi"] >= self.RANGE_SHORT_MIN_RSI)
            & (dataframe["range_rsi"] < dataframe["range_rsi"].shift(1))
            & (
                dataframe["market_regime_score"]
                <= self.RANGE_SHORT_MAX_REGIME
            )
            & (dataframe["breadth"] <= self.RANGE_SHORT_MAX_BREADTH)
            & (
                dataframe["btc_trend_4h"]
                <= self.RANGE_SHORT_MAX_BTC_TREND_4H
            )
            & (
                dataframe["btc_trend_1d"]
                <= self.RANGE_SHORT_MAX_BTC_TREND_1D
            )
            & (
                dataframe["symbol_ema55_atr"]
                >= self.RANGE_SHORT_MIN_SYMBOL_EMA55_ATR
            )
            & (
                (
                    dataframe["market_efficiency"]
                    >= self.RANGE_SHORT_ORDERLY_EFFICIENCY
                )
                | (
                    dataframe["grid_volume_ratio"]
                    >= self.RANGE_SHORT_CONFIRMING_VOLUME
                )
            )
        )
        long_entry = _causal_signal_cap(
            range_regime & long_rejection,
            dataframe["date"],
            daily_limit=self.RANGE_DAILY_LIMIT,
            minimum_bar_gap=self.RANGE_MINIMUM_BAR_GAP,
        ).astype(bool)
        short_entry = _causal_signal_cap(
            range_regime & short_rejection,
            dataframe["date"],
            daily_limit=self.RANGE_DAILY_LIMIT,
            minimum_bar_gap=self.RANGE_MINIMUM_BAR_GAP,
        ).astype(bool)

        dataframe["range_entry_long"] = long_entry.astype(int)
        dataframe["range_entry_short"] = short_entry.astype(int)
        dataframe["range_entry"] = (long_entry | short_entry).astype(int)
        deviation_score = dataframe["range_z"].abs().clip(upper=3.0)
        recovery_score = (
            (dataframe["range_close_position"] - 0.5).abs() * 2.0
        ).clip(upper=1.0)
        neutral_score = (
            1.0
            - dataframe["market_regime_score"].abs()
            / max(self.RANGE_MAX_ABS_REGIME, 1e-12)
        ).clip(lower=0.0, upper=1.0)
        dataframe["range_quality"] = (
            0.55 * deviation_score
            + 0.25 * recovery_score
            + 0.20 * neutral_score
        )
        dataframe["range_rank"] = -dataframe["range_quality"]
        return dataframe

    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        return self._populate_range(dataframe)

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        range_long = dataframe["range_entry_long"].astype(bool)
        range_short = dataframe["range_entry_short"].astype(bool)

        if self.RANGE_ONLY:
            dataframe["enter_long"] = 0
            dataframe["enter_short"] = 0
            dataframe["enter_tag"] = None
            available = pd.Series(True, index=dataframe.index)
        else:
            available = ~(
                dataframe["enter_long"].astype(bool)
                | dataframe["enter_short"].astype(bool)
            )
        selected_long = range_long & available
        selected_short = range_short & available
        dataframe.loc[selected_long, "enter_long"] = 1
        dataframe.loc[selected_short, "enter_short"] = 1
        for index in dataframe.index[selected_long]:
            quality = int(round(float(dataframe.at[index, "range_quality"]) * 100))
            dataframe.at[index, "enter_tag"] = f"range_v14_long_q{quality}"
        for index in dataframe.index[selected_short]:
            quality = int(round(float(dataframe.at[index, "range_quality"]) * 100))
            dataframe.at[index, "enter_tag"] = f"range_v14_short_q{quality}"

        pair = str(metadata.get("pair") or "")
        if pair:
            ranks = {} if self.RANGE_ONLY else dict(
                self._candidate_ranks.get(pair, {})
            )
            available_times = (
                pd.to_datetime(dataframe["date"], utc=True)
                + pd.Timedelta(self.timeframe)
            )
            for index in dataframe.index[selected_long | selected_short]:
                ranks[("range", int(available_times.at[index].timestamp()))] = (
                    float(dataframe.at[index, "range_rank"])
                )
            self._candidate_ranks[pair] = ranks
        return dataframe

    def _latest_signal_row(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> Series | None:
        if component != "range":
            return super()._latest_signal_row(pair, component, current_time)
        row = self._latest_row(pair, current_time)
        if row is None:
            return None
        return row if int(row.get("range_entry", 0) or 0) == 1 else None

    def _component_limits_allow(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> bool:
        trades = Trade.get_trades_proxy()
        if component in {"grid", "range"} and any(
            trade.is_open
            and self._component(getattr(trade, "enter_tag", None))
            in {"grid", "range"}
            for trade in trades
        ):
            return False
        if component != "range":
            return super()._component_limits_allow(
                pair,
                component,
                current_time,
            )

        for trade in trades:
            if trade.is_open or trade.pair != pair:
                continue
            if self._component(getattr(trade, "enter_tag", None)) != "range":
                continue
            close_time = self._trade_close_time(trade)
            if close_time is not None and current_time < close_time + timedelta(
                hours=6
            ):
                return False
        day = current_time.astimezone(timezone.utc).date()
        entries_today = sum(
            1
            for trade in trades
            if self._component(getattr(trade, "enter_tag", None)) == "range"
            and self._trade_open_time(trade).date() == day
        )
        return entries_today < self.RANGE_DAILY_LIMIT

    @staticmethod
    def _v14_range_has_spare_slot(
        open_components: tuple[str, ...],
        max_open_trades: int,
        breakout_candidate: bool,
        grid_candidate: bool,
    ) -> bool:
        """Return whether Range can enter without displacing primary work."""

        if grid_candidate or any(
            component in {"grid", "range"}
            for component in open_components
        ):
            return False
        remaining = int(max_open_trades) - len(open_components)
        if remaining <= 0:
            return False
        breakout_reservation = int(
            breakout_candidate and "breakout" not in open_components
        )
        return remaining > breakout_reservation

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
        if component != "range":
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
        if not self._component_limits_allow(pair, component, current_time):
            return False
        open_trades = Trade.get_trades_proxy(is_open=True)
        open_components = tuple(
            self._component(getattr(trade, "enter_tag", None))
            for trade in open_trades
        )
        if not self._v14_range_has_spare_slot(
            open_components,
            self.MAX_OPEN_TRADES,
            bool(self._ranked_candidates("breakout", current_time)),
            bool(self._ranked_candidates("grid", current_time)),
        ):
            return False
        ranked = self._ranked_candidates("range", current_time)
        return bool(ranked) and ranked[0][1] == pair

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
        if self._component(entry_tag) != "range":
            return super().custom_stake_amount(
                pair,
                current_time,
                current_rate,
                proposed_stake,
                min_stake,
                max_stake,
                leverage,
                entry_tag,
                side,
                **kwargs,
            )
        row = self._latest_signal_row(pair, "range", current_time)
        if row is None or current_rate <= 0.0:
            return 0.0
        equity = float(self.wallets.get_total_stake_amount())
        starting = float(self.wallets.get_starting_balance()) * (
            self.RANGE_LEDGER_START_SHARE
        )
        range_equity = starting
        range_peak = starting
        boundary = current_time
        if boundary.tzinfo is None:
            boundary = boundary.replace(tzinfo=timezone.utc)
        completed: list[tuple[datetime, float]] = []
        for trade in Trade.get_trades_proxy(is_open=False):
            if self._component(getattr(trade, "enter_tag", None)) != "range":
                continue
            close_time = self._trade_close_time(trade)
            profit = getattr(trade, "close_profit_abs", None)
            if close_time is None or close_time > boundary or profit is None:
                continue
            value = float(profit)
            if np.isfinite(value):
                completed.append((close_time, value))
        completed.sort(key=lambda item: item[0])
        for _close_time, profit in completed:
            range_equity += profit
            range_peak = max(range_peak, range_equity)
        if equity <= 0.0 or range_equity <= 0.0:
            return 0.0
        range_drawdown = max(0.0, 1.0 - range_equity / range_peak)
        progress = np.clip(
            (
                range_drawdown
                - self.RANGE_LEDGER_RISK_START_DRAWDOWN
            )
            / max(
                self.RANGE_LEDGER_RISK_FULL_DRAWDOWN
                - self.RANGE_LEDGER_RISK_START_DRAWDOWN,
                1e-12,
            ),
            0.0,
            1.0,
        )
        portfolio_scale = 1.0 - float(progress) * (
            1.0 - self.RANGE_LEDGER_MIN_SCALE
        )
        atr_value = max(float(row.get("atr", 0.0)), 1e-12)
        unit_loss_ratio = (
            self.RANGE_STOP_ATR * atr_value / current_rate
            + 2.0 * self.SIDE_COST
        )
        notional = equity * self.RANGE_RISK_PCT / max(unit_loss_ratio, 1e-12)
        notional = min(notional, equity * self.RANGE_MAX_NOTIONAL)
        stake = notional / max(1.0, float(leverage)) * portfolio_scale
        mode = self._adaptive_state_mode()
        if mode is not None:
            self._persist_adaptive_high_water(mode, current_time)
        scaled = min(float(max_stake), max(0.0, float(stake)))
        if min_stake is not None and scaled < float(min_stake):
            return 0.0
        return scaled

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: Order,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        if self._component(getattr(trade, "enter_tag", None)) != "range":
            super().order_filled(pair, trade, order, current_time, **kwargs)
            return
        initial_entry = (
            getattr(order, "ft_order_side", None)
            == getattr(trade, "entry_side", None)
            and int(getattr(trade, "nr_of_successful_entries", 0)) == 1
        )
        if not initial_entry:
            return
        row = self._latest_signal_row(pair, "range", current_time)
        if row is None or not self._claim_fill_state_update(trade, order):
            return
        atr_value = max(float(row.get("atr", 0.0)), 1e-12)
        side = -1.0 if trade.is_short else 1.0
        minimum_target = float(trade.open_rate) + side * (
            self.RANGE_MIN_TARGET_ATR * atr_value
        )
        midpoint = float(row.get("range_mid", trade.open_rate))
        target = (
            min(midpoint, minimum_target)
            if trade.is_short
            else max(midpoint, minimum_target)
        )
        trade.set_custom_data("range_initial_atr", atr_value)
        trade.set_custom_data("range_target", target)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs: Any,
    ) -> float | None:
        if self._component(getattr(trade, "enter_tag", None)) != "range":
            return super().custom_stoploss(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
                after_fill,
                **kwargs,
            )
        atr_value = max(
            float(trade.get_custom_data("range_initial_atr", 0.0)),
            1e-12,
        )
        side = -1.0 if trade.is_short else 1.0
        stop_rate = float(trade.open_rate) - side * self.RANGE_STOP_ATR * atr_value
        favorable_value = trade.min_rate if trade.is_short else trade.max_rate
        favorable_rate = float(
            current_rate if favorable_value is None else favorable_value
        )
        favorable_atr = side * (favorable_rate - float(trade.open_rate)) / atr_value
        if favorable_atr >= self.RANGE_BREAKEVEN_TRIGGER_ATR:
            stop_rate = float(trade.open_rate) + side * (
                self.RANGE_BREAKEVEN_LOCK_ATR * atr_value
                + 2.0 * self.SIDE_COST * float(trade.open_rate)
            )
        value = stoploss_from_absolute(
            stop_rate,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
        return abs(float(value))

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs: Any,
    ) -> Any:
        if self._component(getattr(trade, "enter_tag", None)) == "range":
            return None
        return super().adjust_trade_position(
            trade,
            current_time,
            current_rate,
            current_profit,
            min_stake,
            max_stake,
            current_entry_rate,
            current_exit_rate,
            current_entry_profit,
            current_exit_profit,
            **kwargs,
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        if self._component(getattr(trade, "enter_tag", None)) != "range":
            return super().custom_exit(
                pair,
                trade,
                current_time,
                current_rate,
                current_profit,
                **kwargs,
            )
        # The auxiliary may use an idle tactical slot, but it must not keep
        # that slot once a primary Grid candidate becomes executable.  Exit
        # processing precedes new entries in Freqtrade's candle loop, so this
        # causal preemption lets the ranked Grid campaign enter on the same
        # completed candle.  In LIVE the same candidate map is built from the
        # synchronized closed 1h bar.
        if self._ranked_candidates("grid", current_time):
            return "range_v14_primary_preempt"
        target = float(trade.get_custom_data("range_target", 0.0))
        if target > 0.0 and (
            (trade.is_short and current_rate <= target)
            or (not trade.is_short and current_rate >= target)
        ):
            return "range_v14_mid_reversion"
        opened = self._trade_open_time(trade)
        held = current_time - opened
        if held >= timedelta(hours=self.RANGE_MAX_HOLD_HOURS):
            return "range_v14_time_stop"
        if held < timedelta(hours=self.RANGE_EARLY_FAILURE_HOURS):
            return None
        row = self._latest_row(pair, current_time)
        if row is None:
            return None
        side = -1.0 if trade.is_short else 1.0
        adverse_regime = side * float(row.get("market_regime_score", 0.0)) <= (
            -self.RANGE_FAILURE_REGIME
        )
        adverse_slope = side * float(row.get("range_ema_slope_atr", 0.0)) <= (
            -self.RANGE_FAILURE_SLOPE_ATR
        )
        if adverse_regime or adverse_slope:
            return "range_v14_regime_failure"
        return None


class _V14RelativeRangeSleeveMixin(_V14RangeSleeveMixin):
    """Fade cross-market residual extremes during persistent compression.

    A plain price band confuses a market-wide move with target-symbol
    mispricing.  This alternative standardizes each contract's completed 4h
    return against the synchronized BTC/ETH benchmark and only fades a tail
    after the candle itself starts reverting.  The 72h opportunity state and
    all z-score moments are causal; no future constituent or candle is used.
    """

    RELATIVE_RANGE_Z_ENTRY = 1.60
    RELATIVE_RANGE_Z_MAX = 3.50
    RELATIVE_RANGE_Z_EXIT = 0.20
    RELATIVE_RANGE_LOOKBACK = 72
    RELATIVE_RANGE_MIN_PERSISTENCE = 0.55
    RELATIVE_RANGE_MAX_ABS_REGIME = 0.70
    RELATIVE_RANGE_MAX_SYMBOL_EFFICIENCY = 0.48
    RELATIVE_RANGE_MAX_ABS_BTC_TREND_4H = 0.65
    RELATIVE_RANGE_MAX_ABS_BTC_TREND_1D = 0.55
    RELATIVE_RANGE_MIN_VOLUME = 0.55
    RELATIVE_RANGE_MAX_VOLUME = 1.80

    RANGE_RISK_PCT = 0.012
    RANGE_MAX_NOTIONAL = 2.00
    RANGE_STOP_ATR = 1.10
    RANGE_MIN_TARGET_ATR = 0.45
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.50
    RANGE_MAX_HOLD_HOURS = 12
    RANGE_EARLY_FAILURE_HOURS = 4
    RANGE_DAILY_LIMIT = 3
    RANGE_MINIMUM_BAR_GAP = 6

    def _populate_range(self, dataframe: DataFrame) -> DataFrame:
        close = dataframe["close"].astype(float)
        atr = dataframe["atr"].clip(lower=1e-12)
        benchmark_return_4h = 0.5 * (
            dataframe["btc_return_4h"] + dataframe["eth_return_4h"]
        )
        residual = dataframe["symbol_return_4h"] - benchmark_return_4h
        trailing = residual.shift(1)
        center = trailing.rolling(
            self.RELATIVE_RANGE_LOOKBACK,
            min_periods=self.RELATIVE_RANGE_LOOKBACK // 2,
        ).mean()
        scale = trailing.rolling(
            self.RELATIVE_RANGE_LOOKBACK,
            min_periods=self.RELATIVE_RANGE_LOOKBACK // 2,
        ).std(ddof=0).clip(lower=0.0025)
        zscore = (residual - center) / scale
        candle_range = (dataframe["high"] - dataframe["low"]).clip(
            lower=1e-12
        )
        close_position = (close - dataframe["low"]) / candle_range

        dataframe["range_mid"] = dataframe["fast_ema"]
        dataframe["range_lower"] = dataframe["fast_ema"] - atr
        dataframe["range_upper"] = dataframe["fast_ema"] + atr
        dataframe["range_z"] = zscore
        dataframe["range_residual_4h"] = residual
        dataframe["range_residual_center_72h"] = center
        dataframe["range_residual_scale_72h"] = scale
        dataframe["range_band_atr"] = 2.0
        dataframe["range_rsi"] = _rsi(close, 14)
        dataframe["range_close_position"] = close_position
        dataframe["range_ema_slope_atr"] = (
            dataframe["fast_ema"] - dataframe["fast_ema"].shift(6)
        ) / atr

        ready = dataframe[
            [
                "market_regime_score",
                "market_efficiency",
                "market_dispersion_7d",
                "low_opportunity_fraction_72h",
                "btc_trend_4h",
                "btc_trend_1d",
                "symbol_efficiency_12h",
                "grid_volume_ratio",
            ]
        ].notna().all(axis=1) & zscore.notna()
        range_regime = (
            ready
            & (
                dataframe["low_opportunity_fraction_72h"]
                >= self.RELATIVE_RANGE_MIN_PERSISTENCE
            )
            & (
                dataframe["market_regime_score"].abs()
                <= self.RELATIVE_RANGE_MAX_ABS_REGIME
            )
            & dataframe["market_efficiency"].between(0.07, 0.31)
            & dataframe["market_dispersion_7d"].between(0.010, 0.050)
            & (
                dataframe["symbol_efficiency_12h"].abs()
                <= self.RELATIVE_RANGE_MAX_SYMBOL_EFFICIENCY
            )
            & (
                dataframe["btc_trend_4h"].abs()
                <= self.RELATIVE_RANGE_MAX_ABS_BTC_TREND_4H
            )
            & (
                dataframe["btc_trend_1d"].abs()
                <= self.RELATIVE_RANGE_MAX_ABS_BTC_TREND_1D
            )
            & dataframe["grid_volume_ratio"].between(
                self.RELATIVE_RANGE_MIN_VOLUME,
                self.RELATIVE_RANGE_MAX_VOLUME,
            )
            & (atr / close).between(0.0035, 0.060)
        )
        lower_tail = zscore.between(
            -self.RELATIVE_RANGE_Z_MAX,
            -self.RELATIVE_RANGE_Z_ENTRY,
        )
        upper_tail = zscore.between(
            self.RELATIVE_RANGE_Z_ENTRY,
            self.RELATIVE_RANGE_Z_MAX,
        )
        long_reversal = (
            lower_tail
            & (zscore > zscore.shift(1))
            & (close > dataframe["open"])
            & (close_position >= 0.58)
            & (dataframe["range_rsi"] <= 46.0)
        )
        short_reversal = (
            upper_tail
            & (zscore < zscore.shift(1))
            & (close < dataframe["open"])
            & (close_position <= 0.42)
            & (dataframe["range_rsi"] >= 54.0)
        )
        raw_long = range_regime & long_reversal
        raw_short = range_regime & short_reversal
        capped = _causal_signal_cap(
            raw_long | raw_short,
            dataframe["date"],
            daily_limit=self.RANGE_DAILY_LIMIT,
            minimum_bar_gap=self.RANGE_MINIMUM_BAR_GAP,
        ).astype(bool)
        long_entry = raw_long & capped
        short_entry = raw_short & capped
        dataframe["range_regime"] = range_regime.astype(int)
        dataframe["range_entry_long"] = long_entry.astype(int)
        dataframe["range_entry_short"] = short_entry.astype(int)
        dataframe["range_entry"] = (long_entry | short_entry).astype(int)
        reversal_strength = (close_position - 0.5).abs() * 2.0
        dataframe["range_quality"] = (
            0.60 * zscore.abs().clip(upper=3.5)
            + 0.20 * reversal_strength.clip(upper=1.0)
            + 0.20 * dataframe["low_opportunity_fraction_72h"].clip(
                lower=0.0, upper=1.0
            )
        )
        dataframe["range_rank"] = -dataframe["range_quality"]
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs: Any,
    ) -> str | bool | None:
        reason = super().custom_exit(
            pair,
            trade,
            current_time,
            current_rate,
            current_profit,
            **kwargs,
        )
        if reason or self._component(getattr(trade, "enter_tag", None)) != "range":
            return reason
        row = self._latest_row(pair, current_time)
        if row is None:
            return None
        zscore = float(row.get("range_z", np.nan))
        if np.isfinite(zscore) and (
            (trade.is_short and zscore <= self.RELATIVE_RANGE_Z_EXIT)
            or (not trade.is_short and zscore >= -self.RELATIVE_RANGE_Z_EXIT)
        ):
            return "range_v14_relative_reversion"
        return None


class RangeV14ChopStandaloneFreqtrade(
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Standalone Range sleeve used to validate edge before composition."""

    RANGE_ONLY = True
    MAX_OPEN_TRADES = 1
    STRATEGY_VERSION = "range_v14_chop_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_standalone_high_water"


class RangeV14ChopEma35StandaloneFreqtrade(
    RangeV14ChopStandaloneFreqtrade,
):
    """Neighbor used to verify the slow-trend extension boundary."""

    RANGE_SHORT_MIN_SYMBOL_EMA55_ATR = 0.35
    STRATEGY_VERSION = "range_v14_chop_ema35_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_ema35_high_water"


class RangeV14ChopEma50StandaloneFreqtrade(
    RangeV14ChopStandaloneFreqtrade,
):
    """Neighbor used to verify the slow-trend extension boundary."""

    RANGE_SHORT_MIN_SYMBOL_EMA55_ATR = 0.50
    STRATEGY_VERSION = "range_v14_chop_ema50_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_ema50_high_water"


class RangeV14ChopBe35StandaloneFreqtrade(
    RangeV14ChopStandaloneFreqtrade,
):
    """Fast capital-protection neighbor for the short-horizon sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.35
    STRATEGY_VERSION = "range_v14_chop_be35_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_be35_high_water"


class RangeV14ChopBe45StandaloneFreqtrade(
    RangeV14ChopStandaloneFreqtrade,
):
    """Middle capital-protection neighbor for the short-horizon sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = "range_v14_chop_be45_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_be45_high_water"


class RangeV14ChopBe55StandaloneFreqtrade(
    RangeV14ChopStandaloneFreqtrade,
):
    """Conservative capital-protection neighbor for the range sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.55
    STRATEGY_VERSION = "range_v14_chop_be55_standalone_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_chop_be55_high_water"


class RangeV14PersistentChopBe35StandaloneFreqtrade(
    _V14RangePersistenceMixin,
    RangeV14ChopBe35StandaloneFreqtrade,
):
    """Cross-regime persistence filter with fast capital protection."""

    STRATEGY_VERSION = "range_v14_persistent_chop_be35_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_persistent_chop_be35_high_water"


class RangeV14PersistentChopBe45StandaloneFreqtrade(
    _V14RangePersistenceMixin,
    RangeV14ChopBe45StandaloneFreqtrade,
):
    """Cross-regime persistence filter with a wider profit path."""

    STRATEGY_VERSION = "range_v14_persistent_chop_be45_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_persistent_chop_be45_high_water"


class RangeV14DynamicPersistentChopStandaloneFreqtrade(
    _V14DynamicUniverseMixin,
    RangeV14PersistentChopBe35StandaloneFreqtrade,
):
    """Standalone persistent Range edge on the causal monthly Top50."""

    STRATEGY_VERSION = "range_v14_dynamic_persistent_chop_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_dynamic_persistent_chop_high_water"


class RangeV14DynamicPersistentChop45StandaloneFreqtrade(
    _V14DynamicUniverseMixin,
    RangeV14PersistentChopBe45StandaloneFreqtrade,
):
    """Wider profit path selected by the four-regime exact replay."""

    STRATEGY_VERSION = "range_v14_dynamic_persistent_chop45_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_dynamic_persistent_chop45_high_water"


class RangeV14DynamicPersistentBalancedStandaloneFreqtrade(
    RangeV14DynamicPersistentChopStandaloneFreqtrade,
):
    """Broader structural neighbor; loosens each frequency axis modestly."""

    RANGE_DEVIATION = 1.35
    RANGE_MAX_MARKET_EFFICIENCY = 0.30
    RANGE_MAX_BAND_ATR = 2.75
    RANGE_MAX_ABS_EMA_SLOPE_ATR = 0.32
    RANGE_MAX_VOLUME_RATIO = 1.50
    RANGE_DAILY_LIMIT = 4
    RANGE_MINIMUM_BAR_GAP = 5
    STRATEGY_VERSION = "range_v14_dynamic_persistent_balanced_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_dynamic_persistent_balanced_high_water"


class RangeV14DynamicPersistentWideStandaloneFreqtrade(
    RangeV14DynamicPersistentChopStandaloneFreqtrade,
):
    """Frequency stress neighbor used to reject an overly narrow optimum."""

    RANGE_DEVIATION = 1.25
    RANGE_MAX_MARKET_EFFICIENCY = 0.33
    RANGE_MAX_BAND_ATR = 3.00
    RANGE_MAX_ABS_EMA_SLOPE_ATR = 0.40
    RANGE_MAX_VOLUME_RATIO = 1.65
    RANGE_DAILY_LIMIT = 4
    RANGE_MINIMUM_BAR_GAP = 4
    STRATEGY_VERSION = "range_v14_dynamic_persistent_wide_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_dynamic_persistent_wide_high_water"


class RangeV14RelativeResidualStandaloneFreqtrade(
    _V14RelativeRangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Standalone cross-market residual-reversion research sleeve."""

    RANGE_ONLY = True
    MAX_OPEN_TRADES = 1
    STRATEGY_VERSION = "range_v14_relative_residual_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_relative_residual_high_water"


class RangeV14RelativeResidualZ14StandaloneFreqtrade(
    RangeV14RelativeResidualStandaloneFreqtrade,
):
    RELATIVE_RANGE_Z_ENTRY = 1.40
    STRATEGY_VERSION = "range_v14_relative_residual_z14_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_relative_residual_z14_high_water"


class RangeV14RelativeResidualZ18StandaloneFreqtrade(
    RangeV14RelativeResidualStandaloneFreqtrade,
):
    RELATIVE_RANGE_Z_ENTRY = 1.80
    STRATEGY_VERSION = "range_v14_relative_residual_z18_research_20260809"
    ADAPTIVE_STATE_BASENAME = "range_v14_relative_residual_z18_high_water"


class BreakoutV14GridPullbackOnlyResearchFreqtrade(
    _V14GridTruePullbackMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 plus only the Grid-long true-pullback correction."""

    STRATEGY_VERSION = "breakout_v14_grid_pullback_only_research_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_pullback_only_high_water"


class BreakoutV14GridPullbackContinuousRiskResearchFreqtrade(
    _V14GridFalsePullbackContinuousRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 timing with smooth Grid-long false-pullback sizing."""

    STRATEGY_VERSION = "breakout_v14_grid_pullback_continuous_risk_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_grid_pullback_continuous_risk_high_water"
    )


class BreakoutV14GridPullbackContinuousRisk4ResearchFreqtrade(
    BreakoutV14GridPullbackContinuousRiskResearchFreqtrade,
):
    """Neighbor: reach the defensive floor at a 0.4% rebound."""

    V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H = 0.004
    STRATEGY_VERSION = "breakout_v14_grid_pullback_continuous_risk4_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_pullback_risk4_high_water"


class BreakoutV14GridPullbackContinuousRisk6ResearchFreqtrade(
    BreakoutV14GridPullbackContinuousRiskResearchFreqtrade,
):
    """Neighbor: reach the defensive floor at a 0.6% rebound."""

    V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H = 0.006
    STRATEGY_VERSION = "breakout_v14_grid_pullback_continuous_risk6_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_pullback_risk6_high_water"


class BreakoutV14SmoothGridConflictBaselineResearchFreqtrade(
    _V14SmoothGridConflictRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Ablation: only replace the V13 binary Grid-short conflict gate."""

    STRATEGY_VERSION = "breakout_v14_smooth_grid_conflict_baseline_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_grid_conflict_high_water"


class BreakoutV14SmoothConflictContinuousGridRiskResearchFreqtrade(
    _V14GridFalsePullbackContinuousRiskMixin,
    _V14SmoothGridConflictRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Continuous Grid-long and Grid-short risk without entry deletion."""

    STRATEGY_VERSION = "breakout_v14_smooth_conflict_continuous_grid_risk_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_continuous_grid_high_water"


class BreakoutV14SmoothConflictContinuousGridRisk4ResearchFreqtrade(
    BreakoutV14SmoothConflictContinuousGridRiskResearchFreqtrade,
):
    V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H = 0.004
    STRATEGY_VERSION = "breakout_v14_smooth_continuous_grid_risk4_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_continuous_grid_risk4_high_water"


class BreakoutV14SmoothConflictContinuousGridRisk6ResearchFreqtrade(
    BreakoutV14SmoothConflictContinuousGridRiskResearchFreqtrade,
):
    V14_GRID_PULLBACK_FULL_COMPRESSION_RETURN_4H = 0.006
    STRATEGY_VERSION = "breakout_v14_smooth_continuous_grid_risk6_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_continuous_grid_risk6_high_water"


class BreakoutV14GridSleeveHighWaterResearchFreqtrade(
    _V14GridSleeveHighWaterMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 signals with an independent, continuous Grid risk ledger."""

    STRATEGY_VERSION = "breakout_v14_grid_sleeve_high_water_research_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_sleeve_high_water"


class BreakoutV14GridSleeveHighWaterGentleResearchFreqtrade(
    BreakoutV14GridSleeveHighWaterResearchFreqtrade,
):
    """Structural neighbor: later and gentler Grid sleeve compression."""

    V14_GRID_LEDGER_RISK_START_DRAWDOWN = 0.06
    V14_GRID_LEDGER_RISK_MID_DRAWDOWN = 0.16
    V14_GRID_LEDGER_RISK_FULL_DRAWDOWN = 0.28
    V14_GRID_LEDGER_MID_SCALE = 0.55
    V14_GRID_LEDGER_MIN_SCALE = 0.25
    STRATEGY_VERSION = (
        "breakout_v14_grid_sleeve_high_water_gentle_research_20260809"
    )
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_sleeve_high_water_gentle"


class BreakoutV14DynamicUniverseV13ParityResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 logic on the causal historical monthly Top50; research only."""

    STRATEGY_VERSION = "breakout_v14_dynamic_universe_v13_parity_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_dynamic_v13_parity_high_water"


class BreakoutV14DynamicUniverseV12ParityResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV12RegimeAdaptiveGridV9SelectedFreqtrade,
):
    """Frozen V12 on the same causal monthly Top50 robustness harness."""

    STRATEGY_VERSION = "breakout_v14_dynamic_universe_v12_parity_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_dynamic_v12_parity_high_water"


class BreakoutV14CrossUniverseQualityV13ResearchFreqtrade(
    _V14CrossUniverseSignalQualityMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Static Active50 research control for the cross-universe quality layer."""

    STRATEGY_VERSION = "breakout_v14_cross_universe_quality_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_cross_universe_quality_high_water"


class BreakoutV14CrossUniverseQuality08V13ResearchFreqtrade(
    BreakoutV14CrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.08
    STRATEGY_VERSION = "breakout_v14_cross_universe_quality08_v13_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_cross_universe_quality08_high_water"
    )


class BreakoutV14CrossUniverseQuality10V13ResearchFreqtrade(
    BreakoutV14CrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.10
    STRATEGY_VERSION = "breakout_v14_cross_universe_quality10_v13_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_cross_universe_quality10_high_water"
    )


class BreakoutV14CrossUniverseQuality12V13ResearchFreqtrade(
    BreakoutV14CrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.12
    STRATEGY_VERSION = "breakout_v14_cross_universe_quality12_v13_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_cross_universe_quality12_high_water"
    )


class BreakoutV14LateChaseOnlyV13ResearchFreqtrade(
    BreakoutV14CrossUniverseQualityV13ResearchFreqtrade,
):
    """Ablation: Breakout-long exhaustion gate without the Grid gate."""

    V14_GRID_SHORT_MIN_RETURN_4H = float("-inf")
    STRATEGY_VERSION = "breakout_v14_late_chase_only_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_late_chase_only_high_water"


class BreakoutV14GridReboundOnlyV13ResearchFreqtrade(
    BreakoutV14CrossUniverseQualityV13ResearchFreqtrade,
):
    """Ablation: Grid-short rebound gate without the Breakout gate."""

    V14_BO_LONG_MAX_RETURN_4H = float("inf")
    STRATEGY_VERSION = "breakout_v14_grid_rebound_only_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_grid_rebound_only_high_water"


class BreakoutV14SmoothQualityRiskV13ResearchFreqtrade(
    _V14ContinuousEntryQualityRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 signals with continuous, non-deleting structural risk curves."""

    STRATEGY_VERSION = "breakout_v14_smooth_quality_risk_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_quality_risk_high_water"


class BreakoutV14SmoothQualityRiskGentleV13ResearchFreqtrade(
    BreakoutV14SmoothQualityRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_RISK_MIN_SCALE = 0.70
    V14_GRID_SHORT_RISK_MIN_SCALE = 0.70
    V14_GRID_LONG_RISK_MIN_SCALE = 0.70
    STRATEGY_VERSION = (
        "breakout_v14_smooth_quality_risk_gentle_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_smooth_quality_risk_gentle_high_water"
    )


class BreakoutV14SmoothGridRiskOnlyV13ResearchFreqtrade(
    BreakoutV14SmoothQualityRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_RISK_MIN_SCALE = 1.0
    STRATEGY_VERSION = "breakout_v14_smooth_grid_risk_only_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_smooth_grid_risk_only_high_water"


class BreakoutV14SmoothGridShortRiskOnlyV13ResearchFreqtrade(
    BreakoutV14SmoothGridRiskOnlyV13ResearchFreqtrade,
):
    V14_GRID_LONG_RISK_MIN_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v14_smooth_grid_short_risk_only_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_smooth_grid_short_risk_only_high_water"
    )


class BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade(
    BreakoutV14SmoothGridRiskOnlyV13ResearchFreqtrade,
):
    V14_GRID_SHORT_RISK_MIN_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v14_smooth_grid_long_risk_only_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_smooth_grid_long_risk_only_high_water"
    )


class BreakoutV14GridLongRiskPersistentRangeV13ResearchFreqtrade(
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Cross-year Grid-long risk winner plus a preemptible Range auxiliary."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_grid_long_risk_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_grid_long_risk_persistent_range_v13_high_water"
    )


class BreakoutV14SmoothGovernorGridLongRiskV13ResearchFreqtrade(
    _V14ContinuousPortfolioGovernorMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Grid-long risk winner with a smooth, evidence-latched governor."""

    STRATEGY_VERSION = (
        "breakout_v14_smooth_governor_grid_long_risk_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_smooth_governor_grid_long_risk_high_water"
    )


class BreakoutV14ChopBreakoutRiskV13ResearchFreqtrade(
    _V14ChopBreakoutRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 with only the persistent-chop Breakout-short risk curve."""

    STRATEGY_VERSION = "breakout_v14_chop_breakout_risk_v13_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_chop_breakout_risk_high_water"


class BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Causal monthly Top50 plus the universe-robust quality layer."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_cross_universe_quality_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_cross_universe_quality_high_water"
    )


class BreakoutV14DynamicCrossUniverseQuality08V13ResearchFreqtrade(
    BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.08
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_cross_universe_quality08_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_cross_universe_quality08_high_water"
    )


class BreakoutV14DynamicCrossUniverseQuality10V13ResearchFreqtrade(
    BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.10
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_cross_universe_quality10_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_cross_universe_quality10_high_water"
    )


class BreakoutV14DynamicCrossUniverseQuality12V13ResearchFreqtrade(
    BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.12
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_cross_universe_quality12_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_cross_universe_quality12_high_water"
    )


class BreakoutV14DynamicLateChaseOnlyV13ResearchFreqtrade(
    BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade,
):
    V14_GRID_SHORT_MIN_RETURN_4H = float("-inf")
    STRATEGY_VERSION = "breakout_v14_dynamic_late_chase_only_v13_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_late_chase_only_high_water"
    )


class BreakoutV14DynamicGridReboundOnlyV13ResearchFreqtrade(
    BreakoutV14DynamicCrossUniverseQualityV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = float("inf")
    STRATEGY_VERSION = "breakout_v14_dynamic_grid_rebound_only_v13_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_grid_rebound_only_high_water"
    )


class BreakoutV14DynamicSmoothQualityRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14ContinuousEntryQualityRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_smooth_quality_risk_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_smooth_quality_risk_high_water"
    )


class BreakoutV14DynamicSmoothQualityRiskGentleV13ResearchFreqtrade(
    BreakoutV14DynamicSmoothQualityRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_RISK_MIN_SCALE = 0.70
    V14_GRID_SHORT_RISK_MIN_SCALE = 0.70
    V14_GRID_LONG_RISK_MIN_SCALE = 0.70
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_smooth_quality_risk_gentle_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_smooth_quality_risk_gentle_high_water"
    )


class BreakoutV14DynamicSmoothGridRiskOnlyV13ResearchFreqtrade(
    BreakoutV14DynamicSmoothQualityRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_RISK_MIN_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_smooth_grid_risk_only_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_smooth_grid_risk_only_high_water"
    )


class BreakoutV14DynamicSmoothGridLongRiskOnlyV13ResearchFreqtrade(
    BreakoutV14DynamicSmoothGridRiskOnlyV13ResearchFreqtrade,
):
    """Causal monthly Top50 version of the Grid-long-only risk curve."""

    V14_GRID_SHORT_RISK_MIN_SCALE = 1.0
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_smooth_grid_long_risk_only_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_smooth_grid_long_risk_only_high_water"
    )


class BreakoutV14DynamicGridLongRiskPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV14GridLongRiskPersistentRangeV13ResearchFreqtrade,
):
    """Dynamic Top50 composition with preemptible Range and Grid-long curve."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_grid_long_risk_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_grid_long_risk_persistent_range_high_water"
    )


class BreakoutV14DynamicQualityGridLongRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Dynamic-universe quality gates plus the cross-year Grid-long curve."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_grid_long_risk_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_grid_long_risk_high_water"
    )


class BreakoutV14DynamicQualityGridLongRiskPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Full dynamic candidate with a preemptible idle-capacity Range sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_grid_long_risk_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_grid_long_risk_range_high_water"
    )


class BreakoutV14DynamicQualitySmoothGovernorGridLongRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14ContinuousPortfolioGovernorMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Dynamic quality candidate with the continuous portfolio governor."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_smooth_governor_grid_long_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_smooth_governor_grid_long_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkGridLongRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Dynamic quality candidate with continuous benchmark confirmation."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_grid_long_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_grid_long_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Ablation: benchmark-confirmed Breakout risk without Grid-long sizing."""

    STRATEGY_VERSION = "breakout_v14_dynamic_quality_benchmark_risk_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_risk_high_water"
    )


class BreakoutV14DynamicLeadershipBenchmarkRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14LeadershipAwareSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Isolated test of the leader exception without other Grid risk changes."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_leadership_benchmark_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_leadership_benchmark_risk_high_water"
    )


class BreakoutV14DynamicQuality08BenchmarkRiskV13ResearchFreqtrade(
    BreakoutV14DynamicQualityBenchmarkRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.08
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality08_benchmark_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality08_benchmark_risk_high_water"
    )


class BreakoutV14DynamicQuality10BenchmarkRiskV13ResearchFreqtrade(
    BreakoutV14DynamicQualityBenchmarkRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.10
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality10_benchmark_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality10_benchmark_risk_high_water"
    )


class BreakoutV14DynamicQuality12BenchmarkRiskV13ResearchFreqtrade(
    BreakoutV14DynamicQualityBenchmarkRiskV13ResearchFreqtrade,
):
    V14_BO_LONG_MAX_RETURN_4H = 0.12
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality12_benchmark_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality12_benchmark_risk_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkPullbackRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14GridFalsePullbackContinuousRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Benchmark-confirmed Breakout plus continuous true-pullback Grid risk."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_pullback_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_pullback_risk_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkReentryV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _BreakoutV12BullReentryMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Capped bull pullback/rebreak participation on top of the V13 core."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_reentry_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_reentry_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkReentryStrongV13ResearchFreqtrade(
    BreakoutV14DynamicQualityBenchmarkReentryV13ResearchFreqtrade,
):
    """Structural neighbor requiring stronger broad-market confirmation."""

    BO_V12_REENTRY_MIN_REGIME = 0.35
    BO_V12_REENTRY_MIN_BREADTH = 0.60
    BO_V12_REENTRY_MIN_SLOW_BREADTH = 0.55
    BO_V12_REENTRY_MIN_BTC_DAILY_TREND = 0.15
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_reentry_strong_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_reentry_strong_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkChopRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14ChopBreakoutRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Keep Breakout active in chop while continuously reducing weak shorts."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_chop_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_chop_risk_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkNarrowGovernorV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14NarrowContinuousPortfolioGovernorMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Candidate with an 18%-centered continuous portfolio transition."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_narrow_governor_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_narrow_governor_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkNarrowCliffsV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14NarrowSmoothGridConflictRiskMixin,
    _V14NarrowContinuousPortfolioGovernorMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Remove both selected risk cliffs while retaining their centers."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_narrow_cliffs_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_narrow_cliffs_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Primary candidate plus preemptible, ledger-isolated Range sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.35
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_range_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_range_high_water"
    )


class BreakoutV14DynamicQualityBenchmarkRange45V13ResearchFreqtrade(
    BreakoutV14DynamicQualityBenchmarkRangeV13ResearchFreqtrade,
):
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_benchmark_range45_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_benchmark_range45_high_water"
    )


class BreakoutV14DynamicQualityBalancedRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14CrowdedShallowGridShortRiskMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Dynamic candidate with continuous risk on all audited weak structures."""

    STRATEGY_VERSION = "breakout_v14_dynamic_quality_balanced_risk_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_balanced_risk_high_water"
    )


class BreakoutV14DynamicLeadershipBalancedRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14LeadershipAwareSignalQualityMixin,
    _V14BenchmarkConfirmedBreakoutLongRiskMixin,
    _V14CrowdedShallowGridShortRiskMixin,
    BreakoutV14SmoothGridLongRiskOnlyV13ResearchFreqtrade,
):
    """Balanced dynamic candidate with a narrow leader continuation path."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_leadership_balanced_risk_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_leadership_balanced_risk_high_water"
    )


class BreakoutV14DynamicChopBreakoutRiskV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14ChopBreakoutRiskMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_chop_breakout_risk_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_chop_breakout_risk_high_water"
    )


class BreakoutV14RangeResearchGridV9Freqtrade(
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """V13 plus the independent Range tactical sleeve; research only."""

    STRATEGY_VERSION = "breakout_v14_range_research_grid_v9_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_range_research_grid_v9_high_water"


class BreakoutV14PersistentRangeV13ResearchFreqtrade(
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Frozen V13 plus only the cross-regime validated range auxiliary."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = "breakout_v14_persistent_range_v13_research_20260809"
    ADAPTIVE_STATE_BASENAME = "breakout_v14_persistent_range_v13_high_water"


class BreakoutV14ChopRiskPersistentRangeV13ResearchFreqtrade(
    _V14ChopBreakoutRiskMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Smooth chop allocation plus the isolated spare-capacity Range sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_chop_risk_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_chop_risk_persistent_range_high_water"
    )


class BreakoutV14DynamicChopRiskPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14ChopBreakoutRiskMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_chop_risk_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_chop_risk_persistent_range_high_water"
    )


class BreakoutV14DynamicQualityPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_persistent_range_high_water"
    )


class BreakoutV14DynamicQualityChopPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    _V14CrossUniverseSignalQualityMixin,
    _V14ChopBreakoutRiskMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_dynamic_quality_chop_persistent_range_v13_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_quality_chop_persistent_range_high_water"
    )


class BreakoutV14GridHighWaterPersistentRangeResearchFreqtrade(
    _V14GridSleeveHighWaterMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Range auxiliary plus an independent Grid-sleeve high-water ledger."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_grid_high_water_persistent_range_research_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_grid_high_water_persistent_range_high_water"
    )


class BreakoutV14PullbackRangeResearchGridV9Freqtrade(
    _V14GridTruePullbackMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Range sleeve plus the structural Grid-long pullback correction."""

    STRATEGY_VERSION = "breakout_v14_pullback_range_research_grid_v9_20260809"
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_pullback_range_research_grid_v9_high_water"
    )


class BreakoutV14PersistentRangePullbackResearchGridV9Freqtrade(
    _V14GridTruePullbackMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Grid true-pullback plus the cross-regime validated range sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_persistent_range_pullback_research_grid_v9_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_persistent_range_pullback_grid_v9_high_water"
    )


class BreakoutV14ContinuousGridRiskPersistentRangeResearchFreqtrade(
    _V14GridFalsePullbackContinuousRiskMixin,
    _V14RangePrimaryIsolationMixin,
    _V14RangePersistenceMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Continuous Grid risk plus the validated range auxiliary sleeve."""

    RANGE_BREAKEVEN_TRIGGER_ATR = 0.45
    STRATEGY_VERSION = (
        "breakout_v14_continuous_grid_risk_persistent_range_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_continuous_grid_risk_persistent_range_high_water"
    )


class BreakoutV14DynamicPersistentRangePullbackResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV14PersistentRangePullbackResearchGridV9Freqtrade,
):
    """Full causal-universe composition candidate; research only."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_persistent_range_pullback_research_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_persistent_range_pullback_high_water"
    )


class BreakoutV14DynamicPersistentRangeV13ResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV14PersistentRangeV13ResearchFreqtrade,
):
    """Dynamic historical Top50 without the rejected hard Grid ablation."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_persistent_range_v13_research_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_persistent_range_v13_high_water"
    )


class BreakoutV14DynamicGridHighWaterPersistentRangeResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV14GridHighWaterPersistentRangeResearchFreqtrade,
):
    """Causal historical Top50 plus range and Grid sleeve allocation."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_grid_high_water_range_research_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_grid_high_water_range_high_water"
    )


class BreakoutV14DynamicContinuousGridRiskPersistentRangeResearchFreqtrade(
    _V14DynamicUniverseMixin,
    BreakoutV14ContinuousGridRiskPersistentRangeResearchFreqtrade,
):
    """Dynamic-universe version of the continuous-allocation candidate."""

    STRATEGY_VERSION = (
        "breakout_v14_dynamic_continuous_grid_risk_range_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_dynamic_continuous_grid_risk_range_high_water"
    )


class BreakoutV14SmoothRiskPullbackRangeResearchGridV9Freqtrade(
    _V14SmoothGridConflictRiskMixin,
    _V14GridTruePullbackMixin,
    _V14RangeSleeveMixin,
    BreakoutV13BullCaptureGridRelativeSqueezeSelectedFreqtrade,
):
    """Full V14 research stack with continuous Grid conflict sizing."""

    STRATEGY_VERSION = (
        "breakout_v14_smooth_risk_pullback_range_research_grid_v9_20260809"
    )
    ADAPTIVE_STATE_BASENAME = (
        "breakout_v14_smooth_risk_pullback_range_research_grid_v9_high_water"
    )
