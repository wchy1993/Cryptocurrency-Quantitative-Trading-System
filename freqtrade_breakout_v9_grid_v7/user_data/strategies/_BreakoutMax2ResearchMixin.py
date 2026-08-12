from __future__ import annotations

from datetime import datetime, timedelta, timezone

from freqtrade.persistence import Trade

from _BreakoutOnlyResearchMixin import BreakoutOnlyResearchMixin


class BreakoutMax2ResearchMixin(BreakoutOnlyResearchMixin):
    """Run two Breakout campaigns while keeping the Grid sleeve disabled.

    The frozen combined strategies normally allow only one open campaign per
    component.  This research adapter changes only that portfolio constraint:
    two distinct Breakout pairs may be open at once.  Existing same-pair,
    cooldown and daily-entry guards remain in force.
    """

    MAX_OPEN_TRADES = 2
    BREAKOUT_OPEN_LIMIT = 2

    def _component_limits_allow(
        self,
        pair: str,
        component: str,
        current_time: datetime,
    ) -> bool:
        if component != "breakout":
            return False

        trades = Trade.get_trades_proxy()
        open_breakout = [
            trade
            for trade in trades
            if trade.is_open
            and self._component(getattr(trade, "enter_tag", None))
            == "breakout"
        ]
        if len(open_breakout) >= self.BREAKOUT_OPEN_LIMIT:
            return False
        if any(trade.pair == pair for trade in open_breakout):
            return False

        for trade in trades:
            if trade.is_open or trade.pair != pair:
                continue
            if self._component(getattr(trade, "enter_tag", None)) != component:
                continue
            close_time = self._trade_close_time(trade)
            if (
                close_time is not None
                and current_time < close_time + timedelta(minutes=120)
            ):
                return False

        day = current_time.astimezone(timezone.utc).date()
        entries_today = sum(
            1
            for trade in trades
            if self._component(getattr(trade, "enter_tag", None)) == component
            and self._trade_open_time(trade).date() == day
        )
        return entries_today < 5
