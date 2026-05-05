from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RiskConfig
from .models import Candle, Direction, EquityPoint, Position, Signal, Trade
from .risk import RiskManager
from .strategy import VolatilityBreakoutScalper


@dataclass(frozen=True)
class BacktestResult:
    summary: dict[str, Any]
    trades: list[Trade]
    equity_curve: list[EquityPoint]


class Backtester:
    def __init__(
        self,
        candles: list[Candle],
        strategy: VolatilityBreakoutScalper,
        risk_config: RiskConfig,
    ) -> None:
        if not candles:
            raise ValueError("candles cannot be empty")
        self.candles = candles
        self.strategy = strategy
        self.risk_config = risk_config
        self.risk = RiskManager(risk_config)
        self.equity = risk_config.initial_equity
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self.position: Position | None = None
        self.peak_equity = risk_config.initial_equity

    def run(self) -> BacktestResult:
        self.strategy.prepare(self.candles)

        for index, candle in enumerate(self.candles):
            mark_equity = self._mark_equity(candle.close)
            self.risk.on_bar(candle, mark_equity)
            self._record_equity(candle, mark_equity)

            if self.position:
                exit_price, reason = self._check_forced_exit(candle)
                if exit_price is not None:
                    self._close_position(candle, exit_price, reason)
                    continue
                exit_price, reason = self._manage_open_position(index, candle)
                if exit_price is not None:
                    self._close_position(candle, exit_price, reason)
                    continue

            signal = self.strategy.signal(index, self.candles)

            if self.position:
                if signal.direction != Direction.FLAT and signal.direction != self.position.direction:
                    self._close_position(candle, self._exit_execution_price(candle.close, self.position.direction), "strategy_exit")
                continue

            if signal.direction == Direction.FLAT:
                continue

            can_enter, reason = self.risk.can_enter(self.equity)
            if not can_enter:
                continue

            qty, size_reason = self.risk.size_position(self.equity, candle.close, signal)
            if size_reason != "ok":
                continue

            self._open_position(candle, signal, qty)

        if self.position:
            last = self.candles[-1]
            self._close_position(last, self._exit_execution_price(last.close, self.position.direction), "end_of_data")
            self._record_equity(last, self.equity)

        return BacktestResult(
            summary=self._summary(),
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

    def _open_position(self, candle: Candle, signal: Signal, qty: float) -> None:
        entry_price = self._entry_execution_price(candle.close, signal.direction)
        notional = qty * entry_price
        entry_fee = self._fee(notional)
        self.equity -= entry_fee

        if signal.direction == Direction.LONG:
            stop_price = entry_price * (1.0 - signal.stop_loss_pct)
            take_profit_price = entry_price * (1.0 + signal.take_profit_pct)
        else:
            stop_price = entry_price * (1.0 + signal.stop_loss_pct)
            take_profit_price = entry_price * (1.0 - signal.take_profit_pct)

        self.position = Position(
            direction=signal.direction,
            qty=qty,
            entry_price=entry_price,
            entry_time=candle.timestamp,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            entry_fee=entry_fee,
            peak_price=entry_price,
            trough_price=entry_price,
            max_holding_bars=signal.max_holding_bars,
        )

    def _close_position(self, candle: Candle, exit_price: float, reason: str) -> None:
        if not self.position:
            return

        position = self.position
        gross_pnl = position.direction.value * position.qty * (exit_price - position.entry_price)
        exit_fee = self._fee(position.qty * exit_price)
        fees = position.entry_fee + exit_fee
        net_pnl = gross_pnl - exit_fee
        self.equity += net_pnl

        trade = Trade(
            direction=position.direction,
            entry_time=position.entry_time,
            exit_time=candle.timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=gross_pnl - fees,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.risk.on_trade_closed(trade.net_pnl)
        self.position = None

    def _check_forced_exit(self, candle: Candle) -> tuple[float | None, str]:
        if not self.position:
            return None, ""

        liquidation_price = self._liquidation_price(self.position)
        if self.position.direction == Direction.LONG:
            if candle.low <= liquidation_price:
                return self._exit_execution_price(liquidation_price, self.position.direction), "liquidation"
            stop_hit = candle.low <= self.position.stop_price
            take_profit_hit = candle.high >= self.position.take_profit_price
            if stop_hit:
                return self._exit_execution_price(self.position.stop_price, self.position.direction), "stop_loss"
            if take_profit_hit:
                return self._exit_execution_price(self.position.take_profit_price, self.position.direction), "take_profit"
        else:
            if candle.high >= liquidation_price:
                return self._exit_execution_price(liquidation_price, self.position.direction), "liquidation"
            stop_hit = candle.high >= self.position.stop_price
            take_profit_hit = candle.low <= self.position.take_profit_price
            if stop_hit:
                return self._exit_execution_price(self.position.stop_price, self.position.direction), "stop_loss"
            if take_profit_hit:
                return self._exit_execution_price(self.position.take_profit_price, self.position.direction), "take_profit"

        return None, ""

    def _manage_open_position(self, index: int, candle: Candle) -> tuple[float | None, str]:
        if not self.position:
            return None, ""

        position = self.position
        position.bars_held += 1
        atr_value = self.strategy.atr_at(index - 1)
        config = self.strategy.config

        if position.direction == Direction.LONG:
            position.peak_price = max(position.peak_price, candle.high)
            favorable_move = position.peak_price - position.entry_price
            if atr_value > 0 and config.breakeven_atr > 0 and favorable_move >= atr_value * config.breakeven_atr:
                position.stop_price = max(position.stop_price, position.entry_price)
            if atr_value > 0 and config.trailing_stop_atr > 0:
                activation = atr_value * max(config.trailing_activation_atr, 0.0)
                if favorable_move >= activation:
                    position.stop_price = max(position.stop_price, position.peak_price - atr_value * config.trailing_stop_atr)
        else:
            position.trough_price = min(position.trough_price, candle.low)
            favorable_move = position.entry_price - position.trough_price
            if atr_value > 0 and config.breakeven_atr > 0 and favorable_move >= atr_value * config.breakeven_atr:
                position.stop_price = min(position.stop_price, position.entry_price)
            if atr_value > 0 and config.trailing_stop_atr > 0:
                activation = atr_value * max(config.trailing_activation_atr, 0.0)
                if favorable_move >= activation:
                    position.stop_price = min(position.stop_price, position.trough_price + atr_value * config.trailing_stop_atr)

        max_holding_bars = position.max_holding_bars or config.max_holding_bars
        if max_holding_bars > 0 and position.bars_held >= max_holding_bars:
            return self._exit_execution_price(candle.close, position.direction), "time_stop"

        return None, ""

    def _entry_execution_price(self, price: float, direction: Direction) -> float:
        slip = self.risk_config.slippage_bps / 10_000.0
        if direction == Direction.LONG:
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _exit_execution_price(self, price: float, direction: Direction) -> float:
        slip = self.risk_config.slippage_bps / 10_000.0
        if direction == Direction.LONG:
            return price * (1.0 - slip)
        return price * (1.0 + slip)

    def _liquidation_price(self, position: Position) -> float:
        leverage_loss_pct = 1.0 / max(self.risk_config.max_leverage, 1e-12)
        maintenance = self.risk_config.maintenance_margin_pct
        if position.direction == Direction.LONG:
            return position.entry_price * max(0.0, 1.0 - leverage_loss_pct + maintenance)
        return position.entry_price * (1.0 + leverage_loss_pct - maintenance)

    def _fee(self, notional: float) -> float:
        return abs(notional) * self.risk_config.fee_bps / 10_000.0

    def _mark_equity(self, mark_price: float) -> float:
        if not self.position:
            return self.equity
        return self.equity + self.position.unrealized_pnl(mark_price)

    def _record_equity(self, candle: Candle, mark_equity: float) -> None:
        self.peak_equity = max(self.peak_equity, mark_equity)
        drawdown = 0.0 if self.peak_equity <= 0 else (self.peak_equity - mark_equity) / self.peak_equity
        self.equity_curve.append(EquityPoint(candle.timestamp, mark_equity, drawdown))

    def _summary(self) -> dict[str, Any]:
        total = len(self.trades)
        wins = [trade for trade in self.trades if trade.net_pnl > 0]
        losses = [trade for trade in self.trades if trade.net_pnl <= 0]
        gross_profit = sum(trade.net_pnl for trade in wins)
        gross_loss = abs(sum(trade.net_pnl for trade in losses))
        max_drawdown = max((point.drawdown_pct for point in self.equity_curve), default=0.0)
        profit_factor = None if gross_loss == 0 else gross_profit / gross_loss
        avg_trade = 0.0 if total == 0 else sum(trade.net_pnl for trade in self.trades) / total
        return {
            "initial_equity": self.risk_config.initial_equity,
            "final_equity": self.equity,
            "net_profit": self.equity - self.risk_config.initial_equity,
            "net_return_pct": (self.equity / self.risk_config.initial_equity - 1.0) * 100.0,
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": 0.0 if total == 0 else len(wins) / total * 100.0,
            "profit_factor": profit_factor,
            "avg_trade": avg_trade,
            "max_drawdown_pct": max_drawdown * 100.0,
        }
