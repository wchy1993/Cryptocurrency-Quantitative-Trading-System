from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from freqtrade.enums import RunMode

from BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade import (
    BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade,
)


logger = logging.getLogger(__name__)


class BreakoutV11AdaptiveGridV8DualSideFreqtrade(
    BreakoutV11DefensiveBoS3Risk50Recovery60Q112GridV8DualSideFreqtrade
):
    """Selected v11 entry point with restart-safe runtime high-water state.

    Selection gates (Active50, 200 USDT, Max2, isolated futures) require exact
    v10F/Grid-v8 trade-row parity over three and six months, plus higher
    one-year net profit and profit factor with lower minute-wallet drawdown.

    This entry point persists the already-used realized-equity high-water mark
    in LIVE/DRY-RUN so a process restart cannot silently re-arm full risk.  It
    also contains the execution-only tail guard which promotes a partial exit
    to a full exit when exchange precision would otherwise leave an
    untradeable remainder and cause Freqtrade to reject the entire reduction.
    Backtesting and hyperopt never read or write the runtime high-water state.
    """

    ADAPTIVE_STATE_VERSION = 1
    ADAPTIVE_STATE_DIR = Path(__file__).resolve().parents[1] / "strategy_state"
    ADAPTIVE_STATE_BASENAME = "breakout_v11_adaptive_grid_v8_high_water"

    @staticmethod
    def _split_position_adjustment(
        adjustment: Any,
    ) -> tuple[float | None, str | None, bool]:
        if isinstance(adjustment, tuple):
            stake, tag = adjustment
            return stake, tag, True
        return adjustment, None, False

    def _promote_untradeable_tail_to_full_exit(
        self,
        pair: str,
        trade: Any,
        current_exit_rate: float,
        min_stake: float | None,
        adjustment: Any,
    ) -> Any:
        """Avoid rejecting an otherwise valid partial exit because of dust.

        Freqtrade converts a requested stake reduction back to contract
        precision before checking the value of the remaining position.  A
        nearly complete Grid take-profit can therefore round to one remaining
        contract whose value is below Binance's minimum.  Freqtrade rejects
        the *whole* reduction in that case, leaving the profitable position
        untouched and retrying every loop.

        Promote only that boundary case to a complete reduction.  Normal
        partial exits, entries and strategy thresholds are unchanged.
        """

        requested_stake, order_tag, returned_tuple = (
            self._split_position_adjustment(adjustment)
        )
        if requested_stake is None or float(requested_stake) >= 0.0:
            return adjustment

        trade_stake = max(float(getattr(trade, "stake_amount", 0.0)), 0.0)
        trade_amount = max(float(getattr(trade, "amount", 0.0)), 0.0)
        if trade_stake <= 0.0 or trade_amount <= 0.0:
            return adjustment

        exit_stake = min(abs(float(requested_stake)), trade_stake)
        raw_remaining_stake = max(trade_stake - exit_stake, 0.0)
        if raw_remaining_stake <= 1e-12:
            return adjustment

        minimum = max(float(min_stake or 0.0), 0.0)
        should_promote = minimum > 0.0 and raw_remaining_stake < minimum
        remaining_notional: float | None = None
        minimum_exit_notional: float | None = None

        # Match Freqtrade's own post-precision dust check when the runtime
        # exchange adapter is available.  The raw-stake fallback above keeps
        # the guard deterministic in unit tests and unusual provider modes.
        exchange = getattr(getattr(self, "dp", None), "_exchange", None)
        if exchange is not None and current_exit_rate > 0.0:
            try:
                requested_amount = exchange.amount_to_contract_precision(
                    pair,
                    exit_stake * trade_amount / trade_stake,
                )
                remaining_amount = max(
                    trade_amount - float(requested_amount),
                    0.0,
                )
                minimum_exit_notional = exchange.get_min_pair_stake_amount(
                    pair,
                    current_exit_rate,
                    self.stoploss,
                    float(getattr(trade, "leverage", 1.0)),
                )
                remaining_notional = remaining_amount * current_exit_rate
                if (
                    minimum_exit_notional is not None
                    and remaining_notional > 0.0
                    and remaining_notional < float(minimum_exit_notional)
                ):
                    should_promote = True
            except (AttributeError, TypeError, ValueError, ArithmeticError):
                # Freqtrade will still apply its normal validation.  The
                # conservative raw-stake check remains available as fallback.
                pass

        if not should_promote:
            return adjustment

        logger.warning(
            "Promoting partial exit to full exit to avoid untradeable tail: "
            "trade_id=%s pair=%s tag=%s requested_stake=%.8f "
            "trade_stake=%.8f raw_remaining_stake=%.8f "
            "remaining_notional=%s minimum_exit=%s",
            getattr(trade, "id", None),
            pair,
            order_tag,
            exit_stake,
            trade_stake,
            raw_remaining_stake,
            (
                "unknown"
                if remaining_notional is None
                else f"{remaining_notional:.8f}"
            ),
            (
                "unknown"
                if minimum_exit_notional is None
                else f"{float(minimum_exit_notional):.8f}"
            ),
        )
        promoted = -trade_stake
        if returned_tuple:
            return promoted, order_tag
        return promoted

    def adjust_trade_position(
        self,
        trade: Any,
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
        adjustment = super().adjust_trade_position(
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
        return self._promote_untradeable_tail_to_full_exit(
            str(getattr(trade, "pair", "")),
            trade,
            current_exit_rate,
            min_stake,
            adjustment,
        )

    def _adaptive_state_mode(self) -> str | None:
        runmode = self.config.get("runmode")
        if runmode == RunMode.LIVE:
            return "live"
        if runmode == RunMode.DRY_RUN:
            return "dryrun"
        return None

    def _adaptive_state_path(self, mode: str) -> Path:
        return self.ADAPTIVE_STATE_DIR / (
            f"{self.ADAPTIVE_STATE_BASENAME}.{mode}.json"
        )

    @staticmethod
    def _state_error(path: Path, message: str) -> RuntimeError:
        return RuntimeError(f"v11 权益高水位状态无效：{path}：{message}")

    def _load_adaptive_high_water(self, mode: str) -> None:
        path = self._adaptive_state_path(mode)
        self._adaptive_last_persisted_peak = 0.0
        if not path.is_file():
            logger.info("v11 %s high-water state: new session", mode)
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            version = int(payload.get("version", 0))
            stored_mode = str(payload.get("mode") or "")
            peak = float(payload.get("peak_equity", 0.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise self._state_error(path, str(exc)) from exc
        if version != self.ADAPTIVE_STATE_VERSION:
            raise self._state_error(path, f"version={version}")
        if stored_mode != mode:
            raise self._state_error(path, f"mode={stored_mode!r}")
        if not math.isfinite(peak) or peak <= 0.0:
            raise self._state_error(path, f"peak_equity={peak!r}")
        self._peak_equity = max(float(self._peak_equity), peak)
        self._adaptive_last_persisted_peak = peak
        logger.info(
            "v11 %s high-water restored: %.8f USDT",
            mode,
            peak,
        )

    def _persist_adaptive_high_water(
        self,
        mode: str,
        current_time: datetime,
    ) -> None:
        peak = float(self._peak_equity)
        previous = float(
            getattr(self, "_adaptive_last_persisted_peak", 0.0)
        )
        if not math.isfinite(peak) or peak <= previous + 1e-9:
            return
        timestamp = current_time
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        path = self._adaptive_state_path(mode)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "version": self.ADAPTIVE_STATE_VERSION,
            "strategy": self.__class__.__name__,
            "mode": mode,
            "peak_equity": peak,
            "updated_at": timestamp.astimezone(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            try:
                os.write(descriptor, encoded)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if mode == "live":
                raise RuntimeError(
                    f"v11 LIVE 权益高水位无法安全保存，已拒绝开仓：{exc}"
                ) from exc
            logger.error("v11 DRY-RUN high-water save failed: %s", exc)
            return
        self._adaptive_last_persisted_peak = peak

    def bot_start(self, **kwargs: Any) -> None:
        super().bot_start(**kwargs)
        self._adaptive_last_persisted_peak = 0.0
        mode = self._adaptive_state_mode()
        if mode is not None:
            self._load_adaptive_high_water(mode)

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
        mode = self._adaptive_state_mode()
        if mode is not None:
            self._persist_adaptive_high_water(mode, current_time)
        return stake
