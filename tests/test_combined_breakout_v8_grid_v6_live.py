from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from crypto_scalper.binance_client import (
    BinanceFuturesClient,
    BinanceRateLimitError,
    SymbolRules,
)
from crypto_scalper.combined_breakout_v8_grid_v6_live import (
    CombinedBreakoutV8GridV6LiveTrader,
    _round_reduce_only_quantity,
    _transport_code_hashes,
)
from crypto_scalper.combined_volatility_trend_grid_backtest import (
    BREAKOUT_KEY,
    GRID_KEY,
)
from crypto_scalper.combined_volatility_trend_grid_shadow import _utc_now
from crypto_scalper.gui import (
    EXECUTION_MODE_DRY_RUN,
    EXECUTION_MODE_LIVE,
    LIVE_GUI_CONFIG_PATH,
    STRATEGY_MODE_COMBINED_LIVE,
    STRATEGY_MODE_COMBINED_SHADOW,
    TradingApp,
    _account_display_equity_baseline,
    _authorize_live_gui_session,
    _config_with_strategy_mode,
    _config_with_strategy_selection,
    _detect_strategy_mode,
    _execution_mode_for_config,
    _lock_live_authorization_for_persistence,
)
from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_trader import AccountSnapshot
from crypto_scalper.models import Candle, Direction
from crypto_scalper.trend_grid import TrendGridSignal
from crypto_scalper.trend_grid_optimize import GridCandidate
from crypto_scalper.volatility_breakout import DualThrustSignal
from crypto_scalper.volatility_breakout_optimize import Candidate, minute_token
from crypto_scalper.volatility_breakout_v4_research import V4MarketSnapshot


ROOT = Path(__file__).resolve().parents[1]
LIVE_CONFIG = ROOT / LIVE_GUI_CONFIG_PATH
LIVE_MANIFEST = (
    ROOT / "config.gui.breakout-v8-grid-v6-max2-live-manifest.json"
)


class RecordingClient(BinanceFuturesClient):
    def __init__(self) -> None:
        super().__init__("key", "secret", environment="testnet")
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, path, dict(params or {})))
        return {}


class FakeExchange:
    def __init__(self, now: datetime | None = None) -> None:
        self.api_key = "fake-key"
        self.api_secret = "fake-secret"
        self.now = now or datetime(2026, 7, 24, 12, 0)
        self.equity = 200.0
        self.position_amounts: dict[str, float] = {}
        self.entry_prices: dict[str, float] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.algo_orders: dict[str, dict[str, Any]] = {}
        self.market_calls: list[dict[str, Any]] = []
        self.limit_calls: list[dict[str, Any]] = []
        self.algo_calls: list[dict[str, Any]] = []
        self.candle_high = 100.0
        self.candle_low = 100.0
        self.candle_close = 100.0
        self.mark_price = 100.0
        self.entry_fill_ratio = 1.0
        self.reduce_only_fill_ratio = 1.0
        self.fail_open_orders = False
        self.prepared: list[tuple[str, str, str | int]] = []
        self.test_order_calls: list[dict[str, Any]] = []
        self._next_id = 1000
        self.account_calls = 0
        self.open_orders_calls = 0
        self.open_algo_orders_calls = 0

    def position_mode(self) -> bool:
        return False

    def set_margin_type(
        self, symbol: str, margin_type: str
    ) -> dict[str, Any]:
        self.prepared.append(("margin", symbol, margin_type))
        return {}

    def set_leverage(
        self, symbol: str, leverage: int
    ) -> dict[str, Any]:
        self.prepared.append(("leverage", symbol, leverage))
        return {}

    def test_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        *,
        new_client_order_id: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.test_order_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "clientOrderId": new_client_order_id,
            }
        )
        return {}

    def account(self) -> dict[str, Any]:
        self.account_calls += 1
        positions = []
        unrealized = 0.0
        for symbol, amount in self.position_amounts.items():
            if abs(amount) <= 1e-12:
                continue
            entry = self.entry_prices.get(symbol, 100.0)
            mark = 100.0
            pnl = amount * (mark - entry)
            unrealized += pnl
            positions.append(
                {
                    "symbol": symbol,
                    "positionAmt": str(amount),
                    "entryPrice": str(entry),
                    "markPrice": str(mark),
                    "unrealizedProfit": str(pnl),
                    "notional": str(amount * mark),
                    "leverage": "10",
                    "marginType": "cross",
                    "positionSide": "BOTH",
                    "liquidationPrice": "0",
                }
            )
        return {
            "totalMarginBalance": str(self.equity + unrealized),
            "totalWalletBalance": str(self.equity),
            "availableBalance": str(self.equity),
            "totalInitialMargin": "0",
            "totalMaintMargin": "0",
            "totalUnrealizedProfit": str(unrealized),
            "positions": positions,
        }

    def symbol_rules(self, symbol: str) -> SymbolRules:
        return SymbolRules(symbol, "0.001", "0.001", "0.01", "5")

    def klines(
        self, _symbol: str, interval: str, _limit: int
    ) -> list[Candle]:
        if interval == "1m":
            return [
                Candle(
                    self.now,
                    100.0,
                    self.candle_high,
                    self.candle_low,
                    self.candle_close,
                    1_000_000.0,
                )
            ]
        return []

    def premium_index(
        self, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            {
                "symbol": symbol or "BTCUSDT",
                "markPrice": str(self.mark_price),
            }
        ]

    def open_orders(
        self, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        self.open_orders_calls += 1
        if self.fail_open_orders:
            raise RuntimeError("simulated open-orders outage")
        return [
            dict(row)
            for row in self.orders.values()
            if row["status"] in {"NEW", "PARTIALLY_FILLED"}
            and (symbol is None or row["symbol"] == symbol)
        ]

    def open_algo_orders(
        self, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        self.open_algo_orders_calls += 1
        return [
            dict(row)
            for row in self.algo_orders.values()
            if row["algoStatus"] == "NEW"
            and (symbol is None or row["symbol"] == symbol)
        ]

    def query_order(
        self,
        symbol: str,
        *,
        order_id: int | str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is not None:
            row = next(
                (
                    candidate
                    for candidate in self.orders.values()
                    if str(candidate.get("orderId")) == str(order_id)
                ),
                None,
            )
        else:
            row = self.orders.get(str(orig_client_order_id))
        if row is None or row["symbol"] != symbol:
            raise RuntimeError("order not found")
        return dict(row)

    def query_algo_order(
        self,
        *,
        algo_id: int | str | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        del algo_id
        row = self.algo_orders.get(str(client_algo_id))
        if row is None:
            raise RuntimeError("algo order not found")
        return dict(row)

    def new_market_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        reduce_only: bool = False,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        client_id = str(new_client_order_id)
        if client_id in self.orders:
            return dict(self.orders[client_id])
        qty = float(quantity)
        amount = self.position_amounts.get(symbol, 0.0)
        fill_ratio = (
            self.reduce_only_fill_ratio
            if reduce_only
            else self.entry_fill_ratio
        )
        executed_qty = float(
            self.symbol_rules(symbol).round_quantity(
                qty * fill_ratio
            )
        )
        signed = (
            executed_qty
            if side.upper() == "BUY"
            else -executed_qty
        )
        if reduce_only:
            if amount > 0.0:
                amount = max(0.0, amount + signed)
            elif amount < 0.0:
                amount = min(0.0, amount + signed)
        else:
            amount += signed
            self.entry_prices[symbol] = 100.0
        self.position_amounts[symbol] = amount
        self._next_id += 1
        row = {
            "symbol": symbol,
            "side": side.upper(),
            "clientOrderId": client_id,
            "orderId": self._next_id,
            "status": (
                "FILLED"
                if reduce_only or self.entry_fill_ratio >= 1.0
                else "PARTIALLY_FILLED"
            ),
            "origQty": quantity,
            "executedQty": str(executed_qty),
            "avgPrice": "100",
            "reduceOnly": reduce_only,
        }
        self.orders[client_id] = row
        self.market_calls.append(dict(row))
        return dict(row)

    def new_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        price: str,
        *,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        del time_in_force
        client_id = str(new_client_order_id)
        if client_id in self.orders:
            return dict(self.orders[client_id])
        self._next_id += 1
        row = {
            "symbol": symbol,
            "side": side.upper(),
            "clientOrderId": client_id,
            "orderId": self._next_id,
            "status": "NEW",
            "origQty": quantity,
            "executedQty": "0",
            "avgPrice": "0",
            "price": price,
            "reduceOnly": reduce_only,
        }
        self.orders[client_id] = row
        self.limit_calls.append(dict(row))
        return dict(row)

    def _new_algo(
        self,
        kind: str,
        symbol: str,
        side: str,
        stop_price: str,
        quantity: str,
        reduce_only: bool,
        new_client_algo_id: str | None,
    ) -> dict[str, Any]:
        client_id = str(new_client_algo_id)
        if client_id in self.algo_orders:
            return dict(self.algo_orders[client_id])
        self._next_id += 1
        row = {
            "symbol": symbol,
            "side": side.upper(),
            "clientAlgoId": client_id,
            "algoId": self._next_id,
            "algoStatus": "NEW",
            "orderType": kind,
            "quantity": quantity,
            "triggerPrice": stop_price,
            "reduceOnly": reduce_only,
        }
        self.algo_orders[client_id] = row
        self.algo_calls.append(dict(row))
        return dict(row)

    def new_stop_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: str,
        quantity: str,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
        new_client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        del working_type
        return self._new_algo(
            "STOP_MARKET",
            symbol,
            side,
            stop_price,
            quantity,
            reduce_only,
            new_client_algo_id,
        )

    def new_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: str,
        quantity: str,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
        new_client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        del working_type
        return self._new_algo(
            "TAKE_PROFIT_MARKET",
            symbol,
            side,
            stop_price,
            quantity,
            reduce_only,
            new_client_algo_id,
        )

    def cancel_order(
        self,
        symbol: str,
        *,
        order_id: int | str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        del symbol, order_id
        row = self.orders[str(orig_client_order_id)]
        row["status"] = "CANCELED"
        return dict(row)

    def cancel_algo_order(
        self,
        *,
        algo_id: int | str | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        del algo_id
        row = self.algo_orders[str(client_algo_id)]
        row["algoStatus"] = "CANCELED"
        return dict(row)

    def fill_limit(self, client_id: str) -> None:
        row = self.orders[client_id]
        assert row["status"] in {"NEW", "PARTIALLY_FILLED"}
        already = float(row["executedQty"])
        remaining = float(row["origQty"]) - already
        row["status"] = "FILLED"
        row["executedQty"] = row["origQty"]
        row["avgPrice"] = row["price"]
        qty = remaining
        signed = qty if row["side"] == "BUY" else -qty
        symbol = row["symbol"]
        amount = self.position_amounts.get(symbol, 0.0)
        if row["reduceOnly"]:
            if amount > 0:
                amount = max(0.0, amount + signed)
            else:
                amount = min(0.0, amount + signed)
        else:
            amount += signed
        self.position_amounts[symbol] = amount

    def fill_limit_partial(
        self, client_id: str, executed_fraction: float
    ) -> None:
        row = self.orders[client_id]
        assert row["status"] == "NEW"
        qty = float(
            self.symbol_rules(row["symbol"]).round_quantity(
                float(row["origQty"]) * executed_fraction
            )
        )
        row["status"] = "PARTIALLY_FILLED"
        row["executedQty"] = str(qty)
        row["avgPrice"] = row["price"]
        signed = qty if row["side"] == "BUY" else -qty
        symbol = row["symbol"]
        amount = self.position_amounts.get(symbol, 0.0)
        if row["reduceOnly"]:
            if amount > 0:
                amount = max(0.0, amount + signed)
            else:
                amount = min(0.0, amount + signed)
        else:
            amount += signed
        self.position_amounts[symbol] = amount


def _armed_config(tmp_path: Path):
    config = load_live_config(LIVE_CONFIG)
    live = replace(
        config.combined_breakout_v8_grid_v6_live,
        armed=True,
        live_confirmation_text=(
            config.combined_breakout_v8_grid_v6_live
            .required_live_confirmation_text
        ),
        state_path=str(tmp_path / "live-state.json"),
        event_log_path=str(tmp_path / "live-events.jsonl"),
        report_path=str(tmp_path / "live-report.json"),
    )
    return replace(
        config,
        exchange=replace(config.exchange, environment="testnet"),
        combined_breakout_v8_grid_v6_live=live,
    )


def _snapshot(symbol: str, now: datetime) -> V4MarketSnapshot:
    return V4MarketSnapshot(
        available_minute=minute_token(now),
        symbol=symbol,
        btc_return_4h=0.0,
        eth_return_4h=0.0,
        breadth_above_ema21=0.5,
        breadth_change_4h=0.0,
        symbol_return_4h=0.0,
        symbol_efficiency_12h=0.0,
        market_efficiency_12h=0.5,
        symbol_ema55_atr=-1.0,
    )


def _breakout_candidate(symbol: str, now: datetime) -> Candidate:
    return Candidate(
        DualThrustSignal(
            event_id=f"bo-{symbol}",
            symbol=symbol,
            direction=Direction.LONG,
            signal_bar_time=now,
            signal_available_time=now,
            raw_signal_price=100.0,
            session_open=99.0,
            upper_band=99.5,
            lower_band=95.0,
            dual_thrust_range=5.0,
            atr_value=2.0,
            trend_alignment_atr=1.0,
            volume_ratio=2.2,
            body_atr=3.0,
            directional_close_position=0.8,
            range_atr=2.0,
            breakout_extension_atr=1.0,
            quality_score=2.0,
        ),
        minute_token(now),
        btc_return_4h=0.0,
    )


def _grid_candidate(symbol: str, now: datetime) -> GridCandidate:
    return GridCandidate(
        TrendGridSignal(
            event_id=f"grid-{symbol}",
            symbol=symbol,
            direction=Direction.SHORT,
            signal_bar_time=now,
            signal_available_time=now,
            raw_signal_price=100.0,
            atr_value=2.0,
            fast_ema=99.0,
            slow_ema=101.0,
            fast_slope_atr=-0.2,
            slow_slope_atr=-0.1,
            alignment_atr=0.6,
            extension_atr=0.3,
            directional_close_position=0.8,
            volume_ratio=1.0,
            quality_score=0.6,
        ),
        minute_token(now),
    )


def _install_context(
    trader: CombinedBreakoutV8GridV6LiveTrader,
    now: datetime,
    *symbols: str,
) -> None:
    minute = minute_token(now)
    trader._market_context = {
        minute - minute % 60: {
            symbol: _snapshot(symbol, now) for symbol in symbols
        }
    }


def test_live_config_is_independent_and_locked_by_default(
    tmp_path: Path,
) -> None:
    config = load_live_config(LIVE_CONFIG)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        replace(
            config,
            combined_breakout_v8_grid_v6_live=replace(
                config.combined_breakout_v8_grid_v6_live,
                state_path=str(tmp_path / "state.json"),
                event_log_path=str(tmp_path / "events.jsonl"),
                report_path=str(tmp_path / "report.json"),
            ),
        ),
        FakeExchange(),
    )

    assert _detect_strategy_mode(config) == STRATEGY_MODE_COMBINED_LIVE
    assert config.trading.dry_run is False
    assert config.combined_breakout_v8_grid_v6_live.armed is False
    with pytest.raises(RuntimeError, match="locked"):
        trader.validate_startup_settings_only()
    assert not trader.state_path.exists()


def test_live_candidate_gate_maps_the_live_drawdown_limit(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path),
        exchange,
    )
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    trader.state["peak_equity"] = exchange.equity
    _install_context(trader, now, "BTCUSDT")
    candidate = _breakout_candidate("BTCUSDT", now)

    assert (
        trader.live.hard_drawdown_stop_pct
        == trader.live.max_drawdown_pct
        == 0.60
    )
    assert (
        trader._candidate_reject_reason(
            BREAKOUT_KEY,
            candidate,
            now,
        )
        is None
    )

    trader.state["peak_equity"] = 200.0
    exchange.equity = 79.0
    assert (
        trader._candidate_reject_reason(
            BREAKOUT_KEY,
            candidate,
            now,
        )
        == "hard_drawdown_stop"
    )


def test_live_acceptance_requires_full_authenticated_stress_run(
    tmp_path: Path,
) -> None:
    config = load_live_config(LIVE_CONFIG)
    acceptance_path = tmp_path / "transport-acceptance.json"
    live = replace(
        config.combined_breakout_v8_grid_v6_live,
        state_path=str(tmp_path / "live-state.json"),
        event_log_path=str(tmp_path / "live-events.jsonl"),
        report_path=str(tmp_path / "live-report.json"),
        transport_acceptance_report_path=str(acceptance_path),
    )
    trader = CombinedBreakoutV8GridV6LiveTrader(
        replace(
            config,
            combined_breakout_v8_grid_v6_live=live,
        ),
        FakeExchange(),
    )
    report = {
        "schema_version": 2,
        "transport_version": live.transport_version,
        "environment": "mainnet",
        "mode": "MAINNET_DRY_RUN_PUBLIC_ONLY",
        "user_stream_required": False,
        "passed": True,
        "duration_seconds": 45.0,
        "cycles": 200,
        "criteria": {"user_stream_connected": True},
        "live_config_hash": trader.config_hash,
        "strategy_source_hashes": trader.source_bundle["hashes"],
        "transport_code_hashes": _transport_code_hashes(),
    }
    acceptance_path.write_text(
        json.dumps(report), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="user stream"):
        trader.validate_transport_acceptance()

    report.update(
        {
            "mode": "MAINNET_DRY_RUN_NO_ORDER_WITH_USER_STREAM",
            "user_stream_required": True,
            "duration_seconds": 10.0,
        }
    )
    acceptance_path.write_text(
        json.dumps(report), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="full stress run"):
        trader.validate_transport_acceptance()

    report["duration_seconds"] = 45.0
    acceptance_path.write_text(
        json.dumps(report), encoding="utf-8"
    )
    trader.validate_transport_acceptance()
    assert (
        trader.state["transport_acceptance"]["report_path"]
        == str(acceptance_path)
    )


def test_live_manifest_hashes_match_frozen_files() -> None:
    manifest = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))

    for relative, expected in manifest["hashes"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_live_risk_profile_matches_frozen_backtest_envelope() -> None:
    config = load_live_config(LIVE_CONFIG)
    live = config.combined_breakout_v8_grid_v6_live
    combined = json.loads(
        (ROOT / live.source_combined_config_path).read_text(
            encoding="utf-8"
        )
    )
    breakout = json.loads(
        (
            ROOT
            / combined["source_configs"]["volatility_breakout"]["path"]
        ).read_text(encoding="utf-8")
    )
    grid = json.loads(
        (
            ROOT
            / combined["source_configs"][
                "dynamic_trend_following_grid"
            ]["path"]
        ).read_text(encoding="utf-8")
    )

    assert live.risk_scale == 1.0
    assert (
        live.max_gross_notional_multiple
        == combined["portfolio"]["max_gross_notional_multiple"]
        == 9.0
    )
    assert (
        live.max_drawdown_pct
        == combined["portfolio"]["hard_drawdown_stop_pct"]
        == 0.60
    )
    assert live.max_daily_loss_pct == 1.0
    assert config.risk.max_account_margin_usage_pct == 1.0
    assert config.risk.max_symbol_margin_pct == 1.0
    assert config.risk.min_available_balance_usdt == 0.0
    assert (
        breakout["operational_portfolio"]["max_notional_multiple"]
        * live.risk_scale
        == 9.0
    )
    assert (
        grid["portfolio"]["max_notional_multiple"]
        * live.risk_scale
        == 5.0
    )
    assert live.order_id_prefix == "b8g6r1"
    assert "risk1" in live.state_path
    assert "risk1" in live.event_log_path
    assert "risk1" in live.report_path


def test_dry_run_and_live_use_distinct_mainnet_ledgers() -> None:
    dry = load_live_config(
        ROOT / "config.gui.breakout-v8-grid-v6-max2-shadow.json"
    )
    live = load_live_config(LIVE_CONFIG)
    dry_state = (
        dry.combined_volatility_trend_grid_shadow.state_path
    )
    live_state = (
        live.combined_breakout_v8_grid_v6_live.state_path.format(
            environment=live.exchange.environment
        )
    )

    assert dry.exchange.environment == "mainnet"
    assert live.exchange.environment == "mainnet"
    assert dry.trading.dry_run is True
    assert live.trading.dry_run is False
    assert dry_state != live_state


def test_live_config_pins_frozen_combined_source_hash(
    tmp_path: Path,
) -> None:
    config = _armed_config(tmp_path)
    live = replace(
        config.combined_breakout_v8_grid_v6_live,
        source_combined_config_sha256="0" * 64,
    )
    trader = CombinedBreakoutV8GridV6LiveTrader(
        replace(
            config,
            combined_breakout_v8_grid_v6_live=live,
        ),
        FakeExchange(),
    )

    with pytest.raises(RuntimeError, match="source hash differs"):
        trader.validate_startup_settings_only()


def test_live_confirmation_contract_cannot_be_configured_away(
    tmp_path: Path,
) -> None:
    config = _armed_config(tmp_path)
    live = replace(
        config.combined_breakout_v8_grid_v6_live,
        live_confirmation_text="",
        required_live_confirmation_text="",
        runtime_confirmation_text="",
    )
    trader = CombinedBreakoutV8GridV6LiveTrader(
        replace(
            config,
            combined_breakout_v8_grid_v6_live=live,
        ),
        FakeExchange(),
    )

    with pytest.raises(RuntimeError, match="confirmation contract changed"):
        trader.validate_startup_settings_only()


def test_live_one_way_and_protective_contract_cannot_be_disabled(
    tmp_path: Path,
) -> None:
    config = _armed_config(tmp_path)
    config = replace(
        config,
        trading=replace(
            config.trading,
            require_one_way_mode=False,
            use_protective_orders=False,
        ),
    )
    trader = CombinedBreakoutV8GridV6LiveTrader(
        config, FakeExchange()
    )

    with pytest.raises(RuntimeError, match="requires One-way"):
        trader.validate_startup_settings_only()


def test_gui_mode_switch_cannot_blend_live_and_shadow() -> None:
    live_config = load_live_config(LIVE_CONFIG)
    dry = _config_with_strategy_mode(
        live_config, STRATEGY_MODE_COMBINED_SHADOW
    )
    dry = _config_with_strategy_selection(
        dry, indicator_enabled=True, vbp_enabled=True
    )
    assert dry.trading.dry_run is True
    assert dry.combined_volatility_trend_grid_shadow.enabled is True
    assert dry.combined_breakout_v8_grid_v6_live.enabled is False

    live = _config_with_strategy_mode(
        live_config, STRATEGY_MODE_COMBINED_LIVE
    )
    live = _config_with_strategy_selection(
        live, indicator_enabled=True, vbp_enabled=True
    )
    assert live.trading.dry_run is False
    assert live.combined_breakout_v8_grid_v6_live.enabled is True
    assert live.combined_volatility_trend_grid_shadow.enabled is False
    assert live.dual_thrust_shadow.enabled is False
    assert live.vbp_strategy.enabled is False

    manually_changed_environment = replace(
        live_config,
        exchange=replace(
            live_config.exchange,
            environment="testnet",
            api_key_env="SHOULD_NOT_SURVIVE",
            api_secret_env="SHOULD_NOT_SURVIVE",
        ),
    )
    for mode in (
        STRATEGY_MODE_COMBINED_SHADOW,
        STRATEGY_MODE_COMBINED_LIVE,
    ):
        mainnet_only = _config_with_strategy_mode(
            manually_changed_environment, mode
        )
        assert mainnet_only.exchange.environment == "mainnet"
        assert (
            mainnet_only.exchange.api_key_env
            == "BINANCE_FUTURES_API_KEY"
        )
        assert (
            mainnet_only.exchange.api_secret_env
            == "BINANCE_FUTURES_API_SECRET"
        )


def test_gui_never_persists_live_runtime_authorization() -> None:
    config = _armed_config(Path("/tmp/unused-v8-v6-live-test"))
    config = replace(
        config,
        trading=replace(
            config.trading,
            mainnet_confirmation_text="CONFIRM_MAINNET",
        ),
    )

    locked = _lock_live_authorization_for_persistence(config)

    assert locked.combined_breakout_v8_grid_v6_live.armed is False
    assert (
        locked.combined_breakout_v8_grid_v6_live.live_confirmation_text
        == ""
    )
    assert locked.trading.mainnet_confirmation_text == ""


def test_simplified_gui_routes_modes_and_authorizes_only_this_run() -> None:
    dry = load_live_config(
        ROOT / "config.gui.breakout-v8-grid-v6-max2-shadow.json"
    )
    live = load_live_config(LIVE_CONFIG)

    assert _execution_mode_for_config(dry) == EXECUTION_MODE_DRY_RUN
    assert _execution_mode_for_config(live) == EXECUTION_MODE_LIVE
    assert dry.macro_events.enabled is False
    assert live.macro_events.enabled is False

    authorized = _authorize_live_gui_session(live)
    assert authorized.trading.mainnet_confirmation_text == "CONFIRM_MAINNET"
    assert authorized.combined_breakout_v8_grid_v6_live.armed is True
    assert (
        authorized.combined_breakout_v8_grid_v6_live.live_confirmation_text
        == "CONFIRM_BREAKOUT_V8_GRID_V6_LIVE"
    )

    locked = _lock_live_authorization_for_persistence(authorized)
    assert locked.trading.mainnet_confirmation_text == ""
    assert locked.combined_breakout_v8_grid_v6_live.armed is False
    assert (
        locked.combined_breakout_v8_grid_v6_live.live_confirmation_text
        == ""
    )


def test_gui_account_profit_baselines_are_separate_from_risk_capital() -> None:
    dry = load_live_config(
        ROOT / "config.gui.breakout-v8-grid-v6-max2-shadow.json"
    )
    live = load_live_config(LIVE_CONFIG)

    assert dry.risk.starting_capital_usdt == 200.0
    assert live.risk.starting_capital_usdt == 200.0
    assert _account_display_equity_baseline(dry, 289.0) == 200.0
    assert _account_display_equity_baseline(live, None) == 200.0
    assert _account_display_equity_baseline(live, 289.0) == 289.0
    assert live.risk.starting_capital_usdt == 200.0


def test_live_account_refresh_renders_zero_profit_without_changing_risk() -> None:
    live = load_live_config(LIVE_CONFIG)
    values = {
        key: MagicMock()
        for key in (
            "equity",
            "available",
            "unrealized",
            "capital_pnl",
            "initial_margin_usage",
            "position_count",
        )
    }
    positions = MagicMock()
    positions.get_children.return_value = ()
    app = SimpleNamespace(
        _live_display_equity_baseline=None,
        _read_config=lambda: live,
        log=MagicMock(),
        summary_vars=values,
        summary_labels={"equity": MagicMock()},
        _set_summary_value_style=MagicMock(),
        status_label=MagicMock(),
        status_var=MagicMock(),
        positions=positions,
    )
    snapshot = AccountSnapshot(
        equity=289.0,
        wallet_balance=289.0,
        available_balance=289.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        total_unrealized_pnl=0.0,
        positions={},
        position_rows=(),
        position_mode="ONE-WAY",
    )

    TradingApp._render_account(
        app,
        snapshot,
        reset_live_display_baseline=True,
    )

    values["equity"].set.assert_called_once_with("289.00 U")
    values["capital_pnl"].set.assert_called_once_with("+0.00 U")
    assert app._live_display_equity_baseline == 289.0
    assert live.risk.starting_capital_usdt == 200.0


def test_live_refresh_requests_a_new_gui_profit_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import crypto_scalper.gui as gui_module

    live = load_live_config(LIVE_CONFIG)
    snapshot = AccountSnapshot(
        equity=289.0,
        wallet_balance=289.0,
        available_balance=289.0,
        initial_margin=0.0,
        maintenance_margin=0.0,
        total_unrealized_pnl=0.0,
        positions={},
        position_rows=(),
        position_mode="ONE-WAY",
    )

    class FakeLiveTrader:
        state_path = Path("/tmp/gui-live-display-test.json")

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def snapshot_account(self) -> AccountSnapshot:
            return snapshot

    monkeypatch.setattr(
        gui_module,
        "CombinedBreakoutV8GridV6LiveTrader",
        FakeLiveTrader,
    )
    app = SimpleNamespace(
        _client_for_config=lambda _config: SimpleNamespace(
            api_key="fake-key"
        ),
        log_from_thread=MagicMock(),
        account_from_thread=MagicMock(),
    )

    TradingApp._refresh_account_worker(app, live)

    app.account_from_thread.assert_called_once_with(
        snapshot,
        reset_live_display_baseline=True,
    )


def test_gui_log_display_and_file_begin_with_timestamp(
    tmp_path: Path,
) -> None:
    app = SimpleNamespace(
        _log_file_path=tmp_path / "gui.log",
        log_text=MagicMock(),
    )

    TradingApp.log(app, "测试日志")

    displayed = app.log_text.insert.call_args.args[1]
    datetime.strptime(displayed[1:20], "%Y-%m-%d %H:%M:%S")
    assert displayed[0] == "["
    assert displayed[20:] == "] 测试日志\n"
    assert app._log_file_path.read_text(encoding="utf-8") == displayed


def test_binance_client_supports_idempotent_query_cancel_and_test_order() -> None:
    client = RecordingClient()
    client.query_order("btcusdt", orig_client_order_id="b8g6-test")
    client.cancel_order("btcusdt", orig_client_order_id="b8g6-test")
    client.query_algo_order(client_algo_id="b8g6-stop")
    client.cancel_algo_order(client_algo_id="b8g6-stop")
    client.new_limit_order(
        "btcusdt",
        "sell",
        "0.1",
        "101",
        reduce_only=True,
        new_client_order_id="b8g6-tp",
    )
    client.new_stop_market_order(
        "btcusdt",
        "sell",
        "99",
        "0.1",
        new_client_algo_id="b8g6-stop",
    )
    client.test_order(
        "btcusdt",
        "buy",
        "0.1",
        new_client_order_id="b8g6-test-order",
    )
    client.income_history(
        "btcusdt",
        start_time=1000,
        end_time=2000,
    )

    assert client.calls[0] == (
        "GET",
        "/fapi/v1/order",
        {"symbol": "BTCUSDT", "origClientOrderId": "b8g6-test"},
    )
    assert client.calls[1][0:2] == ("DELETE", "/fapi/v1/order")
    assert client.calls[2] == (
        "GET",
        "/fapi/v1/algoOrder",
        {"clientAlgoId": "b8g6-stop"},
    )
    assert client.calls[3][0:2] == ("DELETE", "/fapi/v1/algoOrder")
    assert client.calls[4][2]["reduceOnly"] == "true"
    assert client.calls[4][2]["newClientOrderId"] == "b8g6-tp"
    assert client.calls[5][2]["clientAlgoId"] == "b8g6-stop"
    assert client.calls[6][0:2] == ("POST", "/fapi/v1/order/test")
    assert client.calls[7] == (
        "GET",
        "/fapi/v1/income",
        {
            "limit": 1000,
            "symbol": "BTCUSDT",
            "startTime": 1000,
            "endTime": 2000,
        },
    )


def test_canceled_algo_order_cannot_masquerade_as_live_protection(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    client_id = "b8g6-bo-stop-1-deadbeef"
    exchange._new_algo(
        "STOP_MARKET",
        "BTCUSDT",
        "SELL",
        "99",
        "0.1",
        True,
        client_id,
    )
    exchange.algo_orders[client_id]["algoStatus"] = "CANCELED"

    with pytest.raises(RuntimeError, match="is CANCELED"):
        trader._algo_order_idempotent(
            kind="stop",
            symbol="BTCUSDT",
            side="SELL",
            quantity="0.1",
            trigger_price="99",
            client_id=client_id,
        )


def test_breakout_live_entry_is_idempotent_protected_and_restart_safe(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    config = _armed_config(tmp_path)
    trader = CombinedBreakoutV8GridV6LiveTrader(config, exchange)
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")

    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    assert exchange.prepared == [
        ("margin", "BTCUSDT", "CROSSED"),
        ("leverage", "BTCUSDT", 10),
    ]
    assert len(exchange.test_order_calls) == 1
    entry_calls = [
        row for row in exchange.market_calls if not row["reduceOnly"]
    ]
    assert len(entry_calls) == 1
    assert exchange.position_amounts["BTCUSDT"] > 0
    assert {row["orderType"] for row in exchange.open_algo_orders()} == {
        "STOP_MARKET",
        "TAKE_PROFIT_MARKET",
    }
    assert all(row["reduceOnly"] for row in exchange.algo_calls)

    same_id = entry_calls[0]["clientOrderId"]
    replay = trader._market_order_idempotent(
        symbol="BTCUSDT",
        side="BUY",
        quantity=entry_calls[0]["origQty"],
        reduce_only=False,
        client_id=same_id,
    )
    assert replay["status"] == "FILLED"
    assert len(
        [row for row in exchange.market_calls if not row["reduceOnly"]]
    ) == 1

    restored = CombinedBreakoutV8GridV6LiveTrader(config, exchange)
    restored.validate_startup()
    assert restored.state["breakout_exchange"]["symbol"] == "BTCUSDT"
    assert restored.state["operational_halt"] is False


def test_filled_market_response_with_zero_fill_fields_is_requeried(
    tmp_path: Path,
) -> None:
    class DelayedFinalFillExchange(FakeExchange):
        def new_market_order(
            self,
            symbol: str,
            side: str,
            quantity: str,
            reduce_only: bool = False,
            new_client_order_id: str | None = None,
        ) -> dict[str, Any]:
            final = super().new_market_order(
                symbol,
                side,
                quantity,
                reduce_only=reduce_only,
                new_client_order_id=new_client_order_id,
            )
            if reduce_only:
                return final
            immediate = dict(final)
            immediate["executedQty"] = "0"
            immediate["avgPrice"] = "0"
            return immediate

    now = _utc_now().replace(second=0, microsecond=0)
    exchange = DelayedFinalFillExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")

    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )

    assert trader.state["operational_halt"] is False
    assert trader.state["breakout_exchange"]["entry_price"] == 100.0
    assert len(
        [row for row in exchange.market_calls if not row["reduceOnly"]]
    ) == 1
    events = [
        json.loads(line)
        for line in trader.event_log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    confirmation = [
        row
        for row in events
        if row["event_type"]
        == "market_fill_confirmed_after_query"
    ]
    assert len(confirmation) == 1
    assert confirmation[0]["attempts"] == 2


def test_filled_market_response_can_derive_price_from_cumulative_quote(
    tmp_path: Path,
) -> None:
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), FakeExchange()
    )

    quantity, price = trader._filled_order(
        {
            "status": "FILLED",
            "executedQty": "5",
            "avgPrice": "0",
            "cumQuote": "501.25",
        }
    )

    assert quantity == 5.0
    assert price == pytest.approx(100.25)


def test_reduce_only_quantity_snaps_only_float_step_artifact() -> None:
    rules = SymbolRules("TRUMPUSDT", "0.01", "0.01", "0.001", "5")
    artifact = 30.9 + 30.9 + 30.9

    assert artifact == 92.69999999999999
    assert rules.round_quantity(artifact) == "92.69"
    assert _round_reduce_only_quantity(rules, artifact) == "92.7"
    assert _round_reduce_only_quantity(rules, 92.699) == "92.69"


def test_terminal_standard_and_finished_algo_cancel_are_idempotent(
    tmp_path: Path,
) -> None:
    class TerminalCancelExchange(FakeExchange):
        def __init__(self) -> None:
            super().__init__()
            self.standard_cancel_calls = 0
            self.algo_cancel_calls = 0

        def cancel_order(
            self,
            symbol: str,
            *,
            order_id: int | str | None = None,
            orig_client_order_id: str | None = None,
        ) -> dict[str, Any]:
            del symbol, order_id, orig_client_order_id
            self.standard_cancel_calls += 1
            raise RuntimeError("Unknown order sent")

        def cancel_algo_order(
            self,
            *,
            algo_id: int | str | None = None,
            client_algo_id: str | None = None,
        ) -> dict[str, Any]:
            del algo_id, client_algo_id
            self.algo_cancel_calls += 1
            raise RuntimeError("Unknown order sent")

    exchange = TerminalCancelExchange()
    exchange.orders["terminal-standard"] = {
        "symbol": "TRUMPUSDT",
        "side": "BUY",
        "clientOrderId": "terminal-standard",
        "orderId": 101,
        "status": "EXPIRED",
        "origQty": "1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": True,
    }
    exchange.algo_orders["terminal-algo"] = {
        "symbol": "TRUMPUSDT",
        "side": "BUY",
        "clientAlgoId": "terminal-algo",
        "algoId": 102,
        "algoStatus": "FINISHED",
        "orderType": "STOP_MARKET",
        "quantity": "1",
        "triggerPrice": "1.59",
        "reduceOnly": True,
    }
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )

    trader._safe_cancel_standard(
        "TRUMPUSDT", "terminal-standard"
    )
    trader._safe_cancel_algo("terminal-algo")

    assert exchange.standard_cancel_calls == 0
    assert exchange.algo_cancel_calls == 0
    skipped = [
        row
        for row in trader.state["order_journal"]
        if row["action"] == "terminal_order_cancel_skipped"
    ]
    assert {row["status"] for row in skipped} == {
        "EXPIRED",
        "FINISHED",
    }


def test_finished_protective_stop_residual_is_verified_and_flattened(
    tmp_path: Path,
) -> None:
    class CentStepExchange(FakeExchange):
        def symbol_rules(self, symbol: str) -> SymbolRules:
            return SymbolRules(
                symbol, "0.01", "0.01", "0.001", "5"
            )

    now = _utc_now().replace(second=0, microsecond=0)
    exchange = CentStepExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "ETHUSDT")
    assert trader._open_grid_candidate(
        _grid_candidate("ETHUSDT", now), now
    )

    campaign = trader.state["grid_campaign"]
    first_lot = next(iter(campaign["lots"].values()))
    first_lot["quantity"] = 92.7
    campaign["levels"][0]["quantity"] = 92.7
    trader.state["grid_campaign"] = campaign
    exchange.position_amounts["ETHUSDT"] = -0.01
    exchange.entry_prices["ETHUSDT"] = 1.561

    protection = trader.state["protective_orders"][
        GRID_KEY
    ]
    stop_client_id = protection["stop_client_id"]
    stop = exchange.algo_orders[stop_client_id]
    stop["algoStatus"] = "FINISHED"
    stop["quantity"] = "92.69"
    exchange._next_id += 1
    stop_actual_order_id = exchange._next_id
    stop["actualOrderId"] = stop_actual_order_id
    exchange.orders["exchange-stop-child"] = {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "clientOrderId": "exchange-stop-child",
        "orderId": stop_actual_order_id,
        "status": "FILLED",
        "origQty": "92.69",
        "executedQty": "92.69",
        "avgPrice": "1.591",
        "cumQuote": str(92.69 * 1.591),
        "reduceOnly": True,
    }

    account = trader.reconcile()

    assert "ETHUSDT" not in account.positions
    assert exchange.position_amounts["ETHUSDT"] == pytest.approx(0.0)
    assert trader.state["grid_campaign"] is None
    assert trader.state["grid_exchange"] is None
    assert trader.state["pending_orders"] == {}
    assert trader.state["protective_orders"] == {}
    assert trader.state["operational_halt"] is False
    assert trader.state["circuit_breaker"] is False
    residual_exits = [
        row
        for row in exchange.market_calls
        if row["reduceOnly"]
        and "-resid-" in row["clientOrderId"]
    ]
    assert len(residual_exits) == 1
    assert residual_exits[0]["origQty"] == "0.01"
    assert (
        trader.state["trades"][-1]["exit_reason"]
        == "protective_stop_residual_closed"
    )
    events = [
        json.loads(line)
        for line in trader.event_log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert any(
        row["event_type"] == "protective_stop_residual_closed"
        and row["stop_fill_quantity"] == pytest.approx(92.69)
        and row["residual_quantity"] == pytest.approx(0.01)
        for row in events
    )


def test_verified_flat_emergency_round_trip_clears_only_reviewed_halt(
    tmp_path: Path,
) -> None:
    class TradeHistoryExchange(FakeExchange):
        def user_trades(
            self,
            symbol: str | None = None,
            limit: int = 1000,
            start_time: int | None = None,
            end_time: int | None = None,
        ) -> list[dict[str, Any]]:
            del limit, start_time, end_time
            rows = []
            for index, order in enumerate(self.market_calls, start=1):
                if symbol is not None and order["symbol"] != symbol:
                    continue
                quantity = float(order["executedQty"])
                price = float(order["avgPrice"])
                rows.append(
                    {
                        "symbol": order["symbol"],
                        "id": index,
                        "orderId": order["orderId"],
                        "side": order["side"],
                        "price": str(price),
                        "qty": str(quantity),
                        "quoteQty": str(quantity * price),
                        "commission": str(
                            quantity * price * 0.0005
                        ),
                        "commissionAsset": "USDT",
                        "realizedPnl": (
                            "-0.01" if order["reduceOnly"] else "0"
                        ),
                    }
                )
            return rows

    now = _utc_now().replace(second=0, microsecond=0)
    exchange = TradeHistoryExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    model = trader.state["breakout_position"]
    entry = trader.state["breakout_exchange"]
    entry_id = entry["entry_client_id"]
    event_id = entry["event_id"]
    emergency_id = trader._client_id(
        BREAKOUT_KEY, event_id, "emerg", 0
    )
    quantity = float(entry["quantity"])
    exchange.new_market_order(
        "BTCUSDT",
        "SELL",
        str(quantity),
        reduce_only=True,
        new_client_order_id=emergency_id,
    )
    for row in exchange.algo_orders.values():
        row["algoStatus"] = "CANCELED"
    trader.state["breakout_position"] = None
    trader.state["breakout_exchange"] = None
    trader.state["protective_orders"] = {}
    trader.state["pending_orders"] = {
        entry_id: {
            "strategy": BREAKOUT_KEY,
            "event_id": event_id,
            "symbol": "BTCUSDT",
            "direction": Direction.LONG.name,
            "client_id": entry_id,
            "created_at": now.isoformat(),
            "exchange_status": "FILLED",
            "model": model,
        }
    }
    trader.state["operational_halt"] = True
    trader.state["halt_reason"] = (
        "BTCUSDT entry fill was not final: test | "
        "BTCUSDT protective stop placement failed: test | "
        "BTCUSDT filled entry has no exchange position; "
        "trade-history review required"
    )

    result = trader.recover_verified_flat_emergency_round_trip(
        expected_entry_client_id=entry_id,
        expected_emergency_client_id=emergency_id,
    )

    assert result["cleared"] is True
    assert result["already_recovered"] is False
    assert trader.state["operational_halt"] is False
    assert trader.state["halt_reason"] == ""
    assert trader.state["pending_orders"] == {}
    assert len(trader.state["recovery_journal"]) == 1
    assert (
        trader.state["recovery_journal"][0]["entry_client_id"]
        == entry_id
    )
    assert trader.state["trades"][-1]["pnl_source"] == (
        "verified_exchange_trade_history"
    )
    assert trader.state["daily_entries"][BREAKOUT_KEY]
    assert "BTCUSDT" in trader.state["cooldown_until"][BREAKOUT_KEY]
    assert (
        trader.state["seen_events"][BREAKOUT_KEY][event_id]["status"]
        == "emergency_flattened_after_execution_fault"
    )
    repeated = trader.recover_verified_flat_emergency_round_trip(
        expected_entry_client_id=entry_id,
        expected_emergency_client_id=emergency_id,
    )
    assert repeated["already_recovered"] is True


def test_flat_emergency_recovery_refuses_unrelated_halt(
    tmp_path: Path,
) -> None:
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), FakeExchange()
    )
    trader.state["operational_halt"] = True
    trader.state["halt_reason"] = "BTCUSDT direction differs from live ledger"
    trader.state["pending_orders"] = {
        "reviewed-entry": {
            "strategy": BREAKOUT_KEY,
            "event_id": "event",
            "symbol": "BTCUSDT",
            "direction": Direction.LONG.name,
            "client_id": "reviewed-entry",
            "created_at": _utc_now().isoformat(),
            "model": {},
        }
    }

    with pytest.raises(RuntimeError, match="unrelated recovery reason"):
        trader.recover_verified_flat_emergency_round_trip(
            expected_entry_client_id="reviewed-entry"
        )

    assert trader.state["operational_halt"] is True
    assert "reviewed-entry" in trader.state["pending_orders"]


def test_verified_flat_protective_stop_incident_recovery_is_strict(
    tmp_path: Path,
) -> None:
    class IncidentExchange(FakeExchange):
        def __init__(self) -> None:
            super().__init__()
            self.trade_rows: list[dict[str, Any]] = []
            self.income_rows: list[dict[str, Any]] = []

        def symbol_rules(self, symbol: str) -> SymbolRules:
            return SymbolRules(
                symbol, "0.01", "0.01", "0.001", "5"
            )

        def user_trades(
            self,
            symbol: str | None = None,
            limit: int = 1000,
            start_time: int | None = None,
            end_time: int | None = None,
        ) -> list[dict[str, Any]]:
            del limit, start_time, end_time
            return [
                dict(row)
                for row in self.trade_rows
                if symbol is None or row["symbol"] == symbol
            ]

        def income_history(
            self,
            symbol: str | None = None,
            income_type: str | None = None,
            limit: int = 1000,
            start_time: int | None = None,
            end_time: int | None = None,
        ) -> list[dict[str, Any]]:
            del limit, start_time, end_time
            return [
                dict(row)
                for row in self.income_rows
                if (symbol is None or row["symbol"] == symbol)
                and (
                    income_type is None
                    or row["incomeType"] == income_type
                )
            ]

    exchange = IncidentExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    symbol = "TRUMPUSDT"
    event_id = "3926b7712530497f5b36aca7"
    entry_ids = [
        trader._client_id(GRID_KEY, event_id, "entry", 0),
        trader._client_id(GRID_KEY, event_id, "ge1", 0),
        trader._client_id(GRID_KEY, event_id, "ge2", 0),
    ]
    take_profit_ids = [
        trader._client_id(GRID_KEY, event_id, "tp0", 0),
        trader._client_id(GRID_KEY, event_id, "tp1", 0),
        trader._client_id(GRID_KEY, event_id, "tp2", 0),
    ]
    primary_stop_id = trader._client_id(
        GRID_KEY, event_id, "stop", 3
    )
    residual_stop_id = trader._client_id(
        GRID_KEY, event_id, "stop", 4
    )

    next_order_id = 2000
    entry_prices = (1.553, 1.561, 1.569)
    for index, (client_id, price) in enumerate(
        zip(entry_ids, entry_prices), start=1
    ):
        next_order_id += 1
        exchange.orders[client_id] = {
            "symbol": symbol,
            "side": "SELL",
            "clientOrderId": client_id,
            "orderId": next_order_id,
            "status": "FILLED",
            "origQty": "30.9",
            "executedQty": "30.9",
            "avgPrice": str(price),
            "cumQuote": str(30.9 * price),
            "reduceOnly": False,
            "updateTime": index * 1000,
        }
        exchange.trade_rows.append(
            {
                "symbol": symbol,
                "orderId": next_order_id,
                "side": "SELL",
                "qty": "30.9",
                "quoteQty": str(30.9 * price),
                "commission": "0.02",
                "commissionAsset": "USDT",
                "realizedPnl": "0",
            }
        )
    for client_id in take_profit_ids:
        next_order_id += 1
        exchange.orders[client_id] = {
            "symbol": symbol,
            "side": "BUY",
            "clientOrderId": client_id,
            "orderId": next_order_id,
            "status": "EXPIRED",
            "origQty": "30.9",
            "executedQty": "0",
            "avgPrice": "0",
            "cumQuote": "0",
            "reduceOnly": True,
            "updateTime": 4000,
        }

    stop_specs = (
        (primary_stop_id, 92.69, 1.591, -2.7807, 5000),
        (residual_stop_id, 0.01, 1.592, -0.00031, 6000),
    )
    for stop_client_id, quantity, price, pnl, update_time in stop_specs:
        next_order_id += 1
        actual_order_id = next_order_id
        exchange.algo_orders[stop_client_id] = {
            "symbol": symbol,
            "side": "BUY",
            "clientAlgoId": stop_client_id,
            "algoId": actual_order_id + 1000,
            "algoStatus": "FINISHED",
            "orderType": "STOP_MARKET",
            "quantity": str(quantity),
            "triggerPrice": "1.59",
            "actualOrderId": actual_order_id,
            "reduceOnly": True,
            "updateTime": update_time,
        }
        actual_client_id = f"actual-{stop_client_id}"
        exchange.orders[actual_client_id] = {
            "symbol": symbol,
            "side": "BUY",
            "clientOrderId": actual_client_id,
            "orderId": actual_order_id,
            "status": "FILLED",
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "avgPrice": str(price),
            "cumQuote": str(quantity * price),
            "reduceOnly": True,
            "updateTime": update_time,
        }
        exchange.trade_rows.append(
            {
                "symbol": symbol,
                "orderId": actual_order_id,
                "side": "BUY",
                "qty": str(quantity),
                "quoteQty": str(quantity * price),
                "commission": "0.03",
                "commissionAsset": "USDT",
                "realizedPnl": str(pnl),
            }
        )

    commission = sum(
        float(row["commission"]) for row in exchange.trade_rows
    )
    realized = sum(
        float(row["realizedPnl"]) for row in exchange.trade_rows
    )
    exchange.income_rows = [
        {
            "symbol": symbol,
            "incomeType": "COMMISSION",
            "income": str(-commission),
        },
        {
            "symbol": symbol,
            "incomeType": "REALIZED_PNL",
            "income": str(realized),
        },
        {
            "symbol": symbol,
            "incomeType": "FUNDING_FEE",
            "income": "0.005",
        },
    ]
    reviewed_ids = (
        entry_ids
        + take_profit_ids
        + [primary_stop_id, residual_stop_id]
    )
    trader.state["order_journal"] = [
        {
            "time": _utc_now().isoformat(),
            "action": "reviewed",
            "symbol": symbol,
            "client_id": client_id,
        }
        for client_id in reviewed_ids
    ]
    trader.state["trades"] = [
        {
            "time": _utc_now().isoformat(),
            "strategy": GRID_KEY,
            "symbol": symbol,
            "exit_reason": "exchange_closed_or_protection_triggered",
            "pnl_source": "exchange_trade_history",
        }
    ]
    trader.state["operational_halt"] = True
    trader.state["halt_reason"] = (
        f"{symbol} quantity differs: ledger=92.7 exchange=0.01 | "
        "reconciliation failed 2 times: BinanceApiError: "
        "Order would immediately trigger. | "
        "reconciliation failed 3 times: BinanceApiError: "
        "Unknown order sent."
    )
    trader.state["circuit_breaker"] = True
    trader.state["circuit_reason"] = (
        "API failures=3: BinanceApiError: Unknown order sent."
    )

    with pytest.raises(
        RuntimeError,
        match="deterministic Grid event IDs",
    ):
        trader.recover_verified_flat_protective_stop_incident(
            expected_symbol=symbol,
            expected_event_id="wrong-event",
            expected_entry_client_ids=entry_ids,
            expected_primary_stop_client_id=primary_stop_id,
            expected_residual_stop_client_id=residual_stop_id,
            expected_take_profit_client_ids=take_profit_ids,
        )
    assert trader.state["operational_halt"] is True
    assert trader.state["circuit_breaker"] is True

    result = (
        trader.recover_verified_flat_protective_stop_incident(
            expected_symbol=symbol,
            expected_event_id=event_id,
            expected_entry_client_ids=entry_ids,
            expected_primary_stop_client_id=primary_stop_id,
            expected_residual_stop_client_id=residual_stop_id,
            expected_take_profit_client_ids=take_profit_ids,
        )
    )

    assert result["cleared"] is True
    assert result["already_recovered"] is False
    assert result["record"]["entry_quantity"] == pytest.approx(92.7)
    assert result["record"]["exit_quantity"] == pytest.approx(92.7)
    assert result["record"]["realized_pnl"] == pytest.approx(
        -2.78101
    )
    assert result["record"]["funding_pnl_usdt"] == pytest.approx(
        0.005
    )
    assert trader.state["operational_halt"] is False
    assert trader.state["circuit_breaker"] is False
    assert trader.state["halt_reason"] == ""
    assert trader.state["circuit_reason"] == ""
    assert len(trader.state["recovery_journal"]) == 1
    trade = trader.state["trades"][0]
    assert trade["exit_reason"] == (
        "verified_protective_stop_with_residual_close"
    )
    assert trade["pnl_source"] == (
        "verified_exchange_trades_and_income_history"
    )
    assert exchange.market_calls == []
    repeated = (
        trader.recover_verified_flat_protective_stop_incident(
            expected_symbol=symbol,
            expected_event_id=event_id,
            expected_entry_client_ids=entry_ids,
            expected_primary_stop_client_id=primary_stop_id,
            expected_residual_stop_client_id=residual_stop_id,
            expected_take_profit_client_ids=take_profit_ids,
        )
    )
    assert repeated["already_recovered"] is True
    assert len(trader.state["recovery_journal"]) == 1


def test_missing_protective_stop_is_repaired_on_reconcile(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    stop = next(
        row
        for row in exchange.open_algo_orders()
        if row["orderType"] == "STOP_MARKET"
    )
    exchange.algo_orders[stop["clientAlgoId"]]["algoStatus"] = "CANCELED"
    before = len(
        [row for row in exchange.algo_calls if row["orderType"] == "STOP_MARKET"]
    )

    trader.reconcile()

    after = len(
        [row for row in exchange.algo_calls if row["orderType"] == "STOP_MARKET"]
    )
    assert after == before + 1
    assert any(
        row["orderType"] == "STOP_MARKET"
        for row in exchange.open_algo_orders()
    )


def test_quantity_mismatch_halts_but_protects_full_exchange_quantity(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    exchange.position_amounts["BTCUSDT"] += 0.002
    actual = exchange.position_amounts["BTCUSDT"]

    trader.reconcile()

    stop = next(
        row
        for row in exchange.open_algo_orders("BTCUSDT")
        if row["orderType"] == "STOP_MARKET"
    )
    assert float(stop["quantity"]) == pytest.approx(actual)
    assert trader.state["operational_halt"] is True
    assert "quantity differs" in trader.state["halt_reason"]


def test_partial_entry_is_canceled_emergency_closed_and_halted(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    exchange.entry_fill_ratio = 0.5

    with pytest.raises(RuntimeError, match="not fully filled"):
        trader._open_breakout_candidate(
            _breakout_candidate("BTCUSDT", now), now
        )

    entries = [
        row for row in exchange.market_calls if not row["reduceOnly"]
    ]
    exits = [row for row in exchange.market_calls if row["reduceOnly"]]
    assert len(entries) == 1
    assert exchange.orders[entries[0]["clientOrderId"]]["status"] == "CANCELED"
    assert len(exits) == 1
    assert exchange.position_amounts["BTCUSDT"] == 0.0
    assert trader.state["breakout_position"] is None
    assert trader.state["operational_halt"] is True


def test_restart_recovers_exchange_fill_from_pending_entry_intent(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    config = _armed_config(tmp_path)
    trader = CombinedBreakoutV8GridV6LiveTrader(config, exchange)
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    model = trader.state["breakout_position"]
    entry = trader.state["breakout_exchange"]
    entry_id = entry["entry_client_id"]
    trader.state["pending_orders"][entry_id] = {
        "strategy": BREAKOUT_KEY,
        "event_id": entry["event_id"],
        "symbol": entry["symbol"],
        "direction": entry["direction"],
        "client_id": entry_id,
        "created_at": entry["opened_at"],
        "model": model,
    }
    trader.state["breakout_position"] = None
    trader.state["breakout_exchange"] = None
    trader._persist_state()

    restored = CombinedBreakoutV8GridV6LiveTrader(config, exchange)
    restored.validate_startup()

    assert restored.state["breakout_position"] is not None
    assert restored.state["breakout_exchange"]["symbol"] == "BTCUSDT"
    assert entry_id not in restored.state["pending_orders"]
    assert restored.state["operational_halt"] is False


def test_unresolved_pending_entry_blocks_duplicate_and_halts(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    model = trader.state["breakout_position"]
    entry = trader.state["breakout_exchange"]
    entry_id = entry["entry_client_id"]
    exchange.position_amounts["BTCUSDT"] = 0.0
    exchange.orders.pop(entry_id)
    for row in exchange.algo_orders.values():
        row["algoStatus"] = "CANCELED"
    trader.state["breakout_position"] = None
    trader.state["breakout_exchange"] = None
    trader.state["protective_orders"] = {}
    trader.state["pending_orders"][entry_id] = {
        "strategy": BREAKOUT_KEY,
        "event_id": entry["event_id"],
        "symbol": entry["symbol"],
        "direction": entry["direction"],
        "client_id": entry_id,
        "created_at": (
            _utc_now() - timedelta(minutes=2)
        ).isoformat(),
        "model": model,
    }
    before = len(exchange.market_calls)

    trader.reconcile()
    opened = trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )

    assert opened is False
    assert len(exchange.market_calls) == before
    assert trader.state["operational_halt"] is True
    assert "initial entry intent unresolved" in trader.state["halt_reason"]


def test_grid_fill_reconciles_quantity_and_replaces_protective_stop(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "ETHUSDT")
    assert trader._open_grid_candidate(
        _grid_candidate("ETHUSDT", now), now
    )
    initial_stop_count = len(
        [row for row in exchange.algo_calls if row["orderType"] == "STOP_MARKET"]
    )
    assert all(
        row["reduceOnly"]
        for row in exchange.open_orders("ETHUSDT")
    )
    trigger_id, trigger = next(
        (client_id, row)
        for client_id, row in trader.state["grid_exchange"][
            "orders"
        ].items()
        if row["kind"] == "entry_trigger"
        and row["status"] == "ARMED"
    )
    exchange.candle_high = float(trigger["price"]) + 0.01
    exchange.candle_close = float(trigger["price"]) + 0.01

    trader._trigger_grid_entries(now)

    campaign = trader.state["grid_campaign"]
    assert len(campaign["lots"]) == 2
    stop_count = len(
        [row for row in exchange.algo_calls if row["orderType"] == "STOP_MARKET"]
    )
    assert stop_count == initial_stop_count + 1
    assert any(
        row["reduceOnly"]
        for row in exchange.open_orders("ETHUSDT")
    )
    assert not any(
        not row["reduceOnly"]
        for row in exchange.open_orders("ETHUSDT")
    )
    assert any(
        row["clientOrderId"] == trigger_id
        and not row["reduceOnly"]
        for row in exchange.market_calls
    )
    assert trader.state["operational_halt"] is False


def test_grid_partial_take_profit_updates_lot_and_stop_without_halt(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "ETHUSDT")
    assert trader._open_grid_candidate(
        _grid_candidate("ETHUSDT", now), now
    )
    take_profit = next(
        row
        for row in exchange.open_orders("ETHUSDT")
        if row["reduceOnly"]
    )
    initial_quantity = abs(exchange.position_amounts["ETHUSDT"])
    exchange.fill_limit_partial(
        take_profit["clientOrderId"], 0.5
    )

    trader.reconcile()

    campaign = trader.state["grid_campaign"]
    remaining_lot = float(campaign["lots"]["0"]["quantity"])
    exchange_quantity = abs(exchange.position_amounts["ETHUSDT"])
    stop = next(
        row
        for row in exchange.open_algo_orders("ETHUSDT")
        if row["orderType"] == "STOP_MARKET"
    )
    assert remaining_lot < initial_quantity
    assert remaining_lot == pytest.approx(exchange_quantity)
    assert float(stop["quantity"]) == pytest.approx(exchange_quantity)
    assert trader.state["operational_halt"] is False


def test_circuit_breaker_blocks_grid_scale_in(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "ETHUSDT")
    assert trader._open_grid_candidate(
        _grid_candidate("ETHUSDT", now), now
    )
    trigger = next(
        row
        for row in trader.state["grid_exchange"]["orders"].values()
        if row["kind"] == "entry_trigger"
        and row["status"] == "ARMED"
    )
    exchange.candle_high = float(trigger["price"]) + 0.01
    exchange.candle_close = float(trigger["price"]) + 0.01
    entries_before = len(
        [row for row in exchange.market_calls if not row["reduceOnly"]]
    )
    trader._trip_circuit("test circuit")

    trader._trigger_grid_entries(now)

    entries_after = len(
        [row for row in exchange.market_calls if not row["reduceOnly"]]
    )
    assert entries_after == entries_before
    assert len(trader.state["grid_campaign"]["lots"]) == 1


def test_external_position_halts_and_drawdown_trips_entry_circuit(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    exchange.position_amounts["XRPUSDT"] = 1.0
    exchange.entry_prices["XRPUSDT"] = 100.0

    trader.reconcile()

    assert trader.state["operational_halt"] is True
    assert "unmanaged exchange position" in trader.state["halt_reason"]

    trader.state["peak_equity"] = 200.0
    exchange.equity = 79.0
    trader._update_risk_circuits(trader.snapshot_account())
    assert trader.state["circuit_breaker"] is True
    assert "drawdown" in trader.state["circuit_reason"]


def test_entry_balance_and_margin_guards_trip_circuit(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    account = trader.snapshot_account()

    low_balance = replace(
        account,
        available_balance=(
            trader.config.risk.min_available_balance_usdt - 0.01
        ),
    )
    assert (
        trader._entry_account_allows(
            BREAKOUT_KEY, "BTCUSDT", low_balance
        )
        is False
    )
    assert trader.state["circuit_breaker"] is True
    assert "available_balance" in trader.state["circuit_reason"]

    margin_trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path / "margin"), exchange
    )
    high_margin = replace(
        account,
        initial_margin=(
            account.equity
            * margin_trader.config.risk.max_account_margin_usage_pct
        ),
    )
    assert (
        margin_trader._entry_account_allows(
            BREAKOUT_KEY, "BTCUSDT", high_margin
        )
        is False
    )
    assert margin_trader.state["circuit_breaker"] is True
    assert "initial_margin_usage" in margin_trader.state["circuit_reason"]


def test_repeated_reconciliation_failure_latches_operational_halt(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    exchange.fail_open_orders = True

    for _ in range(2):
        with pytest.raises(
            RuntimeError, match="simulated open-orders outage"
        ):
            trader.reconcile()

    assert trader.state["reconcile_failures"] == 2
    assert trader.state["operational_halt"] is True
    assert "reconciliation failed 2 times" in trader.state["halt_reason"]


def test_reconciliation_uses_one_consolidated_exchange_snapshot(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    exchange.account_calls = 0
    exchange.open_orders_calls = 0
    exchange.open_algo_orders_calls = 0

    trader.reconcile()

    assert exchange.account_calls == 1
    assert exchange.open_orders_calls == 1
    assert exchange.open_algo_orders_calls == 1


def test_rate_limit_cooldown_does_not_latch_reconciliation_halt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()

    def limited(_symbol: str | None = None):
        raise BinanceRateLimitError(
            429,
            "Too many requests",
            retry_after_seconds=65,
            proactive=False,
        )

    monkeypatch.setattr(exchange, "open_orders", limited)
    with pytest.raises(BinanceRateLimitError):
        trader.reconcile()

    assert trader.state["reconcile_failures"] == 0
    assert trader.state["operational_halt"] is False
    assert trader.state["rate_limit_cooldown_until"] is not None


def test_untracked_owned_order_is_canceled_and_latches_halt(
    tmp_path: Path,
) -> None:
    exchange = FakeExchange()
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    unknown_id = f"{trader.live.order_id_prefix}-unknown-open"
    exchange.new_limit_order(
        "BTCUSDT",
        "BUY",
        "0.1",
        "99",
        reduce_only=False,
        new_client_order_id=unknown_id,
    )

    trader.reconcile()

    assert exchange.orders[unknown_id]["status"] == "CANCELED"
    assert trader.state["operational_halt"] is True
    assert "untracked strategy orders canceled" in trader.state["halt_reason"]


def test_all_live_exit_orders_are_reduce_only(tmp_path: Path) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )

    trader._close_live_position(
        BREAKOUT_KEY,
        "BTCUSDT",
        Direction.LONG,
        "test_exit",
    )

    exits = [row for row in exchange.market_calls if row["reduceOnly"]]
    assert len(exits) == 1
    assert exchange.position_amounts["BTCUSDT"] == 0.0


def test_partial_exit_keeps_ledger_and_protection_and_halts(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now)
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )
    trader.validate_startup()
    trader.state["started_at"] = (
        now - timedelta(minutes=1)
    ).isoformat()
    _install_context(trader, now, "BTCUSDT")
    assert trader._open_breakout_candidate(
        _breakout_candidate("BTCUSDT", now), now
    )
    exchange.reduce_only_fill_ratio = 0.5

    trader._close_live_position(
        BREAKOUT_KEY,
        "BTCUSDT",
        Direction.LONG,
        "partial_test_exit",
    )

    assert exchange.position_amounts["BTCUSDT"] > 0.0
    assert trader.state["breakout_position"] is not None
    assert trader.state["operational_halt"] is True
    assert "residual" in trader.state["halt_reason"]
    assert any(
        row["orderType"] == "STOP_MARKET"
        for row in exchange.open_algo_orders("BTCUSDT")
    )


def test_stale_market_data_trips_entry_circuit(
    tmp_path: Path,
) -> None:
    now = _utc_now().replace(second=0, microsecond=0)
    exchange = FakeExchange(now - timedelta(minutes=10))
    trader = CombinedBreakoutV8GridV6LiveTrader(
        _armed_config(tmp_path), exchange
    )

    assert trader._market_data_is_fresh(now) is False
    assert trader.state["circuit_breaker"] is True
    assert "market_data_stale" in trader.state["circuit_reason"]
