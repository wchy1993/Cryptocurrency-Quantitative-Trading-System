from __future__ import annotations

import json
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .binance_client import BinanceApiError, BinanceFuturesClient, SymbolRules
from .live_config import LiveAppConfig
from .live_trader import AccountSnapshot, LivePosition
from .models import Direction
from .order_flow_absorption_scalper import (
    AggTradeEvent,
    BookSnapshot,
    MarketContext,
    OfasAction,
    OrderFlowAbsorptionScalper,
)


LogCallback = Callable[[str], None]
AccountCallback = Callable[[AccountSnapshot], None]
MAINNET_PUBLIC_WS_URL = "wss://fstream.binance.com/public/ws"
MAINNET_MARKET_WS_URL = "wss://fstream.binance.com/market/ws"
TESTNET_WS_URL = "wss://stream.binancefuture.com/ws"
SYSTEM_CA_FILE = "/etc/ssl/cert.pem"
STREAMS_PER_CONNECTION = 20


class OfasPaperTrader:
    """Realtime OFAS runner with conservative local paper fills.

    Market data is real Binance Futures data. Orders and positions remain local;
    this runner deliberately refuses to start when dry_run is disabled.
    """

    def __init__(
        self,
        config: LiveAppConfig,
        client: BinanceFuturesClient,
        logger: LogCallback | None = None,
        account_callback: AccountCallback | None = None,
    ) -> None:
        if not config.ofas_strategy.enabled:
            raise ValueError("OFAS strategy is not enabled")
        if not config.trading.dry_run:
            raise RuntimeError("OFAS GUI runner currently supports dry-run only")
        self.config = config
        self.client = client
        self.log = logger or print
        self.account_callback = account_callback
        self.symbols = tuple(config.ofas_strategy.symbols)
        if not self.symbols:
            raise ValueError("OFAS symbol list is empty")
        self._lock = threading.RLock()
        self._rules: dict[str, SymbolRules] = {}
        self._engine: OrderFlowAbsorptionScalper | None = None
        self._book_ready = {symbol: False for symbol in self.symbols}
        self._last_depth_update_id: dict[str, int] = {}
        self._latest_price: dict[str, float] = {}
        self._funding: dict[str, float] = {}
        self._realized_pnl = 0.0
        self._entry_fees: dict[str, float] = {}
        self._connected = threading.Event()
        self._ws_threads: list[threading.Thread] = []
        self._websockets: dict[str, Any] = {}
        self._connected_routes: set[str] = set()
        self._expected_routes: set[str] = set()
        self._route_symbols: dict[str, tuple[str, ...]] = {}
        self._route_message_counts: dict[str, int] = {}
        self._subscription_ids: dict[int, str] = {}
        self._next_subscription_id = 1
        self._ready_count_logged = 0
        self._resync_requested: set[str] = set()
        self._stop_event: threading.Event | None = None
        self._last_account_callback = 0.0
        self._last_status_log = 0.0
        self._last_funding_refresh = 0.0
        self._last_message_time = 0.0

    def run_forever(self, stop_event: threading.Event) -> None:
        self._stop_event = stop_event
        self._load_rules()
        self._engine = OrderFlowAbsorptionScalper(self.config.ofas_strategy, self._rules)
        self.log(f"OFAS dry-run 启动：实时行情、虚拟下单，扫描 {len(self.symbols)} 个币")
        self._start_websocket()
        if not self._connected.wait(20):
            self._stop_websocket()
            raise RuntimeError("OFAS WebSocket connection timed out")

        self._wait_for_books(stop_event)
        while not stop_event.is_set():
            started = time.monotonic()
            self._evaluate_all()
            self._refresh_funding_if_due()
            self._publish_account_if_due()
            if self._last_message_time and time.time() - self._last_message_time > 10:
                self.log("OFAS WebSocket 超过10秒无行情，禁止新开仓并等待自动重连")
                self._last_message_time = time.time()
            elapsed = time.monotonic() - started
            stop_event.wait(max(0.05, self.config.ofas_strategy.evaluation_interval_ms / 1000.0 - elapsed))

        self._stop_websocket()
        self._publish_account(force=True)
        self.log("OFAS dry-run 已停止")

    def snapshot_account(self) -> AccountSnapshot:
        with self._lock:
            engine = self._require_engine()
            positions: dict[str, LivePosition] = {}
            rows: list[LivePosition] = []
            unrealized_total = 0.0
            initial_margin = 0.0
            maintenance_margin = 0.0
            for symbol, runtime in engine.runtime.items():
                position = runtime.position
                if not position:
                    continue
                mark = self._latest_price.get(symbol, position.entry_price)
                unrealized = position.side.value * position.quantity * (mark - position.entry_price)
                notional = abs(position.quantity * mark)
                unrealized_total += unrealized
                initial_margin += notional / max(self.config.ofas_strategy.leverage, 1.0)
                maintenance_margin += notional * 0.005
                live = LivePosition(
                    symbol=symbol,
                    position_side="SIM",
                    direction=position.side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    mark_price=mark,
                    notional=notional,
                    unrealized_pnl=unrealized,
                    leverage=max(1, int(self.config.ofas_strategy.leverage)),
                    margin_type="SIM",
                    liquidation_price=None,
                    entry_reason=f"OFAS {position.quality_bucket} score={position.quality_score:.1f}",
                    opened_at=position.entry_time.replace(tzinfo=timezone.utc),
                )
                positions[symbol] = live
                rows.append(live)
            wallet = self.config.risk.starting_capital_usdt + self._realized_pnl
            equity = wallet + unrealized_total
            return AccountSnapshot(
                equity=equity,
                wallet_balance=wallet,
                available_balance=max(0.0, equity - initial_margin),
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                total_unrealized_pnl=unrealized_total,
                positions=positions,
                position_rows=tuple(rows),
                position_mode="OFAS PAPER",
            )

    def _load_rules(self) -> None:
        unsupported: list[str] = []
        for symbol in self.symbols:
            try:
                self._rules[symbol] = self.client.symbol_rules(symbol)
            except BinanceApiError:
                unsupported.append(symbol)
        if unsupported:
            self.log(f"OFAS 跳过交易所不支持币种: {','.join(unsupported)}")
            self.symbols = tuple(symbol for symbol in self.symbols if symbol in self._rules)
            self._book_ready = {symbol: False for symbol in self.symbols}
        if not self.symbols:
            raise RuntimeError("OFAS 没有可用交易币种")

    def _start_websocket(self) -> None:
        try:
            import websocket  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("缺少 websocket-client，请先执行 pip install -e .") from exc
        public_url = MAINNET_PUBLIC_WS_URL if self.config.exchange.environment == "mainnet" else TESTNET_WS_URL
        market_url = MAINNET_MARKET_WS_URL if self.config.exchange.environment == "mainnet" else TESTNET_WS_URL
        routes: list[tuple[str, str, list[str], tuple[str, ...]]] = []
        for offset in range(0, len(self.symbols), STREAMS_PER_CONNECTION):
            symbols = self.symbols[offset:offset + STREAMS_PER_CONNECTION]
            shard = offset // STREAMS_PER_CONNECTION + 1
            routes.extend((
                (
                    f"public-{shard}",
                    public_url,
                    [f"{symbol.lower()}@depth10@100ms" for symbol in symbols],
                    symbols,
                ),
                (
                    f"market-{shard}",
                    market_url,
                    [f"{symbol.lower()}@aggTrade" for symbol in symbols],
                    symbols,
                ),
            ))
        self._expected_routes = {route for route, _, _, _ in routes}
        self._route_symbols = {route: symbols for route, _, _, symbols in routes}
        self._route_message_counts = {route: 0 for route in self._expected_routes}
        for route, url, streams, _symbols in routes:
            thread = threading.Thread(
                target=self._websocket_loop,
                args=(route, url, streams),
                daemon=True,
                name=f"ofas-{route}-data",
            )
            self._ws_threads.append(thread)
            thread.start()
            time.sleep(0.25)

    def _websocket_loop(self, route: str, url: str, streams: list[str]) -> None:
        import websocket

        backoff = 1.0
        while self._stop_event and not self._stop_event.is_set():
            ws = websocket.WebSocketApp(
                url,
                on_open=lambda opened, r=route, s=streams: self._on_ws_open(opened, r, s),
                on_message=lambda opened, raw, r=route: self._on_ws_message(opened, raw, r),
                on_error=lambda opened, error, r=route: self._on_ws_error(opened, error, r),
                on_close=lambda closed, code, message, r=route: self._on_ws_close(closed, code, message, r),
            )
            self._websockets[route] = ws
            ssl_options: dict[str, Any] = {"cert_reqs": ssl.CERT_REQUIRED}
            if Path(SYSTEM_CA_FILE).exists():
                ssl_options["ca_certs"] = SYSTEM_CA_FILE
            # Binance sends protocol-level pings and websocket-client replies
            # automatically. An additional client ping timer can falsely time
            # out while a high-volume depth callback is being processed.
            ws.run_forever(ping_interval=0, sslopt=ssl_options)
            if self._stop_event.is_set():
                break
            self._stop_event.wait(backoff)
            backoff = min(30.0, backoff * 2.0)
        with self._lock:
            self._connected_routes.discard(route)
            self._connected.clear()

    def _stop_websocket(self) -> None:
        for ws in list(self._websockets.values()):
            try:
                ws.close()
            except Exception:
                pass
        for thread in self._ws_threads:
            if thread.is_alive():
                thread.join(timeout=5)

    def _on_ws_open(self, ws: Any, route: str, streams: list[str]) -> None:
        with self._lock:
            if route.startswith("public-"):
                for symbol in self._route_symbols.get(route, ()):
                    self._last_depth_update_id.pop(symbol, None)
                    self._book_ready[symbol] = False
                    if self._engine:
                        self._process_actions(self._engine.on_order_book_invalid(symbol, "websocket_reconnected"))
            subscription_id = self._next_subscription_id
            self._next_subscription_id += 1
            self._subscription_ids[subscription_id] = route
        ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": subscription_id}))
        with self._lock:
            self._connected_routes.add(route)
            if self._connected_routes == self._expected_routes:
                self._connected.set()
        self.log(f"OFAS {route} WebSocket 已连接，订阅 {len(streams)} 个实时流")

    def _on_ws_message(self, _ws: Any, raw: str, route: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self.log(f"OFAS {route} 收到无法解析的 WebSocket 消息")
            return
        if isinstance(message, dict) and "id" in message:
            subscription_id = message.get("id")
            subscribed_route = self._subscription_ids.get(subscription_id, route)
            if message.get("result") is None and "code" not in message:
                self.log(f"OFAS {subscribed_route} 订阅已确认")
            else:
                self.log(
                    f"OFAS {subscribed_route} 订阅被拒绝 "
                    f"code={message.get('code')} msg={message.get('msg', message)}"
                )
            return
        data = message.get("data", message) if isinstance(message, dict) else {}
        if not isinstance(data, dict):
            return
        event_type = data.get("e")
        symbol = str(data.get("s", "")).upper()
        if symbol not in self._book_ready:
            return
        timestamp = _event_time(data)
        self._last_message_time = time.time()
        with self._lock:
            self._route_message_counts[route] = self._route_message_counts.get(route, 0) + 1
            engine = self._engine
            if engine is None:
                return
            if event_type == "aggTrade":
                event = AggTradeEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=float(data["p"]),
                    quantity=float(data["q"]),
                    buyer_is_maker=bool(data["m"]),
                    aggregate_trade_id=int(data["a"]),
                )
                self._latest_price[symbol] = event.price
                engine.on_agg_trade(event)
                self._maybe_fill_maker(symbol, event)
            elif event_type == "depthUpdate":
                update_id = int(data["u"])
                previous_id = int(data["pu"]) if data.get("pu") is not None else None
                last_id = self._last_depth_update_id.get(symbol)
                if last_id is not None and previous_id != last_id:
                    self.log(f"{symbol}: OFAS L10快照序号跳跃 last={last_id} pu={previous_id}，取消信号并用当前完整L10恢复")
                    self._process_actions(engine.on_order_book_invalid(symbol, "partial_depth_gap"))
                engine.on_book_snapshot(
                    BookSnapshot(
                        symbol=symbol,
                        timestamp=timestamp,
                        last_update_id=update_id,
                        bids=tuple((float(price), float(qty)) for price, qty in data.get("b", ())),
                        asks=tuple((float(price), float(qty)) for price, qty in data.get("a", ())),
                    )
                )
                self._last_depth_update_id[symbol] = update_id
                self._book_ready[symbol] = engine.books[symbol].valid
                ready = sum(self._book_ready.values())
                if ready > self._ready_count_logged and (ready == len(self.symbols) or ready % 10 == 0):
                    self._ready_count_logged = ready
                    self.log(f"OFAS 本地L10订单簿初始化进度 {ready}/{len(self.symbols)}")
                book = engine.books[symbol]
                if book.valid:
                    bid, _ = book.best_bid
                    ask, _ = book.best_ask
                    self._latest_price[symbol] = (bid + ask) / 2.0

    def _on_ws_error(self, _ws: Any, error: Any, route: str) -> None:
        self.log(f"OFAS {route} WebSocket 错误: {type(error).__name__}: {error}")

    def _on_ws_close(self, _ws: Any, code: Any, message: Any, route: str) -> None:
        with self._lock:
            self._connected_routes.discard(route)
            self._connected.clear()
            if route.startswith("public-"):
                for symbol in self._route_symbols.get(route, ()):
                    self._last_depth_update_id.pop(symbol, None)
                    self._book_ready[symbol] = False
                    if self._engine:
                        self._process_actions(self._engine.on_order_book_invalid(symbol, "websocket_disconnected"))
        if self._stop_event and not self._stop_event.is_set():
            self.log(
                f"OFAS {route} WebSocket 已断开 code={code} message={message or ''}，本轮禁止新开仓"
            )

    def _wait_for_books(self, stop_event: threading.Event) -> None:
        deadline = time.time() + 20.0
        while not stop_event.is_set() and time.time() < deadline:
            ready = sum(self._book_ready.values())
            if ready == len(self.symbols):
                break
            stop_event.wait(0.10)
        ready = sum(self._book_ready.values())
        self.log(f"OFAS 本地L10订单簿初始化完成 {ready}/{len(self.symbols)}")
        if ready < len(self.symbols):
            silent = [route for route, count in self._route_message_counts.items() if count == 0]
            missing = [symbol for symbol, is_ready in self._book_ready.items() if not is_ready]
            self.log(
                f"OFAS 行情初始化不完整：无数据连接={','.join(sorted(silent)) or '无'} "
                f"缺少盘口={','.join(missing)}"
            )

    def _evaluate_all(self) -> None:
        with self._lock:
            snapshot = self.snapshot_account()
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            btc_2s = btc_5s = 0.0
            engine = self._require_engine()
            if "BTCUSDT" in engine.books and engine.books["BTCUSDT"].valid:
                btc_features = engine.flows["BTCUSDT"].features(now, engine.books["BTCUSDT"])
                if btc_features:
                    btc_2s = btc_features.return_2s * 10_000.0
                    btc_5s = btc_features.return_5s * 10_000.0
        for symbol in self.symbols:
            with self._lock:
                if not self._book_ready.get(symbol):
                    continue
                self._evaluate_symbol(symbol, snapshot, now, btc_2s, btc_5s)

    def _evaluate_symbol(
        self,
        symbol: str,
        snapshot: AccountSnapshot,
        now: datetime,
        btc_2s: float,
        btc_5s: float,
    ) -> None:
        engine = self._require_engine()
        actions = engine.evaluate(
            symbol,
            MarketContext(
                timestamp=now,
                equity=snapshot.equity,
                available_balance=snapshot.available_balance,
                open_positions=len(snapshot.positions),
                btc_return_2s_bps=btc_2s,
                btc_return_5s_bps=btc_5s,
                btc_data_age_ms=0 if self._book_ready.get("BTCUSDT") else 99_999,
                funding_rate=self._funding.get(symbol, 0.0),
            ),
        )
        self._process_actions(actions)

    def _process_actions(self, actions: list[OfasAction]) -> None:
        for action in actions:
            if action.action == "PENDING_CREATED":
                self.log(f"{action.symbol}: OFAS {action.side.name} shock pending event={action.event_id}")
            elif action.action == "PLACE_ENTRY":
                score = float(action.metadata.get("quality_score", 0.0))
                bucket = action.metadata.get("quality_bucket", "")
                self.log(
                    f"{action.symbol}: OFAS {action.side.name} ready {bucket}/{score:.1f} "
                    f"{action.order_type} price={action.price:.8g} qty={action.quantity:.8g}"
                )
                if action.order_type == "MARKET":
                    fill = self._market_entry_price(action)
                    self._fill_entry(action.symbol, fill, action.quantity)
            elif action.action in {"PENDING_CANCELLED", "CANCEL_ENTRY"}:
                failed = action.metadata.get("failed_checks", ())
                detail = f" failed={','.join(str(reason) for reason in failed)}" if failed else ""
                self.log(f"{action.symbol}: OFAS 信号取消 reason={action.reason}{detail}")
            elif action.action == "RESYNC_BOOK":
                self._book_ready[action.symbol] = False
                self._resync_requested.add(action.symbol)
                self.log(f"{action.symbol}: OFAS 盘口序号断裂，等待 snapshot 重建")
            elif action.action == "EXIT_MARKET":
                self._close_position(action)
            elif action.action == "PLACE_PROTECTIVE_STOP":
                self.log(f"{action.symbol}: OFAS dry-run 保护止损={action.price:.8g}")
            elif action.action == "PLACE_TAKE_PROFIT":
                self.log(f"{action.symbol}: OFAS dry-run 止盈={action.price:.8g}")

    def _maybe_fill_maker(self, symbol: str, trade: AggTradeEvent) -> None:
        engine = self._require_engine()
        order = engine.runtime[symbol].working_order
        if not order or order.order_type != "LIMIT_MAKER":
            return
        tick = float(self._rules[symbol].price_tick)
        penetrated = (
            order.side == Direction.LONG
            and trade.taker_direction == Direction.SHORT
            and trade.price <= order.price - tick
        ) or (
            order.side == Direction.SHORT
            and trade.taker_direction == Direction.LONG
            and trade.price >= order.price + tick
        )
        if not penetrated:
            return
        fill_quantity = min(order.quantity, trade.quantity)
        if fill_quantity <= 0:
            return
        self._fill_entry(symbol, order.price, fill_quantity)

    def _fill_entry(self, symbol: str, price: float, quantity: float) -> None:
        engine = self._require_engine()
        actions = engine.on_entry_fill(symbol, datetime.now(timezone.utc).replace(tzinfo=None), price, quantity)
        position = engine.runtime[symbol].position
        if position:
            fee_rate = self.config.ofas_strategy.maker_fee_bps / 10_000.0 if position.entry_order_type == "LIMIT_MAKER" else self.config.ofas_strategy.taker_fee_bps / 10_000.0
            self._entry_fees[symbol] = price * quantity * fee_rate
            self.log(f"{symbol}: OFAS dry-run 成交 {position.side.name} entry={price:.8g} qty={quantity:.8g}")
        self._process_actions(actions)
        self._publish_account(force=True)

    def _close_position(self, action: OfasAction) -> None:
        engine = self._require_engine()
        position = engine.runtime[action.symbol].position
        if not position:
            return
        mark = self._latest_price.get(action.symbol, position.entry_price)
        slippage = self.config.ofas_strategy.estimated_slippage_bps / 10_000.0
        exit_price = mark * (1.0 - slippage if position.side == Direction.LONG else 1.0 + slippage)
        gross = position.side.value * position.quantity * (exit_price - position.entry_price)
        exit_fee = exit_price * position.quantity * self.config.ofas_strategy.taker_fee_bps / 10_000.0
        fee = self._entry_fees.pop(action.symbol, 0.0) + exit_fee
        net = gross - fee
        self._realized_pnl += net
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        engine.on_position_closed(
            action.symbol,
            now,
            net,
            exit_price=exit_price,
            gross_pnl=gross,
            fee=fee,
            slippage=abs(exit_price - mark) * position.quantity,
            exit_reason=action.reason,
        )
        self.log(
            f"{action.symbol}: OFAS dry-run 平仓 exit={exit_price:.8g} "
            f"gross={gross:+.4f}U fee={fee:.4f}U net={net:+.4f}U reason={action.reason}"
        )
        self._publish_account(force=True)

    def _market_entry_price(self, action: OfasAction) -> float:
        engine = self._require_engine()
        book = engine.books[action.symbol]
        raw = book.best_ask[0] if action.side == Direction.LONG else book.best_bid[0]
        slippage = self.config.ofas_strategy.estimated_slippage_bps / 10_000.0
        return raw * (1.0 + slippage if action.side == Direction.LONG else 1.0 - slippage)

    def _refresh_funding_if_due(self) -> None:
        now = time.time()
        if now - self._last_funding_refresh < 300:
            return
        self._last_funding_refresh = now
        try:
            rows = self.client.premium_indexes()
        except Exception as exc:
            self.log(f"OFAS funding 刷新失败 ({type(exc).__name__}: {exc})")
            return
        allowed = set(self.symbols)
        self._funding.update({
            str(row.get("symbol", "")).upper(): float(row.get("lastFundingRate", 0.0))
            for row in rows
            if str(row.get("symbol", "")).upper() in allowed
        })

    def _publish_account_if_due(self) -> None:
        if time.time() - self._last_account_callback >= 1.0:
            self._publish_account()

    def _publish_account(self, force: bool = False) -> None:
        if not self.account_callback:
            return
        now = time.time()
        if not force and now - self._last_account_callback < 1.0:
            return
        self._last_account_callback = now
        self.account_callback(self.snapshot_account())
        if now - self._last_status_log >= 60:
            self._last_status_log = now
            summary = self._require_engine().summary()
            top_rejects = sorted(
                summary["reject_reasons"].items(), key=lambda item: item[1], reverse=True
            )[:5]
            reject_text = ",".join(f"{reason}:{count}" for reason, count in top_rejects) or "无"
            self.log(
                f"OFAS 统计 pending={summary['pending_count']} confirmed={summary['confirmed_count']} "
                f"entries={summary['entry_count']} rejects={summary['reject_count']} "
                f"realized={self._realized_pnl:+.4f}U top_rejects={reject_text}"
            )

    def _require_engine(self) -> OrderFlowAbsorptionScalper:
        if self._engine is None:
            raise RuntimeError("OFAS engine is not initialized")
        return self._engine


def _event_time(data: dict[str, Any]) -> datetime:
    timestamp_ms = int(data.get("T") or data.get("E") or int(time.time() * 1000))
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)
