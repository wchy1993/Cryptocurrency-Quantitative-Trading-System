from __future__ import annotations

import hashlib
import hmac
import json
import os
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Candle


MAINNET_BASE_URL = "https://fapi.binance.com"
TESTNET_BASE_URL = "https://testnet.binancefuture.com"


class BinanceApiError(RuntimeError):
    def __init__(self, status: int | None, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    quantity_step: Decimal
    min_quantity: Decimal
    price_tick: Decimal
    min_notional: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity_step", Decimal(str(self.quantity_step)))
        object.__setattr__(self, "min_quantity", Decimal(str(self.min_quantity)))
        object.__setattr__(self, "price_tick", Decimal(str(self.price_tick)))
        object.__setattr__(self, "min_notional", Decimal(str(self.min_notional)))

    def round_quantity(self, quantity: float) -> str:
        value = Decimal(str(quantity))
        if value <= 0:
            return "0"
        rounded = (value / self.quantity_step).to_integral_value(rounding=ROUND_DOWN) * self.quantity_step
        return _decimal_to_string(rounded)

    def round_price(self, price: float) -> str:
        value = Decimal(str(price))
        rounded = (value / self.price_tick).to_integral_value(rounding=ROUND_DOWN) * self.price_tick
        return _decimal_to_string(rounded)


class BinanceFuturesClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        environment: str = "testnet",
        recv_window: int = 5_000,
        timeout_seconds: int = 10,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.timeout_seconds = timeout_seconds
        normalized = environment.strip().lower()
        if normalized not in {"testnet", "mainnet"}:
            raise ValueError("environment must be testnet or mainnet")
        self.environment = normalized
        self.base_url = TESTNET_BASE_URL if normalized == "testnet" else MAINNET_BASE_URL
        self._rules: dict[str, SymbolRules] = {}
        self._rules_loaded = False
        ca_file = "/etc/ssl/cert.pem"
        self._ssl_context = ssl.create_default_context(cafile=ca_file if os.path.exists(ca_file) else None)

    def ping(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/ping")

    def server_time(self) -> int:
        payload = self._request("GET", "/fapi/v1/time")
        return int(payload["serverTime"])

    def exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def symbol_rules(self, symbol: str) -> SymbolRules:
        symbol = symbol.upper()
        cached = self._rules.get(symbol)
        if cached:
            return cached
        if not self._rules_loaded:
            self._load_symbol_rules()
        cached = self._rules.get(symbol)
        if cached:
            return cached
        raise BinanceApiError(None, f"symbol not found in exchangeInfo: {symbol}")

    def _load_symbol_rules(self) -> None:
        info = self.exchange_info()
        for item in info.get("symbols", []):
            item_symbol = str(item.get("symbol", "")).upper()
            if not item_symbol:
                continue
            filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
            lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
            price = filters.get("PRICE_FILTER") or {}
            notional = filters.get("MIN_NOTIONAL") or {}
            rules = SymbolRules(
                symbol=item_symbol,
                quantity_step=Decimal(str(lot.get("stepSize", "0.001"))),
                min_quantity=Decimal(str(lot.get("minQty", "0"))),
                price_tick=Decimal(str(price.get("tickSize", "0.01"))),
                min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "5")))),
            )
            self._rules[item_symbol] = rules
        self._rules_loaded = True

    def depth(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        if limit not in {5, 10, 20, 50, 100, 500, 1000}:
            raise ValueError("depth limit must be one of 5, 10, 20, 50, 100, 500, 1000")
        payload = self._request("GET", "/fapi/v1/depth", {"symbol": symbol.upper(), "limit": limit})
        if not isinstance(payload, dict):
            raise BinanceApiError(None, f"invalid depth response for {symbol}")
        return payload

    def premium_index(self, symbol: str) -> dict[str, Any]:
        payload = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        if not isinstance(payload, dict):
            raise BinanceApiError(None, f"invalid premium index response for {symbol}")
        return payload

    def premium_indexes(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/fapi/v1/premiumIndex")
        if not isinstance(payload, list):
            raise BinanceApiError(None, "invalid premium index response")
        return [row for row in payload if isinstance(row, dict)]

    def klines(self, symbol: str, interval: str, limit: int = 200) -> list[Candle]:
        rows = self._request("GET", "/fapi/v1/klines", {"symbol": symbol.upper(), "interval": interval, "limit": limit})
        candles = []
        for row in rows:
            timestamp = datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
            candles.append(
                Candle(
                    timestamp=timestamp,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        for candle in candles:
            candle.validate()
        return candles

    def account(self) -> dict[str, Any]:
        return self._signed_request("GET", "/fapi/v2/account")

    def position_mode(self) -> bool:
        payload = self._signed_request("GET", "/fapi/v1/positionSide/dual")
        value = payload.get("dualSidePosition")
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"

    def position_risk(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._signed_request("GET", "/fapi/v2/positionRisk", params)
        return payload if isinstance(payload, list) else [payload]

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._signed_request("GET", "/fapi/v1/openOrders", params)
        return payload if isinstance(payload, list) else [payload]

    def open_algo_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._signed_request("GET", "/fapi/v1/openAlgoOrders", params)
        return payload if isinstance(payload, list) else [payload]

    def user_trades(
        self,
        symbol: str | None = None,
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol.upper()
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        payload = self._signed_request("GET", "/fapi/v1/userTrades", params)
        return payload if isinstance(payload, list) else [payload]

    def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol.upper(), "leverage": leverage})

    def set_margin_type(self, symbol: str, margin_type: str) -> dict[str, Any]:
        try:
            return self._signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol.upper(), "marginType": margin_type.upper()})
        except BinanceApiError as exc:
            if isinstance(exc.payload, dict) and int(exc.payload.get("code", 0)) == -4046:
                return {"code": -4046, "msg": "No need to change margin type."}
            raise

    def cancel_all_open_orders(self, symbol: str) -> dict[str, Any]:
        return self._signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()})

    def cancel_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not client_order_id:
            raise ValueError("order_id or client_order_id is required")
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._signed_request("DELETE", "/fapi/v1/order", params)

    def query_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not client_order_id:
            raise ValueError("order_id or client_order_id is required")
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._signed_request("GET", "/fapi/v1/order", params)

    def cancel_all_algo_open_orders(self, symbol: str) -> dict[str, Any]:
        return self._signed_request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol.upper()})

    def new_market_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        reduce_only: bool = False,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
        return self._signed_request("POST", "/fapi/v1/order", params)

    def new_post_only_order(
        self,
        symbol: str,
        side: str,
        price: str,
        quantity: str,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTX",
            "price": price,
            "quantity": quantity,
            "newOrderRespType": "ACK",
        }
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
        return self._signed_request("POST", "/fapi/v1/order", params)

    def new_stop_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: str,
        quantity: str,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "STOP_MARKET",
            "triggerPrice": stop_price,
            "quantity": quantity,
            "workingType": working_type,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

    def new_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: str,
        quantity: str,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": stop_price,
            "quantity": quantity,
            "workingType": working_type,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

    def _signed_request(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key or not self.api_secret:
            raise BinanceApiError(None, "missing Binance API credentials")
        signed = dict(params or {})
        signed["timestamp"] = int(time.time() * 1000)
        signed["recvWindow"] = self.recv_window
        query = urlencode(signed, doseq=True)
        signature = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        signed["signature"] = signature
        return self._request(method, path, signed, signed=True)

    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = params or {}
        query = urlencode(params, doseq=True)
        url = f"{self.base_url}{path}"
        data = None
        if method.upper() in {"GET", "DELETE"} and query:
            url = f"{url}?{query}"
        elif query:
            data = query.encode("utf-8")

        headers = {"User-Agent": "crypto-scalper/0.1"}
        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        if signed and self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        request = Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise BinanceApiError(exc.code, _error_message(body), _json_or_text(body)) from exc
        except URLError as exc:
            raise BinanceApiError(None, f"network error: {exc.reason}") from exc

        if not body:
            return {}
        return json.loads(body)


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _error_message(value: str) -> str:
    parsed = _json_or_text(value)
    if isinstance(parsed, dict):
        return str(parsed.get("msg", parsed))
    return str(parsed)


def _decimal_to_string(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")
