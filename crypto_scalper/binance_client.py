from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .binance_rate_limit import (
    RequestWeightBudget,
    RequestWeightCooldown,
)
from .models import Candle


MAINNET_BASE_URL = "https://fapi.binance.com"
TESTNET_BASE_URL = "https://testnet.binancefuture.com"


class BinanceApiError(RuntimeError):
    def __init__(
        self,
        status: int | None,
        message: str,
        payload: Any = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload
        self.headers = headers or {}


class BinanceRateLimitError(BinanceApiError):
    def __init__(
        self,
        status: int | None,
        message: str,
        payload: Any = None,
        *,
        retry_after_seconds: float,
        proactive: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            status,
            message,
            payload,
            headers=headers,
        )
        self.retry_after_seconds = max(
            0.0, float(retry_after_seconds)
        )
        self.proactive = bool(proactive)


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
        request_weight_budget: RequestWeightBudget | None = None,
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
        self.request_weight_budget = (
            request_weight_budget or RequestWeightBudget()
        )
        self._market_stream_cache: Any | None = None

    def attach_market_stream_cache(self, cache: Any | None) -> None:
        self._market_stream_cache = cache

    def rate_limit_status(self) -> dict[str, Any]:
        return self.request_weight_budget.status().as_dict()

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
        info = self.exchange_info()
        for item in info.get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            filters = {entry["filterType"]: entry for entry in item.get("filters", [])}
            lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
            price = filters.get("PRICE_FILTER") or {}
            notional = filters.get("MIN_NOTIONAL") or {}
            rules = SymbolRules(
                symbol=symbol,
                quantity_step=Decimal(str(lot.get("stepSize", "0.001"))),
                min_quantity=Decimal(str(lot.get("minQty", "0"))),
                price_tick=Decimal(str(price.get("tickSize", "0.01"))),
                min_notional=Decimal(str(notional.get("notional", notional.get("minNotional", "5")))),
            )
            self._rules[symbol] = rules
            return rules
        raise BinanceApiError(None, f"symbol not found in exchangeInfo: {symbol}")

    def klines(self, symbol: str, interval: str, limit: int = 200) -> list[Candle]:
        normalized_symbol = symbol.upper()
        if self._market_stream_cache is not None:
            cached = self._market_stream_cache.candles(
                normalized_symbol, interval, limit
            )
            if cached is not None:
                return cached
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
        if self._market_stream_cache is not None:
            self._market_stream_cache.seed_candles(
                normalized_symbol, interval, candles
            )
        return candles

    def premium_index(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol and self._market_stream_cache is not None:
            cached = self._market_stream_cache.premium_index(
                symbol.upper()
            )
            if cached is not None:
                return [cached]
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._request("GET", "/fapi/v1/premiumIndex", params)
        return payload if isinstance(payload, list) else [payload]

    def funding_rate_history(
        self,
        symbol: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": max(1, min(1000, int(limit))),
        }
        if start_time is not None:
            params["startTime"] = max(0, int(start_time))
        if end_time is not None:
            params["endTime"] = max(0, int(end_time))
        payload = self._request("GET", "/fapi/v1/fundingRate", params)
        return payload if isinstance(payload, list) else []

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

    def leverage_brackets(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._signed_request("GET", "/fapi/v1/leverageBracket", params)
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

    def cancel_all_algo_open_orders(self, symbol: str) -> dict[str, Any]:
        return self._signed_request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol.upper()})

    def start_user_data_stream(self) -> str:
        payload = self._api_key_request(
            "POST", "/fapi/v1/listenKey"
        )
        listen_key = str(payload.get("listenKey", ""))
        if not listen_key:
            raise BinanceApiError(
                None, "Binance user-data stream returned no listenKey"
            )
        return listen_key

    def keepalive_user_data_stream(
        self, listen_key: str
    ) -> dict[str, Any]:
        return self._api_key_request(
            "PUT",
            "/fapi/v1/listenKey",
            {"listenKey": listen_key},
        )

    def close_user_data_stream(
        self, listen_key: str
    ) -> dict[str, Any]:
        return self._api_key_request(
            "DELETE",
            "/fapi/v1/listenKey",
            {"listenKey": listen_key},
        )

    def query_order(
        self,
        symbol: str,
        *,
        order_id: int | str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not orig_client_order_id:
            raise ValueError("query_order requires order_id or orig_client_order_id")
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._signed_request("GET", "/fapi/v1/order", params)

    def cancel_order(
        self,
        symbol: str,
        *,
        order_id: int | str | None = None,
        orig_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if order_id is None and not orig_client_order_id:
            raise ValueError("cancel_order requires order_id or orig_client_order_id")
        params: dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = order_id
        if orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._signed_request("DELETE", "/fapi/v1/order", params)

    def query_algo_order(
        self,
        *,
        algo_id: int | str | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        if algo_id is None and not client_algo_id:
            raise ValueError("query_algo_order requires algo_id or client_algo_id")
        params: dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        return self._signed_request("GET", "/fapi/v1/algoOrder", params)

    def cancel_algo_order(
        self,
        *,
        algo_id: int | str | None = None,
        client_algo_id: str | None = None,
    ) -> dict[str, Any]:
        if algo_id is None and not client_algo_id:
            raise ValueError("cancel_algo_order requires algo_id or client_algo_id")
        params: dict[str, Any] = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id:
            params["clientAlgoId"] = client_algo_id
        return self._signed_request("DELETE", "/fapi/v1/algoOrder", params)

    def test_order(
        self,
        symbol: str,
        side: str,
        quantity: str,
        *,
        order_type: str = "MARKET",
        price: str | None = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        new_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT test_order requires price")
            params["price"] = price
            params["timeInForce"] = time_in_force.upper()
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id
        return self._signed_request("POST", "/fapi/v1/order/test", params)

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
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force.upper(),
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
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
        new_client_algo_id: str | None = None,
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
        if new_client_algo_id:
            params["clientAlgoId"] = new_client_algo_id
        return self._signed_request("POST", "/fapi/v1/algoOrder", params)

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
        if new_client_algo_id:
            params["clientAlgoId"] = new_client_algo_id
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
        return self._request(method, path, signed, api_key=True)

    def _api_key_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if not self.api_key:
            raise BinanceApiError(
                None, "Binance API key is required for this endpoint"
            )
        return self._request(method, path, params, api_key=True)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        api_key: bool = False,
    ) -> Any:
        params = params or {}
        try:
            self.request_weight_budget.reserve(method, path, params)
        except RequestWeightCooldown as exc:
            raise BinanceRateLimitError(
                exc.status,
                str(exc),
                retry_after_seconds=exc.retry_after_seconds,
                proactive=exc.proactive,
            ) from exc
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
        if api_key and self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        request = Request(url, data=data, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                response_headers = {
                    str(key): str(value)
                    for key, value in response.headers.items()
                }
                self.request_weight_budget.observe_response(
                    response_headers
                )
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            response_headers = {
                str(key): str(value)
                for key, value in (exc.headers or {}).items()
            }
            if exc.code in {418, 429}:
                cooldown = (
                    self.request_weight_budget.enter_exchange_cooldown(
                        exc.code, response_headers
                    )
                )
                raise BinanceRateLimitError(
                    exc.code,
                    _error_message(body),
                    _json_or_text(body),
                    retry_after_seconds=(
                        cooldown.retry_after_seconds
                    ),
                    proactive=False,
                    headers=response_headers,
                ) from exc
            self.request_weight_budget.observe_response(
                response_headers
            )
            raise BinanceApiError(
                exc.code,
                _error_message(body),
                _json_or_text(body),
                headers=response_headers,
            ) from exc
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
