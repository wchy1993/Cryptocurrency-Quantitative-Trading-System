from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

import crypto_scalper.binance_client as binance_client_module
from crypto_scalper.binance_client import (
    BinanceFuturesClient,
    BinanceRateLimitError,
)
from crypto_scalper.binance_rate_limit import (
    RequestWeightBudget,
    RequestWeightCooldown,
    endpoint_request_weight,
)
from crypto_scalper.binance_streams import (
    BinanceFuturesStreamCache,
    MAINNET_MARKET_STREAM_URL,
)
from crypto_scalper.models import Candle


def test_request_weight_estimates_penalize_account_wide_routes() -> None:
    assert (
        endpoint_request_weight(
            "GET", "/fapi/v1/openOrders", {}
        )
        == 40
    )
    assert (
        endpoint_request_weight(
            "GET",
            "/fapi/v1/openOrders",
            {"symbol": "BTCUSDT"},
        )
        == 1
    )
    assert (
        endpoint_request_weight(
            "GET", "/fapi/v1/klines", {"limit": 500}
        )
        == 5
    )
    assert (
        endpoint_request_weight(
            "GET", "/fapi/v1/klines", {"limit": 1_500}
        )
        == 10
    )


def test_request_weight_budget_stops_before_exchange_limit() -> None:
    now = [120.0]
    budget = RequestWeightBudget(
        limit=100,
        soft_limit_ratio=0.50,
        clock=lambda: now[0],
    )

    budget.reserve("GET", "/fapi/v1/openOrders", {})
    budget.observe_response({"X-MBX-USED-WEIGHT-1M": "48"})

    with pytest.raises(RequestWeightCooldown) as raised:
        budget.reserve("GET", "/fapi/v1/openOrders", {})

    assert raised.value.proactive is True
    assert budget.status().proactive_cooldown_count == 1
    assert budget.status().effective_used_weight == 48

    now[0] = 181.1
    assert budget.reserve("GET", "/fapi/v1/ping", {}) == 1


def test_http_429_forces_cooldown_and_blocks_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def rate_limited(*_args, **_kwargs):
        calls.append(True)
        headers = Message()
        headers["Retry-After"] = "12"
        headers["X-MBX-USED-WEIGHT-1M"] = "2400"
        raise HTTPError(
            "https://fapi.binance.com/fapi/v1/ping",
            429,
            "Too Many Requests",
            headers,
            BytesIO(
                json.dumps(
                    {
                        "code": -1003,
                        "msg": "Too many requests",
                    }
                ).encode("utf-8")
            ),
        )

    monkeypatch.setattr(
        binance_client_module, "urlopen", rate_limited
    )
    client = BinanceFuturesClient(environment="mainnet")

    with pytest.raises(BinanceRateLimitError) as first:
        client.ping()
    assert first.value.status == 429
    assert first.value.retry_after_seconds == pytest.approx(12.0)

    with pytest.raises(BinanceRateLimitError) as second:
        client.ping()
    assert second.value.proactive is False
    assert len(calls) == 1
    assert client.rate_limit_status()["rate_limit_count"] == 1


class _FakeStreamClient:
    environment = "mainnet"

    def __init__(self) -> None:
        self.listen_key_starts = 0
        self.listen_key_closes = 0

    def start_user_data_stream(self) -> str:
        self.listen_key_starts += 1
        return "listen-key"

    def keepalive_user_data_stream(self, _listen_key: str):
        return {}

    def close_user_data_stream(self, _listen_key: str):
        self.listen_key_closes += 1
        return {}


class _FakeWebSocketApp:
    def __init__(self, url: str, **callbacks) -> None:
        self.url = url
        self.callbacks = callbacks
        self.closed = threading.Event()
        self.sent: list[str] = []

    def send(self, message: str) -> None:
        self.sent.append(message)

    def run_forever(self, **_kwargs) -> None:
        self.callbacks["on_open"](self)
        if "/private/ws/" in self.url:
            self.callbacks["on_message"](
                self,
                json.dumps(
                    {
                        "e": "ACCOUNT_UPDATE",
                        "E": 1_700_000_000_000,
                        "a": {"B": [], "P": []},
                    }
                ),
            )
        else:
            self.callbacks["on_message"](
                self,
                json.dumps({"result": None, "id": 1}),
            )
            self.callbacks["on_message"](
                self,
                json.dumps(
                    {
                        "stream": "btcusdt@kline_1m",
                        "data": {
                            "e": "kline",
                            "s": "BTCUSDT",
                            "k": {
                                "t": 1_700_000_060_000,
                                "s": "BTCUSDT",
                                "i": "1m",
                                "o": "101",
                                "h": "103",
                                "l": "100",
                                "c": "102",
                                "v": "12",
                            },
                        },
                    }
                ),
            )
            self.callbacks["on_message"](
                self,
                json.dumps(
                    {
                        "stream": "btcusdt@markPrice@1s",
                        "data": {
                            "e": "markPriceUpdate",
                            "s": "BTCUSDT",
                            "p": "102.5",
                            "i": "102.4",
                            "r": "0.0001",
                            "T": 1_700_000_100_000,
                        },
                    }
                ),
            )
        self.closed.wait(timeout=2.0)

    def close(self) -> None:
        self.closed.set()


def test_websocket_cache_updates_market_and_user_state() -> None:
    client = _FakeStreamClient()
    apps: list[_FakeWebSocketApp] = []

    def factory(url: str, **callbacks):
        app = _FakeWebSocketApp(url, **callbacks)
        apps.append(app)
        return app

    cache = BinanceFuturesStreamCache(
        client,  # type: ignore[arg-type]
        ("BTCUSDT",),
        websocket_app_factory=factory,
        listen_key_keepalive_seconds=300,
    )
    cache.seed_candles(
        "BTCUSDT",
        "1m",
        [
            Candle(
                datetime.utcfromtimestamp(1_700_000_000),
                100,
                101,
                99,
                100.5,
                10,
            )
        ],
    )
    stop = threading.Event()
    cache.start(stop, include_user_stream=True)
    try:
        assert cache.wait_until_ready(
            1.0, require_user_stream=True
        )
        deadline = time.time() + 1.0
        while (
            cache.health()["market_event_count"] < 2
            or cache.health()["user_event_count"] < 1
        ) and time.time() < deadline:
            time.sleep(0.01)

        rows = cache.candles("BTCUSDT", "1m", 10)
        assert rows is not None
        assert rows[-1].close == pytest.approx(102.0)
        assert cache.premium_index("BTCUSDT")["markPrice"] == "102.5"
        revision, events = cache.recent_user_events()
        assert revision == 1
        assert events[-1]["e"] == "ACCOUNT_UPDATE"
        health = cache.health()
        assert health["market_connected"] is True
        assert health["market_subscription_acknowledged"] is True
        assert health["market_healthy"] is True
        assert health["user_connected"] is True
        assert health["cache_hits"] >= 2
        assert any(app.sent for app in apps)
        market_app = next(
            app for app in apps if "/market/stream" in app.url
        )
        assert (
            market_app.url
            == "wss://fstream.binance.com/market/stream"
        )
        assert (
            MAINNET_MARKET_STREAM_URL
            == "wss://fstream.binance.com/market/stream"
        )
    finally:
        stop.set()
        cache.stop()

    assert client.listen_key_starts == 1
    assert client.listen_key_closes == 1


def test_websocket_cache_keeps_time_order_when_event_precedes_seed() -> None:
    cache = BinanceFuturesStreamCache(
        _FakeStreamClient(),  # type: ignore[arg-type]
        ("BTCUSDT",),
        websocket_app_factory=lambda *_args, **_kwargs: None,
    )
    current_ms = 1_700_000_120_000
    cache._on_market_message(
        None,
        json.dumps(
            {
                "data": {
                    "e": "kline",
                    "s": "BTCUSDT",
                    "k": {
                        "t": current_ms,
                        "s": "BTCUSDT",
                        "i": "1m",
                        "o": "102",
                        "h": "104",
                        "l": "101",
                        "c": "103",
                        "v": "13",
                    },
                }
            }
        ),
    )
    older = Candle(
        datetime.utcfromtimestamp((current_ms - 60_000) / 1_000),
        100,
        102,
        99,
        101,
        12,
    )
    current = Candle(
        datetime.utcfromtimestamp(current_ms / 1_000),
        102,
        104,
        101,
        103,
        13,
    )
    cache.seed_candles("BTCUSDT", "1m", [older, current])

    rows = cache.candles("BTCUSDT", "1m", 10)
    assert rows is not None
    assert [row.timestamp for row in rows] == sorted(
        row.timestamp for row in rows
    )
    assert rows[-1].timestamp == current.timestamp
