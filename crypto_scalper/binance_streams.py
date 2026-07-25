from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any, Callable

from .binance_client import (
    BinanceFuturesClient,
    BinanceRateLimitError,
)
from .models import Candle

try:
    import websocket
except ImportError:  # pragma: no cover - exercised by startup validation
    websocket = None


# Binance permanently retired the unrouted USD-M Futures market path on
# 2026-04-23.  Klines, mark price, aggregate trades, and tickers belong on the
# regular ``/market`` route; the legacy ``/stream`` endpoint still upgrades the
# socket but no longer publishes those event types.
MAINNET_MARKET_STREAM_URL = (
    "wss://fstream.binance.com/market/stream"
)
MAINNET_USER_STREAM_URL = "wss://fstream.binance.com/private/ws"
TESTNET_MARKET_STREAM_URL = (
    "wss://fstream.binancefuture.com/stream"
)
TESTNET_USER_STREAM_URL = "wss://fstream.binancefuture.com/ws"


class BinanceFuturesStreamCache:
    """Thread-safe USD-M market and user-data WebSocket cache.

    Historical candle warm-up remains REST-backed.  Once seeded, live kline
    and mark-price events update the same cache so strategy code can keep using
    the normal ``client.klines`` and ``client.premium_index`` interfaces
    without repeatedly polling Binance.
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        symbols: tuple[str, ...],
        *,
        logger: Callable[[str], None] | None = None,
        stale_after_seconds: float = 180.0,
        listen_key_keepalive_seconds: float = 1_800.0,
        reconnect_max_seconds: float = 30.0,
        websocket_app_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if stale_after_seconds <= 0.0:
            raise ValueError("stream stale threshold must be positive")
        if listen_key_keepalive_seconds <= 0.0:
            raise ValueError("listen-key keepalive must be positive")
        self.client = client
        self.symbols = tuple(
            dict.fromkeys(str(symbol).upper() for symbol in symbols)
        )
        self.logger = logger or (lambda _message: None)
        self.stale_after_seconds = float(stale_after_seconds)
        self.listen_key_keepalive_seconds = float(
            listen_key_keepalive_seconds
        )
        self.reconnect_max_seconds = max(
            1.0, float(reconnect_max_seconds)
        )
        self.websocket_app_factory = (
            websocket_app_factory
            or (
                websocket.WebSocketApp
                if websocket is not None
                else None
            )
        )
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._candles: dict[
            tuple[str, str], OrderedDict[int, Candle]
        ] = {}
        self._seeded: set[tuple[str, str]] = set()
        self._series_updated_at: dict[tuple[str, str], float] = {}
        self._marks: dict[str, tuple[dict[str, Any], float]] = {}
        self._user_events: deque[dict[str, Any]] = deque(maxlen=2_000)
        self._user_revision = 0
        self._market_connected = False
        self._user_connected = False
        self._market_subscription_acknowledged = False
        self._market_subscription_error = ""
        self._last_market_event_epoch = 0.0
        self._last_user_event_epoch = 0.0
        self._market_event_count = 0
        self._user_event_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._rest_seed_count = 0
        self._market_reconnects = 0
        self._user_reconnects = 0
        self._last_market_error = ""
        self._last_user_error = ""
        self._listen_key: str | None = None
        self._listen_key_expired = False
        self._last_keepalive_monotonic = 0.0
        self._market_app: Any | None = None
        self._user_app: Any | None = None
        self._threads: list[threading.Thread] = []
        self._external_stop: threading.Event | None = None
        self._local_stop = threading.Event()
        self._include_user_stream = True

    @property
    def user_revision(self) -> int:
        with self._lock:
            return self._user_revision

    def start(
        self,
        stop_event: threading.Event,
        *,
        include_user_stream: bool = True,
    ) -> None:
        if self.websocket_app_factory is None:
            raise RuntimeError(
                "websocket-client is required for Binance LIVE streams"
            )
        with self._lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._external_stop = stop_event
            self._local_stop.clear()
            self._include_user_stream = bool(include_user_stream)
            self._threads = [
                threading.Thread(
                    target=self._market_loop,
                    name="binance-usdm-market-stream",
                    daemon=True,
                )
            ]
            if include_user_stream:
                self._threads.extend(
                    [
                        threading.Thread(
                            target=self._user_loop,
                            name="binance-usdm-user-stream",
                            daemon=True,
                        ),
                        threading.Thread(
                            target=self._keepalive_loop,
                            name="binance-usdm-listen-key",
                            daemon=True,
                        ),
                    ]
                )
            for thread in self._threads:
                thread.start()

    def stop(self) -> None:
        self._local_stop.set()
        with self._condition:
            for app in (self._market_app, self._user_app):
                if app is not None:
                    try:
                        app.close()
                    except Exception:
                        pass
            self._condition.notify_all()
        for thread in tuple(self._threads):
            thread.join(timeout=3.0)
        listen_key: str | None
        with self._lock:
            listen_key = self._listen_key
            self._listen_key = None
            self._market_connected = False
            self._user_connected = False
        if listen_key:
            try:
                self.client.close_user_data_stream(listen_key)
            except Exception:
                pass

    def wait_until_ready(
        self,
        timeout_seconds: float,
        *,
        require_user_stream: bool,
    ) -> bool:
        deadline = self._monotonic() + max(0.0, timeout_seconds)
        with self._condition:
            while True:
                ready = (
                    self._market_connected
                    and self._market_subscription_acknowledged
                    and self._market_event_count > 0
                    and (
                        self._user_connected
                        or not require_user_stream
                    )
                )
                if ready:
                    return True
                remaining = deadline - self._monotonic()
                if remaining <= 0.0 or self._should_stop():
                    return False
                self._condition.wait(timeout=min(0.25, remaining))

    def candles(
        self, symbol: str, interval: str, limit: int
    ) -> list[Candle] | None:
        key = (symbol.upper(), str(interval))
        with self._lock:
            series = self._candles.get(key)
            updated = self._series_updated_at.get(key, 0.0)
            fresh = (
                updated > 0.0
                and self._monotonic() - updated
                <= self.stale_after_seconds
            )
            if key not in self._seeded or not series or not fresh:
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            # REST warm-up can race the first WebSocket event.  Always expose
            # candles in exchange-time order even when the current candle was
            # inserted before its older history arrived.
            return [
                candle
                for _timestamp, candle in sorted(series.items())
            ][-max(1, int(limit)) :]

    def seed_candles(
        self,
        symbol: str,
        interval: str,
        candles: list[Candle],
    ) -> None:
        key = (symbol.upper(), str(interval))
        with self._lock:
            series = self._candles.setdefault(key, OrderedDict())
            for candle in candles:
                timestamp_ms = int(
                    candle.timestamp.replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                    * 1_000
                )
                series[timestamp_ms] = candle
            series = OrderedDict(sorted(series.items()))
            self._candles[key] = series
            self._trim_series(series, interval)
            self._seeded.add(key)
            self._series_updated_at[key] = self._monotonic()
            self._rest_seed_count += 1

    def premium_index(
        self, symbol: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._marks.get(symbol.upper())
            if row is None:
                self._cache_misses += 1
                return None
            payload, updated = row
            if (
                self._monotonic() - updated
                > self.stale_after_seconds
            ):
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return dict(payload)

    def recent_user_events(
        self, *, after_revision: int = 0
    ) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            if after_revision >= self._user_revision:
                return self._user_revision, []
            return self._user_revision, [
                dict(event) for event in self._user_events
            ]

    def health(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            market_age = (
                max(0.0, now - self._last_market_event_epoch)
                if self._last_market_event_epoch
                else None
            )
            return {
                "market_connected": self._market_connected,
                "market_subscription_acknowledged": (
                    self._market_subscription_acknowledged
                ),
                "market_subscription_error": (
                    self._market_subscription_error
                ),
                "market_healthy": (
                    self._market_connected
                    and self._market_subscription_acknowledged
                    and market_age is not None
                    and market_age <= self.stale_after_seconds
                ),
                "user_connected": self._user_connected,
                "last_market_event_at": (
                    self._last_market_event_epoch or None
                ),
                "last_user_event_at": (
                    self._last_user_event_epoch or None
                ),
                "market_event_age_seconds": market_age,
                "user_event_age_seconds": (
                    max(0.0, now - self._last_user_event_epoch)
                    if self._last_user_event_epoch
                    else None
                ),
                "market_event_count": self._market_event_count,
                "user_event_count": self._user_event_count,
                "user_revision": self._user_revision,
                "seeded_series": len(self._seeded),
                "symbols_with_stream_data": len(
                    {
                        symbol
                        for symbol, _interval in self._candles
                    }
                ),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "rest_seed_count": self._rest_seed_count,
                "market_reconnects": self._market_reconnects,
                "user_reconnects": self._user_reconnects,
                "last_market_error": self._last_market_error,
                "last_user_error": self._last_user_error,
                "listen_key_active": bool(self._listen_key),
            }

    def _market_loop(self) -> None:
        backoff = 1.0
        while not self._should_stop():
            app = self.websocket_app_factory(
                self._market_url(),
                on_open=self._on_market_open,
                on_message=self._on_market_message,
                on_error=self._on_market_error,
                on_close=self._on_market_close,
            )
            with self._lock:
                self._market_app = app
            try:
                app.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                self._on_market_error(app, exc)
            finally:
                with self._condition:
                    self._market_connected = False
                    self._market_app = None
                    self._condition.notify_all()
            if self._wait(backoff):
                break
            with self._lock:
                self._market_reconnects += 1
            backoff = min(self.reconnect_max_seconds, backoff * 2.0)

    def _user_loop(self) -> None:
        backoff = 1.0
        while not self._should_stop():
            try:
                listen_key = self._ensure_listen_key()
            except BinanceRateLimitError as exc:
                self._set_user_error(exc)
                if self._wait(exc.retry_after_seconds):
                    break
                continue
            except Exception as exc:
                self._set_user_error(exc)
                if self._wait(backoff):
                    break
                backoff = min(
                    self.reconnect_max_seconds, backoff * 2.0
                )
                continue
            app = self.websocket_app_factory(
                f"{self._user_url()}/{listen_key}",
                on_open=self._on_user_open,
                on_message=self._on_user_message,
                on_error=self._on_user_error,
                on_close=self._on_user_close,
            )
            with self._lock:
                self._user_app = app
            try:
                app.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                self._on_user_error(app, exc)
            finally:
                with self._condition:
                    self._user_connected = False
                    self._user_app = None
                    expired = self._listen_key_expired
                    self._condition.notify_all()
                if expired:
                    self._discard_listen_key()
            if self._wait(backoff):
                break
            with self._lock:
                self._user_reconnects += 1
            backoff = min(self.reconnect_max_seconds, backoff * 2.0)

    def _keepalive_loop(self) -> None:
        while not self._should_stop():
            if self._wait(5.0):
                return
            with self._lock:
                listen_key = self._listen_key
                elapsed = (
                    self._monotonic()
                    - self._last_keepalive_monotonic
                )
            if (
                not listen_key
                or elapsed < self.listen_key_keepalive_seconds
            ):
                continue
            try:
                self.client.keepalive_user_data_stream(listen_key)
                with self._lock:
                    self._last_keepalive_monotonic = (
                        self._monotonic()
                    )
            except BinanceRateLimitError as exc:
                self._set_user_error(exc)
                if self._wait(exc.retry_after_seconds):
                    return
            except Exception as exc:
                self._set_user_error(exc)
                with self._lock:
                    self._listen_key_expired = True
                    app = self._user_app
                if app is not None:
                    try:
                        app.close()
                    except Exception:
                        pass

    def _on_market_open(self, app: Any) -> None:
        streams: list[str] = []
        for symbol in self.symbols:
            name = symbol.lower()
            streams.extend(
                (
                    f"{name}@kline_1m",
                    f"{name}@kline_1h",
                    f"{name}@markPrice@1s",
                )
            )
        app.send(
            json.dumps(
                {
                    "method": "SUBSCRIBE",
                    "params": streams,
                    "id": 1,
                },
                separators=(",", ":"),
            )
        )
        with self._condition:
            self._market_connected = True
            self._market_subscription_acknowledged = False
            self._market_subscription_error = ""
            self._last_market_error = ""
            self._condition.notify_all()

    def _on_market_message(self, _app: Any, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return
        if payload.get("id") == 1:
            if "result" in payload and payload.get("result") is None:
                with self._condition:
                    self._market_subscription_acknowledged = True
                    self._condition.notify_all()
                return
            error = payload.get("error") or payload.get("msg") or payload
            with self._condition:
                self._market_subscription_error = str(error)
                self._last_market_error = (
                    f"subscription rejected: {error}"
                )
                self._condition.notify_all()
            try:
                _app.close()
            except Exception:
                pass
            return
        data = payload.get("data", payload)
        event_type = str(data.get("e", ""))
        if event_type == "kline" and isinstance(data.get("k"), dict):
            self._update_kline(data)
        elif event_type == "markPriceUpdate":
            self._update_mark(data)
        else:
            return
        with self._condition:
            self._last_market_event_epoch = self._clock()
            self._market_event_count += 1
            self._condition.notify_all()

    def _on_market_error(self, _app: Any, error: Any) -> None:
        with self._condition:
            self._last_market_error = (
                f"{type(error).__name__}: {error}"
            )
            self._condition.notify_all()

    def _on_market_close(
        self, _app: Any, _status: Any, message: Any
    ) -> None:
        with self._condition:
            self._market_connected = False
            self._market_subscription_acknowledged = False
            if message:
                self._last_market_error = str(message)
            self._condition.notify_all()

    def _on_user_open(self, _app: Any) -> None:
        with self._condition:
            self._user_connected = True
            self._last_user_error = ""
            self._condition.notify_all()

    def _on_user_message(self, app: Any, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return
        event_type = str(payload.get("e", ""))
        with self._condition:
            self._last_user_event_epoch = self._clock()
            self._user_event_count += 1
            self._user_revision += 1
            self._user_events.append(dict(payload))
            if event_type == "listenKeyExpired":
                self._listen_key_expired = True
            self._condition.notify_all()
        if event_type == "listenKeyExpired":
            try:
                app.close()
            except Exception:
                pass

    def _on_user_error(self, _app: Any, error: Any) -> None:
        self._set_user_error(error)

    def _on_user_close(
        self, _app: Any, _status: Any, message: Any
    ) -> None:
        with self._condition:
            self._user_connected = False
            if message:
                self._last_user_error = str(message)
            self._condition.notify_all()

    def _update_kline(self, data: dict[str, Any]) -> None:
        raw = data["k"]
        symbol = str(raw.get("s") or data.get("s") or "").upper()
        interval = str(raw.get("i", ""))
        if not symbol or interval not in {"1m", "1h"}:
            return
        timestamp_ms = int(raw["t"])
        candle = Candle(
            timestamp=datetime.fromtimestamp(
                timestamp_ms / 1_000.0, tz=timezone.utc
            ).replace(tzinfo=None),
            open=float(raw["o"]),
            high=float(raw["h"]),
            low=float(raw["l"]),
            close=float(raw["c"]),
            volume=float(raw["v"]),
        )
        candle.validate()
        key = (symbol, interval)
        with self._lock:
            series = self._candles.setdefault(key, OrderedDict())
            series[timestamp_ms] = candle
            if series and timestamp_ms < next(reversed(series)):
                series = OrderedDict(sorted(series.items()))
                self._candles[key] = series
            else:
                series.move_to_end(timestamp_ms)
            self._trim_series(series, interval)
            self._series_updated_at[key] = self._monotonic()

    def _update_mark(self, data: dict[str, Any]) -> None:
        symbol = str(data.get("s", "")).upper()
        price = data.get("p")
        if not symbol or price is None:
            return
        with self._lock:
            self._marks[symbol] = (
                {
                    "symbol": symbol,
                    "markPrice": str(price),
                    "indexPrice": str(data.get("i", "")),
                    "lastFundingRate": str(data.get("r", "")),
                    "nextFundingTime": data.get("T"),
                },
                self._monotonic(),
            )

    def _ensure_listen_key(self) -> str:
        with self._lock:
            if self._listen_key and not self._listen_key_expired:
                return self._listen_key
        listen_key = self.client.start_user_data_stream()
        with self._lock:
            self._listen_key = listen_key
            self._listen_key_expired = False
            self._last_keepalive_monotonic = self._monotonic()
        return listen_key

    def _discard_listen_key(self) -> None:
        with self._lock:
            listen_key = self._listen_key
            self._listen_key = None
            self._listen_key_expired = False
        if listen_key:
            try:
                self.client.close_user_data_stream(listen_key)
            except Exception:
                pass

    def _set_user_error(self, error: Any) -> None:
        with self._condition:
            self._last_user_error = (
                f"{type(error).__name__}: {error}"
            )
            self._condition.notify_all()

    def _trim_series(
        self, series: OrderedDict[int, Candle], interval: str
    ) -> None:
        maximum = 1_600 if interval == "1m" else 550
        while len(series) > maximum:
            series.popitem(last=False)

    def _market_url(self) -> str:
        return (
            MAINNET_MARKET_STREAM_URL
            if self.client.environment == "mainnet"
            else TESTNET_MARKET_STREAM_URL
        )

    def _user_url(self) -> str:
        return (
            MAINNET_USER_STREAM_URL
            if self.client.environment == "mainnet"
            else TESTNET_USER_STREAM_URL
        )

    def _should_stop(self) -> bool:
        return self._local_stop.is_set() or bool(
            self._external_stop is not None
            and self._external_stop.is_set()
        )

    def _wait(self, seconds: float) -> bool:
        deadline = self._monotonic() + max(0.0, seconds)
        while not self._should_stop():
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False
            self._local_stop.wait(min(0.5, remaining))
        return True
