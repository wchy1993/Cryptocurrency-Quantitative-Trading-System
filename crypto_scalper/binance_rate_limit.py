from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Mapping


DEFAULT_REQUEST_WEIGHT_LIMIT = 2_400


class RequestWeightCooldown(RuntimeError):
    """Raised before a REST call when the shared IP budget is cooling down."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float,
        proactive: bool,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        self.proactive = bool(proactive)
        self.status = status


@dataclass(frozen=True)
class RequestWeightStatus:
    limit: int
    soft_limit: int
    window_epoch_minute: int
    estimated_used_weight: int
    observed_used_weight: int
    effective_used_weight: int
    cooldown_until_epoch: float
    cooldown_remaining_seconds: float
    request_count: int
    response_count: int
    rate_limit_count: int
    proactive_cooldown_count: int
    endpoint_counts: dict[str, int]
    endpoint_estimated_weight: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RequestWeightBudget:
    """Conservative, header-aware Binance request-weight governor.

    Binance request weight is shared by every process behind the same public
    IP.  The local estimate therefore cannot be treated as the source of truth:
    successful response headers are folded into the same rolling state and
    always win when they report a higher number.
    """

    def __init__(
        self,
        *,
        limit: int = DEFAULT_REQUEST_WEIGHT_LIMIT,
        soft_limit_ratio: float = 0.65,
        default_cooldown_seconds: float = 65.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if limit <= 0:
            raise ValueError("request-weight limit must be positive")
        if not 0.1 <= soft_limit_ratio < 1.0:
            raise ValueError(
                "request-weight soft-limit ratio must be in [0.1, 1.0)"
            )
        if default_cooldown_seconds <= 0.0:
            raise ValueError("default cooldown must be positive")
        self.limit = int(limit)
        self.soft_limit = max(
            1, int(math.floor(self.limit * soft_limit_ratio))
        )
        self.default_cooldown_seconds = float(
            default_cooldown_seconds
        )
        self._clock = clock
        self._lock = threading.RLock()
        now = self._clock()
        self._window_epoch_minute = int(now // 60)
        self._estimated_used_weight = 0
        self._observed_used_weight = 0
        self._cooldown_until_epoch = 0.0
        self._request_count = 0
        self._response_count = 0
        self._rate_limit_count = 0
        self._proactive_cooldown_count = 0
        self._endpoint_counts: dict[str, int] = {}
        self._endpoint_estimated_weight: dict[str, int] = {}

    def reserve(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        weight = endpoint_request_weight(method, path, params)
        endpoint = f"{method.upper()} {path}"
        with self._lock:
            now = self._clock()
            self._roll_window(now)
            remaining = self._cooldown_until_epoch - now
            if remaining > 0.0:
                raise RequestWeightCooldown(
                    "Binance REST request blocked by active cooldown",
                    retry_after_seconds=remaining,
                    proactive=False,
                )
            effective = max(
                self._estimated_used_weight,
                self._observed_used_weight,
            )
            if effective + weight > self.soft_limit:
                next_window = (int(now // 60) + 1) * 60 + 1.0
                self._cooldown_until_epoch = max(
                    self._cooldown_until_epoch, next_window
                )
                self._proactive_cooldown_count += 1
                raise RequestWeightCooldown(
                    "Binance REST soft request-weight limit reached",
                    retry_after_seconds=next_window - now,
                    proactive=True,
                )
            self._estimated_used_weight += weight
            self._request_count += 1
            self._endpoint_counts[endpoint] = (
                self._endpoint_counts.get(endpoint, 0) + 1
            )
            self._endpoint_estimated_weight[endpoint] = (
                self._endpoint_estimated_weight.get(endpoint, 0) + weight
            )
            return weight

    def observe_response(self, headers: Mapping[str, Any]) -> None:
        with self._lock:
            now = self._clock()
            self._roll_window(now)
            observed = _used_weight_1m(headers)
            if observed is not None:
                self._observed_used_weight = max(
                    self._observed_used_weight, observed
                )
            self._response_count += 1

    def enter_exchange_cooldown(
        self,
        status: int,
        headers: Mapping[str, Any],
    ) -> RequestWeightCooldown:
        with self._lock:
            now = self._clock()
            self._roll_window(now)
            observed = _used_weight_1m(headers)
            if observed is not None:
                self._observed_used_weight = max(
                    self._observed_used_weight, observed
                )
            retry_after = _retry_after_seconds(headers, now)
            if retry_after is None:
                retry_after = (
                    300.0
                    if int(status) == 418
                    else self.default_cooldown_seconds
                )
            retry_after = max(1.0, retry_after)
            self._cooldown_until_epoch = max(
                self._cooldown_until_epoch, now + retry_after
            )
            self._rate_limit_count += 1
            self._response_count += 1
            return RequestWeightCooldown(
                f"Binance HTTP {status} forced REST cooldown",
                retry_after_seconds=(
                    self._cooldown_until_epoch - now
                ),
                proactive=False,
                status=int(status),
            )

    def remaining_cooldown_seconds(self) -> float:
        with self._lock:
            return max(
                0.0, self._cooldown_until_epoch - self._clock()
            )

    def status(self) -> RequestWeightStatus:
        with self._lock:
            now = self._clock()
            self._roll_window(now)
            effective = max(
                self._estimated_used_weight,
                self._observed_used_weight,
            )
            return RequestWeightStatus(
                limit=self.limit,
                soft_limit=self.soft_limit,
                window_epoch_minute=self._window_epoch_minute,
                estimated_used_weight=self._estimated_used_weight,
                observed_used_weight=self._observed_used_weight,
                effective_used_weight=effective,
                cooldown_until_epoch=self._cooldown_until_epoch,
                cooldown_remaining_seconds=max(
                    0.0, self._cooldown_until_epoch - now
                ),
                request_count=self._request_count,
                response_count=self._response_count,
                rate_limit_count=self._rate_limit_count,
                proactive_cooldown_count=(
                    self._proactive_cooldown_count
                ),
                endpoint_counts=dict(self._endpoint_counts),
                endpoint_estimated_weight=dict(
                    self._endpoint_estimated_weight
                ),
            )

    def _roll_window(self, now: float) -> None:
        current = int(now // 60)
        if current == self._window_epoch_minute:
            return
        self._window_epoch_minute = current
        self._estimated_used_weight = 0
        self._observed_used_weight = 0


def endpoint_request_weight(
    method: str,
    path: str,
    params: Mapping[str, Any] | None = None,
) -> int:
    """Return a conservative USD-M REST request-weight estimate."""

    del method
    values = params or {}
    if path == "/fapi/v1/klines":
        limit = int(values.get("limit", 500))
        if limit < 100:
            return 1
        if limit < 500:
            return 2
        if limit <= 1_000:
            return 5
        return 10
    if path in {
        "/fapi/v1/openOrders",
        "/fapi/v1/openAlgoOrders",
    }:
        return 1 if values.get("symbol") else 40
    if path in {
        "/fapi/v2/account",
        "/fapi/v3/account",
        "/fapi/v2/positionRisk",
        "/fapi/v3/positionRisk",
        "/fapi/v1/userTrades",
    }:
        return 5
    if path == "/fapi/v1/premiumIndex":
        return 1 if values.get("symbol") else 10
    return 1


def _header(
    headers: Mapping[str, Any], name: str
) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if str(key).lower() == target:
            return str(value)
    return None


def _used_weight_1m(
    headers: Mapping[str, Any],
) -> int | None:
    raw = _header(headers, "X-MBX-USED-WEIGHT-1M")
    if raw is None:
        return None
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(
    headers: Mapping[str, Any],
    now: float,
) -> float | None:
    raw = _header(headers, "Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, parsed.timestamp() - now)
