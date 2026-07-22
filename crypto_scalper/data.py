from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .models import Candle


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def load_candles_csv(
    path: str | Path,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Candle]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}")
        start_token = start.isoformat(timespec="seconds") if start is not None else None
        end_token = end.isoformat(timespec="seconds") if end is not None else None
        candles = []
        for row in reader:
            timestamp_text = row["timestamp"]
            # Binance research files are UTC-naive, sorted ISO-8601 timestamps.
            # Lexical filtering avoids constructing millions of out-of-window
            # datetime/float objects during walk-forward research.
            if start_token is not None and timestamp_text < start_token:
                continue
            if end_token is not None and timestamp_text > end_token:
                break
            candles.append(
                Candle(
                    timestamp=parse_timestamp(timestamp_text),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )

    candles.sort(key=lambda candle: candle.timestamp)
    for candle in candles:
        candle.validate()
    return candles


def write_candles_csv(path: str | Path, candles: Iterable[Candle]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "timestamp": candle.timestamp.isoformat(),
                    "open": f"{candle.open:.8f}",
                    "high": f"{candle.high:.8f}",
                    "low": f"{candle.low:.8f}",
                    "close": f"{candle.close:.8f}",
                    "volume": f"{candle.volume:.6f}",
                }
            )


def download_binance_futures_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    limit: int = 1500,
    sleep_seconds: float = 0.05,
    timeout_seconds: int = 20,
) -> list[Candle]:
    if start >= end:
        raise ValueError("start must be before end")
    if limit <= 0 or limit > 1500:
        raise ValueError("limit must be between 1 and 1500")

    step_ms = interval_to_milliseconds(interval)
    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end)
    candles: list[Candle] = []
    seen_open_times: set[int] = set()

    while start_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        request = Request(
            f"{BINANCE_FUTURES_KLINES_URL}?{urlencode(params)}",
            headers={"User-Agent": "crypto-scalper/0.1"},
            method="GET",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            rows = json.loads(response.read().decode("utf-8"))
        if not rows:
            break

        last_open_ms = start_ms
        for row in rows:
            open_ms = int(row[0])
            last_open_ms = open_ms
            if open_ms in seen_open_times:
                continue
            seen_open_times.add(open_ms)
            candle = Candle(
                timestamp=datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).replace(tzinfo=None),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            candle.validate()
            candles.append(candle)

        next_start_ms = last_open_ms + step_ms
        if next_start_ms <= start_ms:
            break
        start_ms = next_start_ms
        if len(rows) >= limit and start_ms < end_ms and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    candles.sort(key=lambda candle: candle.timestamp)
    return candles


def interval_to_milliseconds(interval: str) -> int:
    if len(interval) < 2:
        raise ValueError(f"unsupported interval: {interval}")
    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"unsupported interval: {interval}") from exc
    factors = {
        "m": 60_000,
        "h": 60 * 60_000,
        "d": 24 * 60 * 60_000,
        "w": 7 * 24 * 60 * 60_000,
    }
    if value <= 0 or unit not in factors:
        raise ValueError(f"unsupported interval: {interval}")
    return value * factors[unit]


def generate_sample_candles(
    bars: int = 2_000,
    start_price: float = 60_000.0,
    seed: int = 42,
    start: datetime | None = None,
) -> list[Candle]:
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    rng = random.Random(seed)
    timestamp = start or datetime(2025, 1, 1, 0, 0, 0)
    price = start_price
    candles: list[Candle] = []

    drift = 0.0
    volatility = 0.0009
    for index in range(bars):
        if index % 360 == 0:
            drift = rng.choice((-0.00005, 0.0, 0.00006))
            volatility = rng.choice((0.00055, 0.0009, 0.0014))

        shock = rng.gauss(drift, volatility)
        if rng.random() < 0.012:
            shock += rng.choice((-1, 1)) * rng.uniform(0.002, 0.007)

        open_price = price
        close_price = max(1.0, open_price * math.exp(shock))
        body_high = max(open_price, close_price)
        body_low = min(open_price, close_price)
        wick_scale = abs(rng.gauss(0.0, volatility * 0.8)) + 0.00015
        high_price = body_high * (1.0 + wick_scale)
        low_price = max(1.0, body_low * (1.0 - wick_scale))
        volume = max(0.0, rng.lognormvariate(4.2, 0.45) * (1.0 + abs(shock) * 800.0))

        candles.append(
            Candle(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
            )
        )
        price = close_price
        timestamp += timedelta(minutes=1)

    return candles


def _to_utc_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(value.timestamp() * 1000)
