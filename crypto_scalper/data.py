from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import Candle


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def load_candles_csv(path: str | Path) -> list[Candle]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(missing)}")
        candles = [
            Candle(
                timestamp=parse_timestamp(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for row in reader
        ]

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
