#!/usr/bin/env python3
"""Build causal 15m candles for Breakout V16 from the frozen 1m archive.

The V16 research strategy keeps the original 1h decision clock.  Its two
30-minute phases and four 15-minute path observations are therefore derived
from the 60 one-minute candles that belong to the same completed hour.  This
utility deliberately writes only a new timeframe; it never modifies the 1m
or 1h V15 source data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


OHLCV_AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _symbols(config_path: Path) -> Iterable[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for pair in config["exchange"]["pair_whitelist"]:
        yield pair.replace("/", "_").replace(":", "_")


def _complete_resample(source: pd.DataFrame, minutes: int) -> pd.DataFrame:
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    indexed = frame.set_index("date")
    rule = f"{minutes}min"
    sampled = indexed.resample(rule, label="left", closed="left").agg(
        OHLCV_AGGREGATION
    )
    counts = indexed["close"].resample(rule, label="left", closed="left").count()
    sampled = sampled.loc[counts == minutes].dropna().reset_index()
    return sampled[["date", "open", "high", "low", "close", "volume"]]


def build(
    data_root: Path,
    config_path: Path,
    minutes: int,
    overwrite: bool,
) -> None:
    futures = data_root / "futures"
    if not futures.is_dir():
        raise FileNotFoundError(f"Futures data directory not found: {futures}")

    completed = 0
    skipped = 0
    for symbol in _symbols(config_path):
        source_path = futures / f"{symbol}-1m-futures.feather"
        target_path = futures / f"{symbol}-{minutes}m-futures.feather"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing V16 source data: {source_path}")
        if target_path.exists() and not overwrite:
            skipped += 1
            continue

        source = pd.read_feather(source_path)
        sampled = _complete_resample(source, minutes)
        temporary = target_path.with_suffix(target_path.suffix + ".tmp")
        sampled.to_feather(temporary)
        temporary.replace(target_path)
        completed += 1
        print(
            f"{symbol}: {len(source):,} x 1m -> {len(sampled):,} x "
            f"{minutes}m ({sampled['date'].min()} .. {sampled['date'].max()})"
        )

    print(f"complete={completed} skipped={skipped} timeframe={minutes}m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.backtest.json"))
    parser.add_argument("--minutes", type=int, choices=(15,), default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(
        data_root=args.data_root.resolve(),
        config_path=args.config.resolve(),
        minutes=args.minutes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
