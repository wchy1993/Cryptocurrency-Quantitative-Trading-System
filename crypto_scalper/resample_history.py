from __future__ import annotations

import argparse
import re
from pathlib import Path

from .data import interval_to_milliseconds, load_candles_csv, write_candles_csv
from .models import Candle


_FILENAME_RE = re.compile(r"^(?P<symbol>.+)_(?P<timeframe>[0-9]+[mhdw])_(?P<start>[0-9]{8})_(?P<end>[0-9]{8})\.csv$")


def resample_history(input_dir: str, output_dir: str, source_timeframe: str, target_timeframe: str) -> int:
    factor = _resample_factor(source_timeframe, target_timeframe)
    source_root = Path(input_dir)
    target_root = Path(output_dir)
    written = 0

    for source_path in sorted(source_root.glob(f"*_{source_timeframe}_*.csv")):
        match = _FILENAME_RE.match(source_path.name)
        if not match:
            continue
        candles = load_candles_csv(source_path)
        resampled = _resample(candles, factor)
        if not resampled:
            continue
        target_name = f"{match.group('symbol')}_{target_timeframe}_{match.group('start')}_{match.group('end')}.csv"
        write_candles_csv(target_root / target_name, resampled)
        written += 1
    return written


def _resample_factor(source_timeframe: str, target_timeframe: str) -> int:
    source_ms = interval_to_milliseconds(source_timeframe)
    target_ms = interval_to_milliseconds(target_timeframe)
    if target_ms < source_ms:
        raise ValueError(f"{target_timeframe} is shorter than {source_timeframe}")
    factor, remainder = divmod(target_ms, source_ms)
    if remainder:
        raise ValueError(f"{target_timeframe} is not an even multiple of {source_timeframe}")
    return factor


def _resample(candles: list[Candle], factor: int) -> list[Candle]:
    output: list[Candle] = []
    for start in range(0, len(candles), factor):
        chunk = candles[start:start + factor]
        if len(chunk) < factor:
            break
        output.append(
            Candle(
                timestamp=chunk[-1].timestamp,
                open=chunk[0].open,
                high=max(candle.high for candle in chunk),
                low=min(candle.low for candle in chunk),
                close=chunk[-1].close,
                volume=sum(candle.volume for candle in chunk),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-timeframe", default="15m")
    parser.add_argument("--target-timeframe", default="30m")
    args = parser.parse_args()

    written = resample_history(args.input_dir, args.output_dir, args.source_timeframe, args.target_timeframe)
    print(f"written={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
