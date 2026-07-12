from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SliceSpec:
    name: str
    start: str
    end: str


def _parse_slice(value: str) -> SliceSpec:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("slice must be name,start_iso,end_iso")
    return SliceSpec(*parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream time slices from large Binance candle CSV directories")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--slice", dest="slices", type=_parse_slice, action="append", required=True)
    args = parser.parse_args()

    symbols = json.loads(args.config.read_text())["trading"]["symbols"]
    for spec in args.slices:
        (args.output_root / spec.name).mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        matches = sorted(args.source.glob(f"{symbol}_1m_*.csv"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one source file for {symbol}, found {len(matches)}")
        source = matches[0]
        outputs = {
            spec: (args.output_root / spec.name / source.name).open("w", encoding="utf-8", newline="")
            for spec in args.slices
        }
        try:
            with source.open(encoding="utf-8", newline="") as handle:
                header = handle.readline()
                for output in outputs.values():
                    output.write(header)
                for line in handle:
                    timestamp = line.split(",", 1)[0]
                    for spec, output in outputs.items():
                        if spec.start <= timestamp < spec.end:
                            output.write(line)
        finally:
            for output in outputs.values():
                output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
