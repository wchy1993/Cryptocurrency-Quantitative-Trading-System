from __future__ import annotations

import argparse
import csv
import io
import json
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
METRICS_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    day: str
    status: str
    detail: str = ""


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / max(0.1, requests_per_second)
        self._lock = threading.Lock()
        self._next_request = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self._interval
        if delay:
            time.sleep(delay)


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def archive_name(symbol: str, day: date) -> str:
    return f"{symbol}-metrics-{day.isoformat()}.zip"


def archive_url(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol}/{archive_name(symbol, day)}"


def _valid_archive(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return len(names) == 1 and names[0].endswith(".csv") and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def download_archive(
    symbol: str,
    day: date,
    raw_root: Path,
    limiter: RateLimiter,
    retries: int,
    timeout_seconds: float,
) -> DownloadResult:
    symbol_root = raw_root / symbol
    symbol_root.mkdir(parents=True, exist_ok=True)
    destination = symbol_root / archive_name(symbol, day)
    missing_marker = destination.with_suffix(".missing")
    if destination.exists() and _valid_archive(destination):
        return DownloadResult(symbol, day.isoformat(), "cached")
    if missing_marker.exists():
        return DownloadResult(symbol, day.isoformat(), "missing_cached")

    for attempt in range(retries + 1):
        limiter.wait()
        request = Request(archive_url(symbol, day), headers={"User-Agent": "crypto-scalper-history/1.0"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            temp_path = destination.with_suffix(f".zip.part.{os.getpid()}.{threading.get_ident()}")
            temp_path.write_bytes(payload)
            if not _valid_archive(temp_path):
                temp_path.unlink(missing_ok=True)
                raise zipfile.BadZipFile("downloaded archive failed CRC/schema validation")
            temp_path.replace(destination)
            missing_marker.unlink(missing_ok=True)
            return DownloadResult(symbol, day.isoformat(), "downloaded")
        except HTTPError as exc:
            if exc.code == 404:
                missing_marker.write_text("404\n", encoding="ascii")
                return DownloadResult(symbol, day.isoformat(), "missing", "HTTP 404")
            detail = f"HTTP {exc.code}"
        except Exception as exc:  # Network and archive failures share the retry policy.
            detail = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    return DownloadResult(symbol, day.isoformat(), "failed", detail)


def read_archive_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if tuple(reader.fieldnames or ()) != METRICS_COLUMNS:
                raise ValueError(f"unexpected metrics columns in {path}: {reader.fieldnames}")
            return list(reader)


def _iso_timestamp(value: str) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return parsed.isoformat()


def merge_symbol(
    symbol: str,
    start: date,
    end: date,
    raw_root: Path,
    output_root: Path,
) -> dict[str, object]:
    rows_by_time: dict[str, dict[str, str]] = {}
    archive_days = 0
    for day in daterange(start, end):
        path = raw_root / symbol / archive_name(symbol, day)
        if not path.exists():
            continue
        archive_days += 1
        for row in read_archive_rows(path):
            timestamp = _iso_timestamp(row["create_time"])
            rows_by_time[timestamp] = row

    output_root.mkdir(parents=True, exist_ok=True)
    start_tag = start.strftime("%Y%m%d")
    end_tag = end.strftime("%Y%m%d")
    oi_path = output_root / f"{symbol}_oi_5m_{start_tag}_{end_tag}.csv"
    taker_path = output_root / f"{symbol}_taker_ratio_5m_{start_tag}_{end_tag}.csv"
    ordered = sorted(rows_by_time.items())

    with oi_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "sumOpenInterest", "sumOpenInterestValue"))
        writer.writeheader()
        for timestamp, row in ordered:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "sumOpenInterest": row["sum_open_interest"],
                    "sumOpenInterestValue": row["sum_open_interest_value"],
                }
            )

    # The public metrics archive contains a buy/sell volume ratio, not raw buy/sell
    # volumes. Preserve that distinction rather than fabricating volumes from it.
    with taker_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "buySellRatio"))
        writer.writeheader()
        for timestamp, row in ordered:
            writer.writerow({"timestamp": timestamp, "buySellRatio": row["sum_taker_long_short_vol_ratio"]})

    timestamps = [datetime.fromisoformat(timestamp) for timestamp, _ in ordered]
    gap_count = sum(
        (right - left) > timedelta(minutes=6)
        for left, right in zip(timestamps, timestamps[1:])
    )
    expected_rows = ((end - start).days + 1) * 288
    return {
        "symbol": symbol,
        "archive_days": archive_days,
        "rows": len(ordered),
        "first_timestamp": ordered[0][0] if ordered else None,
        "last_timestamp": ordered[-1][0] if ordered else None,
        "coverage_pct": len(ordered) / expected_rows * 100.0 if expected_rows else 0.0,
        "gap_count": gap_count,
        "oi_file": str(oi_path),
        "taker_ratio_file": str(taker_path),
    }


def parse_symbols(value: str) -> tuple[str, ...]:
    path = Path(value)
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        return tuple(str(item).upper() for item in config["trading"]["symbols"])
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and merge Binance Vision USD-M daily metrics")
    parser.add_argument("--symbols", required=True, help="comma-separated symbols or a config JSON path")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--raw-dir", default="data/binance_metrics_daily_raw")
    parser.add_argument("--output-dir", default="data/binance_derivatives_metrics_5m")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    symbols = parse_symbols(args.symbols)
    if not symbols:
        parser.error("no symbols selected")

    raw_root = Path(args.raw_dir)
    output_root = Path(args.output_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    days = tuple(daterange(args.start, args.end))
    tasks = [(symbol, day) for symbol in symbols for day in days]
    limiter = RateLimiter(args.requests_per_second)
    results: list[DownloadResult] = []
    started = time.monotonic()
    print(
        f"Downloading {len(tasks)} daily archives for {len(symbols)} symbols "
        f"at <= {args.requests_per_second:.2f} requests/s...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                download_archive,
                symbol,
                day,
                raw_root,
                limiter,
                args.retries,
                args.timeout_seconds,
            )
            for symbol, day in tasks
        ]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % max(1, args.progress_every) == 0 or index == len(tasks):
                counts: dict[str, int] = {}
                for item in results:
                    counts[item.status] = counts.get(item.status, 0) + 1
                elapsed = max(0.001, time.monotonic() - started)
                print(f"progress={index}/{len(tasks)} rate={index / elapsed:.2f}/s status={counts}", flush=True)

    coverage = [merge_symbol(symbol, args.start, args.end, raw_root, output_root) for symbol in symbols]
    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[item.status] = status_counts.get(item.status, 0) + 1
    manifest = {
        "source": BASE_URL,
        "source_schema": list(METRICS_COLUMNS),
        "requested_start": args.start.isoformat(),
        "requested_end": args.end.isoformat(),
        "symbols": list(symbols),
        "rate_limit_requests_per_second": args.requests_per_second,
        "zero_fill_used": False,
        "raw_taker_volumes_available": False,
        "taker_ratio_preserved_without_volume_fabrication": True,
        "status_counts": status_counts,
        "failed": [asdict(item) for item in results if item.status == "failed"],
        "missing": [asdict(item) for item in results if item.status in {"missing", "missing_cached"}],
        "coverage": coverage,
    }
    manifest_path = output_root / f"coverage_{args.start.strftime('%Y%m%d')}_{args.end.strftime('%Y%m%d')}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}", flush=True)
    print(f"status={status_counts}", flush=True)


if __name__ == "__main__":
    main()
