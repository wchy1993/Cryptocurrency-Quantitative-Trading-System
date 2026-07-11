from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data import parse_timestamp
from .live_config import load_live_config
from .risk import FundingRate


FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/daily/aggTrades"


def download_funding_history(
    symbols: Iterable[str],
    start: datetime,
    end: datetime,
    output_dir: str | Path,
    sleep_seconds: float = 1.0,
    timeout_seconds: int = 30,
) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for symbol in symbols:
        path = output / f"{symbol.upper()}_funding.csv"
        existing = _read_funding_csv(path)
        cursor = max(start, existing[-1].timestamp + timedelta(milliseconds=1)) if existing else start
        rows = list(existing)
        while cursor < end:
            params = {"symbol": symbol.upper(), "startTime": _to_ms(cursor), "endTime": _to_ms(end), "limit": 1000}
            payload = _request_json(f"{FUNDING_URL}?{urlencode(params)}", timeout_seconds, sleep_seconds)
            if not payload:
                break
            for item in payload:
                timestamp = datetime.fromtimestamp(int(item["fundingTime"]) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
                if start <= timestamp <= end:
                    rows.append(FundingRate(timestamp, float(item["fundingRate"])))
            next_cursor = datetime.fromtimestamp(int(payload[-1]["fundingTime"]) / 1000.0, tz=timezone.utc).replace(tzinfo=None) + timedelta(milliseconds=1)
            if next_cursor <= cursor or len(payload) < 1000:
                break
            cursor = next_cursor
            time.sleep(max(0.0, sleep_seconds))
        unique = {item.timestamp: item for item in rows}
        _write_funding_csv(path, [unique[key] for key in sorted(unique)])
        written += 1
        print(f"funding {symbol}: {len(unique)} rows -> {path}", flush=True)
        time.sleep(max(0.0, sleep_seconds))
    return written


def download_aggtrade_archives(
    symbols: Iterable[str],
    start: date,
    end: date,
    output_dir: str | Path,
    sleep_seconds: float = 2.0,
    timeout_seconds: int = 120,
    max_files: int = 0,
) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for day in _dates(start, end):
        for symbol in symbols:
            symbol = symbol.upper()
            target_dir = output / symbol
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{symbol}-aggTrades-{day.isoformat()}.zip"
            target = target_dir / filename
            missing = target.with_suffix(".missing")
            if target.exists() or missing.exists():
                continue
            url = f"{ARCHIVE_ROOT}/{symbol}/{filename}"
            try:
                _download_resumable(url, target, timeout_seconds, sleep_seconds)
            except HTTPError as exc:
                if exc.code == 404:
                    missing.write_text("not listed or no archive\n", encoding="ascii")
                    print(f"missing {symbol} {day}", flush=True)
                else:
                    raise
            else:
                downloaded += 1
                print(f"aggTrades {symbol} {day} -> {target}", flush=True)
            time.sleep(max(0.0, sleep_seconds))
            if max_files > 0 and downloaded >= max_files:
                return downloaded
    return downloaded


def load_funding_rate_directory(path: str | Path, symbols: Iterable[str]) -> dict[str, tuple[FundingRate, ...]]:
    root = Path(path)
    output: dict[str, tuple[FundingRate, ...]] = {}
    if not root.exists():
        return output
    for symbol in symbols:
        rows = _read_funding_csv(root / f"{symbol.upper()}_funding.csv")
        if rows:
            output[symbol.upper()] = tuple(rows)
    return output


def _request_json(url: str, timeout_seconds: int, base_sleep: float) -> Any:
    for attempt in range(7):
        request = Request(url, headers={"User-Agent": "crypto-scalper/0.2"}, method="GET")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == 6:
                raise
            retry_after = float(exc.headers.get("Retry-After", 0) or 0)
            time.sleep(max(retry_after, base_sleep * (2 ** attempt), 1.0))
        except URLError:
            if attempt == 6:
                raise
            time.sleep(max(base_sleep * (2 ** attempt), 1.0))
    raise RuntimeError("request retry loop exhausted")


def _download_resumable(url: str, target: Path, timeout_seconds: int, base_sleep: float) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(7):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "crypto-scalper/0.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                append = offset > 0 and getattr(response, "status", 200) == 206
                with partial.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            with zipfile.ZipFile(partial) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"corrupt member {bad}")
            partial.replace(target)
            return
        except HTTPError as exc:
            if exc.code == 404:
                raise
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == 6:
                raise
            retry_after = float(exc.headers.get("Retry-After", 0) or 0)
            time.sleep(max(retry_after, base_sleep * (2 ** attempt), 1.0))
        except (URLError, OSError, zipfile.BadZipFile, ValueError):
            if attempt == 6:
                raise
            time.sleep(max(base_sleep * (2 ** attempt), 1.0))


def _read_funding_csv(path: Path) -> list[FundingRate]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [FundingRate(parse_timestamp(row["timestamp"]), float(row["rate"])) for row in csv.DictReader(handle)]


def _write_funding_csv(path: Path, rows: Iterable[FundingRate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("timestamp", "rate"))
        writer.writeheader()
        for row in rows:
            writer.writerow({"timestamp": row.timestamp.isoformat(), "rate": f"{row.rate:.12g}"})


def _dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)


def _to_ms(value: datetime) -> int:
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return int(aware.timestamp() * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rate-limited realistic Binance futures data downloader")
    parser.add_argument("kind", choices=("funding", "aggtrades", "all"))
    parser.add_argument("--config", default="config.live_safe.json")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--funding-dir", default="data/binance_funding_90d")
    parser.add_argument("--aggtrades-dir", default="data/binance_aggtrades_90d")
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--max-files", type=int, default=0, help="Stop after this many new aggTrade files; zero means unlimited")
    args = parser.parse_args()
    config = load_live_config(args.config)
    symbols = tuple(config.trading.symbols)
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if args.kind in {"funding", "all"}:
        download_funding_history(symbols, start, end, args.funding_dir, args.sleep)
    if args.kind in {"aggtrades", "all"}:
        download_aggtrade_archives(symbols, start.date(), end.date(), args.aggtrades_dir, args.sleep, max_files=args.max_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
