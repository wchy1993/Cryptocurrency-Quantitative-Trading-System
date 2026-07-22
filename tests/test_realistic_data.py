from __future__ import annotations

from pathlib import Path
from datetime import datetime

from crypto_scalper.data import load_candles_csv
from crypto_scalper.realistic_data import load_funding_rate_directory


def test_load_funding_directory_accepts_dated_archive_schema(tmp_path: Path) -> None:
    (tmp_path / "BTCUSDT_funding_20250101_20250102.csv").write_text(
        "timestamp,funding_rate,mark_price\n"
        "2025-01-01T00:00:00,0.0001,100000\n",
        encoding="utf-8",
    )
    (tmp_path / "BTCUSDT_funding.csv").write_text(
        "timestamp,rate\n"
        "2025-01-01T08:00:00,-0.0002\n",
        encoding="utf-8",
    )

    result = load_funding_rate_directory(tmp_path, ("BTCUSDT", "ETHUSDT"))

    assert tuple(result) == ("BTCUSDT",)
    assert [row.rate for row in result["BTCUSDT"]] == [0.0001, -0.0002]


def test_candle_csv_window_is_inclusive_and_does_not_change_values(tmp_path: Path) -> None:
    path = tmp_path / "BTCUSDT_1m_test.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T00:00:00,100,101,99,100.5,10\n"
        "2026-01-01T00:01:00,101,102,100,101.5,11\n"
        "2026-01-01T00:02:00,102,103,101,102.5,12\n",
        encoding="utf-8",
    )

    candles = load_candles_csv(
        path,
        start=datetime(2026, 1, 1, 0, 1),
        end=datetime(2026, 1, 1, 0, 2),
    )

    assert [candle.timestamp.minute for candle in candles] == [1, 2]
    assert [candle.close for candle in candles] == [101.5, 102.5]
