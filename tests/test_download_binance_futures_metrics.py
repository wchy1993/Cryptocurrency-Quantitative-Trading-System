from __future__ import annotations

import csv
import zipfile
from datetime import date
from pathlib import Path

from scripts.download_binance_futures_metrics import METRICS_COLUMNS
from scripts.download_binance_futures_metrics import archive_name
from scripts.download_binance_futures_metrics import merge_symbol


def test_merge_preserves_real_oi_and_taker_ratio_without_fabricating_volumes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    symbol_root = raw_root / "BTCUSDT"
    symbol_root.mkdir(parents=True)
    archive_path = symbol_root / archive_name("BTCUSDT", date(2025, 6, 12))
    payload = (
        ",".join(METRICS_COLUMNS)
        + "\n"
        + "2025-06-12 00:05:00,BTCUSDT,10,1000,1,1,1,1.25\n"
        + "2025-06-12 00:10:00,BTCUSDT,11,1100,1,1,1,0.80\n"
    )
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-metrics-2025-06-12.csv", payload)

    coverage = merge_symbol(
        "BTCUSDT",
        date(2025, 6, 12),
        date(2025, 6, 12),
        raw_root,
        tmp_path / "merged",
    )

    oi_rows = list(csv.DictReader(Path(str(coverage["oi_file"])).open(encoding="utf-8")))
    taker_rows = list(csv.DictReader(Path(str(coverage["taker_ratio_file"])).open(encoding="utf-8")))
    assert oi_rows[0] == {
        "timestamp": "2025-06-12T00:05:00",
        "sumOpenInterest": "10",
        "sumOpenInterestValue": "1000",
    }
    assert taker_rows[0] == {"timestamp": "2025-06-12T00:05:00", "buySellRatio": "1.25"}
    assert "buyVol" not in taker_rows[0]
    assert "sellVol" not in taker_rows[0]
    assert coverage["rows"] == 2
    assert coverage["gap_count"] == 0
