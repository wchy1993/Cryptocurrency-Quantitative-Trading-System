from __future__ import annotations

import pytest

from scripts.mtf_momentum_reset_full_year import _monthly_path


def test_monthly_path_carries_equity_across_month_boundary() -> None:
    path = {
        "trades": [
            {
                "entry_time": "2026-01-31T23:00:00",
                "net_pnl": 20.0,
                "equity_after": 220.0,
            },
            {
                "entry_time": "2026-02-01T01:00:00",
                "net_pnl": -10.0,
                "equity_after": 210.0,
            },
        ]
    }

    monthly = _monthly_path(path)

    assert monthly["2026-01"]["starting_equity"] == pytest.approx(200.0)
    assert monthly["2026-01"]["ending_equity"] == pytest.approx(220.0)
    assert monthly["2026-02"]["starting_equity"] == pytest.approx(220.0)
    assert monthly["2026-02"]["ending_equity"] == pytest.approx(210.0)
    assert monthly["2026-02"]["return_pct"] == pytest.approx(-10.0 / 220.0 * 100.0)
