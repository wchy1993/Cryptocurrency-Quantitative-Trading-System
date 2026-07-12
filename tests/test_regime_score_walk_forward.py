from __future__ import annotations

import runpy
from datetime import datetime, timedelta
from pathlib import Path


BUILD = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "regime_score_walk_forward.py"))["build"]


def _row(timestamp: datetime, pnl: float) -> dict[str, object]:
    return {
        "event_id": f"event-{timestamp.isoformat()}-{pnl}",
        "strategy": "volume_breakout_pullback",
        "status": "traded",
        "timestamp": timestamp.isoformat(),
        "entry_time": timestamp.isoformat(),
        "symbol": "BTCUSDT",
        "full_cost_net_pnl": pnl,
        "trend_score": 85.0,
        "reversal_score": 30.0,
        "score_gap": 55.0,
    }


def test_validation_selects_rule_but_unseen_test_can_reject_it() -> None:
    rows = []
    validation_start = datetime(2025, 10, 1)
    test_start = datetime(2026, 1, 1)
    for index in range(30):
        rows.append(_row(validation_start + timedelta(days=index * 3), 0.4 if index % 5 else -0.3))
    for index in range(20):
        rows.append(_row(test_start + timedelta(days=index * 2), 0.2 if index % 3 else -0.8))

    report = BUILD({"alpha_candidate_diagnostics": rows})

    assert report["vbp"]["selected_rule"] is not None
    assert report["vbp"]["status"] == "failed_unseen_test"
    assert report["vbp"]["test_passed"] is False
    assert report["reversal"]["status"] == "no_proven_edge"
