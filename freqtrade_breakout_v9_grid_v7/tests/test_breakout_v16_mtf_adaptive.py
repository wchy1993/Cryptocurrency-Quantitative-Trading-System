from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV16Fixed50MtfAdaptiveBreakoutMax2ResearchFreqtrade import (  # noqa: E402
    BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade,
    BreakoutV16Fixed50MtfFailureProtectedFast10BreakoutMax2ResearchFreqtrade,
    BreakoutV16Fixed50MtfObserveBreakoutMax2ResearchFreqtrade,
    _V16IntrahourPathMixin,
)


V15_FROZEN_HASHES = {
    "BreakoutV15Fixed50BreakoutMax2LiveParityFreqtrade.py": (
        "ecdd0550823527a4a01283c780cd4cd6e5482dca967039f1a21b3f1a78fef2c4"
    ),
    "BreakoutV15Fixed50BreakoutMax2ResearchFreqtrade.py": (
        "f9e717b954e6ce0733811cb9434065b1d14954ccfee595162a3fe885ced98422"
    ),
}


def _quarters(start: str, count: int = 4) -> pd.DataFrame:
    dates = pd.date_range(start, periods=count, freq="15min", tz="UTC")
    rows = []
    for index, date in enumerate(dates):
        open_rate = 100.0 - index
        close_rate = open_rate - 1.0
        rows.append(
            {
                "date": date,
                "open": open_rate,
                "high": open_rate + 1.0,
                "low": close_rate - 1.0,
                "close": close_rate,
                "volume": 10.0 * (index + 1),
            }
        )
    return pd.DataFrame(rows)


def test_v15_baseline_sources_remain_byte_for_byte_frozen() -> None:
    for filename, expected in V15_FROZEN_HASHES.items():
        digest = hashlib.sha256((STRATEGY_DIR / filename).read_bytes()).hexdigest()
        assert digest == expected


def test_quarter_frame_uses_only_four_complete_candles_in_each_hour() -> None:
    complete = _quarters("2026-08-10 16:00:00", 4)
    incomplete_future_hour = _quarters("2026-08-10 17:00:00", 2)
    result = _V16IntrahourPathMixin._v16_quarter_frame(
        pd.concat([complete, incomplete_future_hour], ignore_index=True)
    )

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2026-08-10 16:00:00+00:00")
    assert result.iloc[0]["v16_mtf_ready"] == 1
    assert result.iloc[0]["v16_short_directional_quarters"] == 4
    assert result.iloc[0]["v16_short_hour_directional_return"] > 0.0


def test_future_hour_cannot_change_previous_hour_features() -> None:
    first_hour = _quarters("2026-08-10 16:00:00", 4)
    before = _V16IntrahourPathMixin._v16_quarter_frame(first_hour).iloc[0]
    extreme_future = _quarters("2026-08-10 17:00:00", 4)
    extreme_future.loc[:, ["open", "high", "low", "close"]] *= 10.0
    after = _V16IntrahourPathMixin._v16_quarter_frame(
        pd.concat([first_hour, extreme_future], ignore_index=True)
    ).iloc[0]

    pd.testing.assert_series_equal(before, after, check_names=False)


def test_observe_mode_marks_watch_without_changing_v15_entry() -> None:
    frame = pd.DataFrame(
        {
            "bo_entry_short": [1],
            "bo_entry_long": [0],
            "bo_entry": [1],
            "v16_mtf_ready": [1],
            "v16_short_last15_rejection": [0.30],
            "v16_short_last15_share": [0.10],
            "v16_short_close_from_extreme": [0.10],
            "v16_short_hour_directional_return": [0.009],
            "v16_short_directional_quarters": [4],
        }
    )
    selected = BreakoutV16Fixed50MtfObserveBreakoutMax2ResearchFreqtrade
    marked = selected._v16_mark_path_states(selected, frame)

    assert marked.iloc[0]["v16_no_follow_watch"] == 1
    assert marked.iloc[0]["v16_short_exhaustion"] == 0
    assert marked.iloc[0]["bo_entry_short"] == 1
    assert marked.iloc[0]["bo_entry"] == 1


def test_no_follow_is_time_bounded_and_requires_no_favorable_progress() -> None:
    selected = BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade

    assert selected._v16_no_follow_reason(selected, 14, -0.46, 0.0) is None
    assert (
        selected._v16_no_follow_reason(selected, 15, -0.46, 0.0)
        == "bo_v16_no_follow_15m"
    )
    assert selected._v16_no_follow_reason(selected, 15, -0.46, 0.01) is None
    assert (
        selected._v16_no_follow_reason(selected, 30, -0.61, 0.05)
        == "bo_v16_no_follow_30m"
    )
    assert selected._v16_no_follow_reason(selected, 61, -1.0, -0.5) is None


def test_fast10_neighbor_changes_only_first_confirmation_time() -> None:
    base = BreakoutV16Fixed50MtfFailureProtectedBreakoutMax2ResearchFreqtrade
    fast = BreakoutV16Fixed50MtfFailureProtectedFast10BreakoutMax2ResearchFreqtrade

    assert base.V16_NO_FOLLOW_STAGE1_MINUTES == 15
    assert fast.V16_NO_FOLLOW_STAGE1_MINUTES == 10
    assert fast.V16_NO_FOLLOW_STAGE1_CURRENT_R == base.V16_NO_FOLLOW_STAGE1_CURRENT_R
    assert fast.V16_NO_FOLLOW_STAGE2_CURRENT_R == base.V16_NO_FOLLOW_STAGE2_CURRENT_R
