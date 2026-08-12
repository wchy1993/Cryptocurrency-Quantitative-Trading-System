from __future__ import annotations

from pathlib import Path
import sys

from pandas import DataFrame, isna
import pytest
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
GRID_STRATEGY_DIR = (
    PROJECT_DIR.parent
    / "freqtrade_grid_v8_dual_side"
    / "user_data"
    / "strategies"
)
for path in (STRATEGY_DIR, GRID_STRATEGY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GridV15Fixed50QualityProtectedResearchFreqtrade import (  # noqa: E402
    GridV15Fixed50QualityBalancedResearchFreqtrade,
    GridV15Fixed50QualityConviction115ResearchFreqtrade,
    GridV15Fixed50QualityConviction125ResearchFreqtrade,
    GridV15Fixed50QualityPfResearchFreqtrade,
    GridV15Fixed50QualityPfConviction105ResearchFreqtrade,
    GridV15Fixed50QualityPfConviction110ResearchFreqtrade,
    GridV15Fixed50QualityPfShortFocusResearchFreqtrade,
    GridV15Fixed50QualityProfitLooseResearchFreqtrade,
    GridV15Fixed50QualityProfitResearchFreqtrade,
)


def _scale(strategy_class: type, **row: float) -> float:
    strategy = object.__new__(strategy_class)
    return strategy._grid_v15_quality_scale(row, "short")


def test_score_neighborhood_is_ordered() -> None:
    central = GridV15Fixed50QualityBalancedResearchFreqtrade
    pf = GridV15Fixed50QualityPfResearchFreqtrade
    profit = GridV15Fixed50QualityProfitResearchFreqtrade
    assert pf.GRID_V15_SCORE4_SCALE < central.GRID_V15_SCORE4_SCALE
    assert profit.GRID_V15_SCORE4_SCALE > central.GRID_V15_SCORE4_SCALE
    assert pf.GRID_V15_SCORE5_SCALE == pytest.approx(0.25)
    assert profit.GRID_V15_SCORE5_SCALE == pytest.approx(0.25)


def test_high_conviction_short_keeps_full_risk() -> None:
    assert _scale(
        GridV15Fixed50QualityBalancedResearchFreqtrade,
        grid_score=6,
        grid_extension=0.40,
        grid_volume_ratio=0.84,
    ) == pytest.approx(1.0)


def test_extension_and_participation_axes_are_smooth_and_bounded() -> None:
    strategy = GridV15Fixed50QualityProfitResearchFreqtrade
    full = _scale(
        strategy,
        grid_score=6,
        grid_extension=0.42,
        grid_volume_ratio=0.82,
    )
    middle = _scale(
        strategy,
        grid_score=6,
        grid_extension=0.47,
        grid_volume_ratio=0.78,
    )
    defensive = _scale(
        strategy,
        grid_score=6,
        grid_extension=0.52,
        grid_volume_ratio=0.74,
    )
    assert full == pytest.approx(1.0)
    assert defensive == pytest.approx(0.20)
    assert defensive < middle < full


def test_long_and_loose_neighbor_contracts() -> None:
    central = object.__new__(GridV15Fixed50QualityBalancedResearchFreqtrade)
    loose = GridV15Fixed50QualityProfitLooseResearchFreqtrade
    assert central._grid_v15_quality_scale({}, "long") == pytest.approx(0.50)
    assert loose.GRID_V15_EXTENSION_MIN_SCALE > (
        GridV15Fixed50QualityProfitResearchFreqtrade.GRID_V15_EXTENSION_MIN_SCALE
    )
    assert loose.GRID_V15_VOLUME_MIN_SCALE > (
        GridV15Fixed50QualityProfitResearchFreqtrade.GRID_V15_VOLUME_MIN_SCALE
    )


def test_candidate_remains_grid_only_and_single_sleeve() -> None:
    candidate = GridV15Fixed50QualityProfitResearchFreqtrade
    assert candidate.MAX_OPEN_TRADES == 1
    assert "GridV15Fixed50OnlyResearchFreqtrade" in {
        base.__name__ for base in candidate.__mro__
    }


@pytest.mark.parametrize(
    ("strategy", "expected"),
    (
        (GridV15Fixed50QualityPfConviction105ResearchFreqtrade, 1.05),
        (GridV15Fixed50QualityPfConviction110ResearchFreqtrade, 1.10),
        (GridV15Fixed50QualityConviction115ResearchFreqtrade, 1.15),
        (GridV15Fixed50QualityConviction125ResearchFreqtrade, 1.25),
    ),
)
def test_score_six_reallocation_is_narrow_and_explicit(
    strategy: type,
    expected: float,
) -> None:
    assert _scale(
        strategy,
        grid_score=6,
        grid_extension=0.40,
        grid_volume_ratio=0.84,
    ) == pytest.approx(expected)
    assert strategy.GRID_V15_MAX_SCALE == pytest.approx(expected)


def test_short_focus_removes_long_signal_and_rank_only() -> None:
    strategy = object.__new__(
        GridV15Fixed50QualityPfShortFocusResearchFreqtrade
    )
    strategy.timeframe = "1h"
    strategy._candidate_ranks = {
        "BTC/USDT:USDT": {
            ("grid", 3_600): 0.20,
            ("grid", 7_200): 0.30,
        }
    }
    source = DataFrame(
        {
            "date": ["1970-01-01 00:00:00+00:00", "1970-01-01 01:00:00+00:00"],
            "enter_long": [1, 0],
            "enter_short": [0, 1],
            "enter_tag": ["grid_v8_long_s6", "grid_v8_short_s6"],
        }
    )
    parent = GridV15Fixed50QualityPfResearchFreqtrade
    with patch.object(parent, "populate_entry_trend", return_value=source.copy()):
        result = strategy.populate_entry_trend(
            source.copy(),
            {"pair": "BTC/USDT:USDT"},
        )
    assert int(result.iloc[0]["enter_long"]) == 0
    assert isna(result.iloc[0]["enter_tag"])
    assert int(result.iloc[1]["enter_short"]) == 1
    assert result.iloc[1]["enter_tag"] == "grid_v8_short_s6"
    assert strategy._candidate_ranks["BTC/USDT:USDT"] == {
        ("grid", 7_200): 0.30
    }
