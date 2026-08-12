from __future__ import annotations

from pathlib import Path
import sys

import pytest


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

from BreakoutV16GridV15QualityPfCombinedResearchFreqtrade import (  # noqa: E402
    BreakoutV16GridV15QualityPfCombinedResearchFreqtrade,
)


def test_combined_contract_keeps_both_sleeves_and_shared_max2() -> None:
    selected = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    mro_names = {base.__name__ for base in selected.__mro__}

    assert selected.MAX_OPEN_TRADES == 2
    assert "BreakoutOnlyResearchMixin" not in mro_names
    assert "GridOnlyResearchMixin" not in mro_names
    assert "_V16IntrahourPathMixin" in mro_names
    assert "_GridV15QualityProtectedRiskMixin" in mro_names
    assert "BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade" in mro_names


def test_combined_contract_uses_selected_v16_management() -> None:
    selected = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade

    assert selected.V16_REJECT_SHORT_EXHAUSTION is False
    assert selected.V16_ENABLE_NO_FOLLOW_EXIT is True
    assert selected.V16_ENABLE_WATCH_PROFIT_FLOOR is True
    assert selected.V16_NO_FOLLOW_STAGE1_MINUTES == 15
    assert selected.V16_CONFIRM_TIMEFRAME == "15m"


def test_combined_contract_uses_exact_grid_pf_risk_surface() -> None:
    selected = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    strategy = object.__new__(selected)

    assert selected.GRID_V15_SCORE4_SCALE == pytest.approx(0.60)
    assert selected.GRID_V15_SCORE5_SCALE == pytest.approx(0.25)
    assert selected.GRID_V15_LONG_SCALE == pytest.approx(0.35)
    assert strategy._grid_v15_quality_scale(
        {
            "grid_score": 6,
            "grid_extension": 0.40,
            "grid_volume_ratio": 0.84,
        },
        "short",
    ) == pytest.approx(1.0)


def test_overlay_order_routes_grid_quality_after_v16_base_sizing() -> None:
    selected = BreakoutV16GridV15QualityPfCombinedResearchFreqtrade
    names = [base.__name__ for base in selected.__mro__]

    assert names.index("_GridV15QualityProtectedRiskMixin") < names.index(
        "_V16IntrahourPathMixin"
    )
    assert names.index("_V16IntrahourPathMixin") < names.index(
        "BreakoutV15Fixed50ShortImpulseStableSelectedFreqtrade"
    )
