from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from BreakoutV16GridV15PrecisionGuardLiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade,
)
from BreakoutV16Score2StructuralDecayResearchFreqtrade import (  # noqa: E402
    BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade,
    BreakoutV16GridV15PrecisionGuardScore2DecayStrictShapeResearchFreqtrade,
)
from BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade import (  # noqa: E402
    BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade,
)


def _selected() -> BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade:
    return object.__new__(
        BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade
    )


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "range_atr": 1.20,
        "lower_wick_ratio": 0.60,
        "close_position": 0.90,
        "bull_body_atr": 0.35,
        "reclaim_atr": 0.40,
        "short_momentum_4h_atr": -0.75,
        "maximum_r": 0.95,
        "current_r": -0.15,
    }
    values.update(overrides)
    return values


def test_candidate_changes_only_custom_exit() -> None:
    frozen = BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade
    candidate = BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade

    assert candidate.populate_indicators is frozen.populate_indicators
    assert candidate.populate_entry_trend is frozen.populate_entry_trend
    assert candidate.custom_stake_amount is frozen.custom_stake_amount
    assert candidate.leverage is frozen.leverage
    assert candidate.custom_stoploss is frozen.custom_stoploss
    assert candidate.adjust_trade_position is frozen.adjust_trade_position
    assert candidate.custom_exit is not frozen.custom_exit


def test_production_adapter_promotes_selected_candidate_without_overrides() -> None:
    selected = BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade
    production = (
        BreakoutV16GridV15PrecisionGuardScore2DecayLiveParityFreqtrade
    )

    assert issubclass(production, selected)
    assert production.custom_exit is selected.custom_exit
    assert production.populate_entry_trend is selected.populate_entry_trend
    assert production.custom_stake_amount is selected.custom_stake_amount
    assert production.custom_stoploss is selected.custom_stoploss
    assert production.adjust_trade_position is selected.adjust_trade_position
    assert (
        production.ADAPTIVE_STATE_BASENAME
        == BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade
        .ADAPTIVE_STATE_BASENAME
    )


def test_strict_shape_neighbour_only_tightens_rejection_shape() -> None:
    selected = BreakoutV16GridV15PrecisionGuardScore2DecayResearchFreqtrade
    strict = (
        BreakoutV16GridV15PrecisionGuardScore2DecayStrictShapeResearchFreqtrade
    )

    assert strict.V16_S2_MIN_HOLD_MINUTES == selected.V16_S2_MIN_HOLD_MINUTES
    assert strict.V16_S2_MAX_MFE_R == selected.V16_S2_MAX_MFE_R
    assert strict.V16_S2_MIN_RANGE_ATR == 0.80
    assert strict.V16_S2_MIN_LOWER_WICK_RATIO == 0.50
    assert strict.V16_S2_MIN_CLOSE_POSITION == selected.V16_S2_MIN_CLOSE_POSITION
    assert strict.V16_S2_MIN_BULL_BODY_ATR == selected.V16_S2_MIN_BULL_BODY_ATR
    assert strict.V16_S2_MIN_RECLAIM_ATR == selected.V16_S2_MIN_RECLAIM_ATR


def test_central_rule_requires_every_score2_decay_axis() -> None:
    selected = _selected()
    check = selected._v16_s2_decay_reason
    expected = selected.V16_S2_EXIT_REASON

    assert (
        check(
            score=2,
            is_short=True,
            holding_minutes=480.0,
            features=_features(),
        )
        == expected
    )
    assert check(
        score=3,
        is_short=True,
        holding_minutes=480.0,
        features=_features(),
    ) is None
    assert check(
        score=2,
        is_short=False,
        holding_minutes=480.0,
        features=_features(),
    ) is None
    assert check(
        score=2,
        is_short=True,
        holding_minutes=479.99,
        features=_features(),
    ) is None

    rejected = {
        "maximum_r": 1.25,
        "current_r": 0.01,
        "range_atr": 0.64,
        "lower_wick_ratio": 0.29,
        "close_position": 0.79,
        "bull_body_atr": 0.14,
        "reclaim_atr": 0.09,
        "short_momentum_4h_atr": 0.01,
    }
    for name, value in rejected.items():
        assert check(
            score=2,
            is_short=True,
            holding_minutes=480.0,
            features=_features(**{name: value}),
        ) is None


class _Trade:
    pair = "DASH/USDT:USDT"
    enter_tag = "bo_v9_s2_r30294_c0_l0"
    is_short = True
    open_rate = 29.78
    min_rate = 29.56
    open_date_utc = datetime(2026, 8, 18, 21, 0, tzinfo=timezone.utc)

    def get_custom_data(self, key: str, default: object = None) -> object:
        values = {
            "bo_score": 2,
            "initial_unit_risk": 0.17715040681067645,
        }
        return values.get(key, default)


def test_completed_history_ignores_unfinished_hour() -> None:
    selected = _selected()
    dates = pd.date_range("2026-08-19 03:00:00+00:00", periods=7, freq="1h")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [29.79, 29.63, 29.77, 29.73, 29.70, 29.77, 29.73],
            "high": [29.79, 29.80, 29.80, 29.74, 29.78, 29.77, 29.94],
            "low": [29.62, 29.63, 29.70, 29.68, 29.56, 29.68, 29.70],
            "close": [29.64, 29.76, 29.73, 29.69, 29.76, 29.73, 29.87],
            "atr": [0.16818494390996944] * 7,
        }
    )
    selected.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (frame, None)
    )

    # At 08:00 UTC, the 08:00 candle is still forming.  The completed
    # rejection is the 07:00 candle and must be the latest usable row.
    history = selected._v16_s2_completed_history(
        _Trade.pair,
        datetime(2026, 8, 19, 8, 0, tzinfo=timezone.utc),
        _Trade(),
    )
    assert history is not None
    _previous, latest, _four_hours_ago, _campaign = history
    assert latest["date"] == pd.Timestamp("2026-08-19 07:00:00+00:00")

    assert selected._v16_s2_completed_history(
        _Trade.pair,
        datetime(2026, 8, 19, 8, 5, 1, tzinfo=timezone.utc),
        _Trade(),
    ) is None
