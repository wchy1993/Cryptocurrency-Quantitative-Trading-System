#!/usr/bin/env python3
"""Join a V15 exact backtest to its causal completed-candle entry context."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from pandas import DataFrame


PROJECT_DIR = Path(__file__).resolve().parents[1]
STRATEGY_DIR = PROJECT_DIR / "user_data" / "strategies"
if str(PROJECT_DIR / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "tools"))
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

import v11_entry_feature_audit as frozen_audit  # noqa: E402


FEATURE_COLUMNS = tuple(
    dict.fromkeys(
        (
            *frozen_audit.FEATURE_COLUMNS,
            "market_regime_score",
            "market_efficiency",
            "market_dispersion_7d",
            "low_opportunity_fraction_72h",
            "market_median_return_24h",
            "breadth_change_24h",
            "breadth_travel_24h",
            "grid_quality",
            "bo_range_atr",
            "btc_trend_4h",
            "btc_trend_1d",
        )
    )
)


def _strategy_class(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    strategy_class = getattr(module, class_name, None)
    if strategy_class is None:
        raise ValueError(
            f"module {module_name!r} has no strategy class {class_name!r}"
        )
    return strategy_class


def _archive_context(path: Path) -> tuple[str, dict, dict]:
    payload = frozen_audit._archive_payload(path)
    strategy_name, result = next(iter(payload["strategy"].items()))
    with zipfile.ZipFile(path) as archive:
        config_name = next(
            name for name in archive.namelist() if name.endswith("_config.json")
        )
        config = json.loads(archive.read(config_name))
    return strategy_name, result, config


def _analyze_frames(
    strategy: object,
    frames: dict[str, DataFrame],
    pairs: tuple[str, ...],
) -> dict[str, DataFrame]:
    strategy.dp = frozen_audit.FrozenDataProvider(frames, pairs)
    strategy.bot_start()
    analyzed: dict[str, DataFrame] = {}
    for pair in pairs:
        frame = strategy.populate_indicators(frames[pair], {"pair": pair})
        frame = strategy.populate_entry_trend(frame, {"pair": pair})
        atr = frame["atr"].clip(lower=1e-12)
        frame["session_return"] = frame["close"] / frame["session_open"] - 1.0
        frame["return_8h"] = frame["close"] / frame["close"].shift(8) - 1.0
        frame["return_12h"] = frame["close"] / frame["close"].shift(12) - 1.0
        frame["return_24h"] = frame["close"] / frame["close"].shift(24) - 1.0
        frame["high_24h"] = frame["high"].rolling(24, min_periods=1).max()
        frame["low_24h"] = frame["low"].rolling(24, min_periods=1).min()
        frame["from_high_24h"] = frame["close"] / frame["high_24h"] - 1.0
        frame["from_low_24h"] = frame["close"] / frame["low_24h"] - 1.0
        frame["atr_pct"] = atr / frame["close"].clip(lower=1e-12)
        frame["below_fast_age"] = frozen_audit._consecutive(
            frame["close"] < frame["fast_ema"]
        )
        frame["above_fast_age"] = frozen_audit._consecutive(
            frame["close"] > frame["fast_ema"]
        )
        frame["ema_spread_atr"] = (
            frame["fast_ema"] - frame["slow_ema"]
        ) / atr
        analyzed[pair] = frame.set_index("date", drop=False)
    return analyzed


def build_audit(
    archive_path: Path,
    module_name: str,
    datadir: Path,
) -> DataFrame:
    strategy_name, result, config = _archive_context(archive_path)
    pairs = tuple(result["pairlist"])
    end = pd.Timestamp(result["backtest_end"])
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    frames = frozen_audit._load_frames(pairs, datadir, end)
    strategy = _strategy_class(module_name, strategy_name)(config)
    analyzed = _analyze_frames(strategy, frames, pairs)

    rows: list[dict] = []
    for trade in result["trades"]:
        pair = str(trade["pair"])
        open_date = pd.Timestamp(trade["open_date"])
        signal_date = open_date - pd.Timedelta(hours=1)
        signal = analyzed[pair].loc[signal_date]
        if isinstance(signal, DataFrame):
            signal = signal.iloc[-1]
        is_short = bool(trade["is_short"])
        side = -1.0 if is_short else 1.0
        row = {
            "strategy": strategy_name,
            "pair": pair,
            "open_date": open_date,
            "signal_date": signal_date,
            "close_date": pd.Timestamp(trade["close_date"]),
            "component": (
                "breakout"
                if str(trade["enter_tag"]).startswith("bo_")
                else "grid"
            ),
            "side": "short" if is_short else "long",
            "enter_tag": str(trade["enter_tag"]),
            "profit_abs": float(trade["profit_abs"]),
            "profit_ratio": float(trade["profit_ratio"]),
            "stake_amount": float(trade["stake_amount"]),
            "leverage": float(trade["leverage"]),
            "trade_duration": int(trade["trade_duration"]),
            "exit_reason": str(trade["exit_reason"]),
        }
        for column in FEATURE_COLUMNS:
            if column == "date":
                continue
            value = signal.get(column, np.nan)
            row[column] = value.item() if hasattr(value, "item") else value

        row["directional_return_4h"] = side * float(row["symbol_return_4h"])
        row["directional_return_8h"] = side * float(row["return_8h"])
        row["directional_return_12h"] = side * float(row["return_12h"])
        row["directional_return_24h"] = side * float(row["return_24h"])
        row["directional_ema55_atr"] = side * float(row["symbol_ema55_atr"])
        row["directional_breadth"] = (
            1.0 - float(row["breadth"])
            if is_short
            else float(row["breadth"])
        )
        rows.append(row)
    return DataFrame(rows).sort_values("open_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--datadir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = build_audit(args.archive, args.module, args.datadir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(f"{args.output} rows={len(audit)}")


if __name__ == "__main__":
    main()
