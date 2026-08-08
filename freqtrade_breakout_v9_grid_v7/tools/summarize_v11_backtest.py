#!/usr/bin/env python3
"""Print comparable metrics and a deterministic trade-row hash.

This helper is read-only.  It accepts one or more Freqtrade result archives
and emits one JSON object per archive so research and frozen baselines can be
checked with the same metric definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


def load_result(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as package:
        result_name = next(
            name
            for name in package.namelist()
            if name.endswith(".json")
            and not name.endswith("_config.json")
        )
        payload = json.loads(package.read(result_name))
    return next(iter(payload["strategy"].values()))


def trade_hash(trades: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        trades,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def entry_sequence_hash(trades: list[dict[str, Any]]) -> str:
    """Hash only causal entry identity, independent of stake and PnL."""
    entries = [
        {
            "pair": trade["pair"],
            "open_date": trade["open_date"],
            "enter_tag": trade.get("enter_tag"),
            "is_short": trade["is_short"],
        }
        for trade in trades
    ]
    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def component_summary(
    trades: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    selected = [
        trade
        for trade in trades
        if str(trade.get("enter_tag") or "").startswith(prefix)
    ]
    profits = [float(trade["profit_abs"]) for trade in selected]
    gross_profit = sum(value for value in profits if value > 0.0)
    gross_loss = -sum(value for value in profits if value < 0.0)
    return {
        "trades": len(selected),
        "net_profit_usdt": sum(profits),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0.0 else 0.0
        ),
        "win_rate_pct": (
            100.0 * sum(value > 0.0 for value in profits) / len(selected)
            if selected
            else 0.0
        ),
        "mean_profit_ratio_pct": (
            100.0
            * sum(float(trade["profit_ratio"]) for trade in selected)
            / len(selected)
            if selected
            else 0.0
        ),
    }


def summarize(path: Path) -> dict[str, Any]:
    result = load_result(path)
    wallet = result["wallet_stats"]
    trades = result["trades"]
    return {
        "archive": str(path),
        "strategy": result["strategy_name"],
        "start": result["backtest_start"],
        "end": result["backtest_end"],
        "trades": result["total_trades"],
        "net_profit_usdt": result["profit_total_abs"],
        "profit_factor": result["profit_factor"],
        "win_rate_pct": result["winrate"] * 100.0,
        "wallet_max_drawdown_pct": (
            wallet["max_relative_drawdown"] * 100.0
        ),
        "wallet_low_usdt": wallet["low_balance"],
        "liquidations": sum(
            trade.get("exit_reason") == "liquidation" for trade in trades
        ),
        "components": {
            "breakout": component_summary(trades, "bo_"),
            "grid": component_summary(trades, "grid_"),
        },
        "entry_sequence_sha256": entry_sequence_hash(trades),
        "trade_rows_sha256": trade_hash(trades),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        print(json.dumps(summarize(archive), ensure_ascii=False))


if __name__ == "__main__":
    main()
