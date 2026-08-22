from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
from math import inf
from pathlib import Path
import re
from typing import Any, Callable, Iterable
from zipfile import ZipFile

import pandas as pd

from freqtrade.data.metrics import (
    calculate_max_drawdown_from_balance,
    calculate_sharpe_from_balance,
    calculate_sortino_from_balance,
)


Trade = dict[str, Any]


def _strategy_names(path: Path) -> list[str]:
    with ZipFile(path) as archive:
        result_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("_config.json")
        )
        result = json.loads(archive.read(result_name))
    return list(result["strategy"])


def _load_archive(
    path: Path,
    requested_strategy: str | None = None,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    with ZipFile(path) as archive:
        result_name = next(
            name
            for name in archive.namelist()
            if name.endswith(".json") and not name.endswith("_config.json")
        )
        result = json.loads(archive.read(result_name))
        strategy_name = requested_strategy or next(iter(result["strategy"]))
        if strategy_name not in result["strategy"]:
            available = ", ".join(result["strategy"])
            raise ValueError(
                f"Strategy {strategy_name!r} is not in {path}; available: {available}"
            )
        strategy = result["strategy"][strategy_name]
        wallet_name = next(
            name
            for name in archive.namelist()
            if name.endswith(f"_{strategy_name}_wallet.feather")
        )
        wallet = pd.read_feather(BytesIO(archive.read(wallet_name)))
    return strategy_name, strategy, wallet


def _profit_factor(trades: Iterable[Trade]) -> float:
    profits = [float(trade["profit_abs"]) for trade in trades]
    gross_profit = sum(value for value in profits if value > 0.0)
    gross_loss = -sum(value for value in profits if value < 0.0)
    return inf if gross_loss == 0.0 and gross_profit > 0.0 else (
        gross_profit / gross_loss if gross_loss > 0.0 else 0.0
    )


def _trade_metrics(trades: Iterable[Trade]) -> dict[str, Any]:
    rows = list(trades)
    profits = [float(trade["profit_abs"]) for trade in rows]
    wins = sum(value > 0.0 for value in profits)
    draws = sum(value == 0.0 for value in profits)
    losses = len(rows) - wins - draws
    return {
        "trades": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "winrate_pct": 100.0 * wins / len(rows) if rows else 0.0,
        "net_profit": sum(profits),
        "profit_factor": _profit_factor(rows),
    }


def _component(trade: Trade) -> str:
    return "grid" if str(trade.get("enter_tag", "")).startswith("grid_") else "breakout"


def _side(trade: Trade) -> str:
    return "short" if bool(trade["is_short"]) else "long"


def _grid_score(trade: Trade) -> str:
    match = re.search(r"_s(?P<score>\d+)$", str(trade.get("enter_tag", "")))
    return f"s{match.group('score')}" if match else "unknown"


def _closed_year(trade: Trade) -> str:
    return str(trade["close_date"])[:4]


def _group(
    trades: Iterable[Trade],
    key: Callable[[Trade], str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Trade]] = defaultdict(list)
    for trade in trades:
        groups[key(trade)].append(trade)
    return {
        name: _trade_metrics(rows)
        for name, rows in sorted(groups.items())
    }


def _wallet_metrics(wallet: pd.DataFrame) -> dict[str, Any]:
    if wallet.empty:
        return {}
    drawdown = calculate_max_drawdown_from_balance(wallet, relative=True)
    return {
        "start_balance": float(wallet.iloc[0]["total_quote"]),
        "end_balance": float(wallet.iloc[-1]["total_quote"]),
        "return_pct": 100.0
        * (
            float(wallet.iloc[-1]["total_quote"])
            / float(wallet.iloc[0]["total_quote"])
            - 1.0
        ),
        "strict_drawdown_pct": 100.0 * float(drawdown.relative_account_drawdown),
        "strict_drawdown_abs": float(drawdown.drawdown_abs),
        "drawdown_peak": str(drawdown.high_date),
        "drawdown_trough": str(drawdown.low_date),
        "wallet_sharpe": float(calculate_sharpe_from_balance(wallet)),
        "wallet_sortino": float(calculate_sortino_from_balance(wallet)),
    }


def summarize(
    path: Path,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    strategy_name, strategy, wallet = _load_archive(path, strategy_name)
    trades: list[Trade] = strategy["trades"]
    overall = _trade_metrics(trades)
    overall.update(
        {
            "final_balance": float(strategy["final_balance"]),
            "summary_drawdown_pct": 100.0
            * float(strategy["max_drawdown_account"]),
            "sharpe": float(strategy["sharpe"]),
            "sortino": float(strategy["sortino"]),
        }
    )
    overall.update(_wallet_metrics(wallet))

    years: dict[str, dict[str, Any]] = {}
    for year in sorted({_closed_year(trade) for trade in trades}):
        year_trades = [trade for trade in trades if _closed_year(trade) == year]
        year_wallet = wallet[wallet["date"].dt.year == int(year)].copy()
        year_metrics = _trade_metrics(year_trades)
        year_metrics.update(_wallet_metrics(year_wallet))
        year_metrics["side"] = _group(year_trades, _side)
        year_metrics["component"] = _group(year_trades, _component)
        grid_trades = [trade for trade in year_trades if _component(trade) == "grid"]
        year_metrics["grid_score"] = _group(grid_trades, _grid_score)
        years[year] = year_metrics

    winning_profits = sorted(
        (float(trade["profit_abs"]) for trade in trades if trade["profit_abs"] > 0.0),
        reverse=True,
    )
    net_profit = float(overall["net_profit"])
    concentration = {
        f"top{count}_pct": (
            100.0 * sum(winning_profits[:count]) / net_profit
            if net_profit > 0.0
            else 0.0
        )
        for count in (1, 3, 5, 10)
    }

    trigger_tags = {
        "grid_v18_saturated_partial",
        "grid_v19_score5_mature_reversal_partial",
        "grid_v19_long_saturated_partial",
        "bo_v19_s3_short_failed_follow_partial",
    }
    triggers = []
    for trade in trades:
        matched = sorted(
            {
                str(order.get("ft_order_tag"))
                for order in trade.get("orders", [])
                if order.get("ft_order_tag") in trigger_tags
            }
        )
        if matched:
            triggers.append(
                {
                    "pair": trade["pair"],
                    "side": _side(trade),
                    "enter_tag": trade["enter_tag"],
                    "open_date": trade["open_date"],
                    "close_date": trade["close_date"],
                    "profit_abs": float(trade["profit_abs"]),
                    "profit_ratio": float(trade["profit_ratio"]),
                    "trigger_tags": matched,
                }
            )

    grid = [trade for trade in trades if _component(trade) == "grid"]
    return {
        "archive": str(path.resolve()),
        "strategy": strategy_name,
        "timerange": strategy["timerange"],
        "timeframe": strategy["timeframe"],
        "timeframe_detail": strategy["timeframe_detail"],
        "overall": overall,
        "side": _group(trades, _side),
        "component": _group(trades, _component),
        "component_side": _group(
            trades, lambda trade: f"{_component(trade)}_{_side(trade)}"
        ),
        "grid_score": _group(grid, _grid_score),
        "years": years,
        "profit_concentration": concentration,
        "year_profit_share_pct": {
            year: 100.0 * float(metrics["net_profit"]) / net_profit
            for year, metrics in years.items()
        },
        "dynamic_rule_triggers": triggers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize continuous-wallet exact-1m V19 backtest archives."
    )
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument(
        "--strategy",
        help="Summarize only this strategy from each archive (default: all).",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    summaries = []
    for path in args.archives:
        names = [args.strategy] if args.strategy else _strategy_names(path)
        summaries.extend(summarize(path, name) for name in names)
    print(
        json.dumps(
            summaries,
            ensure_ascii=False,
            indent=args.indent,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
