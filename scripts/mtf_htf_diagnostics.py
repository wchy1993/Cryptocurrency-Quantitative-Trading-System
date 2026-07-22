from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


MTF_REASON_TOKEN = "mtf_4h_rsi_regime_pullback"


def _timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _tags(trade: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for token in str(trade.get("entry_reason", "")).split():
        if "=" in token:
            key, value = token.split("=", 1)
            output[key] = value
    return output


def _initial_risk(trade: dict[str, Any]) -> float:
    return (
        abs(float(trade.get("entry_price", 0.0)) - float(trade.get("initial_stop_price", 0.0)))
        * abs(float(trade.get("quantity", trade.get("qty", 0.0))))
    )


def _trade_values(trade: dict[str, Any]) -> tuple[float, float, float]:
    risk = _initial_risk(trade)
    net_pnl = float(trade.get("net_pnl", 0.0))
    cost = float(trade.get("fee", 0.0)) + float(trade.get("slippage_cost", 0.0))
    if risk <= 0:
        return net_pnl, 0.0, 0.0
    return net_pnl, net_pnl / risk, cost / risk


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    net_values: list[float] = []
    net_r_values: list[float] = []
    cost_r_values: list[float] = []
    gross_r_values: list[float] = []
    for trade in trades:
        net, net_r, cost_r = _trade_values(trade)
        net_values.append(net)
        net_r_values.append(net_r)
        cost_r_values.append(cost_r)
        risk = _initial_risk(trade)
        gross_r_values.append(float(trade.get("gross_pnl", 0.0)) / risk if risk > 0 else 0.0)
    wins = [value for value in net_r_values if value > 0]
    losses = [value for value in net_r_values if value <= 0]
    gross_profit_r = sum(wins)
    gross_loss_r = abs(sum(losses))
    return {
        "trades": len(trades),
        "net_pnl": sum(net_values),
        "net_r": sum(net_r_values),
        "expectancy_r": sum(net_r_values) / len(net_r_values) if net_r_values else 0.0,
        "profit_factor_r": gross_profit_r / gross_loss_r if gross_loss_r else (float("inf") if gross_profit_r else 0.0),
        "win_rate_pct": len(wins) / len(trades) * 100.0 if trades else 0.0,
        "average_cost_r": sum(cost_r_values) / len(cost_r_values) if cost_r_values else 0.0,
        "gross_expectancy_r": sum(gross_r_values) / len(gross_r_values) if gross_r_values else 0.0,
    }


def _group(trades: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        grouped[key_fn(trade)].append(trade)
    return {key: _metrics(rows) for key, rows in sorted(grouped.items())}


def _stop_bucket(trade: dict[str, Any]) -> str:
    entry = max(abs(float(trade.get("entry_price", 0.0))), 1e-12)
    stop_bps = abs(float(trade.get("entry_price", 0.0)) - float(trade.get("initial_stop_price", 0.0))) / entry * 10_000.0
    if stop_bps < 50:
        return "lt_50bps"
    if stop_bps < 75:
        return "50_75bps"
    if stop_bps < 100:
        return "75_100bps"
    if stop_bps < 150:
        return "100_150bps"
    return "gte_150bps"


def _cost_space_bucket(trade: dict[str, Any]) -> str:
    risk = _initial_risk(trade)
    cost = float(trade.get("fee", 0.0)) + float(trade.get("slippage_cost", 0.0))
    ratio = 2.0 * risk / max(cost, 1e-12)
    if ratio < 4:
        return "target_cost_lt_4x"
    if ratio < 6:
        return "target_cost_4_6x"
    if ratio < 8:
        return "target_cost_6_8x"
    if ratio < 12:
        return "target_cost_8_12x"
    return "target_cost_gte_12x"


def _split(trades: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    return [trade for trade in trades if start <= _timestamp(trade.get("entry_time")) < end]


def _split_report(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _metrics(trades),
        "by_side": _group(trades, lambda trade: str(trade.get("side", "unknown"))),
        "by_trigger": _group(trades, lambda trade: _tags(trade).get("trigger", "unknown")),
        "by_rsi_bucket": _group(trades, lambda trade: _tags(trade).get("rsi_bucket", "unknown")),
        "by_volume_bucket": _group(trades, lambda trade: _tags(trade).get("vol", "unknown")),
        "by_funding_bucket": _group(trades, lambda trade: _tags(trade).get("funding", "unknown")),
        "by_btc_state": _group(trades, lambda trade: _tags(trade).get("btc", "unknown")),
        "by_stop_bucket": _group(trades, _stop_bucket),
        "by_target_cost_bucket": _group(trades, _cost_space_bucket),
        "by_exit_reason": _group(trades, lambda trade: str(trade.get("exit_reason", "unknown")).split()[0]),
    }


def _candidate_filter_report(trades: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    filters: dict[str, Callable[[dict[str, Any]], bool]] = {
        "structure_only": lambda trade: _tags(trade).get("trigger") == "structure_break",
        "non_negative_funding": lambda trade: _tags(trade).get("funding") != "funding_neg",
        "volume_1_to_2x": lambda trade: _tags(trade).get("vol") in {"vol_1_1p5x", "vol_1p5_2x"},
        "volume_1p5_to_2x": lambda trade: _tags(trade).get("vol") == "vol_1p5_2x",
        "stop_below_100bps": lambda trade: _stop_bucket(trade) in {"lt_50bps", "50_75bps", "75_100bps"},
        "target_cost_6_to_8x": lambda trade: _cost_space_bucket(trade) == "target_cost_6_8x",
        "structure_non_negative_funding": lambda trade: (
            _tags(trade).get("trigger") == "structure_break"
            and _tags(trade).get("funding") != "funding_neg"
        ),
        "structure_funding_volume_1_to_2x": lambda trade: (
            _tags(trade).get("trigger") == "structure_break"
            and _tags(trade).get("funding") != "funding_neg"
            and _tags(trade).get("vol") in {"vol_1_1p5x", "vol_1p5_2x"}
        ),
        "structure_funding_stop_below_100bps": lambda trade: (
            _tags(trade).get("trigger") == "structure_break"
            and _tags(trade).get("funding") != "funding_neg"
            and _stop_bucket(trade) in {"lt_50bps", "50_75bps", "75_100bps"}
        ),
    }
    return {
        name: _metrics([trade for trade in trades if predicate(trade)])
        for name, predicate in filters.items()
    }


def diagnose(report: dict[str, Any]) -> dict[str, Any]:
    trades = [
        trade
        for trade in report.get("trades", [])
        if MTF_REASON_TOKEN in str(trade.get("entry_reason", ""))
    ]
    boundaries = {
        "train": (datetime(2025, 6, 12), datetime(2026, 1, 1)),
        "validation": (datetime(2026, 1, 1), datetime(2026, 4, 1)),
        "historical_test": (datetime(2026, 4, 1), datetime(2026, 6, 6)),
    }
    split_trades = {
        name: _split(trades, start, end)
        for name, (start, end) in boundaries.items()
    }
    return {
        "splits": {name: _split_report(rows) for name, rows in split_trades.items()},
        "candidate_filter_counterfactuals": {
            name: _candidate_filter_report(rows)
            for name, rows in split_trades.items()
        },
        "historical_test_is_not_untouched": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MTF high-timeframe trades by time split")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    payload = diagnose(report)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
