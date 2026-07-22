from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from crypto_scalper.mtf_candidate_quality import candidate_metrics, grouped_report
from crypto_scalper.mtf_momentum_reset import STAGE18_EXPERIMENTS, MtfSetupExperiment
from scripts.mtf_candidate_quality_research import _timestamp
from scripts.mtf_momentum_reset_research import _concentration_stress, _dedupe_event_clusters


SELECTED_EXPERIMENT = "s18_c_long_rank_cost_btc4h_guard"


def _eligible_rows(payload: dict[str, Any], experiment: MtfSetupExperiment) -> list[dict[str, Any]]:
    rows = [
        row
        for row in payload.get("rows", [])
        if not row.get("shadow_reject_reason")
        and bool(row.get("passes_frozen_rank_floor"))
        and bool(row.get("independent_from_structure_window"))
        and experiment.accepts(row)
    ]
    return _dedupe_event_clusters(rows, 4)


def _path_metrics(
    rows: list[dict[str, Any]],
    starting_equity: float,
    risk_pct: float,
    compound: bool,
    exclude_event_ids: set[str] | None = None,
    exclude_symbols: set[str] | None = None,
    priority_fields: tuple[str, ...] = ("rank_score",),
) -> dict[str, Any]:
    excluded_events = exclude_event_ids or set()
    excluded_symbols = exclude_symbols or set()
    def priority_key(row: dict[str, Any]) -> tuple[float, ...]:
        return tuple(-float(row.get(field, -999.0)) for field in priority_fields)

    ordered = sorted(
        (
            row
            for row in rows
            if str(row.get("event_id")) not in excluded_events
            and str(row.get("symbol")) not in excluded_symbols
        ),
        key=lambda row: (
            _timestamp(row.get("signal_available_time")),
            *priority_key(row),
            str(row.get("symbol")),
        ),
    )
    equity = starting_equity
    peak = equity
    max_drawdown = 0.0
    open_until = datetime.min
    daily_entries: dict[Any, int] = defaultdict(int)
    symbol_cooldown_until: dict[str, datetime] = {}
    selected: list[dict[str, Any]] = []
    monthly: dict[str, dict[str, float]] = defaultdict(lambda: {"trade_count": 0, "net_pnl": 0.0})
    fixed_risk_usdt = starting_equity * risk_pct

    for row in ordered:
        signal_time = _timestamp(row.get("signal_available_time"))
        entry_time = _timestamp(row.get("shadow_entry_time") or row.get("signal_available_time"))
        exit_value = row.get("shadow_exit_time")
        if not exit_value:
            continue
        exit_time = _timestamp(exit_value)
        symbol = str(row.get("symbol"))
        if entry_time < open_until:
            continue
        if signal_time < symbol_cooldown_until.get(symbol, datetime.min):
            continue
        if daily_entries[entry_time.date()] >= 2:
            continue
        pnl_r = float(row.get("shadow_net_r", 0.0))
        risk_usdt = equity * risk_pct if compound else fixed_risk_usdt
        net_pnl = risk_usdt * pnl_r
        equity += net_pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / max(peak, 1e-12))
        open_until = max(entry_time, exit_time)
        symbol_cooldown_until[symbol] = open_until + timedelta(hours=12)
        daily_entries[entry_time.date()] += 1
        month = entry_time.strftime("%Y-%m")
        monthly[month]["trade_count"] += 1
        monthly[month]["net_pnl"] += net_pnl
        selected.append(
            {
                "event_id": row.get("event_id"),
                "symbol": symbol,
                "side": row.get("side"),
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "pnl_r": pnl_r,
                "risk_usdt": risk_usdt,
                "net_pnl": net_pnl,
                "equity_after": equity,
            }
        )

    wins = [item["net_pnl"] for item in selected if item["net_pnl"] > 0.0]
    losses = [item["net_pnl"] for item in selected if item["net_pnl"] <= 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "starting_equity": starting_equity,
        "final_equity": equity,
        "net_pnl": equity - starting_equity,
        "return_pct": (equity / starting_equity - 1.0) * 100.0,
        "trade_count": len(selected),
        "win_rate_pct": len(wins) / len(selected) * 100.0 if selected else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else ("Infinity" if gross_profit else 0.0),
        "expectancy_r": sum(item["pnl_r"] for item in selected) / len(selected) if selected else 0.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "compound": compound,
        "risk_pct": risk_pct,
        "max_open_positions": 1,
        "max_daily_trades": 2,
        "symbol_cooldown_hours": 12,
        "priority_fields": list(priority_fields),
        "monthly": dict(sorted(monthly.items())),
        "trades": selected,
    }


def _path_stress(
    rows: list[dict[str, Any]],
    starting_equity: float,
    risk_pct: float,
    priority_fields: tuple[str, ...] = ("rank_score",),
) -> dict[str, Any]:
    baseline = _path_metrics(rows, starting_equity, risk_pct, True, priority_fields=priority_fields)
    ordered_winners = sorted(baseline["trades"], key=lambda item: item["net_pnl"], reverse=True)
    stress: dict[str, Any] = {
        "compound": baseline,
        "fixed_risk": _path_metrics(
            rows,
            starting_equity,
            risk_pct,
            False,
            priority_fields=priority_fields,
        ),
    }
    for count in (1, 3, 5):
        excluded = {str(item["event_id"]) for item in ordered_winners[:count]}
        stress[f"exclude_top_{count}_path_aware"] = _path_metrics(
            rows,
            starting_equity,
            risk_pct,
            True,
            exclude_event_ids=excluded,
            priority_fields=priority_fields,
        )
    symbol_pnl: dict[str, float] = defaultdict(float)
    for trade in baseline["trades"]:
        symbol_pnl[str(trade["symbol"])] += float(trade["net_pnl"])
    if symbol_pnl:
        best_symbol = max(symbol_pnl, key=symbol_pnl.get)
        stress["exclude_best_symbol_path_aware"] = {
            "excluded_symbol": best_symbol,
            **_path_metrics(
                rows,
                starting_equity,
                risk_pct,
                True,
                exclude_symbols={best_symbol},
                priority_fields=priority_fields,
            ),
        }
    return stress


def build_selection_report(
    train_path: str,
    validation_path: str,
    historical_path: str,
    starting_equity: float = 200.0,
    risk_pct: float = 0.0321839081,
) -> dict[str, Any]:
    payloads = {
        "train": json.loads(Path(train_path).read_text(encoding="utf-8")),
        "validation": json.loads(Path(validation_path).read_text(encoding="utf-8")),
        "historical": json.loads(Path(historical_path).read_text(encoding="utf-8")),
    }
    experiment_reports: dict[str, Any] = {}
    for experiment in STAGE18_EXPERIMENTS:
        by_split: dict[str, Any] = {}
        for split, payload in payloads.items():
            rows = _eligible_rows(payload, experiment)
            by_split[split] = {
                "event_metrics": candidate_metrics(rows),
                "by_month": grouped_report(rows, "month"),
                "by_side": grouped_report(rows, "side"),
                "by_symbol": grouped_report(rows, "symbol"),
                "concentration": _concentration_stress(rows),
                "event_ids": [row["event_id"] for row in rows],
            }
        experiment_reports[experiment.name] = {
            "parameters": experiment.__dict__,
            "splits": by_split,
        }

    selected = next(item for item in STAGE18_EXPERIMENTS if item.name == SELECTED_EXPERIMENT)
    selected_paths = {
        split: _path_stress(_eligible_rows(payload, selected), starting_equity, risk_pct)
        for split, payload in payloads.items()
    }
    return {
        "research_name": "mtf_1h_reset_30m_release_stage18_selection",
        "selection_policy": {
            "candidate_generation": "train feature buckets only",
            "parameter_selection": "validation; simplest adjacent threshold preferred",
            "historical_use": "evaluation only; previously observed, not untouched",
            "selected_experiment": SELECTED_EXPERIMENT,
            "portfolio_status": "normalized risk path diagnostic; not production integration",
        },
        "source_reports": {
            "train": train_path,
            "validation": validation_path,
            "historical": historical_path,
        },
        "experiment_budget": len(STAGE18_EXPERIMENTS),
        "experiments": experiment_reports,
        "selected_normalized_paths": selected_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and stress the MTF momentum-reset setup")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--starting-equity", type=float, default=200.0)
    parser.add_argument("--risk-pct", type=float, default=0.0321839081)
    args = parser.parse_args()
    payload = build_selection_report(
        args.train,
        args.validation,
        args.historical,
        args.starting_equity,
        args.risk_pct,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    selected = payload["selection_policy"]["selected_experiment"]
    print(
        json.dumps(
            {
                "selected": selected,
                "event_metrics": {
                    split: payload["experiments"][selected]["splits"][split]["event_metrics"]
                    for split in ("train", "validation", "historical")
                },
                "normalized_paths": {
                    split: {
                        key: value
                        for key, value in payload["selected_normalized_paths"][split]["compound"].items()
                        if key not in {"trades", "monthly"}
                    }
                    for split in ("train", "validation", "historical")
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
