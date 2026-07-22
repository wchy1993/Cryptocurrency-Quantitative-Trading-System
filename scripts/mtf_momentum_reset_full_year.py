from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from crypto_scalper.mtf_candidate_quality import candidate_metrics
from scripts.mtf_momentum_reset_regime import (
    SELECTED_REGIME_EXPERIMENT,
    build_regime_report,
)
from scripts.mtf_momentum_reset_selection import _path_stress


FROZEN_PRIORITY_FIELDS = ("target_to_cost_ratio", "rank_score")
FROZEN_RISK_PCT = 0.0321839081


def _accepted_rows(regime_report: dict[str, Any]) -> list[dict[str, Any]]:
    accepted_ids = set(
        regime_report["experiments"][SELECTED_REGIME_EXPERIMENT]["full_year"]["event_ids"]
    )
    return [
        row
        for row in regime_report["rows"]["full_year"]
        if str(row.get("event_id")) in accepted_ids
    ]


def _monthly_path(path: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in path.get("trades", []):
        grouped[str(trade["entry_time"])[:7]].append(trade)
    output: dict[str, dict[str, float | int]] = {}
    for month, trades in sorted(grouped.items()):
        ordered = sorted(trades, key=lambda item: str(item["entry_time"]))
        starting_equity = float(ordered[0]["equity_after"]) - float(ordered[0]["net_pnl"])
        ending_equity = float(ordered[-1]["equity_after"])
        net_pnl = sum(float(item["net_pnl"]) for item in ordered)
        wins = [float(item["net_pnl"]) for item in ordered if float(item["net_pnl"]) > 0.0]
        losses = [float(item["net_pnl"]) for item in ordered if float(item["net_pnl"]) <= 0.0]
        gross_loss = abs(sum(losses))
        output[month] = {
            "trade_count": len(ordered),
            "starting_equity": starting_equity,
            "ending_equity": ending_equity,
            "net_pnl": net_pnl,
            "return_pct": (ending_equity / max(starting_equity, 1e-12) - 1.0) * 100.0,
            "profit_factor": sum(wins) / gross_loss if gross_loss else ("Infinity" if wins else 0.0),
        }
    return output


def build_full_year_report(
    config_path: str,
    data_dir: str,
    event_report_path: str,
    starting_equity: float = 200.0,
    risk_pct: float = FROZEN_RISK_PCT,
    progress: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_payload = json.loads(Path(event_report_path).read_text(encoding="utf-8"))
    regime_report = build_regime_report(
        config_path,
        data_dir,
        {"full_year": event_report_path},
        progress=progress,
    )
    accepted = _accepted_rows(regime_report)
    stresses = _path_stress(
        accepted,
        starting_equity,
        risk_pct,
        priority_fields=FROZEN_PRIORITY_FIELDS,
    )
    for path in stresses.values():
        if isinstance(path, dict) and "trades" in path:
            path["monthly_summary"] = _monthly_path(path)
    report = {
        "research_name": "mtf_1h_reset_30m_release_frozen_full12m_continuous_path",
        "parameter_status": "frozen before this run; no full-year parameter selection",
        "source_manifest": "config.mtf-htf.momentum-reset-stage20-manifest.json",
        "config": config_path,
        "data_dir": data_dir,
        "event_report": event_report_path,
        "start": event_payload["start"],
        "end": event_payload["end"],
        "effective_end_after_embargo": event_payload["effective_end_after_embargo"],
        "symbols": event_payload["symbols"],
        "symbol_count": len(event_payload["symbols"]),
        "starting_equity": starting_equity,
        "risk_pct": risk_pct,
        "frozen_rules": {
            "side": "LONG",
            "minimum_rank_score": 4.25,
            "minimum_target_to_full_cost_ratio": 12.0,
            "minimum_btc_closed_4h_return": -0.001,
            "minimum_directional_breadth_above_ema21": 0.55,
            "candidate_priority": list(FROZEN_PRIORITY_FIELDS),
            "max_open_positions": 1,
            "max_daily_trades": 2,
            "symbol_cooldown_hours": 12,
        },
        "event_counts": event_payload["counts"],
        "accepted_event_metrics": candidate_metrics(accepted),
        "accepted_event_count": len(accepted),
        "continuous_paths": stresses,
        "interpretation": {
            "portfolio_reset": False,
            "path_aware_exclusions": True,
            "historical_period_is_untouched": False,
            "status": "historical continuous-path evaluation; not live approval",
        },
    }
    return report, regime_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen MTF reset setup as one continuous full-year path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regime-output", required=True)
    parser.add_argument("--starting-equity", type=float, default=200.0)
    parser.add_argument("--risk-pct", type=float, default=FROZEN_RISK_PCT)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    report, regime_report = build_full_year_report(
        args.config,
        args.data_dir,
        args.events,
        starting_equity=args.starting_equity,
        risk_pct=args.risk_pct,
        progress=not args.no_progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    regime_output = Path(args.regime_output)
    regime_output.parent.mkdir(parents=True, exist_ok=True)
    regime_output.write_text(json.dumps(regime_report, indent=2, ensure_ascii=False), encoding="utf-8")
    compound = report["continuous_paths"]["compound"]
    print(
        json.dumps(
            {
                "accepted_event_count": report["accepted_event_count"],
                "trade_count": compound["trade_count"],
                "starting_equity": compound["starting_equity"],
                "final_equity": compound["final_equity"],
                "net_pnl": compound["net_pnl"],
                "return_pct": compound["return_pct"],
                "win_rate_pct": compound["win_rate_pct"],
                "profit_factor": compound["profit_factor"],
                "expectancy_r": compound["expectancy_r"],
                "max_drawdown_pct": compound["max_drawdown_pct"],
                "monthly": compound["monthly_summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
