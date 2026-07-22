from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.mtf_momentum_reset_regime import SELECTED_REGIME_EXPERIMENT
from scripts.mtf_momentum_reset_selection import _path_stress


@dataclass(frozen=True)
class PriorityExperiment:
    name: str
    fields: tuple[str, ...]


STAGE20_PRIORITY_EXPERIMENTS: tuple[PriorityExperiment, ...] = (
    PriorityExperiment("s20_a_frozen_rank", ("rank_score",)),
    PriorityExperiment("s20_b_trend_score", ("trend_score", "rank_score")),
    PriorityExperiment("s20_c_score_gap", ("score_gap", "rank_score")),
    PriorityExperiment("s20_d_target_cost", ("target_to_cost_ratio", "rank_score")),
    PriorityExperiment(
        "s20_e_30m_alignment",
        ("ema9_ema21_alignment_30m_atr", "rank_score"),
    ),
    PriorityExperiment(
        "s20_f_trend_then_gap",
        ("trend_score", "score_gap", "rank_score"),
    ),
)


def _selected_regime_rows(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    selected_ids = set(
        payload["experiments"][SELECTED_REGIME_EXPERIMENT][split]["event_ids"]
    )
    return [
        row
        for row in payload["rows"][split]
        if str(row.get("event_id")) in selected_ids
    ]


def _pf(payload: dict[str, Any]) -> float:
    value = payload.get("profit_factor", 0.0)
    return float("inf") if value == "Infinity" else float(value)


def _selection_score(experiment: dict[str, Any]) -> float:
    train = experiment["splits"]["train"]
    validation = experiment["splits"]["validation"]
    if validation["compound"]["trade_count"] < 30 or train["compound"]["trade_count"] < 60:
        return -999.0
    return min(
        _pf(train["compound"]),
        _pf(validation["compound"]),
        _pf(validation["exclude_top_1_path_aware"]),
    )


def build_priority_report(
    regime_report_path: str,
    starting_equity: float = 200.0,
    risk_pct: float = 0.0321839081,
) -> dict[str, Any]:
    source = json.loads(Path(regime_report_path).read_text(encoding="utf-8"))
    experiments: dict[str, Any] = {}
    for experiment in STAGE20_PRIORITY_EXPERIMENTS:
        splits = {
            split: _path_stress(
                _selected_regime_rows(source, split),
                starting_equity,
                risk_pct,
                priority_fields=experiment.fields,
            )
            for split in ("train", "validation", "historical")
        }
        experiments[experiment.name] = {
            "priority_fields": list(experiment.fields),
            "splits": splits,
        }
        experiments[experiment.name]["selection_score"] = _selection_score(
            experiments[experiment.name]
        )

    selected = min(
        STAGE20_PRIORITY_EXPERIMENTS,
        key=lambda item: (
            -experiments[item.name]["selection_score"],
            -_pf(experiments[item.name]["splits"]["train"]["compound"]),
            float(experiments[item.name]["splits"]["train"]["compound"]["max_drawdown_pct"]),
            len(item.fields),
            item.name,
        ),
    )
    return {
        "research_name": "mtf_1h_reset_30m_release_stage20_candidate_priority",
        "source_report": regime_report_path,
        "selected_regime_experiment": SELECTED_REGIME_EXPERIMENT,
        "experiment_budget": len(STAGE20_PRIORITY_EXPERIMENTS),
        "selection_policy": {
            "selected_priority_experiment": selected.name,
            "objective": "maximize the minimum of train PF, validation PF, and validation path-aware exclude-top-1 PF",
            "minimum_samples": {"train": 60, "validation": 30},
            "tie_break": "higher train PF, lower train drawdown, fewer priority fields, then stable experiment name",
            "historical_role": "evaluation only; not used for selection",
            "path_status": "normalized risk diagnostic; no production integration",
        },
        "experiments": experiments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank simultaneous MTF momentum-reset candidates")
    parser.add_argument("--regime-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--starting-equity", type=float, default=200.0)
    parser.add_argument("--risk-pct", type=float, default=0.0321839081)
    args = parser.parse_args()
    payload = build_priority_report(
        args.regime_report,
        starting_equity=args.starting_equity,
        risk_pct=args.risk_pct,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    selected = payload["selection_policy"]["selected_priority_experiment"]
    print(
        json.dumps(
            {
                "selected": selected,
                "selection_score": payload["experiments"][selected]["selection_score"],
                "splits": {
                    split: {
                        key: value
                        for key, value in payload["experiments"][selected]["splits"][split][
                            "compound"
                        ].items()
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
