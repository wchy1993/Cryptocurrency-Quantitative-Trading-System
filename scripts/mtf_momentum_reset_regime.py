from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_execution_backtest import _resample_to_timeframe
from crypto_scalper.live_portfolio_backtest import _load_symbol_data
from crypto_scalper.models import Direction
from crypto_scalper.mtf_candidate_quality import candidate_metrics, grouped_report, quantile_bucket_report
from crypto_scalper.mtf_momentum_reset import STAGE18_EXPERIMENTS
from crypto_scalper.regime_score import RegimeScoreEngine, snapshot_payload
from scripts.mtf_momentum_reset_research import _concentration_stress, _dedupe_event_clusters
from scripts.mtf_momentum_reset_selection import SELECTED_EXPERIMENT
from scripts.mtf_momentum_reset_selection import _path_stress


REGIME_EXPERIMENTS: tuple[tuple[str, Callable[[dict[str, Any]], bool]], ...] = (
    ("s19_a_selected_without_market_gate", lambda row: True),
    (
        "s19_b_directional_breadth_ema55",
        lambda row: float(row.get("directional_breadth_ema21", 0.0)) >= 0.55,
    ),
    (
        "s19_c_existing_trend_regime",
        lambda row: float(row.get("trend_score", 0.0)) >= 60.0
        and float(row.get("score_gap", -999.0)) >= 10.0,
    ),
    (
        "s19_d_breadth55_trend50",
        lambda row: float(row.get("directional_breadth_ema21", 0.0)) >= 0.55
        and float(row.get("trend_score", 0.0)) >= 50.0,
    ),
)

SELECTED_REGIME_EXPERIMENT = "s19_b_directional_breadth_ema55"


def _build_frames(config_path: str, data_dir: str, progress: bool) -> tuple[Any, dict[str, dict[str, Any]]]:
    config = load_live_config(config_path)
    frames: dict[str, dict[str, Any]] = {timeframe: {} for timeframe in ("15m", "30m", "1h")}
    symbols = tuple(config.trading.symbols)
    for index, symbol in enumerate(symbols, start=1):
        loaded = _load_symbol_data(data_dir, (symbol,), "1m")
        if symbol not in loaded:
            continue
        for timeframe in frames:
            frames[timeframe][symbol] = _resample_to_timeframe(loaded[symbol], "1m", timeframe)
        if progress:
            print(f"[{index}/{len(symbols)}] loaded breadth frames for {symbol}", flush=True)
    return config, frames


def _selected_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    experiment = next(item for item in STAGE18_EXPERIMENTS if item.name == SELECTED_EXPERIMENT)
    return _dedupe_event_clusters(
        [
            row
            for row in payload.get("rows", [])
            if not row.get("shadow_reject_reason")
            and bool(row.get("passes_frozen_rank_floor"))
            and experiment.accepts(row)
        ],
        4,
    )


def build_regime_report(
    config_path: str,
    data_dir: str,
    source_reports: dict[str, str],
    progress: bool = True,
) -> dict[str, Any]:
    config, frames = _build_frames(config_path, data_dir, progress)
    score_config = replace(config.regime_score, enabled=True)
    engine = RegimeScoreEngine(score_config, frames)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, source_path in source_reports.items():
        payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
        enriched: list[dict[str, Any]] = []
        for row in _selected_rows(payload):
            direction = Direction.LONG if str(row.get("side")) == "LONG" else Direction.SHORT
            snapshot = engine.score(str(row["symbol"]), _timestamp(row["signal_available_time"]), direction)
            snapshot_row = snapshot_payload(snapshot)
            features = snapshot_row.pop("regime_score_features", {})
            components = snapshot_row.pop("regime_score_components", {})
            enriched.append({**row, **snapshot_row, **features, "regime_score_components": components})
        split_rows[split] = enriched

    reports: dict[str, Any] = {}
    accepted_by_experiment: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, predicate in REGIME_EXPERIMENTS:
        split_report: dict[str, Any] = {}
        accepted_by_split: dict[str, list[dict[str, Any]]] = {}
        for split, rows in split_rows.items():
            accepted = [row for row in rows if predicate(row)]
            accepted_by_split[split] = accepted
            split_report[split] = {
                "metrics": candidate_metrics(accepted),
                "by_month": grouped_report(accepted, "month"),
                "by_symbol": grouped_report(accepted, "symbol"),
                "concentration": _concentration_stress(accepted),
                "event_ids": [row["event_id"] for row in accepted],
            }
        reports[name] = split_report
        accepted_by_experiment[name] = accepted_by_split
    bucket_fields = (
        "trend_score",
        "score_gap",
        "directional_breadth_return",
        "directional_breadth_ema21",
        "btc_alignment_atr",
        "ema21_slope_1h_atr",
        "ema9_ema21_alignment_30m_atr",
        "atr_expansion_30m_ratio",
    )
    return {
        "research_name": "mtf_momentum_reset_market_regime_stage19",
        "config": config_path,
        "data_dir": data_dir,
        "source_reports": source_reports,
        "selected_setup": SELECTED_EXPERIMENT,
        "experiment_budget": len(REGIME_EXPERIMENTS),
        "selection_policy": {
            "selected_regime_experiment": SELECTED_REGIME_EXPERIMENT,
            "basis": "positive train and validation; simplest rule with adequate sample preferred",
            "historical_role": "evaluation only, not parameter selection",
        },
        "method": {
            "engine": "existing RegimeScoreEngine",
            "features": "point-in-time closed 15m/30m/1h universe",
            "selection": "train/validation only; historical evaluation only",
        },
        "experiments": reports,
        "selected_normalized_paths": {
            split: _path_stress(rows, 200.0, 0.0321839081)
            for split, rows in accepted_by_experiment[SELECTED_REGIME_EXPERIMENT].items()
        },
        "feature_buckets": {
            split: {
                field: quantile_bucket_report(rows, field, buckets=3, min_sample=20)
                for field in bucket_fields
            }
            for split, rows in split_rows.items()
        },
        "rows": split_rows,
    }


def _timestamp(value: Any):
    from scripts.mtf_candidate_quality_research import _timestamp as parse

    return parse(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply point-in-time market regime diagnostics to MTF reset events")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--historical", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    payload = build_regime_report(
        args.config,
        args.data_dir,
        {
            "train": args.train,
            "validation": args.validation,
            "historical": args.historical,
        },
        progress=not args.no_progress,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    split: report[split]["metrics"]
                    for split in ("train", "validation", "historical")
                }
                for name, report in payload["experiments"].items()
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
