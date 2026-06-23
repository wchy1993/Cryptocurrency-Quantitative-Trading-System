from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .data import load_candles_csv, parse_timestamp
from .live_execution_backtest import run_execution_backtest
from .models import Candle


VBP_STRATEGY = "volume_breakout_pullback"
DEFAULT_THRESHOLD = 0.55
BASELINE_POLICY_VERSION = "vbp_candidate_policy_v1"
FEATURE_VERSION = "vbp_lgbm_features_v2"
LABEL_VERSION = "vbp_full_cost_r_labels_v2"
COST_MODEL_VERSION = "full_cost_live_execution_v1"
MODEL_TARGETS = (
    "p_good_trade",
    "expected_net_r",
    "p_large_loss",
    "p_false_breakout",
    "expected_mfe_r",
    "expected_mae_r",
)
METADATA_COLUMNS = {
    "symbol",
    "entry_time",
    "exit_time",
    "entry_reason",
    "exit_reason",
    "strategy",
    "side",
    "label",
    "net_pnl",
    "gross_pnl",
    "fee",
    "slippage_cost",
    "funding",
    "notional",
    "pnl_r",
    "return_pct",
    "raw_entry_price",
    "raw_exit_price",
    "qty",
    "quantity",
    "entry_fee",
    "exit_fee",
    "fees",
    "execution_gross_pnl",
    "net_bps",
    "exit_price",
    "mfe",
    "mae",
    "bars",
    "scale_ins",
    "hold_minutes",
    "avg_hold_minutes",
    "event_cluster_id",
    "baseline_policy_version",
    "strategy_config_hash",
    "feature_version",
    "label_version",
    "cost_model_version",
    "net_r",
    "net_bps",
    "mfe_r",
    "mae_r",
    "good_trade_label",
    "large_loss_label",
    "false_breakout_label",
    "expected_net_r_target",
    "expected_mfe_r_target",
    "expected_mae_r_target",
    "sample_weight",
    "ml_quality_score",
    "ml_selected",
    "p_good_trade",
    "expected_net_r",
    "p_large_loss",
    "p_false_breakout",
    "expected_mfe_r",
    "expected_mae_r",
}


@dataclass
class SymbolFeatureCache:
    timestamps: list[datetime]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]
    close_prefix: list[float]
    volume_prefix: list[float]
    high_1d: list[float]
    low_1d: list[float]
    high_7d: list[float]
    low_7d: list[float]
    high_30d: list[float]
    low_30d: list[float]


@dataclass(frozen=True)
class VbpMlDecision:
    allowed: bool
    score: float
    risk_multiplier: float
    reason: str
    predictions: dict[str, float]


class MarketBreadthRuntimeCache:
    def __init__(self, candles_by_symbol: dict[str, list[Candle]]) -> None:
        self.candles_by_symbol = candles_by_symbol
        reference = candles_by_symbol.get("BTCUSDT") or next(iter(candles_by_symbol.values()), [])
        self.timestamps = [candle.timestamp for candle in reference]
        self.memo: dict[datetime, dict[str, float]] = {}

    def features_at(self, entry_time: datetime) -> dict[str, float]:
        key = entry_time.replace(second=0, microsecond=0)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        index = bisect.bisect_right(self.timestamps, entry_time) - 1
        if index < 100:
            result = _empty_market_breadth_features()
            self.memo[key] = result
            return result
        ret_5m: list[float] = []
        ret_15m: list[float] = []
        ret_1h: list[float] = []
        above_ema21 = 0
        above_ma99 = 0
        simultaneous_breakouts = 0
        simultaneous_candidates = 0
        major_up = 0
        major_total = 0
        for symbol, candles in self.candles_by_symbol.items():
            if index >= len(candles) or index < 100:
                continue
            close = candles[index].close
            r5 = _ret_candles(candles, index, 5)
            r15 = _ret_candles(candles, index, 15)
            r60 = _ret_candles(candles, index, 60)
            if math.isfinite(r5):
                ret_5m.append(r5)
            if math.isfinite(r15):
                ret_15m.append(r15)
            if math.isfinite(r60):
                ret_1h.append(r60)
            if close > _ema_candles_at(candles, index, 21):
                above_ema21 += 1
            if close > _sma_candles_at(candles, index, 99):
                above_ma99 += 1
            recent_high = max((candle.high for candle in candles[index - 48:index]), default=close)
            if close > recent_high:
                simultaneous_breakouts += 1
            if close >= recent_high * 0.995:
                simultaneous_candidates += 1
            if symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
                major_total += 1
                if r15 > 0 and r60 > 0:
                    major_up += 1
        total = max(1, len(ret_1h))
        top10 = sorted(ret_1h, reverse=True)[:10]
        result = {
            "market_positive_return_ratio_5m": sum(1 for value in ret_5m if value > 0) / max(1, len(ret_5m)),
            "market_positive_return_ratio_15m": sum(1 for value in ret_15m if value > 0) / max(1, len(ret_15m)),
            "market_positive_return_ratio_1h": sum(1 for value in ret_1h if value > 0) / total,
            "market_above_ema21_ratio": above_ema21 / total,
            "market_above_ma99_ratio": above_ma99 / total,
            "market_average_return_15m": sum(ret_15m) / max(1, len(ret_15m)),
            "market_average_return_1h": sum(ret_1h) / total,
            "market_median_return_15m": _median(ret_15m),
            "simultaneous_breakout_count": float(simultaneous_breakouts),
            "simultaneous_vbp_candidate_count": float(simultaneous_candidates),
            "top10_market_return": sum(top10) / max(1, len(top10)),
            "market_breadth_change": (sum(1 for value in ret_5m if value > 0) / max(1, len(ret_5m))) - (sum(1 for value in ret_1h if value > 0) / total),
            "major_coins_alignment": major_up / max(1, major_total),
        }
        self.memo[key] = result
        if len(self.memo) > 20_000:
            self.memo.clear()
        return result


class VbpLightGbmFilter:
    def __init__(
        self,
        model_path: str,
        threshold: float,
        candles_by_symbol: dict[str, list[Candle]],
        mode: str = "filter",
        reject_enabled: bool | None = None,
        sizing_enabled: bool = True,
        min_score: float = -0.25,
        neutral_score: float = 0.0,
        high_score: float = 0.35,
        min_risk_multiplier: float = 0.35,
        max_risk_multiplier: float = 1.15,
        fail_open_without_schedule: bool = True,
    ) -> None:
        lgb = _import_lightgbm()
        path = Path(model_path)
        self.mode = str(mode or "filter").lower()
        self.reject_enabled = bool(self.mode in {"filter", "reject", "score_size_filter"} if reject_enabled is None else reject_enabled)
        self.sizing_enabled = bool(sizing_enabled and self.mode in {"score_size", "score_size_filter", "sizing"})
        self.min_score = float(min_score)
        self.neutral_score = float(neutral_score)
        self.high_score = float(high_score)
        self.min_risk_multiplier = max(0.0, float(min_risk_multiplier))
        self.max_risk_multiplier = max(self.min_risk_multiplier, float(max_risk_multiplier))
        self.fail_open_without_schedule = bool(fail_open_without_schedule)
        self.schedule: list[dict[str, Any]] = []
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.threshold = float(payload.get("threshold", threshold))
            base = path.parent
            for window in payload.get("windows", []):
                models: dict[str, Any] = {}
                if "model_paths" in window:
                    for name, raw_path in dict(window["model_paths"]).items():
                        model_file = Path(str(raw_path))
                        if not model_file.is_absolute():
                            model_file = base / model_file
                        models[name] = lgb.Booster(model_file=str(model_file))
                    first_model = next(iter(models.values()))
                else:
                    model_file = Path(str(window["model_path"]))
                    if not model_file.is_absolute():
                        model_file = base / model_file
                    first_model = lgb.Booster(model_file=str(model_file))
                    models["p_good_trade"] = first_model
                self.schedule.append(
                    {
                        "start": parse_timestamp(str(window["test_start"])),
                        "end": parse_timestamp(str(window["test_end"])),
                        "model": first_model,
                        "models": models,
                        "feature_columns": list(first_model.feature_name()),
                        "decision_rule": dict(window.get("decision_rule") or {}),
                    }
                )
            if not self.schedule:
                raise RuntimeError(f"empty VBP ML schedule: {model_path}")
            self.model = None
            self.feature_columns = list(self.schedule[0]["feature_columns"])
        else:
            self.model = lgb.Booster(model_file=model_path)
            self.threshold = threshold
            self.models = {"p_good_trade": self.model}
            self.feature_columns = list(self.model.feature_name())
        self.candles_by_symbol = candles_by_symbol
        self.caches: dict[str, SymbolFeatureCache] = {}
        self.cache_order: deque[str] = deque()
        self.cache_limit = max(4, int(os.environ.get("VBP_ML_SYMBOL_CACHE_LIMIT", "16")))
        self.market_breadth_cache = MarketBreadthRuntimeCache(candles_by_symbol)
        self.score_count = 0
        _progress(
            f"path-aware ML filter loaded lazy_symbol_cache_limit={self.cache_limit} "
            f"market_symbols={len(candles_by_symbol)} mode={self.mode} "
            f"reject={int(self.reject_enabled)} sizing={int(self.sizing_enabled)}"
        )

    def _cache_for_symbol(self, symbol: str) -> SymbolFeatureCache | None:
        cached = self.caches.get(symbol)
        if cached is not None:
            return cached
        candles = self.candles_by_symbol.get(symbol)
        if not candles:
            return None
        cached = _feature_cache(candles)
        self.caches[symbol] = cached
        self.cache_order.append(symbol)
        while len(self.caches) > self.cache_limit and self.cache_order:
            old_symbol = self.cache_order.popleft()
            if old_symbol in {symbol, "BTCUSDT"}:
                self.cache_order.append(old_symbol)
                break
            self.caches.pop(old_symbol, None)
        return cached

    def _prediction_bundle(
        self,
        symbol: str,
        entry_time: Any,
        entry_price: float,
        entry_reason: str,
        signal_time: Any = None,
    ) -> tuple[dict[str, float], dict[str, Any] | None]:
        entry_time_text = entry_time.isoformat() if hasattr(entry_time, "isoformat") else str(entry_time)
        signal_time_text = signal_time.isoformat() if hasattr(signal_time, "isoformat") else (str(signal_time) if signal_time is not None else entry_time_text)
        scoped_caches: dict[str, SymbolFeatureCache] = {}
        symbol_cache = self._cache_for_symbol(symbol)
        if symbol_cache is not None:
            scoped_caches[symbol] = symbol_cache
        btc_cache = symbol_cache if symbol == "BTCUSDT" else self._cache_for_symbol("BTCUSDT")
        if btc_cache is not None:
            scoped_caches["BTCUSDT"] = btc_cache
        rows = _build_dataset_rows(
            [
                {
                    "symbol": symbol,
                    "strategy": VBP_STRATEGY,
                    "side": "LONG",
                    "entry_time": entry_time_text,
                    "exit_time": "",
                    "signal_time": signal_time_text,
                    "signal_available_time": entry_time_text,
                    "entry_price": entry_price,
                    "exit_price": 0.0,
                    "entry_reason": entry_reason,
                    "exit_reason": "",
                    "net_pnl": 0.0,
                    "gross_pnl": 0.0,
                    "fee": 0.0,
                    "slippage_cost": 0.0,
                    "funding": 0.0,
                    "notional": 0.0,
                    "return_pct": 0.0,
                    "pnl_r": 0.0,
                    "mfe": 0.0,
                    "mae": 0.0,
                    "hold_minutes": 0.0,
                    "label": 0,
                }
            ],
            scoped_caches,
            market_breadth_cache=self.market_breadth_cache,
        )
        if not rows:
            return {}, None
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("NumPy is not installed; install it before using --vbp-ml-model") from exc
        models = getattr(self, "models", {"p_good_trade": self.model})
        feature_columns = self.feature_columns
        selected = None
        if self.schedule:
            parsed_time = parse_timestamp(entry_time_text)
            for window in self.schedule:
                if window["start"] <= parsed_time < window["end"]:
                    selected = window
                    break
            if selected is None:
                return {"allow_without_schedule": 1.0}, None
            models = selected.get("models") or {"p_good_trade": selected["model"]}
            feature_columns = selected["feature_columns"]
        matrix = np.array([[_number(rows[0].get(column)) for column in feature_columns]], dtype=float)
        predictions: dict[str, float] = {}
        for name, model in models.items():
            predictions[name] = float(model.predict(matrix)[0])
        for probability_key in ("p_good_trade", "p_large_loss", "p_false_breakout"):
            if probability_key in predictions:
                predictions[probability_key] = max(0.0, min(1.0, predictions[probability_key]))
        if "expected_net_r" in predictions:
            predictions["quality_score"] = _quality_score(predictions)
        elif "p_good_trade" in predictions:
            predictions["quality_score"] = predictions["p_good_trade"]
        return predictions, selected

    def score(self, symbol: str, entry_time: Any, entry_price: float, entry_reason: str, signal_time: Any = None) -> float:
        predictions, _ = self._prediction_bundle(symbol, entry_time, entry_price, entry_reason, signal_time=signal_time)
        if "allow_without_schedule" in predictions:
            return 1.0
        return float(predictions.get("quality_score", predictions.get("p_good_trade", 0.0)))

    def decision(self, symbol: str, entry_time: Any, entry_price: float, entry_reason: str, signal_time: Any = None) -> VbpMlDecision:
        predictions, window = self._prediction_bundle(symbol, entry_time, entry_price, entry_reason, signal_time=signal_time)
        if "allow_without_schedule" in predictions:
            allowed = self.fail_open_without_schedule
            decision = VbpMlDecision(
                allowed=allowed,
                score=1.0 if allowed else 0.0,
                risk_multiplier=1.0 if allowed else 0.0,
                reason="ml_no_schedule_fail_open" if allowed else "ml_no_schedule_fail_close",
                predictions=predictions,
            )
            self._log_score_progress(symbol, entry_time, decision.score, decision.allowed, decision.risk_multiplier)
            return decision
        if not predictions:
            allowed = not self.reject_enabled
            decision = VbpMlDecision(
                allowed=allowed,
                score=0.0,
                risk_multiplier=self.min_risk_multiplier if allowed and self.sizing_enabled else (1.0 if allowed else 0.0),
                reason="ml_missing_features_allow" if allowed else "ml_missing_features_reject",
                predictions={},
            )
            self._log_score_progress(symbol, entry_time, decision.score, decision.allowed, decision.risk_multiplier)
            return decision

        rule = dict(window.get("decision_rule") or {}) if window else {}
        reject_enabled = bool(rule.get("reject_enabled", self.reject_enabled))
        sizing_enabled = bool(rule.get("sizing_enabled", self.sizing_enabled))
        if "expected_net_r" in predictions:
            score = float(predictions.get("quality_score", 0.0))
            if self.mode == "score_size_filter":
                allowed_by_rule = _soft_filter_allows(predictions, rule, self.min_score)
            else:
                allowed_by_rule = _decision_allows(predictions, rule)
        else:
            score = float(predictions.get("p_good_trade", 0.0))
            allowed_by_rule = score >= float(rule.get("p_good_trade_min", self.threshold))
        allowed = allowed_by_rule if reject_enabled else True
        risk_multiplier = (
            _score_risk_multiplier(
                score,
                rule,
                self.min_score,
                self.neutral_score,
                self.high_score,
                self.min_risk_multiplier,
                self.max_risk_multiplier,
            )
            if sizing_enabled
            else 1.0
        )
        if not allowed:
            risk_multiplier = 0.0
        if reject_enabled:
            reason = "ml_allowed" if allowed else "ml_reject_rule"
        else:
            reason = "ml_score_sizing"
        decision = VbpMlDecision(
            allowed=allowed,
            score=score,
            risk_multiplier=risk_multiplier,
            reason=reason,
            predictions=predictions,
        )
        self._log_score_progress(symbol, entry_time, decision.score, decision.allowed, decision.risk_multiplier)
        return decision

    def allows(self, symbol: str, entry_time: Any, entry_price: float, entry_reason: str, signal_time: Any = None) -> tuple[bool, float]:
        decision = self.decision(symbol, entry_time, entry_price, entry_reason, signal_time=signal_time)
        return decision.allowed, decision.score

    def _log_score_progress(self, symbol: str, entry_time: Any, score: float, allowed: bool, risk_multiplier: float = 1.0) -> None:
        self.score_count += 1
        if self.score_count == 1 or self.score_count % 50 == 0:
            _progress(
                f"path-aware ML scored={self.score_count} time={entry_time} "
                f"symbol={symbol} score={score:.4f} allowed={int(allowed)} ml_mult={risk_multiplier:.3f} "
                f"symbol_cache={len(self.caches)} breadth_cache={len(self.market_breadth_cache.memo)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a LightGBM meta-filter for VBP candidates")
    parser.add_argument("--config", default="config.live_safe.json")
    parser.add_argument("--execution-data-dir", default="data/binance_1m_365d_top100")
    parser.add_argument("--initial-equity", type=float, default=160.0)
    parser.add_argument("--backtest-json", default=None, help="Use an existing --include-trades backtest JSON instead of running a backtest")
    parser.add_argument("--dataset-csv", default=None, help="Use an existing VBP ML dataset CSV and skip candidate generation")
    parser.add_argument("--executed-trade-mode", action="store_true", help="Train from executed VBP trades instead of candidate signals")
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--test-start", default=None, help="UTC ISO timestamp. Rows before this time are train, rows at/after are test")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Used only when --test-start is omitted")
    parser.add_argument("--label-min-r", type=float, default=0.0, help="Positive label if aggregated VBP pnl_r is at least this value")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num-boost-round", type=int, default=250)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--training-window-mode", choices=["expanding", "rolling", "recency_weighted"], default="expanding")
    parser.add_argument("--recency-half-life-months", type=float, default=3.0)
    parser.add_argument("--recency-weight-min", type=float, default=0.25)
    parser.add_argument("--compare-training-windows", action="store_true")
    parser.add_argument("--embargo-minutes", type=int, default=120)
    parser.add_argument("--min-selected-per-window", type=int, default=10)
    parser.add_argument("--path-aware-backtest", action="store_true")
    parser.add_argument("--path-aware-live-safe", action="store_true", help="Run path-aware backtest with reject_enabled=false live-safe score sizing")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    _progress(f"output_dir={output_dir}")
    config_hash = _file_sha256(Path(args.config))
    if args.dataset_csv:
        _progress(f"loading existing dataset csv: {args.dataset_csv}")
        positions = _read_dataset_csv(Path(args.dataset_csv))
        if len(positions) < 30:
            raise RuntimeError(f"not enough VBP samples for training: {len(positions)}")
        _progress("augmenting existing candidate dataset with current point-in-time features")
        symbols = sorted(set(_config_symbols(args.config)) | {str(row["symbol"]) for row in positions} | {"BTCUSDT"})
        symbol_caches = _load_symbol_caches(args.execution_data_dir, symbols, progress=True)
        rows = _build_dataset_rows(positions, symbol_caches, strategy_config_hash=config_hash, progress=True)
        rows.sort(key=lambda row: row["entry_time"])
        _write_csv(output_dir / "vbp_ml_dataset.csv", rows)
        _write_dataset_reports(output_dir, rows)
        _progress(f"dataset_rows={len(rows)}")
        if args.walk_forward:
            metrics = _run_training_window_comparison(rows, args, output_dir) if args.compare_training_windows else _run_walk_forward(rows, args, output_dir)
            print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))
            return 0
        train_rows, test_rows = _split_rows(rows, args.test_start, args.test_ratio)
        feature_columns = _feature_columns(rows)
        _progress(f"training single split model train={len(train_rows)} test={len(test_rows)} features={len(feature_columns)}")
        model, train_pred, test_pred = _train_lightgbm(train_rows, test_rows, feature_columns, args.num_boost_round)
        _write_csv(output_dir / "vbp_ml_train.csv", train_rows)
        _write_csv(output_dir / "vbp_ml_test.csv", test_rows)
        _write_prediction_csv(output_dir / "train_predictions.csv", train_rows, train_pred)
        _write_prediction_csv(output_dir / "test_predictions.csv", test_rows, test_pred)
        model.save_model(str(output_dir / "vbp_lightgbm_model.txt"))
        _write_feature_importance(output_dir / "feature_importance.csv", model, feature_columns)
        metrics = _single_split_metrics(args, rows, train_rows, test_rows, train_pred, test_pred, feature_columns)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_summary(output_dir / "summary.md", metrics, output_dir)
        print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))
        return 0
    if args.backtest_json:
        _progress(f"loading existing backtest json: {args.backtest_json}")
        backtest_payload = json.loads(Path(args.backtest_json).read_text(encoding="utf-8"))
        trades = backtest_payload.get("trades") or []
        positions = _aggregate_vbp_positions(trades, label_min_r=args.label_min_r)
    else:
        trade_start = parse_timestamp(args.trade_start) if args.trade_start else None
        trade_end = parse_timestamp(args.trade_end) if args.trade_end else None
        candidate_rows: list[dict[str, Any]] = []
        _progress("generating VBP candidate dataset with full-cost path labels")
        backtest_payload = run_execution_backtest(
            args.config,
            args.execution_data_dir,
            initial_equity=args.initial_equity,
            include_trades=args.executed_trade_mode,
            compact=True,
            progress=True,
            trade_start=trade_start,
            trade_end=trade_end,
            cost_experiment="full_cost",
            vbp_candidate_rows=None if args.executed_trade_mode else candidate_rows,
        )
        if args.executed_trade_mode:
            trades = backtest_payload.get("trades") or []
            positions = _aggregate_vbp_positions(trades, label_min_r=args.label_min_r)
        else:
            positions = candidate_rows
    _progress(f"raw_samples={len(positions)}")
    if not positions:
        raise RuntimeError("no VBP samples found; make sure VBP is enabled")

    _progress("loading 1m candles for feature extraction")
    symbols = sorted(set(_config_symbols(args.config)) | {str(row["symbol"]) for row in positions} | {"BTCUSDT"})
    symbol_caches = _load_symbol_caches(args.execution_data_dir, symbols, progress=True)
    _progress("building point-in-time feature matrix")
    rows = _build_dataset_rows(positions, symbol_caches, strategy_config_hash=config_hash, progress=True)
    if len(rows) < 30:
        raise RuntimeError(f"not enough VBP samples for training: {len(rows)}")

    rows.sort(key=lambda row: row["entry_time"])
    _write_csv(output_dir / "vbp_ml_dataset.csv", rows)
    _write_dataset_reports(output_dir, rows)
    _progress(f"dataset_rows={len(rows)}")
    if args.walk_forward:
        metrics = _run_training_window_comparison(rows, args, output_dir) if args.compare_training_windows else _run_walk_forward(rows, args, output_dir)
        print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))
        return 0

    train_rows, test_rows = _split_rows(rows, args.test_start, args.test_ratio)
    if not train_rows or not test_rows:
        raise RuntimeError(f"invalid train/test split: train={len(train_rows)} test={len(test_rows)}")

    feature_columns = _feature_columns(rows)
    _progress(f"training single split model train={len(train_rows)} test={len(test_rows)} features={len(feature_columns)}")
    model, train_pred, test_pred = _train_lightgbm(train_rows, test_rows, feature_columns, args.num_boost_round)

    _write_csv(output_dir / "vbp_ml_train.csv", train_rows)
    _write_csv(output_dir / "vbp_ml_test.csv", test_rows)
    _write_prediction_csv(output_dir / "train_predictions.csv", train_rows, train_pred)
    _write_prediction_csv(output_dir / "test_predictions.csv", test_rows, test_pred)
    model.save_model(str(output_dir / "vbp_lightgbm_model.txt"))
    _write_feature_importance(output_dir / "feature_importance.csv", model, feature_columns)

    metrics = _single_split_metrics(args, rows, train_rows, test_rows, train_pred, test_pred, feature_columns)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(output_dir / "summary.md", metrics, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2, ensure_ascii=False))
    return 0


def _run_training_window_comparison(rows: list[dict[str, Any]], args: Any, output_dir: Path) -> dict[str, Any]:
    specs = [
        ("rolling_6m", "rolling", 6),
        ("rolling_9m", "rolling", 9),
        ("recency_weighted_9m", "recency_weighted", 9),
    ]
    comparison_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for offset, (name, mode, months) in enumerate(specs, start=1):
        _progress(f"training_window_compare {offset}/{len(specs)} name={name} mode={mode} train_months={months}")
        child_args = argparse.Namespace(**vars(args))
        child_args.training_window_mode = mode
        child_args.train_months = months
        child_output = output_dir / name
        child_output.mkdir(parents=True, exist_ok=True)
        try:
            metrics = _run_walk_forward(rows, child_args, child_output)
        except RuntimeError as exc:
            if "no trainable windows" not in str(exc):
                raise
            _progress(f"training_window_compare skipped name={name} reason={exc}")
            comparison_rows.append(
                {
                    "name": name,
                    "training_window_mode": mode,
                    "train_months": months,
                    "status": "skipped",
                    "skip_reason": str(exc),
                    "output_dir": str(child_output),
                }
            )
            continue
        results[name] = metrics
        fixed = metrics.get("walk_forward_fixed_trade_filter", {})
        path = metrics.get("path_aware_backtest", {})
        comparison_rows.append(
            {
                "name": name,
                "status": "completed",
                "training_window_mode": mode,
                "train_months": months,
                "sample_count": metrics.get("sample_count", 0),
                "window_count": metrics.get("window_count", 0),
                "selected_trades": fixed.get("selected_trades", 0),
                "selected_rate": fixed.get("selected_rate", 0.0),
                "selected_net_pnl": fixed.get("selected_net_pnl", 0.0),
                "selected_profit_factor": fixed.get("selected_profit_factor", 0.0),
                "selected_expectancy": fixed.get("selected_expectancy", 0.0),
                "selected_max_drawdown": fixed.get("selected_max_drawdown", 0.0),
                "path_final_equity": path.get("final_equity", ""),
                "path_net_pnl": path.get("net_pnl", ""),
                "path_max_drawdown_pct": path.get("max_drawdown_pct", ""),
                "path_trade_count": path.get("trade_count", ""),
                "path_profit_factor": path.get("profit_factor", ""),
                "output_dir": str(child_output),
            }
        )
    _write_csv(output_dir / "training_window_comparison.csv", comparison_rows)
    completed_rows = [row for row in comparison_rows if row.get("status") == "completed"]
    if not completed_rows:
        raise RuntimeError("training window comparison produced no completed runs")
    best = max(
        completed_rows,
        key=lambda row: _float(row.get("selected_expectancy"), 0.0)
        + min(_float(row.get("selected_profit_factor"), 0.0), 5.0) * 0.05
        - _float(row.get("selected_max_drawdown"), 0.0) * 0.001,
    )
    metrics = {
        "comparison": "training_windows",
        "best_name": best["name"],
        "best_output_dir": best["output_dir"],
        "runs": comparison_rows,
        "details": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def _training_window_start(first_row_time: datetime, validation_start: datetime, args: Any) -> datetime:
    mode = str(getattr(args, "training_window_mode", "expanding"))
    if mode == "expanding":
        return _month_start(first_row_time)
    candidate = _add_months(validation_start, -max(1, int(getattr(args, "train_months", 6))))
    first = _month_start(first_row_time)
    return candidate if candidate > first else first


def _weighted_train_rows(
    train_rows: list[dict[str, Any]],
    validation_start: datetime,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    copied = [dict(row) for row in train_rows]
    mode = str(getattr(args, "training_window_mode", "expanding"))
    if mode == "recency_weighted":
        half_life = max(0.25, float(getattr(args, "recency_half_life_months", 3.0)))
        minimum = max(0.01, min(1.0, float(getattr(args, "recency_weight_min", 0.25))))
        for row in copied:
            age_months = max(0.0, (validation_start - _row_time(row)).total_seconds() / (86400.0 * 30.4375))
            recency_weight = max(minimum, 0.5 ** (age_months / half_life))
            row["sample_weight"] = _float(row.get("sample_weight"), 1.0) * recency_weight
    weights = [max(0.0, _float(row.get("sample_weight"), 1.0)) for row in copied]
    if not weights:
        return copied, {"mean": 0.0, "min": 0.0, "max": 0.0}
    return copied, {
        "mean": sum(weights) / len(weights),
        "min": min(weights),
        "max": max(weights),
    }


def _run_walk_forward(rows: list[dict[str, Any]], args: Any, output_dir: Path) -> dict[str, Any]:
    rows = _prepare_model_rows(rows)
    feature_columns = _feature_columns(rows)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    first_row_time = _row_time(rows[0])
    last_row_time = _row_time(rows[-1])
    if args.test_start:
        test_start = _month_start(parse_timestamp(args.test_start))
    else:
        test_start = _add_months(_month_start(first_row_time), max(1, int(args.train_months)) + 1)
    last_month = _month_start(last_row_time)
    windows: list[dict[str, Any]] = []
    window_metrics: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
    all_test_predictions: list[dict[str, float]] = []
    model_comparison_rows: list[dict[str, Any]] = []
    _progress(
        f"walk_forward_start={test_start.isoformat()} last_month={last_month.isoformat()} "
        f"mode={getattr(args, 'training_window_mode', 'expanding')} train_months={args.train_months} "
        f"embargo_minutes={args.embargo_minutes} features={len(feature_columns)}"
    )

    current = test_start
    while current <= last_month:
        validation_start = _add_months(current, -1)
        train_start = _training_window_start(first_row_time, validation_start, args)
        test_end = _add_months(current, 1)
        embargo_cutoff = validation_start
        if int(args.embargo_minutes) > 0:
            from datetime import timedelta

            embargo_cutoff = validation_start - timedelta(minutes=max(0, int(args.embargo_minutes)))
        train_rows = [row for row in rows if train_start <= _row_time(row) < embargo_cutoff]
        train_rows, train_weight_summary = _weighted_train_rows(train_rows, validation_start, args)
        validation_rows = [row for row in rows if validation_start <= _row_time(row) < current]
        test_rows = [row for row in rows if current <= _row_time(row) < test_end]
        window_name = current.strftime("%Y_%m")
        _progress(
            f"window={window_name} train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)} "
            f"train_start={train_start.date()} weight_mean={train_weight_summary['mean']:.3f}"
        )
        if len(train_rows) < 100 or len(validation_rows) < 20 or len(test_rows) < 10 or len({int(row['good_trade_label']) for row in train_rows}) < 2:
            _progress(f"window={window_name} skipped insufficient samples or labels")
            current = test_end
            continue
        bundle = _train_model_bundle(train_rows, validation_rows, feature_columns, args.num_boost_round)
        validation_predictions = _predict_model_bundle(bundle, validation_rows, feature_columns)
        test_predictions = _predict_model_bundle(bundle, test_rows, feature_columns)
        decision_rule = _select_decision_rule(validation_rows, validation_predictions, int(args.min_selected_per_window))
        validation_selected = [_decision_allows(pred, decision_rule) for pred in validation_predictions]
        test_selected = [_decision_allows(pred, decision_rule) for pred in test_predictions]
        _progress(
            f"window={window_name} trained "
            f"auc={_classification_metrics_for_label(test_rows, [p['p_good_trade'] for p in test_predictions], 'good_trade_label', args.threshold)['auc']:.4f} "
            f"selected={sum(1 for item in test_selected if item)}/{len(test_rows)} "
            f"rule={decision_rule.get('name', 'combined')}"
        )
        model_paths: dict[str, str] = {}
        for target, model in bundle.items():
            model_file = models_dir / f"vbp_{target}_{window_name}.txt"
            model.save_model(str(model_file))
            model_paths[target] = str(model_file.relative_to(output_dir))
        _write_prediction_bundle_csv(output_dir / f"validation_predictions_{window_name}.csv", validation_rows, validation_predictions, validation_selected)
        _write_prediction_bundle_csv(output_dir / f"test_predictions_{window_name}.csv", test_rows, test_predictions, test_selected)
        validation_trade_filter = _trade_filter_metrics_from_selection(validation_rows, validation_selected)
        test_trade_filter = _trade_filter_metrics_from_selection(test_rows, test_selected)
        random_filter = _random_filter_metrics(test_rows, sum(1 for item in test_selected if item), seed=int(current.strftime("%Y%m")))
        rule_score_filter = _simple_rule_score_metrics(test_rows, sum(1 for item in test_selected if item))
        model_comparison_rows.extend(
            [
                {"window": window_name, "model": "rules_only_baseline", **_trade_filter_metrics_from_selection(test_rows, [True] * len(test_rows))},
                {"window": window_name, "model": "combined_quality_model", **test_trade_filter},
                {"window": window_name, "model": "random_same_selection_rate", **random_filter},
                {"window": window_name, "model": "simple_rule_quality_score", **rule_score_filter},
            ]
        )
        metrics = {
            "window": window_name,
            "train_start": train_start.isoformat(),
            "validation_start": validation_start.isoformat(),
            "test_start": current.isoformat(),
            "test_end": test_end.isoformat(),
            "train_count": len(train_rows),
            "validation_count": len(validation_rows),
            "test_count": len(test_rows),
            "train_weight_mean": train_weight_summary["mean"],
            "train_weight_min": train_weight_summary["min"],
            "train_weight_max": train_weight_summary["max"],
            "positive_rate_train": _label_rate(train_rows, "good_trade_label"),
            "positive_rate_validation": _label_rate(validation_rows, "good_trade_label"),
            "positive_rate_test": _label_rate(test_rows, "good_trade_label"),
            "classification": _classification_metrics_for_label(test_rows, [p["p_good_trade"] for p in test_predictions], "good_trade_label", args.threshold),
            "large_loss_classification": _classification_metrics_for_label(test_rows, [p["p_large_loss"] for p in test_predictions], "large_loss_label", 0.5),
            "false_breakout_classification": _classification_metrics_for_label(test_rows, [p["p_false_breakout"] for p in test_predictions], "false_breakout_label", 0.5),
            "validation_fixed_trade_filter": validation_trade_filter,
            "fixed_trade_filter": test_trade_filter,
            "decision_rule": decision_rule,
            "model_paths": model_paths,
        }
        window_metrics.append(metrics)
        windows.append(
            {
                "test_start": current.isoformat(),
                "test_end": test_end.isoformat(),
                "model_paths": model_paths,
                "decision_rule": decision_rule,
                "train_start": train_start.isoformat(),
                "validation_start": validation_start.isoformat(),
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "test_count": len(test_rows),
                "training_window_mode": getattr(args, "training_window_mode", "expanding"),
                "train_months": int(args.train_months),
                "train_weight_mean": train_weight_summary["mean"],
            }
        )
        all_test_rows.extend(test_rows)
        all_test_predictions.extend(test_predictions)
        current = test_end

    if not windows:
        raise RuntimeError("walk-forward produced no trainable windows")

    schedule = {
        "strategy": VBP_STRATEGY,
        "mode": "combined_quality_score_sizing",
        "threshold": float(args.threshold),
        "training_window_mode": getattr(args, "training_window_mode", "expanding"),
        "train_months": int(args.train_months),
        "recency_half_life_months": float(getattr(args, "recency_half_life_months", 3.0)),
        "recency_weight_min": float(getattr(args, "recency_weight_min", 0.25)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "baseline_policy_version": BASELINE_POLICY_VERSION,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "windows": windows,
    }
    schedule_path = output_dir / "model_schedule.json"
    schedule_path.write_text(json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "best_balanced_config.json").write_text(json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8")
    live_safe = json.loads(json.dumps(schedule))
    live_safe["live_mode"] = "score_sizing_fail_open"
    live_safe["notes"] = "First-stage VBP scoring assistant. It can be configured as reject-only or score-based sizing; it never changes direction, stop, take profit, or cost model."
    for window in live_safe.get("windows", []):
        rule = dict(window.get("decision_rule") or {})
        rule["reject_enabled"] = False
        rule["sizing_enabled"] = True
        window["decision_rule"] = rule
    (output_dir / "best_live_safe_config.json").write_text(json.dumps(live_safe, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "walk_forward_windows.csv", _flatten_window_metrics(window_metrics))
    _write_csv(output_dir / "model_metrics.csv", _flatten_window_metrics(window_metrics))
    all_test_selected = [_decision_allows(pred, _window_rule_for_row(row, windows)) for row, pred in zip(all_test_rows, all_test_predictions)]
    _write_prediction_bundle_csv(output_dir / "walk_forward_test_predictions.csv", all_test_rows, all_test_predictions, all_test_selected)
    _write_csv(output_dir / "score_decile_report.csv", _score_decile_report(all_test_rows, all_test_predictions))
    _write_csv(output_dir / "probability_distribution_by_month.csv", _probability_distribution_by_month(all_test_rows, all_test_predictions))
    _write_csv(output_dir / "calibration_report.csv", _calibration_report(all_test_rows, all_test_predictions, "p_good_trade", "good_trade_label"))
    _write_csv(output_dir / "model_comparison_report.csv", model_comparison_rows)
    _write_csv(output_dir / "threshold_coverage_report.csv", _threshold_coverage_report(window_metrics))

    metrics: dict[str, Any] = {
        "config": args.config,
        "execution_data_dir": args.execution_data_dir,
        "initial_equity": args.initial_equity,
        "sample_count": len(rows),
        "feature_count": len(feature_columns),
        "threshold": args.threshold,
        "embargo_minutes": args.embargo_minutes,
        "train_months": args.train_months,
        "training_window_mode": getattr(args, "training_window_mode", "expanding"),
        "recency_half_life_months": float(getattr(args, "recency_half_life_months", 3.0)),
        "recency_weight_min": float(getattr(args, "recency_weight_min", 0.25)),
        "window_count": len(windows),
        "first_test_start": windows[0]["test_start"],
        "last_test_end": windows[-1]["test_end"],
        "walk_forward_fixed_trade_filter": _trade_filter_metrics_from_selection(all_test_rows, all_test_selected),
        "walk_forward_classification": _classification_metrics_for_label(all_test_rows, [p["p_good_trade"] for p in all_test_predictions], "good_trade_label", args.threshold),
        "schedule_path": str(schedule_path),
        "windows": window_metrics,
    }

    if args.path_aware_backtest:
        _progress("running path-aware backtest with walk-forward model schedule")
        path_model_path = output_dir / "best_live_safe_config.json" if getattr(args, "path_aware_live_safe", False) else schedule_path
        _progress(f"path-aware model schedule={path_model_path} live_safe={int(bool(getattr(args, 'path_aware_live_safe', False)))}")
        path_result = run_execution_backtest(
            args.config,
            args.execution_data_dir,
            initial_equity=args.initial_equity,
            include_trades=False,
            compact=True,
            progress=True,
            trade_start=parse_timestamp(windows[0]["test_start"]),
            trade_end=parse_timestamp(windows[-1]["test_end"]),
            cost_experiment="full_cost",
            vbp_ml_model_path=str(path_model_path),
            vbp_ml_threshold=float(args.threshold),
            vbp_ml_mode="score_size",
        )
        (output_dir / "path_aware_backtest.json").write_text(json.dumps(path_result, indent=2, ensure_ascii=False), encoding="utf-8")
        metrics["path_aware_backtest"] = {
            "initial_equity": path_result.get("initial_equity"),
            "final_equity": path_result.get("final_equity"),
            "net_pnl": path_result.get("net_pnl"),
            "net_return_pct": path_result.get("net_return_pct"),
            "max_drawdown_pct": path_result.get("max_drawdown_pct"),
            "trade_count": path_result.get("trade_count"),
            "win_rate_pct": path_result.get("win_rate_pct"),
            "profit_factor": path_result.get("profit_factor"),
            "strategy_buckets": path_result.get("strategy_buckets"),
            "vbp_stats": path_result.get("vbp_stats"),
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_walk_forward_summary(output_dir / "summary.md", metrics, output_dir)
    return metrics


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("optimization_runs") / f"vbp_lightgbm_{stamp}"


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _config_symbols(config_path: str) -> list[str]:
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except OSError:
        return []
    return [str(symbol) for symbol in payload.get("trading", {}).get("symbols", [])]


def _event_cluster_id(symbol: str, signal_time: datetime, level: float) -> str:
    minute_bucket = signal_time.replace(second=0, microsecond=0)
    rounded_level = "nan" if not math.isfinite(level) else f"{level:.8g}"
    return f"{symbol}|LONG|{minute_bucket.isoformat()}|{rounded_level}"


def _write_dataset_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    clusters = defaultdict(int)
    months = defaultdict(int)
    for row in rows:
        clusters[str(row.get("event_cluster_id", ""))] += 1
        months[_row_time(row).strftime("%Y-%m")] += 1
    duplicate_count = sum(max(0, count - 1) for count in clusters.values())
    dataset_summary = {
        "sample_count": len(rows),
        "unique_event_count": len(clusters),
        "duplicate_candidate_count": duplicate_count,
        "average_candidates_per_event": len(rows) / max(1, len(clusters)),
        "baseline_policy_version": BASELINE_POLICY_VERSION,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "month_counts": dict(sorted(months.items())),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(
        output_dir / "event_dedup_summary.csv",
        [
            {
                "raw_candidate_count": len(rows),
                "unique_event_count": len(clusters),
                "duplicate_candidate_count": duplicate_count,
                "average_candidates_per_event": len(rows) / max(1, len(clusters)),
            }
        ],
    )
    _write_csv(
        output_dir / "label_distribution.csv",
        [
            {
                "label": label,
                "positive_rate": _label_rate(rows, label),
                "positive_count": sum(1 for row in rows if int(row.get(label, 0)) == 1),
                "sample_count": len(rows),
            }
            for label in ("good_trade_label", "large_loss_label", "false_breakout_label")
        ],
    )
    audit = [
        "# Feature leakage audit",
        "",
        "Excluded future/path/outcome columns are listed in METADATA_COLUMNS and are not used as model features.",
        "",
        "Key excluded fields: exit_price, raw_exit_price, fee, slippage_cost, funding, net_pnl, pnl_r, net_r, net_bps, MFE/MAE, bars, hold_minutes, labels, model predictions, event/version metadata.",
        "",
        "Candidate generation and full-cost shadow outcome are frozen by baseline_policy_version and cost_model_version.",
    ]
    (output_dir / "feature_leakage_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def _import_lightgbm() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    for name in ("dask", "dask.array", "dask.dataframe", "dask.distributed"):
        sys.modules[name] = None
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LightGBM is not installed. Install it with: python3 -m pip install lightgbm numpy"
        ) from exc
    return lgb


def _progress(message: str) -> None:
    print(f"[vbp-ml] {message}", file=sys.stderr, flush=True)


def _row_time(row: dict[str, Any]) -> datetime:
    return parse_timestamp(str(row["entry_time"]))


def _month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year = month_index // 12
    month = month_index % 12 + 1
    return datetime(year, month, 1)


def _flatten_window_metrics(window_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in window_metrics:
        fixed = item["fixed_trade_filter"]
        classification = item["classification"]
        rows.append(
            {
                "window": item["window"],
                "train_start": item["train_start"],
                "test_start": item["test_start"],
                "test_end": item["test_end"],
                "train_count": item["train_count"],
                "validation_count": item.get("validation_count", 0),
                "test_count": item["test_count"],
                "train_weight_mean": item.get("train_weight_mean", 0.0),
                "train_weight_min": item.get("train_weight_min", 0.0),
                "train_weight_max": item.get("train_weight_max", 0.0),
                "positive_rate_train": item["positive_rate_train"],
                "positive_rate_validation": item.get("positive_rate_validation", 0.0),
                "positive_rate_test": item["positive_rate_test"],
                "auc": classification["auc"],
                "precision": classification["precision"],
                "recall": classification["recall"],
                "baseline_net_pnl": fixed["baseline_net_pnl"],
                "baseline_profit_factor": fixed["baseline_profit_factor"],
                "selected_trades": fixed["selected_trades"],
                "selected_rate": fixed["selected_rate"],
                "selected_net_pnl": fixed["selected_net_pnl"],
                "selected_profit_factor": fixed["selected_profit_factor"],
                "rejected_net_pnl": fixed["rejected_net_pnl"],
                "decision_rule": json.dumps(item.get("decision_rule", {}), sort_keys=True),
            }
        )
    return rows


def _single_split_metrics(
    args: Any,
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_pred: list[float],
    test_pred: list[float],
    feature_columns: list[str],
) -> dict[str, Any]:
    sweep = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        item = _trade_filter_metrics(test_rows, test_pred, threshold)
        item["threshold"] = threshold
        sweep.append(item)
    return {
        "config": args.config,
        "execution_data_dir": args.execution_data_dir,
        "initial_equity": args.initial_equity,
        "sample_count": len(rows),
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "feature_count": len(feature_columns),
        "positive_rate_train": _positive_rate(train_rows),
        "positive_rate_test": _positive_rate(test_rows),
        "threshold": args.threshold,
        "train": _classification_metrics(train_rows, train_pred, args.threshold),
        "test": _classification_metrics(test_rows, test_pred, args.threshold),
        "train_trade_filter": _trade_filter_metrics(train_rows, train_pred, args.threshold),
        "test_trade_filter": _trade_filter_metrics(test_rows, test_pred, args.threshold),
        "threshold_sweep_test": sweep,
    }


def _write_walk_forward_summary(path: Path, metrics: dict[str, Any], output_dir: Path) -> None:
    fixed = metrics["walk_forward_fixed_trade_filter"]
    clf = metrics["walk_forward_classification"]
    path_aware = metrics.get("path_aware_backtest")
    path_text = ""
    if path_aware:
        path_text = f"""
## Path-aware backtest

- final equity: {path_aware['final_equity']:.4f}
- net pnl: {path_aware['net_pnl']:.4f}
- net return pct: {path_aware['net_return_pct']:.4f}
- max drawdown pct: {path_aware['max_drawdown_pct']:.4f}
- trade count: {path_aware['trade_count']}
- win rate pct: {path_aware['win_rate_pct']:.4f}
- profit factor: {path_aware['profit_factor']}
"""
    text = f"""# VBP LightGBM walk-forward

This run keeps the original VBP signal policy frozen and trains only a reject-only meta-filter.
Thresholds and decision rules are selected on validation months; test months are evaluation-only.

- samples: {metrics['sample_count']}
- windows: {metrics['window_count']}
- training window mode: {metrics.get('training_window_mode', 'expanding')}
- train months: {metrics['train_months']}
- recency half-life months: {metrics.get('recency_half_life_months', 0)}
- recency weight min: {metrics.get('recency_weight_min', 0)}
- embargo minutes: {metrics.get('embargo_minutes', 0)}
- first test: {metrics['first_test_start']}
- last test: {metrics['last_test_end']}
- threshold: {metrics['threshold']}

## Fixed candidate-list diagnostics

- baseline candidates: {int(fixed['baseline_trades'])}
- baseline net pnl: {fixed['baseline_net_pnl']:.4f}
- baseline PF: {fixed['baseline_profit_factor']:.4f}
- selected candidates: {int(fixed['selected_trades'])}
- selected rate: {fixed['selected_rate']:.4f}
- selected net pnl: {fixed['selected_net_pnl']:.4f}
- selected PF: {fixed['selected_profit_factor']:.4f}
- rejected net pnl: {fixed['rejected_net_pnl']:.4f}

## Classification

- auc: {clf['auc']:.4f}
- precision: {clf['precision']:.4f}
- recall: {clf['recall']:.4f}
- accuracy: {clf['accuracy']:.4f}
{path_text}
## Files

- dataset: `{output_dir / 'vbp_ml_dataset.csv'}`
- schedule: `{output_dir / 'model_schedule.json'}`
- windows: `{output_dir / 'walk_forward_windows.csv'}`
- predictions: `{output_dir / 'walk_forward_test_predictions.csv'}`
- score deciles: `{output_dir / 'score_decile_report.csv'}`
- probability distribution: `{output_dir / 'probability_distribution_by_month.csv'}`
- calibration: `{output_dir / 'calibration_report.csv'}`
- model comparison: `{output_dir / 'model_comparison_report.csv'}`
- threshold coverage: `{output_dir / 'threshold_coverage_report.csv'}`
- metrics: `{output_dir / 'metrics.json'}`
- balanced config: `{output_dir / 'best_balanced_config.json'}`
- live safe config: `{output_dir / 'best_live_safe_config.json'}`
"""
    path.write_text(text, encoding="utf-8")


def _aggregate_vbp_positions(trades: list[dict[str, Any]], label_min_r: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        if trade.get("strategy") != VBP_STRATEGY:
            continue
        if str(trade.get("side") or trade.get("direction")) != "LONG":
            continue
        entry_reason = str(trade.get("entry_reason", ""))
        if "vbp_volume_breakout_pullback" not in entry_reason:
            continue
        key = (str(trade.get("symbol")), str(trade.get("entry_time")), entry_reason)
        grouped[key].append(trade)

    rows: list[dict[str, Any]] = []
    for (symbol, entry_time, entry_reason), parts in grouped.items():
        parts.sort(key=lambda part: str(part.get("exit_time", "")))
        entry_price = _float(parts[0].get("entry_price"))
        notional = sum(abs(_float(part.get("notional"))) for part in parts)
        net_pnl = sum(_float(part.get("net_pnl")) for part in parts)
        gross_pnl = sum(_float(part.get("gross_pnl")) for part in parts)
        fee = sum(_float(part.get("fee", part.get("fees"))) for part in parts)
        slippage = sum(_float(part.get("slippage_cost")) for part in parts)
        funding = sum(_float(part.get("funding")) for part in parts)
        stop_pct = _reason_float(entry_reason, "stop", 0.0) / 100.0
        risk_usdt = max(notional * stop_pct, 1e-12)
        pnl_r = net_pnl / risk_usdt
        rows.append(
            {
                "symbol": symbol,
                "strategy": VBP_STRATEGY,
                "side": "LONG",
                "entry_time": entry_time,
                "exit_time": str(parts[-1].get("exit_time", "")),
                "entry_price": entry_price,
                "exit_price": _float(parts[-1].get("exit_price")),
                "entry_reason": entry_reason,
                "exit_reason": "|".join(sorted({str(part.get("exit_reason", "")) for part in parts if part.get("exit_reason")})),
                "net_pnl": net_pnl,
                "gross_pnl": gross_pnl,
                "fee": fee,
                "slippage_cost": slippage,
                "funding": funding,
                "notional": notional,
                "return_pct": net_pnl / max(notional, 1e-12),
                "pnl_r": pnl_r,
                "mfe": max(_float(part.get("mfe")) for part in parts),
                "mae": min(_float(part.get("mae")) for part in parts),
                "hold_minutes": max(_float(part.get("hold_minutes", part.get("avg_hold_minutes"))) for part in parts),
                "label": 1 if pnl_r >= label_min_r and net_pnl > 0 else 0,
            }
        )
    return rows


def _load_symbol_caches(data_dir: str, symbols: list[str], progress: bool = False) -> dict[str, SymbolFeatureCache]:
    root = Path(data_dir)
    caches: dict[str, SymbolFeatureCache] = {}
    total = len(symbols)
    for offset, symbol in enumerate(symbols, start=1):
        matches = sorted(root.glob(f"{symbol}_1m_*.csv"))
        if not matches:
            if progress:
                _progress(f"load_candles {offset}/{total} {symbol} missing")
            continue
        candles = load_candles_csv(matches[-1])
        caches[symbol] = _feature_cache(candles)
        if progress and (offset == 1 or offset == total or offset % 5 == 0):
            _progress(f"load_candles {offset}/{total} {symbol} rows={len(candles)} cached={len(caches)}")
    return caches


def _read_dataset_csv(path: Path) -> list[dict[str, Any]]:
    text_columns = {
        "symbol",
        "strategy",
        "side",
        "direction",
        "entry_time",
        "exit_time",
        "entry_reason",
        "exit_reason",
        "strategy_bucket",
        "reason",
        "signal_time",
            "signal_available_time",
            "skip_reason",
            "event_cluster_id",
            "baseline_policy_version",
            "strategy_config_hash",
            "feature_version",
            "label_version",
            "cost_model_version",
        }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_columns:
                    row[key] = value
                else:
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
                    else:
                        if key == "label":
                            row[key] = int(number)
                        else:
                            row[key] = number
            rows.append(row)
    return rows


def _build_dataset_rows(
    positions: list[dict[str, Any]],
    caches: dict[str, SymbolFeatureCache],
    strategy_config_hash: str = "",
    market_breadth_cache: MarketBreadthRuntimeCache | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    btc = caches.get("BTCUSDT")
    rows: list[dict[str, Any]] = []
    total = len(positions)
    for offset, position in enumerate(positions, start=1):
        symbol = str(position["symbol"])
        cache = caches.get(symbol)
        if cache is None:
            if progress and offset % 500 == 0:
                _progress(f"build_features {offset}/{total} rows={len(rows)} skipped_missing_cache")
            continue
        entry_time = parse_timestamp(str(position["entry_time"]))
        index = bisect.bisect_right(cache.timestamps, entry_time) - 1
        if index < 200:
            if progress and offset % 500 == 0:
                _progress(f"build_features {offset}/{total} rows={len(rows)} skipped_warmup")
            continue
        entry_reason = str(position["entry_reason"])
        signal_time = parse_timestamp(str(position.get("signal_time") or position.get("entry_time")))
        breakout_index = bisect.bisect_right(cache.timestamps, signal_time) - 1
        level = _reason_float(entry_reason, "level", math.nan)
        bottom = _reason_float(entry_reason, "bottom", math.nan)
        target = _reason_float(entry_reason, "target", math.nan)
        tp1 = _reason_float(entry_reason, "tp1", math.nan)
        stop_pct = _reason_float(entry_reason, "stop", math.nan) / 100.0
        row = dict(position)
        entry_price = _float(position["entry_price"])
        net_r = _float(row.get("pnl_r", row.get("net_r", 0.0)))
        mfe_r, mae_r = _r_mfe_mae(row, stop_pct)
        row.update(
            {
                "baseline_policy_version": BASELINE_POLICY_VERSION,
                "strategy_config_hash": strategy_config_hash,
                "feature_version": FEATURE_VERSION,
                "label_version": LABEL_VERSION,
                "cost_model_version": COST_MODEL_VERSION,
                "event_cluster_id": _event_cluster_id(symbol, signal_time, level),
                "entry_hour": float(entry_time.hour),
                "entry_weekday": float(entry_time.weekday()),
                "hour_sin": math.sin(2.0 * math.pi * entry_time.hour / 24.0),
                "hour_cos": math.cos(2.0 * math.pi * entry_time.hour / 24.0),
                "vbp_level": level,
                "vbp_bottom": bottom,
                "vbp_target": target,
                "vbp_tp1": tp1,
                "vbp_stop_pct": stop_pct,
                "entry_to_level_pct": _safe_div(entry_price - level, level),
                "entry_to_bottom_pct": _safe_div(entry_price - bottom, bottom),
                "entry_to_target_pct": _safe_div(entry_price - target, target),
                "tp1_distance_pct": _safe_div(tp1 - entry_price, entry_price),
                "net_r": net_r,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
            }
        )
        row.update(_symbol_features(cache, index, prefix="symbol"))
        if breakout_index >= 120 and breakout_index <= index:
            row.update(_vbp_specific_features(cache, breakout_index, index, level, bottom, target, entry_price))
        row.update(_market_breadth_features_fast(caches, market_breadth_cache, entry_time))
        if btc is not None:
            btc_index = bisect.bisect_right(btc.timestamps, entry_time) - 1
            if btc_index >= 200:
                row.update(_symbol_features(btc, btc_index, prefix="btc"))
                row["symbol_minus_btc_ret_1h"] = row.get("symbol_ret_60m", math.nan) - row.get("btc_ret_60m", math.nan)
                row["symbol_minus_btc_ret_4h"] = row.get("symbol_ret_240m", math.nan) - row.get("btc_ret_240m", math.nan)
        _apply_label_targets(row)
        rows.append(row)
        if progress and (offset == 1 or offset == total or offset % 250 == 0):
            _progress(f"build_features {offset}/{total} rows={len(rows)} latest={symbol}")
    return rows


def _symbol_features(cache: SymbolFeatureCache, index: int, prefix: str) -> dict[str, float]:
    close = cache.close[index]
    high = cache.high[index]
    low = cache.low[index]
    open_price = cache.open[index]
    candle_range = max(high - low, 1e-12)
    sma7 = _prefix_mean(cache.close_prefix, index, 7)
    sma25 = _prefix_mean(cache.close_prefix, index, 25)
    sma60 = _prefix_mean(cache.close_prefix, index, 60)
    sma99 = _prefix_mean(cache.close_prefix, index, 99)
    ema9 = _ema_at(cache.close, index, 9)
    ema21 = _ema_at(cache.close, index, 21)
    atr14 = _atr_at(cache.high, cache.low, cache.close, index, 14)
    rsi6 = _rsi_at(cache.close, index, 6)
    rsi14 = _rsi_at(cache.close, index, 14)
    macd_hist = _macd_hist_at(cache.close, index)
    macd_hist_prev = _macd_hist_at(cache.close, max(0, index - 3))
    high_1d = cache.high_1d[index]
    low_1d = cache.low_1d[index]
    high_7d = cache.high_7d[index]
    low_7d = cache.low_7d[index]
    high_30d = cache.high_30d[index]
    low_30d = cache.low_30d[index]
    return {
        f"{prefix}_ret_5m": _ret(cache.close, index, 5),
        f"{prefix}_ret_15m": _ret(cache.close, index, 15),
        f"{prefix}_ret_30m": _ret(cache.close, index, 30),
        f"{prefix}_ret_60m": _ret(cache.close, index, 60),
        f"{prefix}_ret_240m": _ret(cache.close, index, 240),
        f"{prefix}_ret_1440m": _ret(cache.close, index, 1440),
        f"{prefix}_atr14_pct": _safe_div(atr14, close),
        f"{prefix}_volume_ratio_20": _safe_div(cache.volume[index], _prefix_mean(cache.volume_prefix, index, 20)),
        f"{prefix}_volume_ratio_60": _safe_div(cache.volume[index], _prefix_mean(cache.volume_prefix, index, 60)),
        f"{prefix}_range_pct": _safe_div(candle_range, close),
        f"{prefix}_close_position": _safe_div(close - low, candle_range),
        f"{prefix}_upper_wick_ratio": _safe_div(high - max(open_price, close), candle_range),
        f"{prefix}_lower_wick_ratio": _safe_div(min(open_price, close) - low, candle_range),
        f"{prefix}_body_pct": _safe_div(close - open_price, open_price),
        f"{prefix}_dist_sma7": _safe_div(close - sma7, close),
        f"{prefix}_dist_sma25": _safe_div(close - sma25, close),
        f"{prefix}_dist_sma60": _safe_div(close - sma60, close),
        f"{prefix}_dist_sma99": _safe_div(close - sma99, close),
        f"{prefix}_dist_ema9": _safe_div(close - ema9, close),
        f"{prefix}_dist_ema21": _safe_div(close - ema21, close),
        f"{prefix}_rsi6": rsi6,
        f"{prefix}_rsi14": rsi14,
        f"{prefix}_macd_hist": macd_hist,
        f"{prefix}_macd_hist_slope_3": macd_hist - macd_hist_prev,
        f"{prefix}_range_pos_1d": _range_position(close, low_1d, high_1d),
        f"{prefix}_range_pos_7d": _range_position(close, low_7d, high_7d),
        f"{prefix}_range_pos_30d": _range_position(close, low_30d, high_30d),
    }


def _vbp_specific_features(
    cache: SymbolFeatureCache,
    breakout_index: int,
    entry_index: int,
    level: float,
    bottom: float,
    target: float,
    entry_price: float,
) -> dict[str, float]:
    breakout_close = cache.close[breakout_index]
    breakout_open = cache.open[breakout_index]
    breakout_high = cache.high[breakout_index]
    breakout_low = cache.low[breakout_index]
    breakout_volume = cache.volume[breakout_index]
    breakout_range = max(breakout_high - breakout_low, 1e-12)
    atr = _atr_at(cache.high, cache.low, cache.close, breakout_index, 14)
    pre_start = max(0, breakout_index - 48)
    pre_high = max(cache.high[pre_start:breakout_index] or [breakout_high])
    pre_low = min(cache.low[pre_start:breakout_index] or [breakout_low])
    pre_range = max(pre_high - pre_low, 1e-12)
    pre_volume = _prefix_mean(cache.volume_prefix, max(0, breakout_index - 1), min(48, max(1, breakout_index - pre_start)))
    recent_volume = _prefix_mean(cache.volume_prefix, max(0, breakout_index - 1), min(12, max(1, breakout_index)))
    pullback_start = min(len(cache.close) - 1, breakout_index + 1)
    pullback_end = max(pullback_start, entry_index)
    pullback_lows = cache.low[pullback_start:pullback_end + 1]
    pullback_volumes = cache.volume[pullback_start:pullback_end + 1]
    pullback_low = min(pullback_lows or [entry_price])
    pullback_volume_avg = sum(pullback_volumes) / max(1, len(pullback_volumes))
    entry_close = cache.close[entry_index]
    entry_open = cache.open[entry_index]
    entry_high = cache.high[entry_index]
    entry_low = cache.low[entry_index]
    entry_range = max(entry_high - entry_low, 1e-12)
    sma7 = _prefix_mean(cache.close_prefix, entry_index, 7)
    sma25 = _prefix_mean(cache.close_prefix, entry_index, 25)
    sma99 = _prefix_mean(cache.close_prefix, entry_index, 99)
    ema9 = _ema_at(cache.close, entry_index, 9)
    ema21 = _ema_at(cache.close, entry_index, 21)
    previous_15m = _ret(cache.close, max(0, breakout_index - 1), 15)
    previous_1h = _ret(cache.close, max(0, breakout_index - 1), 60)
    consecutive_green = 0
    for current in range(max(0, breakout_index - 12), breakout_index + 1):
        if cache.close[current] > cache.open[current]:
            consecutive_green += 1
        else:
            consecutive_green = 0
    inside_range = sum(1 for current in range(pre_start, breakout_index) if pre_low <= cache.close[current] <= pre_high)
    failed_breakouts = sum(1 for current in range(pre_start, breakout_index) if cache.high[current] > level and cache.close[current] <= level)
    return {
        "breakout_distance_pct": _safe_div(breakout_close - level, level),
        "breakout_distance_atr": _safe_div(breakout_close - level, atr),
        "breakout_body_atr": _safe_div(breakout_close - breakout_open, atr),
        "breakout_range_atr": _safe_div(breakout_range, atr),
        "breakout_close_position": _safe_div(breakout_close - breakout_low, breakout_range),
        "breakout_upper_wick_ratio": _safe_div(breakout_high - max(breakout_open, breakout_close), breakout_range),
        "breakout_lower_wick_ratio": _safe_div(min(breakout_open, breakout_close) - breakout_low, breakout_range),
        "distance_from_breakout_level_atr": _safe_div(entry_price - level, atr),
        "distance_from_vbp_target_atr": _safe_div(entry_price - target, atr),
        "channel_width_atr": _safe_div(pre_range, atr),
        "channel_width_pct": _safe_div(pre_range, breakout_close),
        "pre_breakout_range_atr": _safe_div(pre_range, atr),
        "pre_breakout_volatility": _safe_div(pre_range, pre_low),
        "pre_breakout_volatility_compression": _safe_div(pre_range, max(cache.high_1d[breakout_index] - cache.low_1d[breakout_index], 1e-12)),
        "pre_breakout_volume_contraction": _safe_div(recent_volume, pre_volume),
        "pre_breakout_bars_inside_range": float(inside_range),
        "breakout_attempt_count": float(failed_breakouts + 1),
        "previous_failed_breakout_count": float(failed_breakouts),
        "pullback_depth_atr": _safe_div(breakout_close - pullback_low, atr),
        "pullback_depth_pct": _safe_div(breakout_close - pullback_low, breakout_close),
        "pullback_depth_vs_breakout_distance": _safe_div(breakout_close - pullback_low, breakout_close - level),
        "pullback_bars": float(max(0, entry_index - breakout_index)),
        "pullback_duration_minutes": float(max(0, entry_index - breakout_index)),
        "pullback_volume_vs_breakout_volume": _safe_div(pullback_volume_avg, breakout_volume),
        "pullback_volume_contraction": _safe_div(pullback_volume_avg, pre_volume),
        "pullback_low_vs_vbp_level": _safe_div(pullback_low - level, level),
        "pullback_close_vs_vbp_level": _safe_div(entry_close - level, level),
        "pullback_broke_breakout_level": 1.0 if pullback_low < level else 0.0,
        "pullback_broke_bottom": 1.0 if pullback_low < bottom else 0.0,
        "pullback_lower_wick_ratio": _safe_div(min(entry_open, entry_close) - entry_low, entry_range),
        "confirmation_body_atr": _safe_div(entry_close - entry_open, atr),
        "confirmation_close_position": _safe_div(entry_close - entry_low, entry_range),
        "confirmation_upper_wick_ratio": _safe_div(entry_high - max(entry_open, entry_close), entry_range),
        "confirmation_lower_wick_ratio": _safe_div(min(entry_open, entry_close) - entry_low, entry_range),
        "confirmation_volume_ratio": _safe_div(cache.volume[entry_index], pre_volume),
        "confirmation_close_vs_breakout_level": _safe_div(entry_close - level, level),
        "confirmation_close_vs_ema9": _safe_div(entry_close - ema9, entry_close),
        "confirmation_close_vs_ema21": _safe_div(entry_close - ema21, entry_close),
        "confirmation_chase_pct": _safe_div(entry_close - level, level),
        "time_since_breakout_minutes": float(max(0, entry_index - breakout_index)),
        "return_before_breakout_15m": previous_15m,
        "return_before_breakout_1h": previous_1h,
        "consecutive_green_bars": float(consecutive_green),
        "distance_from_ma7_atr": _safe_div(entry_close - sma7, atr),
        "distance_from_ma25_atr": _safe_div(entry_close - sma25, atr),
        "distance_from_ma99_atr": _safe_div(entry_close - sma99, atr),
        "rsi_overheat_distance": max(0.0, _rsi_at(cache.close, entry_index, 6) - 80.0),
        "price_already_extended_flag": 1.0 if _ret(cache.close, entry_index, 60) > 0.06 else 0.0,
    }


def _market_breadth_features(caches: dict[str, SymbolFeatureCache], entry_time: datetime) -> dict[str, float]:
    ret_5m: list[float] = []
    ret_15m: list[float] = []
    ret_1h: list[float] = []
    above_ema21 = 0
    above_ma99 = 0
    simultaneous_breakouts = 0
    simultaneous_candidates = 0
    major_up = 0
    major_total = 0
    for symbol, cache in caches.items():
        index = bisect.bisect_right(cache.timestamps, entry_time) - 1
        if index < 100:
            continue
        r5 = _ret(cache.close, index, 5)
        r15 = _ret(cache.close, index, 15)
        r60 = _ret(cache.close, index, 60)
        if math.isfinite(r5):
            ret_5m.append(r5)
        if math.isfinite(r15):
            ret_15m.append(r15)
        if math.isfinite(r60):
            ret_1h.append(r60)
        close = cache.close[index]
        if close > _ema_at(cache.close, index, 21):
            above_ema21 += 1
        if close > _prefix_mean(cache.close_prefix, index, 99):
            above_ma99 += 1
        if index >= 48 and close > max(cache.high[index - 48:index]):
            simultaneous_breakouts += 1
        if index >= 48 and close >= max(cache.high[index - 48:index]) * 0.995:
            simultaneous_candidates += 1
        if symbol in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}:
            major_total += 1
            if r15 > 0 and r60 > 0:
                major_up += 1
    total = max(1, len(ret_1h))
    top10 = sorted(ret_1h, reverse=True)[:10]
    return {
        "market_positive_return_ratio_5m": sum(1 for value in ret_5m if value > 0) / max(1, len(ret_5m)),
        "market_positive_return_ratio_15m": sum(1 for value in ret_15m if value > 0) / max(1, len(ret_15m)),
        "market_positive_return_ratio_1h": sum(1 for value in ret_1h if value > 0) / total,
        "market_above_ema21_ratio": above_ema21 / total,
        "market_above_ma99_ratio": above_ma99 / total,
        "market_average_return_15m": sum(ret_15m) / max(1, len(ret_15m)),
        "market_average_return_1h": sum(ret_1h) / total,
        "market_median_return_15m": _median(ret_15m),
        "simultaneous_breakout_count": float(simultaneous_breakouts),
        "simultaneous_vbp_candidate_count": float(simultaneous_candidates),
        "top10_market_return": sum(top10) / max(1, len(top10)),
        "market_breadth_change": (sum(1 for value in ret_5m if value > 0) / max(1, len(ret_5m))) - (sum(1 for value in ret_1h if value > 0) / total),
        "major_coins_alignment": major_up / max(1, major_total),
    }


def _market_breadth_features_fast(
    caches: dict[str, SymbolFeatureCache],
    runtime_cache: MarketBreadthRuntimeCache | None,
    entry_time: datetime,
) -> dict[str, float]:
    if runtime_cache is not None:
        return runtime_cache.features_at(entry_time)
    if len(caches) <= 2:
        return _empty_market_breadth_features()
    return _market_breadth_features(caches, entry_time)


def _empty_market_breadth_features() -> dict[str, float]:
    return {
        "market_positive_return_ratio_5m": 0.0,
        "market_positive_return_ratio_15m": 0.0,
        "market_positive_return_ratio_1h": 0.0,
        "market_above_ema21_ratio": 0.0,
        "market_above_ma99_ratio": 0.0,
        "market_average_return_15m": 0.0,
        "market_average_return_1h": 0.0,
        "market_median_return_15m": 0.0,
        "simultaneous_breakout_count": 0.0,
        "simultaneous_vbp_candidate_count": 0.0,
        "top10_market_return": 0.0,
        "market_breadth_change": 0.0,
        "major_coins_alignment": 0.0,
    }


def _ret_candles(candles: list[Candle], index: int, bars: int) -> float:
    if index < bars or index >= len(candles):
        return math.nan
    previous = candles[index - bars].close
    return _safe_div(candles[index].close - previous, previous)


def _sma_candles_at(candles: list[Candle], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    window = candles[start:index + 1]
    return sum(candle.close for candle in window) / max(1, len(window))


def _ema_candles_at(candles: list[Candle], index: int, period: int) -> float:
    start = max(0, index - period * 10)
    alpha = 2.0 / (period + 1.0)
    value = candles[start].close
    for candle in candles[start + 1:index + 1]:
        value = alpha * candle.close + (1.0 - alpha) * value
    return value


def _split_rows(rows: list[dict[str, Any]], test_start: str | None, test_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_start:
        cutoff = parse_timestamp(test_start)
        train = [row for row in rows if parse_timestamp(str(row["entry_time"])) < cutoff]
        test = [row for row in rows if parse_timestamp(str(row["entry_time"])) >= cutoff]
        return train, test
    ratio = min(0.8, max(0.05, test_ratio))
    split = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - ratio))))
    return rows[:split], rows[split:]


def _feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in METADATA_COLUMNS:
                continue
            if isinstance(value, (bool, int, float)):
                columns.add(key)
    return sorted(columns)


def _prepare_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        stop_pct = _float(row.get("vbp_stop_pct"), _reason_float(str(row.get("entry_reason", "")), "stop", 0.0) / 100.0)
        row["net_r"] = _float(row.get("net_r", row.get("pnl_r")))
        mfe_r, mae_r = _r_mfe_mae(row, stop_pct)
        row["mfe_r"] = mfe_r
        row["mae_r"] = mae_r
        _apply_label_targets(row)
        if not row.get("event_cluster_id"):
            row["event_cluster_id"] = _event_cluster_id(str(row.get("symbol", "")), _row_time(row), _float(row.get("vbp_level", math.nan)))
    _downweight_duplicate_events(rows)
    return rows


def _apply_label_targets(row: dict[str, Any]) -> None:
    net_r = _float(row.get("net_r", row.get("pnl_r")))
    stop_pct = _float(row.get("vbp_stop_pct"), _reason_float(str(row.get("entry_reason", "")), "stop", 0.0) / 100.0)
    mfe_r, mae_r = _r_mfe_mae(row, stop_pct)
    exit_reason = str(row.get("exit_reason") or row.get("reason") or "")
    row["net_r"] = net_r
    row["mfe_r"] = mfe_r
    row["mae_r"] = mae_r
    row["expected_net_r_target"] = net_r
    row["expected_mfe_r_target"] = mfe_r
    row["expected_mae_r_target"] = mae_r
    row["good_trade_label"] = 1 if net_r >= 0.25 and mfe_r >= 0.75 and mae_r <= 1.25 else 0
    row["large_loss_label"] = 1 if net_r <= -0.70 or "stop_loss" in exit_reason or (mae_r >= 0.85 and mfe_r < 0.35) else 0
    broke_level = _float(row.get("pullback_broke_breakout_level"), 0.0) > 0
    row["false_breakout_label"] = 1 if broke_level or net_r <= -0.35 or ("stop_loss" in exit_reason and mfe_r < 0.50) else 0
    if "sample_weight" not in row:
        row["sample_weight"] = 0.35 if -0.15 < net_r < 0.20 else 1.0
    row["label"] = int(row["good_trade_label"])


def _r_mfe_mae(row: dict[str, Any], stop_pct: float) -> tuple[float, float]:
    stop = max(stop_pct, 1e-6)
    mfe = _float(row.get("mfe"))
    mae = _float(row.get("mae"))
    return max(0.0, mfe / stop), abs(min(0.0, mae) / stop)


def _downweight_duplicate_events(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("event_cluster_id", ""))] += 1
    for row in rows:
        count = max(1, counts.get(str(row.get("event_cluster_id", "")), 1))
        if count > 1:
            row["sample_weight"] = _float(row.get("sample_weight"), 1.0) / math.sqrt(count)


def _train_model_bundle(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_columns: list[str],
    num_boost_round: int,
) -> dict[str, Any]:
    return {
        "p_good_trade": _train_lgbm_target(train_rows, validation_rows, feature_columns, "good_trade_label", "binary", num_boost_round),
        "expected_net_r": _train_lgbm_target(train_rows, validation_rows, feature_columns, "expected_net_r_target", "regression", num_boost_round),
        "p_large_loss": _train_lgbm_target(train_rows, validation_rows, feature_columns, "large_loss_label", "binary", num_boost_round),
        "p_false_breakout": _train_lgbm_target(train_rows, validation_rows, feature_columns, "false_breakout_label", "binary", num_boost_round),
        "expected_mfe_r": _train_lgbm_target(train_rows, validation_rows, feature_columns, "expected_mfe_r_target", "regression", num_boost_round),
        "expected_mae_r": _train_lgbm_target(train_rows, validation_rows, feature_columns, "expected_mae_r_target", "regression", num_boost_round),
    }


def _train_lgbm_target(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    feature_columns: list[str],
    target: str,
    objective: str,
    num_boost_round: int,
) -> Any:
    lgb = _import_lightgbm()
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is not installed. Install it with: python3 -m pip install numpy") from exc
    x_train = np.array([[_number(row.get(column)) for column in feature_columns] for row in train_rows], dtype=float)
    y_train = np.array([_float(row.get(target)) for row in train_rows], dtype=float)
    x_valid = np.array([[_number(row.get(column)) for column in feature_columns] for row in validation_rows], dtype=float)
    y_valid = np.array([_float(row.get(target)) for row in validation_rows], dtype=float)
    weights = np.array([max(0.05, _float(row.get("sample_weight"), 1.0)) for row in train_rows], dtype=float)
    effective_objective = objective
    if objective == "binary" and len(set(float(value) for value in y_train)) < 2:
        effective_objective = "regression"
    params = {
        "objective": effective_objective,
        "metric": "binary_logloss" if effective_objective == "binary" else "l2",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 45,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(x_train, label=y_train, weight=weights, feature_name=feature_columns)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, feature_name=feature_columns)
    return lgb.train(
        params,
        train_set,
        num_boost_round=max(30, num_boost_round),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )


def _predict_model_bundle(models: dict[str, Any], rows: list[dict[str, Any]], feature_columns: list[str]) -> list[dict[str, float]]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is not installed. Install it with: python3 -m pip install numpy") from exc
    if not rows:
        return []
    matrix = np.array([[_number(row.get(column)) for column in feature_columns] for row in rows], dtype=float)
    raw: dict[str, list[float]] = {}
    for target, model in models.items():
        raw[target] = [float(value) for value in model.predict(matrix)]
    predictions: list[dict[str, float]] = []
    for index in range(len(rows)):
        item = {target: raw[target][index] for target in raw}
        for probability_key in ("p_good_trade", "p_large_loss", "p_false_breakout"):
            if probability_key in item:
                item[probability_key] = max(0.0, min(1.0, item[probability_key]))
        item["quality_score"] = _quality_score(item)
        predictions.append(item)
    return predictions


def _train_lightgbm(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_columns: list[str],
    num_boost_round: int,
) -> tuple[Any, list[float], list[float]]:
    lgb = _import_lightgbm()
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is not installed. Install it with: python3 -m pip install numpy") from exc

    x_train = np.array([[_number(row.get(column)) for column in feature_columns] for row in train_rows], dtype=float)
    y_train = np.array([int(row["label"]) for row in train_rows], dtype=int)
    x_test = np.array([[_number(row.get(column)) for column in feature_columns] for row in test_rows], dtype=float)
    y_test = np.array([int(row["label"]) for row in test_rows], dtype=int)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 0.5,
        "verbosity": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_columns)
    valid_set = lgb.Dataset(x_test, label=y_test, reference=train_set, feature_name=feature_columns)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=max(20, num_boost_round),
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    train_pred = [float(value) for value in model.predict(x_train)]
    test_pred = [float(value) for value in model.predict(x_test)]
    return model, train_pred, test_pred


def _classification_metrics(rows: list[dict[str, Any]], probabilities: list[float], threshold: float) -> dict[str, float]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 1)
    tn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 0)
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 1)
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 0)
    total = max(1, len(rows))
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "auc": _auc(y_true, probabilities),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _classification_metrics_for_label(rows: list[dict[str, Any]], probabilities: list[float], label_key: str, threshold: float) -> dict[str, float]:
    y_true = [int(row.get(label_key, 0)) for row in rows]
    y_pred = [1 if probability >= threshold else 0 for probability in probabilities]
    tp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 1)
    tn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 0)
    fp = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 0 and pred == 1)
    fn = sum(1 for actual, pred in zip(y_true, y_pred) if actual == 1 and pred == 0)
    total = max(1, len(rows))
    return {
        "accuracy": (tp + tn) / total,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "auc": _auc(y_true, probabilities),
        "brier": sum((prob - actual) ** 2 for prob, actual in zip(probabilities, y_true)) / total,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_decision_rule(rows: list[dict[str, Any]], predictions: list[dict[str, float]], min_selected: int) -> dict[str, Any]:
    if not rows or not predictions:
        return _fallback_decision_rule()
    candidates: list[dict[str, Any]] = []
    scores = [pred["quality_score"] for pred in predictions]
    risk_floor = _quantile(scores, 0.20)
    risk_neutral = _quantile(scores, 0.50)
    risk_high = _quantile(scores, 0.80)
    for percentile in (0.10, 0.20, 0.30, 0.40, 0.50):
        score_min = _quantile(scores, 1.0 - percentile)
        for net_min in (-0.05, 0.0, 0.05, 0.10, 0.20):
            for large_max in (0.35, 0.45, 0.55, 0.65):
                for false_max in (0.35, 0.45, 0.55, 0.65):
                    for mae_max in (1.25, 1.50, 2.00, 3.00):
                        rule = {
                            "name": f"top_{int(percentile * 100)}_net_{net_min:g}",
                            "quality_score_min": score_min,
                            "expected_net_r_min": net_min,
                            "p_large_loss_max": large_max,
                            "p_false_breakout_max": false_max,
                            "expected_mae_r_max": mae_max,
                            "expected_mfe_r_min": 0.20,
                            "p_good_trade_min": 0.0,
                            "reject_enabled": True,
                            "sizing_enabled": True,
                            "soft_filter_enabled": True,
                            "reject_score_min": risk_floor,
                            "hard_expected_net_r_min": -0.25,
                            "hard_large_loss_max": 0.70,
                            "hard_false_breakout_max": 0.70,
                            "hard_expected_mae_r_max": 3.0,
                            "risk_score_floor": risk_floor,
                            "risk_score_neutral": risk_neutral,
                            "risk_score_high": risk_high,
                            "risk_multiplier_min": 0.35,
                            "risk_multiplier_max": 1.15,
                        }
                        selected = [_decision_allows(pred, rule) for pred in predictions]
                        selected_count = sum(1 for item in selected if item)
                        if selected_count < min_selected:
                            continue
                        metrics = _trade_filter_metrics_from_selection(rows, selected)
                        candidates.append(
                            {
                                **rule,
                                "validation_selected_trades": selected_count,
                                "validation_selected_net_pnl": metrics["selected_net_pnl"],
                                "validation_selected_profit_factor": metrics["selected_profit_factor"],
                                "validation_selected_expectancy": metrics["selected_expectancy"],
                                "validation_selected_max_drawdown": metrics["selected_max_drawdown"],
                                "_rank": metrics["selected_expectancy"]
                                + min(metrics["selected_profit_factor"], 5.0) * 0.05
                                - metrics["selected_max_drawdown"] * 0.001,
                            }
                        )
    if not candidates:
        score_min = _quantile(scores, 0.70)
        return {
            **_fallback_decision_rule(),
            "quality_score_min": score_min,
            "name": "fallback_top_30",
            "validation_selected_trades": sum(1 for pred in predictions if pred["quality_score"] >= score_min),
            "soft_filter_enabled": True,
            "reject_score_min": risk_floor,
            "hard_expected_net_r_min": -0.25,
            "hard_large_loss_max": 0.70,
            "hard_false_breakout_max": 0.70,
            "hard_expected_mae_r_max": 3.0,
            "risk_score_floor": risk_floor,
            "risk_score_neutral": risk_neutral,
            "risk_score_high": risk_high,
        }
    best = max(candidates, key=lambda item: item["_rank"])
    best.pop("_rank", None)
    return best


def _fallback_decision_rule() -> dict[str, Any]:
    return {
        "name": "fallback_allow_balanced",
        "quality_score_min": -999.0,
        "expected_net_r_min": -999.0,
        "p_large_loss_max": 1.0,
        "p_false_breakout_max": 1.0,
        "expected_mae_r_max": 999.0,
        "expected_mfe_r_min": -999.0,
        "p_good_trade_min": 0.0,
        "reject_enabled": False,
        "sizing_enabled": True,
        "soft_filter_enabled": True,
        "reject_score_min": -0.85,
        "hard_expected_net_r_min": -0.25,
        "hard_large_loss_max": 0.70,
        "hard_false_breakout_max": 0.70,
        "hard_expected_mae_r_max": 3.0,
        "risk_score_floor": -0.85,
        "risk_score_neutral": -0.45,
        "risk_score_high": -0.10,
        "risk_multiplier_min": 0.35,
        "risk_multiplier_max": 1.15,
    }


def _decision_allows(prediction: dict[str, float], rule: dict[str, Any]) -> bool:
    score = float(prediction.get("quality_score", _quality_score(prediction)))
    return (
        score >= float(rule.get("quality_score_min", -999.0))
        and float(prediction.get("expected_net_r", 0.0)) >= float(rule.get("expected_net_r_min", -999.0))
        and float(prediction.get("p_large_loss", 0.0)) <= float(rule.get("p_large_loss_max", 1.0))
        and float(prediction.get("p_false_breakout", 0.0)) <= float(rule.get("p_false_breakout_max", 1.0))
        and float(prediction.get("expected_mae_r", 0.0)) <= float(rule.get("expected_mae_r_max", 999.0))
        and float(prediction.get("expected_mfe_r", 0.0)) >= float(rule.get("expected_mfe_r_min", -999.0))
        and float(prediction.get("p_good_trade", 0.0)) >= float(rule.get("p_good_trade_min", 0.0))
    )


def _soft_filter_allows(
    prediction: dict[str, float],
    rule: dict[str, Any],
    default_reject_score_min: float,
) -> bool:
    """Reject only the worst tail; score sizing controls the normal risk range."""
    score = float(prediction.get("quality_score", _quality_score(prediction)))
    reject_score_min = float(rule.get("reject_score_min", rule.get("risk_score_floor", default_reject_score_min)))
    if score < reject_score_min:
        return False

    expected_net_r = float(prediction.get("expected_net_r", 0.0))
    p_large_loss = float(prediction.get("p_large_loss", 0.0))
    p_false_breakout = float(prediction.get("p_false_breakout", 0.0))
    expected_mfe_r = float(prediction.get("expected_mfe_r", 0.0))
    expected_mae_r = float(prediction.get("expected_mae_r", 0.0))

    hard_expected_min = float(rule.get("hard_expected_net_r_min", -0.25))
    hard_large_loss = float(rule.get("hard_large_loss_max", 0.70))
    hard_false_breakout = float(rule.get("hard_false_breakout_max", 0.70))
    hard_mae = float(rule.get("hard_expected_mae_r_max", 3.0))

    if expected_net_r <= hard_expected_min and p_large_loss >= hard_large_loss:
        return False
    if expected_net_r <= hard_expected_min and p_false_breakout >= hard_false_breakout and expected_mfe_r < 0.20:
        return False
    if expected_net_r < 0.0 and expected_mae_r >= hard_mae:
        return False
    return True


def _score_risk_multiplier(
    score: float,
    rule: dict[str, Any],
    default_min_score: float,
    default_neutral_score: float,
    default_high_score: float,
    default_min_multiplier: float,
    default_max_multiplier: float,
) -> float:
    floor = float(rule.get("risk_score_floor", default_min_score))
    neutral = float(rule.get("risk_score_neutral", default_neutral_score))
    high = float(rule.get("risk_score_high", default_high_score))
    min_multiplier = max(0.0, float(rule.get("risk_multiplier_min", default_min_multiplier)))
    max_multiplier = max(min_multiplier, float(rule.get("risk_multiplier_max", default_max_multiplier)))
    if high <= neutral:
        high = neutral + 1e-9
    if neutral <= floor:
        neutral = floor + 1e-9
    if score <= floor:
        return min_multiplier
    if score < neutral:
        progress = (score - floor) / max(neutral - floor, 1e-9)
        return min_multiplier + (1.0 - min_multiplier) * max(0.0, min(1.0, progress))
    if score >= high:
        return max_multiplier
    progress = (score - neutral) / max(high - neutral, 1e-9)
    return 1.0 + (max_multiplier - 1.0) * max(0.0, min(1.0, progress))


def _quality_score(prediction: dict[str, float]) -> float:
    return (
        float(prediction.get("expected_net_r", 0.0))
        + 0.12 * float(prediction.get("expected_mfe_r", 0.0))
        + 0.10 * float(prediction.get("p_good_trade", 0.0))
        - 0.45 * float(prediction.get("p_large_loss", 0.0))
        - 0.35 * float(prediction.get("p_false_breakout", 0.0))
        - 0.12 * float(prediction.get("expected_mae_r", 0.0))
    )


def _trade_filter_metrics(rows: list[dict[str, Any]], probabilities: list[float], threshold: float) -> dict[str, float]:
    selected = [row for row, probability in zip(rows, probabilities) if probability >= threshold]
    rejected = [row for row, probability in zip(rows, probabilities) if probability < threshold]
    return {
        "threshold": threshold,
        "baseline_trades": float(len(rows)),
        "baseline_net_pnl": _net_pnl(rows),
        "baseline_profit_factor": _profit_factor(rows),
        "baseline_max_drawdown": _trade_list_drawdown(rows),
        "selected_trades": float(len(selected)),
        "selected_rate": len(selected) / max(1, len(rows)),
        "selected_net_pnl": _net_pnl(selected),
        "selected_profit_factor": _profit_factor(selected),
        "selected_max_drawdown": _trade_list_drawdown(selected),
        "selected_expectancy": _net_pnl(selected) / max(1, len(selected)),
        "selected_win_rate": _positive_rate(selected),
        "rejected_trades": float(len(rejected)),
        "rejected_net_pnl": _net_pnl(rejected),
        "cost_drag_per_selected_trade": _cost_drag(selected),
    }


def _trade_filter_metrics_from_selection(rows: list[dict[str, Any]], selected_mask: list[bool]) -> dict[str, float]:
    selected = [row for row, selected in zip(rows, selected_mask) if selected]
    rejected = [row for row, selected in zip(rows, selected_mask) if not selected]
    return {
        "baseline_trades": float(len(rows)),
        "baseline_net_pnl": _net_pnl(rows),
        "baseline_profit_factor": _profit_factor(rows),
        "baseline_max_drawdown": _trade_list_drawdown(rows),
        "selected_trades": float(len(selected)),
        "selected_rate": len(selected) / max(1, len(rows)),
        "selected_net_pnl": _net_pnl(selected),
        "selected_profit_factor": _profit_factor(selected),
        "selected_max_drawdown": _trade_list_drawdown(selected),
        "selected_expectancy": _net_pnl(selected) / max(1, len(selected)),
        "selected_win_rate": _net_pnl_positive_rate(selected),
        "rejected_trades": float(len(rejected)),
        "rejected_net_pnl": _net_pnl(rejected),
        "cost_drag_per_selected_trade": _cost_drag(selected),
        "large_loss_rate": _label_rate(selected, "large_loss_label"),
        "false_breakout_rate": _label_rate(selected, "false_breakout_label"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_prediction_csv(path: Path, rows: list[dict[str, Any]], probabilities: list[float]) -> None:
    enriched = []
    for row, probability in zip(rows, probabilities):
        enriched.append({**row, "pred_probability": probability})
    _write_csv(path, enriched)


def _write_prediction_bundle_csv(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, float]],
    selected: list[bool],
) -> None:
    enriched = []
    for row, prediction, is_selected in zip(rows, predictions, selected):
        enriched.append({**row, **prediction, "ml_selected": int(is_selected)})
    _write_csv(path, enriched)


def _random_filter_metrics(rows: list[dict[str, Any]], selected_count: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected_indices = set(indices[: max(0, min(len(rows), selected_count))])
    return _trade_filter_metrics_from_selection(rows, [index in selected_indices for index in range(len(rows))])


def _simple_rule_score(row: dict[str, Any]) -> float:
    return (
        _float(row.get("breakout_distance_atr")) * 0.20
        + _float(row.get("breakout_close_position")) * 0.50
        - _float(row.get("breakout_upper_wick_ratio")) * 0.40
        - abs(_float(row.get("pullback_depth_vs_breakout_distance"))) * 0.20
        - _float(row.get("pullback_volume_vs_breakout_volume")) * 0.20
        + _float(row.get("confirmation_close_position")) * 0.30
        - _float(row.get("confirmation_chase_pct")) * 4.0
        + _float(row.get("market_positive_return_ratio_1h")) * 0.20
    )


def _simple_rule_score_metrics(rows: list[dict[str, Any]], selected_count: int) -> dict[str, float]:
    ranked = sorted(range(len(rows)), key=lambda index: _simple_rule_score(rows[index]), reverse=True)
    selected_indices = set(ranked[: max(0, min(len(rows), selected_count))])
    return _trade_filter_metrics_from_selection(rows, [index in selected_indices for index in range(len(rows))])


def _score_decile_report(rows: list[dict[str, Any]], predictions: list[dict[str, float]]) -> list[dict[str, Any]]:
    pairs = sorted(zip(rows, predictions), key=lambda item: item[1].get("quality_score", 0.0))
    if not pairs:
        return []
    out: list[dict[str, Any]] = []
    for decile in range(10):
        start = int(len(pairs) * decile / 10)
        end = int(len(pairs) * (decile + 1) / 10)
        bucket = pairs[start:end]
        bucket_rows = [row for row, _ in bucket]
        bucket_predictions = [pred for _, pred in bucket]
        out.append(
            {
                "decile": decile + 1,
                "candidate_count": len(bucket_rows),
                "score_min": min((pred.get("quality_score", 0.0) for pred in bucket_predictions), default=0.0),
                "score_max": max((pred.get("quality_score", 0.0) for pred in bucket_predictions), default=0.0),
                "profitable_rate": _net_pnl_positive_rate(bucket_rows),
                "average_net_r": sum(_float(row.get("net_r")) for row in bucket_rows) / max(1, len(bucket_rows)),
                "median_net_r": _median([_float(row.get("net_r")) for row in bucket_rows]),
                "pf": _profit_factor(bucket_rows),
                "expectancy": _net_pnl(bucket_rows) / max(1, len(bucket_rows)),
                "large_loss_rate": _label_rate(bucket_rows, "large_loss_label"),
                "false_breakout_rate": _label_rate(bucket_rows, "false_breakout_label"),
                "average_mfe_r": sum(_float(row.get("mfe_r")) for row in bucket_rows) / max(1, len(bucket_rows)),
                "average_mae_r": sum(_float(row.get("mae_r")) for row in bucket_rows) / max(1, len(bucket_rows)),
            }
        )
    return out


def _probability_distribution_by_month(rows: list[dict[str, Any]], predictions: list[dict[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, float]]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        grouped[_row_time(row).strftime("%Y-%m")].append((row, prediction))
    out: list[dict[str, Any]] = []
    for month, items in sorted(grouped.items()):
        values = sorted(pred.get("p_good_trade", 0.0) for _, pred in items)
        out.append(
            {
                "month": month,
                "count": len(values),
                "min": min(values) if values else 0.0,
                "p05": _quantile(values, 0.05),
                "p10": _quantile(values, 0.10),
                "p25": _quantile(values, 0.25),
                "median": _quantile(values, 0.50),
                "p75": _quantile(values, 0.75),
                "p90": _quantile(values, 0.90),
                "p95": _quantile(values, 0.95),
                "max": max(values) if values else 0.0,
                "actual_good_rate": _label_rate([row for row, _ in items], "good_trade_label"),
            }
        )
    return out


def _calibration_report(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, float]],
    pred_key: str,
    label_key: str,
) -> list[dict[str, Any]]:
    buckets: list[list[tuple[dict[str, Any], dict[str, float]]]] = [[] for _ in range(10)]
    for row, prediction in zip(rows, predictions):
        value = max(0.0, min(0.999999, float(prediction.get(pred_key, 0.0))))
        buckets[int(value * 10)].append((row, prediction))
    out: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        pred_values = [prediction.get(pred_key, 0.0) for _, prediction in bucket]
        actual = [int(row.get(label_key, 0)) for row, _ in bucket]
        out.append(
            {
                "bucket": index,
                "prob_min": index / 10.0,
                "prob_max": (index + 1) / 10.0,
                "count": len(bucket),
                "avg_pred": sum(pred_values) / max(1, len(pred_values)),
                "actual_rate": sum(actual) / max(1, len(actual)),
                "brier": sum((pred - label) ** 2 for pred, label in zip(pred_values, actual)) / max(1, len(actual)),
            }
        )
    return out


def _threshold_coverage_report(window_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in window_metrics:
        rule = item.get("decision_rule", {})
        fixed = item.get("fixed_trade_filter", {})
        rows.append(
            {
                "window": item.get("window"),
                "rule_name": rule.get("name"),
                "quality_score_min": rule.get("quality_score_min"),
                "expected_net_r_min": rule.get("expected_net_r_min"),
                "p_large_loss_max": rule.get("p_large_loss_max"),
                "p_false_breakout_max": rule.get("p_false_breakout_max"),
                "expected_mae_r_max": rule.get("expected_mae_r_max"),
                "selected_trades": fixed.get("selected_trades"),
                "selected_rate": fixed.get("selected_rate"),
                "selected_net_pnl": fixed.get("selected_net_pnl"),
                "selected_profit_factor": fixed.get("selected_profit_factor"),
            }
        )
    return rows


def _window_rule_for_row(row: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any]:
    row_time = _row_time(row)
    for window in windows:
        if parse_timestamp(str(window["test_start"])) <= row_time < parse_timestamp(str(window["test_end"])):
            return dict(window.get("decision_rule") or {})
    return _fallback_decision_rule()


def _write_feature_importance(path: Path, model: Any, feature_columns: list[str]) -> None:
    gains = model.feature_importance(importance_type="gain")
    splits = model.feature_importance(importance_type="split")
    rows = [
        {"feature": feature, "gain": float(gain), "split": int(split)}
        for feature, gain, split in sorted(zip(feature_columns, gains, splits), key=lambda item: item[1], reverse=True)
    ]
    _write_csv(path, rows)


def _write_summary(path: Path, metrics: dict[str, Any], output_dir: Path) -> None:
    test = metrics["test_trade_filter"]
    baseline = test["baseline_net_pnl"]
    selected = test["selected_net_pnl"]
    text = f"""# VBP LightGBM meta-filter

This run trains a reject-only VBP LightGBM filter from VBP candidate-level full-cost shadow outcomes.

- samples: {metrics['sample_count']}
- train: {metrics['train_count']}
- test: {metrics['test_count']}
- features: {metrics['feature_count']}
- threshold: {metrics['threshold']}

## Test classification

- accuracy: {metrics['test']['accuracy']:.4f}
- precision: {metrics['test']['precision']:.4f}
- recall: {metrics['test']['recall']:.4f}
- auc: {metrics['test']['auc']:.4f}

## Test trade-list diagnostic

- baseline net pnl: {baseline:.4f}
- baseline PF: {test['baseline_profit_factor']:.4f}
- selected trades: {int(test['selected_trades'])} / {int(test['baseline_trades'])}
- selected net pnl: {selected:.4f}
- selected PF: {test['selected_profit_factor']:.4f}
- selected max drawdown: {test['selected_max_drawdown']:.4f}
- rejected net pnl: {test['rejected_net_pnl']:.4f}

Note: this is fixed-trade-list diagnostics. It does not recalculate compounding or changed portfolio path after filtering.

## Files

- dataset: `{output_dir / 'vbp_ml_dataset.csv'}`
- train: `{output_dir / 'vbp_ml_train.csv'}`
- test: `{output_dir / 'vbp_ml_test.csv'}`
- model: `{output_dir / 'vbp_lightgbm_model.txt'}`
- metrics: `{output_dir / 'metrics.json'}`
- feature importance: `{output_dir / 'feature_importance.csv'}`
"""
    path.write_text(text, encoding="utf-8")


def _feature_cache(candles: list[Candle]) -> SymbolFeatureCache:
    highs = [candle.high for candle in candles]
    lows = [candle.low for candle in candles]
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    return SymbolFeatureCache(
        timestamps=[candle.timestamp for candle in candles],
        open=[candle.open for candle in candles],
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
        close_prefix=_prefix_sum(closes),
        volume_prefix=_prefix_sum(volumes),
        high_1d=_rolling_extreme(highs, 1440, max),
        low_1d=_rolling_extreme(lows, 1440, min),
        high_7d=_rolling_extreme(highs, 1440 * 7, max),
        low_7d=_rolling_extreme(lows, 1440 * 7, min),
        high_30d=_rolling_extreme(highs, 1440 * 30, max),
        low_30d=_rolling_extreme(lows, 1440 * 30, min),
    )


def _sma(values: list[float], period: int) -> list[float]:
    out: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        out.append(running / min(period, index + 1))
    return out


def _mean_window(values: list[float], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    window = values[start:index + 1]
    return sum(window) / max(1, len(window))


def _prefix_sum(values: list[float]) -> list[float]:
    out = [0.0]
    total = 0.0
    for value in values:
        total += value
        out.append(total)
    return out


def _prefix_mean(prefix: list[float], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    total = prefix[index + 1] - prefix[start]
    return total / max(1, index - start + 1)


def _max_window(values: list[float], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    return max(values[start:index + 1])


def _min_window(values: list[float], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    return min(values[start:index + 1])


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def _ema_at(values: list[float], index: int, period: int) -> float:
    start = max(0, index - period * 10)
    return _ema(values[start:index + 1], period)[-1]


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    true_ranges: list[float] = []
    for index, high in enumerate(highs):
        previous_close = closes[index - 1] if index > 0 else closes[index]
        true_ranges.append(max(high - lows[index], abs(high - previous_close), abs(lows[index] - previous_close)))
    return _sma(true_ranges, period)


def _atr_at(highs: list[float], lows: list[float], closes: list[float], index: int, period: int) -> float:
    start = max(0, index - period + 1)
    true_ranges: list[float] = []
    for current in range(start, index + 1):
        previous_close = closes[current - 1] if current > 0 else closes[current]
        true_ranges.append(max(highs[current] - lows[current], abs(highs[current] - previous_close), abs(lows[current] - previous_close)))
    return sum(true_ranges) / max(1, len(true_ranges))


def _rsi(closes: list[float], period: int) -> list[float]:
    gains = [0.0]
    losses = [0.0]
    for index in range(1, len(closes)):
        delta = closes[index] - closes[index - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = _sma(gains, period)
    avg_loss = _sma(losses, period)
    return [100.0 if loss <= 1e-12 else 100.0 - 100.0 / (1.0 + gain / loss) for gain, loss in zip(avg_gain, avg_loss)]


def _rsi_at(closes: list[float], index: int, period: int) -> float:
    if index <= 0:
        return 50.0
    start = max(1, index - period + 1)
    gains = 0.0
    losses = 0.0
    count = 0
    for current in range(start, index + 1):
        delta = closes[current] - closes[current - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
        count += 1
    avg_gain = gains / max(1, count)
    avg_loss = losses / max(1, count)
    if avg_loss <= 1e-12:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _macd_hist(closes: list[float]) -> list[float]:
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [fast - slow for fast, slow in zip(ema12, ema26)]
    signal = _ema(macd_line, 9)
    return [line - sig for line, sig in zip(macd_line, signal)]


def _macd_hist_at(closes: list[float], index: int) -> float:
    start = max(0, index - 120)
    return _macd_hist(closes[start:index + 1])[-1]


def _rolling_extreme(values: list[float], period: int, fn: Any) -> list[float]:
    is_max = fn is max
    out: list[float] = []
    queue: deque[int] = deque()
    for index, value in enumerate(values):
        while queue and queue[0] <= index - period:
            queue.popleft()
        if is_max:
            while queue and values[queue[-1]] <= value:
                queue.pop()
        else:
            while queue and values[queue[-1]] >= value:
                queue.pop()
        queue.append(index)
        out.append(values[queue[0]])
    return out


def _ret(closes: list[float], index: int, bars: int) -> float:
    if index < bars:
        return math.nan
    previous = closes[index - bars]
    return _safe_div(closes[index] - previous, previous)


def _range_position(close: float, low: float, high: float) -> float:
    return _safe_div(close - low, high - low)


def _reason_float(entry_reason: str, key: str, default: float = math.nan) -> float:
    marker = f"{key}="
    if marker not in entry_reason:
        return default
    raw = entry_reason.split(marker, 1)[1].split()[0].strip().rstrip("%,")
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return math.nan
    return numerator / denominator


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _net_pnl(rows: list[dict[str, Any]]) -> float:
    return sum(_float(row.get("net_pnl")) for row in rows)


def _profit_factor(rows: list[dict[str, Any]]) -> float:
    wins = sum(_float(row.get("net_pnl")) for row in rows if _float(row.get("net_pnl")) > 0)
    losses = -sum(_float(row.get("net_pnl")) for row in rows if _float(row.get("net_pnl")) < 0)
    if losses <= 1e-12:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def _trade_list_drawdown(rows: list[dict[str, Any]]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in rows:
        equity += _float(row.get("net_pnl"))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _positive_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get("label", 0)) == 1) / len(rows)


def _label_rate(rows: list[dict[str, Any]], label_key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.get(label_key, 0)) == 1) / len(rows)


def _net_pnl_positive_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if _float(row.get("net_pnl")) > 0) / len(rows)


def _cost_drag(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total_cost = sum(_float(row.get("fee")) + _float(row.get("slippage_cost")) + _float(row.get("funding")) for row in rows)
    return total_cost / len(rows)


def _quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    q = max(0.0, min(1.0, q))
    index = int(round((len(finite) - 1) * q))
    return finite[index]


def _median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2.0


def _auc(y_true: list[int], probabilities: list[float]) -> float:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(zip(probabilities, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, actual) in enumerate(ordered, start=1):
        if actual == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


if __name__ == "__main__":
    raise SystemExit(main())
