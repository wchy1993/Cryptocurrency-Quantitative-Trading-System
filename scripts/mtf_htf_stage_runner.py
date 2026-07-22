from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_scalper.live_config import load_live_config
from crypto_scalper.live_execution_backtest import _load_symbol_data
from crypto_scalper.live_execution_backtest import _mtf_required_timeframes
from crypto_scalper.live_execution_backtest import _resample_to_timeframe
from crypto_scalper.live_execution_backtest import run_execution_backtest_config
from crypto_scalper.mtf_4h_rsi_regime import load_auxiliary_features
from scripts.mtf_htf_diagnostics import diagnose
from scripts.mtf_htf_summary import summarize


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _config_specs(value: str) -> list[tuple[str, str]]:
    output = []
    for item in value.split(","):
        name, separator, path = item.strip().partition("=")
        if not separator or not name or not path:
            raise ValueError(f"invalid config specification: {item}")
        output.append((name, path))
    return output


def _manifest_configs(path: str) -> list[tuple[str, str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    base_path = str(payload["base_config"])
    base = load_live_config(base_path)
    output = []
    for item in payload.get("experiments", []):
        name = str(item["name"])
        config = base
        mtpc_source = str(item.get("mtpc_source_config", "") or "")
        if mtpc_source:
            config = replace(config, mtpc=load_live_config(mtpc_source).mtpc)
        universe_source = str(item.get("universe_source_config", "") or "")
        if universe_source:
            source_trading = load_live_config(universe_source).trading
            if bool(item.get("append_universe_to_base", False)):
                symbols = tuple(dict.fromkeys((*config.trading.symbols, *source_trading.symbols)))
                entry_symbols = tuple(
                    dict.fromkeys((*config.trading.entry_symbols, *source_trading.entry_symbols))
                )
            else:
                symbols = source_trading.symbols
                entry_symbols = source_trading.entry_symbols
            config = replace(
                config,
                trading=replace(
                    config.trading,
                    symbols=symbols,
                    entry_symbols=entry_symbols,
                ),
            )
        for section in ("trading", "strategy", "risk"):
            overrides = item.get(section, {})
            if overrides:
                config = replace(config, **{section: replace(getattr(config, section), **overrides)})
        mtpc_overrides = item.get("mtpc", {})
        if mtpc_overrides:
            config = replace(config, mtpc=replace(config.mtpc, **mtpc_overrides))
        output.append((name, f"{path}#{name}", config))
    if not output:
        raise ValueError(f"{path}: manifest contains no experiments")
    return output


_EXIT_ONLY_STRATEGY_FIELDS = {
    "mtf_profit_protection_enabled",
    "mtf_move_stop_to_breakeven_r",
    "mtf_breakeven_extra_bps",
    "mtf_trailing_start_r",
    "mtf_trailing_mode",
    "mtf_profit_giveback_r",
    "mtf_trailing_atr15m_mult",
    "mtf_exit_mode",
    "mtf_partial_take_profit_r",
    "mtf_partial_take_profit_fraction",
    "mtf_fail_fast_minutes",
    "mtf_fail_fast_min_r",
    "mtf_exit_on_30m_confirm_lost",
    "mtf_30m_exit_confirm_bars",
    "mtf_30m_exit_require_macd_adverse",
    "mtf_exit_on_1h_confirm_lost",
    "mtf_extra_execution_delay_minutes",
    # These fields schedule already-computed candidates; they do not alter them.
    "mtf_max_open_positions",
    "mtf_max_daily_trades",
    "mtf_symbol_cooldown_hours",
}


def _signal_strategy_key(strategy: Any) -> int:
    values = tuple(
        (item.name, getattr(strategy, item.name))
        for item in fields(strategy)
        if item.name not in _EXIT_ONLY_STRATEGY_FIELDS
    )
    return hash(values)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest:
        configs = _manifest_configs(args.manifest)
    else:
        specs = _config_specs(args.configs)
        configs = [(name, path, load_live_config(path)) for name, path in specs]
    requested = {
        item.strip()
        for item in str(getattr(args, "experiments", "") or "").split(",")
        if item.strip()
    }
    if requested:
        configs = [item for item in configs if item[0] in requested]
        missing = requested - {item[0] for item in configs}
        if missing:
            raise ValueError(f"unknown experiments: {', '.join(sorted(missing))}")
    if not configs:
        raise ValueError("no staged experiments selected")
    symbols = tuple(configs[0][2].trading.symbols)
    timeframe = str(configs[0][2].trading.timeframe)
    for name, _path, config in configs[1:]:
        if tuple(config.trading.symbols) != symbols:
            raise ValueError(f"{name}: staged configs must use the same symbol universe")
        if str(config.trading.timeframe) != timeframe:
            raise ValueError(f"{name}: staged configs must use the same signal timeframe")
    execution = _load_symbol_data(args.execution_data_dir, symbols, "1m")
    loaded_symbols = tuple(symbol for symbol in symbols if symbol in execution)
    execution = {symbol: execution[symbol] for symbol in loaded_symbols}
    signal = {
        symbol: _resample_to_timeframe(candles, "1m", timeframe)
        for symbol, candles in execution.items()
    }
    mtf_timeframes = tuple(
        sorted(
            {
                required
                for _name, _path, config in configs
                for required in _mtf_required_timeframes(config)
            }
        )
    )
    mtf_candles = {
        required: {
            symbol: _resample_to_timeframe(candles, "1m", required)
            for symbol, candles in execution.items()
        }
        for required in mtf_timeframes
    }
    experiments = []
    candidate_caches: dict[int, dict[tuple[Any, ...], Any]] = {}
    auxiliary_caches: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for name, path, config in configs:
        strategy_key = _signal_strategy_key(config.strategy)
        auxiliary_key = (
            str(getattr(config.strategy, "mtf_oi_data_dir", "data/binance_oi_taker_5m")),
            str(getattr(config.strategy, "mtf_funding_data_dir", "data/binance_oi_flush_funding")),
        )
        if auxiliary_key not in auxiliary_caches:
            auxiliary_caches[auxiliary_key] = load_auxiliary_features(
                loaded_symbols,
                auxiliary_key[0],
                auxiliary_key[1],
            )
        report = run_execution_backtest_config(
            config,
            execution,
            signal,
            initial_equity=args.initial_equity,
            include_trades=True,
            compact=True,
            progress=True,
            mtf_candles_by_timeframe=mtf_candles,
            trade_start=_timestamp(args.trade_start),
            trade_end=_timestamp(args.trade_end),
            cost_experiment="full_cost",
            backtest_mode="conservative",
            mtf_candidate_cache=candidate_caches.setdefault(strategy_key, {}),
            mtf_aux_features_override=auxiliary_caches[auxiliary_key],
        )
        experiments.append(
            {
                "name": name,
                "config": path,
                "summary": summarize(report),
                "diagnostics": diagnose(report),
                "report": report,
            }
        )
        gc.collect()
    return {
        "trade_start": args.trade_start,
        "trade_end": args.trade_end,
        "cost_experiment": "full_cost",
        "backtest_mode": "conservative",
        "symbols": list(loaded_symbols),
        "experiments": experiments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged MTF full-cost experiments with one data load")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--configs", help="Comma-separated name=config.json entries")
    source.add_argument("--manifest", help="JSON staged experiment manifest")
    parser.add_argument("--experiments", help="Optional comma-separated experiment names from the manifest")
    parser.add_argument("--execution-data-dir", required=True)
    parser.add_argument("--initial-equity", type=float, default=160.0)
    parser.add_argument("--trade-start", required=True)
    parser.add_argument("--trade-end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(args)
    Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                experiment["name"]: experiment["summary"]
                for experiment in payload["experiments"]
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
