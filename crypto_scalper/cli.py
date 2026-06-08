from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .backtest import Backtester
from .binance_client import BinanceFuturesClient
from .config import load_config
from .data import download_binance_futures_klines, generate_sample_candles, load_candles_csv, write_candles_csv
from .live_config import DEFAULT_SYMBOLS, load_live_config
from .live_trader import BinanceAutoTrader
from .research import run_robust_optimization
from .secrets import mask_secret, read_secret
from .strategy import VolatilityBreakoutScalper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-first crypto futures scalping toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("generate-sample", help="generate deterministic sample 1m OHLCV data")
    sample.add_argument("--output", default="data/sample_btcusdt_1m.csv")
    sample.add_argument("--bars", type=int, default=2_000)
    sample.add_argument("--start-price", type=float, default=60_000.0)
    sample.add_argument("--seed", type=int, default=42)

    backtest = subparsers.add_parser("backtest", help="run a CSV-driven backtest")
    backtest.add_argument("--config", default="config.example.json")
    backtest.add_argument("--data", action="append", default=None, help="override config data.path; can be repeated")
    backtest.add_argument("--data-glob", default=None, help="run all CSV files matching this glob")
    backtest.add_argument("--trades", action="store_true", help="print closed trades")

    optimize = subparsers.add_parser("optimize", help="grid-search strategy parameters on one dataset")
    optimize.add_argument("--config", default="config.example.json")
    optimize.add_argument("--data", action="append", default=None, help="override config data.path; can be repeated")
    optimize.add_argument("--data-glob", default=None, help="evaluate every sampled strategy on all matching CSV files")
    optimize.add_argument(
        "--metric",
        default="profit_score",
        choices=("profit_score", "net_return_pct", "profit_factor", "win_rate_pct", "max_drawdown_pct", "calmar"),
    )
    optimize.add_argument("--top", type=int, default=10)
    optimize.add_argument("--min-trades", type=int, default=20)
    optimize.add_argument("--trials", type=int, default=250, help="number of random parameter sets to evaluate")
    optimize.add_argument("--seed", type=int, default=42)
    optimize.add_argument("--write-config", default=None, help="write the best parameter set to this JSON config")

    robust = subparsers.add_parser("robust-optimize", help="train/validation/test robust parameter search and report")
    robust.add_argument("--config", default="config.optimized.json")
    robust.add_argument("--data", action="append", default=None, help="override config data.path; can be repeated")
    robust.add_argument("--data-glob", default=None, help="evaluate all CSV files matching this glob")
    robust.add_argument("--trials", type=int, default=200)
    robust.add_argument("--seed", type=int, default=42)
    robust.add_argument("--top", type=int, default=8)
    robust.add_argument("--min-trades", type=int, default=120)
    robust.add_argument("--max-drawdown-pct", type=float, default=10.0)
    robust.add_argument("--train-ratio", type=float, default=0.60)
    robust.add_argument("--validation-ratio", type=float, default=0.20)
    robust.add_argument("--report", default="report.md")
    robust.add_argument("--write-config", default="config.robust.json")

    history = subparsers.add_parser("download-history", help="download Binance USD-M futures klines to CSV")
    history.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma/newline separated symbols")
    history.add_argument("--timeframe", default="15m")
    history.add_argument("--days", type=int, default=180)
    history.add_argument("--start", default=None, help="UTC start, e.g. 2025-12-01 or 2025-12-01T00:00:00")
    history.add_argument("--end", default=None, help="UTC end, defaults to now")
    history.add_argument("--output-dir", default="data/binance")
    history.add_argument("--sleep", type=float, default=0.05)

    live = subparsers.add_parser("trade-live", help="run Binance USD-M futures live/testnet trader")
    live.add_argument("--config", default="config.live.example.json")
    live.add_argument("--once", action="store_true", help="run one polling cycle and exit")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-sample":
        candles = generate_sample_candles(args.bars, args.start_price, args.seed)
        write_candles_csv(args.output, candles)
        print(f"wrote {len(candles)} candles to {Path(args.output)}")
        return 0

    if args.command == "backtest":
        config = load_config(args.config)
        data_paths = _resolve_data_paths(args.data, args.data_glob, config.data.path)
        results = _run_backtests(data_paths, config)
        if len(results) == 1:
            print(json.dumps(results[0][1].summary, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(_portfolio_payload(results), indent=2, ensure_ascii=False))
        if args.trades:
            for path, result in results:
                if len(results) > 1:
                    print(f"# {path}")
                for trade in result.trades:
                    print(
                        f"{trade.entry_time.isoformat()} {trade.direction.name:<5} "
                        f"entry={trade.entry_price:.2f} exit={trade.exit_price:.2f} "
                        f"qty={trade.qty:.6f} pnl={trade.net_pnl:.2f} reason={trade.exit_reason}"
                    )
        return 0

    if args.command == "optimize":
        config = load_config(args.config)
        data_paths = _resolve_data_paths(args.data, args.data_glob, config.data.path)
        datasets = [(path, load_candles_csv(path)) for path in data_paths]
        rows = []

        for strategy_config in _sample_strategy_configs(config.strategy, args.trials, args.seed):
            results = [
                (path, Backtester(candles, VolatilityBreakoutScalper(strategy_config), config.risk).run())
                for path, candles in datasets
            ]
            summary = _aggregate_summaries(results)
            if summary["total_trades"] < args.min_trades:
                continue
            params = _strategy_params(strategy_config)
            rows.append(
                {
                    "score": _score(summary, args.metric),
                    "params": params,
                    "summary": summary,
                }
            )

        rows.sort(key=lambda row: row["score"], reverse=args.metric != "max_drawdown_pct")
        if args.write_config and rows:
            _write_config(config, rows[0]["params"], args.write_config)
        print(json.dumps(rows[: max(args.top, 0)], indent=2, ensure_ascii=False))
        return 0

    if args.command == "robust-optimize":
        config = load_config(args.config)
        data_paths = _resolve_data_paths(args.data, args.data_glob, config.data.path)
        payload = run_robust_optimization(
            config=config,
            data_paths=data_paths,
            trials=args.trials,
            seed=args.seed,
            top=args.top,
            min_trades=args.min_trades,
            max_drawdown_pct=args.max_drawdown_pct,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            report_path=args.report,
            write_config_path=args.write_config,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "download-history":
        symbols = _parse_symbols(args.symbols)
        end = _parse_datetime_arg(args.end) if args.end else datetime.now(timezone.utc).replace(tzinfo=None)
        start = _parse_datetime_arg(args.start) if args.start else end - timedelta(days=max(args.days, 1))
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for symbol in symbols:
            candles = download_binance_futures_klines(symbol, args.timeframe, start, end, sleep_seconds=args.sleep)
            output = output_dir / f"{symbol}_{args.timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
            write_candles_csv(output, candles)
            print(f"wrote {len(candles)} candles to {output}")
        return 0

    if args.command == "trade-live":
        config = load_live_config(args.config)
        api_key = read_secret(config.exchange.api_key_env)
        api_secret = read_secret(config.exchange.api_secret_env)
        print(
            f"environment={config.exchange.environment} dry_run={config.trading.dry_run} "
            f"api_key={mask_secret(api_key)}"
        )
        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            environment=config.exchange.environment,
            recv_window=config.exchange.recv_window,
            timeout_seconds=config.exchange.timeout_seconds,
        )
        trader = BinanceAutoTrader(config, client, logger=print)
        if args.once:
            trader.validate_startup()
            trader.run_once()
            return 0
        import threading

        stop_event = threading.Event()
        try:
            trader.run_forever(stop_event)
        except KeyboardInterrupt:
            stop_event.set()
        return 0

    return 2


def _score(summary: dict[str, Any], metric: str) -> float:
    if metric == "profit_score":
        profit_factor = summary.get("profit_factor")
        factor_bonus = 0.0 if profit_factor is None else min(float(profit_factor), 3.0) * 2.0
        datasets = max(float(summary.get("datasets", 1)), 1.0)
        target_trades = max(20.0, datasets * 18.0)
        frequency_ratio = float(summary.get("total_trades", 0)) / target_trades
        frequency_bonus = min(frequency_ratio, 1.5) * 2.0
        sparse_penalty = max(0.0, 1.0 - frequency_ratio) * 8.0
        return float(summary["net_return_pct"]) - float(summary["max_drawdown_pct"]) * 0.75 + factor_bonus + frequency_bonus - sparse_penalty
    if metric == "calmar":
        drawdown = max(float(summary["max_drawdown_pct"]), 0.01)
        return float(summary["net_return_pct"]) / drawdown
    value = summary.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def _resolve_data_paths(data: list[str] | None, data_glob: str | None, default_path: str) -> list[str]:
    paths: list[str] = []
    if data:
        paths.extend(data)
    if data_glob:
        paths.extend(str(path) for path in sorted(Path().glob(data_glob)))
    if not paths:
        paths.append(default_path)
    unique: list[str] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def _run_backtests(data_paths: list[str], config: Any) -> list[tuple[str, Any]]:
    results = []
    for path in data_paths:
        candles = load_candles_csv(path)
        result = Backtester(candles, VolatilityBreakoutScalper(config.strategy), config.risk).run()
        results.append((path, result))
    return results


def _portfolio_payload(results: list[tuple[str, Any]]) -> dict[str, Any]:
    return {
        "summary": _aggregate_summaries(results),
        "datasets": [
            {
                "path": path,
                "summary": result.summary,
            }
            for path, result in results
        ],
    }


def _aggregate_summaries(results: list[tuple[str, Any]]) -> dict[str, Any]:
    initial = sum(float(result.summary["initial_equity"]) for _, result in results)
    final = sum(float(result.summary["final_equity"]) for _, result in results)
    period_days = max((float(result.summary.get("period_days", 0.0)) for _, result in results), default=0.0)
    months = period_days / 30.4375 if period_days > 0 else 0.0
    trades = [trade for _, result in results for trade in result.trades]
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    gross_profit = sum(trade.net_pnl for trade in wins)
    gross_loss = abs(sum(trade.net_pnl for trade in losses))
    total_trades = len(trades)
    return {
        "datasets": len(results),
        "initial_equity": initial,
        "final_equity": final,
        "net_profit": final - initial,
        "net_return_pct": 0.0 if initial <= 0 else (final / initial - 1.0) * 100.0,
        "monthly_return_pct": 0.0 if initial <= 0 or months <= 0 else ((final / initial - 1.0) * 100.0) / months,
        "period_days": period_days,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 0.0 if total_trades == 0 else len(wins) / total_trades * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "avg_trade": 0.0 if total_trades == 0 else sum(trade.net_pnl for trade in trades) / total_trades,
        "max_drawdown_pct": max((float(result.summary["max_drawdown_pct"]) for _, result in results), default=0.0),
    }


def _sample_strategy_configs(base: Any, trials: int, seed: int) -> Iterable[Any]:
    rng = random.Random(seed)
    min_atr_values = sorted({0.0, base.min_atr_pct / 2.0, base.min_atr_pct, base.min_atr_pct * 2.0, 0.001, 0.0015, 0.0025, 0.004})
    spaces = {
        "fast_ema": (5, 8, 12, 18, 24),
        "slow_ema": (21, 34, 55, 89),
        "atr_period": (10, 14, 21),
        "channel_period": (12, 20, 30, 48, 72),
        "min_atr_pct": tuple(min_atr_values),
        "max_atr_pct": (0.012, 0.02, 0.035, 0.05, 0.08),
        "breakout_buffer_atr": (0.0, 0.1, 0.2, 0.35, 0.5),
        "ema_gap_atr": (0.0, 0.1, 0.2, 0.35, 0.5),
        "volume_period": (20, 30, 48),
        "min_volume_ratio": (0.0, 0.8, 1.0, 1.2, 1.5),
        "stop_loss_atr": (0.8, 1.0, 1.2, 1.6, 2.0, 2.5),
        "take_profit_atr": (0.8, 1.0, 1.2, 1.6, 2.4, 3.0, 4.0),
        "breakeven_atr": (0.0, 1.0, 1.5, 2.0),
        "trailing_activation_atr": (1.0, 1.5, 2.0, 3.0),
        "trailing_stop_atr": (0.0, 1.0, 1.5, 2.0, 3.0),
        "max_holding_bars": (8, 12, 24, 48, 72),
        "spike_guard_enabled": (True, False),
        "spike_trade_enabled": (True, False),
        "spike_min_range_atr": (2.5, 3.0, 4.0, 5.0),
        "spike_min_wick_atr": (1.0, 1.4, 2.0),
        "spike_min_wick_ratio": (0.5, 0.6, 0.7),
        "spike_min_volume_ratio": (1.0, 1.2, 1.6),
        "spike_recovery_ratio": (0.35, 0.45, 0.6),
        "spike_stop_atr": (0.5, 0.7, 1.0),
        "spike_take_profit_atr": (0.7, 0.9, 1.2),
        "spike_risk_multiplier": (0.2, 0.35, 0.5),
        "spike_max_holding_bars": (4, 8, 12),
        "allow_short": (True, False),
    }

    yielded = 0
    attempts = 0
    seen: set[tuple[Any, ...]] = set()
    max_attempts = max(trials * 50, 500)
    while yielded < max(trials, 0) and attempts < max_attempts:
        attempts += 1
        values = {name: rng.choice(options) for name, options in spaces.items()}
        if values["fast_ema"] >= values["slow_ema"]:
            continue
        if values["take_profit_atr"] < values["stop_loss_atr"] * 0.6:
            continue
        key = tuple(values[name] for name in sorted(values))
        if key in seen:
            continue
        seen.add(key)
        yielded += 1
        yield replace(base, **values)


def _strategy_params(strategy_config: Any) -> dict[str, Any]:
    names = (
        "fast_ema",
        "slow_ema",
        "atr_period",
        "channel_period",
        "min_atr_pct",
        "max_atr_pct",
        "breakout_buffer_atr",
        "ema_gap_atr",
        "volume_period",
        "min_volume_ratio",
        "stop_loss_atr",
        "take_profit_atr",
        "breakeven_atr",
        "trailing_activation_atr",
        "trailing_stop_atr",
        "max_holding_bars",
        "spike_guard_enabled",
        "spike_trade_enabled",
        "spike_min_range_atr",
        "spike_min_wick_atr",
        "spike_min_wick_ratio",
        "spike_min_volume_ratio",
        "spike_recovery_ratio",
        "spike_stop_atr",
        "spike_take_profit_atr",
        "spike_risk_multiplier",
        "spike_max_holding_bars",
        "allow_short",
        "long_score_threshold",
        "short_score_threshold",
        "long_risk_bias",
        "short_risk_bias",
        "regime_filter_enabled",
        "regime_lookback",
        "long_min_slow_slope_atr",
        "short_max_slow_slope_atr",
        "super_volume_breakout_enabled",
        "super_volume_min_ratio",
        "super_volume_min_breakout_atr",
        "super_volume_min_body_atr",
        "super_volume_confidence_boost",
        "super_volume_risk_multiplier",
        "super_volume_take_profit_multiplier",
    )
    return {name: getattr(strategy_config, name) for name in names}


def _parse_symbols(value: str) -> list[str]:
    symbols: list[str] = []
    for part in value.replace("，", ",").replace("\n", ",").split(","):
        symbol = part.strip().upper()
        if not symbol:
            continue
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        if symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise ValueError("at least one symbol is required")
    return symbols


def _parse_datetime_arg(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _write_config(config: Any, params: dict[str, Any], output_path: str) -> None:
    payload = {
        "data": asdict(config.data),
        "strategy": asdict(replace(config.strategy, **params)),
        "risk": asdict(config.risk),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
