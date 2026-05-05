from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .backtest import Backtester
from .binance_client import BinanceFuturesClient
from .config import load_config
from .data import generate_sample_candles, load_candles_csv, write_candles_csv
from .live_config import load_live_config
from .live_trader import BinanceAutoTrader
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
    backtest.add_argument("--data", default=None, help="override config data.path")
    backtest.add_argument("--trades", action="store_true", help="print closed trades")

    optimize = subparsers.add_parser("optimize", help="grid-search strategy parameters on one dataset")
    optimize.add_argument("--config", default="config.example.json")
    optimize.add_argument("--data", default=None, help="override config data.path")
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
        data_path = args.data or config.data.path
        candles = load_candles_csv(data_path)
        strategy = VolatilityBreakoutScalper(config.strategy)
        result = Backtester(candles, strategy, config.risk).run()
        print(json.dumps(result.summary, indent=2, ensure_ascii=False))
        if args.trades:
            for trade in result.trades:
                print(
                    f"{trade.entry_time.isoformat()} {trade.direction.name:<5} "
                    f"entry={trade.entry_price:.2f} exit={trade.exit_price:.2f} "
                    f"qty={trade.qty:.6f} pnl={trade.net_pnl:.2f} reason={trade.exit_reason}"
                )
        return 0

    if args.command == "optimize":
        config = load_config(args.config)
        data_path = args.data or config.data.path
        candles = load_candles_csv(data_path)
        rows = []

        for strategy_config in _sample_strategy_configs(config.strategy, args.trials, args.seed):
            result = Backtester(candles, VolatilityBreakoutScalper(strategy_config), config.risk).run()
            if result.summary["total_trades"] < args.min_trades:
                continue
            params = _strategy_params(strategy_config)
            rows.append(
                {
                    "score": _score(result.summary, args.metric),
                    "params": params,
                    "summary": result.summary,
                }
            )

        rows.sort(key=lambda row: row["score"], reverse=args.metric != "max_drawdown_pct")
        if args.write_config and rows:
            _write_config(config, rows[0]["params"], args.write_config)
        print(json.dumps(rows[: max(args.top, 0)], indent=2, ensure_ascii=False))
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
        return float(summary["net_return_pct"]) - float(summary["max_drawdown_pct"]) * 0.75 + factor_bonus
    if metric == "calmar":
        drawdown = max(float(summary["max_drawdown_pct"]), 0.01)
        return float(summary["net_return_pct"]) / drawdown
    value = summary.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def _sample_strategy_configs(base: Any, trials: int, seed: int) -> Iterable[Any]:
    rng = random.Random(seed)
    min_atr_values = sorted({0.0, base.min_atr_pct / 2.0, base.min_atr_pct, base.min_atr_pct * 2.0})
    spaces = {
        "fast_ema": (4, 5, 7, 9, 12),
        "slow_ema": (13, 21, 34, 55),
        "channel_period": (8, 10, 14, 20, 30, 40),
        "min_atr_pct": tuple(min_atr_values),
        "max_atr_pct": (0.004, 0.0075, 0.01, 0.02),
        "breakout_buffer_atr": (0.0, 0.05, 0.1, 0.2, 0.35),
        "ema_gap_atr": (0.0, 0.05, 0.1, 0.2, 0.35),
        "min_volume_ratio": (0.0, 0.8, 1.0, 1.2),
        "stop_loss_atr": (0.6, 0.8, 1.0, 1.2, 1.6, 2.0),
        "take_profit_atr": (1.0, 1.2, 1.6, 1.8, 2.4, 3.0, 4.0),
        "breakeven_atr": (0.0, 0.8, 1.0, 1.5),
        "trailing_activation_atr": (0.5, 1.0, 1.5, 2.0),
        "trailing_stop_atr": (0.0, 0.8, 1.0, 1.2, 1.6, 2.0),
        "max_holding_bars": (0, 15, 30, 60, 120),
        "spike_min_range_atr": (2.5, 3.0, 4.0),
        "spike_min_wick_atr": (1.0, 1.4, 2.0),
        "spike_min_wick_ratio": (0.5, 0.6, 0.7),
        "spike_min_volume_ratio": (1.0, 1.2, 1.6),
        "spike_recovery_ratio": (0.35, 0.45, 0.6),
        "spike_stop_atr": (0.5, 0.7, 1.0),
        "spike_take_profit_atr": (0.7, 0.9, 1.2),
        "spike_risk_multiplier": (0.2, 0.35, 0.5),
        "spike_max_holding_bars": (3, 6, 10),
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
        if values["take_profit_atr"] <= values["stop_loss_atr"]:
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
        "channel_period",
        "min_atr_pct",
        "max_atr_pct",
        "breakout_buffer_atr",
        "ema_gap_atr",
        "min_volume_ratio",
        "stop_loss_atr",
        "take_profit_atr",
        "breakeven_atr",
        "trailing_activation_atr",
        "trailing_stop_atr",
        "max_holding_bars",
        "spike_min_range_atr",
        "spike_min_wick_atr",
        "spike_min_wick_ratio",
        "spike_min_volume_ratio",
        "spike_recovery_ratio",
        "spike_stop_atr",
        "spike_take_profit_atr",
        "spike_risk_multiplier",
        "spike_max_holding_bars",
    )
    return {name: getattr(strategy_config, name) for name in names}


def _write_config(config: Any, params: dict[str, Any], output_path: str) -> None:
    payload = {
        "data": asdict(config.data),
        "strategy": asdict(replace(config.strategy, **params)),
        "risk": asdict(config.risk),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
