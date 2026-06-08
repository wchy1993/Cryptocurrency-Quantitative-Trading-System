from __future__ import annotations

import argparse
import bisect
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from .data import download_binance_futures_klines, load_candles_csv, parse_timestamp, write_candles_csv
from .models import Candle, Direction


@dataclass(frozen=True)
class MacroEvent:
    timestamp: Any
    event_type: str
    reference: str
    actual: float
    forecast: float
    unit: str
    source: str = ""

    @property
    def surprise(self) -> float:
        return self.actual - self.forecast


@dataclass(frozen=True)
class MacroEventBacktestConfig:
    initial_equity: float = 160.0
    leverage: float = 50.0
    margin_pct: float = 0.05
    max_position_notional_usdt: float = 500.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    entry_delay_seconds: int = 60
    max_holding_minutes: int = 15
    stop_loss_pct: float = 0.0025
    take_profit_pct: float = 0.0040
    nfp_min_surprise_k: float = 25.0
    cpi_min_surprise_pct: float = 0.10
    event_types: tuple[str, ...] = ("NFP", "CPI_YOY")
    max_entry_gap_seconds: int = 300


def load_macro_events(path: str | Path) -> list[MacroEvent]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp_utc", "event_type", "reference", "actual", "forecast", "unit"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing macro event columns: {', '.join(sorted(missing))}")
        events = [
            MacroEvent(
                timestamp=parse_timestamp(row["timestamp_utc"]),
                event_type=row["event_type"].strip().upper(),
                reference=row["reference"].strip(),
                actual=float(row["actual"]),
                forecast=float(row["forecast"]),
                unit=row["unit"].strip(),
                source=row.get("source", "").strip(),
            )
            for row in reader
        ]
    events.sort(key=lambda item: item.timestamp)
    return events


def run_macro_event_backtest(
    data_dir: str | Path,
    events_path: str | Path,
    symbols: Iterable[str],
    config: MacroEventBacktestConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MacroEventBacktestConfig()
    events = [event for event in load_macro_events(events_path) if event.event_type in cfg.event_types]
    symbol_results = []
    for symbol in _normalize_symbols(symbols):
        candles = _load_symbol_1m(data_dir, symbol)
        symbol_results.append(_run_symbol_backtest(symbol, candles, events, cfg))
    symbol_results.sort(key=lambda item: item["summary"]["net_pnl"], reverse=True)
    return {
        "config": asdict(cfg),
        "events_path": str(events_path),
        "data_dir": str(data_dir),
        "symbols": symbol_results,
        "best_symbol": symbol_results[0]["symbol"] if symbol_results else None,
    }


def optimize_macro_event_backtest(
    data_dir: str | Path,
    events_path: str | Path,
    symbols: Iterable[str],
    initial_equity: float = 160.0,
    top: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_symbols = _normalize_symbols(symbols)
    all_events = load_macro_events(events_path)
    candles_by_symbol = {symbol: _load_symbol_1m(data_dir, symbol) for symbol in normalized_symbols}
    event_sets = (("NFP",), ("CPI_YOY",), ("NFP", "CPI_YOY"))
    for symbol in normalized_symbols:
        candles = candles_by_symbol[symbol]
        for event_types in event_sets:
            for delay in (0, 60, 120, 180):
                for holding in (8, 12, 15, 20, 30):
                    for stop_loss in (0.0020, 0.0025, 0.0035):
                        for take_profit in (0.0030, 0.0040, 0.0055, 0.0070):
                            if take_profit <= stop_loss * 1.1:
                                continue
                            for nfp_threshold in (0.0, 25.0, 50.0):
                                for cpi_threshold in (0.0, 0.10, 0.20):
                                    cfg = MacroEventBacktestConfig(
                                        initial_equity=initial_equity,
                                        entry_delay_seconds=delay,
                                        max_holding_minutes=holding,
                                        stop_loss_pct=stop_loss,
                                        take_profit_pct=take_profit,
                                        nfp_min_surprise_k=nfp_threshold,
                                        cpi_min_surprise_pct=cpi_threshold,
                                        event_types=event_types,
                                    )
                                    events = [event for event in all_events if event.event_type in event_types]
                                    result = _run_symbol_backtest(symbol, candles, events, cfg)
                                    summary = result["summary"]
                                    if summary["trades"] < 4:
                                        continue
                                    rows.append(
                                        {
                                            "score": _macro_score(summary),
                                            "symbol": symbol,
                                            "config": asdict(cfg),
                                            "summary": summary,
                                        }
                                    )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[: max(0, top)]


def _run_symbol_backtest(
    symbol: str,
    candles: list[Candle],
    events: list[MacroEvent],
    config: MacroEventBacktestConfig,
) -> dict[str, Any]:
    timestamps = [candle.timestamp for candle in candles]
    equity = config.initial_equity
    peak = equity
    max_drawdown = 0.0
    trades: list[dict[str, Any]] = []
    monthly: dict[str, dict[str, Any]] = {}

    for event in events:
        direction = _event_direction(event, config)
        if direction == Direction.FLAT:
            continue
        entry_timestamp = event.timestamp + timedelta(seconds=max(0, config.entry_delay_seconds))
        entry_index = bisect.bisect_left(timestamps, entry_timestamp)
        if entry_index >= len(candles):
            continue
        if abs((candles[entry_index].timestamp - entry_timestamp).total_seconds()) > config.max_entry_gap_seconds:
            continue

        entry_candle = candles[entry_index]
        entry_price = _entry_execution_price(config, entry_candle.open, direction)
        notional = min(
            equity * max(0.0, config.margin_pct) * max(1.0, config.leverage),
            max(0.0, config.max_position_notional_usdt),
        )
        if notional <= 0:
            continue
        quantity = notional / entry_price
        entry_fee = _fee(config, notional)
        equity -= entry_fee

        stop_price = entry_price * (1.0 - config.stop_loss_pct) if direction == Direction.LONG else entry_price * (1.0 + config.stop_loss_pct)
        take_profit_price = entry_price * (1.0 + config.take_profit_pct) if direction == Direction.LONG else entry_price * (1.0 - config.take_profit_pct)
        exit_index = min(len(candles) - 1, entry_index + max(1, config.max_holding_minutes))
        exit_price = candles[exit_index].close
        exit_reason = "time_stop"
        for index in range(entry_index, exit_index + 1):
            candle = candles[index]
            if direction == Direction.LONG:
                hit_stop = candle.low <= stop_price
                hit_target = candle.high >= take_profit_price
            else:
                hit_stop = candle.high >= stop_price
                hit_target = candle.low <= take_profit_price
            if hit_stop:
                exit_index = index
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if hit_target:
                exit_index = index
                exit_price = take_profit_price
                exit_reason = "take_profit"
                break

        executed_exit = _exit_execution_price(config, exit_price, direction)
        gross_pnl = direction.value * quantity * (executed_exit - entry_price)
        exit_fee = _fee(config, abs(quantity * executed_exit))
        net_pnl = gross_pnl - entry_fee - exit_fee
        equity += gross_pnl - exit_fee
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 0.0 if peak <= 0 else (peak - equity) / peak)
        trade = {
            "symbol": symbol,
            "event_type": event.event_type,
            "reference": event.reference,
            "event_time": event.timestamp.isoformat(),
            "entry_time": entry_candle.timestamp.isoformat(),
            "exit_time": candles[exit_index].timestamp.isoformat(),
            "direction": direction.name,
            "surprise": event.surprise,
            "actual": event.actual,
            "forecast": event.forecast,
            "entry_price": entry_price,
            "exit_price": executed_exit,
            "notional": notional,
            "gross_pnl": gross_pnl,
            "fees": entry_fee + exit_fee,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / max(notional, 1e-12),
            "reason": exit_reason,
            "holding_minutes": exit_index - entry_index,
            "strategy_bucket": "macro_event",
        }
        trades.append(trade)
        _record_monthly(monthly, trade)

    summary = _summary(config.initial_equity, equity, max_drawdown, trades)
    return {
        "symbol": symbol,
        "summary": summary,
        "monthly": _monthly_summary(monthly),
        "event_types": _event_type_summary(trades),
        "trades": trades,
    }


def _event_direction(event: MacroEvent, config: MacroEventBacktestConfig) -> Direction:
    surprise = event.surprise
    if event.event_type == "NFP":
        threshold = abs(config.nfp_min_surprise_k)
    elif event.event_type == "CPI_YOY":
        threshold = abs(config.cpi_min_surprise_pct)
    else:
        return Direction.FLAT
    if abs(surprise) < threshold:
        return Direction.FLAT
    # Stronger USD / rates pressure is usually a short-term headwind for crypto.
    return Direction.SHORT if surprise > 0 else Direction.LONG


def _load_symbol_1m(data_dir: str | Path, symbol: str) -> list[Candle]:
    root = Path(data_dir)
    matches = sorted(root.glob(f"{symbol}_1m_*.csv"))
    if not matches:
        raise FileNotFoundError(f"no 1m data found for {symbol} in {root}")
    return load_candles_csv(matches[-1])


def download_macro_event_windows(
    output_dir: str | Path,
    events_path: str | Path,
    symbols: Iterable[str],
    minutes_before: int = 45,
    minutes_after: int = 120,
    sleep_seconds: float = 0.25,
) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = load_macro_events(events_path)
    written: list[str] = []
    for symbol in _normalize_symbols(symbols):
        by_timestamp: dict[Any, Candle] = {}
        for event in events:
            start = event.timestamp - timedelta(minutes=max(1, minutes_before))
            end = event.timestamp + timedelta(minutes=max(1, minutes_after))
            candles = download_binance_futures_klines(
                symbol,
                "1m",
                start,
                end,
                sleep_seconds=sleep_seconds,
            )
            for candle in candles:
                by_timestamp[candle.timestamp] = candle
        candles = [by_timestamp[timestamp] for timestamp in sorted(by_timestamp)]
        path = output / f"{symbol}_1m_macro_events_{candles[0].timestamp:%Y%m%d}_{candles[-1].timestamp:%Y%m%d}.csv"
        write_candles_csv(path, candles)
        written.append(str(path))
    return written


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    output: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        if symbol not in output:
            output.append(symbol)
    return output


def _entry_execution_price(config: MacroEventBacktestConfig, price: float, direction: Direction) -> float:
    slip = max(0.0, config.slippage_bps) / 10_000.0
    return price * (1.0 + slip) if direction == Direction.LONG else price * (1.0 - slip)


def _exit_execution_price(config: MacroEventBacktestConfig, price: float, direction: Direction) -> float:
    slip = max(0.0, config.slippage_bps) / 10_000.0
    return price * (1.0 - slip) if direction == Direction.LONG else price * (1.0 + slip)


def _fee(config: MacroEventBacktestConfig, notional: float) -> float:
    return abs(notional) * max(0.0, config.fee_bps) / 10_000.0


def _summary(starting_equity: float, final_equity: float, max_drawdown: float, trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [trade for trade in trades if trade["net_pnl"] > 0]
    losses = [trade for trade in trades if trade["net_pnl"] <= 0]
    gross_profit = sum(trade["net_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl"] for trade in losses))
    return {
        "initial_equity": starting_equity,
        "final_equity": final_equity,
        "net_pnl": final_equity - starting_equity,
        "net_return_pct": 0.0 if starting_equity <= 0 else (final_equity / starting_equity - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": 0.0 if not trades else len(wins) / len(trades) * 100.0,
        "profit_factor": None if gross_loss == 0 else gross_profit / gross_loss,
        "avg_trade_pnl": 0.0 if not trades else sum(trade["net_pnl"] for trade in trades) / len(trades),
    }


def _record_monthly(monthly: dict[str, dict[str, Any]], trade: dict[str, Any]) -> None:
    month = trade["exit_time"][:7]
    bucket = monthly.setdefault(
        month,
        {
            "pnl": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "long": {"trades": 0, "pnl": 0.0},
            "short": {"trades": 0, "pnl": 0.0},
        },
    )
    pnl = float(trade["net_pnl"])
    side = "long" if trade["direction"] == "LONG" else "short"
    bucket["pnl"] += pnl
    bucket["trades"] += 1
    bucket["wins" if pnl > 0 else "losses"] += 1
    bucket[side]["trades"] += 1
    bucket[side]["pnl"] += pnl


def _monthly_summary(monthly: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for month in sorted(monthly):
        bucket = monthly[month]
        trades = int(bucket["trades"])
        rows.append(
            {
                "month": month,
                "pnl": bucket["pnl"],
                "trades": trades,
                "wins": int(bucket["wins"]),
                "losses": int(bucket["losses"]),
                "win_rate_pct": 0.0 if trades <= 0 else bucket["wins"] / trades * 100.0,
                "long_trades": int(bucket["long"]["trades"]),
                "long_pnl": bucket["long"]["pnl"],
                "short_trades": int(bucket["short"]["trades"]),
                "short_pnl": bucket["short"]["pnl"],
            }
        )
    return rows


def _event_type_summary(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for event_type in sorted({trade["event_type"] for trade in trades}):
        event_trades = [trade for trade in trades if trade["event_type"] == event_type]
        summary = _summary(0.0, sum(trade["net_pnl"] for trade in event_trades), 0.0, event_trades)
        rows.append(
            {
                "event_type": event_type,
                "trades": summary["trades"],
                "net_pnl": summary["final_equity"],
                "win_rate_pct": summary["win_rate_pct"],
                "profit_factor": summary["profit_factor"],
                "long_trades": len([trade for trade in event_trades if trade["direction"] == "LONG"]),
                "short_trades": len([trade for trade in event_trades if trade["direction"] == "SHORT"]),
            }
        )
    return rows


def _macro_score(summary: dict[str, Any]) -> float:
    drawdown_penalty = float(summary["max_drawdown_pct"]) * 0.35
    sparse_penalty = max(0.0, 8.0 - float(summary["trades"])) * 0.15
    win_bonus = float(summary["win_rate_pct"]) * 0.01
    return float(summary["net_pnl"]) - drawdown_penalty - sparse_penalty + win_bonus


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest fixed-time US macro event crypto scalps")
    parser.add_argument("--data-dir", default="data/binance_1m_365d_macro")
    parser.add_argument("--events", default="data/macro_events_us_major_2025_2026.csv")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--initial-equity", type=float, default=160.0)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--write-report", default=None)
    parser.add_argument("--download-windows", action="store_true")
    parser.add_argument("--minutes-before", type=int, default=45)
    parser.add_argument("--minutes-after", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    symbols = [part.strip() for part in args.symbols.replace("，", ",").split(",") if part.strip()]
    if args.download_windows:
        payload = download_macro_event_windows(
            args.data_dir,
            args.events,
            symbols,
            minutes_before=args.minutes_before,
            minutes_after=args.minutes_after,
            sleep_seconds=args.sleep,
        )
    elif args.optimize:
        payload = optimize_macro_event_backtest(args.data_dir, args.events, symbols, args.initial_equity, args.top)
    else:
        cfg = replace(MacroEventBacktestConfig(), initial_equity=args.initial_equity)
        payload = run_macro_event_backtest(args.data_dir, args.events, symbols, cfg)

    if args.write_report:
        Path(args.write_report).write_text(_format_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _format_report(payload: Any) -> str:
    if isinstance(payload, list):
        lines = ["# Macro Event Optimization", ""]
        for index, row in enumerate(payload, 1):
            summary = row["summary"]
            cfg = row["config"]
            lines.append(
                f"{index}. {row['symbol']} events={','.join(cfg['event_types'])} "
                f"delay={cfg['entry_delay_seconds']}s hold={cfg['max_holding_minutes']}m "
                f"sl={cfg['stop_loss_pct'] * 100:.2f}% tp={cfg['take_profit_pct'] * 100:.2f}% "
                f"trades={summary['trades']} pnl={summary['net_pnl']:.4f}U "
                f"win={summary['win_rate_pct']:.2f}% dd={summary['max_drawdown_pct']:.2f}%"
            )
        return "\n".join(lines) + "\n"

    lines = ["# Macro Event Backtest", ""]
    for result in payload.get("symbols", []):
        summary = result["summary"]
        lines.append(
            f"## {result['symbol']}\n"
            f"- PnL: {summary['net_pnl']:.4f}U ({summary['net_return_pct']:.2f}%)\n"
            f"- Trades: {summary['trades']} Win: {summary['win_rate_pct']:.2f}% "
            f"PF: {summary['profit_factor']}\n"
            f"- Max DD: {summary['max_drawdown_pct']:.2f}%"
        )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
