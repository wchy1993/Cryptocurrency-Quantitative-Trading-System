# Breakout V16 GUI/LIVE path-parity acceptance

Date: 2026-08-11

## Decision

Accepted for the fixed-50 GUI DRY-RUN/LIVE entry point. The production adapter
inherits the selected V16 research class and adds only the existing completed-
hour synchronization gate in live-like modes.

## Classes compared

- Research: `BreakoutV16Fixed50MtfAdaptiveBreakoutMax2ResearchFreqtrade`
- GUI/LIVE: `BreakoutV16Fixed50MtfAdaptiveBreakoutMax2LiveParityFreqtrade`

The production adapter does not reimplement signals, candidate ranks, stake,
leverage, stops, no-follow exits, protected-runner stops or ordinary exits.
Backtest mode bypasses only the wall-clock batch gate.

## Execution contract

- Timerange: 2026-07-01 through 2026-08-01
- Fixed active 50-pair universe
- 200 USDT initial wallet, unlimited compound stake
- Isolated futures, 10x strategy leverage, maximum two open trades
- 1h signal clock, causal 15m path data and 1m order replay
- Historical 2026 Binance leverage tiers
- 0.08% fee per side

## Result

| Class | Trades | Net profit | PF | Win rate | Max wallet drawdown |
|---|---:|---:|---:|---:|---:|
| V16 research | 19 | +217.33348029U | 4.489912 | 52.63% | 8.63% |
| V16 GUI/LIVE adapter | 19 | +217.33348029U | 4.489912 | 52.63% | 8.63% |

The complete exported trade objects are exactly equal. Their canonical JSON
SHA-256 is
`90f0cbea150e644d9c214cf3bfb44362060bd4d318af8aa686ad02d9f99c2b9b`.

Accepted archive:
`user_data/backtest_results/v16_gui_path_parity_20260811/backtest-result-2026-08-11_12-30-44.zip`

Archive SHA-256:
`46b459b97245bd4bda94538eb3acfa71197a2c75ef298f510c0e2032a3482084`

## LIVE-specific safety boundary

- Entry requires all 50 pairs to finish the same just-closed hourly batch.
- Entry is rejected if context is missing, stale, mismatched or more than 90
  seconds past the hourly boundary.
- The four completed 15m candles belong to the signal hour and never include a
  future quarter.
- `v16_no_follow_watch` is written to Freqtrade trade custom data on the first
  filled entry and therefore persists in the dedicated V16 SQLite ledger.
- V16 DRY-RUN and LIVE use separate ledgers and do not reuse V15 history.
