# Breakout V16 MTF Adaptive vs frozen V15

Date: 2026-08-11
Status: research candidate only; GUI/LIVE remains on frozen V15.

## Frozen baseline and execution contract

- V15 LIVE SHA-256: `ecdd0550823527a4a01283c780cd4cd6e5482dca967039f1a21b3f1a78fef2c4`
- V15 research SHA-256: `f9e717b954e6ce0733811cb9434065b1d14954ccfee595162a3fe885ced98422`
- Both hashes still match `gui_manifest.json` after V16 development.
- Fixed current 50-pair universe, 200 USDT initial wallet, isolated futures,
  10x strategy leverage, maximum two concurrent positions, unlimited compound
  stake, 1h primary clock and 1m detail simulation.
- All accepted comparisons use 0.08% fee per side. A single exploratory run
  that accidentally used the exchange default fee was quarantined and excluded.

## What V16 changes

V16 does not change any V15 entry signal, score, allocation, leverage or normal
runner exit. It adds a causal intrahour path layer to a small subset of V15
breakout shorts:

1. Build four complete 15m candles from the same completed 1h signal candle.
2. Derive the first and second 30m phases from those four quarters.
3. Mark a bounded `no-follow` watch only when the hour is a slow orderly short
   grind (hourly directional move no more than 1%, close within 25% of the hour
   low, and all four quarters progress downward).
4. Keep the V15 entry. Exit after 15 minutes only if the position is at or below
   -0.45R and has never reached positive R. At 30 minutes, exit at or below
   -0.60R only if maximum progress is no more than +0.10R. The rule expires
   after 60 minutes.
5. If a watched trade reaches +1.25R, protect a +0.10R floor after costs so the
   same filter does not truncate a valid runner.

The row stamped at hour `T` uses only `[T, T+1h)` and orders at `T+1h`.
Incomplete quarters and future-hour candles are excluded.

## Completed-window comparison

| Window | Strategy | Trades | Net profit | PF | Win rate | Sharpe | Sortino | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3m: 2026-05-01..2026-08-01 | V15 frozen | 47 | +13,031.54U | 5.597 | 48.94% | 3.879 | 16.639 | 9.80% |
| 3m: 2026-05-01..2026-08-01 | V16 MTF | 47 | +13,202.04U | 5.912 | 48.94% | 3.941 | 17.343 | 8.63% |
| 6m: 2026-02-01..2026-08-01 | V15 frozen | 73 | +73,306.84U | 5.645 | 46.58% | 2.416 | 10.109 | 9.80% |
| 6m: 2026-02-01..2026-08-01 | V16 MTF | 73 | +74,254.00U | 5.962 | 46.58% | 2.452 | 10.588 | 8.63% |
| 12m: 2025-08-01..2026-08-01 | V15 frozen | 118 | +127,555.80U | 5.600 | 36.44% | 1.465 | 7.566 | 8.58% |
| 12m: 2025-08-01..2026-08-01 | V16 MTF | 118 | +129,242.95U | 5.911 | 36.44% | 1.486 | 8.045 | 7.40% |

In the 6m window, V15 and V16 long profit is identical at +37,466.69U.
Short profit improves from +35,840.15U to +36,787.31U. This isolates the
improvement to the intended short-side path management.

## Calendar walk-forward comparison

| Period | Strategy | Trades | Net profit | PF | Win rate | Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | V15 frozen | 112 | +900.34U | 2.709 | 26.79% | 0.937 | 21.91% |
| 2024 | V16 MTF | 112 | +904.30U | 2.710 | 27.68% | 0.938 | 21.91% |
| 2025 | V15 frozen | 78 | +58.85U | 1.330 | 16.67% | 0.247 | 22.04% |
| 2025 | V16 MTF | 78 | +59.09U | 1.332 | 16.67% | 0.248 | 22.05% |
| 2026-01-01..2026-08-01 | V15 frozen | 87 | +139,196.89U | 5.687 | 43.68% | 2.190 | 8.21% |
| 2026-01-01..2026-08-01 | V16 MTF | 87 | +140,993.10U | 6.008 | 43.68% | 2.221 | 7.03% |

The 2024 and 2025 drawdown differences are respectively +0.0079 and +0.0030
percentage points, effectively flat but reported rather than rounded away.
The material improvement is in 2026 and the completed 3m/6m/12m windows.

## Latest BTC full-universe replay

Full fixed-50 replay: 2026-08-09 through 2026-08-10, using the latest local
mainnet 1m archive. The isolated BTC-only replay was rejected because removing
the other 49 pairs changes V15 market-breadth context.

| Strategy | BTC open | BTC close | Hold | Exit | BTC PnL | Trade return |
|---|---|---|---:|---|---:|---:|
| V15 frozen | 2026-08-10 17:00 UTC @ 63,888.6 | 20:36 @ 64,100.6 | 216m | stop loss | -7.8598U | -4.9249% |
| V16 MTF | same | 17:15 @ 63,973.7 | 15m | `bo_v16_no_follow_15m` | -4.6847U | -2.9354% |

V16 preserves the same entry and reduces this replay loss by 3.1750U, or
40.4%. The preceding signal hour had four downward 15m phases and closed near
its low, so it was watched rather than rejected. In the first 15 minutes after
entry it never achieved positive R and reversed beyond -0.45R, triggering the
bounded failure exit. The other replay trade, PENGU long, is byte-for-byte
unchanged in entry, exit and PnL.

## Ablation decisions

- Observe-only MTF produces the exact V15 2024 result, confirming the data
  attachment itself does not shift entries or leak future candles.
- Hard-rejecting short exhaustion improved 2024 but harmed 2025 through slot
  replacement, so the final V16 keeps every V15 entry.
- Reducing stake on the flagged path was rejected because the best local
  settings lost 2026 profit or worsened drawdown.
- Moving the first check from 15m to 10m slightly improved one 2025 XMR trade
  but reduced 2024 profit from +904.30U to +901.28U, so 10m was rejected.

## Verification

- `tests/test_breakout_v16_mtf_adaptive.py`: 6 tests passed.
- V15 LIVE parity and research-adapter suite: 10 tests plus 7 subtests passed.
- `git diff --check`: passed for all V16 source and test files.
- Selected V16 remains research-only. Promotion to GUI/LIVE requires a separate
  parity adapter, persistent trade-state recovery and DRY-RUN acceptance test.
