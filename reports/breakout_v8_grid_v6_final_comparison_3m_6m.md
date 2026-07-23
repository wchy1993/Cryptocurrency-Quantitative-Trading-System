# Breakout v8 + Grid v6 final comparison

## Test contract

- Initial equity: `200.00U`
- Universe: frozen `50` symbols
- Windows: `2026-04-19..2026-07-19` and
  `2026-01-19..2026-07-19`
- Gap-free 1m execution with maker/taker fees, market/stop/take-profit
  slippage and historical funding
- Standalone: one position/campaign; shared account: Breakout max1 +
  Grid max1, globally max2
- Same-bar conflicts use adverse-stop-first ordering
- Frozen Breakout v7, Grid v5, active GUI, GUI state and APT Grid were
  not modified

## Standalone Pareto comparison

| Period | Strategy | Trades | Net | PF | Win | Max DD |
|---|---|---:|---:|---:|---:|---:|
| 3 months | Breakout v7 frozen | 46 | +8,229.43U | 6.898 | 45.65% | 29.07% |
| 3 months | Breakout v8 refined | 45 | +13,826.31U | 8.780 | 46.67% | 28.86% |
| 6 months | Breakout v7 frozen | 84 | +28,371.33U | 6.497 | 39.29% | 30.73% |
| 6 months | Breakout v8 refined | 83 | +50,281.48U | 8.382 | 39.76% | 30.57% |
| 3 months | Grid v5 frozen | 22 | +201.73U | 66.061 | 90.91% | 5.84% |
| 3 months | Grid v6 selected | 22 | +204.44U | 66.502 | 90.91% | 5.84% |
| 6 months | Grid v5 frozen | 39 | +401.44U | 11.615 | 87.18% | 9.82% |
| 6 months | Grid v6 selected | 38 | +461.20U | 33.787 | 89.47% | 7.93% |

## Shared 200U account, max two

| Period | Strategy pair | Trades | Net | PF | Win | Max DD |
|---|---|---:|---:|---:|---:|---:|
| 3 months | Breakout v7 + Grid v5 | 68 | +15,997.33U | 9.121 | 60.29% | 29.07% |
| 3 months | Breakout v8 + Grid v6 | 67 | +23,479.80U | 10.523 | 61.19% | 27.39% |
| 6 months | Breakout v7 + Grid v5 | 123 | +78,737.99U | 8.668 | 54.47% | 31.61% |
| 6 months | Breakout v8 + Grid v6 | 121 | +132,798.19U | 10.195 | 55.37% | 31.16% |

## Selected structural changes

- Breakout v8 uses the same v7 entries but applies entry-time
  confidence-score convex allocation: score-1 short `0.50x`, score-3
  short `0.85x`, score-4 short `1.10x`, score-5 short `2.60x`, capped
  at a total adjusted multiplier of `3.00x`.
- Breakout core stop is tightened from `0.80 ATR` to `0.77 ATR`;
  the selected profile disables the old `5R` breakeven jump while
  retaining the `15R/12R` giveback exit.
- Grid v6 keeps v5 campaign sizing and management, adding a causal
  minimum actual extension guard of `0.05 ATR`. It removes the anomalous
  low-extension campaign without raising the `11%` campaign risk cap.
- A finer Grid threshold and score-risk redistribution search found
  higher-profit alternatives, but each exceeded the selected v6
  drawdown in at least one window, so they were rejected.

## Robustness checks

- Both selected standalone strategies remained profitable under `1.5x`
  execution costs, an additional one-minute entry delay, fixed
  non-compounding risk, removal of the top contributing symbol, and the
  earlier three-month subwindow.
- Breakout v8 refined under `1.5x` costs: 3m `+11,155.56U`, PF `8.036`;
  6m `+36,711.20U`, PF `7.629`.
- Grid v6 under `1.5x` costs: 3m `+182.76U`, PF `19.282`; 6m
  `+410.76U`, PF `17.146`.
