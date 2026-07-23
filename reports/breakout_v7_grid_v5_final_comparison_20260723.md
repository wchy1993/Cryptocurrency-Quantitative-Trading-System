# Breakout v7 and Grid v5 final comparison

## Scope

- Frozen anchor: Breakout v6 + Grid v4, commit `a440442`
- Research window: 2026-01-19 through 2026-07-19
- Recent window: 2026-04-19 through 2026-07-19
- Universe: 50 symbols; initial equity: 200U; compounding enabled
- Execution: gap-free 1m bars, point-in-time features, funding, maker/taker fees,
  2 bps market/take-profit slippage and 5 bps stop slippage
- Stress checks: 1.5x trading costs, one-minute entry delay, fixed risk, early
  three-month window and removal of the best symbol
- Breakout v6, Grid v4, APT Grid and the active GUI configuration are unchanged.

## Strict anchor comparison

| Strategy | Period | Trades | Net profit | PF | Win rate | Max DD |
|---|---|---:|---:|---:|---:|---:|
| Breakout v6 anchor | 3 months | 49 | +3,481.58U | 4.847 | 42.86% | 29.15% |
| Breakout v7 | 3 months | 46 | +8,229.43U | 6.898 | 45.65% | 29.07% |
| Breakout v6 anchor | 6 months | 92 | +10,537.28U | 4.396 | 36.96% | 31.76% |
| Breakout v7 | 6 months | 84 | +28,371.33U | 6.497 | 39.29% | 30.73% |
| Grid v4 anchor | 3 months | 22 | +176.46U | 24.711 | 90.91% | 7.47% |
| Grid v5 | 3 months | 22 | +201.73U | 66.061 | 90.91% | 5.84% |
| Grid v4 anchor | 6 months | 40 | +307.54U | 6.456 | 85.00% | 13.95% |
| Grid v5 | 6 months | 39 | +401.44U | 11.615 | 87.18% | 9.82% |

Breakout v7 increases net profit by 136.4% in the three-month window and
169.2% in the six-month window while improving PF, win rate and normal-path
drawdown in both windows. Grid v5 increases net profit by 14.3% and 30.5%
respectively while improving or preserving win rate and materially reducing
drawdown.

## Pressure checks

| Strategy | Check | Anchor net | New net | New PF | New Max DD |
|---|---|---:|---:|---:|---:|
| Breakout | Early 3 months | +432.27U | +529.02U | 2.301 | 30.73% |
| Breakout | 1.5x costs, 6 months | +7,642.87U | +20,168.78U | 5.952 | 31.34% |
| Breakout | Entry +1m, 6 months | +1,653.30U | +1,831.42U | 2.522 | 42.83% |
| Breakout | Fixed risk, 6 months | +1,076.27U | +1,416.04U | 5.853 | 19.55% |
| Breakout | Remove best symbol, 6 months | +7,792.24U | +17,840.42U | 4.844 | 30.73% |
| Grid | Early 3 months | +80.44U | +112.60U | 4.394 | 9.82% |
| Grid | 1.5x costs, 6 months | +268.74U | +355.66U | 8.456 | 9.91% |
| Grid | Entry +1m, 6 months | +194.86U | +257.28U | 3.719 | 21.75% |
| Grid | Fixed risk, 6 months | +191.77U | +225.98U | 9.728 | 7.17% |
| Grid | Remove best symbol, 6 months | +273.21U | +356.93U | 10.441 | 9.82% |

Both selected versions pass the strict two-window improvement gate and every
pressure scenario remains profitable. Both reports mark the candidate as
`robust=True` and `stress_better=True`; Grid v5 also passes every active
confidence-tier profitability check.

## Design changes

Breakout v7 retains the frozen v6 signal and managed exit structure. It adds a
causal, point-in-time confidence allocator based on signal quality, candle body,
volume, breakout extension and directional breadth. Risk is scaled by direction
and confidence at order submission; no future bar or eventual trade outcome is
used. The optional one-minute confirmation path is implemented and tested, but
the selected configuration uses immediate entry.

Grid v5 retains the frozen v4 gate and grid mechanics. At campaign entry it
scores quality, alignment, extension, volume and market regime, freezes a
tier-specific risk/management profile for that campaign, and rejects the weak
tier. The selected profile keeps the v4 exit geometry; its improvement comes
from causal campaign selection and confidence-based risk allocation rather than
post-trade fitting.

## Limitations and release decision

- Breakout remains a low-win-rate, convex strategy. Its six-month top-five
  trades contribute 70.16% of profit, March remains slightly negative, and the
  one-minute-delay drawdown rises to 42.83%. The positive fixed-risk and
  remove-best-symbol checks reduce, but do not eliminate, concentration risk.
- Grid v5 has only 39 six-month and 22 three-month campaigns. The three-month PF
  of 66.06 is caused partly by a very small gross-loss denominator and should not
  be extrapolated as a live expectation.
- The local neighborhoods around both selected policies were tested. Further
  in-sample gains required sacrificing a strict window, drawdown, coverage or
  pressure stability, so optimization stops here to limit additional overfit.
- These are independent research candidates, not active GUI/live settings. A
  fresh forward dry-run is required before any separate live-integration change.
