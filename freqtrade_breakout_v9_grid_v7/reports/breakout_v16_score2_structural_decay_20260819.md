# Breakout V16 Score-2 structural-decay exit research (2026-08-19)

## Scope and selected rule

This is a research-only `custom_exit` overlay on the frozen
`BreakoutV16GridV15PrecisionGuardGlobalLiveParityFreqtrade` class. It does not
change entries, indicators, leverage, stake sizing, Grid logic, existing stops,
or any existing exit. The inherited V16 + Grid V15 exit is evaluated first.

The selected overlay exits only a Breakout **Score-2 short** when every item is
true:

- the campaign has been open for at least 8 hours;
- completed-path MFE has always remained below 1.25R;
- the latest completed 1h close is no better than 0R for the short;
- the completed 1h candle has range >= 0.65 ATR, lower wick >= 30%, closes in
  its top 20%, has bullish body >= 0.15 ATR, and reclaims the previous close by
  >= 0.10 ATR;
- the latest completed close has made no net downside progress versus four
  completed hourly closes earlier;
- the decision occurs in the first five minutes after the 1h candle completes.

No forming 4h candle is read. The four-hour decay measure is assembled from
five fully completed 1h closes. Six-hour and ten-hour sensitivity neighbours
are kept as separate research classes; the selected boundary is eight hours.

## Exact current-DASH replay (1m detail, same stake and leverage)

Timerange: 2026-08-16 through available data at 2026-08-19 08:00 UTC; wallet
200 USDT, max two positions, fee 0.05% each side.

| Metric | Frozen V16 + Grid V15 | Selected overlay |
|---|---:|---:|
| Trades | 6 | 6 |
| Net PnL | -9.8894 U (-4.94%) | -6.0164 U (-3.01%) |
| PF (U-weighted) | 0.5064 | 0.6277 |
| Max relative drawdown | 8.21% | 6.34% |
| Wins / losses | 2 / 4 | 2 / 4 |

DASH itself used the same 29.77 entry, 88.646129 U stake, and 10x leverage:

| | Frozen | Selected |
|---|---:|---:|
| Exit time (UTC) | 2026-08-19 08:48 | 2026-08-19 07:00 |
| Hold | 708 min | 600 min |
| Exit price | 29.90 | 29.77 |
| PnL ratio | -5.3717% | -1.0005% |
| PnL | -4.7594 U | -0.8865 U |

The looser five-hour maturity boundary exited DASH at 02:00 UTC and a worse
29.79 price. Both six-hour and eight-hour boundaries waited for the 07:00 UTC
confirmed structure and produced the same current-DASH result. Eight hours is
kept because it is the boundary directly used in the cross-year combined run.

## Historical exact affected-trade replays (1m detail)

The causal scan of the frozen 2024-01-01 through 2026-08-11 path found only two
qualifying historical trades. Targeted exact replays gave:

| Trade | Frozen exit | Selected exit | Frozen PnL | Selected PnL | Direct saving |
|---|---|---|---:|---:|---:|
| JTO short, 2025-03-20 | 00:03 UTC, precision soft stop | 00:00 UTC, structure | -42.9506 U | -27.6450 U | +15.3055 U |
| DASH short, 2025-05-24 | 10:50 UTC, precision soft stop | 09:00 UTC, structure | -52.3725 U | -27.2522 U | +25.1203 U |
| **Total** | | | **-95.3231 U** | **-54.8972 U** | **+40.4259 U** |

The JTO targeted replay used current 10x exchange metadata while the frozen
archive used 5x. Its selected ratio and U result above are converted back to
the frozen 5x stake/risk scale. DASH used 10x in both.

## Full-period production-detail baseline and causal projection

The exact frozen production-detail archive contains 530 trades from 2024-01-01
through 2026-08-11. A new full 1m engine run takes roughly 80+ minutes. The
candidate column below is therefore a **causal compounding projection**, not a
second full-engine result: it substitutes the two exact replay outcomes, then
proportionally carries the resulting equity difference through the unchanged
frozen trade path.

| Metric | Exact frozen baseline | Causal projection |
|---|---:|---:|
| Trades | 530 | 530 |
| Net PnL | 3,190,459.94 U | 3,199,333.52 U |
| Total return | 1,595,229.97% | 1,599,666.76% |
| Win rate | 55.47% | 55.47% |
| PF (U-weighted) | 8.38655 | 8.38762 |
| Max relative drawdown | 25.78% | 25.78% |

Projected net-equity improvement is 8,873.58 U. Only 40.43 U is the direct
two-trade saving; the rest is the later high-equity compounding effect. Because
Grid contains minimum-order and partial-tail discontinuities, this projection
must not be presented as an exact full-engine result.

Projected annual breakdown on the frozen path:

| Year | Trades | Long / short | Baseline PnL | Projected PnL | Baseline PF | Projected PF |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 219 | 57 / 162 | 8,612.87 U | 8,612.87 U | 3.6132 | 3.6132 |
| 2025 | 165 | 38 / 127 | 6,388.01 U | 6,430.29 U | 1.4775 | 1.4813 |
| 2026 to Aug 11 | 146 | 41 / 105 | 3,175,459.05 U | 3,184,290.35 U | 8.6470 | 8.6470 |

## Direct 15m execution-sensitivity runs

A direct baseline/main-candidate run and a separate strict-shape-neighbour run
were completed over the same 2024-01-01 through 2026-08-11 period with 15m
detail while retaining Precision Guard's 1m data lookup. The strict neighbour
raises the 1h range floor to 0.80 ATR and the lower-wick floor to 50%, so only
the historical DASH event triggers. These are secondary validations, not
production-detail parity.

| Metric | Frozen | Main candidate | Strict shape |
|---|---:|---:|---:|
| Trades | 517 | 515 | 514 |
| Net PnL | 2,800,442.09 U | 2,829,673.03 U | 2,755,328.43 U |
| Total return | 1,400,221.04% | 1,414,836.52% | 1,377,664.21% |
| PF (U-weighted) | 8.3880 | 8.2734 | 8.4841 |
| PF (sum of trade-return ratios) | 2.2314 | 2.2425 | 2.2728 |
| Max relative drawdown | 24.17% | 23.29% | 21.71% |
| Account drawdown | 2.716% | 2.696% | 2.730% |

The main candidate improves net PnL, both drawdown measures, and trade-return
PF, but reduces U-weighted PF. The strict shape improves both PF definitions
and relative drawdown, but loses 45,113.66 U versus the frozen baseline and
slightly worsens account drawdown. It is therefore rejected rather than chosen
just for its headline PF.

Both overlays change later stake amounts. Grid minimum-tail and partial-exit
boundaries then create a different path (the main candidate has three
baseline-only trades and one candidate-only trade). This is why the direct
15m results and the frozen-path projection are both reported, with neither
disguised as the other.

## Decision

The main overlay is a meaningful research improvement for the exact DASH
failure mode and both historical qualifying events. It is intentionally sparse
and causal. The strict neighbour confirms that tightening thresholds can trade
away too much portfolio profit even when PF looks better. Further threshold
search against these one or two events would be overfitting.

Neither candidate dominates the frozen strategy on net profit, both PF views,
and both drawdown views at once. The main candidate remains the better research
balance, but it is **not promoted to GUI/live**. The frozen production strategy
and GUI remain unchanged.

Verification: 24 focused tests pass, including frozen-method identity, every
rule boundary, completed-candle causality, and unfinished-hour exclusion.
