# 30m Backtest, Daily 10% Stop Only

Data range: 2025-06-05 to 2026-06-05, Binance USD-M futures 30m candles.
Symbols: 60 configured symbols.
Initial equity: 100 USDT.

## Risk Settings

- Daily loss stop: 10% from the current day's starting equity.
- Starting-capital drawdown stop: disabled.
- Weekly profit drawdown stop: disabled.
- Soft drawdown size reduction: disabled.
- Minimum symbol margin pct: 0.3%.
- Max position notional: 10000 USDT.

The minimum symbol margin pct was reduced from 1.0% to 0.3% because the old 1.0% setting conflicts with the 10000 USDT max-position cap once account equity grows above about 33333 USDT. That conflict blocked all June entries in the first daily-only run.

## Summary

| Metric | Result |
| --- | ---: |
| Initial equity | 100.00 |
| Final equity | 45431.65 |
| Net return | 45331.65% |
| Max drawdown | 20.61% |
| Total trades | 12884 |
| Win rate | 88.44% |
| Profit factor | 1.534 |
| Scale-ins | 506 |
| Long PnL | 26166.35 |
| Long trades | 5741 |
| Long win rate | 87.86% |
| Short PnL | 19165.30 |
| Short trades | 7143 |
| Short win rate | 88.91% |

## Strategy Buckets

| Strategy | Trades | PnL | Win | PF | Long PnL | Short PnL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Super-volume startup | 1011 | 10687.17 | 89.02% | 2.512 | 7478.75 | 3208.42 |
| Breakout / breakdown | 1997 | 11527.89 | 88.23% | 1.782 | 8965.80 | 2562.09 |
| Indicator reversal | 8216 | 15847.66 | 88.51% | 1.310 | 6304.47 | 9543.19 |
| Pullback reclaim / reject | 1619 | 6813.71 | 87.89% | 1.579 | 3417.32 | 3396.39 |
| RSI reversal | 41 | 455.22 | 92.68% | 3.571 | 0.00 | 455.22 |

## Monthly Long/Short Table

June 2025 and June 2026 are partial months because the downloaded 365-day range is 2025-06-05 through 2026-06-05.

| Month | Start | End | PnL | Return | Open/Close | Long PnL | Long O/C | Long Win | Short PnL | Short O/C | Short Win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-06 | 100.00 | 186.85 | 86.85 | 86.85% | 768/764 | 54.23 | 333/331 | 90.03% | 33.02 | 435/433 | 91.22% |
| 2025-07 | 187.74 | 429.80 | 242.06 | 128.93% | 1161/1165 | 198.42 | 538/540 | 88.70% | 44.12 | 623/625 | 88.16% |
| 2025-08 | 429.80 | 674.76 | 244.96 | 56.99% | 1047/1045 | 81.39 | 439/439 | 86.33% | 160.85 | 608/606 | 89.11% |
| 2025-09 | 676.04 | 776.76 | 100.73 | 14.90% | 1059/1059 | 80.53 | 494/493 | 84.38% | 23.64 | 565/566 | 85.16% |
| 2025-10 | 775.93 | 1105.27 | 329.34 | 42.44% | 985/981 | 157.21 | 402/399 | 87.22% | 187.53 | 583/582 | 89.52% |
| 2025-11 | 1111.73 | 2835.99 | 1724.26 | 155.10% | 1058/1063 | 896.36 | 473/477 | 88.05% | 820.48 | 585/586 | 90.78% |
| 2025-12 | 2919.22 | 4207.46 | 1288.24 | 44.13% | 1051/1049 | 987.58 | 442/439 | 88.15% | 402.55 | 609/610 | 86.39% |
| 2026-01 | 4219.08 | 9124.99 | 4905.91 | 116.28% | 1156/1158 | 3153.13 | 521/523 | 88.72% | 1763.10 | 635/635 | 88.03% |
| 2026-02 | 9154.86 | 13263.39 | 4108.52 | 44.88% | 906/907 | 824.24 | 364/365 | 86.58% | 3295.01 | 542/542 | 91.14% |
| 2026-03 | 13263.39 | 17712.98 | 4449.59 | 33.55% | 1155/1152 | 563.88 | 515/513 | 84.41% | 3930.12 | 640/639 | 89.83% |
| 2026-04 | 17785.63 | 25892.52 | 8106.89 | 45.58% | 1172/1168 | 7468.47 | 561/557 | 89.05% | 868.36 | 611/611 | 87.23% |
| 2026-05 | 25874.01 | 43148.39 | 17274.39 | 66.76% | 1215/1220 | 11740.58 | 603/608 | 92.43% | 5318.54 | 612/612 | 90.03% |
| 2026-06 | 43150.32 | 45431.65 | 2281.33 | 5.29% | 151/153 | -39.67 | 56/57 | 80.70% | 2317.98 | 95/96 | 95.83% |

