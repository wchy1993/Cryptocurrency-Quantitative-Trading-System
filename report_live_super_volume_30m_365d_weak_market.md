# 30m Weak-Market Filter Backtest, 365d

Data range: 2025-06-05 to 2026-06-05, Binance USD-M futures 30m candles.
Symbols: 60 configured symbols, no extra symbols.
Initial equity: 100 USDT.

## Active Weak-Market Logic

- Weak-market check uses the latest 12 bars on 30m data.
- A weak market is confirmed only when breadth is at or below 40% and the 6-hour average return is at or below -0.6%.
- During confirmed weak markets, long entries must have rank score at least 5.6.
- Confirmed weak-market long size is reduced to 40% risk multiplier.
- Weak-market filter applies to longs only; shorts continue to use the existing strategy filters.

## 12-Month Summary

| Metric | Result |
| --- | ---: |
| Initial equity | 100.00 |
| Final equity | 26965.65 |
| Net return | 26865.65% |
| Max drawdown | 18.65% |
| Total trades | 10753 |
| Win rate | 87.68% |
| Profit factor | 1.586 |
| Long PnL | 18294.98 |
| Long trades | 5443 |
| Long win rate | 87.43% |
| Short PnL | 8570.68 |
| Short trades | 5310 |
| Short win rate | 87.93% |

## Monthly Long/Short Table

June 2025 and June 2026 are partial months because the downloaded 365-day range is 2025-06-05 through 2026-06-05.

| Month | Start | End | PnL | Return | Open/Close | Long PnL | Long O/C | Long Win | Short PnL | Short O/C | Short Win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-06 | 100.00 | 192.55 | 92.55 | 92.55% | 722/718 | 57.72 | 323/321 | 90.34% | 35.26 | 399/397 | 91.18% |
| 2025-07 | 193.48 | 386.87 | 193.39 | 99.96% | 935/939 | 154.97 | 509/511 | 88.06% | 38.93 | 426/428 | 86.21% |
| 2025-08 | 386.87 | 618.18 | 231.31 | 59.79% | 977/975 | 92.87 | 432/432 | 86.57% | 136.85 | 545/543 | 88.95% |
| 2025-09 | 618.94 | 630.25 | 11.31 | 1.83% | 918/919 | 4.50 | 492/491 | 84.73% | 8.86 | 426/428 | 85.28% |
| 2025-10 | 630.50 | 626.35 | -4.14 | -0.66% | 430/428 | 19.77 | 313/311 | 84.89% | -22.42 | 117/117 | 77.78% |
| 2025-11 | 627.34 | 1032.41 | 405.07 | 64.57% | 770/772 | 241.89 | 436/439 | 86.33% | 163.71 | 334/333 | 89.19% |
| 2025-12 | 1054.53 | 1323.18 | 268.65 | 25.48% | 836/834 | 167.78 | 404/401 | 87.03% | 128.94 | 432/433 | 85.68% |
| 2026-01 | 1326.83 | 2727.95 | 1401.13 | 105.60% | 1049/1051 | 930.39 | 509/511 | 88.06% | 473.59 | 540/540 | 87.04% |
| 2026-02 | 2736.74 | 3893.47 | 1156.73 | 42.27% | 745/746 | 188.55 | 342/343 | 85.13% | 971.34 | 403/403 | 90.32% |
| 2026-03 | 3893.47 | 5453.44 | 1559.97 | 40.07% | 1019/1016 | 393.88 | 496/494 | 85.22% | 1182.09 | 523/522 | 89.46% |
| 2026-04 | 5495.94 | 11940.93 | 6444.99 | 117.27% | 1071/1069 | 5972.80 | 527/524 | 89.12% | 599.33 | 544/545 | 86.79% |
| 2026-05 | 11949.80 | 26970.65 | 15020.85 | 125.70% | 1167/1171 | 11143.09 | 611/615 | 92.36% | 3791.67 | 556/556 | 89.03% |
| 2026-06 | 26958.08 | 26965.65 | 7.57 | 0.03% | 114/115 | -1073.22 | 49/50 | 78.00% | 1062.55 | 65/65 | 95.38% |

