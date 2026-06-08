# Leverage Sweep, 30m 365d

Data range: 2025-06-05 to 2026-06-05.
Symbols: 60 configured symbols.
Initial equity: 100 USDT.
Base config: `config.live.optimized_super_volume.json`.
Only `trading.leverage` was changed in memory for each run.

## Summary

| Leverage | Final Equity | Net Return | Max DD | Trades | Win | PF | Long PnL | Short PnL | June PnL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5x | 831.68 | 731.68% | 4.84% | 12263 | 88.81% | 1.599 | 374.37 | 357.32 | 41.80 |
| 10x | 4292.67 | 4192.67% | 8.55% | 12549 | 88.70% | 1.588 | 2278.42 | 1914.25 | 227.68 |
| 15x | 16855.11 | 16755.11% | 11.23% | 12663 | 88.65% | 1.623 | 9698.36 | 7056.75 | 1002.58 |
| 20x | 33985.91 | 33885.91% | 14.80% | 12755 | 88.58% | 1.639 | 20438.13 | 13447.78 | 1939.94 |
| 30x | 45431.65 | 45331.65% | 20.61% | 12884 | 88.44% | 1.534 | 26166.35 | 19165.30 | 2281.33 |
| 50x | 55543.82 | 55443.82% | 29.99% | 12881 | 88.24% | 1.492 | 31441.74 | 24002.09 | 2294.75 |

## Strategy PnL

| Leverage | Super Volume | Breakout | Indicator | Pullback | RSI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5x | 142.20 | 156.63 | 332.44 | 95.28 | 5.13 |
| 10x | 844.74 | 1094.03 | 1574.72 | 648.66 | 30.51 |
| 15x | 3567.79 | 5124.55 | 5114.11 | 2838.88 | 109.78 |
| 20x | 7056.48 | 9355.57 | 11780.42 | 5401.97 | 291.47 |
| 30x | 10687.17 | 11527.89 | 15847.66 | 6813.71 | 455.22 |
| 50x | 13744.60 | 13199.74 | 19796.34 | 8217.44 | 485.70 |

## Readout

- 20x had the best profit factor in this sweep: 1.639, with max drawdown 14.80%.
- 30x produced much higher final equity than 20x, but drawdown rose to 20.61% and profit factor fell to 1.534.
- 50x produced the highest final equity, but max drawdown rose to 29.99% and profit factor fell to 1.492.
- Super-volume startup stayed profitable at every leverage and scaled from 142.20 USDT at 5x to 13744.60 USDT at 50x.

