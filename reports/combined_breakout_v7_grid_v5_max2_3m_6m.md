# Breakout v7 + Grid v5 Shared Max2 Backtest

- Initial equity: `200.00U`
- Windows: 3m `2026-04-19..2026-07-19`; 6m `2026-01-19..2026-07-19`
- 50 symbols; each strategy max one position; shared account max two
- Gap-free 1m execution; full fee, slippage and funding; adverse stop first
- Standalone rows each start from 200U separately; their profits must not be added as one 200U portfolio

| Period | Mode | Trades | Net | PF | Win rate | Max DD |
|---|---|---:|---:|---:|---:|---:|
| 3 months | Breakout v7 standalone | 46 | +8229.43U | 6.898 | 45.65% | 29.07% |
| 3 months | Grid v5 standalone | 22 | +201.73U | 66.061 | 90.91% | 5.84% |
| 3 months | Shared account max2 | 68 | +15997.33U | 9.121 | 60.29% | 29.07% |
| 6 months | Breakout v7 standalone | 84 | +28371.33U | 6.497 | 39.29% | 30.73% |
| 6 months | Grid v5 standalone | 39 | +401.44U | 11.615 | 87.18% | 9.82% |
| 6 months | Shared account max2 | 123 | +78737.99U | 8.668 | 54.47% | 31.61% |

## Shared-account contribution

- 3 months: Breakout `46` trades, `+11763.31U`, PF `7.031`.
- 3 months: Grid `22` trades, `+4234.02U`, PF `218.266`.
- 3 months: two-position time share `12.53%`; reverse-priority net `+15997.33U`, PF `9.121`, DD `29.07%`.
- 6 months: Breakout `84` trades, `+57880.73U`, PF `6.747`.
- 6 months: Grid `39` trades, `+20857.25U`, PF `106.838`.
- 6 months: two-position time share `10.81%`; reverse-priority net `+78737.99U`, PF `8.668`, DD `31.61%`.
