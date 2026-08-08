# Grid v8 dual-side — isolated research

This folder develops a standalone long/short Grid successor without changing
the frozen Grid v7, the current Breakout v10F + Grid v7 GUI, or either active
configuration.

## Acceptance target

The comparison uses the same Freqtrade 2026.6 execution model and current
active-50 static universe as the standalone frozen Grid v7 control:

| Window | Grid v7 trades | Net profit | PF | Wallet max drawdown |
| --- | ---: | ---: | ---: | ---: |
| 3 months | 24 | +134.99U | 2.402 | 11.99% |
| 6 months | 42 | +902.15U | 5.538 | 19.51% |

The candidate must produce both long and short trades, exceed Grid v7 profit
and PF in both windows, and keep Freqtrade wallet-equity maximum drawdown at
or below 10%.

## Fixed execution assumptions

- exact windows `20260419-20260719` and `20260119-20260719`;
- 200 USDT initial wallet with unlimited compounding;
- current GUI active-50 static whitelist;
- one standalone Grid campaign at a time;
- 1h completed-candle signals and 1m intrabar execution detail;
- Binance USDT-M isolated futures and historical funding;
- 0.08% fee/slippage proxy per side and no backtest cache.

The selected research class is `GridV8DualSideSelected`. Run the frozen
comparison with:

```bash
./run_backtest_period.sh GridV8DualSideSelected final_3m 20260419-20260719
./run_backtest_period.sh GridV8DualSideSelected final_6m 20260119-20260719
```

The new strategy imports the frozen Grid v7 execution adapter from the sibling
project. The final manifest pins that dependency by SHA-256 so later changes
cannot be mistaken for the tested version.

## Selected result

The frozen `GridV8DualSideSelected` class passed every requested comparison:

| Window | Trades | Long / short | Net profit | PF | Win rate | Wallet max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 months | 22 | 5 / 17 | +311.10U | 5.772 | 81.82% | 6.97% |
| 6 months | 38 | 7 / 31 | +1,052.08U | 10.805 | 89.47% | 9.04% |

Both directions were profitable in both main windows and there were no
liquidations.  A non-overlapping early three-month slice returned +302.32U
with PF 9.025 and 9.04% drawdown.  Raising the modeled fee from 0.08% to
0.12% per side over six months still returned +924.95U with PF 7.071 and
7.37% drawdown.

The detailed comparison, monthly attribution, limitations, and exact archive
paths are in
[`reports/grid_v8_dual_side_3m_6m_20260729.md`](reports/grid_v8_dual_side_3m_6m_20260729.md).
Machine-readable metrics are in the adjacent JSON report and all tested source
and result hashes are pinned in `selected_manifest.json`.

This remains a historical research result, not a live-profit guarantee.  The
main selection windows were also used during development, so the early-period,
higher-fee, and local-threshold checks reduce but do not eliminate overfitting
risk.
