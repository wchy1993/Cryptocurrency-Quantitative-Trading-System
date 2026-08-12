# Breakout v10F optimization — 2026-07-29

## Scope

- Only the isolated `freqtrade_breakout_v9_grid_v7` directory was changed.
- The frozen `BreakoutV9GridV7Freqtrade` source was not modified.
- Grid v7 signal, sizing, campaign and exit rules were not modified.
- Both comparison windows use 50 pairs, 200 USDT initial equity, shared Max2,
  1h completed-candle signals, 1m execution detail, isolated futures, funding
  and a 0.08% per-side fee/slippage proxy.

## Shared-account results

| Window | Strategy | Trades | Net profit | PF | Win rate | Wallet max drawdown | Liquidations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 months | v9 + Grid v7 baseline | 72 | +24,589.41U | 8.964 | 55.56% | 20.36% | 0 |
| 3 months | v10F + Grid v7 | 54 | +30,224.95U | 21.584 | 70.37% | 12.09% | 0 |
| 6 months | v9 + Grid v7 baseline | 130 | +102,437.31U | 7.236 | 53.85% | 23.95% | 2 |
| 6 months | v10F + Grid v7 | 95 | +221,529.29U | 16.639 | 67.37% | 18.87% | 0 |

## Selected-version attribution

| Window | Component | Trades | Net profit | PF | Win rate |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 months | Breakout v10F | 29 | +22,672.50U | 46.190 | 68.97% |
| 3 months | Frozen Grid v7 contribution | 25 | +7,552.45U | 8.813 | 72.00% |
| 6 months | Breakout v10F | 52 | +157,420.11U | 37.638 | 53.85% |
| 6 months | Frozen Grid v7 contribution | 43 | +64,109.18U | 7.496 | 83.72% |

The Grid contribution changes because both strategies share equity and slots;
its rules are unchanged.

## Structural changes

1. Remove score-one Breakout entries, whose six-month group had negative
   expectancy.
2. Reject short breakouts when broad-market efficiency indicates a mature,
   crowded selloff. A bounded score-two continuation exception requires
   moderate volume, symbol efficiency and EMA displacement.
3. Reject score-four long entries after market breadth expands by more than
   ten percentage points in four hours. This removes late correlated chasing
   while preserving score-five convex setups.
4. Use 5x on Breakout contracts whose exchange maximum is at most 10x, keeping
   the ATR stop inside the liquidation constraint.
5. Derive selective profit floors from completed 1h closes. Intrabar 1m wicks
   cannot arm these floors, and wide trends retain room to run.

## Stopping decision

The selected six-month Breakout component has PF 37.638 and total gross losses
of 4,296.63U. The portfolio's remaining maximum wallet drawdown is driven by a
frozen Grid v7 ONDO campaign, not Breakout. Remaining Breakout losses do not
form another stable, causally explainable cluster across both windows.
Additional filtering was therefore stopped to avoid fitting isolated trades.

These results are historical simulations, not a profit forecast. The extreme
ending balances depend on uncapped compounding, historical liquidity being
available at the modeled cost and the selected trend regime. Forward dry-run
reconciliation remains required before live use.
