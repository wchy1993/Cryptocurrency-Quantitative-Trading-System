# CMIPR: Cross-Sectional Momentum Ignition Pyramid

Strategy name: `cross_sectional_momentum_ignition_pyramid`

CMIPR is an independent multi-timeframe futures trend strategy. It disables the
legacy VBP, indicator reversal, super-volume, and fast-breakout signal paths. It
uses closed 4h/1h/30m/15m/5m candles for decisions and the existing path-aware
1m full-cost engine for execution.

## State machine

Signal state:

`IDLE -> COMPRESSION_WATCH -> IGNITION_PENDING -> ENTRY_CONFIRMING -> INITIAL_ENTRY_READY`

Order safety state:

`ORDER_PENDING -> PARTIAL_FILL -> PROTECTION_PENDING -> PROTECTED`

Cancellation and recovery are explicit:

`CANCEL_PENDING`, `RECOVERY_AFTER_RESTART`, and `EXITING`.

Position state:

`PROTECTED -> ADDON_1_ARMED -> ADDON_1_POSITION -> ADDON_2_ARMED -> FULL_POSITION -> RUNNER -> EXITING -> COOLDOWN`

An entry or add-on fill without an exchange protection stop transitions to
`EXITING`; live execution must immediately reduce or flatten it. A process
restart reconciles exchange position and protective orders before new entries.

## Model variants

`core` uses only historical OHLCV and the full-cost execution data. Missing OI,
taker, funding, or basis values are not replaced with zero.

`derivatives_enhanced` may use OI, taker flow, funding, and basis only when the
coverage audit spans every configured symbol and the complete evaluation
period. It refuses to start otherwise. The current local OI/taker history covers
approximately 2026-05-20 through 2026-06-08 and no basis archive is present, so
the year and three-month studies must use `core`.

## Regime gate

The market gate classifies `BULL_EXPANSION`, `BEAR_EXPANSION`, `NEUTRAL`,
`OVERHEATED_BULL`, `OVEREXTENDED_BEAR`, or `CHAOS_NO_TRADE` from BTC/ETH trend,
cross-sectional breadth, and shock/conflict checks.

Hysteresis is mandatory:

- Entry requires multiple distinct closed 1h bars.
- Entry and exit use different EMA slope and breadth thresholds.
- An active regime has a minimum hold time.
- Exit requires multiple distinct closed 1h bars.
- Repeated scans of the same 1h bar never increment confirmation.

## Ranking, compression, and ignition

Ranking combines 30m/1h/4h return, relative strength versus BTC, volume trend,
and EMA alignment. Longs use only the strongest percentile; shorts use only the
weakest percentile and reduced risk.

Compression uses 30m ATR percentile, ATR versus its recent mean, channel width
in ATR, volume contraction, prior move, and failed-breakout count. Ignition uses
a closed 15m breakout, ATR-normalized distance/body, close location, wick,
volume expansion, and MACD histogram continuation.

The default entry waits for the first healthy 5m pullback. It requires a bounded
pullback depth, lower normalized volume, breakout-level and EMA9 reclaim/reject,
a strong close, a structural stop, and a chase guard. Confirmation is executable
no earlier than the next 1m open.

## Winner-only pyramiding

Initial/add-on risk fractions default to 40/30/30. An add-on requires:

- Positive full-cost executable `current_r` at the current market exit fill.
- A new closed 5m structure break with volume confirmation.
- A genuinely improved structural stop outside the configured ATR noise floor.
- Execution no earlier than the next 1m open.
- Recalculated worst full-cost loss within the original trade risk budget.

Worst loss includes existing and new quantities, entry fees, exit fee, entry
slippage, spread/market impact represented by the existing fill model, stop
slippage, and adverse accrued funding. Quantity is reduced or rejected if the
budget is exceeded. The stop is never moved closer merely to manufacture add-on
capacity, and no losing position can qualify.

## 1m event order and exits

Each 1m event is processed in this order:

1. Existing position stop/exit using the old protection state.
2. Add-ons confirmed on an earlier closed bar.
3. Initial entries confirmed on an earlier closed bar.
4. New signal and add-on decisions, queued for a future 1m open.

Same-bar conflicts use the adverse path. A stop change becomes effective on the
next 1m bar. Exits include structural stop, fail-fast, breakout-level failure,
full-cost breakeven, fixed-R exit when runner is disabled, segmented maximum
profit giveback, closed-15m EMA runner trailing, and time stop.

## Risk and research

Sizing is stop-risk based. The original risk budget is persisted on the position
and cannot be increased by quality score or add-ons. Controls include maximum
positions, same-direction exposure, global entry interval, symbol cooldown,
consecutive-loss pause, daily loss stop, soft drawdown reduction, and hard
drawdown stop.

Each staged-search phase is capped by `max_experiments_per_stage`. Parameters
within the configured PF tolerance are selected by lower complexity and then
lower drawdown. Results below `min_research_trades` are not selectable.

Required stress variants are baseline, fixed risk, no compounding, extra
entry/add-on delay, higher costs, no/one/two add-ons, and no runner. Analytical
top-five-winner and top-symbol exclusions are also reported with an explicit
warning that deleting trades does not replay the portfolio path.

## Time validation policy

The historical workflow is train -> validation -> historical test. Parameters
may be selected only on validation. The 2026-04 through 2026-06 interval has
already been used in prior research and is explicitly not an untouched final
holdout. Final acceptance requires new post-freeze shadow/dry-run observations.
