# Breakout V16 MTF + Grid V15 PF · 共享 Max2 — Freqtrade GUI

The desktop GUI in this directory runs
`BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade`. It composes the
selected V16 causal 15m Breakout management with the frozen Grid V15 PF-control
risk surface. The account has two shared slots: at most one Breakout campaign
and one Grid campaign, with Breakout priority when only one slot remains.
Former strategy files, databases, logs and application bundles remain
available for comparison; this GUI uses new overlays, logs and SQLite ledgers
and never reuses the pure-V16 or former V11/Grid ledger.

The release manifest freezes the production adapter, its complete V16/V15 lineage
and both GUI/backtest configurations. A source mismatch blocks engine startup
instead of silently running a different strategy.

Freqtrade owns the exchange model, order lifecycle, Binance futures
constraints, leverage/liquidation calculations, funding, position
adjustments, stop execution, dry-run/live state and historical backtesting.

## Current active-50 assumptions

- the exact same 50-symbol Binance USDT-M whitelist is used by GUI and
  backtesting
- all 50 contracts were active when audited on 2026-07-29; SEI replaces the
  inactive TON contract
- 1h closed-candle signals, four causal 15m path candles and 1m execution detail
- 200 USDT initial equity with unlimited-stake compounding
- 10x isolated futures, shared maximum two positions
- maximum one Breakout and one Grid campaign at a time
- Breakout-first arbitration when only one slot remains
- same-component duplicate campaigns are blocked; Breakout keeps its 2h
  pair cooldown and five-entry daily limit, while Grid keeps its own campaign
  cooldown and DCA contract
- the optional NFP event sleeve is disabled in this GUI
- conservative 0.08% bundled fee/slippage charge per side
- selected comparison windows: 2025-08-11 to 2026-08-11 and 2024-01-01
  to 2026-08-11

## Reproduce

The checked runtime uses Python 3.12.11 and Freqtrade 2026.6.

```bash
./download_data.sh
./run_backtest_periods.sh
./run_breakout_v10f_backtests.sh
./run_breakout_v11_adaptive_backtests.sh
./run_backtest.sh
./build_report.sh
```

`run_backtest_periods.sh` reproduces the frozen v9 3-month/6-month baseline.
`run_breakout_v10f_backtests.sh` reproduces the selected v10F active-50
3-month/6-month comparison under the same assumptions. `run_backtest.sh` runs
the independent one-year baseline window.

`run_breakout_v11_adaptive_backtests.sh` reproduces the selected v11 Adaptive
3-month, 6-month and one-year validation windows with cache disabled.

The selected optimization results and structural changes are documented in
`reports/breakout_v10f_optimization_20260729.md`. The active-50 rerun and its
comparison with the archived TON universe are documented in
`reports/active50_v10f_grid_v7_3m_6m_20260729.md`.

The selected v15 Breakout-only Max2 comparison is documented in
`reports/breakout_max2_v11_vs_v15_2024_20260731_20260811.md`. The continuous
2024-01-01 through 2026-07-31 path contains 291 trades, +772,008.43U net,
PF 6.890, 28.52% wins and 4.78% Freqtrade summary drawdown. The absolute USDT
result relies on unconstrained compounding and is not a profit forecast.

The V16 research selection is documented in
`reports/breakout_v16_mtf_vs_v15_20260811.md`. The former pure-V16 GUI
acceptance remains in `reports/breakout_v16_gui_live_path_parity_20260811.md`.

The selected V16 + Grid V15 PF composition and its one-year/full-history
results are documented in
`reports/breakout_v16_grid_v15_quality_pf_combined_1y_full_20260812.md`.
The current combined GUI/research path-parity acceptance is documented in
`reports/breakout_v16_grid_v15_gui_live_path_parity_20260812.md`.
The one-year run contains 226 trades with PF 7.064; the continuous 2024+
run contains 546 trades with PF 7.724. Its unconstrained-compounding returns
and multi-million-USDT simulated notionals are mathematical upper bounds, not
an executable profit forecast.

The standalone frozen Grid v7 active-50 versus active-100 scanner test is
documented in
`reports/grid_v7_only_active50_vs_active100_3m_6m_20260729.md`. The tested
active-100 pool lost money in both windows, so it remains a research-only
configuration and is not used by the GUI.

## DRY-RUN / LIVE desktop GUI

The standalone desktop console in this directory runs the selected
`BreakoutV16GridV15QualityPfCombinedLiveParityFreqtrade` class directly
through Freqtrade. The same class is selected by `config.gui.json` and
`config.backtest.json`. It inherits the frozen combined research class without
reimplementing signals, candidate ordering, component arbitration, risk, DCA,
stops or exits in the GUI:

```bash
./launch_gui.command
```

The existing Finder launcher is `V16 Breakout MTF Max2 Trader.app`; its visible
application name is now `V16 + Grid V15 Trader`. It has a native gold-coin
application icon and opens without a Terminal window.
`launch_gui.command` now forwards to the same app bundle for compatibility.
The console starts in DRY-RUN and is never opened automatically by setup or
tests.

The five primary controls are custom-rendered rather than macOS Aqua buttons,
so their faces remain visibly blue (DRY-RUN), orange (LIVE), purple (refresh),
green (start) and red (stop), including clear color-preserving disabled
states.

- DRY-RUN reads Binance mainnet public market data, uses a fixed 200 USDT
  simulated wallet and never submits exchange orders.
- LIVE uses the real Binance USDT-M account. Selecting LIVE and clicking the
  clearly labelled orange/green controls starts the safety preflight directly;
  no typed start phrase is required.
- Freqtrade first boots into `stopped`, then the GUI verifies the frozen
  strategy/config, futures mode, isolated margin, maximum two positions,
  account state and ledger reconciliation before calling the local `/start`
  endpoint.
- DRY-RUN and LIVE use separate SQLite ledgers.
- A manual LIVE account refresh establishes a new session-equity baseline, so
  the GUI shows `0.00 U` session profit at refresh instead of comparing the
  live account with the 200 USDT dry-run wallet.
- Existing exchange positions which cannot be matched to the dedicated
  Freqtrade ledger block LIVE startup to prevent accidental duplicate
  exposure.
- Stateful fill accounting claims each completed exchange order by its unique
  order ID. A repeated callback caused by a later Binance fee update is logged
  and ignored, keeping Breakout partial exits and Grid DCA/TP state restart-safe.
- GUI and engine PID locks prevent two consoles or two combined engines from
  trading the same account. If the GUI crashes while its Freqtrade child is
  still alive, a new console reports the active PID and blocks duplicate
  startup rather than taking over silently.
- Every GUI log line begins with local date and time to the second. Known key
  and secret values are redacted before display and persistence.

The console reads the existing parent `.env` without copying it. Supported
variable names are:

```text
BINANCE_FUTURES_API_KEY
BINANCE_FUTURES_API_SECRET
```

Secrets are passed only to the LIVE child process through its environment.
They are never stored in `config.gui.json`, `gui_manifest.json`, runtime
overlays or GUI logs.

`config.gui.json` is deliberately separate from `config.backtest.json`, but
both now keep the exact same active 50-symbol whitelist and reject markets
which Binance marks inactive. The former TON-based historical comparison
configuration is preserved as `config.backtest.legacy-ton50.json`.

Run the non-network acceptance checks without opening any window:

```bash
.runtime/conda/bin/python freqtrade_gui.py --check
.runtime/conda/bin/python -m unittest discover -s tests -v
```

An optional stopped-state DRY-RUN startup check connects only to Binance public
market endpoints, verifies the local API and account view, requires all 50
pairs to expose complete market-context fields on the same latest closed 1h
candle, submits no real orders, and removes its temporary ledger when complete:

```bash
.runtime/conda/bin/python tools/check_gui_dry_startup.py
```

### Live market-context synchronization

Freqtrade refreshes the whitelist together but analyzes its pairs
sequentially. The strategy captures the exact closed-candle frame
being analyzed for each of the 50 pairs, waits until every pair has the same
latest timestamp, computes the frozen breadth/efficiency/BTC/ETH context once,
and re-analyzes that timestamp once before Freqtrade checks entries. It never
fills a current candle with context from an older candle. If any pair is
missing or stale, that cycle fails closed and retries on the next engine loop.

The LIVE entry window is 90 seconds after the hourly boundary. This replaces
the inherited 30-second expiry which discarded otherwise-valid candidates
when all-pair synchronization took 31-56 seconds. Entry still fails closed
unless all 50 pairs have the exact just-completed candle; an older, newer,
missing or partially synchronized batch is never traded. The audited sample
contained 218 synchronized hours, averaged 24.932 seconds, peaked at 55.973
seconds and crossed the old 30-second limit 9.6% of the time.

Breakout V16 retains the audited V15 entry filters, risk allocation and causal
15m no-follow/profit-floor management. Grid V15 retains the same Grid signals,
ranking, leverage, DCA ladder and exits as the combined research run, with the
frozen PF-control initial-risk scales. Both components pass through the same
synchronized LIVE batch gate. Backtest mode bypasses only that wall-clock gate
and otherwise traverses the same production class.

The combined portfolio high-water mark is persisted separately for DRY-RUN and LIVE
using owner-only atomic state files. Backtest and hyperopt modes neither read
nor write this runtime state, so historical results stay reproducible. A
missing state file starts a new combined session; a corrupt state file blocks startup
instead of silently resetting risk.

The frozen strategy currently uses Freqtrade software-managed dynamic stops
(`stoploss_on_exchange = false`). Keep the engine online during LIVE trading.
If a managed LIVE position exists, stopping requires the typed phrase
`STOP LIVE` and explicit manual takeover.

## Selected strategy composition

- exact fixed-50 Breakout V15 Stable Selected signal/risk/exit lineage;
- V16 causal four-quarter 15m path state, bounded no-follow exit and watched
  runner protection;
- frozen Grid V15 PF-control initial sizing: score 4 at 0.60, score 5 at 0.25
  and Grid long at 0.35 before the existing causal quality axes;
- at most one Breakout plus one Grid campaign, with Breakout priority for the
  final free slot;
- existing component cooldowns, entry caps, Grid DCA and Grid TP accounting are
  retained;
- NFP signals cannot reach stake sizing or order submission;
- liquidation-aware dynamic leverage and all normal exits remain unchanged
  from the selected combined research path.

## Important semantic boundary

The LIVE/DRY-RUN difference is limited to exchange reality and the completed-
candle synchronization gate: actual order-book price, spread, latency, partial
fills, fees and funding can move fills away from a 1m OHLC simulation. Candidate
generation, ranking, Max2 occupancy, risk sizing and exits are not separately
reimplemented in the GUI.

Generated environments, market data, runtime databases, logs and backtest
archives remain inside this directory and are ignored by Git.

The 0.08% cost is a reproducible proxy, not an order-book replay. Freqtrade
uses 1m OHLC detail for fills and cannot model real queue position, latency,
spread expansion or market impact. Forward dry-run/live reconciliation is
still required before treating either simulator's PnL as a live forecast.
