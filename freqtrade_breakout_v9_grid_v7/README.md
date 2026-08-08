# Breakout v11 Adaptive + 双向 Grid v8 — Freqtrade GUI

The desktop GUI in this directory now runs
`BreakoutV11AdaptiveGridV8DualSideFreqtrade`. It combines the selected
Breakout v11 Adaptive implementation with the unchanged long/short Grid v8
source. Former strategy files, databases, logs and the v10F application bundle
remain available for comparison; the v11 GUI uses its own runtime locks,
overlays, logs and SQLite ledgers.

The release manifest freezes the combined adapter, both local Breakout
dependencies, the sibling Grid v8 dependency and the GUI/backtest
configuration. A source mismatch blocks engine startup instead of silently
running a different strategy.

Freqtrade owns the exchange model, order lifecycle, Binance futures
constraints, leverage/liquidation calculations, funding, position
adjustments, stop execution, dry-run/live state and historical backtesting.

## Current active-50 assumptions

- the exact same 50-symbol Binance USDT-M whitelist is used by GUI and
  backtesting
- all 50 contracts were active when audited on 2026-07-29; SEI replaces the
  inactive TON contract
- 1h closed-candle signals and 1m execution detail
- 200 USDT initial equity with unlimited-stake compounding
- 10x isolated futures, shared maximum two positions
- maximum one Breakout and one Grid campaign at a time
- Breakout-first arbitration when only one slot remains
- conservative 0.08% bundled fee/slippage charge per side
- exact windows: 2026-04-19 to 2026-07-19 and 2026-01-19 to
  2026-07-19

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

The exact active-50 v11 Adaptive + dual-side Grid v8 validation is documented
in `reports/breakout_v11_adaptive_grid_v8_3m_6m_1y_20260801.md`. It preserves
the v10F/v8 trade sequence and metrics over three and six months, while the
one-year result is 178 trades, +260,230.62U, PF 13.715, 47.75% wins and 25.99%
wallet drawdown. These are historical simulations, not a profit forecast.

The standalone frozen Grid v7 active-50 versus active-100 scanner test is
documented in
`reports/grid_v7_only_active50_vs_active100_3m_6m_20260729.md`. The tested
active-100 pool lost money in both windows, so it remains a research-only
configuration and is not used by the GUI.

## DRY-RUN / LIVE desktop GUI

The standalone desktop console in this directory runs the selected
`BreakoutV11AdaptiveGridV8DualSideFreqtrade` class directly through Freqtrade:

```bash
./launch_gui.command
```

The recommended Finder launcher is `V11 Adaptive Grid V8 Trader.app`. It has a native
gold-coin application icon and opens without a Terminal window.
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
  and ignored, so Grid entry/TP counters advance exactly once and remain
  restart-safe.
- If a Grid take-profit would leave a post-precision remainder below Binance's
  minimum tradable value, the current v11 execution adapter promotes that
  reduction to a complete exit. This prevents Freqtrade from rejecting the
  entire profitable reduction while retrying an impossible dust remainder.
- GUI and engine PID locks prevent two consoles or two v11/v8 engines from
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
sequentially. The combined strategy now captures the exact closed-candle frame
being analyzed for each of the 50 pairs, waits until every pair has the same
latest timestamp, computes the frozen breadth/efficiency/BTC/ETH context once,
and re-analyzes that timestamp once before Freqtrade checks entries. It never
fills a current candle with context from an older candle. If any pair is
missing or stale, that cycle fails closed and retries on the next engine loop.

The synchronized live market-context adapter remains unchanged from the prior
GUI. Breakout v11 Adaptive adds the audited entry filters and causal portfolio
risk governor documented in the v11 report. Grid v8 signals, spacing, targets,
risk and normal position-management rules remain unchanged; only an
untradeable post-precision tail is promoted to a full exit. Backtesting uses
the same guard through the selected v11 entry point.

The v11 portfolio high-water mark is persisted separately for DRY-RUN and LIVE
using owner-only atomic state files. Backtest and hyperopt modes neither read
nor write this runtime state, so historical results stay reproducible. A
missing state file starts a new v11 session; a corrupt state file blocks startup
instead of silently resetting risk.

The frozen strategy currently uses Freqtrade software-managed dynamic stops
(`stoploss_on_exchange = false`). Keep the engine online during LIVE trading.
If a managed LIVE position exists, stopping requires the typed phrase
`STOP LIVE` and explicit manual takeover.

## Selected strategy composition

- removes score-one Breakout signals;
- rejects crowded short breakouts in highly efficient broad markets, with a
  bounded continuation exception for moderate score-two setups;
- rejects score-four long breakouts after an excessive four-hour market
  breadth expansion;
- uses liquidation-aware leverage on contracts whose exchange leverage cap is
  10x;
- arms selective profit floors only from completed 1h closes, preserving wide
  room for large trend runners;
- keeps the selected v10F rules in recent regimes, while applying the v11
  Adaptive filters and causal drawdown/recovery sizing in older adverse paths;
- replaces Grid v7 with the frozen dual-side Grid v8 implementation, allowing
  both long and short campaigns under the same one-Grid-position limit.

## Important semantic boundary

The combined adapter does not alter the selected Grid v8 component rules. A
Freqtrade `Trade` can maintain only one active
position-adjustment order at once. The original Grid campaign can keep several
refill and take-profit limits working simultaneously and can remain alive with
zero inventory. The adapter therefore uses a small keeper position when a
campaign must survive at zero logical lots. This is an explicit, auditable
approximation rather than an assertion of identical results.

Generated environments, market data, runtime databases, logs and backtest
archives remain inside this directory and are ignored by Git.

The 0.08% cost is a reproducible proxy, not an order-book replay. Freqtrade
uses 1m OHLC detail for fills and cannot model real queue position, latency,
spread expansion or market impact. Forward dry-run/live reconciliation is
still required before treating either simulator's PnL as a live forecast.
