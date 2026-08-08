#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREQTRADE_BIN="$PROJECT_DIR/.runtime/conda/bin/freqtrade"
STRATEGY="${V11_STRATEGY:-BreakoutV11AdaptiveGridV8DualSideFreqtrade}"

run_period() {
  local label="$1"
  local timerange="$2"
  local result_dir="$PROJECT_DIR/user_data/backtest_results/v11_adaptive_final_$label"

  mkdir -p "$result_dir"
  nice -n 15 "$FREQTRADE_BIN" backtesting \
    --config "$PROJECT_DIR/config.backtest.json" \
    --userdir "$PROJECT_DIR/user_data" \
    --strategy "$STRATEGY" \
    --datadir "$PROJECT_DIR/user_data/data/binance" \
    --timerange "$timerange" \
    --timeframe 1h \
    --timeframe-detail 1m \
    --dry-run-wallet 200 \
    --fee 0.0008 \
    --cache none \
    --enable-protections \
    --export trades \
    --backtest-directory "$result_dir" \
    --breakdown month \
    --notes "v11 adaptive Active50 200U Max2 exact $label"
}

run_period "3m" "20260419-20260719"
run_period "6m" "20260119-20260719"
run_period "1y" "20250719-20260719"
