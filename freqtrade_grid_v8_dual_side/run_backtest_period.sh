#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 STRATEGY LABEL TIMERANGE [FEE_PER_SIDE]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PROJECT="$PROJECT_DIR/../freqtrade_breakout_v9_grid_v7"
FREQTRADE_BIN="$BASE_PROJECT/.runtime/conda/bin/freqtrade"
STRATEGY="$1"
LABEL="$2"
TIMERANGE="$3"
FEE_PER_SIDE="${4:-0.0008}"
RESULT_DIR="$PROJECT_DIR/user_data/backtest_results/$STRATEGY/$LABEL"

mkdir -p "$RESULT_DIR"

"$FREQTRADE_BIN" backtesting \
  --config "$PROJECT_DIR/config.active50.backtest.json" \
  --userdir "$PROJECT_DIR/user_data" \
  --strategy "$STRATEGY" \
  --datadir "$BASE_PROJECT/user_data/data/binance" \
  --timerange "$TIMERANGE" \
  --timeframe 1h \
  --timeframe-detail 1m \
  --max-open-trades 1 \
  --dry-run-wallet 200 \
  --fee "$FEE_PER_SIDE" \
  --cache none \
  --enable-protections \
  --export trades \
  --backtest-directory "$RESULT_DIR" \
  --breakdown month \
  --notes "$STRATEGY $LABEL active50 dual-side grid fee=$FEE_PER_SIDE"
