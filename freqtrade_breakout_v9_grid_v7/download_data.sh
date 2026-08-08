#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FREQTRADE_BIN="$PROJECT_DIR/.runtime/conda/bin/freqtrade"

"$FREQTRADE_BIN" download-data \
  --config "$PROJECT_DIR/config.backtest.json" \
  --userdir "$PROJECT_DIR/user_data" \
  --datadir "$PROJECT_DIR/user_data/data/binance" \
  --trading-mode futures \
  --timeframes 1h 1m \
  --timerange 20250601-20260730 \
  --data-format-ohlcv feather
