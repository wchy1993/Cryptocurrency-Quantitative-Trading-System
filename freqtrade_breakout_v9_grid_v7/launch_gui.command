#!/bin/zsh
set -eu

script_dir="${0:A:h}"
cd "$script_dir"

app_bundle="$script_dir/V16 Breakout MTF Max2 Trader.app"
app_executable="$app_bundle/Contents/MacOS/V16BreakoutMtfMax2Trader"
if [[ -d "$app_bundle" && -x "$app_executable" ]]; then
  exec /usr/bin/open "$app_bundle"
fi

runtime_python="$script_dir/.runtime/conda/bin/python"
if [[ ! -x "$runtime_python" ]]; then
  echo "缺少 Freqtrade 独立 Python：$runtime_python"
  read -r "?按回车关闭..."
  exit 1
fi

exec "$runtime_python" "$script_dir/freqtrade_gui.py"
