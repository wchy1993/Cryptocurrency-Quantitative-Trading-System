#!/bin/zsh
set -eu

script_dir="${0:A:h}"
project_dir="${script_dir:h}"
asset_dir="$project_dir/assets"
svg_path="$asset_dir/coin_app_icon.svg"
app_resources="$project_dir/V15 Breakout Max2 Trader.app/Contents/Resources"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/v11-adaptive-grid-v8-icon.XXXXXX")"

cleanup() {
  rm -rf "$temp_dir"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$app_resources"
/usr/bin/qlmanage -t -s 1024 -o "$temp_dir" "$svg_path" >/dev/null 2>&1
rendered="$temp_dir/coin_app_icon.svg.png"
if [[ ! -f "$rendered" ]]; then
  echo "无法从 SVG 生成应用图标"
  exit 1
fi

/usr/bin/sips -s format png "$rendered" --out "$asset_dir/coin_app_icon_1024.png" >/dev/null
/usr/bin/sips -z 256 256 "$rendered" --out "$asset_dir/coin_app_icon_256.png" >/dev/null

iconset="$temp_dir/AppIcon.iconset"
mkdir -p "$iconset"
/usr/bin/sips -z 16 16 "$rendered" --out "$iconset/icon_16x16.png" >/dev/null
/usr/bin/sips -z 32 32 "$rendered" --out "$iconset/icon_16x16@2x.png" >/dev/null
/usr/bin/sips -z 32 32 "$rendered" --out "$iconset/icon_32x32.png" >/dev/null
/usr/bin/sips -z 64 64 "$rendered" --out "$iconset/icon_32x32@2x.png" >/dev/null
/usr/bin/sips -z 128 128 "$rendered" --out "$iconset/icon_128x128.png" >/dev/null
/usr/bin/sips -z 256 256 "$rendered" --out "$iconset/icon_128x128@2x.png" >/dev/null
/usr/bin/sips -z 256 256 "$rendered" --out "$iconset/icon_256x256.png" >/dev/null
/usr/bin/sips -z 512 512 "$rendered" --out "$iconset/icon_256x256@2x.png" >/dev/null
/usr/bin/sips -z 512 512 "$rendered" --out "$iconset/icon_512x512.png" >/dev/null
/usr/bin/sips -z 1024 1024 "$rendered" --out "$iconset/icon_512x512@2x.png" >/dev/null
/usr/bin/iconutil -c icns "$iconset" -o "$asset_dir/AppIcon.icns"
/usr/bin/install -m 0644 "$asset_dir/AppIcon.icns" "$app_resources/AppIcon.icns"
touch "$project_dir/V15 Breakout Max2 Trader.app"

echo "icon=$asset_dir/AppIcon.icns"
echo "app=$project_dir/V15 Breakout Max2 Trader.app"
