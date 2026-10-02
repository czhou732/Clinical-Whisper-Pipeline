#!/usr/bin/env bash
# Render docs/icon/icon.html to clinicalwhisper.icns (the app and dock icon).
set -euo pipefail
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TMP=$(mktemp -d)
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --default-background-color=00000000 \
    --window-size=1024,1024 --screenshot="$TMP/icon_1024.png" "file://$PWD/icon.html" 2>/dev/null
SET="$TMP/clinicalwhisper.iconset"
mkdir -p "$SET"
for s in 16 32 128 256 512; do
    sips -z $s $s "$TMP/icon_1024.png" --out "$SET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d "$TMP/icon_1024.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$SET" -o ../../clinicalwhisper.icns
cp "$TMP/icon_1024.png" icon_1024.png
rm -rf "$TMP"
echo "wrote clinicalwhisper.icns"
