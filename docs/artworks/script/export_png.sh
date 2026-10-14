#!/bin/bash
# Exports every unsim-*.svg in unsim/ to unsim/png/ at 1x and 2x with headless Chrome (same procedure as script/export_png.sh).
# Usage (Git Bash): bash unsim/script/export_png.sh
# The background is transparent unless the SVG paints one; the Chrome profile is kept in the project's tmp/ so the user's profile is not touched.
cd "$(dirname "$0")/.." || exit 1
DIR=$(pwd -W 2>/dev/null || pwd)
PROJ=$(cd .. && (pwd -W 2>/dev/null || pwd))
C="/c/Program Files/Google/Chrome/Application/chrome.exe"
mkdir -p png ../tmp
for f in unsim-*.svg; do
    stem=${f%.svg}
    w=$(head -1 "$f" | grep -o 'width="[0-9]*"' | grep -o '[0-9]*')
    h=$(head -1 "$f" | grep -o 'height="[0-9]*"' | grep -o '[0-9]*')
    for s in 1 2; do
        suf=""; [ $s = 2 ] && suf="@2x"
        "$C" --headless=new --disable-gpu --hide-scrollbars --no-first-run --no-default-browser-check --user-data-dir="$PROJ/tmp/chrome-profile" --default-background-color=00000000 --force-device-scale-factor=$s --window-size=$w,$h --screenshot="$DIR/png/${stem}${suf}.png" "file:///$DIR/$f" 2>&1 | grep -o 'bytes written to file.*'
    done
done
