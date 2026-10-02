#!/bin/bash
# Build "ClinicalWhisper Praat.dmg": Praat voice measures (praat-parselmouth,
# GPL-3.0) as a separate add-on, because ClinicalWhisper itself is MIT and
# does not bundle GPL code. See praat_measures.py and addons.py.
set -e
SO=$(.venv/bin/python -c "import parselmouth; print(parselmouth.__file__)")
VER=$(.venv/bin/python -c "import importlib.metadata as m; print(m.version('praat-parselmouth'))")
OUT="ClinicalWhisper Praat.dmg"; STAGE=".addon_stage_praat"; FOLDER="$STAGE/ClinicalWhisper Praat"
trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"; mkdir -p "$FOLDER/site"
# parselmouth is one compiled module (.so) for this Python; copy it and its metadata.
cp "$SO" "$FOLDER/site/"
cp -R "$(dirname "$SO")"/praat_parselmouth-*.dist-info "$FOLDER/site/"
curl -s https://www.gnu.org/licenses/gpl-3.0.txt -o "$FOLDER/LICENSE-GPL3.txt"
cat > "$FOLDER/SOURCE.txt" <<TXT
praat-parselmouth $VER (GPL-3.0). Source code:
https://github.com/YannickJadoul/Parselmouth/tree/v$VER
Praat (GPL-3.0): https://github.com/praat/praat
TXT
cat > "$FOLDER/README.txt" <<'TXT'
ClinicalWhisper Praat add-on

Adds Praat voice measures for each speaker, defined as in senselab and the
Bridge2AI-Voice project: pitch (with a speaker-adapted range), harmonics-to-
noise ratio, CPPS, jitter, shimmer and spectral slope.

It is separate from the app because Praat is licensed GPL-3.0 (see
LICENSE-GPL3.txt and SOURCE.txt). To install: open ClinicalWhisper, click
"Install an add-on…" at the bottom of the window, and choose this folder.
TXT
hdiutil create -volname "ClinicalWhisper Praat" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
