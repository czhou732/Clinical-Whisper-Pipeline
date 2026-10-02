#!/bin/bash
# Build "ClinicalWhisper Languages.dmg": masking for recordings not in English
# (OpenMed's multilingual privacy filter, MLX build; see addons.py). Installed
# with the app's "Install an add-on…" link or `--batch --install-addon FOLDER`.
set -e

NAME="OpenMed_privacy-filter-multilingual-mlx"
SRC="$HOME/.cache/openmed/$NAME"
OUT="ClinicalWhisper Languages.dmg"
STAGE=".addon_stage_languages"
FOLDER="$STAGE/ClinicalWhisper Languages"

if [ ! -f "$SRC/weights.safetensors" ]; then
    echo "ERROR: $SRC is missing. Mask one non-English recording from source first,"
    echo "or download OpenMed/privacy-filter-multilingual-mlx into that folder."
    exit 1
fi

trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"
mkdir -p "$FOLDER/openmed"
rsync -aL "$SRC" "$FOLDER/openmed/"
cat > "$FOLDER/README.txt" <<'TXT'
ClinicalWhisper Languages add-on

Lets ClinicalWhisper mask names and other identifiers in recordings that are
not in English: Arabic, Bengali, Chinese, Dutch, French, German, Hindi,
Italian, Japanese, Korean, Portuguese, Spanish, Telugu, Turkish and Vietnamese.
Without it, such recordings are refused rather than written unmasked.

To install: open ClinicalWhisper, click "Install an add-on…" at the bottom of
the window, and choose this "ClinicalWhisper Languages" folder.

Clinical scores and the review keyword screen stay English-only.
TXT
hdiutil create -volname "ClinicalWhisper Languages" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
