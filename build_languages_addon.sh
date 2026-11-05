#!/bin/bash
# Build "ClinicalWhisper Languages.dmg": masking for recordings not in English
# (OpenMed's multilingual privacy filter v2, 8-bit MLX build, ~1.5 GB; see
# addons.py). Installed with the app's "Install an add-on…" link or
# `--batch --install-addon FOLDER`.
set -e

REPO="models--OpenMed--privacy-filter-multilingual-v2-mlx-8bit"
HUB="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
SRC="$HUB/$REPO"
OUT="ClinicalWhisper Languages.dmg"
STAGE=".addon_stage_languages"
FOLDER="$STAGE/ClinicalWhisper Languages"

REV="$(cat "$SRC/refs/main" 2>/dev/null || true)"
if [ -z "$REV" ] || [ ! -f "$SRC/snapshots/$REV/weights.safetensors" ]; then
    echo "ERROR: $SRC is missing. Download it first:"
    echo "  hf download OpenMed/privacy-filter-multilingual-v2-mlx-8bit"
    exit 1
fi

trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"
# One snapshot, links resolved to real files, in the Hugging Face layout the
# app links into its model cache at startup.
mkdir -p "$FOLDER/hub/$REPO/refs" "$FOLDER/hub/$REPO/snapshots"
printf '%s' "$REV" > "$FOLDER/hub/$REPO/refs/main"
rsync -aL "$SRC/snapshots/$REV" "$FOLDER/hub/$REPO/snapshots/"
cat > "$FOLDER/README.txt" <<'TXT'
ClinicalWhisper Languages add-on

Lets ClinicalWhisper mask names and other identifiers in recordings that are
not in English: Arabic, Bengali, Chinese, Dutch, French, German, Hindi,
Italian, Japanese, Korean, Portuguese, Spanish, Telugu, Turkish and Vietnamese.
Without it, such recordings are refused rather than written unmasked.

Masking is weaker in some languages than in English; see the measured
recall in the app's documentation before relying on it (Hindi especially).

To install: open ClinicalWhisper, click "Install an add-on…" at the bottom of
the window, and choose this "ClinicalWhisper Languages" folder.

Clinical scores and the review keyword screen stay English-only.
TXT
hdiutil create -volname "ClinicalWhisper Languages" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
