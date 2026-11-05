#!/bin/bash
# Build "ClinicalWhisper Scoring.dmg": the clinical scoring model as an add-on
# for the base app (see addons.py). The disk holds one folder, "ClinicalWhisper
# Scoring", which the app's "Add clinical scoring…" button (or
# `ClinicalWhisper --batch --install-scoring FOLDER`) copies into place.
set -e

MODEL="models--mlx-community--Meta-Llama-3-8B-Instruct-4bit"
HF_HUB="$HOME/.cache/huggingface/hub"
OUT="ClinicalWhisper Scoring.dmg"
STAGE=".addon_stage"
FOLDER="$STAGE/ClinicalWhisper Scoring"

REV=$(cat "$HF_HUB/$MODEL/refs/main" 2>/dev/null || true)
if [ -z "$REV" ] || [ ! -d "$HF_HUB/$MODEL/snapshots/$REV" ]; then
    echo "ERROR: $MODEL is not in the local cache. Run one scored file first."
    exit 1
fi

trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"
mkdir -p "$FOLDER/hub/$MODEL/snapshots"
cp -R "$HF_HUB/$MODEL/refs" "$FOLDER/hub/$MODEL/"
# Dereference the cache's symlinks into real files: the add-on is copied
# somewhere else on the user's Mac, where the blobs/ folder does not exist.
rsync -aL "$HF_HUB/$MODEL/snapshots/$REV" "$FOLDER/hub/$MODEL/snapshots/"

cat > "$FOLDER/README.txt" <<'EOF'
ClinicalWhisper Scoring add-on

This adds clinical scoring to ClinicalWhisper. To install it:

1. Open ClinicalWhisper.
2. Click "Add clinical scoring…" under the Transcribe only option.
3. Choose this "ClinicalWhisper Scoring" folder.

Copying takes about a minute and needs about 5.5 GB of free space. After that
you can eject this disk; the app keeps its own copy.
EOF

echo "Creating $OUT ($(du -sh "$FOLDER" | cut -f1))..."
# The weights are already compressed, so a plain read-only image is as small
# as a compressed one and much faster to build and open.
hdiutil create -volname "ClinicalWhisper Scoring" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
