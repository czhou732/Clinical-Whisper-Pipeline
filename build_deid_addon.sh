#!/bin/bash
# Build "ClinicalWhisper Name Silencing.dmg": the word aligner that lets the app
# silence masked names word by word (facebook/wav2vec2-base-960h, Apache-2.0,
# 0.36 GB; see audio_deid.py). Without it, whole segments are silenced instead.
set -e
MODEL="models--facebook--wav2vec2-base-960h"; HF_HUB="$HOME/.cache/huggingface/hub"
REV=$(cat "$HF_HUB/$MODEL/refs/main" 2>/dev/null || true)
[ -n "$REV" ] && [ -d "$HF_HUB/$MODEL/snapshots/$REV" ] || { echo "ERROR: $MODEL not cached."; exit 1; }
OUT="ClinicalWhisper Name Silencing.dmg"; STAGE=".addon_stage_deid"; FOLDER="$STAGE/ClinicalWhisper Name Silencing"
trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"; mkdir -p "$FOLDER/hub/$MODEL/snapshots"
cp -R "$HF_HUB/$MODEL/refs" "$FOLDER/hub/$MODEL/"
rsync -aL "$HF_HUB/$MODEL/snapshots/$REV" "$FOLDER/hub/$MODEL/snapshots/"
cat > "$FOLDER/README.txt" <<'TXT'
ClinicalWhisper Name Silencing add-on

Lets "Also save a copy of the audio with the masked names silenced" silence
each masked word precisely (English), instead of the whole segment it is in.
Model: facebook/wav2vec2-base-960h (Apache-2.0).

To install: open ClinicalWhisper, click "Install an add-on…", choose this folder.
TXT
hdiutil create -volname "ClinicalWhisper Name Silencing" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
