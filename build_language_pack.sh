#!/bin/bash
# Build "ClinicalWhisper <Language>.dmg": one OpenMed single-language masker
# (Apache-2.0, ~0.6 GB). Usage: ./build_language_pack.sh es|fr|de|it|tr
# The model must have been used once from source (so OpenMed has downloaded
# and converted it into ~/.cache/openmed).
set -e
CODE="$1"
MODEL=$(.venv/bin/python -c "import addons,sys; print(addons.LANGUAGE_PACKS.get('$CODE',''))")
[ -n "$MODEL" ] || { echo "Usage: $0 es|fr|de|it|tr"; exit 1; }
NAME=$(.venv/bin/python -c "print({'es':'Spanish','fr':'French','de':'German','it':'Italian','tr':'Turkish'}['$CODE'])")
CONV="${MODEL/\//_}"; HUBNAME="models--${MODEL//\//--}"; OM="$HOME/.cache/openmed"
[ -f "$OM/$CONV/weights.safetensors" ] || { echo "ERROR: $OM/$CONV missing; run the model once."; exit 1; }
OUT="ClinicalWhisper $NAME.dmg"; STAGE=".addon_stage_lang_$CODE"; FOLDER="$STAGE/ClinicalWhisper $NAME"
trap 'rm -rf ".addon_stage_lang_$CODE"' EXIT
rm -rf ".addon_stage_lang_$CODE"; rm -f "$OUT"
mkdir -p "$FOLDER/openmed" "$FOLDER/hub"
rsync -aL "$OM/$CONV" "$FOLDER/openmed/"
# Config and tokenizer only: the converted weights above are what runs.
rsync -aL --exclude '*.safetensors' --exclude '*.bin' --exclude 'blobs' "$OM/$HUBNAME" "$FOLDER/hub/"
cat > "$FOLDER/README.txt" <<TXT
ClinicalWhisper $NAME language pack

Masks names and other identifiers in $NAME recordings ($MODEL, Apache-2.0).
Without it, $NAME recordings are refused rather than written unmasked.
Clinical scores and the review keyword screen remain English-only.

To install: open ClinicalWhisper, click "Install an add-on…", choose this folder.
TXT
hdiutil create -volname "ClinicalWhisper $NAME" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
