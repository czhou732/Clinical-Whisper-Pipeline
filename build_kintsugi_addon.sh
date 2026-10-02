#!/bin/bash
# Build "ClinicalWhisper Kintsugi.dmg": Kintsugi's open Depression-Anxiety Model
# (KintsugiHealth/dam 3.1, Apache-2.0, 0.74 GB) as an add-on (see kintsugi_dam.py).
set -e
SRC=$(python3 -c "from huggingface_hub import try_to_load_from_cache as t; print(t('KintsugiHealth/dam','dam3.1.ckpt') or '')")
if [ -z "$SRC" ] || [ ! -f "$SRC" ]; then
    echo "ERROR: KintsugiHealth/dam is not in the local Hugging Face cache."; exit 1
fi
OUT="ClinicalWhisper Kintsugi.dmg"; STAGE=".addon_stage_kintsugi"; FOLDER="$STAGE/ClinicalWhisper Kintsugi"
trap 'rm -rf "$STAGE"' EXIT
rm -rf "$STAGE" "$OUT"; mkdir -p "$FOLDER"
cp -L "$SRC" "$FOLDER/dam3.1.ckpt"
cp -L "$(dirname "$SRC")/README.md" "$FOLDER/KINTSUGI_README.md"
cat > "$FOLDER/README.txt" <<'TXT'
ClinicalWhisper Kintsugi add-on

Kintsugi Health's open Depression-Anxiety Model (DAM 3.1, Apache-2.0). It reads
the participant's voice, not their words, and estimates PHQ-9 and GAD-7 bands.
Research estimate only: trained on Kintsugi's data, not validated on this
population or on clinical interviews, and not a diagnosis.

To install: open ClinicalWhisper, click "Install an add-on…" at the bottom of
the window, and choose this folder.
TXT
hdiutil create -volname "ClinicalWhisper Kintsugi" -srcfolder "$STAGE" -ov -format UDRO "$OUT"
ls -lh "$OUT"
