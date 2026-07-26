#!/bin/bash
set -e

echo "Installing packaging dependencies..."
uv pip install pyinstaller pywebview fastapi uvicorn python-multipart

echo "Cleaning up previous builds..."
rm -rf build dist ClinicalWhisper.dmg ClinicalWhisper.spec

echo "Building ClinicalWhisper.app with PyInstaller..."
# --collect-all, not --hidden-import: hidden imports only add .py modules to the
# archive. These packages also ship data files and native binaries — OpenSMILE's
# SMILExtract binary and eGeMAPSv02 .conf files, OpenMED's clinical term lists —
# and without them the app launches but every analysis fails.
uv run pyinstaller --noconfirm \
    --windowed \
    --name "ClinicalWhisper" \
    --icon "clinicalwhisper.icns" \
    --add-data "www:www" \
    --add-data "config.example.yaml:." \
    --collect-all opensmile \
    --collect-all audresample \
    --collect-all audobject \
    --collect-all audformat \
    --collect-all audmath \
    --collect-all audiofile \
    --collect-all audeer \
    --collect-all moss_transcribe_diarize \
    --collect-all openmed \
    --collect-all mlx_lm \
    --collect-all mlx \
    --collect-all soundfile \
    --collect-all transformers \
    --collect-all tokenizers \
    --hidden-import "uvicorn" \
    --hidden-import "fastapi" \
    --hidden-import "multipart" \
    launcher.py

echo "Verifying the bundle contains the ML backends..."
# Pure-python modules live in the PYZ archive, so check for each package's
# data/binary payload instead — that is what --hidden-import used to omit.
# opensmile/core/bin holds SMILExtract; without it acoustics cannot run.
# audresample ships its own .dylib that OpenSMILE loads at runtime; without it
# acoustic extraction fails and every acoustic column comes back null.
MISSING=""
for pkg in opensmile/core/bin opensmile/core/config audresample/core/bin \
           moss_transcribe_diarize openmed mlx_lm _soundfile_data; do
    if ! find dist/ClinicalWhisper.app/Contents -maxdepth 5 -path "*/$pkg" | grep -q .; then
        MISSING="$MISSING $pkg"
    fi
done

# Every native lib these packages need must actually be inside the bundle.
for dylib in libSMILEapi.dylib libaudresample.dylib; do
    if ! find dist/ClinicalWhisper.app/Contents -name "$dylib" | grep -q .; then
        MISSING="$MISSING $dylib"
    fi
done
if [ -n "$MISSING" ]; then
    echo "ERROR: bundle is missing:$MISSING"
    echo "The app would launch but every analysis would fail. Aborting."
    exit 1
fi
echo "All ML backends present."

echo "Creating DMG..."
# Check if hdiutil is available (macOS)
if command -v hdiutil &> /dev/null; then
    mkdir -p dist/dmg_folder
    cp -r dist/ClinicalWhisper.app dist/dmg_folder/
    ln -s /Applications dist/dmg_folder/Applications

    echo "Running hdiutil to package .app into .dmg..."
    hdiutil create -volname "ClinicalWhisper" \
        -srcfolder dist/dmg_folder \
        -ov -format UDZO \
        ClinicalWhisper.dmg

    echo "Successfully created ClinicalWhisper.dmg in the current directory!"
    echo
    echo "NOTE: model weights are NOT bundled. On first run the app downloads"
    echo "      ~1.7 GB (MOSS) + ~4.5 GB (Llama-3 4-bit) + ~0.2 GB (OpenMED)."
    echo "      ffmpeg must also be installed on the target machine."
else
    echo "hdiutil not found. DMG creation skipped (are you on macOS?). The .app is available in the dist/ folder."
fi
