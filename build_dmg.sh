#!/bin/bash
set -e

echo "Installing packaging dependencies..."
uv pip install pyinstaller pywebview fastapi uvicorn python-multipart

# Oldest macOS the app supports. The package installer picks wheels for the
# machine it runs on, so a build on macOS 26 bundled MLX's macOS-26.2 build.
# On an older Mac that library calls Metal functions that do not exist, and
# the app dies with no error the moment MLX runs — a collaborator on an older
# macOS saw exactly that. PyPI publishes MLX for macOS 14, 15 and 26; pin 14.
MIN_MACOS="14.0"
MLX_VERSION=$(uv run python -c "import importlib.metadata as m; print(m.version('mlx'))")
echo "Pinning MLX $MLX_VERSION to its macOS $MIN_MACOS build..."
MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS uv pip install --reinstall --no-deps \
    --python-platform aarch64-apple-darwin \
    "mlx==$MLX_VERSION" "mlx-metal==$MLX_VERSION"

echo "Cleaning up previous builds..."
rm -rf build dist ClinicalWhisper.dmg ClinicalWhisper.spec

echo "Staging model weights for the bundle..."
# Ship the models inside the app so a fresh machine needs nothing but the DMG.
# The HF cache stores snapshots/ as symlinks into blobs/; dereferencing both
# would double 6.6 GB, so copy the dereferenced snapshots and drop the blobs.
# Verified: the loaders read this layout read-only with HF_HUB_OFFLINE=1.
STAGE=".model_stage"   # outside build/ so cleanup does not force a 7 GB re-copy
mkdir -p "$STAGE/hub"

HF_HUB="$HOME/.cache/huggingface/hub"
# The voice model is small (26 MB) but not optional: without it speakers are
# matched across 5-minute windows by a much weaker fallback, and one person
# comes out as several. The offline lock (correctly) refuses to download it.
MODELS="models--OpenMOSS-Team--MOSS-Transcribe-Diarize \
        models--mlx-community--Meta-Llama-3-8B-Instruct-4bit \
        models--OpenMed--OpenMed-PII-SuperClinical-Small-44M-v1 \
        models--Wespeaker--wespeaker-voxceleb-resnet34-LM"

for m in $MODELS; do
    if [ ! -d "$HF_HUB/$m" ]; then
        echo "ERROR: $m is not in the local cache."
        echo "Process one audio file first so the weights download, then rebuild."
        exit 1
    fi
    if [ -d "$STAGE/hub/$m/snapshots" ]; then
        echo "  reusing staged $m"
        continue
    fi
    echo "  staging $m ($(du -sh "$HF_HUB/$m" | cut -f1))"
    mkdir -p "$STAGE/hub/$m"
    cp -R "$HF_HUB/$m/refs" "$STAGE/hub/$m/" 2>/dev/null || true
    rsync -aL "$HF_HUB/$m/snapshots" "$STAGE/hub/$m/"
done

# OpenMED keeps an MLX-converted copy; its HF-form duplicate is already in hub/.
if [ -d "$HOME/.cache/openmed" ]; then
    echo "  staging openmed MLX cache"
    rsync -aL --exclude 'models--*' "$HOME/.cache/openmed/" "$STAGE/openmed/"
fi

echo "  staged total: $(du -sh "$STAGE" | cut -f1)"

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
    --add-data ".model_stage:models" \
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
if [ ! -d dist/ClinicalWhisper.app/Contents/Resources/models/hub ]; then
    MISSING="$MISSING bundled-models"
fi
for m in $MODELS; do
    if [ ! -d "dist/ClinicalWhisper.app/Contents/Resources/models/hub/$m" ]; then
        MISSING="$MISSING $m"
    fi
done

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

echo "Checking every bundled binary runs on macOS $MIN_MACOS..."
# minos is the oldest macOS a Mach-O binary was built for. Anything newer than
# MIN_MACOS may crash natively on a supported Mac, so refuse to ship it.
TOO_NEW=""
while IFS= read -r -d '' bin; do
    minos=$(otool -l "$bin" 2>/dev/null | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')
    if [ -n "$minos" ] && [ "$(printf '%s\n%s\n' "$MIN_MACOS" "$minos" | sort -V | tail -1)" != "$MIN_MACOS" ]; then
        TOO_NEW="$TOO_NEW
  macOS $minos  ${bin#dist/ClinicalWhisper.app/Contents/}"
    fi
done < <(find dist/ClinicalWhisper.app/Contents -type f \( -name '*.dylib' -o -name '*.so' -o -path '*/MacOS/*' \) -print0)
if [ -n "$TOO_NEW" ]; then
    echo "ERROR: these binaries require a newer macOS than $MIN_MACOS:$TOO_NEW"
    exit 1
fi
echo "All binaries run on macOS $MIN_MACOS or later."

# Say so up front: an older Mac gets "requires macOS 14" instead of a crash.
PLIST=dist/ClinicalWhisper.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Delete :LSMinimumSystemVersion" "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Add :LSMinimumSystemVersion string $MIN_MACOS" "$PLIST"
# Editing Info.plist invalidates the bundle signature; an invalid signature on
# Apple Silicon reads as "damaged". Re-apply the ad-hoc signature.
codesign --force --deep --sign - dist/ClinicalWhisper.app
codesign --verify --deep --strict dist/ClinicalWhisper.app

echo "Creating DMG..."
# Check if hdiutil is available (macOS)
if command -v hdiutil &> /dev/null; then
    rm -rf dist/dmg_folder
    mkdir -p dist/dmg_folder
    # ditto, not cp -r: BSD cp follows symlinks, which dereferenced the 126
    # Frameworks -> Resources links and duplicated the whole 7.2 GB model set
    # (an 18 GB folder from an 8.3 GB app). It also preserves the ad-hoc code
    # signature PyInstaller applies.
    ditto dist/ClinicalWhisper.app dist/dmg_folder/ClinicalWhisper.app
    ln -s /Applications dist/dmg_folder/Applications

    echo "Running hdiutil to package .app into .dmg..."
    hdiutil create -volname "ClinicalWhisper" \
        -srcfolder dist/dmg_folder \
        -ov -format UDZO \
        ClinicalWhisper.dmg

    echo "Successfully created ClinicalWhisper.dmg in the current directory!"
    echo
    echo "This DMG is self-contained: model weights are inside the app, and"
    echo "audio decoding uses PyAV, so the target machine needs no Hugging Face"
    echo "download, no account, and no ffmpeg install."
else
    echo "hdiutil not found. DMG creation skipped (are you on macOS?). The .app is available in the dist/ folder."
fi
