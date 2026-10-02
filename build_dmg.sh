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
STAGE=".model_stage"   # models staged for PyInstaller; deleted again before the DMG step
mkdir -p "$STAGE/hub"

HF_HUB="$HOME/.cache/huggingface/hub"
# The voice model is small (26 MB) but not optional: without it speakers are
# matched across 5-minute windows by a much weaker fallback, and one person
# comes out as several. The offline lock (correctly) refuses to download it.
# The clinical scoring model (5.3 GB) is not in the base app: it ships as a
# separate add-on, built by build_scoring_addon.sh (see addons.py).
MODELS="models--OpenMOSS-Team--MOSS-Transcribe-Diarize \
        models--OpenMed--OpenMed-PII-SuperClinical-Small-44M-v1 \
        models--Wespeaker--wespeaker-voxceleb-resnet34-LM"

# A stage left by an older build may hold models or revisions this build
# does not ship (the scoring model, old MOSS revisions): keep only what is
# listed below, at its current revision.
for staged in "$STAGE"/hub/models--*; do
    [ -d "$staged" ] || continue
    case " $MODELS " in
        *" $(basename "$staged") "*) ;;
        *) echo "  removing unlisted staged $(basename "$staged")"; rm -rf "$staged" ;;
    esac
done

for m in $MODELS; do
    if [ ! -d "$HF_HUB/$m" ]; then
        echo "ERROR: $m is not in the local cache."
        echo "Process one audio file first so the weights download, then rebuild."
        exit 1
    fi
    # Only the revision the loaders use. The cache keeps every revision ever
    # downloaded: copying all of snapshots/ shipped MOSS three times (+3.6 GB).
    REV=$(cat "$HF_HUB/$m/refs/main" 2>/dev/null || true)
    if [ -z "$REV" ] || [ ! -d "$HF_HUB/$m/snapshots/$REV" ]; then
        echo "ERROR: $m has no usable refs/main snapshot."
        exit 1
    fi
    if [ "$(ls "$STAGE/hub/$m/snapshots" 2>/dev/null)" = "$REV" ]; then
        echo "  reusing staged $m"
        continue
    fi
    echo "  staging $m"
    rm -rf "$STAGE/hub/$m"
    mkdir -p "$STAGE/hub/$m/snapshots"
    cp -R "$HF_HUB/$m/refs" "$STAGE/hub/$m/"
    rsync -aL "$HF_HUB/$m/snapshots/$REV" "$STAGE/hub/$m/snapshots/"
done

# OpenMED keeps an MLX-converted copy; its HF-form duplicate is already in hub/.
# Only the English masker: other languages are add-ons (addons.py), and the
# cache also holds models tried during development (5.3's first build shipped
# Hindi, Spanish and Chinese ones, +2.1 GB).
OPENMED_MLX="OpenMed_OpenMed-PII-SuperClinical-Small-44M-v1"
if [ ! -f "$HOME/.cache/openmed/$OPENMED_MLX/weights.safetensors" ]; then
    echo "ERROR: $HOME/.cache/openmed/$OPENMED_MLX is missing. Process one recording first."
    exit 1
fi
mkdir -p "$STAGE/openmed"
for staged in "$STAGE"/openmed/*; do
    [ -e "$staged" ] || continue
    [ "$(basename "$staged")" = "$OPENMED_MLX" ] || { echo "  removing unlisted staged $(basename "$staged")"; rm -rf "$staged"; }
done
echo "  staging openmed MLX $OPENMED_MLX"
rsync -aL "$HOME/.cache/openmed/$OPENMED_MLX" "$STAGE/openmed/"

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
    --exclude-module parselmouth \
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
# The version Finder's Get Info shows, from version.py.
APP_VERSION=$(uv run python -c "from version import __version__; print(__version__)")
for key in CFBundleShortVersionString CFBundleVersion; do
    /usr/libexec/PlistBuddy -c "Delete :$key" "$PLIST" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :$key string $APP_VERSION" "$PLIST"
done
# Editing Info.plist invalidates the bundle signature; an invalid signature on
# Apple Silicon reads as "damaged". Re-apply the ad-hoc signature.
codesign --force --deep --sign - dist/ClinicalWhisper.app
codesign --verify --deep --strict dist/ClinicalWhisper.app

echo "Creating DMG..."
# Check if hdiutil is available (macOS)
if command -v hdiutil &> /dev/null; then
    rm -rf dist/dmg_folder
    mkdir -p dist/dmg_folder
    # Move rather than copy: a copy needs another 12 GB of disk, and cp -r
    # also dereferenced the Frameworks -> Resources links (an 18 GB folder).
    # The app is moved back afterwards, even if hdiutil fails.
    rm -rf dist/ClinicalWhisper  # PyInstaller's intermediate folder, unused
    # The models are inside the app now. Their 11 GB staging copy is rebuilt
    # from the Hugging Face cache next time; hdiutil needs the room.
    rm -rf "$STAGE"
    mv dist/ClinicalWhisper.app dist/dmg_folder/ClinicalWhisper.app
    trap 'mv dist/dmg_folder/ClinicalWhisper.app dist/ClinicalWhisper.app 2>/dev/null || true' EXIT
    ln -s /Applications dist/dmg_folder/Applications
    # One-page guide beside the app (source: docs/quickstart.html).
    if [ ! -f "docs/Quick Start.pdf" ]; then
        echo "MISSING docs/Quick Start.pdf: run docs/make_quickstart.sh" >&2
        exit 1
    fi
    cp "docs/Quick Start.pdf" "dist/dmg_folder/Quick Start.pdf"

    echo "Running hdiutil to package .app into .dmg..."
    hdiutil create -volname "ClinicalWhisper" \
        -srcfolder dist/dmg_folder \
        -ov -format UDZO \
        ClinicalWhisper.dmg
    mv dist/dmg_folder/ClinicalWhisper.app dist/ClinicalWhisper.app
    trap - EXIT

    echo "Successfully created ClinicalWhisper.dmg in the current directory!"
    echo
    echo "This DMG is self-contained: model weights are inside the app, and"
    echo "audio decoding uses PyAV, so the target machine needs no Hugging Face"
    echo "download, no account, and no ffmpeg install."
else
    echo "hdiutil not found. DMG creation skipped (are you on macOS?). The .app is available in the dist/ folder."
fi
