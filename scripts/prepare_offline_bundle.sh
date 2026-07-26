#!/bin/bash
#
# Build a self-contained ClinicalWhisper handoff bundle for a machine that has
# never run the app: the DMG, every model weight, and a static ffmpeg.
#
# Run this on a machine that has already processed at least one file, so the
# model caches are populated.
#
#   ./scripts/prepare_offline_bundle.sh /Volumes/USB_DRIVE/ClinicalWhisper
#
# The recipient then runs install.sh from inside that folder. No downloads, no
# Hugging Face account, no Homebrew.

set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "Usage: $0 <destination-folder>"
    echo "Example: $0 /Volumes/USB_DRIVE/ClinicalWhisper"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_HUB="$HOME/.cache/huggingface/hub"
OPENMED_CACHE="$HOME/.cache/openmed"

# Model repos the pipeline needs, in HF cache-directory form.
MODELS=(
    "models--OpenMOSS-Team--MOSS-Transcribe-Diarize"
    "models--mlx-community--Meta-Llama-3-8B-Instruct-4bit"
    "models--OpenMed--OpenMed-PII-SuperClinical-Small-44M-v1"
)

echo "==> Preparing offline bundle in $TARGET"
mkdir -p "$TARGET/models/hub" "$TARGET/bin"

# ── 1. The app ──────────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/ClinicalWhisper.dmg" ]; then
    echo "ERROR: ClinicalWhisper.dmg not found. Run ./build_dmg.sh first."
    exit 1
fi
echo "--> Copying ClinicalWhisper.dmg"
cp "$REPO_DIR/ClinicalWhisper.dmg" "$TARGET/"

# ── 2. Model weights ────────────────────────────────────────────────────────
MISSING=""
for m in "${MODELS[@]}"; do
    if [ -d "$HF_HUB/$m" ]; then
        echo "--> Copying $m ($(du -sh "$HF_HUB/$m" | cut -f1))"
        # -L dereferences the blob symlinks so the copy stands alone.
        cp -RL "$HF_HUB/$m" "$TARGET/models/hub/"
    else
        MISSING="$MISSING $m"
    fi
done

if [ -n "$MISSING" ]; then
    echo
    echo "ERROR: these models are not in the local cache:$MISSING"
    echo "Process one audio file on this machine first so they download, then re-run."
    exit 1
fi

if [ -d "$OPENMED_CACHE" ]; then
    echo "--> Copying OpenMED MLX cache ($(du -sh "$OPENMED_CACHE" | cut -f1))"
    cp -RL "$OPENMED_CACHE" "$TARGET/models/openmed"
fi

# ── 3. ffmpeg ───────────────────────────────────────────────────────────────
FFMPEG="$(command -v ffmpeg || true)"
if [ -n "$FFMPEG" ]; then
    echo "--> Copying ffmpeg from $FFMPEG"
    cp "$FFMPEG" "$TARGET/bin/ffmpeg"
    chmod +x "$TARGET/bin/ffmpeg"
    # Homebrew's ffmpeg links against dylibs under /opt/homebrew and will not
    # run on a machine without Homebrew. Say so rather than let it fail later.
    if otool -L "$TARGET/bin/ffmpeg" 2>/dev/null | grep -q "/opt/homebrew\|/usr/local/Cellar"; then
        echo "    WARNING: this ffmpeg links to Homebrew libraries and will NOT run"
        echo "             on a machine without Homebrew. Replace bin/ffmpeg with a"
        echo "             static build from https://evermeet.cx/ffmpeg/ before handoff."
    fi
else
    echo "    WARNING: no ffmpeg on this machine; download a static build from"
    echo "             https://evermeet.cx/ffmpeg/ into $TARGET/bin/ffmpeg"
fi

# ── 4. Installer for the recipient ──────────────────────────────────────────
cat > "$TARGET/install.sh" <<'INSTALLER'
#!/bin/bash
# ClinicalWhisper offline setup. Run once: ./install.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(uname -m)" != "arm64" ]; then
    echo "WARNING: this Mac is not Apple Silicon ($(uname -m))."
    echo "The app will run but clinical scoring falls back to CPU and will be very slow."
    echo
fi

echo "==> Installing model weights into ~/.cache/huggingface/hub"
mkdir -p "$HOME/.cache/huggingface/hub"
cp -R "$HERE/models/hub/"* "$HOME/.cache/huggingface/hub/"

if [ -d "$HERE/models/openmed" ]; then
    echo "==> Installing OpenMED cache into ~/.cache/openmed"
    mkdir -p "$HOME/.cache/openmed"
    cp -R "$HERE/models/openmed/"* "$HOME/.cache/openmed/"
fi

if [ -f "$HERE/bin/ffmpeg" ]; then
    echo "==> Installing ffmpeg into /usr/local/bin (may ask for your password)"
    sudo mkdir -p /usr/local/bin
    sudo cp "$HERE/bin/ffmpeg" /usr/local/bin/ffmpeg
    sudo chmod +x /usr/local/bin/ffmpeg
fi

# ClinicalWhisper is ad-hoc signed but not notarised, so Gatekeeper flags it as
# coming from an unidentified developer. Clearing the quarantine attribute after
# the app is in place avoids that dialog entirely.
if [ -d "/Applications/ClinicalWhisper.app" ]; then
    echo "==> Clearing the quarantine flag on /Applications/ClinicalWhisper.app"
    xattr -dr com.apple.quarantine /Applications/ClinicalWhisper.app 2>/dev/null || true
fi

echo
echo "Done."
echo
echo "If you have NOT yet copied the app: open ClinicalWhisper.dmg, drag"
echo "ClinicalWhisper into Applications, then run this script once more to clear"
echo "the Gatekeeper warning."
echo
echo "Results are written to ~/Documents/ClinicalWhisper/Output"
INSTALLER
chmod +x "$TARGET/install.sh"

# ── 5. Read-me for the recipient ────────────────────────────────────────────
cat > "$TARGET/READ_ME_FIRST.txt" <<'NOTE'
ClinicalWhisper — offline setup
===============================

1. Open ClinicalWhisper.dmg and drag ClinicalWhisper into Applications.

2. Open Terminal and run:      ./install.sh
   (drag this folder into the Terminal window to get its path)
   It installs the models and clears the Gatekeeper warning on the app.

3. Launch ClinicalWhisper from Applications.

If macOS still says the app "cannot be opened because the developer cannot be
verified", go to System Settings > Privacy & Security, scroll to Security, and
click "Open Anyway" next to ClinicalWhisper. This is expected: the app is not
notarised by Apple.

Everything runs on this machine. No audio, transcript, or result is ever
uploaded. Results appear in the app window and are written to:
    ~/Documents/ClinicalWhisper/Output

Requires an Apple Silicon Mac (M1 or newer). Check with:  uname -m
It should print: arm64
NOTE

echo
echo "==> Bundle ready: $TARGET  ($(du -sh "$TARGET" | cut -f1))"
echo "    Hand off the whole folder. The recipient runs install.sh, then mounts the DMG."
