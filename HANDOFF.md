# ClinicalWhisper — Handoff Guide

For giving ClinicalWhisper to a collaborator on a machine that has never run it.

---

## Short answer: no, the DMG alone is not enough

The `.app` is 1.3 GB, but the model weights are **not** inside it. On a machine
with an empty cache, the DMG on its own will download ~6.4 GB the first time
someone processes a file, and will fail outright if `ffmpeg` is not installed.

Three things have to be on the target machine:

| | What | Size | Why |
|---|---|---|---|
| 1 | `ClinicalWhisper.app` | 1.3 GB | the application |
| 2 | Model weights in `~/.cache` | ~7.7 GB | MOSS, Llama-3, OpenMED |
| 3 | `ffmpeg` on PATH | ~80 MB | decodes and resamples every input file |

`prepare_offline_bundle.sh` packages all three so nothing downloads on the day.

---

## Hardware requirement — check this first

**Apple Silicon Mac (M1 or newer).** Have them run this in Terminal:

```bash
uname -m
```

- `arm64` → good, proceed.
- `x86_64` → Intel Mac. The app runs, but clinical scoring falls back to CPU and
  takes roughly 10–20x longer. Not viable for a live demo.
- Windows or Linux → the DMG does not apply at all; they need the command-line
  install from the README instead.

The bundled OpenSMILE and MLX binaries are `arm64`-only. This is not a soft
preference — check before the meeting, not during it.

---

## Preparing the drive (do this beforehand, on your machine)

Use a **16 GB or larger** USB drive formatted **APFS or exFAT**. FAT32 cannot
hold the individual model files.

```bash
cd ~/Research/Labs/VisionNeuro_Itti/GetBraille/ClinicalWhisper
./build_dmg.sh
./scripts/prepare_offline_bundle.sh /Volumes/YOUR_DRIVE/ClinicalWhisper
```

The script copies the DMG, all three model caches, and `ffmpeg`, then writes
`install.sh` and `READ_ME_FIRST.txt` into the folder. It refuses to run if any
model is missing from your local cache — so process at least one file on your
machine first.

**One caveat it will warn you about:** if your `ffmpeg` came from Homebrew it
links against `/opt/homebrew` libraries and will not run on a machine without
Homebrew. If you see that warning, download a static build from
<https://evermeet.cx/ffmpeg/> and replace `bin/ffmpeg` in the bundle.

Verify before you hand it over:

```bash
du -sh /Volumes/YOUR_DRIVE/ClinicalWhisper     # expect ~9 GB
ls /Volumes/YOUR_DRIVE/ClinicalWhisper          # ClinicalWhisper.dmg, models/, bin/, install.sh
```

---

## Instructions for the recipient

> ### Setting up ClinicalWhisper
>
> **1. Install the app**
> Open `ClinicalWhisper.dmg` and drag **ClinicalWhisper** into your Applications
> folder. Eject the disk image when it finishes copying.
>
> **2. Run the setup script**
> Open Terminal (⌘-Space, type "Terminal"). Type `bash ` — with a space — then
> drag `install.sh` from the folder into the Terminal window and press Return.
>
> It installs the models and clears the macOS security warning on the app. It
> will ask for your Mac password once, to place `ffmpeg`.
>
> **3. Open the app**
> Launch ClinicalWhisper from Applications.
>
> If macOS says the developer cannot be verified: open System Settings →
> Privacy & Security, scroll to Security, and click **Open Anyway**. This is
> expected — the app is not notarized by Apple.
>
> ### Using it
>
> Drag one or more audio files onto the window and click **Process**. Supported:
> `.wav`, `.m4a`, `.mp3`, `.mp4`.
>
> While it runs you will see each stage: transcription, PII scrubbing, acoustic
> extraction, clinical scoring.
>
> When it finishes there are three views:
> - **Scores** — the clinical scores, acoustic measures, and clinical impression
> - **Transcript** — the de-identified transcript with speaker labels
> - **JSON** — the complete analysis record
>
> Use **CSV**, **JSON**, or **Report** at the top right to save wherever you like.
> Nothing is uploaded; everything stays on this Mac. A copy is also written to
> `~/Documents/ClinicalWhisper/Output/`.
>
> ### What to expect
>
> The first file after opening the app takes about 30 seconds longer while the
> models load. After that, roughly 20–40 seconds per short clip. A full-length
> interview takes several minutes — the transcription stage scales with how much
> speech is in the recording.
>
> Processing several files at once is faster than one at a time: the models load
> once and stay loaded for the whole batch.

---

## Known limits worth stating up front

- **Very quiet recordings.** Transcription runs on a loudness-normalized copy,
  which handles the faint pilot recordings. Genuinely silent audio produces a
  clear error rather than an empty result.
- **Speaker labels are anonymous.** MOSS emits `S01`, `S02`, …; the app then
  guesses Interviewer vs Subject from turn length and question ratio. On a
  single-speaker recording it labels everything `Subject`.
- **The clinical scores are model output, not a diagnosis.** They are a research
  instrument and should be reported as such.
- **The app is not notarized.** Every recipient will hit the Gatekeeper warning
  once. `install.sh` clears it; otherwise it is the System Settings step above.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "ffmpeg was not found" | ffmpeg missing or Homebrew-linked | `brew install ffmpeg`, or replace `bin/ffmpeg` with a static build |
| "MOSS returned an empty transcript" | audio is silent or not speech | check the recording plays |
| Stuck on the first file for minutes | first launch — macOS is scanning the 1.3 GB bundle | wait; subsequent launches are fast |
| Acoustic fields blank, warning banner shown | OpenSMILE failed | the banner names the reason; scores from other stages are still valid |
| A file in a batch fails | that file only | the batch continues; the failed tab shows the error |

Results and logs live in `~/Documents/ClinicalWhisper/`. Send that folder's
`Output/` contents when reporting a problem.
