# ClinicalWhisper — Handoff Guide

Giving ClinicalWhisper to a collaborator over a call, with no setup on their end.

---

## What you send

**One file: `ClinicalWhisper.dmg` (~8 GB).**

The app carries its own model weights and its own audio decoder, so the machine
receiving it needs no Hugging Face account, no downloads, no Homebrew, and no
Terminal. It also works with the network off.

Too large to send through Zoom chat — upload it to Google Drive, Dropbox, or
WeTransfer and share the link. Start that upload well before the call; 8 GB
takes a while on most connections.

### The one thing to check first

**Apple Silicon Mac (M1 or newer.)** Ask them to run this in Terminal, or check
 > About This Mac:

```bash
uname -m
```

- `arm64` → good.
- `x86_64` → Intel Mac. The bundled binaries are arm64-only; the app will not
  run. They need the command-line install from the README instead.
- Windows or Linux → the DMG does not apply.

This is worth confirming before you spend an hour uploading 8 GB.

---

## Instructions for the recipient

> ### Installing ClinicalWhisper
>
> **1.** Download `ClinicalWhisper.dmg` and double-click it.
>
> **2.** Drag **ClinicalWhisper** into your Applications folder, then eject the
> disk image.
>
> **3.** The first time only: go to your Applications folder, **right-click**
> ClinicalWhisper and choose **Open**, then confirm.
>
> Double-clicking will not work the first time — macOS blocks apps that are not
> notarized by Apple. If you get "cannot be opened because the developer cannot
> be verified", open System Settings → Privacy & Security, scroll to Security,
> and click **Open Anyway**.
>
> That is the whole setup. Nothing to install, nothing to download.
>
> ### Using it
>
> Drag one or more audio files onto the window and click **Process**.
> Supported: `.wav`, `.m4a`, `.mp3`, `.mp4`.
>
> Optionally fill in a Participant ID and Session before processing — they are
> carried into the results so a CSV can be grouped by participant.
>
> Each stage is shown as it runs — transcription, PII scrubbing, acoustic
> extraction, clinical scoring — with a progress bar during transcription, which
> is the long one. **Stop** cancels a run in progress.
>
> When it finishes there are three tabs:
> - **Scores** — clinical scores, acoustic measures, and the clinical impression
> - **Transcript** — de-identified, with speaker labels
> - **JSON** — the complete analysis record
>
> If the Interviewer/Subject labels came out backwards, **Swap roles & re-score**
> fixes them in a few seconds without re-transcribing.
>
> Save with **CSV**, **JSON**, or **Report** at the top right. A copy is also
> written to `~/Documents/ClinicalWhisper/Output/`.
>
> Nothing is uploaded. Everything runs on this Mac, including with Wi-Fi off.
>
> ### What to expect
>
> The first file after opening the app takes about 15 seconds longer while the
> models load. After that, roughly a minute per short clip. Longer interviews
> take proportionally longer — transcription scales with how much speech is in
> the recording.
>
> Processing several files at once is faster than one at a time: the models load
> once and stay loaded for the whole batch.
>
> Memory is released when a batch finishes, so leaving the app open costs a few
> hundred MB rather than several GB.

---

## Known limits worth stating up front

- **The clinical scores are model output, not a diagnosis.** They are a research
  instrument and should be reported as such.
- **Speaker labels are anonymous.** The model emits `S01`, `S02`, …; the app then
  guesses Interviewer vs Subject from turn length and question ratio. On a
  single-speaker recording it labels everything `Subject`.
- **Very quiet recordings** are handled — transcription runs on a level-normalized
  copy, while acoustic features use the original gain. Genuinely silent audio
  produces a clear error rather than an empty result.
- **The app is not notarized**, hence the one-time right-click → Open.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| App will not open at all | Intel Mac, or Gatekeeper | check `uname -m`; right-click → Open |
| "MOSS returned an empty transcript" | audio is silent or not speech | check the recording plays |
| A file in a batch fails | that file only | the batch continues; the failed tab shows the error |
| Acoustic fields blank, warning banner | OpenSMILE failed | the banner names the reason; other scores are still valid |

Results and logs are in `~/Documents/ClinicalWhisper/`. Send the `Output/`
contents when reporting a problem.

---

## Rebuilding the DMG

```bash
./build_dmg.sh
```

The script stages the model weights out of your local Hugging Face cache into
the bundle, so process at least one file on your machine first — it refuses to
build if a model is missing. It also verifies every ML backend and the bundled
weights landed in the `.app`, and aborts rather than producing a DMG that
launches but cannot analyze anything.
