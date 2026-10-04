"""Coding-ready files for a group discussion: transcript, subtitles, participation, de-id record.

Written next to the analysis JSON when the recording type is "Group
discussion". Every file holds the de-identified text only, with speaker
codes (Moderator, P01, P02 ...) instead of names.

Layout follows what qualitative-analysis software imports:

* ``_transcript.docx`` / ``.txt``: one paragraph per turn, "P03: text", a
  timestamp at the end of each paragraph in the f4/f5 style (#00:01:23-4#).
  MAXQDA's focus-group import codes each paragraph by the name before the
  colon and reads these timestamps; ATLAS.ti reads the same timestamp style.
* ``_subtitles.vtt``: one cue per segment with a voice tag (<v P03>), for
  importing in sync with the audio (ATLAS.ti, MAXQDA).
* ``_participation.csv``: talk time, turns, words and crosstalk per code.
* ``_deid_record.txt``: how the transcript was de-identified, for the IRB
  file. Counts only, never the identifiers.

The DOCX is written with the standard library (no python-docx).
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

CROSSTALK = "[crosstalk]"


def codes_for(roles: dict[str, str], segments: list[dict],
              names: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Speaker label -> code. Moderators keep their role; participants are
    P01, P02 ... in order of first speech; brief speakers are "Other N".
    A code the user typed for a speaker (names) is used as given."""
    first: dict[str, float] = {}
    for s in segments:
        first.setdefault(s["speaker"], s["start"])
    out: dict[str, str] = {}
    mods = sorted((spk for spk, r in roles.items() if r.startswith("Moderator")), key=lambda x: first.get(x, 0))
    for i, spk in enumerate(mods, start=1):
        out[spk] = "Moderator" if i == 1 else f"Moderator {i}"
    parts = sorted((spk for spk, r in roles.items() if r.startswith(("Participant", "Subject", "Interviewer"))
                    and spk not in out), key=lambda x: first.get(x, 0))
    for i, spk in enumerate(parts, start=1):
        out[spk] = f"P{i:02d}"
    others = sorted((spk for spk in first if spk not in out), key=lambda x: first.get(x, 0))
    for i, spk in enumerate(others, start=1):
        out[spk] = f"Other {i}"
    for spk, name in (names or {}).items():
        if name and name.strip():
            out[spk] = name.strip()
    return out


def turns(segments: list[dict], codes: dict[str, str]) -> list[dict]:
    """Consecutive segments of one speaker merged into turns."""
    out: list[dict] = []
    for s in sorted(segments, key=lambda x: x["start"]):
        text = (s.get("text") or "").strip()
        if not text:
            continue
        code = codes.get(s["speaker"], s["speaker"])
        if out and out[-1]["code"] == code:
            out[-1]["text"] += " " + text
            out[-1]["end"] = s["end"]
            out[-1]["crosstalk"] = out[-1]["crosstalk"] or bool(s.get("crosstalk"))
        else:
            out.append({"code": code, "start": s["start"], "end": s["end"], "text": text,
                        "crosstalk": bool(s.get("crosstalk"))})
    return out


def _f4(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    tenth = int((seconds - int(seconds)) * 10)
    return f"#{h:02d}:{m:02d}:{s:02d}-{tenth}#"


def _vtt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _turn_line(t: dict) -> str:
    mark = f" {CROSSTALK}" if t["crosstalk"] else ""
    return f"{t['code']}: {t['text']}{mark} {_f4(t['end'])}"


def header_lines(info: dict) -> list[str]:
    lines = [f"Group discussion: {info.get('session') or info.get('file') or 'session'}",
             f"Processed {info.get('processed', '')} with ClinicalWhisper {info.get('version', '')}.",
             "De-identified: names and other identifiers are replaced by tags such as [first_name_1];",
             "the same tag means the same identifier throughout. Speakers are codes, not names.",
             f"{CROSSTALK} marks turns where two people spoke at once; the words of the second "
             "speaker may be missing there."]
    return lines


def to_txt(turn_list: list[dict], info: dict) -> str:
    return "\n".join(header_lines(info) + [""] + [_turn_line(t) for t in turn_list]) + "\n"


def to_vtt(segments: list[dict], codes: dict[str, str]) -> str:
    out = ["WEBVTT", ""]
    n = 0
    for s in sorted(segments, key=lambda x: x["start"]):
        text = (s.get("text") or "").strip()
        if not text or s["end"] <= s["start"]:
            continue
        n += 1
        code = codes.get(s["speaker"], s["speaker"])
        mark = f" {CROSSTALK}" if s.get("crosstalk") else ""
        # Characters with meaning in VTT cue text are escaped.
        body = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out += [str(n), f"{_vtt_time(s['start'])} --> {_vtt_time(s['end'])}", f"<v {code}>{body}{mark}", ""]
    return "\n".join(out)


def _para(text: str, bold_prefix: str = "") -> str:
    runs = ""
    if bold_prefix:
        runs += f'<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">{escape(bold_prefix)}</w:t></w:r>'
    runs += f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    return f"<w:p>{runs}</w:p>"


def to_docx(turn_list: list[dict], info: dict) -> bytes:
    """A minimal Word document: header lines, then one paragraph per turn."""
    body = "".join(_para(line) for line in header_lines(info)) + _para("")
    for t in turn_list:
        line = _turn_line(t)
        prefix = f"{t['code']}:"
        body += _para(line[len(prefix):], bold_prefix=prefix)
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f'<w:body>{body}<w:sectPr/></w:body></w:document>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" '
                     'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def participation(segments: list[dict], codes: dict[str, str]) -> list[dict]:
    rows: dict[str, dict] = {}
    total = sum(max(0.0, s["end"] - s["start"]) for s in segments) or 1.0
    turn_list = turns(segments, codes)
    for s in segments:
        code = codes.get(s["speaker"], s["speaker"])
        r = rows.setdefault(code, {"code": code, "talk_s": 0.0, "words": 0, "turns": 0,
                                   "crosstalk_segments": 0, "first_speaks_s": s["start"]})
        r["talk_s"] += max(0.0, s["end"] - s["start"])
        r["words"] += len(re.findall(r"\S+", s.get("text") or ""))
        r["crosstalk_segments"] += bool(s.get("crosstalk"))
        r["first_speaks_s"] = min(r["first_speaks_s"], s["start"])
    for t in turn_list:
        rows[t["code"]]["turns"] += 1
    out = []
    for r in sorted(rows.values(), key=lambda r: r["first_speaks_s"]):
        out.append({"code": r["code"], "talk_min": round(r["talk_s"] / 60, 2),
                    "share_of_talk": round(r["talk_s"] / total, 3), "turns": r["turns"],
                    "words": r["words"], "crosstalk_segments": r["crosstalk_segments"],
                    "first_speaks": _vtt_time(r["first_speaks_s"])[:8]})
    return out


def deid_record(info: dict, deid: dict, speaker_changes: list[dict], crosstalk_share: Optional[float]) -> str:
    """One page for the IRB file: what was done and how well it is known to work."""
    lines = [f"De-identification record: {info.get('session') or info.get('file') or 'session'}",
             f"Processed {info.get('processed', '')}, ClinicalWhisper {info.get('version', '')}, "
             "entirely on this computer (no audio or text was sent anywhere).", "",
             "Method",
             "- Text: OpenMed PII model, then ClinicalWhisper's safety-net rules (relationship words,",
             "  titles, workplaces, streets, phone numbers, emails), then every caught name masked",
             "  wherever it recurs in the session.",
             f"- Language: {info.get('language', 'English')}.",
             f"- Audio: {'masked names silenced in a separate copy' if info.get('audio_silenced') else 'not altered'}.",
             "", "Counts (no identifiers are listed)"]
    for label, n in sorted((deid.get("masked_by_type") or {}).items()):
        lines.append(f"- {label}: {n} mention(s)")
    lines += [f"- distinct identifiers: {deid.get('distinct_identifiers', 0)}",
              f"- segments replaced whole with [REDACTED]: {deid.get('segments_redacted_whole', 0)}", "",
              "Measured performance (synthetic test sentences, not this session)",
              "- English: 98-100% of identifiers caught on sentences written after the rules",
              "  (evals/masking/README.md); Microsoft Presidio caught 68-81% of the same.",
              "- Other languages: 74-100% depending on language; read the transcript before sharing.",
              "- Real speech is harder than test sentences. A person should read the transcript",
              "  before it leaves the study team.", "", "Speakers"]
    merged = [c for c in speaker_changes if c.get("kind") == "merged"]
    moved = sum(c["seconds"] for c in speaker_changes if c.get("kind") == "moved")
    split = [c for c in speaker_changes if c.get("kind") not in ("merged", "moved")]
    if merged:
        lines.append(f"- {len(merged)} speaker label(s) had the same voice as another and were joined.")
    if moved:
        lines.append(f"- {moved:.0f} s of speech matched another speaker's voice better and was moved there.")
    for c in split:
        lines.append(f"- A second voice was found under one label from {c['first_s']:.0f} s on "
                     f"({c['seconds']:.0f} s) and given its own code: check the codes around that time.")
    if not speaker_changes:
        lines.append("- Every speaker label held one consistent voice.")
    if crosstalk_share is not None:
        lines.append(f"- Two people spoke at once in {crosstalk_share:.0%} of the speech; the transcript "
                     f"marks those turns {CROSSTALK}.")
    return "\n".join(lines) + "\n"


def write_all(out_dir: Path, base: str, segments: list[dict], roles: dict[str, str], info: dict,
              deid: dict, speaker_changes: list[dict], crosstalk_share: Optional[float],
              names: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Write the four files; returns {kind: path}. Files are private to the user (0600)."""
    import os

    codes = codes_for(roles, segments, names)
    tl = turns(segments, codes)
    paths = {"docx": out_dir / f"{base}_transcript.docx", "txt": out_dir / f"{base}_transcript.txt",
             "vtt": out_dir / f"{base}_subtitles.vtt", "participation": out_dir / f"{base}_participation.csv",
             "deid_record": out_dir / f"{base}_deid_record.txt"}
    paths["docx"].write_bytes(to_docx(tl, info))
    paths["txt"].write_text(to_txt(tl, info), encoding="utf-8")
    paths["vtt"].write_text(to_vtt(segments, codes), encoding="utf-8")
    rows = participation(segments, codes)
    with paths["participation"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["code"])
        w.writeheader()
        w.writerows(rows)
    paths["deid_record"].write_text(deid_record(info, deid, speaker_changes, crosstalk_share), encoding="utf-8")
    for p in paths.values():
        os.chmod(p, 0o600)
    return {k: str(v) for k, v in paths.items()}
