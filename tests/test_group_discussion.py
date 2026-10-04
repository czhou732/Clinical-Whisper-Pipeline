"""Group discussion mode: speaker codes, coding files, label repair, crosstalk."""

import io
import zipfile
from xml.dom import minidom

import numpy as np

import group_exports
import overlap_detector
import speaker_check
import speaker_roles

SEGS = [
    {"speaker": "S01", "start": 0.0, "end": 4.0, "text": "Welcome, everyone. What brought you here?"},
    {"speaker": "S02", "start": 4.5, "end": 9.0, "text": "My sister [first_name_1] told me about it."},
    {"speaker": "S02", "start": 9.0, "end": 11.0, "text": "And I was curious."},
    {"speaker": "S03", "start": 11.5, "end": 15.0, "text": "Same <here> & there.", "crosstalk": True},
    {"speaker": "S01", "start": 3725.2, "end": 3727.0, "text": "Thank you."},
]
ROLES = {"S01": "Moderator 1", "S02": "Participant 2", "S03": "Participant 1"}


def test_codes_follow_first_speech_and_names_override():
    codes = group_exports.codes_for(ROLES, SEGS)
    assert codes == {"S01": "Moderator", "S02": "P01", "S03": "P02"}
    assert group_exports.codes_for(ROLES, SEGS, {"S03": "P17"})["S03"] == "P17"


def test_turns_merge_and_f4_timestamps():
    codes = group_exports.codes_for(ROLES, SEGS)
    tl = group_exports.turns(SEGS, codes)
    assert [t["code"] for t in tl] == ["Moderator", "P01", "P02", "Moderator"]
    txt = group_exports.to_txt(tl, {"session": "FG1"})
    assert "P01: My sister [first_name_1] told me about it. And I was curious. #00:00:11-0#" in txt
    assert "P02: Same <here> & there. [crosstalk] #00:00:15-0#" in txt
    assert "#01:02:07-0#" in txt


def test_docx_is_valid_word_xml():
    codes = group_exports.codes_for(ROLES, SEGS)
    data = group_exports.to_docx(group_exports.turns(SEGS, codes), {"session": "FG1"})
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert set(z.namelist()) == {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
        doc = minidom.parseString(z.read("word/document.xml"))
    text = "".join(t.firstChild.data for t in doc.getElementsByTagName("w:t") if t.firstChild)
    assert "Same <here> & there." in text and "P02:" in text


def test_vtt_cues_escape_and_tag_voices():
    vtt = group_exports.to_vtt(SEGS, group_exports.codes_for(ROLES, SEGS))
    assert vtt.startswith("WEBVTT")
    assert "00:00:11.500 --> 00:00:15.000\n<v P02>Same &lt;here&gt; &amp; there. [crosstalk]" in vtt


def test_participation_and_deid_record_hold_no_text():
    rows = group_exports.participation(SEGS, group_exports.codes_for(ROLES, SEGS))
    assert rows[0]["code"] == "Moderator" and rows[0]["turns"] == 2
    rec = group_exports.deid_record({"session": "FG1"}, {"masked_by_type": {"first_name": 1}},
                                    [{"kind": "merged", "seconds": 40.0}], 0.12)
    assert "first_name: 1 mention" in rec and "12%" in rec and "sister" not in rec


def _voice(rng, base, noise=0.1):
    # Same-person clips then match at about 0.6, as WeSpeaker segments do on AMI.
    v = base + noise * rng.standard_normal(base.shape)
    return v / np.linalg.norm(v)


def _people(rng, n=3, dim=64):
    return [p / np.linalg.norm(p) for p in rng.standard_normal((n, dim))]


def test_refine_merges_one_voice_under_two_labels():
    rng = np.random.default_rng(0)
    a, b = _people(rng, 2)
    segs = [{"speaker": lab, "start": i * 3.0, "end": i * 3.0 + 2.5}
            for i, lab in enumerate(["S01"] * 10 + ["S02"] * 10 + ["S03"] * 10)]
    emb = {i: _voice(rng, a if segs[i]["speaker"] in ("S01", "S03") else b) for i in range(30)}
    out, changes = speaker_check.refine(segs, emb)
    assert {s["speaker"] for s in out[:10]} == {s["speaker"] for s in out[20:]}
    assert any(c["kind"] == "merged" for c in changes)


def test_refine_gives_a_late_joiner_hidden_under_a_label_their_own_code():
    rng = np.random.default_rng(1)
    a, b, joiner = _people(rng, 3)
    segs, emb = [], {}
    for i in range(40):
        lab = "S01" if i % 2 == 0 else "S02"
        voice = a if lab == "S01" else (joiner if i > 20 else b)
        segs.append({"speaker": lab, "start": i * 10.0, "end": i * 10.0 + 8.0})
        emb[i] = _voice(rng, voice)
    out, changes = speaker_check.refine(segs, emb)
    late = {out[i]["speaker"] for i in range(21, 40, 2)}
    early = {out[i]["speaker"] for i in range(1, 20, 2)}
    assert len(late) == 1 and late != early


def test_refine_leaves_clean_labels_alone():
    rng = np.random.default_rng(2)
    people = _people(rng, 3)
    segs = [{"speaker": f"S0{i % 3 + 1}", "start": i * 3.0, "end": i * 3.0 + 2.5} for i in range(30)]
    emb = {i: _voice(rng, people[i % 3]) for i in range(30)}
    out, changes = speaker_check.refine(segs, emb)
    assert changes == [] and [s["speaker"] for s in out] == [s["speaker"] for s in segs]


def test_crosstalk_marks_segments_with_enough_overlap():
    segs = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 5.0}, {"start": 6.0, "end": 9.0}]
    assert overlap_detector.mark_segments(segs, [(3.5, 4.6)]) == 2
    assert segs[0].get("crosstalk") and segs[1].get("crosstalk") and not segs[2].get("crosstalk")


def test_group_choice_forces_group_roles_with_two_main_speakers():
    segs = [{"speaker": "S01", "start": 0, "end": 5, "text": "What do you think about the app? Why?"},
            {"speaker": "S02", "start": 5, "end": 100, "text": "I think it is fine. " * 40}]
    assert speaker_roles.assign(segs)["mode"] == "interview"
    assert speaker_roles.assign(segs, group=True)["mode"] == "group"


def test_risk_flags_off_by_default_in_groups():
    from inference_pipeline import _clinical_review
    segs = [{"speaker": "S02", "start": 0, "end": 3, "text": "I wanted to kill myself last year."}]
    roles = {"S01": "Moderator 1", "S02": "Participant 1"}
    off = _clinical_review(segs, roles, {"recording_type": "group"})
    assert off["items"] == [] and "off for group" in off["note"]
    on = _clinical_review(segs, roles, {"recording_type": "group", "review_flags": {"group_enabled": True}})
    assert on["items"]


def test_header_lines_are_not_read_as_speakers_by_maxqda():
    # MAXQDA's focus-group import: a colon in the first 63 characters marks a speaker.
    for line in group_exports.header_lines({"session": "FG 3: pilot", "processed": "2026-10-04", "version": "5.3.0"}):
        assert ":" not in line[:63], line
