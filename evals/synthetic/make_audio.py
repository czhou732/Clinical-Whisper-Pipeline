#!/usr/bin/env python3
"""Synthetic test recordings, spoken by macOS's built-in voices.

No real person is in them, so they can be shared with the team, attached to
bug reports and committed to test plans. Each scenario exercises a specific
part of the pipeline; README.md (written next to the audio) says which.

    python evals/synthetic/make_audio.py [OUT_DIR]

Default OUT_DIR: ~/ClinicalWhisper/Test audio/synthetic. Needs macOS (say,
afconvert). Voices: Samantha, Daniel, Karen, Moira, and the Spanish voices
"Eddy (Spanish (Spain))" and "Flo (Spanish (Mexico))".
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

GAP_SAMPLES = 9600  # 0.6 s of silence between turns, at 16 kHz

QUESTIONS = [
    ("Thanks for coming in today. Where did you grow up?", "neutral"),
    ("What do you enjoy doing on the weekends?", "positive"),
    ("How has your mood been over the past two weeks?", "negative"),
    ("What kind of work do you do?", "neutral"),
    ("Is there anything you look forward to lately?", "positive"),
    ("What has been the hardest part of this month?", "negative"),
    ("Where do you live now?", "neutral"),
    ("What makes you happy these days?", "positive"),
    ("Have you been feeling worried or stressed about anything?", "negative"),
    ("What did you do last weekend?", "neutral"),
    ("What was the best part of your week?", "positive"),
    ("Thank you. That's everything for today.", None),
]

ANHEDONIC = [
    "I grew up in a small town outside Houston, near the refinery. My parents still live there, in the same house, and I drive back to see them a few times a year when I can get the time off work.",
    "Not much, honestly.",
    "Pretty low. Most days I feel kind of heavy and down, like nothing is going to get better. I wake up early and just lie there thinking about everything that went wrong at work and with my brother, and by the afternoon I am exhausted even though I have not done anything at all.",
    "I work in logistics at a warehouse. I manage the evening shift schedules, which means I deal with call outs and overtime and making sure the trucks get loaded on time. It is a lot of spreadsheets and phone calls.",
    "No. Not really.",
    "Probably losing interest in the things I used to love. I used to play guitar every night and I used to go running with my friend on Sundays, and I just stopped both. I do not even miss them, which scares me a little, because they used to be the best part of my week.",
    "I live in an apartment on the east side with one roommate. It is close to work, about fifteen minutes, so the commute is easy, and there is a park nearby, although I have not been there in a while.",
    "I don't know. Maybe sleeping.",
    "Money, mostly. My hours got cut in the spring and I have been behind on rent twice. I keep thinking about what happens if I lose the job entirely, and it is hard to stop thinking about it once it starts, especially at night.",
    "I stayed home. I watched some television and cleaned the kitchen, then I went to the store to buy groceries for the week and came back and mostly stayed on the couch.",
    "Nothing stands out.",
    "Okay, thanks.",
]

ENGAGED = [
    "I grew up in a small town outside Houston. My parents still live there and I visit when I can.",
    "Oh, I love the weekends. I play guitar with two friends on Saturday mornings, we are learning some old blues songs right now, and on Sundays I go running along the bayou trail. Last week I finally ran ten miles without stopping, which I have been working toward all summer, and I was really proud of that.",
    "Pretty good, actually. A little tired some days with work, but overall I feel steady and pretty hopeful.",
    "I work in logistics at a warehouse, scheduling the evening shift.",
    "Yes, definitely. My sister is getting married in November and I am giving a toast, so I have been writing it, and we are planning a trip to the coast afterwards. I also signed up for a half marathon in the spring, so I am excited to train for that with my running group.",
    "Work got busy, and that was stressful for a couple of weeks.",
    "I live in an apartment on the east side with a roommate.",
    "Music, mostly, and being outside. Playing with my friends, cooking a big dinner on Sunday night and inviting people over. Even small things, like a good cup of coffee on the balcony in the morning before work, make me happy.",
    "A little bit about money, but it is manageable.",
    "I went to a friend's birthday party.",
    "Probably the run on Sunday, the weather was perfect, and afterwards we all got breakfast tacos and sat outside for two hours just talking and laughing.",
    "Thank you!",
]

FOCUS_GROUP = [
    ("Samantha", "Welcome everyone, and thank you for joining. Your participation is completely voluntary, and you may skip any question. With your permission, we will record this session."),
    ("Daniel", "Thanks. Let's start with your morning routine. What about you, Karen? Where do you run into trouble?"),
    ("Karen", "Well, usually I make coffee first, and then I try to find my keys, which is honestly the hardest part of my morning because they are never where I left them."),
    ("Samantha", "That makes sense. And you, Moira? How do you get around in the morning?"),
    ("Moira", "I mostly use my cane and an app on my phone that reads labels out loud. It works, but it is slow and people can hear it, which I do not love in public."),
    ("Daniel", "Okay, let's move on to the next question. What would you want from a wearable device?"),
    ("Karen", "Something quiet that just tells me what is in front of me without talking too much. I would also like it to read my mail, especially bills."),
    ("Moira", "For me privacy matters most. I would not want it recording people around me, and I would want it to look discreet."),
    ("Samantha", "Thank you both. Any other thoughts before we finish?"),
    ("Karen", "Just that a good color detector would really help me pick out clothes in the morning."),
    ("Moira", "Same here, and maybe vibration instead of speech when I am walking outside."),
]

SPANISH = [
    ("Eddy (Spanish (Spain))", "Buenos días. ¿Cómo se ha sentido estas últimas dos semanas? ¿Ha perdido interés en las cosas que antes disfrutaba?"),
    ("Flo (Spanish (Mexico))", "Pues la verdad, no mucho. Mi hermana María me llama todos los días, pero ya no tengo ganas de salir con ella ni con mis amigos de Guadalajara."),
    ("Eddy (Spanish (Spain))", "Entiendo. ¿Y cómo ha dormido usted?"),
    ("Flo (Spanish (Mexico))", "Muy mal. Me despierto a las tres de la mañana y no puedo volver a dormir."),
]

README = """Synthetic test recordings (macOS voices; no real people)
Regenerate with evals/synthetic/make_audio.py in the ClinicalWhisper repo.

interview_anhedonic.wav  2 speakers, ~2.3 min. Short answers to every positive
    question ("What do you enjoy...?" -> "Not much, honestly."), long answers to
    neutral and negative ones. Tests: roles (interviewer = Samantha), measured
    elaboration (positive/neutral words-per-answer ratio ~0.07), the clinical
    scorer's quoted evidence, and the 3-minute minimum for scores (1.6 min of
    participant speech: scores are refused unless the minimum is lowered).
interview_engaged.wav    Same questions; the participant elaborates on pleasant
    topics and answers briefly otherwise. The control for the file above: the
    positive/neutral ratio should be well above 1 and anhedonia content low.
focus_group.wav          4 speakers, ~1.3 min. Two moderators (Samantha reads
    the consent lines, Daniel hands the floor), two participants. Tests group
    mode, moderator detection with brief moderators, per-speaker measures, and
    "Remember this voice".
spanish_interview.wav    2 speakers in Spanish. Tests language detection and
    that a non-English recording is refused (or masked by the Languages add-on).
"""


def _clip(voice: str, text: str, tmp: Path, n: int) -> bytes:
    aiff, wav_path = tmp / f"{n}.aiff", tmp / f"{n}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav_path)],
                   check=True)
    with wave.open(str(wav_path)) as w:
        return w.readframes(w.getnframes())


def _write(path: Path, turns: list[tuple[str, str]]) -> None:
    with tempfile.TemporaryDirectory() as d, wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        for n, (voice, text) in enumerate(turns):
            out.writeframes(_clip(voice, text, Path(d), n) + b"\x00\x00" * GAP_SAMPLES)
    print("wrote", path)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "ClinicalWhisper" / "Test audio" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    interview = lambda answers: [t for (q, _), a in zip(QUESTIONS, answers)
                                 for t in (("Samantha", q), ("Daniel", a))]
    _write(out / "interview_anhedonic.wav", interview(ANHEDONIC))
    _write(out / "interview_engaged.wav", interview(ENGAGED))
    _write(out / "focus_group.wav", FOCUS_GROUP)
    _write(out / "spanish_interview.wav", SPANISH)
    (out / "README.txt").write_text(README)


if __name__ == "__main__":
    main()
