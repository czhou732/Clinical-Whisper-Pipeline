"""Which language a transcript is in, so masking and scoring can follow it.

The transcription model handles 50+ languages, but the rest of the pipeline is
language-specific: the default masking model is English-only (on other
languages it misses names, which is worse than failing), the clinical scores
and the review keywords were built and validated in English, and the filler
list is English.

Detection needs no model. Non-Latin scripts identify themselves; Latin-script
languages are told apart by their most common function words, which a
transcript of a few hundred words contains many times over.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish", "vi": "Vietnamese",
    "id": "Indonesian", "tl": "Tagalog", "pl": "Polish", "ro": "Romanian",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "he": "Hebrew",
    "hi": "Hindi", "bn": "Bengali", "te": "Telugu", "th": "Thai", "ru": "Russian",
    "uk": "Ukrainian", "el": "Greek", "fa": "Persian", "ur": "Urdu",
}

# Frequent function words; few are shared, and none are shared by every list.
_STOPWORDS = {
    "en": "the and to of a i that it you in is was for on but so like have not with my what just they be this do know yeah me we had said yes he she are think don't i'm it's",
    "es": "de que y el la en no los se lo un por con para una es me pero mi como más muy eso yo está",
    "fr": "le de et les des que je la est pas un une en pour qui dans ce il ne vous mais ça oui",
    "de": "der die und ich das ist nicht zu ein es den mit sie auf auch eine wir dass aber ja so",
    "it": "di che e il la non è un per in sono mi ma una con lo ho anche gli sì cosa",
    "pt": "de que não o a e é um para com uma os eu se mas na no isso muito também sim",
    "nl": "de het een en van ik je dat is niet op te ze maar met dat wel ook zijn",
    "tr": "bir ve bu da de çok ne ben için ama değil var gibi mi o şey evet yani",
    "vi": "của và là có không tôi các cho được này những người một với trong cũng thì",
    "id": "yang dan di itu ini dengan untuk tidak ada saya ke dari akan juga bisa kita",
    "tl": "ang ng sa na mga at ay hindi ko ako po siya ito lang kasi yung",
    "pl": "nie i w to się na że jest z co jak tak ale do już tylko jestem",
    "ro": "și de în nu că este o pe la cu mai un ce sunt am dar foarte",
}
_ALL = {lang: set(words.split()) for lang, words in _STOPWORDS.items()}
# Only words no other list has: "a", "de", "no" and "so" each belong to
# several languages and would make every transcript look mixed.
_STOP = {lang: words - set().union(*(w for other, w in _ALL.items() if other != lang))
         for lang, words in _ALL.items()}
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

# Script -> language, for scripts that (nearly) belong to one language.
_SCRIPTS = [
    ("HANGUL", "ko"), ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("CJK", "zh"),
    ("ARABIC", "ar"), ("HEBREW", "he"), ("DEVANAGARI", "hi"), ("BENGALI", "bn"),
    ("TELUGU", "te"), ("THAI", "th"), ("CYRILLIC", "ru"), ("GREEK", "el"),
]


def _script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for key, lang in _SCRIPTS:
            if key in name:
                counts[lang] += 1
                break
        else:
            counts["latin"] += 1
    return counts


def detect(text: str) -> dict:
    """``{"code", "name", "confidence", "other_share"}`` for a transcript.

    ``other_share`` is the fraction of letters not in the main language's
    script, or of words matching another language's list: a mostly-English
    interview with a stretch in Spanish still needs a masker that reads Spanish.
    """
    text = re.sub(r"\[[a-z_]+_\d+\]", " ", text or "")  # masking tags are not language
    scripts = _script_counts(text)
    letters = sum(scripts.values())
    if letters < 20:
        return {"code": "und", "name": "Unknown", "confidence": 0.0, "other_share": 0.0}

    top, top_n = scripts.most_common(1)[0]
    if top != "latin":
        # Japanese mixes kana with kanji: any kana at all means Japanese.
        code = "ja" if top == "zh" and scripts.get("ja") else top
        return {"code": code, "name": NAMES.get(code, code), "confidence": round(top_n / letters, 2),
                "other_share": round(1 - top_n / letters, 3)}

    words = [w.lower() for w in _WORD.findall(text)]
    hits = Counter()
    for w in words:
        for lang, stop in _STOP.items():
            if w in stop:
                hits[lang] += 1
    if not hits:
        return {"code": "und", "name": "Unknown", "confidence": 0.0,
                "other_share": round(1 - top_n / letters, 3)}
    (code, n), *rest = hits.most_common()
    runner_up = rest[0][1] if rest else 0
    # A non-English verdict needs real evidence: a short excerpt full of names
    # ("So we had Maria, we had Carlos") otherwise reads as Spanish by chance.
    if code != "en" and (n < 4 or len(words) < 20):
        return {"code": "und", "name": "Unknown", "confidence": 0.0,
                "other_share": round(1 - top_n / letters, 3)}
    confidence = round(n / (n + runner_up), 2)
    non_latin = 1 - top_n / letters
    # Words of a second language well above chance (shared words like "a",
    # "de", "no" give every list some hits) suggest a stretch in it.
    second = runner_up / max(n, 1)
    # A handful of hits is chance on a short clip; ask for at least five.
    other = max(non_latin, second if second > 0.15 and runner_up >= 5 else 0.0)
    return {"code": code, "name": NAMES.get(code, code), "confidence": confidence,
            "other_share": round(other, 3)}


def is_english(result: dict) -> bool:
    """English, with nothing else in it worth masking in another language."""
    return result.get("code") == "en" and result.get("other_share", 0.0) < 0.1
