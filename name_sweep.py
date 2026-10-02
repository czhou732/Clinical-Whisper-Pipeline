"""A second check for names outside English, using the clinical scoring model.

OpenMED's multilingual masker misses many names and places in short spoken
sentences outside English. Measured on evals/masking/multilingual.py (held-out
sentences, one at a time), the model and the language rules caught 64% in
Chinese, 81% Japanese, 75% Korean, 83% Hindi and 88% Spanish; adding this
check (Llama-3-8B, already on the Mac when the Research scoring add-on is
installed) raised that to 80% / 89% / 79% / 91% / 100%, with at most 5% of
other characters masked. English is 98.9% without it. The model reads the
original lines on the Mac and lists names and places; only strings found
verbatim in those lines are masked, so a made-up answer can over-mask but
never unmask anything. Nothing leaves the Mac, and the list is not kept.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable

log = logging.getLogger("ClinicalWhisper")

PROMPT = ("Below are numbered lines from an interview transcript. List every name of a person "
          "and every place name (city, town, region) that appears in them, copied exactly as "
          "written in the original script, one per line. Do not translate. Do not list common "
          "nouns, family words or job titles. If there are none, write NONE.\n\n{lines}")
CHUNK = 15  # lines per request: long enough for context, short enough to list everything
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")


def _keep(value: str, joined: str) -> bool:
    """Whether a listed string is safe and useful to mask."""
    if len(value) < 2 or value.upper() == "NONE" or value not in joined:
        return False
    if _LATIN.match(value):
        # Latin script: a name is capitalised; "mi" or "de" would mask half the text.
        return len(value) >= 3 and value[:1].isupper()
    return True


def candidates(texts: list[str], ask: Callable[[str], str]) -> list[str]:
    """Names and places ``ask`` (a language model) finds in ``texts``, verbatim only."""
    found: dict[str, None] = {}
    for i in range(0, len(texts), CHUNK):
        chunk = [t for t in texts[i:i + CHUNK] if t and t.strip()]
        if not chunk:
            continue
        answer = ask(PROMPT.format(lines="\n".join(f"{k + 1}. {t}" for k, t in enumerate(chunk))))
        joined = "\n".join(chunk)
        for line in answer.split("<|eot_id|>")[0].splitlines():
            value = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip(" ,.。、")
            if _keep(value, joined):
                found.setdefault(value)
    return list(found)


def llama(model_name: str) -> Callable[[str], str]:
    """``ask`` backed by the scoring model (loaded once, greedy decoding)."""
    import llm_clinical_scorer as lcs

    model, tok = lcs._mlx_model(model_name)
    sampler = lcs.make_sampler(temp=0.0)

    def ask(prompt: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        answer = lcs.generate(model, tok, prompt=text, max_tokens=300, sampler=sampler)
        # Llama-3 under mlx_lm can run past its end-of-turn marker and repeat
        # itself; everything after the first marker is not part of the answer.
        return answer.split("<|eot_id|>")[0]
    return ask


def mask(texts: list[str], originals: list[str], ask: Callable[[str], str],
         number: Callable[[str, str], int]) -> tuple[list[str], int]:
    """Mask names ``ask`` finds in ``originals`` wherever they are still exposed in ``texts``."""
    import pii_rules

    names: Iterable[str] = candidates(originals, ask)
    # Numbers only for names still exposed somewhere, so names the masker
    # already caught are not counted twice in the summary.
    found = [(v, f"[full_name_{number('full_name', pii_rules._key(v))}]") for v in names
             if any(v in t for t in texts) or _partly_exposed(v, texts)]
    if not found:
        return texts, 0
    texts, total = pii_rules.propagate(texts, found)
    texts, extra = _finish_partial(texts, found)
    return texts, total + extra


_TAGS = r"(?:\[[a-z_]+(?:_\d+)?\])"


def _partly_exposed(value: str, texts: list[str]) -> bool:
    """Whether some piece of ``value`` sits next to a tag (see _finish_partial)."""
    rx = _partial_rx(value)
    return rx is not None and any(rx.search(t) for t in texts)


def _partial_rx(value: str) -> "re.Pattern | None":
    """Pieces of a no-space-script name touching a tag; None for other scripts."""
    import pii_rules_intl

    if len(pii_rules_intl.SCRIPT.findall(value)) < 2:
        return None
    pieces = {value[i:j] for i in range(len(value)) for j in range(i + 1, len(value) + 1)}
    pieces.discard(value)
    alts = []
    for p in sorted(pieces, key=len, reverse=True):
        e = re.escape(p)
        if value.startswith(p):
            alts.append(rf"{e}(?={_TAGS})")
        if value.endswith(p):
            alts.append(rf"(?<=\]){e}")
        alts.append(rf"(?<=\]){e}(?={_TAGS})")
    return re.compile("|".join(alts))


def _finish_partial(texts: list[str], found: list[tuple[str, str]]) -> tuple[list[str], int]:
    """Mask what is left of a found name that the masker tagged only in part.

    "山田拓海" may come back from OpenMED as "[last_name_1]拓[first_name_1]";
    the whole name is then not in the text, so the leftover "拓" is masked
    when it touches a tag and is a piece of a name the model listed (a prefix
    before a tag, a suffix after one, or any piece between two). Only scripts
    without spaces, where names have no word edges to match on.
    """
    total = 0
    for value, tag in found:
        rx = _partial_rx(value)
        if rx is None:
            continue
        out = []
        for t in texts:
            new, n = rx.subn(tag, t)
            out.append(new)
            total += n
        texts = out
    return texts, total
