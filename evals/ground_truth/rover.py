"""ROVER: combine several transcripts of the same audio by word-level voting.

Each recogniser fails on different words, so where two of three agree the
majority is usually right (Fiscus, 1997, "A post-processing system to yield
reduced word error rates: Recognizer Output Voting Error Reduction").

The hypotheses are aligned into one word-transition network: the first is the
pivot, and each further one is aligned to the network by edit distance, adding
a slot (filled with "" for the others) wherever it inserts a word. Each slot
then takes the word most systems put there; "" winning means the word is
dropped. Ties go to the earlier system in the list, so put the best single
system first.

Used for the evaluation in evals/ground_truth (does voting beat MOSS alone on
AMI?), not in the app: shipping it would mean running two more recognisers.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

EPS = ""


def _align(network: list[list[str]], words: list[str]) -> list[tuple[Optional[int], Optional[int]]]:
    """Edit-distance alignment of ``words`` against network slots.

    A slot matches a word at no cost if any system already has that word
    there. Returns pairs (slot index or None, word index or None).
    """
    n, m = len(network), len(words)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        slot = network[i - 1]
        for j in range(1, m + 1):
            sub = 0 if words[j - 1] in slot else 1
            cost[i][j] = min(cost[i - 1][j - 1] + sub, cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    out: list[tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (0 if words[j - 1] in network[i - 1] else 1):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def combine(hypotheses: Sequence[str]) -> str:
    """The voted transcript of several hypotheses of the same audio."""
    hyps = [h.split() for h in hypotheses if h is not None]
    if not hyps:
        return ""
    network: list[list[str]] = [[w] for w in hyps[0]]  # slot -> word per system so far
    for k, words in enumerate(hyps[1:], start=1):
        new: list[list[str]] = []
        for slot_i, word_j in _align(network, words):
            if slot_i is None:  # insertion: a slot no earlier system filled
                new.append([EPS] * k + [words[word_j]])
            elif word_j is None:
                new.append(network[slot_i] + [EPS])
            else:
                new.append(network[slot_i] + [words[word_j]])
        network = new
    voted = []
    for slot in network:
        counts = Counter(slot)
        best = max(counts.values())
        # Ties go to the earliest system that has one of the tied words.
        word = next(w for w in slot if counts[w] == best)
        if word != EPS:
            voted.append(word)
    return " ".join(voted)


def combine_segments(per_system: Sequence[list[dict]]) -> list[dict]:
    """Vote segment by segment when the systems share segment boundaries.

    The usual setup here: MOSS fixes who spoke when, and every recogniser
    re-transcribes MOSS's segments, so segment i is the same audio in each.
    """
    base = per_system[0]
    out = []
    for i, seg in enumerate(base):
        texts = [sys_segs[i].get("text", "") for sys_segs in per_system if i < len(sys_segs)]
        out.append({**seg, "text": combine(texts)})
    return out
