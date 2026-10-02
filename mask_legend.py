"""A plain-English key to the tags that replace identifiers in a transcript.

``[first_name_2]`` means little to a reader. The key says what was hidden
and how the numbering works, without revealing anything that was masked.
"""

from __future__ import annotations

import re
from collections import defaultdict

_TAG = re.compile(r"\[([a-z_]+?)_(\d+)\]")

# Plural, plain names for the masker's labels; others fall back to the label.
_NAMES = {
    "first_name": "first names", "last_name": "surnames", "name": "names",
    "city": "cities", "state": "states", "country": "countries",
    "street_address": "street addresses", "zipcode": "zip codes", "location": "places",
    "date": "dates", "time": "times", "age": "ages",
    "email": "email addresses", "phone_number": "phone numbers", "url": "web addresses",
    "organization": "organizations", "company_name": "companies",
    "occupation": "job titles", "ssn": "ID numbers", "medical_record_number": "record numbers",
}


def legend(text: str) -> str:
    """Lines explaining the masking tags in ``text``; empty if there are none."""
    found: dict[str, set[int]] = defaultdict(set)
    for label, n in _TAG.findall(text or ""):
        found[label].add(int(n))
    if not found:
        return ""
    lines = [("Hidden identifiers. Each tag stands for one specific word, and the same "
              "tag means the same word everywhere in this recording: [first_name_1] is "
              "always the same person, and [first_name_2] is a different one.")]
    for label in sorted(found, key=lambda k: (-len(found[k]), k)):
        nums = sorted(found[label])
        what = _NAMES.get(label, label.replace("_", " ") + "s")
        count = len(nums)
        if count == 1:
            what = what[:-1] if what.endswith("s") and not what.endswith("ss") else what
            if what.endswith("ie"):  # cities -> citie -> city
                what = what[:-2] + "y"
        examples = ", ".join(f"[{label}_{n}]" for n in nums[:3]) + (" ..." if count > 3 else "")
        lines.append(f"  {count} different {what}: {examples}" if count > 1
                     else f"  1 {what}: {examples}")
    return "\n".join(lines)
