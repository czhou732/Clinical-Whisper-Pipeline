#!/usr/bin/env python3
"""A labelled set for measuring how many identifiers the masker catches.

Interview-style sentences with invented identifiers in known positions. All
names, places and numbers are made up; the set can be shared and committed.
It deliberately includes the hard cases seen in real transcripts:

* a name after a relationship word with no title ("my sister Elena");
* names that are also ordinary words (Hope, Grace, Will, June, Summer);
* names from many backgrounds (Hispanic, Chinese, Indian, Arabic, African,
  Vietnamese, Korean, Anglo), since recall can differ by origin;
* places, streets, workplaces, dates, ages, phone numbers and emails as a
  speech recogniser writes them ("five five five...", "dot com");
* the same name repeated later in the transcript.

Writes evals/masking/set.jsonl: {"id", "text", "entities": [{"start", "end", "type", "origin"}]}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

NAMES = {
    "anglo": ["Emily", "Jacob", "Megan", "Tyler", "Hannah", "Connor", "Abigail", "Logan"],
    "hispanic": ["Elena", "Mateo", "Lucia", "Diego", "Camila", "Javier", "Ximena", "Rodrigo"],
    "chinese": ["Wei", "Xiaoming", "Mei", "Jun", "Yuxuan", "Lihua", "Zhang Wei", "Chengdong"],
    "indian": ["Priya", "Arjun", "Ananya", "Rohan", "Deepika", "Vikram", "Sanjana", "Aditya"],
    "arabic": ["Omar", "Fatima", "Yusuf", "Layla", "Khalid", "Amira", "Tariq", "Noor"],
    "african": ["Chidi", "Amara", "Kwame", "Ngozi", "Tendai", "Abena", "Oluwaseun", "Zuri"],
    "vietnamese_korean": ["Linh", "Minh", "Thao", "Ji-woo", "Seo-yeon", "Hyun", "Bao", "Min-jun"],
    "common_word": ["Hope", "Grace", "Will", "June", "Summer", "Rose", "Faith", "Hunter"],
}
SURNAMES = ["Delgado", "Nguyen", "Okafor", "Patel", "Chen", "Haddad", "Kowalski", "Brennan",
            "Ramirez", "Liu", "Mensah", "Sharma", "Park", "Feldman", "Abdullah", "Tran"]
CITIES = ["Sacramento", "Tucson", "Duluth", "Fresno", "Boise", "Shenzhen", "Pune", "Lagos",
          "Monterrey", "Hanoi", "Wugang", "Katy", "Pasadena", "Riverside", "Dearborn", "Chandler"]
STREETS = ["Hollis Street", "Maple Avenue", "Figueroa Street", "Elm Court", "Ocean Drive",
           "Juniper Lane", "Vermont Avenue", "Kingsley Road"]
ORGS = ["Riverside Logistics", "Saint Mary's Hospital", "Bayview High School", "Kroger",
        "Lincoln Elementary", "the Hollis Clinic", "Desert Valley Bank", "Pinecrest Dental"]
RELATIONS = ["sister", "brother", "mom", "dad", "friend", "cousin", "wife", "husband", "son",
             "daughter", "roommate", "boss", "therapist", "neighbor", "girlfriend", "boyfriend"]
MONTHS = ["January", "March", "June", "August", "October", "December"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Each template returns (text, entities) via fill(); slots are {NAME}, {FULL},
# {CITY}, {STREET}, {ORG}, {DATE}, {AGE}, {PHONE}, {EMAIL}.
TEMPLATES = [
    "Mostly I talk to my {REL} {NAME}, she checks on me every day.",
    "My {REL} {NAME} drove me here today.",
    "I live in {CITY} with my {REL} {NAME}.",
    "{NAME} and I used to go running on Sundays.",
    "My name is {FULL} and I'm {AGE} years old.",
    "I work at {ORG} over on {STREET}.",
    "I moved from {CITY} back in {DATE}.",
    "You can reach me at {PHONE}.",
    "My email is {EMAIL}.",
    "Doctor {SURNAME} at {ORG} changed my medication.",
    "I told {NAME} I couldn't come, and {NAME} got upset.",
    "We grew up on {STREET} in {CITY}.",
    "My {REL}, {FULL}, is the only one who knows.",
    "Last {DATE} I stayed with {NAME} for a week.",
    "I haven't seen {NAME} since the funeral.",
    "Honestly {NAME} is the reason I'm still going.",
]


# Written after the safety-net rules, never used to tune them: the honest test.
HELD_OUT = [
    "Yeah so {NAME} kind of took over after that, {NAME} handles the bills now.",
    "Her name's {NAME}, we met at {ORG} like ten years ago.",
    "I'd been seeing a counselor named {FULL} for a while.",
    "We drove up from {CITY} on {DATE} and stayed through the weekend.",
    "Text me at {PHONE} if anything changes.",
    "The place on {STREET}, near the {ORG}, that's where it happened.",
    "It's {EMAIL}, all lowercase.",
    "Uh, {SURNAME}, Doctor {SURNAME}, she's the one who referred me.",
    "My {REL} {NAME} and her kids are staying with us in {CITY}.",
    "I turned {AGE} last month and nobody called.",
    "{FULL} from church brought food over.",
    "Okay. So. {NAME}. {NAME} is my {REL}, and honestly that's complicated.",
]

# Written Oct 2 2026, after every rule and fix, before any run on it: the
# replication set (held_out_2.jsonl). New phrasing, more speech disfluency.
HELD_OUT_2 = [
    "So um {NAME} called again last night, like three times.",
    "I grew up outside {CITY}, it's a small place.",
    "He works over at {ORG} doing nights.",
    "Her number's {PHONE}, if you need it.",
    "Yeah it's {EMAIL}.",
    "Mm, {FULL}. That was my old case manager.",
    "We were living on {STREET} back then.",
    "I think it was {DATE}, maybe a little after.",
    "{NAME}, my {REL}, she doesn't really get it.",
    "I was {AGE} when that happened.",
    "Then Doctor {SURNAME} said we should try something else.",
    "I keep thinking about {NAME}, you know?",
]


def _spoken_phone(rng: random.Random) -> str:
    return " ".join(DIGITS[rng.randrange(10)] for _ in range(3)) + ", " + \
        " ".join(DIGITS[rng.randrange(10)] for _ in range(4))


def fill(template: str, rng: random.Random) -> tuple[str, list[dict]]:
    origin = rng.choice(list(NAMES))
    name = rng.choice(NAMES[origin])
    values = {
        "NAME": (name, "first_name", origin),
        "FULL": (f"{name} {rng.choice(SURNAMES)}", "full_name", origin),
        "SURNAME": (rng.choice(SURNAMES), "last_name", "surname"),
        "CITY": (rng.choice(CITIES), "city", "place"),
        "STREET": (rng.choice(STREETS), "street", "place"),
        "ORG": (rng.choice(ORGS), "organization", "place"),
        "DATE": (f"{rng.choice(MONTHS)} {rng.randint(1, 28)}", "date", "date"),
        "AGE": (str(rng.randint(19, 78)), "age", "number"),
        "PHONE": (_spoken_phone(rng) if rng.random() < 0.5 else f"555-{rng.randint(1000, 9999)}",
                  "phone", "number"),
        "EMAIL": (f"{name.lower().replace(' ', '')} dot {rng.choice(SURNAMES).lower()} at gmail dot com",
                  "email", "email"),
        "REL": (rng.choice(RELATIONS), None, None),
    }
    text, ents, i = "", [], 0
    while i < len(template):
        if template[i] == "{":
            j = template.index("}", i)
            value, etype, eorigin = values[template[i + 1:j]]
            if etype:
                # "the" in "the Hollis Clinic" identifies no one; it is not part of the target.
                lead = 4 if etype == "organization" and value.startswith("the ") else 0
                ents.append({"start": len(text) + lead, "end": len(text) + len(value),
                             "type": etype, "origin": eorigin})
            text += value
            i = j + 1
        else:
            text += template[i]
            i += 1
    return text, ents


def main() -> None:
    rng = random.Random(20261001)
    rows = []
    for k in range(25):  # 25 rounds of every template, different fillers each time
        for t_idx, template in enumerate(TEMPLATES):
            text, ents = fill(template, rng)
            rows.append({"id": f"{t_idx:02d}_{k:02d}", "text": text, "entities": ents})
    out = Path(__file__).with_name("set.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} sentences, {sum(len(r['entities']) for r in rows)} identifiers -> {out}")
    held = []
    rng = random.Random(777)
    for k in range(25):
        for t_idx, template in enumerate(HELD_OUT):
            text, ents = fill(template, rng)
            held.append({"id": f"h{t_idx:02d}_{k:02d}", "text": text, "entities": ents})
    out = Path(__file__).with_name("held_out.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in held))
    print(f"{len(held)} held-out sentences, {sum(len(r['entities']) for r in held)} identifiers -> {out}")
    held2 = []
    rng = random.Random(20261002)
    for k in range(25):
        for t_idx, template in enumerate(HELD_OUT_2):
            text, ents = fill(template, rng)
            held2.append({"id": f"r{t_idx:02d}_{k:02d}", "text": text, "entities": ents})
    out = Path(__file__).with_name("held_out_2.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in held2))
    print(f"{len(held2)} replication sentences, {sum(len(r['entities']) for r in held2)} identifiers -> {out}")


if __name__ == "__main__":
    main()
