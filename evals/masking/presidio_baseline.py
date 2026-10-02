#!/usr/bin/env python3
"""Mask the evaluation sentences with Microsoft Presidio, for comparison.

Presidio (MIT) is the most widely used open-source de-identification tool. It
is run here as shipped: AnalyzerEngine with its default recognisers and
spaCy en_core_web_lg, every entity type, default score threshold. Each
finding is replaced by a lowercase tag ("[person]") so evaluate.py scores it
exactly like ClinicalWhisper's masker.

It needs its own environment (it is not part of the app):

    uv venv -p 3.12 .venv-presidio
    uv pip install -p .venv-presidio/bin/python presidio-analyzer presidio-anonymizer \\
        "en_core_web_lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
    .venv-presidio/bin/python evals/masking/presidio_baseline.py --set held_out.jsonl --out /tmp/presidio_held_out.jsonl
    .venv/bin/python evals/masking/evaluate.py --set held_out.jsonl --masked /tmp/presidio_held_out.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="set.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()
    rows = [json.loads(l) for l in Path(__file__).with_name(args.set).read_text().splitlines()]
    with args.out.open("w") as f:
        for r in rows:
            found = analyzer.analyze(text=r["text"], language="en")
            ops = {e.entity_type: OperatorConfig("replace", {"new_value": f"[{e.entity_type.lower()}]"})
                   for e in found}
            masked = anonymizer.anonymize(text=r["text"], analyzer_results=found, operators=ops).text
            f.write(json.dumps({"id": r["id"], "masked": masked}) + "\n")
    print(f"{len(rows)} sentences -> {args.out}")


if __name__ == "__main__":
    main()
