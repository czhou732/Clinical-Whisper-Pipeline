#!/usr/bin/env python3
"""
Clinical Scoring Evaluation Runner — ClinicalWhisper v5

Evaluates ``llm_clinical_scorer.score_transcript`` — the module that actually
ships in the v5 pipeline — against ``evals/clinical_golden_dataset.jsonl``.

Replaces the v3 sentiment gate (``run_evals.py``), which scored
``sentiment_analyzer.py``: a module no longer imported by
``inference_pipeline.py`` and structurally unable to detect clinical masking
or anhedonia.

Inputs are built through the real pipeline helpers — ``process_segments`` and
``build_acoustic_prompt_context`` — so the eval exercises the v5 path from
diarized segments onward rather than hand-written prompt strings.

Gate: at least 85% of assertions must hold.

Usage:
  python evals/run_clinical_evals.py                 # full run (needs local LLM)
  python evals/run_clinical_evals.py --self-test     # model-free, <1s, CI path
  python evals/run_clinical_evals.py --no-cache      # force re-scoring
  python evals/run_clinical_evals.py --only anh_01   # single case
  python evals/run_clinical_evals.py --report r.json # write report artifact

Exit codes:
  0  gate passed
  1  gate failed
  2  cannot run (no LLM backend and results not cached) — see --fallback-self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from acoustic_context import build_acoustic_prompt_context  # noqa: E402
from llm_clinical_scorer import (  # noqa: E402
    CLINICAL_SCORING_PROMPT,
    REQUIRED_SCORE_KEYS,
    _load_scoring_config,
    score_transcript,
)
from transcript_formatter import process_segments  # noqa: E402

log = logging.getLogger("ClinicalWhisper.evals")

DEFAULT_DATASET = EVALS_DIR / "clinical_golden_dataset.jsonl"
DEFAULT_CACHE_DIR = EVALS_DIR / ".cache"
GATE = 0.85

_RELATION_OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and schema-check the clinical golden dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    cases: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{lineno} is not valid JSON: {exc}") from exc

    if not cases:
        raise ValueError(f"Golden dataset is empty: {path}")

    seen: set[str] = set()
    for case in cases:
        validate_case(case, seen)
    return cases


def validate_case(case: dict[str, Any], seen_ids: set[str]) -> None:
    """Raise ValueError if a dataset record is malformed.

    A silently malformed case would quietly shrink the assertion count and
    inflate the pass rate, so every field is checked rather than defaulted.
    """
    case_id = case.get("id")
    if not case_id:
        raise ValueError(f"Case is missing 'id': {case!r:.120}")
    if case_id in seen_ids:
        raise ValueError(f"Duplicate case id: {case_id}")
    seen_ids.add(case_id)

    for field in ("category", "segments", "expect"):
        if field not in case:
            raise ValueError(f"[{case_id}] missing required field '{field}'")

    if not case["segments"]:
        raise ValueError(f"[{case_id}] 'segments' is empty")

    for seg in case["segments"]:
        for field in ("start", "end", "speaker", "text"):
            if field not in seg:
                raise ValueError(f"[{case_id}] segment missing '{field}'")

    if not case["expect"] and not case.get("relations"):
        raise ValueError(f"[{case_id}] asserts nothing — no 'expect' bands and no 'relations'")

    for dim, band in case["expect"].items():
        if dim not in REQUIRED_SCORE_KEYS:
            raise ValueError(f"[{case_id}] unknown scoring dimension '{dim}'")
        if not isinstance(band, list) or len(band) != 2:
            raise ValueError(f"[{case_id}] band for '{dim}' must be [lo, hi], got {band!r}")
        lo, hi = band
        if not all(isinstance(v, (int, float)) for v in (lo, hi)):
            raise ValueError(f"[{case_id}] band for '{dim}' must be numeric, got {band!r}")
        if not (0 <= lo <= hi <= 10):
            raise ValueError(f"[{case_id}] band for '{dim}' must satisfy 0 <= lo <= hi <= 10, got {band!r}")

    for rel in case.get("relations", []):
        if not isinstance(rel, list) or len(rel) != 3:
            raise ValueError(f"[{case_id}] relation must be [dim, op, dim], got {rel!r}")
        left, op, right = rel
        if op not in _RELATION_OPS:
            raise ValueError(f"[{case_id}] unknown relation operator '{op}'")
        for dim in (left, right):
            if dim not in REQUIRED_SCORE_KEYS:
                raise ValueError(f"[{case_id}] unknown scoring dimension '{dim}' in relation")


# ---------------------------------------------------------------------------
# Prompt inputs — built through the real pipeline helpers
# ---------------------------------------------------------------------------

def build_inputs(case: dict[str, Any]) -> tuple[str, str]:
    """Build (structured_transcript, acoustic_context) exactly as the pipeline does."""
    structured = process_segments(case["segments"])
    transcript = structured.get("structured_transcript", "")
    if not transcript:
        raise ValueError(f"[{case['id']}] produced an empty structured transcript")

    acoustics = case.get("acoustics", {})
    context = build_acoustic_prompt_context(
        acoustics.get("overall", {}),
        acoustics.get("by_speaker", {}),
    )
    return transcript, context


def cache_key(model: str, transcript: str, context: str) -> str:
    """Hash the full prompt and model — the exact inputs that determine the output.

    Editing the prompt template, the model, or a case invalidates the entry, so
    the gate re-scores precisely when the thing it gates changes.
    """
    prompt = CLINICAL_SCORING_PROMPT.format(transcript=transcript, acoustic_context=context)
    return hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def get_scores(
    case: dict[str, Any],
    config: dict[str, Any],
    cache_dir: Optional[Path],
) -> tuple[dict[str, Any], bool]:
    """Return (scoring_dict, from_cache) for one case."""
    transcript, context = build_inputs(case)
    model = config.get("llm_scoring", {}).get("mlx_model", "unknown")

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{cache_key(model, transcript, context)}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8")), True
            except json.JSONDecodeError:
                log.warning("Discarding corrupt cache entry: %s", cache_path.name)

    scoring = score_transcript(transcript, context, config)

    if cache_path is not None and not _scoring_failed(scoring):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(scoring, indent=2), encoding="utf-8")

    return scoring, False


def failed_checks(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Every assertion a case would have made, marked failed.

    Used when the scorer itself fails. It must cover bands AND relations —
    counting only bands would quietly shrink the denominator on exactly the
    cases where the model is broken.
    """
    dims = sorted(case["expect"]) + [
        f"{left} {op} {right}" for left, op, right in case.get("relations", [])
    ]
    return [{"kind": "scorer", "dimension": d, "expected": "valid score",
             "actual": "scorer failure", "passed": False, "deviation": None}
            for d in dims]


def _scoring_failed(scoring: dict[str, Any]) -> Optional[str]:
    """Return an error string if the scorer fell back to defaults, else None.

    ``score_transcript`` returns DEFAULT_SCORES with ``_meta.error`` when
    generation fails and sets ``_parse_error`` when the response is unparseable.
    Both must fail the case — otherwise a dead model scores a placid 5/10 across
    the board and reads as a pass.
    """
    if scoring.get("_parse_error"):
        return f"unparseable LLM response: {scoring['_parse_error']}"
    error = scoring.get("_meta", {}).get("error")
    if error:
        return f"LLM generation failed: {error}"
    return None


def check_case(case: dict[str, Any], scoring: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate every assertion for one case."""
    checks: list[dict[str, Any]] = []

    for dim, (lo, hi) in sorted(case["expect"].items()):
        value = scoring.get(dim)
        if not isinstance(value, (int, float)):
            checks.append({
                "kind": "band", "dimension": dim, "expected": f"[{lo}, {hi}]",
                "actual": repr(value), "passed": False, "deviation": None,
            })
            continue
        passed = lo <= value <= hi
        deviation = 0.0 if passed else float(min(abs(value - lo), abs(value - hi)))
        checks.append({
            "kind": "band", "dimension": dim, "expected": f"[{lo}, {hi}]",
            "actual": value, "passed": passed, "deviation": deviation,
        })

    for left, op, right in case.get("relations", []):
        lval, rval = scoring.get(left), scoring.get(right)
        if not (isinstance(lval, (int, float)) and isinstance(rval, (int, float))):
            passed = False
        else:
            passed = _RELATION_OPS[op](lval, rval)
        checks.append({
            "kind": "relation", "dimension": f"{left} {op} {right}",
            "expected": f"{left} {op} {right}", "actual": f"{lval} vs {rval}",
            "passed": passed, "deviation": None,
        })

    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list[dict[str, Any]], elapsed: float, gate: float) -> float:
    """Print the per-category and per-dimension tables. Returns overall accuracy."""
    total_passed = sum(r["passed_checks"] for r in results)
    total_checks = sum(len(r["checks"]) for r in results)
    accuracy = total_passed / total_checks if total_checks else 0.0

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    by_dimension: dict[str, dict[str, float]] = defaultdict(lambda: {"passed": 0, "total": 0, "dev": 0.0})

    for res in results:
        cat = by_category[res["category"]]
        cat["passed"] += res["passed_checks"]
        cat["total"] += len(res["checks"])
        for chk in res["checks"]:
            dim = by_dimension[chk["dimension"]]
            dim["total"] += 1
            dim["passed"] += 1 if chk["passed"] else 0
            if chk["deviation"]:
                dim["dev"] += chk["deviation"]

    print("-" * 68)
    print(f"{'Category':<26} {'Passed':<10} {'Checks':<10} {'Accuracy':<10}")
    print("-" * 68)
    for cat in sorted(by_category):
        c = by_category[cat]
        acc = c["passed"] / c["total"] if c["total"] else 0.0
        status = "PASS" if acc >= gate else "WARN"
        print(f"{cat:<26} {c['passed']:<10} {c['total']:<10} {acc*100:>5.1f}%  [{status}]")
    print("-" * 68)
    print(f"{'OVERALL':<26} {total_passed:<10} {total_checks:<10} {accuracy*100:>5.1f}%")
    print()

    print("-" * 68)
    print(f"{'Dimension / relation':<34} {'Passed':<9} {'Checks':<9} {'MeanMiss':<9}")
    print("-" * 68)
    for dim in sorted(by_dimension):
        d = by_dimension[dim]
        failed = d["total"] - d["passed"]
        mean_dev = d["dev"] / failed if failed else 0.0
        print(f"{dim:<34} {int(d['passed']):<9} {int(d['total']):<9} {mean_dev:>6.2f}")
    print("-" * 68)
    print("MeanMiss = mean distance outside the expected band, over failing checks only.")
    print(f"\nTime: {elapsed:.1f}s\n")

    failures = [r for r in results if r["passed_checks"] < len(r["checks"]) or r.get("error")]
    if failures:
        print("FAILURES:")
        for res in failures:
            print(f"  [{res['id']}] ({res['category']}) {res['description']}")
            if res.get("error"):
                print(f"    ERROR: {res['error']}")
            for chk in res["checks"]:
                if not chk["passed"]:
                    print(f"    {chk['kind']}: {chk['dimension']} — expected "
                          f"{chk['expected']}, got {chk['actual']}")
        print()

    return accuracy


def write_report(path: Path, results: list[dict[str, Any]], accuracy: float,
                 model: str, elapsed: float, gate: float) -> None:
    """Write the machine-readable report artifact used for the preprint."""
    payload = {
        "model": model,
        "gate": gate,
        "accuracy": round(accuracy, 4),
        "passed": accuracy >= gate,
        "total_checks": sum(len(r["checks"]) for r in results),
        "passed_checks": sum(r["passed_checks"] for r in results),
        "elapsed_seconds": round(elapsed, 1),
        "cases": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# Self-test — model-free, runs anywhere
# ---------------------------------------------------------------------------

def self_test(dataset_path: Path) -> bool:
    """Validate the dataset and the assertion logic without loading a model.

    This is the CI path: a GitHub Ubuntu runner cannot host an 8B model, but it
    can still catch a malformed dataset, an unknown dimension name, an inverted
    band, or a case whose segments fail to build a transcript.
    """
    print("=" * 68)
    print("ClinicalWhisper — Clinical Scoring Eval SELF-TEST (no model)")
    print("=" * 68)

    cases = load_dataset(dataset_path)
    print(f"\n  ✓ dataset schema valid — {len(cases)} cases")

    total_checks = 0
    for case in cases:
        transcript, context = build_inputs(case)
        if "Subject:" not in transcript:
            raise ValueError(
                f"[{case['id']}] no turn was classified as Subject — "
                f"classify_speakers produced: {transcript[:120]!r}"
            )
        if not context.strip():
            raise ValueError(f"[{case['id']}] produced an empty acoustic context")
        total_checks += len(case["expect"]) + len(case.get("relations", []))
    print(f"  ✓ all {len(cases)} cases build a structured transcript and acoustic context")
    print(f"  ✓ {total_checks} assertions defined "
          f"(gate requires {int(-(-total_checks * GATE // 1))} to pass)")

    # Exercise the assertion logic against synthetic scores.
    probe = {
        "id": "_probe", "category": "_probe", "expect": {"affect_flatness": [6, 10]},
        "relations": [["elaboration_negative", ">", "elaboration_positive"]],
    }
    hit = check_case(probe, {"affect_flatness": 8, "elaboration_negative": 7, "elaboration_positive": 2})
    miss = check_case(probe, {"affect_flatness": 2, "elaboration_negative": 1, "elaboration_positive": 9})
    assert all(c["passed"] for c in hit), "assertion logic failed to pass a satisfying score"
    assert not any(c["passed"] for c in miss), "assertion logic passed a violating score"
    assert miss[0]["deviation"] == 4.0, f"band deviation miscomputed: {miss[0]['deviation']}"

    # A fallback/unparseable result must be detected as a failure.
    assert _scoring_failed({"_meta": {"error": "boom"}}), "generation failure not detected"
    assert _scoring_failed({"_parse_error": "bad json"}), "parse failure not detected"
    assert not _scoring_failed({"_meta": {"model": "x"}}), "healthy result flagged as failed"

    # A scorer failure must cost the case every assertion it would have made,
    # relations included — otherwise a broken model shrinks the denominator.
    for case in cases + [probe]:
        expected_n = len(case["expect"]) + len(case.get("relations", []))
        actual_n = len(failed_checks(case))
        assert actual_n == expected_n, (
            f"[{case['id']}] scorer-failure path counts {actual_n} assertions, "
            f"expected {expected_n}"
        )
    print("  ✓ assertion, deviation, and failure-detection logic verified")

    print("\nPASS: self-test complete. This does NOT evaluate model quality —\n"
          "      run without --self-test on a machine with the local LLM for that.\n")
    return True


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def mlx_available() -> bool:
    """True if the MLX backend that llm_clinical_scorer prefers can be used."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    cases = load_dataset(dataset_path)

    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
        if not cases:
            print(f"No cases matched --only {args.only}", file=sys.stderr)
            return 1

    config = _load_scoring_config(args.config)
    if args.max_tokens is not None:
        # Diagnostic override. The gate itself runs on the shipping config —
        # this exists to tell "the model judged wrongly" apart from "the
        # response was cut off before the JSON closed".
        config["llm_scoring"]["max_tokens"] = args.max_tokens
    model = config.get("llm_scoring", {}).get("mlx_model", "unknown")
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    print("=" * 68)
    print("ClinicalWhisper v5 — Clinical Scoring Evaluation")
    print("=" * 68)
    print(f"\nModel:   {model}")
    print(f"Dataset: {dataset_path.name} ({len(cases)} cases)")
    print(f"Cache:   {'disabled' if cache_dir is None else cache_dir}")
    print()

    # If anything would need the model and no backend exists, say so plainly
    # rather than emitting a run of fallback zeros that looks like a real result.
    if not mlx_available():
        uncached = [
            c for c in cases
            if cache_dir is None
            or not (cache_dir / f"{cache_key(model, *build_inputs(c))}.json").exists()
        ]
        if uncached:
            print("=" * 68)
            print("CANNOT RUN: no local LLM backend (mlx_lm on Apple Silicon) and "
                  f"{len(uncached)} case(s) are not cached.")
            print("This machine cannot evaluate the shipping clinical scorer.")
            print("=" * 68)
            if args.fallback_self_test:
                print("\nFalling back to --self-test.\n")
                return 0 if self_test(dataset_path) else 1
            return 2

    results: list[dict[str, Any]] = []
    cache_hits = 0
    start = time.time()

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case['id']} ({case['category']})...", end="", flush=True)
        try:
            scoring, from_cache = get_scores(case, config, cache_dir)
        except Exception as exc:
            print(f" ERROR: {exc}")
            results.append({
                "id": case["id"], "category": case["category"],
                "description": case.get("description", ""), "error": str(exc),
                "scores": {}, "checks": [], "passed_checks": 0,
            })
            continue

        cache_hits += 1 if from_cache else 0
        error = _scoring_failed(scoring)
        checks = [] if error else check_case(case, scoring)
        passed = sum(1 for c in checks if c["passed"])

        if error:
            checks = failed_checks(case)

        results.append({
            "id": case["id"], "category": case["category"],
            "description": case.get("description", ""), "error": error,
            "scores": {k: scoring.get(k) for k in REQUIRED_SCORE_KEYS},
            "checks": checks, "passed_checks": passed,
        })
        marker = "cached" if from_cache else f"{scoring.get('_meta', {}).get('elapsed_seconds', '?')}s"
        print(f" {passed}/{len(checks)} ({marker})")

    elapsed = time.time() - start
    print(f"\n{cache_hits}/{len(cases)} case(s) served from cache.\n")

    accuracy = print_report(results, elapsed, args.gate)

    if args.report:
        write_report(Path(args.report), results, accuracy, model, elapsed, args.gate)

    if accuracy < args.gate:
        print(f"FAIL: {accuracy*100:.1f}% of assertions hold — below the "
              f"{args.gate*100:.0f}% gate. Commit blocked.")
        return 1

    print(f"PASS: {accuracy*100:.1f}% of assertions hold — meets the {args.gate*100:.0f}% gate.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_clinical_evals",
        description="Evaluate the v5 LLM clinical scorer against the golden dataset.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to the golden dataset JSONL.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory for cached scorer output.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write the cache.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these case ids.")
    parser.add_argument("--gate", type=float, default=GATE, help=f"Assertion pass-rate gate (default {GATE}).")
    parser.add_argument("--max-tokens", type=int, default=None, metavar="N",
                        help="Diagnostic: override llm_scoring.max_tokens for this run.")
    parser.add_argument("--report", default=None, metavar="PATH", help="Write a JSON report artifact.")
    parser.add_argument("--self-test", action="store_true", help="Model-free dataset and logic check.")
    parser.add_argument("--fallback-self-test", action="store_true",
                        help="If no LLM backend is available, run --self-test instead of exiting 2.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show scorer logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        if args.self_test:
            return 0 if self_test(Path(args.dataset)) else 1
        return run(args)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"\nFAIL: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
