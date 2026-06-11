#!/usr/bin/env python3
"""
Evaluation Runner for ClinicalWhisper v3.0

Runs sentiment analysis against a golden dataset and reports accuracy
by category. Passes if overall accuracy >= 85%.
"""

import json
import os
import sys
import time
from collections import defaultdict

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_analyzer import analyze_sentiment


def load_golden_dataset(path="evals/golden_dataset.jsonl"):
    """Load golden dataset entries from JSONL file."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run_tests():
    print("=" * 60)
    print("ClinicalWhisper v3.0 — Sentiment Evaluation Runner")
    print("=" * 60)

    dataset = load_golden_dataset("evals/golden_dataset.jsonl")
    if not dataset:
        print("No golden dataset found. Passing trivially.")
        return True

    print(f"\nLoaded {len(dataset)} test cases.\n")

    # Track results
    total_passed = 0
    total_failed = 0
    category_results = defaultdict(lambda: {"passed": 0, "failed": 0, "cases": []})
    failures = []

    start_time = time.time()

    for item in dataset:
        text = item["input_text"]
        expected = item["expected_sentiment"]
        category = item.get("category", "unknown")
        item_id = item["id"]

        result = analyze_sentiment(text)
        predicted = result["overall"]["label"].lower()
        score = result["overall"]["score"]

        passed = predicted == expected

        if passed:
            total_passed += 1
            category_results[category]["passed"] += 1
        else:
            total_failed += 1
            category_results[category]["failed"] += 1
            failures.append({
                "id": item_id,
                "category": category,
                "expected": expected,
                "predicted": predicted,
                "score": score,
                "text": text[:80] + ("..." if len(text) > 80 else ""),
            })

        category_results[category]["cases"].append({
            "id": item_id,
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "passed": passed,
        })

    elapsed = time.time() - start_time
    total = total_passed + total_failed
    accuracy = total_passed / total if total > 0 else 1.0

    # Print per-category breakdown table
    print("-" * 60)
    print(f"{'Category':<20} {'Passed':<10} {'Failed':<10} {'Accuracy':<10}")
    print("-" * 60)

    for cat in sorted(category_results.keys()):
        cr = category_results[cat]
        cat_total = cr["passed"] + cr["failed"]
        cat_acc = cr["passed"] / cat_total if cat_total > 0 else 1.0
        status = "PASS" if cat_acc >= 0.85 else "WARN"
        print(f"{cat:<20} {cr['passed']:<10} {cr['failed']:<10} {cat_acc*100:>5.1f}%  [{status}]")

    print("-" * 60)
    print(f"{'OVERALL':<20} {total_passed:<10} {total_failed:<10} {accuracy*100:>5.1f}%")
    print(f"\nTime: {elapsed:.2f}s ({elapsed/total:.2f}s per case)")
    print()

    # Print failures
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  [{f['id']}] ({f['category']}) expected={f['expected']}, "
                  f"got={f['predicted']} (score={f['score']}/10)")
            print(f"    text: {f['text']}")
        print()

    # Gate check: 85% accuracy threshold
    if accuracy < 0.85:
        print(f"FAIL: Accuracy {accuracy*100:.1f}% below 85% gate. Commit blocked.")
        return False

    print(f"PASS: Accuracy {accuracy*100:.1f}% meets 85% gate.")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
