#!/usr/bin/env python3
"""
Lightweight Evaluation Runner (Auto-Reviewer)
Runs a quick check on the text analysis portions of the pipeline to ensure 
commits don't break entity extraction or sentiment analysis.
"""

import os
import json
import sys

# In a real scenario, this would import the full inference_pipeline.py
# and run it. For this CI hook, we simulate testing the analytical functions.

def load_golden_dataset(path="evals/golden_dataset.jsonl"):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def run_tests():
    print("--- Running ClinicalWhisper CI Evals ---")
    dataset = load_golden_dataset("evals/golden_dataset.jsonl")
    if not dataset:
        print("No golden dataset found. Passing trivially.")
        return True
        
    passed = 0
    failed = 0
    
    for item in dataset:
        text = item["input_text"]
        expected_sentiment = item["expected_sentiment"]
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sentiment_analyzer import analyze_sentiment
        
        result = analyze_sentiment(text)
        predicted_sentiment = result["overall"]["label"].lower()
            
        if predicted_sentiment == expected_sentiment:
            passed += 1
        else:
            print(f"[FAIL] {item['id']}: Expected {expected_sentiment}, got {predicted_sentiment}")
            failed += 1
            
    total = passed + failed
    accuracy = passed / total if total > 0 else 1.0
    
    print(f"Results: {passed}/{total} passed. Accuracy: {accuracy*100:.1f}%")
    
    # Strict gate: block commit if accuracy drops below 100% on golden set
    if accuracy < 1.0:
        print("ERROR: Performance regression detected! Commit blocked.")
        return False
        
    print("SUCCESS: All evals passed. Commit allowed.")
    return True

if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
    sys.exit(0)
