import json
import os
import sys

# Ensure current directory is in path
sys.path.append(os.getcwd())

from cognis_expert import predict_precision_diagnostic

def test_precision():
    print("Testing 'whille' typo scenario...")
    res = predict_precision_diagnostic(
        student_id="std_001",
        code="whille x < 10:",
        cursor={"lineNumber": 1, "column": 7},
        key_pressed="e",
        student_history={"past_errors": {}, "struggle_concepts": [], "hint_style": "socratic"}
    )
    print(json.dumps(res, indent=2))

    print("\nTesting 'if x > 5' missing colon scenario...")
    res = predict_precision_diagnostic(
        student_id="std_001",
        code="if x > 5",
        cursor={"lineNumber": 1, "column": 9},
        key_pressed="Enter",
        student_history={"past_errors": {}, "struggle_concepts": [], "hint_style": "socratic"}
    )
    print(json.dumps(res, indent=2))

    print("\nTesting correct code scenario...")
    res = predict_precision_diagnostic(
        student_id="std_001",
        code="if x == 5:",
        cursor={"lineNumber": 1, "column": 10},
        key_pressed=":",
        student_history={"past_errors": {}, "struggle_concepts": [], "hint_style": "socratic"}
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test_precision()
