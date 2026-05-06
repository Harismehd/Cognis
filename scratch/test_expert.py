import json
from cognis_expert import predict_intervention

def test_inference():
    print("Testing 'rint' typo scenario...")
    res = predict_intervention(
        student_id="std_001",
        code="p",
        cursor={"lineNumber": 1, "column": 2},
        key_pressed="r",
        student_history={"past_errors": {}, "struggle_concepts": [], "hint_style": "direct"}
    )
    print(json.dumps(res, indent=2))

    print("\nTesting 'if x =' assignment scenario...")
    res = predict_intervention(
        student_id="std_001",
        code="if x ",
        cursor={"lineNumber": 1, "column": 6},
        key_pressed="=",
        student_history={"past_errors": {"conditionals": 2}, "struggle_concepts": ["conditionals"], "hint_style": "socratic"}
    )
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    test_inference()
