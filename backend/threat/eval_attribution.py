
from r4_detector import classify

TEST_CASES = [
    {
        "name": "Confirmed attack indicators",
        "expected": "attack",
        "physics": [
            {
                "kind": "invalid_reading",
                "detail": "Sensor produced an invalid reading",
                "score": 1.0
            }
        ],
        "replay": [
            {
                "kind": "repeated_segment",
                "detail": "Repeated sensor sequence detected",
                "score": 1.0
            }
        ],
        "context": {}
    },
    {
        "name": "Known natural city event",
        "expected": "natural",
        "physics": [],
        "replay": [],
        "context": {"events": ["match_end"]}
    },
    {
        "name": "No supporting evidence",
        "expected": "unknown",
        "physics": [],
        "replay": [],
        "context": {}
    },
    {
        "name": "Physics anomaly only",
        "expected": "unknown",
        "physics": [
            {
                "kind": "invalid_reading",
                "detail": "Invalid sensor value",
                "score": 0.8
            }
        ],
        "replay": [],
        "context": {}
    },
    {
        "name": "Replay anomaly only",
        "expected": "unknown",
        "physics": [],
        "replay": [
            {
                "kind": "possible_replay",
                "detail": "Repeated sensor tick",
                "score": 0.8
            }
        ],
        "context": {}
    }
]


def run_evaluation():
    total = len(TEST_CASES)
    correct = 0

    labels = ["natural", "attack", "unknown"]

    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }

    print("\n========== R4 ATTRIBUTION EVALUATION ==========\n")

    for case in TEST_CASES:
        result = classify(
            case["physics"],
            case["replay"],
            case["context"]
        )

        predicted = result["label"]
        expected = case["expected"]

        if predicted == expected:
            correct += 1

        confusion[expected][predicted] += 1

        status = "PASS" if predicted == expected else "FAIL"

        print(f"{status}: {case['name']}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  Confidence: {result['confidence']}")
        print()

    accuracy = (correct / total) * 100

    print("========== RESULTS ==========")
    print(f"Total scenarios: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nConfusion Matrix")
    print("Rows = actual, Columns = predicted")
    print("             Natural  Attack  Unknown")

    for actual in labels:
        values = confusion[actual]
        print(
            f"{actual:10} "
            f"{values['natural']:7} "
            f"{values['attack']:7} "
            f"{values['unknown']:8}"
        )

    return accuracy


if __name__ == "__main__":
    run_evaluation()
