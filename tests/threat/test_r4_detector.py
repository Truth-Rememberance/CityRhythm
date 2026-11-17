
import unittest

from r4_detector import (
    classify,
    physics_consistency,
    replay_detector,
    sensor_trust,
)


class TestR4Detector(unittest.TestCase):

    def test_natural_event_classification(self):
        result = classify(
            [],
            [],
            {"events": ["match_end"]}
        )

        self.assertEqual(result["label"], "natural")

    def test_unknown_without_context(self):
        result = classify([], [], {})

        self.assertEqual(result["label"], "unknown")


    def test_valid_reading(self):
        readings = [
            {"sensor_id": "sensor1", "value": 0.5}
        ]
        self.assertEqual(physics_consistency(readings), [])

    def test_invalid_reading(self):
        readings = [
            {"sensor_id": "sensor1", "value": 1.5}
        ]
        evidence = physics_consistency(readings)
        self.assertTrue(len(evidence) > 0)

    def test_replay_detected(self):
        tick = [{"sensor_id": "s1", "value": 0.5}]
        history = [tick, tick, tick]

        evidence = replay_detector(history)
        self.assertTrue(len(evidence) > 0)

    def test_no_replay(self):
        history = [
            [{"sensor_id": "s1", "value": 0.2}],
            [{"sensor_id": "s1", "value": 0.5}],
            [{"sensor_id": "s1", "value": 0.8}],
        ]

        evidence = replay_detector(history)
        self.assertEqual(evidence, [])

    def test_replay_alone_is_unknown(self):
        result = classify(
            [],
            [{"kind": "possible_replay", "detail": "Repeated pattern", "score": 0.8}]
        )
        self.assertEqual(result["label"], "unknown")

    def test_combined_evidence_is_attack(self):
        result = classify(
            [{"kind": "out_of_range", "detail": "Invalid value", "score": 0.9}],
            [{"kind": "possible_replay", "detail": "Repeated pattern", "score": 0.8}]
        )
        self.assertEqual(result["label"], "attack")

    def test_sensor_trust(self):
        history = [
            [{"sensor_id": "sensor1", "value": 0.5}],
            [{"sensor_id": "sensor1", "value": 0.6}],
        ]

        result = sensor_trust(history)
        self.assertIn("sensor1", result)


if __name__ == "__main__":
    unittest.main()
