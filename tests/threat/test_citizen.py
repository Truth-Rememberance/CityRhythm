
import unittest

from threat.citizen import extract_report, report_sensor_agreement


class TestCitizenReports(unittest.TestCase):

    def test_traffic_extraction(self):
        result = extract_report("There is a huge traffic jam")
        self.assertEqual(result["category"], "traffic")

    def test_power_extraction(self):
        result = extract_report("Electricity power outage")
        self.assertEqual(result["category"], "power")

    def test_unknown_report(self):
        result = extract_report("Hello everyone")
        self.assertEqual(result["category"], "unknown")

    def test_kannada_detection(self):
        result = extract_report("ನೀರು ಸರಬರಾಜು ಇಲ್ಲ")
        self.assertEqual(result["language"], "Kannada")

    def test_unverified_burst(self):
        reports = [
            {
                "zone": "silk_board",
                "text": "Traffic jam reported"
            }
            for _ in range(5)
        ]

        readings = []

        result = report_sensor_agreement(
            reports, readings
        )

        self.assertEqual(result[0]["unverified"], 5)
        self.assertTrue(result[0]["suspicious_burst"])


if __name__ == "__main__":
    unittest.main()
