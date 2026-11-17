
import unittest

from threat.citizen import (
    find_suspicious_bursts,
    parse_timestamp
)


class TestCitizenBursts(unittest.TestCase):

    def test_timestamp_parsing(self):
        result = parse_timestamp(
            "2026-11-17T10:30:00Z"
        )
        self.assertIsNotNone(result)

    def test_invalid_timestamp(self):
        self.assertIsNone(
            parse_timestamp("not-a-timestamp")
        )

    def test_suspicious_report_burst(self):
        reports = [
            {
                "zone": "silk_board",
                "timestamp": f"2026-11-17T10:0{i}:00Z",
                "verified": False
            }
            for i in range(5)
        ]

        result = find_suspicious_bursts(reports)

        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0]["zone"], "silk_board")

    def test_no_burst_when_reports_are_spread_out(self):
        reports = [
            {
                "zone": "silk_board",
                "timestamp": f"2026-11-17T10:{i * 10:02d}:00Z",
                "verified": False
            }
            for i in range(5)
        ]

        result = find_suspicious_bursts(reports)

        self.assertEqual(result, [])

    def test_verified_reports_do_not_form_burst(self):
        reports = [
            {
                "zone": "silk_board",
                "timestamp": f"2026-11-17T10:0{i}:00Z",
                "verified": True
            }
            for i in range(5)
        ]

        result = find_suspicious_bursts(reports)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
