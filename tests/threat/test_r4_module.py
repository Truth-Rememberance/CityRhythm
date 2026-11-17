
import unittest

from threat.module import ThreatModule


class FakeContext:

    def latest(self, channel, zone=None):
        if channel == "sync.graph":
            return {
                "v": 1,
                "channel": "sync.graph",
                "zone": None,
                "payload": {
                    "edges": [
                        {
                            "src": "silk_board.traffic",
                            "dst": "silk_board.power",
                            "lag_min": 5,
                            "strength": 0.9
                        }
                    ]
                }
            }

        return None


class TestThreatModule(unittest.TestCase):

    def test_simulation_tick(self):

        module = ThreatModule()

        tick = {
            "v": 1,
            "channel": "sim.tick",
            "run_id": "r_calm_1001",
            "t": 4140,
            "zone": None,
            "payload": {
                "readings": [
                    {
                        "zone": "silk_board",
                        "system": "traffic",
                        "metric": "flow",
                        "value": 0.71,
                        "sensor_id": "silk_board.traffic.01"
                    },
                    {
                        "zone": "silk_board",
                        "system": "power",
                        "metric": "load",
                        "value": 0.64,
                        "sensor_id": "silk_board.power.01"
                    }
                ],
                "context": {
                    "temp_c": 27.4,
                    "events": [],
                    "is_holiday": False
                },
                "reports": []
            }
        }

        events = module.on_tick(tick, FakeContext())

        channels = [event["channel"] for event in events]

        self.assertIn("threat.assessment", channels)
        self.assertIn("threat.trust", channels)

        print("\nR4 EVENTS:")
        print(events)


if __name__ == "__main__":
    unittest.main()
