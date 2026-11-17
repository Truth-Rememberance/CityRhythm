
import unittest

from threat.redteam import redteam


class TestRedTeam(unittest.TestCase):

    def test_attack_path(self):

        graph = {
            "edges": [
                {
                    "src": "silk_board.traffic",
                    "dst": "silk_board.power",
                    "lag_min": 5,
                    "strength": 0.9
                },
                {
                    "src": "silk_board.power",
                    "dst": "silk_board.water",
                    "lag_min": 8,
                    "strength": 0.8
                }
            ]
        }

        results = redteam(graph)

        self.assertTrue(len(results) > 0)
        self.assertEqual(
            results[0]["entry_node"],
            "silk_board.traffic"
        )


if __name__ == "__main__":
    unittest.main()

