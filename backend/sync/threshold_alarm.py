"""R3: naive threshold alarm. This is the baseline the Sync Index has to beat."""

THRESHOLD = 0.9
SUSTAINED_MIN = 5


class ThresholdAlarm:
    """Fires once per node when its value stays above the threshold for
    sustained_min minutes in a row. Re-arms after the value drops back."""

    def __init__(self, threshold=THRESHOLD, sustained_min=SUSTAINED_MIN):
        self.threshold = threshold
        self.sustained_min = sustained_min
        self._run = {}
        self._fired = set()

    def update(self, readings):
        """readings: the sim.tick readings list for ONE tick (one per node).
        Returns a list of baseline.alarm payload dicts for alarms that fire now."""
        events = []
        for r in readings:
            node = f'{r["zone"]}.{r["system"]}'
            if r["value"] > self.threshold:
                self._run[node] = self._run.get(node, 0) + 1
                if self._run[node] >= self.sustained_min and node not in self._fired:
                    self._fired.add(node)
                    events.append({
                        "zone": r["zone"],
                        "system": r["system"],
                        "metric": r["metric"],
                        "value": r["value"],
                        "threshold": self.threshold,
                        "sustained_min": self._run[node],
                    })
            else:
                self._run[node] = 0
                self._fired.discard(node)
        return events
