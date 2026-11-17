
import hashlib
import json
import math
from collections import Counter


# ============================================================
# 1. PHYSICS CONSISTENCY
# ============================================================

def physics_consistency(readings):
    """
    Validate normalized sensor readings.

    Expected sensor values: 0 to 1.
    Returns a list of anomaly evidence.
    """

    evidence = []

    for reading in readings:
        sensor_id = reading.get("sensor_id", "unknown")
        value = reading.get("value")

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            evidence.append({
                "kind": "invalid_reading",
                "detail": f"Invalid numeric reading from {sensor_id}",
                "score": 0.9
            })
            continue

        if not 0 <= value <= 1:
            evidence.append({
                "kind": "out_of_range",
                "detail": (
                    f"Sensor {sensor_id} reported "
                    f"out-of-range value {value}"
                ),
                "score": 0.9
            })

    return evidence


# ============================================================
# 2. REPLAY ATTACK DETECTION
# ============================================================

def replay_detector(history, repeat_limit=3, max_period=5):
    """
    Detect suspicious replay patterns using:

    1. Consecutive identical tick hashes.
    2. Repeated-segment hashes.
    3. Numeric autocorrelation.

    history = [
        [reading1, reading2],
        [reading1, reading2],
        ...
    ]
    """

    evidence = []

    if len(history) < repeat_limit:
        return evidence

    def fingerprint(tick):
        canonical = []

        for reading in tick:
            value = reading.get("value")

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                continue

            canonical.append({
                "sensor_id": reading.get("sensor_id"),
                "value": round(float(value), 6)
            })

        canonical.sort(key=lambda item: str(item["sensor_id"]))

        raw = json.dumps(canonical, sort_keys=True)

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    hashes = [fingerprint(tick) for tick in history]

    # --------------------------------------------------------
    # A. Consecutive identical tick detection
    # --------------------------------------------------------

    if hashes:
        repeated_count = 1

        for index in range(len(hashes) - 1, 0, -1):
            if hashes[index] == hashes[index - 1]:
                repeated_count += 1
            else:
                break

        if repeated_count >= repeat_limit:
            evidence.append({
                "kind": "possible_replay",
                "detail": (
                    f"{repeated_count} consecutive identical "
                    "sensor ticks detected"
                ),
                "score": 0.9
            })

    # --------------------------------------------------------
    # B. Repeated-segment hash detection
    # --------------------------------------------------------

    segment_found = False

    for period in range(1, min(max_period, len(hashes) // 2) + 1):

        first_segment = hashes[-2 * period:-period]
        second_segment = hashes[-period:]

        if (
            len(first_segment) == period
            and first_segment == second_segment
        ):
            evidence.append({
                "kind": "repeated_segment",
                "detail": (
                    f"Repeated sensor sequence detected "
                    f"with period {period} ticks"
                ),
                "score": 0.9
            })

            segment_found = True
            break

    # --------------------------------------------------------
    # C. Numeric autocorrelation
    # --------------------------------------------------------

    values = []

    for tick in history:

        valid_values = [
            float(reading["value"])
            for reading in tick
            if isinstance(reading.get("value"), (int, float))
            and not isinstance(reading.get("value"), bool)
            and math.isfinite(float(reading["value"]))
        ]

        if valid_values:
            values.append(sum(valid_values) / len(valid_values))
        else:
            values.append(None)

    for period in range(1, min(max_period, len(values) // 2) + 1):

        pairs = [
            (values[i], values[i - period])
            for i in range(period, len(values))
            if values[i] is not None
            and values[i - period] is not None
        ]

        if len(pairs) < 4:
            continue

        current = [pair[0] for pair in pairs]
        previous = [pair[1] for pair in pairs]

        mean_current = sum(current) / len(current)
        mean_previous = sum(previous) / len(previous)

        numerator = sum(
            (a - mean_current) * (b - mean_previous)
            for a, b in pairs
        )

        variance_current = sum(
            (a - mean_current) ** 2 for a in current
        )

        variance_previous = sum(
            (b - mean_previous) ** 2 for b in previous
        )

        denominator = math.sqrt(
            variance_current * variance_previous
        )

        if denominator == 0:
            continue

        correlation = numerator / denominator

        if correlation >= 0.95:

            # Avoid duplicate evidence for the same exact replay.
            if not segment_found:
                evidence.append({
                    "kind": "periodic_autocorrelation",
                    "detail": (
                        f"Strong periodic correlation at "
                        f"lag {period}: {correlation:.3f}"
                    ),
                    "score": round(correlation, 3)
                })

            break

    return evidence


# ============================================================
# 3. SENSOR TRUST
# ============================================================

def sensor_trust(history):
    """
    Calculate a prototype trust score for each sensor.

    Trust considers:
    - Valid readings.
    - Repeated identical readings.

    This is a heuristic score, not a calibrated probability.
    """

    sensor_values = {}

    for tick in history:

        for reading in tick:

            sensor_id = reading.get("sensor_id")

            if not sensor_id:
                continue

            sensor_values.setdefault(sensor_id, []).append(
                reading.get("value")
            )

    trust_scores = {}

    for sensor_id, values in sensor_values.items():

        valid_values = [
            value for value in values
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        ]

        if not values:
            trust_scores[sensor_id] = 0.0
            continue

        valid_ratio = len(valid_values) / len(values)

        if valid_values:
            counts = Counter(valid_values)

            most_common_count = counts.most_common(1)[0][1]

            repeat_ratio = most_common_count / len(valid_values)
        else:
            repeat_ratio = 1.0

        trust = (
            valid_ratio * 0.7
            + (1 - repeat_ratio) * 0.3
        )

        trust_scores[sensor_id] = round(
            max(0.0, min(1.0, trust)), 3
        )

    return trust_scores


# ============================================================
# 4. THREAT CLASSIFICATION
# ============================================================


def classify(physics_evidence, replay_evidence, context=None):
    """
    Classify a zone as natural, attack or unknown.

    Natural:
        No attack indicators and supporting simulation context.

    Attack:
        Physics anomaly and replay evidence together.

    Unknown:
        Insufficient or conflicting evidence.
    """

    context = context or {}

    evidence = list(physics_evidence) + list(replay_evidence)

    has_physics = bool(physics_evidence)
    has_replay = bool(replay_evidence)

    # Strong combined indicators
    if has_physics and has_replay:
        return {
            "label": "attack",
            "confidence": 0.95,
            "entry_node": None,
            "evidence": evidence
        }

    # One indicator is not enough to confirm an attack
    if has_physics or has_replay:
        return {
            "label": "unknown",
            "confidence": 0.60,
            "entry_node": None,
            "evidence": evidence
        }

    # Check known contextual events from simulation
    events = context.get("events", [])

    if events:
        evidence.append({
            "kind": "contextual_event",
            "detail": "Simulation reports a known city event",
            "score": 0.8
        })

        return {
            "label": "natural",
            "confidence": 0.80,
            "entry_node": None,
            "evidence": evidence
        }

    return {
        "label": "unknown",
        "confidence": 0.0,
        "entry_node": None,
        "evidence": evidence
    }



# ============================================================
# 5. QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":

    sample_reading = {
        "sensor_id": "silk_board.traffic.01",
        "value": 0.65,
        "zone": "silk_board",
        "system": "traffic"
    }

    print("\n--- PHYSICS CHECK ---")

    physics = physics_consistency([sample_reading])
    print(physics)

    print("\n--- REPLAY CHECK ---")

    history = [
        [sample_reading],
        [sample_reading],
        [sample_reading]
    ]

    replay = replay_detector(history)
    print(replay)

    print("\n--- SENSOR TRUST ---")

    trust = sensor_trust(history)
    print(trust)

    print("\n--- THREAT CLASSIFICATION ---")

    assessment = classify(
        [{"kind": "out_of_range",
          "detail": "Test anomaly",
          "score": 0.9}],
        replay
    )

    print(assessment)
