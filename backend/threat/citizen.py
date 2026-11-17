
import re
from collections import Counter
from datetime import datetime, timezone


KEYWORDS = {
    "traffic": [
        "traffic", "jam", "congestion", "ರಸ್ತೆ", "ಟ್ರಾಫಿಕ್",
        "jam agide", "traffic jam"
    ],
    "power": [
        "power", "electricity", "current", "outage",
        "ವಿದ್ಯುತ್", "current illa"
    ],
    "water": [
        "water", "supply", "leak", "ಪानी", "ನೀರು",
        "water illa"
    ]
}


def detect_language(text):
    """Identify a likely report language."""

    if not isinstance(text, str):
        return "unknown"

    if re.search(r"[\u0C80-\u0CFF]", text):
        return "Kannada"

    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"

    return "English/Hinglish"


def extract_report(text, llm=None):
    """
    Extract structured information from a citizen complaint.
    Uses llm.complete when available; otherwise uses keyword fallback.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "category": "unknown",
            "language": "unknown",
            "keywords": [],
            "extraction_method": "fallback"
        }

    language = detect_language(text)
    cleaned = text.lower()

    # Optional LLM extraction.
    if llm is not None:
        try:
            result = llm.complete(
                "Extract the infrastructure category from this "
                "citizen complaint. Return JSON with category, "
                "language and keywords. Categories: traffic, "
                "power, water, unknown.\nComplaint: " + text
            )

            if isinstance(result, dict):
                category = result.get("category", "unknown")

                if category in ("traffic", "power", "water", "unknown"):
                    return {
                        "category": category,
                        "language": result.get("language", language),
                        "keywords": result.get("keywords", []),
                        "extraction_method": "llm"
                    }

        except Exception:
            pass

    # Keyword fallback.
    matched = []

    for category, words in KEYWORDS.items():
        for word in words:
            if word in cleaned:
                matched.append((category, word))

    categories = [item[0] for item in matched]

    category = (
        Counter(categories).most_common(1)[0][0]
        if categories else "unknown"
    )

    return {
        "category": category,
        "language": language,
        "keywords": [item[1] for item in matched],
        "extraction_method": "fallback"
    }


def parse_timestamp(value):
    """
    Convert supported timestamps into UTC datetime.

    Supports:
    - Unix timestamps
    - ISO 8601 strings
    - Datetime objects

    Returns None when timestamp is unavailable or invalid.
    """

    if isinstance(value, datetime):
        result = value

    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            result = datetime.fromtimestamp(value, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return None

    else:
        return None

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)

    return result.astimezone(timezone.utc)


def find_suspicious_bursts(
    reports,
    window_minutes=5,
    min_reports=5,
    unverified_ratio=0.8
):
    """
    Find groups of unverified reports arriving within a time window.

    Only reports with valid timestamps are considered.

    A suspicious burst is an alert for investigation,
    not proof of a fake campaign.
    """

    if window_minutes <= 0 or min_reports < 1:
        raise ValueError("Invalid burst detection settings")

    window_seconds = window_minutes * 60

    by_zone = {}

    for report in reports:
        timestamp = parse_timestamp(
            report.get("timestamp")
            or report.get("ts")
            or report.get("created_at")
        )

        if timestamp is None:
            continue

        zone = report.get("zone")

        if not zone:
            continue

        by_zone.setdefault(zone, []).append(
            (timestamp, report)
        )

    bursts = []

    for zone, entries in by_zone.items():

        entries.sort(key=lambda item: item[0])

        # Sliding window over timestamped reports.
        left = 0

        for right in range(len(entries)):

            while (
                entries[right][0] - entries[left][0]
            ).total_seconds() > window_seconds:
                left += 1

            current = entries[left:right + 1]

            if len(current) < min_reports:
                continue

            unverified = [
                report for _, report in current
                if report.get("verified") is False
            ]

            ratio = len(unverified) / len(current)

            if ratio >= unverified_ratio:

                burst = {
                    "zone": zone,
                    "report_count": len(current),
                    "unverified_count": len(unverified),
                    "unverified_ratio": round(ratio, 3),
                    "window_minutes": window_minutes,
                    "start_time": entries[left][0].isoformat(),
                    "end_time": entries[right][0].isoformat(),
                    "suspicious": True
                }

                # Avoid reporting duplicate overlapping windows
                # with the same start and end.
                if not bursts or not any(
                    item["zone"] == zone
                    and item["start_time"] == burst["start_time"]
                    and item["end_time"] == burst["end_time"]
                    for item in bursts
                ):
                    bursts.append(burst)

    return bursts


def report_sensor_agreement(reports, readings):
    """
    Compare citizen complaints with sensor readings.

    Reports without matching sensor evidence are marked unverified.

    Also identifies suspicious timestamped report bursts.
    """

    results = []

    zones = set(
        report.get("zone")
        for report in reports
        if report.get("zone")
    )

    # Track individual report verification for burst analysis.
    processed_reports = []

    for zone in zones:

        zone_reports = [
            report for report in reports
            if report.get("zone") == zone
        ]

        zone_readings = [
            reading for reading in readings
            if reading.get("zone") == zone
        ]

        verified = 0
        unverified = 0

        for report in zone_reports:

            extracted = extract_report(
                report.get("text", "")
            )

            category = extracted["category"]

            matching_reading = any(
                reading.get("system") == category
                and isinstance(
                    reading.get("value"), (int, float)
                )
                and not isinstance(reading.get("value"), bool)
                and 0 <= reading["value"] <= 1
                and reading["value"] >= 0.7
                for reading in zone_readings
            )

            if matching_reading:
                verified += 1
            else:
                unverified += 1

            processed = dict(report)
            processed["verified"] = matching_reading
            processed_reports.append(processed)

        total = len(zone_reports)

        # Preserve the original aggregate burst check
        # for backward compatibility.
        suspicious_burst = (
            total >= 5
            and unverified / total >= 0.8
        )

        results.append({
            "zone": zone,
            "total_reports": total,
            "verified": verified,
            "unverified": unverified,
            "suspicious_burst": suspicious_burst
        })

    # New timestamp-aware detection.
    timed_bursts = find_suspicious_bursts(
        processed_reports,
        window_minutes=5,
        min_reports=5,
        unverified_ratio=0.8
    )

    for result in results:
        result["timed_suspicious_bursts"] = [
            burst for burst in timed_bursts
            if burst["zone"] == result["zone"]
        ]

    return results
