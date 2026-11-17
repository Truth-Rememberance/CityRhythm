from backend.sync.mock_emitters import (
    ALARM_TICK,
    FIRST_CRITICAL_TICK,
    FIRST_DRIFTING_TICK,
    ZONES,
    mock_events,
)
from backend.sync.sync_index import load_bands, status

INDEX_FIELDS = {
    "zone",
    "sync_index",
    "status",
    "phase_coherence",
    "csd_score",
    "coupling_break",
    "phases",
}


def test_mock_events_are_deterministic():
    assert mock_events(2150) == mock_events(2150)
    assert mock_events(2150, fill_nulls=True) == mock_events(2150, fill_nulls=True)


def test_every_tick_has_sync_indexes_for_all_zones():
    for tick in (0, 1, 1999, 2000, FIRST_DRIFTING_TICK, FIRST_CRITICAL_TICK, 2600, ALARM_TICK):
        events = mock_events(tick)
        indexes = [event for event in events if event["channel"] == "sync.index"]
        assert len(indexes) == 8
        assert {event["zone"] for event in indexes} == set(ZONES)


def test_status_values_match_configured_bands():
    bands = load_bands()
    for tick in (0, 1999, 2000, FIRST_DRIFTING_TICK, FIRST_CRITICAL_TICK, 2600, 2700):
        for event in mock_events(tick):
            if event["channel"] == "sync.index":
                payload = event["payload"]
                assert payload["status"] == status(payload["sync_index"], bands)


def test_warning_story_has_one_drifting_warning_before_baseline_alarm():
    warnings = []
    baseline_alarms = []
    for tick in range(ALARM_TICK + 1):
        for event in mock_events(tick):
            if event["zone"] != "silk_board":
                continue
            if event["channel"] == "sync.warning":
                warnings.append((tick, event["payload"]))
            elif event["channel"] == "baseline.alarm":
                baseline_alarms.append((tick, event["payload"]))

    drifting = [(tick, payload) for tick, payload in warnings if payload["level"] == "drifting" and not payload["baseline_fired"]]
    critical = [(tick, payload) for tick, payload in warnings if payload["level"] == "critical" and not payload["baseline_fired"]]
    assert len(drifting) == 1
    assert len(critical) == 1
    assert len(baseline_alarms) == 1
    assert drifting[0][0] == FIRST_DRIFTING_TICK
    assert critical[0][0] == FIRST_CRITICAL_TICK
    assert drifting[0][0] < baseline_alarms[0][0]
    assert baseline_alarms[0][0] == ALARM_TICK
    update = [payload for tick, payload in warnings if tick == ALARM_TICK and payload["baseline_fired"]]
    assert len(update) == 1
    assert update[0]["lead_min"] == ALARM_TICK - FIRST_DRIFTING_TICK


def test_sync_index_fields_and_null_fill_modes():
    null_events = [event for event in mock_events(2000) if event["channel"] == "sync.index"]
    filled_events = [event for event in mock_events(2000, fill_nulls=True) if event["channel"] == "sync.index"]
    assert len(null_events) == len(filled_events) == 8
    for event in null_events:
        payload = event["payload"]
        assert set(payload) == INDEX_FIELDS
        assert payload["phase_coherence"] is None
        assert payload["coupling_break"] is None
        assert payload["phases"] is None
    for event in filled_events:
        payload = event["payload"]
        assert set(payload) == INDEX_FIELDS
        assert 0.0 <= payload["phase_coherence"] <= 1.0
        assert 0.0 <= payload["coupling_break"] <= 1.0
        assert set(payload["phases"]) == {"traffic", "power", "water"}
        assert all(0.0 <= angle <= 6.283185307179586 for angle in payload["phases"].values())


def test_graph_is_emitted_every_minute_divisible_by_sixty():
    assert any(event["channel"] == "sync.graph" for event in mock_events(0))
    assert any(event["channel"] == "sync.graph" for event in mock_events(60))
    assert not any(event["channel"] == "sync.graph" for event in mock_events(59))
    graph = next(event for event in mock_events(2580) if event["channel"] == "sync.graph")
    assert graph["payload"]["nodes"] == list(ZONES)
    assert len(graph["payload"]["edges"]) == 7
    assert all(edge["source"] == "silk_board" for edge in graph["payload"]["edges"])
