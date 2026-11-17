from backend.sync.threshold_alarm import ThresholdAlarm


def _reading(value, zone="silk_board", system="traffic", metric="flow"):
    return {"zone": zone, "system": system, "metric": metric, "value": value}


def test_no_alarm_below_threshold():
    alarm = ThresholdAlarm(threshold=0.9, sustained_min=5)
    for _ in range(50):
        assert alarm.update([_reading(0.8)]) == []


def test_fires_exactly_on_the_sustained_minute():
    alarm = ThresholdAlarm(threshold=0.9, sustained_min=5)
    fired_at = [i for i in range(10) if alarm.update([_reading(0.95)])]
    assert fired_at == [4]


def test_fires_only_once_while_above():
    alarm = ThresholdAlarm(threshold=0.9, sustained_min=3)
    total = sum(len(alarm.update([_reading(0.95)])) for _ in range(30))
    assert total == 1


def test_short_spike_does_not_fire_and_alarm_rearms():
    alarm = ThresholdAlarm(threshold=0.9, sustained_min=3)
    for v in (0.95, 0.95, 0.5):
        assert alarm.update([_reading(v)]) == []
    fired = [bool(alarm.update([_reading(0.95)])) for _ in range(3)]
    assert fired == [False, False, True]


def test_payload_fields():
    alarm = ThresholdAlarm(threshold=0.9, sustained_min=1)
    (event,) = alarm.update([_reading(0.97)])
    assert set(event) == {"zone", "system", "metric", "value", "threshold", "sustained_min"}
    assert event["zone"] == "silk_board" and event["threshold"] == 0.9
