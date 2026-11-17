from backend.sync.fake_ticks import generate
from scripts.eval_leadtime import evaluate_envelopes


def _evaluate(threshold=1.0, sustained=15):
    return evaluate_envelopes(
        generate(2, 3, stress_from=2000),
        incident_tick=2000,
        baseline_threshold=threshold,
        baseline_sustained=sustained,
    )


def test_lead_time_is_deterministic():
    first = _evaluate()
    second = _evaluate()

    assert first == second


def test_lead_time_matches_reported_ticks():
    result = _evaluate()

    if result["first_warning_tick"] is not None and result["first_baseline_alarm_tick"] is not None:
        assert result["lead_min"] == (
            result["first_baseline_alarm_tick"] - result["first_warning_tick"]
        )


def test_first_warning_is_at_or_after_incident():
    result = _evaluate()

    assert result["first_warning_tick"] is not None
    assert result["first_warning_tick"] >= 2000


def test_stricter_baseline_alarms_later():
    # checks that baseline params are passed through; lead values on fake data are not meaningful because SyncModule uses the real-data baseline
    lower_baseline = _evaluate(threshold=0.9, sustained=5)
    default_baseline = _evaluate(threshold=1.0, sustained=15)
    stricter_baseline = _evaluate(threshold=1.1, sustained=15)

    assert lower_baseline["first_baseline_alarm_tick"] < default_baseline["first_baseline_alarm_tick"]
    assert default_baseline["first_baseline_alarm_tick"] < stricter_baseline["first_baseline_alarm_tick"]
