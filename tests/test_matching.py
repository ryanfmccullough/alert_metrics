import numpy as np
import pandas as pd
import pytest

from alert_metrics import (
    alert_ppv,
    event_sensitivity,
    false_alerts,
    match_alerts_to_events,
    median_warning_time,
    warning_times,
)


def test_hand_case(small_cohort):
    alerts, events, observation = small_cohort
    m = match_alerts_to_events(alerts, events, window=24.0, observation=observation)
    assert event_sensitivity(m) == 1.0
    assert alert_ppv(m) == pytest.approx(3 / 7)
    assert median_warning_time(m) == pytest.approx(20.0)
    ev = m.events.iloc[0]
    assert ev["detected"] and ev["first_alert"] == 30.0 and ev["n_alerts_in_window"] == 3
    fa = false_alerts(m)
    assert sorted(fa["time"].tolist()) == [5.0, 10.0, 20.0, 60.0]


def test_alert_at_event_time_is_not_a_warning():
    alerts = pd.DataFrame({"subject": ["A", "A"], "time": [50.0, 26.0]})
    events = pd.DataFrame({"subject": ["A"], "time": [50.0]})
    m = match_alerts_to_events(alerts, events, window=24.0)
    # 26 h is exactly 24 h before: inside the closed-left window; 50 h is not.
    assert m.events.iloc[0]["detected"]
    assert m.events.iloc[0]["warning_time"] == 24.0
    assert m.alerts.set_index("time").loc[50.0, "true_alert"] == False  # noqa: E712
    assert m.alerts.set_index("time").loc[26.0, "true_alert"] == True  # noqa: E712


def test_alert_before_window_opens_is_false():
    alerts = pd.DataFrame({"subject": ["A"], "time": [20.0]})
    events = pd.DataFrame({"subject": ["A"], "time": [50.0]})
    m = match_alerts_to_events(alerts, events, window=24.0)
    assert not m.events.iloc[0]["detected"]
    assert alert_ppv(m) == 0.0
    assert np.isnan(median_warning_time(m))
    assert median_warning_time(m, undetected="zero") == 0.0


def test_two_close_events_attribute_alert_to_nearest_upcoming():
    alerts = pd.DataFrame({"subject": ["A"], "time": [40.0]})
    events = pd.DataFrame({"subject": ["A", "A"], "time": [50.0, 60.0]})
    m = match_alerts_to_events(alerts, events, window=24.0)
    assert m.alerts.iloc[0]["event_time"] == 50.0
    # both events count the alert as inside their window
    assert m.events["detected"].tolist() == [True, True]
    assert m.events["warning_time"].tolist() == [10.0, 20.0]


def test_undetected_zero_option(small_cohort):
    alerts, events, observation = small_cohort
    events2 = pd.concat([events, pd.DataFrame({"subject": ["B"], "time": [47.0]})])
    m = match_alerts_to_events(alerts, events2, window=24.0, observation=observation)
    # B's alert at 20 h is 27 h before the event: outside the window.
    assert event_sensitivity(m) == 0.5
    assert warning_times(m, "exclude").tolist() == [20.0]
    assert sorted(warning_times(m, "zero").tolist()) == [0.0, 20.0]


def test_observation_filters_alerts(small_cohort):
    alerts, events, observation = small_cohort
    obs = observation.copy()
    obs.loc[obs["subject"] == "A", "end"] = 35.0  # drop alerts at 40, 49.5, 60
    m = match_alerts_to_events(alerts, events, window=24.0, observation=obs)
    assert len(m.alerts) == 4
    assert m.events.iloc[0]["n_alerts_in_window"] == 1
    assert median_warning_time(m) == 20.0


def test_subject_without_alerts_is_undetected():
    alerts = pd.DataFrame({"subject": ["B"], "time": [1.0]})
    events = pd.DataFrame({"subject": ["A"], "time": [50.0]})
    m = match_alerts_to_events(alerts, events, window=24.0)
    assert event_sensitivity(m) == 0.0


def test_bad_window():
    alerts = pd.DataFrame({"subject": ["A"], "time": [1.0]})
    events = pd.DataFrame({"subject": ["A"], "time": [5.0]})
    with pytest.raises(ValueError):
        match_alerts_to_events(alerts, events, window=0.0)
