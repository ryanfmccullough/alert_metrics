import numpy as np
import pandas as pd
import pytest

from alert_metrics import (
    alerts_per_100_days,
    alerts_per_subject_day,
    count_alerts,
    observed_days,
    observed_hours,
)


def test_observed_hours_merges_overlaps():
    obs = pd.DataFrame(
        {
            "subject": ["A", "A", "A", "B"],
            "start": [0.0, 10.0, 30.0, 0.0],
            "end": [20.0, 15.0, 40.0, 24.0],
        }
    )
    # A: [0,20] ∪ [10,15] ∪ [30,40] = 20 + 10 = 30 h; B: 24 h
    assert observed_hours(obs) == pytest.approx(54.0)
    assert observed_days(obs) == pytest.approx(54.0 / 24.0)


def test_alerts_per_100_days_hand_case(small_cohort):
    alerts, _, observation = small_cohort
    # 7 alerts over 148 h = 6.1667 days
    expected = 100.0 * 7 / (148.0 / 24.0)
    assert alerts_per_100_days(alerts, observation) == pytest.approx(expected)


def test_alerts_outside_observation_do_not_count(small_cohort):
    alerts, _, observation = small_cohort
    extra = pd.concat(
        [alerts, pd.DataFrame({"subject": ["B", "C"], "time": [60.0, 1.0]})], ignore_index=True
    )
    assert count_alerts(extra, observation) == 7
    assert count_alerts(extra) == 9


def test_per_subject_rate(small_cohort):
    alerts, _, observation = small_cohort
    rate = alerts_per_subject_day(alerts, observation)
    assert rate["A"] == pytest.approx(5 / (100 / 24))
    assert rate["B"] == pytest.approx(2 / (48 / 24))


def test_bad_intervals_raise():
    obs = pd.DataFrame({"subject": ["A"], "start": [5.0], "end": [1.0]})
    with pytest.raises(ValueError):
        observed_hours(obs)


def test_empty_observation_gives_nan():
    alerts = pd.DataFrame({"subject": ["A"], "time": [1.0]})
    obs = pd.DataFrame({"subject": ["A"], "start": [0.0], "end": [0.0]})
    assert np.isnan(alerts_per_100_days(alerts, obs))
