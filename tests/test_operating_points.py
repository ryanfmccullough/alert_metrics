import numpy as np
import pytest

from alert_metrics import (
    evaluate_alerts,
    operating_point_table,
    ppv_at_sensitivity,
    threshold_at_alert_budget,
    threshold_at_sensitivity,
)


def test_evaluate_alerts_hand_case(small_cohort):
    alerts, events, observation = small_cohort
    r = evaluate_alerts(alerts, events, observation, window=24.0)
    assert r["n_alerts"] == 7 and r["n_events"] == 1
    assert r["sensitivity"] == 1.0
    assert r["ppv"] == pytest.approx(3 / 7)
    assert r["median_warning_time"] == 20.0
    assert r["alerts_per_100_days"] == pytest.approx(100 * 7 / (148 / 24))


def test_table_monotone_in_threshold(synthetic_scores):
    scores, events, observation = synthetic_scores
    tab = operating_point_table(scores, events, observation, window=24.0)
    assert (np.diff(tab["alerts_per_100_days"]) <= 1e-9).all()
    assert (np.diff(tab["sensitivity"]) <= 1e-9).all()
    assert tab.iloc[0]["sensitivity"] == 1.0


def test_refractory_reduces_burden_without_losing_first_warning(synthetic_scores):
    scores, events, observation = synthetic_scores
    grid = np.array([2.0])
    a = operating_point_table(scores, events, observation, window=24.0, thresholds=grid)
    b = operating_point_table(
        scores, events, observation, window=24.0, thresholds=grid, refractory=6.0
    )
    assert b["alerts_per_100_days"].iloc[0] < a["alerts_per_100_days"].iloc[0]
    assert b["sensitivity"].iloc[0] == a["sensitivity"].iloc[0]
    # the first alert of each event is unchanged by a refractory rule, so warning time is too
    assert b["median_warning_time"].iloc[0] == a["median_warning_time"].iloc[0]


def test_operating_point_selectors(synthetic_scores):
    scores, events, observation = synthetic_scores
    tab = operating_point_table(scores, events, observation, window=24.0)
    row = threshold_at_sensitivity(tab, 0.9)
    assert row["sensitivity"] >= 0.9
    assert (tab[tab["threshold"] > row["threshold"]]["sensitivity"] < 0.9).all()
    row2 = threshold_at_alert_budget(tab, 50.0)
    assert row2["alerts_per_100_days"] <= 50.0
    assert (tab[tab["threshold"] < row2["threshold"]]["alerts_per_100_days"] > 50.0).all()
    assert 0.0 <= ppv_at_sensitivity(tab, 0.9) <= 1.0


def test_rising_score_gives_warning_time_near_full_window(synthetic_scores):
    scores, events, observation = synthetic_scores
    tab = operating_point_table(
        scores, events, observation, window=24.0, thresholds=np.array([3.0])
    )
    # the ramp reaches 3.0 only in the last third of the 24 h before each event
    assert 0 < tab["median_warning_time"].iloc[0] < 24.0
    assert tab["sensitivity"].iloc[0] == 1.0


def test_unreachable_floor_raises(synthetic_scores):
    scores, events, observation = synthetic_scores
    tab = operating_point_table(
        scores, events, observation, window=24.0, thresholds=np.array([100.0])
    )
    with pytest.raises(ValueError):
        threshold_at_sensitivity(tab, 0.5)
