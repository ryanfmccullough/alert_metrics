import numpy as np
import pandas as pd

from alert_metrics import evaluate_alerts, match_alerts_to_events, observed_hours
from alert_metrics.bootstrap import (
    METRICS,
    bootstrap_evaluate,
    paired_difference,
    paired_resamples,
    percentile_interval,
    resample_metrics,
    resample_weights,
    subject_summary,
    weighted_median_rows,
)


def cohort(seed=0, n=60):
    rng = np.random.default_rng(seed)
    subjects = [f"s{i:03d}" for i in range(n)]
    observation = pd.DataFrame({"subject": subjects, "start": 0.0, "end": 500.0})
    ev_rows, al_rows = [], []
    for i, s in enumerate(subjects):
        if i % 3 == 0:  # an event, detected with two alerts in the window for most subjects
            e = float(rng.uniform(100, 450))
            ev_rows.append((s, e))
            if i % 6 == 0:
                al_rows += [(s, e - 10.0), (s, e - 3.0)]
        for _ in range(int(rng.integers(0, 4))):  # false alerts anywhere
            al_rows.append((s, float(rng.uniform(0, 60))))
    events = pd.DataFrame(ev_rows, columns=["subject", "time"])
    alerts = pd.DataFrame(al_rows, columns=["subject", "time"])
    return alerts, events, observation


def test_point_estimates_match_evaluate_alerts():
    alerts, events, observation = cohort()
    ref = evaluate_alerts(alerts, events, observation, 24.0)
    m = match_alerts_to_events(alerts, events, 24.0, observation)
    s = subject_summary(m, observation)
    pe = s.point_estimates()
    for k in METRICS:
        assert np.isclose(pe[k], ref[k]), (k, pe[k], ref[k])
    assert np.isclose(s.observed_hours.sum(), observed_hours(observation))
    assert len(s.subjects) == 60  # subjects without alerts or events still count


def test_weighted_median_matches_numpy_on_expanded_multisets():
    rng = np.random.default_rng(1)
    values = np.sort(rng.normal(size=15))
    weights = rng.integers(0, 4, size=(200, 15))
    weights[0] = 0  # empty multiset
    med = weighted_median_rows(values, weights)
    assert np.isnan(med[0])
    for r in range(1, 200):
        expanded = np.repeat(values, weights[r])
        if expanded.size:
            assert np.isclose(med[r], np.median(expanded))
        else:
            assert np.isnan(med[r])


def test_resample_weights_are_multiplicities():
    w = resample_weights(10, 50, seed=3)
    assert w.shape == (50, 10) and (w.sum(axis=1) == 10).all() and (w >= 0).all()
    assert np.array_equal(w, resample_weights(10, 50, seed=3))


def test_bootstrap_intervals_bracket_point_estimates_and_are_reproducible():
    alerts, events, observation = cohort()
    out = bootstrap_evaluate(alerts, events, observation, 24.0, n_resamples=300, seed=0)
    for k in METRICS:
        assert out[f"{k}_lo"] <= out[k] <= out[f"{k}_hi"], k
    assert out["alerts_per_100_days_hi"] > out["alerts_per_100_days_lo"]
    again = bootstrap_evaluate(alerts, events, observation, 24.0, n_resamples=300, seed=0)
    assert out == again
    m = match_alerts_to_events(alerts, events, 24.0, observation)
    res = resample_metrics(subject_summary(m, observation), 300, seed=0)
    assert list(res.columns) == list(METRICS) and len(res) == 300


def test_percentile_interval_ignores_nan():
    lo, hi = percentile_interval(np.array([np.nan, 1.0, 2.0, 3.0, 4.0]), 0.5)
    assert lo == 1.75 and hi == 3.25
    assert all(np.isnan(x) for x in percentile_interval(np.array([np.nan])))


def test_paired_difference_is_zero_for_identical_streams_and_signed_for_more_alerts():
    alerts, events, observation = cohort()
    m = match_alerts_to_events(alerts, events, 24.0, observation)
    s = subject_summary(m, observation)
    d = paired_difference(s, s, n_resamples=100)
    for k in METRICS:
        assert d[f"{k}_diff"] == 0.0 and d[f"{k}_diff_lo"] == 0.0 and d[f"{k}_diff_hi"] == 0.0
    # one extra false alert for every subject: burden rises for every resample
    extra = pd.DataFrame({"subject": observation["subject"], "time": 70.0})
    m2 = match_alerts_to_events(pd.concat([alerts, extra]), events, 24.0, observation)
    s2 = subject_summary(m2, observation)
    d2 = paired_difference(s, s2, n_resamples=100)
    assert d2["alerts_per_100_days_diff"] > 0 and d2["alerts_per_100_days_diff_lo"] > 0
    assert d2["sensitivity_diff"] == 0.0
    pr = paired_resamples({"a": s, "b": s2}, n_resamples=20)
    assert pr.shape == (20, 2 * len(METRICS))
