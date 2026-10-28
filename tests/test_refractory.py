import numpy as np
import pandas as pd
import pytest

from alert_metrics import apply_refractory, persistence_mask, refractory_mask, threshold_alerts
from alert_metrics.refractory import run_lengths


def test_refractory_mask_hand_case():
    times = np.array([0.0, 1.0, 2.0, 6.0, 6.5, 13.0])
    keep = refractory_mask(times, refractory=6.0)
    # keep 0; suppress (0, 6]; keep 6.5; suppress (6.5, 12.5]; keep 13
    assert keep.tolist() == [True, False, False, False, True, True]


def test_refractory_zero_keeps_everything():
    times = np.array([0.0, 0.0, 1.0])
    assert refractory_mask(times, 0.0).all()


def test_refractory_requires_sorted():
    with pytest.raises(ValueError):
        refractory_mask(np.array([2.0, 1.0]), 1.0)


def test_kept_alerts_are_at_least_refractory_apart():
    rng = np.random.default_rng(1)
    times = np.sort(rng.uniform(0, 1000, 5000))
    keep = refractory_mask(times, 3.0)
    kept = times[keep]
    assert np.all(np.diff(kept) > 3.0)
    # and nothing could have been kept in between: every dropped alert lies within 3 h of a kept one
    dropped = times[~keep]
    prev_kept = kept[np.searchsorted(kept, dropped, side="right") - 1]
    assert np.all(dropped - prev_kept <= 3.0)


def test_apply_refractory_per_subject():
    alerts = pd.DataFrame(
        {"subject": ["A", "A", "B", "B", "A"], "time": [0.0, 1.0, 0.0, 10.0, 7.0]}
    )
    out = apply_refractory(alerts, refractory=6.0)
    assert out.values.tolist() == [["A", 0.0], ["A", 7.0], ["B", 0.0], ["B", 10.0]]


def test_persistence_mask():
    above = np.array([True, True, False, True, True, True, False])
    assert persistence_mask(above, 1).tolist() == above.tolist()
    assert persistence_mask(above, 2).tolist() == [False, True, False, False, True, True, False]
    assert persistence_mask(above, 3).tolist() == [False, False, False, False, False, True, False]


def test_threshold_alerts_with_nan_and_persistence():
    scores = pd.DataFrame(
        {
            "subject": ["A"] * 6,
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "score": [0.9, np.nan, 0.9, 0.9, 0.1, 0.9],
        }
    )
    al = threshold_alerts(scores, threshold=0.5)
    assert al["time"].tolist() == [0.0, 2.0, 3.0, 5.0]
    al2 = threshold_alerts(scores, threshold=0.5, persistence=2)
    assert al2["time"].tolist() == [3.0]  # NaN breaks the run at 1.0
    al3 = threshold_alerts(scores, threshold=0.5, refractory=2.0)
    assert al3["time"].tolist() == [0.0, 3.0]  # 2.0 is inside (0, 2]


def test_run_lengths_matches_loop_and_resets_at_subject_boundaries():
    rng = np.random.default_rng(5)
    above = rng.random(5000) < 0.7
    ref = np.zeros(above.size, dtype=int)
    cur = 0
    for i, a in enumerate(above):
        cur = cur + 1 if a else 0
        ref[i] = cur
    assert np.array_equal(run_lengths(above), ref)
    first = np.zeros(above.size, dtype=bool)
    first[[0, 10, 11, 2500]] = True
    out = run_lengths(above, first)
    assert out[10] == int(above[10]) and out[11] == int(above[11]) and out[2500] == int(above[2500])
    assert np.array_equal(out[:10], ref[:10])


def test_threshold_alerts_persistence_does_not_cross_subjects():
    scores = pd.DataFrame(
        {
            "subject": ["A", "A", "A", "B", "B", "B"],
            "time": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "score": [0.9, 0.9, 0.9, 0.9, 0.1, 0.9],
        }
    )
    al = threshold_alerts(scores, 0.5, persistence=2)
    assert al.values.tolist() == [["A", 1.0], ["A", 2.0]]
    al3 = threshold_alerts(scores, 0.5, persistence=3)
    assert al3.values.tolist() == [["A", 2.0]]
