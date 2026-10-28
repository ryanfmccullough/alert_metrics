import numpy as np
import pytest

from alert_metrics import (
    brier_decomposition,
    brier_score,
    calibration_slope_intercept,
    calibration_summary,
    expected_calibration_error,
    reliability_bins,
)


def _calibrated(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_reliability_bins_hand_case():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 1])
    rb = reliability_bins(p, y, n_bins=10)
    assert rb.count.tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert rb.confidence[1] == pytest.approx(0.15)
    assert rb.observed[1] == pytest.approx(0.5)
    assert np.isnan(rb.confidence[2])


def test_ece_small_for_calibrated_scores():
    p, y = _calibrated()
    assert expected_calibration_error(p, y) < 0.02


def test_ece_large_for_miscalibrated_scores():
    p, y = _calibrated()
    assert expected_calibration_error(np.clip(p**3, 0, 1), y) > 0.1


def test_brier_and_decomposition_identity():
    p, y = _calibrated()
    d = brier_decomposition(p, y)
    assert d["brier"] == pytest.approx(brier_score(p, y))
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin_variance"]
    )
    assert d["reliability"] < 0.001


def test_slope_near_one_for_calibrated_and_below_one_for_overconfident():
    p, y = _calibrated()
    fit = calibration_slope_intercept(p, y)
    assert fit["slope"] == pytest.approx(1.0, abs=0.05)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.05)
    logit = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-2.0 * logit))  # predictions twice as extreme
    fit2 = calibration_slope_intercept(over, y)
    assert fit2["slope"] == pytest.approx(0.5, abs=0.05)


def test_summary_keys():
    p, y = _calibrated(2000)
    s = calibration_summary(p, y)
    for k in ("ece", "brier", "reliability", "resolution", "uncertainty", "slope", "intercept"):
        assert k in s


def test_validation():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.2]), np.array([1]))
    with pytest.raises(ValueError):
        brier_score(np.array([0.5]), np.array([2]))
    with pytest.raises(ValueError):
        calibration_slope_intercept(np.array([0.2, 0.4]), np.array([1, 1]))
