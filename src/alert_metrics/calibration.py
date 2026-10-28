"""Calibration summaries for probabilistic scores.

Everything operates on flat NumPy arrays of predicted probabilities and binary
labels so it can be tested without a dataset.

Definitions
-----------
reliability bins
    Equal-width bins on ``[0, 1]``; per bin the mean predicted probability
    (confidence) and the observed positive rate. The reliability diagram
    plots one against the other (DeGroot & Fienberg 1983).
expected calibration error (ECE)
    Count-weighted mean absolute gap between confidence and observed rate
    across bins (Naeini, Cooper & Hauskrecht 2015). Depends on the binning;
    the bin count is an argument and should be reported.
Brier score
    Mean squared error between probability and label (Brier 1950). Its
    decomposition into reliability, resolution and uncertainty (Murphy 1973)
    is returned by :func:`brier_decomposition`.
calibration slope and intercept
    Logistic regression of the label on the logit of the predicted
    probability (Cox 1958; Miller, Hui & Tierney 1991). Slope 1 and intercept
    0 mean calibration-in-the-large and spread are both right; slope below 1
    indicates overfitting (predictions too extreme).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBins:
    """Per-bin statistics behind a reliability diagram."""

    edges: np.ndarray  # (n_bins + 1,)
    confidence: np.ndarray  # mean predicted probability per bin, NaN where empty
    observed: np.ndarray  # observed positive rate per bin, NaN where empty
    count: np.ndarray  # samples per bin


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError("probs and labels must have the same shape")
    if p.size == 0:
        raise ValueError("empty input")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probs must lie in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    return p, y


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> ReliabilityBins:
    """Equal-width reliability bins."""
    p, y = _check(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    conf_sum = np.bincount(idx, weights=p, minlength=n_bins)
    obs_sum = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(count > 0, conf_sum / count, np.nan)
        observed = np.where(count > 0, obs_sum / count, np.nan)
    return ReliabilityBins(edges=edges, confidence=confidence, observed=observed, count=count)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins."""
    rb = reliability_bins(probs, labels, n_bins)
    m = rb.count > 0
    gap = np.abs(rb.confidence[m] - rb.observed[m])
    return float(np.sum(gap * rb.count[m]) / np.sum(rb.count[m]))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between probabilities and labels."""
    p, y = _check(probs, labels)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """Murphy (1973) decomposition: ``brier = reliability - resolution + uncertainty``.

    Computed on the same equal-width bins as :func:`reliability_bins`, so the
    identity holds up to the within-bin variance term, which is also returned
    (``brier - (reliability - resolution + uncertainty)``).
    """
    p, y = _check(probs, labels)
    rb = reliability_bins(p, y, n_bins)
    m = rb.count > 0
    n = p.size
    base = float(y.mean())
    reliability = float(np.sum(rb.count[m] * (rb.confidence[m] - rb.observed[m]) ** 2) / n)
    resolution = float(np.sum(rb.count[m] * (rb.observed[m] - base) ** 2) / n)
    uncertainty = base * (1 - base)
    brier = brier_score(p, y)
    return {
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "within_bin_variance": brier - (reliability - resolution + uncertainty),
    }


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibration_slope_intercept(
    probs: np.ndarray, labels: np.ndarray, max_iter: int = 100, tol: float = 1e-10
) -> dict[str, float]:
    """Cox calibration: fit ``logit P(y=1) = a + b * logit(p)`` by Newton's method.

    Returns ``{"intercept": a, "slope": b, "intercept_in_the_large": c}`` where
    ``c`` is the intercept of the model with slope fixed at 1 (calibration in
    the large). Labels must contain both classes.
    """
    p, y = _check(probs, labels)
    if y.min() == y.max():
        raise ValueError("labels must contain both classes")
    x = _logit(p)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        z = X @ beta
        mu = 1.0 / (1.0 + np.exp(-z))
        w = mu * (1 - mu)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X + 1e-12 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    # intercept with slope fixed at 1
    c = 0.0
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-(c + x)))
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu)) + 1e-12
        step = g / h
        c += step
        if abs(step) < tol:
            break
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_in_the_large": float(c),
    }


def calibration_summary(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """ECE, Brier (with decomposition), slope and intercept in one dictionary."""
    out: dict[str, float] = {"ece": expected_calibration_error(probs, labels, n_bins)}
    out.update(brier_decomposition(probs, labels, n_bins))
    y = np.asarray(labels).ravel()
    if y.min() != y.max():
        out.update(calibration_slope_intercept(probs, labels))
    out["n_bins"] = float(n_bins)
    return out


__all__ = [
    "ReliabilityBins",
    "brier_decomposition",
    "brier_score",
    "calibration_slope_intercept",
    "calibration_summary",
    "expected_calibration_error",
    "reliability_bins",
]
