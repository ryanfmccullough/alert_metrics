"""Subject-level bootstrap intervals for the event-level metrics.

Alerts and events cluster within subjects: one patient with a long stay and
a noisy signal can contribute hundreds of alerts, and the events of one
patient share that patient's physiology. Resampling alerts or events one
at a time therefore understates uncertainty; the unit of resampling is the
subject (the cluster bootstrap, Davison & Hinkley 1997, section 3.8; Efron &
Tibshirani 1993 for the percentile interval).

Every metric reported by :func:`alert_metrics.evaluate_alerts` is a ratio of
per-subject sums (burden, sensitivity, PPV) or a quantile of the multiset of
per-event warning times. A resample of subjects with replacement therefore
needs only the per-subject summary and the integer multiplicities of the
draw: no re-matching, so a thousand resamples cost a fraction of a second.

Paired comparisons between two alert streams on the same subjects use the
same draws for both, which is what makes the interval on the difference
narrower than the two marginal intervals suggest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .burden import observed_hours
from .matching import Matched, match_alerts_to_events
from .streams import DEFAULT_COLUMNS, Columns, check_observation

METRICS = (
    "alerts_per_100_days",
    "sensitivity",
    "ppv",
    "median_warning_time",
    "median_warning_time_zero_undetected",
)


@dataclass
class SubjectSummary:
    """Per-subject sufficient statistics of one evaluation.

    Arrays are aligned with ``subjects``. ``warning_times`` holds, per
    subject, the warning times of that subject's detected events (empty for
    subjects without detected events).
    """

    subjects: np.ndarray
    observed_hours: np.ndarray
    n_alerts: np.ndarray
    n_true_alerts: np.ndarray
    n_events: np.ndarray
    n_detected: np.ndarray
    warning_times: list[np.ndarray]

    def point_estimates(self) -> dict[str, float]:
        """The metrics of the full sample, equal to :func:`evaluate_alerts`'s."""
        w = np.ones(len(self.subjects), dtype=int)
        return {k: float(v[0]) for k, v in _metrics_from_weights(self, w[None, :]).items()}

    def aligned_to(self, subjects: np.ndarray) -> SubjectSummary:
        """The same summary on a given subject list (zeros for subjects not seen)."""
        subjects = np.asarray(subjects)
        pos = {s: i for i, s in enumerate(self.subjects)}
        n = subjects.size
        hours = np.zeros(n)
        n_al = np.zeros(n, dtype=int)
        n_true = np.zeros(n, dtype=int)
        n_ev = np.zeros(n, dtype=int)
        n_det = np.zeros(n, dtype=int)
        wts: list[np.ndarray] = [np.zeros(0)] * n
        for j, s in enumerate(subjects):
            i = pos.get(s)
            if i is None:
                continue
            hours[j] = self.observed_hours[i]
            n_al[j] = self.n_alerts[i]
            n_true[j] = self.n_true_alerts[i]
            n_ev[j] = self.n_events[i]
            n_det[j] = self.n_detected[i]
            wts[j] = self.warning_times[i]
        return SubjectSummary(subjects, hours, n_al, n_true, n_ev, n_det, wts)


def subject_summary(
    matched: Matched,
    observation: pd.DataFrame,
    cols: Columns = DEFAULT_COLUMNS,
) -> SubjectSummary:
    """Per-subject statistics from a matching result and the observation intervals.

    Subjects are the union of those with observation, events or alerts, so
    that monitored subjects without any event or alert still contribute
    observed time to the burden denominator.
    """
    obs = check_observation(observation, cols)
    ev = matched.events
    al = matched.alerts
    subjects = np.array(
        sorted(set(obs[cols.subject]) | set(ev[cols.subject]) | set(al[cols.subject]), key=str)
    )
    pos = {s: i for i, s in enumerate(subjects)}
    n = subjects.size
    hours = np.zeros(n)
    for s, g in obs.groupby(cols.subject, sort=False):
        hours[pos[s]] = observed_hours(g, cols)
    n_al = np.zeros(n, dtype=int)
    n_true = np.zeros(n, dtype=int)
    if len(al):
        for s, g in al.groupby(cols.subject, sort=False):
            n_al[pos[s]] = len(g)
            n_true[pos[s]] = int(g["true_alert"].sum())
    n_ev = np.zeros(n, dtype=int)
    n_det = np.zeros(n, dtype=int)
    wts: list[np.ndarray] = [np.zeros(0)] * n
    if len(ev):
        for s, g in ev.groupby(cols.subject, sort=False):
            i = pos[s]
            n_ev[i] = len(g)
            det = g["detected"].to_numpy(dtype=bool)
            n_det[i] = int(det.sum())
            wts[i] = g["warning_time"].to_numpy(dtype=float)[det]
    return SubjectSummary(subjects, hours, n_al, n_true, n_ev, n_det, wts)


def resample_weights(n_subjects: int, n_resamples: int, seed: int = 0) -> np.ndarray:
    """Integer multiplicities of each subject in each resample (``n_resamples x n_subjects``)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_subjects, size=(n_resamples, n_subjects))
    w = np.zeros((n_resamples, n_subjects), dtype=np.int64)
    rows = np.repeat(np.arange(n_resamples), n_subjects)
    np.add.at(w, (rows, draws.ravel()), 1)
    return w


def weighted_median_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Median of the multiset given by integer ``weights`` per row of ``values``.

    ``values`` is sorted ascending and shared by all rows; ``weights`` is
    ``n_rows x len(values)``. Returns the median that ``numpy.median`` would
    give on the expanded multiset (the mean of the two middle elements when
    the total count is even), NaN for rows with zero total weight.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights)
    out = np.full(w.shape[0], np.nan)
    if v.size == 0:
        return out
    cum = np.cumsum(w, axis=1)
    total = cum[:, -1]
    for r in np.flatnonzero(total > 0):
        t = int(total[r])
        k_lo = (t + 1) // 2  # 1-indexed rank of the lower middle element
        k_hi = t // 2 + 1  # upper middle element (equal to k_lo when t is odd)
        i_lo = int(np.searchsorted(cum[r], k_lo, side="left"))
        i_hi = int(np.searchsorted(cum[r], k_hi, side="left"))
        out[r] = 0.5 * (v[i_lo] + v[i_hi])
    return out


def _metrics_from_weights(summary: SubjectSummary, w: np.ndarray) -> dict[str, np.ndarray]:
    """All metrics for every row of multiplicities ``w``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        days = (w @ summary.observed_hours) / 24.0
        n_alerts = w @ summary.n_alerts
        burden = np.where(days > 0, 100.0 * n_alerts / days, np.nan)
        n_events = w @ summary.n_events
        sens = np.where(n_events > 0, (w @ summary.n_detected) / n_events, np.nan)
        ppv = np.where(n_alerts > 0, (w @ summary.n_true_alerts) / n_alerts, np.nan)
    # warning times: one column per detected event, weighted by its subject's multiplicity
    owner = np.concatenate(
        [np.full(len(x), i, dtype=int) for i, x in enumerate(summary.warning_times)]
        or [np.zeros(0, dtype=int)]
    )
    wt = np.concatenate(summary.warning_times or [np.zeros(0)])
    order = np.argsort(wt, kind="stable")
    wt_sorted, owner_sorted = wt[order], owner[order]
    med = weighted_median_rows(wt_sorted, w[:, owner_sorted])
    # zero for undetected events: prepend the undetected counts as zeros
    n_undet = summary.n_events - summary.n_detected
    zero_owner = np.repeat(np.arange(len(summary.subjects)), n_undet)
    v0 = np.concatenate([np.zeros(zero_owner.size), wt_sorted])
    w0 = np.concatenate([w[:, zero_owner], w[:, owner_sorted]], axis=1)
    med0 = weighted_median_rows(v0, w0)
    return {
        "alerts_per_100_days": burden,
        "sensitivity": sens,
        "ppv": ppv,
        "median_warning_time": med,
        "median_warning_time_zero_undetected": med0,
    }


def resample_metrics(
    summary: SubjectSummary, n_resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """One row per resample with every metric; the same ``seed`` gives the same draws."""
    w = resample_weights(len(summary.subjects), n_resamples, seed)
    return pd.DataFrame(_metrics_from_weights(summary, w))


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval, NaNs ignored; ``(nan, nan)`` when nothing is finite."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(v, [a, 1.0 - a])
    return (float(lo), float(hi))


def bootstrap_evaluate(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    observation: pd.DataFrame,
    window: float,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    cols: Columns = DEFAULT_COLUMNS,
) -> dict[str, float]:
    """Point estimates with subject-bootstrap percentile intervals (``<metric>_lo`` and ``_hi``)."""
    m = match_alerts_to_events(alerts, events, window, observation, cols)
    s = subject_summary(m, observation, cols)
    out: dict[str, float] = s.point_estimates()
    out["n_subjects"] = float(len(s.subjects))
    out["n_resamples"] = float(n_resamples)
    res = resample_metrics(s, n_resamples, seed)
    for k in METRICS:
        lo, hi = percentile_interval(res[k].to_numpy(), level)
        out[f"{k}_lo"], out[f"{k}_hi"] = lo, hi
    return out


def paired_resamples(
    summaries: dict[str, SubjectSummary], n_resamples: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Metrics of several alert streams under the same subject draws.

    Streams are aligned to the union of their subjects first. The result
    has a two-level column index ``(stream name, metric)``; differences
    between streams on the same row are paired.
    """
    all_subjects = np.array(
        sorted(set().union(*(set(s.subjects) for s in summaries.values())), key=str)
    )
    w = resample_weights(all_subjects.size, n_resamples, seed)
    parts = {}
    for name, s in summaries.items():
        parts[name] = pd.DataFrame(_metrics_from_weights(s.aligned_to(all_subjects), w))
    return pd.concat(parts, axis=1)


def paired_difference(
    summary_a: SubjectSummary,
    summary_b: SubjectSummary,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """``b - a`` for every metric, point estimate and paired percentile interval."""
    res = paired_resamples({"a": summary_a, "b": summary_b}, n_resamples, seed)
    pa, pb = summary_a.point_estimates(), summary_b.point_estimates()
    out: dict[str, float] = {}
    for k in METRICS:
        d = (res[("b", k)] - res[("a", k)]).to_numpy()
        out[f"{k}_diff"] = pb[k] - pa[k]
        out[f"{k}_diff_lo"], out[f"{k}_diff_hi"] = percentile_interval(d, level)
    return out


__all__ = [
    "METRICS",
    "SubjectSummary",
    "bootstrap_evaluate",
    "paired_difference",
    "paired_resamples",
    "percentile_interval",
    "resample_metrics",
    "resample_weights",
    "subject_summary",
    "weighted_median_rows",
]
