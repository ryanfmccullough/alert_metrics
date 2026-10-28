"""Turning a score stream into an alert stream: thresholds, refractory periods, persistence.

A continuous score crosses a threshold many times in a row; a bedside alert
system does not re-alarm on every sample. Two textbook suppressions:

refractory period
    After an alert fires at time ``t``, no further alert fires in
    ``(t, t + refractory]``. This is the "silence after alarm" rule of most
    monitors and of the HeRO sepsis display, which updates hourly.
persistence
    An alert fires only when the score has been at or above threshold for
    ``k`` consecutive samples. This filters single-sample spikes.

Both are deterministic functions of the score stream and are therefore part
of the model under evaluation: two models with identical scores but different
refractory rules have different alarm burden and warning time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .streams import DEFAULT_COLUMNS, Columns


def refractory_mask(times: np.ndarray, refractory: float) -> np.ndarray:
    """Boolean mask of alerts kept after applying a refractory period.

    ``times`` must be sorted ascending (one subject). The first alert is kept;
    an alert at ``t`` is kept iff ``t > t_last_kept + refractory``. A zero
    refractory period means no suppression (every alert is kept, duplicates
    included).
    """
    times = np.asarray(times, dtype=float)
    n = times.size
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    if refractory < 0:
        raise ValueError("refractory must be non-negative")
    if n > 1 and np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted ascending")
    if refractory == 0:
        keep[:] = True
        return keep
    i = 0
    while i < n:
        keep[i] = True
        i = int(np.searchsorted(times, times[i] + refractory, side="right"))
    return keep


def apply_refractory(
    alerts: pd.DataFrame,
    refractory: float,
    cols: Columns = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """Apply a refractory period per subject to an alert table."""
    if refractory == 0 or len(alerts) == 0:
        return alerts.sort_values([cols.subject, cols.time], kind="stable").reset_index(drop=True)
    al = alerts.sort_values([cols.subject, cols.time], kind="stable").reset_index(drop=True)
    keep = np.zeros(len(al), dtype=bool)
    t_all = al[cols.time].to_numpy(dtype=float)
    for idx in al.groupby(cols.subject, sort=False).indices.values():
        keep[idx] = refractory_mask(t_all[idx], refractory)
    return al.loc[keep].reset_index(drop=True)


def persistence_mask(above: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of samples where ``above`` has been True for ``k`` consecutive samples.

    ``above`` is a boolean array in time order (one subject). With ``k=1`` the
    mask equals ``above``.
    """
    above = np.asarray(above, dtype=bool)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1 or above.size == 0:
        return above.copy()
    return run_lengths(above) >= k


def run_lengths(above: np.ndarray, first_of_group: np.ndarray | None = None) -> np.ndarray:
    """Length of the run of True values ending at each position (0 where False).

    Vectorised: with ``cs`` the cumulative count of True values, the run at a
    position is ``cs`` minus ``cs`` at the last reset, where a reset is a
    False value or, when ``first_of_group`` is given, the first sample of a
    subject. Runs never cross a subject boundary.
    """
    above = np.asarray(above, dtype=bool)
    n = above.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cs = np.cumsum(above, dtype=np.int64)
    base = np.where(above, 0, cs)
    if first_of_group is not None:
        first = np.asarray(first_of_group, dtype=bool)
        base = np.where(first, cs - above.astype(np.int64), base)
    base = np.maximum.accumulate(base)
    return cs - base


def threshold_alerts(
    scores: pd.DataFrame,
    threshold: float,
    refractory: float = 0.0,
    persistence: int = 1,
    cols: Columns = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """Alert table from a score table (``subject``, ``time``, ``score``).

    An alert fires at a sample when the score is ``>= threshold`` for
    ``persistence`` consecutive samples, subject to a refractory period.
    Samples with NaN scores never fire and break persistence runs.
    """
    sc = scores[[cols.subject, cols.time, cols.score]].sort_values(
        [cols.subject, cols.time], kind="stable"
    )
    s = sc[cols.score].to_numpy(dtype=float)
    above_all = np.where(np.isnan(s), False, s >= threshold)
    if persistence == 1:
        fire = above_all
    else:
        subj = sc[cols.subject].to_numpy()
        first = np.ones(len(sc), dtype=bool)
        first[1:] = subj[1:] != subj[:-1]
        fire = run_lengths(above_all, first) >= persistence
    alerts = sc.loc[fire, [cols.subject, cols.time]].reset_index(drop=True)
    return apply_refractory(alerts, refractory, cols)


__all__ = [
    "apply_refractory",
    "persistence_mask",
    "refractory_mask",
    "run_lengths",
    "threshold_alerts",
]
