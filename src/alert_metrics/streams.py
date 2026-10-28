"""Input conventions shared by every function in the package.

Three tables describe an evaluation. All times are floats on one common axis
(hours is the convention used in the docstrings; any unit works as long as it
is used consistently):

``alerts``
    One row per alert. Columns ``subject`` and ``time``.
``events``
    One row per outcome event (the thing the alert was meant to warn about).
    Columns ``subject`` and ``time``.
``observation``
    One row per monitored interval. Columns ``subject``, ``start``, ``end``.
    A subject may have several intervals (monitoring gaps, censored periods).
    Alerts outside every interval of their subject are ignored, and alarm
    burden is normalised by the total observed time.

Column names are configurable through :class:`Columns`; the defaults are the
names above.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Columns:
    """Column names used by the three input tables."""

    subject: str = "subject"
    time: str = "time"
    start: str = "start"
    end: str = "end"
    score: str = "score"


DEFAULT_COLUMNS = Columns()


def _require(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; has {list(df.columns)}")


def check_alerts(alerts: pd.DataFrame, cols: Columns = DEFAULT_COLUMNS) -> pd.DataFrame:
    """Validate an alert table and return a copy sorted by subject and time."""
    _require(alerts, (cols.subject, cols.time), "alerts")
    out = alerts[[cols.subject, cols.time]].copy()
    if out[cols.time].isna().any():
        raise ValueError("alerts contain NaN times")
    return out.sort_values([cols.subject, cols.time], kind="stable").reset_index(drop=True)


def check_events(events: pd.DataFrame, cols: Columns = DEFAULT_COLUMNS) -> pd.DataFrame:
    """Validate an event table and return a copy sorted by subject and time."""
    _require(events, (cols.subject, cols.time), "events")
    out = events[[cols.subject, cols.time]].copy()
    if out[cols.time].isna().any():
        raise ValueError("events contain NaN times")
    return out.sort_values([cols.subject, cols.time], kind="stable").reset_index(drop=True)


def check_observation(observation: pd.DataFrame, cols: Columns = DEFAULT_COLUMNS) -> pd.DataFrame:
    """Validate observation intervals: finite, start <= end, sorted."""
    _require(observation, (cols.subject, cols.start, cols.end), "observation")
    out = observation[[cols.subject, cols.start, cols.end]].copy()
    if out[[cols.start, cols.end]].isna().any().any():
        raise ValueError("observation intervals contain NaN")
    if (out[cols.end] < out[cols.start]).any():
        raise ValueError("observation intervals must satisfy start <= end")
    return out.sort_values([cols.subject, cols.start], kind="stable").reset_index(drop=True)


def inside_observation(
    table: pd.DataFrame,
    observation: pd.DataFrame,
    cols: Columns = DEFAULT_COLUMNS,
) -> np.ndarray:
    """Boolean mask: which rows of ``table`` (subject, time) fall inside an observed interval.

    Intervals are closed, ``start <= time <= end``. Rows whose subject has no
    interval are marked False.
    """
    obs = check_observation(observation, cols)
    mask = np.zeros(len(table), dtype=bool)
    if len(table) == 0:
        return mask
    groups = {s: g for s, g in obs.groupby(cols.subject, sort=False)}
    for subject, idx in table.groupby(cols.subject, sort=False).indices.items():
        g = groups.get(subject)
        if g is None:
            continue
        t = table[cols.time].to_numpy()[idx]
        starts = g[cols.start].to_numpy()
        ends = g[cols.end].to_numpy()
        hit = np.zeros(len(t), dtype=bool)
        for s, e in zip(starts, ends, strict=True):
            hit |= (t >= s) & (t <= e)
        mask[idx] = hit
    return mask


__all__ = [
    "DEFAULT_COLUMNS",
    "Columns",
    "check_alerts",
    "check_events",
    "check_observation",
    "inside_observation",
]
