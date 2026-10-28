"""Alarm burden: how many alerts a unit absorbs per unit of monitored time.

The textbook quantity is alerts per 100 patient-days (or infant-days):
``100 * n_alerts / observed_days``. It is the number a charge nurse feels,
and it is comparable across cohorts of different size and length of stay,
which a raw alert count or an alert fraction per window is not.
"""

from __future__ import annotations

import pandas as pd

from .streams import DEFAULT_COLUMNS, Columns, check_alerts, check_observation, inside_observation


def observed_hours(observation: pd.DataFrame, cols: Columns = DEFAULT_COLUMNS) -> float:
    """Total monitored time, summed over all intervals of all subjects.

    Overlapping intervals of the same subject are merged first so that time is
    not double counted.
    """
    obs = check_observation(observation, cols)
    total = 0.0
    for _, g in obs.groupby(cols.subject, sort=False):
        cur_s, cur_e = None, None
        for s, e in zip(g[cols.start].to_numpy(), g[cols.end].to_numpy(), strict=True):
            if cur_s is None:
                cur_s, cur_e = s, e
            elif s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
        if cur_s is not None:
            total += cur_e - cur_s
    return float(total)


def observed_days(observation: pd.DataFrame, cols: Columns = DEFAULT_COLUMNS) -> float:
    """``observed_hours`` divided by 24. Assumes the time axis is in hours."""
    return observed_hours(observation, cols) / 24.0


def count_alerts(
    alerts: pd.DataFrame,
    observation: pd.DataFrame | None = None,
    cols: Columns = DEFAULT_COLUMNS,
) -> int:
    """Number of alerts, restricted to observed intervals when ``observation`` is given."""
    al = check_alerts(alerts, cols)
    if observation is None:
        return int(len(al))
    return int(inside_observation(al, observation, cols).sum())


def alerts_per_100_days(
    alerts: pd.DataFrame,
    observation: pd.DataFrame,
    cols: Columns = DEFAULT_COLUMNS,
) -> float:
    """Alerts per 100 patient-days: ``100 * n_alerts / observed_days``.

    Only alerts inside observed intervals count. Returns NaN when nothing was
    observed.
    """
    days = observed_days(observation, cols)
    if days <= 0:
        return float("nan")
    return 100.0 * count_alerts(alerts, observation, cols) / days


def alerts_per_subject_day(
    alerts: pd.DataFrame,
    observation: pd.DataFrame,
    cols: Columns = DEFAULT_COLUMNS,
) -> pd.Series:
    """Per-subject alert rate (alerts per observed day), indexed by subject.

    Useful for seeing whether burden is concentrated on a few patients.
    """
    al = check_alerts(alerts, cols)
    obs = check_observation(observation, cols)
    al = al[inside_observation(al, obs, cols)]
    n = al.groupby(cols.subject).size()
    hours = obs.groupby(cols.subject).apply(
        lambda g: observed_hours(g.assign(**{cols.subject: g.name}), cols), include_groups=False
    )
    days = hours / 24.0
    rate = n.reindex(days.index).fillna(0) / days.replace(0, float("nan"))
    rate.name = "alerts_per_day"
    return rate


__all__ = [
    "alerts_per_100_days",
    "alerts_per_subject_day",
    "count_alerts",
    "observed_days",
    "observed_hours",
]
