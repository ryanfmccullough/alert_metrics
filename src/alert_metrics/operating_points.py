"""Operating-point tables: what each threshold costs and buys, at the event level.

For a score stream, a warning window, and a refractory rule, the table reports
per threshold: alerts per 100 patient-days, event sensitivity, alert PPV,
median warning time among detected events, and the number of alerts. The
alarm-burden column is the one a unit feels; the warning-time column is the
one a clinician asks about; AUROC answers neither question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .burden import alerts_per_100_days, observed_days
from .matching import (
    alert_ppv,
    event_sensitivity,
    match_alerts_to_events,
    median_warning_time,
)
from .refractory import threshold_alerts
from .streams import DEFAULT_COLUMNS, Columns, check_events, check_observation


def evaluate_alerts(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    observation: pd.DataFrame,
    window: float,
    cols: Columns = DEFAULT_COLUMNS,
) -> dict[str, float]:
    """All event-level metrics for one alert stream."""
    m = match_alerts_to_events(alerts, events, window, observation, cols)
    return {
        "n_alerts": float(len(m.alerts)),
        "n_events": float(len(m.events)),
        "observed_days": observed_days(observation, cols),
        "alerts_per_100_days": alerts_per_100_days(alerts, observation, cols),
        "sensitivity": event_sensitivity(m),
        "ppv": alert_ppv(m),
        "median_warning_time": median_warning_time(m, "exclude"),
        "median_warning_time_zero_undetected": median_warning_time(m, "zero"),
    }


def operating_point_table(
    scores: pd.DataFrame,
    events: pd.DataFrame,
    observation: pd.DataFrame,
    window: float,
    thresholds: np.ndarray | None = None,
    refractory: float = 0.0,
    persistence: int = 1,
    cols: Columns = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """One row per threshold.

    ``thresholds`` defaults to 101 evenly spaced values over the observed
    score range (quantile spacing is often better for skewed scores; pass your
    own grid). ``window`` is the warning window; ``refractory`` and
    ``persistence`` are passed to :func:`alert_metrics.refractory.threshold_alerts`.
    """
    ev = check_events(events, cols)
    obs = check_observation(observation, cols)
    s = scores[cols.score].to_numpy(dtype=float)
    s = s[~np.isnan(s)]
    if thresholds is None:
        if s.size == 0:
            raise ValueError("no finite scores")
        thresholds = np.linspace(float(s.min()), float(s.max()), 101)
    rows = []
    for t in np.asarray(thresholds, dtype=float):
        al = threshold_alerts(scores, t, refractory=refractory, persistence=persistence, cols=cols)
        row = {"threshold": float(t)}
        row.update(evaluate_alerts(al, ev, obs, window, cols))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_at_sensitivity(table: pd.DataFrame, floor: float) -> pd.Series:
    """Highest threshold whose event sensitivity is at least ``floor``.

    The highest such threshold has the fewest alerts among those meeting the
    floor, which is how a unit would set it.
    """
    ok = table[table["sensitivity"] >= floor]
    if ok.empty:
        raise ValueError(f"no threshold reaches sensitivity {floor}")
    return ok.loc[ok["threshold"].idxmax()]


def threshold_at_alert_budget(table: pd.DataFrame, alerts_per_100_days: float) -> pd.Series:
    """Lowest threshold whose alarm burden fits the budget; reports what it buys."""
    ok = table[table["alerts_per_100_days"] <= alerts_per_100_days]
    if ok.empty:
        raise ValueError(f"no threshold fits a budget of {alerts_per_100_days} alerts per 100 days")
    return ok.loc[ok["threshold"].idxmin()]


def ppv_at_sensitivity(table: pd.DataFrame, floor: float) -> float:
    """PPV at the operating point chosen by :func:`threshold_at_sensitivity`."""
    return float(threshold_at_sensitivity(table, floor)["ppv"])


__all__ = [
    "evaluate_alerts",
    "operating_point_table",
    "ppv_at_sensitivity",
    "threshold_at_alert_budget",
    "threshold_at_sensitivity",
]
