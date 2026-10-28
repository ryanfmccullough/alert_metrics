"""Event-level matching of alerts to outcomes: sensitivity, PPV, warning time.

An alert is *true* if it fires inside the warning window ``[t_event - window,
t_event)`` of some event of the same subject; otherwise it is *false*. An
event is *detected* if at least one alert fires inside its warning window.

Warning time (lead time) of a detected event is ``t_event - t_first_alert``
where ``t_first_alert`` is the earliest alert inside the window. Alerts that
fire before the window opens do not count toward that event: a warning
earlier than the horizon is, by construction, indistinguishable from a false
alert for a different reason, which is why the horizon has to be fixed in
advance and reported.

These are the definitions used, with minor variations, across the early-warning
literature (e.g., the 24 h windows of neonatal sepsis scores and the 12 to 24 h
windows of adult deterioration scores). The package does not pick a window for
you; every function takes it as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .streams import (
    DEFAULT_COLUMNS,
    Columns,
    check_alerts,
    check_events,
    inside_observation,
)


@dataclass
class Matched:
    """Result of matching alerts to events.

    ``events`` has one row per event with columns ``subject``, ``time``,
    ``detected`` (bool), ``first_alert`` (time of the first alert in the
    window, NaN if undetected), ``warning_time`` (``time - first_alert``, NaN
    if undetected) and ``n_alerts_in_window``.

    ``alerts`` has one row per alert with columns ``subject``, ``time``,
    ``true_alert`` (bool) and ``event_time`` (the matched event, NaN for a
    false alert). When an alert sits inside the windows of two events it is
    attributed to the earlier (nearest upcoming) event.
    """

    events: pd.DataFrame
    alerts: pd.DataFrame
    window: float


def match_alerts_to_events(
    alerts: pd.DataFrame,
    events: pd.DataFrame,
    window: float,
    observation: pd.DataFrame | None = None,
    cols: Columns = DEFAULT_COLUMNS,
) -> Matched:
    """Match alerts to events within a warning window of length ``window``.

    If ``observation`` is given, alerts outside observed intervals are dropped
    before matching (events are kept regardless; whether an event is
    evaluable is the caller's decision and belongs in the event table).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    al = check_alerts(alerts, cols)
    ev = check_events(events, cols)
    if observation is not None:
        al = al[inside_observation(al, observation, cols)].reset_index(drop=True)

    ev_detected = np.zeros(len(ev), dtype=bool)
    ev_first = np.full(len(ev), np.nan)
    ev_count = np.zeros(len(ev), dtype=int)
    al_true = np.zeros(len(al), dtype=bool)
    al_event = np.full(len(al), np.nan)

    al_groups = al.groupby(cols.subject, sort=False).indices
    for subject, ev_idx in ev.groupby(cols.subject, sort=False).indices.items():
        a_idx = al_groups.get(subject)
        if a_idx is None:
            continue
        a_t = al[cols.time].to_numpy()[a_idx]  # sorted
        e_t = ev[cols.time].to_numpy()[ev_idx]  # sorted
        # For each alert, the nearest event strictly after it.
        nxt = np.searchsorted(e_t, a_t, side="right")
        has_next = nxt < len(e_t)
        gap = np.full(len(a_t), np.inf)
        gap[has_next] = e_t[nxt[has_next]] - a_t[has_next]
        is_true = has_next & (gap <= window) & (gap > 0)
        al_true[a_idx] = is_true
        al_event[a_idx[is_true]] = e_t[nxt[is_true]]
        for ei, et in zip(ev_idx, e_t, strict=True):
            lo = np.searchsorted(a_t, et - window, side="left")
            hi = np.searchsorted(a_t, et, side="left")  # alerts strictly before the event
            n_in = hi - lo
            ev_count[ei] = n_in
            if n_in > 0:
                ev_detected[ei] = True
                ev_first[ei] = a_t[lo]

    ev_out = ev.copy()
    ev_out["detected"] = ev_detected
    ev_out["first_alert"] = ev_first
    ev_out["warning_time"] = ev_out[cols.time] - ev_out["first_alert"]
    ev_out["n_alerts_in_window"] = ev_count
    al_out = al.copy()
    al_out["true_alert"] = al_true
    al_out["event_time"] = al_event
    return Matched(events=ev_out, alerts=al_out, window=float(window))


def event_sensitivity(matched: Matched) -> float:
    """Fraction of events with at least one alert in their warning window."""
    n = len(matched.events)
    return float(matched.events["detected"].mean()) if n else float("nan")


def alert_ppv(matched: Matched) -> float:
    """Fraction of alerts that fall inside the warning window of some event."""
    n = len(matched.alerts)
    return float(matched.alerts["true_alert"].mean()) if n else float("nan")


def warning_times(matched: Matched, undetected: str = "exclude") -> np.ndarray:
    """Warning times, one per event.

    ``undetected="exclude"`` returns only detected events (the usual
    "median warning time among detected events"); ``"zero"`` keeps undetected
    events with a warning time of 0 so that a score cannot improve its median
    by detecting fewer events. Report both, or report the first alongside
    sensitivity.
    """
    wt = matched.events["warning_time"].to_numpy(dtype=float)
    if undetected == "exclude":
        return wt[~np.isnan(wt)]
    if undetected == "zero":
        return np.where(np.isnan(wt), 0.0, wt)
    raise ValueError("undetected must be 'exclude' or 'zero'")


def median_warning_time(matched: Matched, undetected: str = "exclude") -> float:
    """Median of :func:`warning_times`; NaN if there are no qualifying events."""
    wt = warning_times(matched, undetected)
    return float(np.median(wt)) if wt.size else float("nan")


def false_alerts(matched: Matched) -> pd.DataFrame:
    """The false alerts, for inspection (subject and time)."""
    return matched.alerts.loc[~matched.alerts["true_alert"]].reset_index(drop=True)


__all__ = [
    "Matched",
    "alert_ppv",
    "event_sensitivity",
    "false_alerts",
    "match_alerts_to_events",
    "median_warning_time",
    "warning_times",
]
