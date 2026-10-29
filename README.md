# alert_metrics

Event-level evaluation of continuous early-warning scores: alarm burden, warning time, refractory suppression, calibration, and operating-point tables.

A risk score is not an alert system. Between the two sit a threshold, a refractory rule, and a warning window, and those choices decide what a unit experiences: how many alerts per day, how many of them precede an event, and how far ahead of the event the first one fires. AUROC answers none of those questions. This package computes the quantities that do, with textbook definitions and the citation for each one in the module docstrings.

## What it computes

- **Alarm burden**: alerts per 100 patient-days, counted only inside observed intervals (`burden.py`).
- **Event matching**: an alert is true if it fires inside the warning window `[t_event - window, t_event)` of an event of the same subject; an event is detected if at least one alert does. From that: event sensitivity, alert PPV, and warning time per event (`matching.py`).
- **Score to alerts**: threshold crossing with a refractory period (silence after an alert) and a persistence rule (k consecutive samples above threshold) (`refractory.py`).
- **Calibration**: reliability bins, expected calibration error, Brier score with the Murphy decomposition, and the Cox calibration slope and intercept (`calibration.py`).
- **Operating points**: one row per threshold with burden, sensitivity, PPV, and median warning time; selectors for the threshold that meets a sensitivity floor with the fewest alerts, or that fits an alert budget (`operating_points.py`).
- **Uncertainty**: subject-level (cluster) bootstrap percentile intervals for every metric, and paired differences between two alert streams under the same resamples (`bootstrap.py`).

## Input conventions

Three tables, all with float times on one common axis (hours by convention):

| table | columns | one row per |
|---|---|---|
| `alerts` | `subject`, `time` | alert |
| `events` | `subject`, `time` | outcome event |
| `observation` | `subject`, `start`, `end` | monitored interval (several per subject allowed) |

Scores for `operating_point_table` are a table with `subject`, `time`, `score`. Column names are configurable through `Columns`.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
```

```python
import pandas as pd
from alert_metrics import operating_point_table, threshold_at_sensitivity

scores = pd.DataFrame({"subject": ..., "time": ..., "score": ...})
events = pd.DataFrame({"subject": ..., "time": ...})
observation = pd.DataFrame({"subject": ..., "start": ..., "end": ...})

table = operating_point_table(scores, events, observation, window=24.0, refractory=6.0)
row = threshold_at_sensitivity(table, floor=0.80)
print(row[["threshold", "alerts_per_100_days", "ppv", "median_warning_time"]])
```

## Definitions in one place

- Warning window `W`: fixed in advance and reported. An alert earlier than `W` before an event is a false alert by construction.
- Warning time of a detected event: `t_event - t_first_alert_in_window`. Reported as the median over detected events, and also with undetected events scored as 0 so that a score cannot improve its median by detecting fewer events.
- Alerts per 100 patient-days: `100 * n_alerts / observed_days`, with alerts outside observed intervals excluded and overlapping intervals merged.
- Refractory period `R`: after an alert at `t`, nothing fires in `(t, t + R]`. A refractory rule cannot change the first alert before an event, so it changes burden without changing warning time.
- PPV at a sensitivity floor: the PPV at the highest threshold whose event sensitivity meets the floor.

## Layout

```
src/alert_metrics/
  streams.py           input tables, validation, observation masks
  burden.py            alerts per 100 patient-days
  matching.py          alert-to-event matching, sensitivity, PPV, warning time
  refractory.py        threshold crossing, refractory period, persistence
  calibration.py       reliability bins, ECE, Brier, calibration slope
  operating_points.py  per-threshold table and operating-point selectors
tests/                 hand-checkable cases and property tests; no dataset needed
paper/                 JOSS submission draft
```

## Status

Scaffold with 34 passing tests. `STATUS.md` lists what is done and what is open. Engineering verification only; nothing here is clinical advice.

## Citing

See `CITATION.cff`. A Zenodo DOI is minted at each tagged release.

## Author

Ryan F. McCullough, [ryanfmccullough.com](https://ryanfmccullough.com). MIT License.
