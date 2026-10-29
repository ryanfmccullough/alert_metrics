# Status

Last updated 2026-10-28.

## Done

- Input conventions (`alerts`, `events`, `observation`) with validation and observation masks.
- Alarm burden: alerts per 100 patient-days with interval merging; per-subject rates.
- Event matching: true/false alerts, detected events, warning time per event, nearest-upcoming-event attribution.
- Refractory and persistence rules; `threshold_alerts` turns a score table into an alert table. Persistence runs are computed in vectorised form (`run_lengths`), with resets at subject boundaries: six million samples in well under a second. A pre-sorted score table with numeric subject ids skips the per-call sort.
- Calibration: reliability bins, ECE, Brier with Murphy decomposition, Cox slope and intercept.
- Operating-point table over thresholds and selectors (sensitivity floor, alert budget, PPV at floor).
- 37 tests on hand-checkable cases and synthetic streams. `ruff check` and `ruff format` clean.

## Open

- Bootstrap confidence intervals (resample subjects) for every metric in `evaluate_alerts`.
- Per-subject-day burden distribution and a "share of alerts from the top 10% of patients" summary.
- Time-varying observation: allow a per-sample validity mask instead of intervals.
- Temperature and Platt recalibration helpers (fit on one split, apply on another).
- Plotting helpers: reliability diagram, sensitivity against alerts per 100 patient-days, warning-time histogram.
- JOSS: finish `paper/paper.md`, add a worked example notebook on synthetic streams, CI badge, Zenodo DOI on the first tag.
