"""alert_metrics: event-level evaluation of continuous early-warning scores.

Alarm burden (alerts per 100 patient-days), event sensitivity, alert PPV,
warning time, refractory and persistence rules, calibration summaries, and
operating-point tables. Textbook definitions with citations in each module.
Nothing here is clinical advice; it is a scoring library.
"""

from .burden import (
    alerts_per_100_days,
    alerts_per_subject_day,
    count_alerts,
    observed_days,
    observed_hours,
)
from .calibration import (
    ReliabilityBins,
    brier_decomposition,
    brier_score,
    calibration_slope_intercept,
    calibration_summary,
    expected_calibration_error,
    reliability_bins,
)
from .matching import (
    Matched,
    alert_ppv,
    event_sensitivity,
    false_alerts,
    match_alerts_to_events,
    median_warning_time,
    warning_times,
)
from .operating_points import (
    evaluate_alerts,
    operating_point_table,
    ppv_at_sensitivity,
    threshold_at_alert_budget,
    threshold_at_sensitivity,
)
from .refractory import apply_refractory, persistence_mask, refractory_mask, threshold_alerts
from .streams import Columns, check_alerts, check_events, check_observation, inside_observation

__version__ = "0.1.1"

__all__ = [
    "Columns",
    "Matched",
    "ReliabilityBins",
    "__version__",
    "alert_ppv",
    "alerts_per_100_days",
    "alerts_per_subject_day",
    "apply_refractory",
    "brier_decomposition",
    "brier_score",
    "calibration_slope_intercept",
    "calibration_summary",
    "check_alerts",
    "check_events",
    "check_observation",
    "count_alerts",
    "evaluate_alerts",
    "event_sensitivity",
    "expected_calibration_error",
    "false_alerts",
    "inside_observation",
    "match_alerts_to_events",
    "median_warning_time",
    "observed_days",
    "observed_hours",
    "operating_point_table",
    "persistence_mask",
    "ppv_at_sensitivity",
    "refractory_mask",
    "reliability_bins",
    "threshold_alerts",
    "threshold_at_alert_budget",
    "threshold_at_sensitivity",
    "warning_times",
]
