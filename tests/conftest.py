import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_cohort():
    """Two subjects, hand-checkable.

    Subject A: observed 0 to 100 h, one event at 50 h, alerts at 10, 30, 40, 49.5, 60.
    Subject B: observed 0 to 48 h, no events, alerts at 5 and 20.
    With a 24 h window: A's alerts at 30, 40, 49.5 are true (first at 30, warning 20 h);
    10 and 60 are false; both of B's are false. Sensitivity 1/1, PPV 3/7.
    """
    alerts = pd.DataFrame(
        {
            "subject": ["A"] * 5 + ["B"] * 2,
            "time": [10.0, 30.0, 40.0, 49.5, 60.0, 5.0, 20.0],
        }
    )
    events = pd.DataFrame({"subject": ["A"], "time": [50.0]})
    observation = pd.DataFrame({"subject": ["A", "B"], "start": [0.0, 0.0], "end": [100.0, 48.0]})
    return alerts, events, observation


@pytest.fixture
def synthetic_scores():
    """A score stream where the score rises before each event.

    40 subjects, hourly samples over 240 h; subjects with an event at 120 h get
    a score that climbs linearly over the 24 h before the event.
    """
    rng = np.random.default_rng(0)
    rows = []
    events = []
    for s in range(40):
        has_event = s % 4 == 0
        t = np.arange(0.0, 240.0, 1.0)
        score = rng.normal(0.0, 1.0, t.size)
        if has_event:
            pre = (t >= 96) & (t < 120)
            score[pre] += np.linspace(0.5, 4.0, pre.sum())
            events.append({"subject": f"s{s}", "time": 120.0})
        rows.append(pd.DataFrame({"subject": f"s{s}", "time": t, "score": score}))
    scores = pd.concat(rows, ignore_index=True)
    events = pd.DataFrame(events)
    observation = pd.DataFrame(
        {"subject": [f"s{s}" for s in range(40)], "start": 0.0, "end": 240.0}
    )
    return scores, events, observation
