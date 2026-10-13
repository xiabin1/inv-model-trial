from __future__ import annotations

import pandas as pd

from src.walk_forward import test_periods as periods_of, training_rows

DAYS = list(pd.bdate_range("2026-01-05", "2026-02-27"))


def labeled_rows() -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS, "symbol": "A", "label": 1})


def test_monday_model_uses_labels_up_to_previous_thursday():
    monday = pd.Timestamp("2026-01-26")
    rows = training_rows(labeled_rows(), DAYS, monday, "expanding")
    # Friday 01-23's label needs Monday's open, so it is not known yet.
    assert rows["date"].max() == pd.Timestamp("2026-01-22")
    assert rows["date"].min() == DAYS[0]


def test_rolling_window_keeps_last_n_days():
    rows = training_rows(labeled_rows(), DAYS, pd.Timestamp("2026-02-02"), 5)
    assert rows["date"].tolist() == list(pd.bdate_range("2026-01-23", "2026-01-29"))


def test_periods_group_by_week_month_or_all():
    test_days = DAYS[10:]
    weeks = periods_of(test_days, "week")
    assert all(pd.DatetimeIndex(w).to_period("W-FRI").nunique() == 1 for w in weeks)
    assert sum(len(w) for w in weeks) == len(test_days)
    assert [len(m) for m in periods_of(test_days, "month")] == [10, 20]
    assert periods_of(test_days, "never") == [test_days]
    assert periods_of(test_days, "day") == [[day] for day in test_days]
