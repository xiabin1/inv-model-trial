from __future__ import annotations

import pandas as pd
import pytest

from src.data.raw.corporate_actions import load_actions
from src.data.raw.dbn_store import files_by_day
from src.data.raw.ticker_renames import apply_renames, load_renames, parse_renames, ticker_for_date

RENAMES = parse_renames(
    [
        {"symbol": "FISV", "old": "FI", "until": "2025-11-10"},
        {"symbol": "BNY", "old": "BK", "until": "2026-05-20"},
    ]
)


def bars(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(UTC time, symbol, close) rows as minimal bars."""
    return pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime([row[0] for row in rows]).tz_localize("UTC"),
            "symbol": [row[1] for row in rows],
            "close": [row[2] for row in rows],
        }
    )


def test_ticker_for_date_on_and_after_the_change():
    assert ticker_for_date("FISV", pd.Timestamp("2025-11-10"), RENAMES) == "FI"
    assert ticker_for_date("FISV", pd.Timestamp("2025-11-11"), RENAMES) == "FISV"
    assert ticker_for_date("AAPL", pd.Timestamp("2025-01-02"), RENAMES) == "AAPL"


def test_old_and_new_ticker_load_as_one_series():
    out = apply_renames(
        bars([("2025-11-07 21:00", "FI", 68.0), ("2025-11-10 20:59", "FI", 67.5), ("2025-11-11 14:30", "FISV", 67.9)]),
        RENAMES,
    )
    assert out["symbol"].tolist() == ["FISV"] * 3
    assert out["close"].tolist() == [68.0, 67.5, 67.9]


def test_reused_ticker_before_the_change_is_dropped():
    out = apply_renames(
        bars([("2026-05-20 20:59", "BNY", 10.2), ("2026-05-20 20:59", "BK", 139.0), ("2026-05-21 13:30", "BNY", 135.0)]),
        RENAMES,
    )
    assert list(zip(out["symbol"], out["close"])) == [("BNY", 139.0), ("BNY", 135.0)]


def test_old_ticker_after_the_change_is_not_renamed():
    out = apply_renames(bars([("2026-01-05 14:30", "FI", 10.0)]), RENAMES)
    assert out["symbol"].tolist() == ["FI"]


@pytest.mark.parametrize(
    "entries",
    [[{"symbol": "FISV", "old": "FI"}], [{"symbol": "X", "old": "A", "until": "2025-01-02"}, {"symbol": "X", "old": "B", "until": "2025-02-03"}]],
)
def test_parse_rejects_bad_entries(entries):
    with pytest.raises(ValueError):
        parse_renames(entries)


def test_files_by_day_keeps_top_up_files():
    days = files_by_day(["raw/20250102.ohlcv-1m.dbn.zst", "raw/20250102.fix.ohlcv-1m.dbn.zst", "raw/20250103.ohlcv-1m.dbn.zst"])
    assert days[pd.Timestamp("2025-01-02")] == ["raw/20250102.fix.ohlcv-1m.dbn.zst", "raw/20250102.ohlcv-1m.dbn.zst"]
    assert len(days[pd.Timestamp("2025-01-03")]) == 1


def test_repo_config_loads():
    renames = load_renames()
    assert {rename.symbol for rename in renames} >= {"FISV", "PSKY", "MRSH", "P", "BNY", "ECHO", "VMRK"}
    assert all(action.verified for action in load_actions())
