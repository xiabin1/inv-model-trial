from __future__ import annotations

import pandas as pd
import pytest

from src.data.raw.session_bars import load_session_bars, session_daily_bars, session_filename
from src.data.storage import Storage


def minute(symbol: str, ny_times: list[str], prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    times = pd.to_datetime(ny_times).tz_localize("America/New_York").tz_convert("UTC")
    volumes = volumes or [100.0] * len(prices)
    frame = pd.DataFrame(
        {"datetime_utc": times, "symbol": symbol, "open": prices, "high": prices, "low": prices,
         "close": prices, "volume": volumes}
    )
    frame["vwap"] = frame["close"]
    frame["vwap_volume"] = frame["vwap"] * frame["volume"]
    return frame


def test_extended_hours_are_excluded():
    # Like Gartner on 2025-06-11: a stray pre-market print, then a normal session.
    bars = minute(
        "IT",
        ["2025-06-11 07:05", "2025-06-11 09:30", "2025-06-11 12:00", "2025-06-11 15:59", "2025-06-11 17:30"],
        [169.81, 421.15, 436.19, 413.41, 900.0],
    )
    day = session_daily_bars(bars).iloc[0]
    assert (day["open"], day["high"], day["low"], day["close"]) == (421.15, 436.19, 413.41, 413.41)
    assert day["volume"] == 300
    assert day["datetime_utc"] == pd.Timestamp("2025-06-11", tz="UTC")


def test_half_day_session_ends_at_13_00():
    bars = minute("AAA", ["2025-11-28 09:30", "2025-11-28 12:59", "2025-11-28 13:00", "2025-11-28 15:00"],
                  [10.0, 11.0, 50.0, 60.0])
    day = session_daily_bars(bars).iloc[0]
    assert day["close"] == 11.0 and day["high"] == 11.0


def test_one_bar_per_day_and_symbol_with_volume_weighted_vwap():
    bars = pd.concat(
        [
            minute("AAA", ["2025-11-24 09:30", "2025-11-24 10:00"], [10.0, 20.0], [100.0, 300.0]),
            minute("AAA", ["2025-11-25 09:30"], [30.0]),
            minute("BBB", ["2025-11-24 09:31"], [5.0]),
        ]
    )
    daily = session_daily_bars(bars)
    assert len(daily) == 3
    aaa = daily[(daily["symbol"] == "AAA") & (daily["datetime_utc"] == pd.Timestamp("2025-11-24", tz="UTC"))].iloc[0]
    assert aaa["vwap"] == pytest.approx((10 * 100 + 20 * 300) / 400)


def test_round_trip_through_storage_applies_split_adjustment(tmp_path):
    storage = Storage({"backend": "local", "local": {"root": str(tmp_path)}})
    day = pd.Timestamp("2025-11-14")
    bars = session_daily_bars(minute("NFLX", ["2025-11-14 09:30", "2025-11-14 15:59"], [1100.0, 1115.0]))
    path = f"raw/session_daily/v1/{session_filename(day)}"
    storage.write_parquet(bars, path)
    assert session_filename(day) == "20251114.session-1d.parquet"
    raw = load_session_bars(storage, [path], adjust=False)
    adjusted = load_session_bars(storage, [path])  # NFLX 10-for-1 on 2025-11-17 is in the actions file
    assert raw["close"].item() == 1115.0
    assert adjusted["close"].item() == pytest.approx(111.5)
