from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.features import minutes_feature_manager as mfm
from src.data.features.minutes_feature_manager import (
    MinutesFeatureManager,
    intraday_base_fields,
    session_bounds,
)

SYMBOLS = ["AAA", "BBB"]
DAYS = ["2025-11-19", "2025-11-20", "2025-11-21", "2025-11-24", "2025-11-25", "2025-11-26", "2025-11-28"]


def minute_bars(day: str, seed: int, until: str = "16:00", close_at: str = "16:00") -> pd.DataFrame:
    """Random-walk 1m bars for SYMBOLS from 09:30 up to min(until, close_at), New York time."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{day} 09:30", f"{day} 16:00", freq="1min", inclusive="left")
    frames = []
    for number, symbol in enumerate(SYMBOLS):
        close = 100 * (number + 1) * np.exp(np.cumsum(rng.normal(0, 0.001, len(times))))
        open_ = np.r_[close[0], close[:-1]]
        frame = pd.DataFrame(
            {
                "datetime_utc": times.tz_localize("America/New_York").tz_convert("UTC"),
                "symbol": symbol,
                "open": open_,
                "high": np.maximum(open_, close) * 1.0005,
                "low": np.minimum(open_, close) * 0.9995,
                "close": close,
                "volume": rng.integers(100, 1000, len(times)).astype(float),
            }
        )
        frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3
        frame["vwap_volume"] = frame["vwap"] * frame["volume"]
        frames.append(frame)
    bars = pd.concat(frames, ignore_index=True)
    end = min(pd.Timestamp(f"{day} {until}"), pd.Timestamp(f"{day} {close_at}")).tz_localize("America/New_York")
    return bars[bars["datetime_utc"] < end].reset_index(drop=True)


def write_config(root: Path, **overrides) -> Path:
    config = {
        "version": {"features": "v1", "raw_data": "v1", "symbols": "v1"},
        "features": {
            "frequency": "10m",
            "feature_set": "Alpha158",
            "custom": {"minutes_vwap_spread": "$close / $vwap - 1"},
            "intraday": {
                "ret_day": "$close / $open - 1",
                "ret_gap": "$open / $prev_close - 1",
                "rvol": "$volume / Mean(Ref($volume, 1), 3)",
                "vol_intraday": "$ret_std",
            },
        },
        "schedule": {"cutoff_before_close": 10, "lookback_days": 3, "start_date": "2025-11-24", "end_date": "2025-11-28"},
        "storage": {
            "backend": "local",
            "local": {"root": str(root)},
            "input": {"raw": "raw/minutes", "symbols": "symbols"},
            "output": {"features": "features/minutes"},
        },
    }
    for section, values in overrides.items():
        config[section].update(values)
    path = root / "minutes_feature_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Empty raw files plus symbol snapshots; the DBN loader is replaced by minute_bars."""
    raw_dir = tmp_path / "raw/minutes/v1"
    raw_dir.mkdir(parents=True)
    symbol_dir = tmp_path / "symbols/v1"
    symbol_dir.mkdir(parents=True)
    for day in DAYS:
        (raw_dir / f"{pd.Timestamp(day):%Y%m%d}.ohlcv-1m.dbn.zst").touch()
        pd.DataFrame({"symbol": SYMBOLS, "index_group": ["sp500", "nasdaq100"]}).to_csv(
            symbol_dir / f"{day}.csv", index=False
        )
    overrides: dict[str, pd.DataFrame] = {}

    def fake_load(storage, paths):
        day = pd.to_datetime(Path(paths[0]).name[:8], format="%Y%m%d")
        key = f"{day:%Y-%m-%d}"
        if key in overrides:
            return overrides[key]
        close_at = "13:00" if key == "2025-11-28" else "16:00"
        return minute_bars(key, seed=DAYS.index(key), close_at=close_at)

    monkeypatch.setattr(mfm, "load_stored_dbn_bars", fake_load)
    return tmp_path, overrides


def test_session_bounds_regular_and_half_day():
    _, close, cutoff = session_bounds(pd.Timestamp("2025-11-26"), 10)
    assert cutoff.tz_convert("America/New_York").strftime("%H:%M") == "15:50"
    assert close.tz_convert("America/New_York").strftime("%H:%M") == "16:00"
    _, close, cutoff = session_bounds(pd.Timestamp("2025-11-28"), 10)
    assert cutoff.tz_convert("America/New_York").strftime("%H:%M") == "12:50"
    assert close.tz_convert("America/New_York").strftime("%H:%M") == "13:00"


def test_intraday_base_fields_stop_before_cutoff():
    bars = minute_bars("2025-11-26", seed=1)
    _, _, cutoff = session_bounds(pd.Timestamp("2025-11-26"), 10)
    base = intraday_base_fields(bars, cutoff)
    aaa = bars[(bars["symbol"] == "AAA") & (bars["datetime_utc"] < cutoff)]
    assert base.loc["AAA", "open"] == aaa["open"].iloc[0]
    assert base.loc["AAA", "close"] == aaa["close"].iloc[-1]  # 15:49 bar
    assert base.loc["AAA", "volume"] == aaa["volume"].sum()
    assert base.loc["AAA", "high"] == aaa["high"].max()
    at_1520 = aaa[aaa["datetime_utc"] < cutoff - pd.Timedelta(minutes=30)]["close"].iloc[-1]
    assert base.loc["AAA", "close_30m"] == at_1520
    assert base.loc["AAA", "drawdown"] <= 0


def test_frequency_must_line_up_with_cutoff(tmp_path):
    with pytest.raises(ValueError, match="straddle the cutoff"):
        MinutesFeatureManager(write_config(tmp_path, features={"frequency": "15m"}))


def test_repo_config_cutoff_is_1558_with_aligned_bars():
    config = Path(__file__).resolve().parents[1] / "config" / "feature" / "minutes_feature_config.yaml"
    manager = MinutesFeatureManager(config)
    _, _, cutoff = session_bounds(pd.Timestamp("2025-11-26"), manager.cutoff_before_close)
    assert cutoff.tz_convert("America/New_York").strftime("%H:%M") == "15:58"


def test_future_looking_formula_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="looks into the future"):
        MinutesFeatureManager(write_config(tmp_path, features={"custom": {"peek": "Ref($close, -1) / $close"}}))


def test_unknown_intraday_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown field"):
        MinutesFeatureManager(write_config(tmp_path, features={"intraday": {"bad": "$closee / $open"}}))


def test_batch_output_one_row_per_symbol_and_half_day_skipped(data_root):
    root, _ = data_root
    features = MinutesFeatureManager(write_config(root)).calculate_features()
    assert sorted(features["date"].dt.strftime("%Y-%m-%d").unique()) == ["2025-11-24", "2025-11-25", "2025-11-26"]
    assert features.groupby("date").size().eq(len(SYMBOLS)).all()
    assert set(features["index_group"]) == {"sp500", "nasdaq100"}
    assert (root / "features/minutes/v1/2025-11-26.parquet").exists()
    assert not (root / "features/minutes/v1/2025-11-28.parquet").exists()
    assert features[["MA60", "ret_gap", "rvol"]].notna().all().all()


def test_close_cutoff_is_last_price_before_cutoff(data_root):
    root, _ = data_root
    features = MinutesFeatureManager(write_config(root)).calculate_features("2025-11-26", "2025-11-26")
    bars = minute_bars("2025-11-26", seed=DAYS.index("2025-11-26"))
    cutoff = session_bounds(pd.Timestamp("2025-11-26"), 10)[2]
    at_1549 = bars[(bars["symbol"] == "AAA") & (bars["datetime_utc"] < cutoff)]["close"].iloc[-1]
    assert features.loc[features["symbol"] == "AAA", "close_cutoff"].item() == at_1549


def test_bars_after_cutoff_do_not_change_features(data_root):
    """Batch (full sessions) must equal a live run whose last day stops at 15:49."""
    root, overrides = data_root
    batch = MinutesFeatureManager(write_config(root)).calculate_features("2025-11-26", "2025-11-26")

    live_bars = minute_bars("2025-11-26", seed=DAYS.index("2025-11-26"), until="15:50")
    overrides["2025-11-26"] = live_bars
    live = MinutesFeatureManager(write_config(root)).calculate_features("2025-11-26", "2025-11-26")
    pd.testing.assert_frame_equal(batch, live)

    wild = minute_bars("2025-11-26", seed=DAYS.index("2025-11-26"))
    after = wild["datetime_utc"] >= session_bounds(pd.Timestamp("2025-11-26"), 10)[2]
    wild.loc[after, ["open", "high", "low", "close", "vwap"]] *= 3
    wild.loc[after, "volume"] *= 50
    overrides["2025-11-26"] = wild
    shocked = MinutesFeatureManager(write_config(root)).calculate_features("2025-11-26", "2025-11-26")
    pd.testing.assert_frame_equal(batch, shocked)


def test_chunked_batch_matches_single_day(data_root, monkeypatch):
    root, _ = data_root
    monkeypatch.setattr(mfm, "CHUNK_SESSIONS", 1)
    chunked = MinutesFeatureManager(write_config(root)).calculate_features()
    single = MinutesFeatureManager(write_config(root)).calculate_features("2025-11-25", "2025-11-25")
    pd.testing.assert_frame_equal(
        chunked[chunked["date"] == "2025-11-25"].reset_index(drop=True), single
    )


def test_state_features_parallel_matches_serial():
    day = pd.Timestamp(DAYS[1])
    bars = minute_bars(DAYS[1], seed=1)
    cutoff = session_bounds(day, 10)[2]
    state = mfm.summarize_session(bars, day, "10m", 10)["state_bars"]
    fields, columns = ["Mean($close, 5)/$close", "Std($close, 10)/$close"], ["MA5", "STD10"]
    serial = mfm.state_features(state, {day: cutoff}, pd.Timedelta(minutes=10), fields, columns)
    parallel = mfm.state_features(state, {day: cutoff}, pd.Timedelta(minutes=10), fields, columns, workers=2)
    pd.testing.assert_frame_equal(serial.sort_values("symbol").reset_index(drop=True), parallel.sort_values("symbol").reset_index(drop=True))
