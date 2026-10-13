from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.features.combined_feature_manager import CombinedFeatureManager
from src.data.features.label_manager import LabelManager

STORAGE = {"backend": "local"}
# Thu, Fri, Mon (after a weekend), Tue, Wed; Thanksgiving Thu 11-27 is closed.
DAYS = ["2025-11-20", "2025-11-21", "2025-11-24", "2025-11-25", "2025-11-26"]


def daily_bars(day: str, symbols: list[str]) -> pd.DataFrame:
    opens = {"AAA": 100.0, "BBB": 50.0}
    bump = DAYS.index(day) if day in DAYS else 9
    frame = pd.DataFrame(
        {
            "datetime_utc": pd.Timestamp(day, tz="UTC"),
            "symbol": symbols,
            "open": [opens[s] + bump for s in symbols],
            "high": [opens[s] + bump + 2 for s in symbols],
            "low": [opens[s] + bump - 2 for s in symbols],
            "close": [opens[s] + bump + 1 for s in symbols],
            "volume": 1000.0,
        }
    )
    frame["vwap"] = frame["close"]
    frame["vwap_volume"] = frame["vwap"] * frame["volume"]
    return frame


def write_config(root: Path, name: str, config: dict) -> Path:
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def label_root(tmp_path):
    raw = tmp_path / "raw/session_daily/v1"
    raw.mkdir(parents=True)
    minutes = tmp_path / "features/minutes/v1"
    minutes.mkdir(parents=True)
    for day in DAYS:
        # BBB has no daily bar on Tue 11-25.
        symbols = ["AAA"] if day == "2025-11-25" else ["AAA", "BBB"]
        daily_bars(day, symbols).to_parquet(raw / f"{pd.Timestamp(day):%Y%m%d}.session-1d.parquet")
        pd.DataFrame(
            {"date": pd.Timestamp(day), "symbol": ["AAA", "BBB"], "close_cutoff": [200.0, 80.0]}
        ).to_parquet(minutes / f"{day}.parquet")

    config = {
        "version": {"labels": "v1", "raw_data": "v1", "minute_features": "v1"},
        "label": {"field": "Ref($open, -1) / $close_cutoff - 1", "column": "label"},
        "schedule": {"start_date": "2025-11-20", "end_date": "2025-11-26"},
        "storage": {
            **STORAGE,
            "local": {"root": str(tmp_path)},
            "input": {"raw": "raw/session_daily", "minute_features": "features/minutes"},
            "output": {"labels": "labels"},
        },
    }
    return tmp_path, write_config(tmp_path, "label", config)


def label_of(labels: pd.DataFrame, day: str, symbol: str) -> float | None:
    row = labels[(labels["date"] == day) & (labels["symbol"] == symbol)]
    return None if row.empty else row["label"].item()


def test_label_uses_next_session_open(label_root):
    _, config = label_root
    labels = LabelManager(config).calculate_labels()
    # Thu -> Fri open (100+1), Fri -> Mon open (100+2): the weekend is skipped.
    assert label_of(labels, "2025-11-20", "AAA") == pytest.approx(101 / 200 - 1)
    assert label_of(labels, "2025-11-21", "AAA") == pytest.approx(102 / 200 - 1)


def test_missing_next_session_gives_no_label(label_root):
    _, config = label_root
    labels = LabelManager(config).calculate_labels()
    # BBB has no Tue bar: Mon's label must not fall back to Wed's open.
    assert label_of(labels, "2025-11-24", "BBB") is None
    assert label_of(labels, "2025-11-25", "BBB") == pytest.approx((50 + 4) / 80 - 1)


def test_last_day_has_no_label_until_next_session(label_root):
    root, config = label_root
    labels = LabelManager(config).calculate_labels()
    # Wed 11-26: the next session (Fri 11-28) has no raw file yet.
    assert label_of(labels, "2025-11-26", "AAA") is None
    assert (root / "labels/v1/2025-11-25.csv").exists()
    assert not (root / "labels/v1/2025-11-26.csv").exists()


def test_label_rejects_unknown_field(tmp_path):
    config = {
        "version": {"labels": "v1", "raw_data": "v1", "minute_features": "v1"},
        "label": {"field": "Ref($open, -1) / $close_cutof - 1", "column": "label"},
        "schedule": {"start_date": "2025-11-20", "end_date": "2025-11-26"},
        "storage": {
            **STORAGE,
            "input": {"raw": "raw/daily", "minute_features": "features/minutes"},
            "output": {"labels": "labels"},
        },
    }
    with pytest.raises(ValueError, match="unknown field"):
        LabelManager(write_config(tmp_path, "label", config))


@pytest.fixture
def combined_root(tmp_path):
    for sub in ("features/minutes/v1", "features/daily/v1", "labels/v1"):
        (tmp_path / sub).mkdir(parents=True)
    for day in ["2025-11-21", "2025-11-24", "2025-11-25"]:
        pd.DataFrame(
            {
                "date": pd.Timestamp(day),
                "symbol": ["AAA", "BBB"],
                "index_group": ["sp500", "nasdaq100"],
                "MA5": [0.99, 1.01],
                "close_cutoff": [200.0, 80.0],
            }
        ).to_parquet(tmp_path / f"features/minutes/v1/{day}.parquet")
    for day in ["2025-11-19", "2025-11-20", "2025-11-21", "2025-11-24"]:
        symbols = ["AAA"] if day == "2025-11-21" else ["AAA", "BBB"]  # BBB missing on Fri
        pd.DataFrame(
            {
                "date": pd.Timestamp(day),
                "symbol": symbols,
                "index_group": "stale",
                "MA5": [float(day[-2:])] * len(symbols),
            }
        ).to_parquet(tmp_path / f"features/daily/v1/{day}.parquet")
    pd.DataFrame({"date": ["2025-11-21"] * 2, "symbol": ["AAA", "BBB"], "label": [0.02, -0.01]}).to_csv(
        tmp_path / "labels/v1/2025-11-21.csv", index=False
    )
    config = {
        "version": {"daily_features": "v1", "minute_features": "v1", "labels": "v1", "combined_features": "v1"},
        "schedule": {"start_date": "2025-11-21", "end_date": "2025-11-25"},
        "storage": {
            **STORAGE,
            "local": {"root": str(tmp_path)},
            "input": {"minute_features": "features/minutes", "daily_features": "features/daily", "labels": "labels"},
            "output": {"features": "features/combined"},
        },
    }
    return tmp_path, write_config(tmp_path, "combined", config)


def test_combined_joins_previous_session_exactly(combined_root):
    _, config = combined_root
    out = CombinedFeatureManager(config).combine()
    row = lambda day, symbol: out[(out["date"] == day) & (out["symbol"] == symbol)].iloc[0]
    # Mon 11-24 takes Fri 11-21's daily features.
    assert row("2025-11-24", "AAA")["daily_feature_date"] == pd.Timestamp("2025-11-21")
    assert row("2025-11-24", "AAA")["daily_MA5"] == 21.0
    # BBB has no Fri daily row: empty, not Thu's stale values.
    assert np.isnan(row("2025-11-24", "BBB")["daily_MA5"])
    assert row("2025-11-21", "BBB")["daily_MA5"] == 20.0


def test_combined_columns_labels_and_row_count(combined_root):
    root, config = combined_root
    out = CombinedFeatureManager(config).combine()
    assert len(out) == 6
    assert {"min_MA5", "daily_MA5", "label"} <= set(out.columns)
    assert "close_cutoff" not in out.columns and "min_close_cutoff" not in out.columns
    assert set(out["index_group"]) == {"sp500", "nasdaq100"}
    assert out.loc[out["date"] == "2025-11-21", "label"].tolist() == [0.02, -0.01]
    assert out.loc[out["date"] == "2025-11-25", "label"].isna().all()
    assert (root / "features/combined/v1/2025-11-25.parquet").exists()


def test_combined_rejects_duplicate_daily_rows(combined_root):
    root, config = combined_root
    duplicate = pd.read_parquet(root / "features/daily/v1/2025-11-20.parquet")
    pd.concat([duplicate, duplicate]).to_parquet(root / "features/daily/v1/2025-11-20.parquet")
    with pytest.raises(pd.errors.MergeError):
        CombinedFeatureManager(config).combine()
