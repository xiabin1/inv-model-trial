#!/usr/bin/env python3
"""Check stored features for look-ahead: recompute recent days the way a live run would.

For each of the last N sessions D that have combined features:

1. Minute features: recomputed with D's minute bars cut at the cutoff (as the 15:50
   pull sees them) and no later days; must equal the stored batch values.
2. Daily features of D: recomputed with session bars up to D only; must equal the
   stored values (combined rows for D+1 use them).
3. Combined rows: daily features come from the previous session; no label, return or
   cutoff-price column is a model input.
4. Formulas are unchanged when prices are scaled (as by a split): corporate-action
   adjustment of earlier bars uses later information, which is harmless only if no
   feature depends on the price level.

Nothing is written to storage.

    python3 scripts/check_leakage.py            # last 5 sessions
    python3 scripts/check_leakage.py --days 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.features import daily_feature_manager as dfm  # noqa: E402
from src.data.features import minutes_feature_manager as mfm  # noqa: E402
from src.data.features.combined_feature_manager import previous_sessions  # noqa: E402
from src.data.features.expression_engine import evaluate_expressions  # noqa: E402
from src.data.storage import Storage  # noqa: E402
from src.live import PROFILES  # noqa: E402

PATHS: dict = dict(PROFILES["basic"])  # set from --profile in main()

FORBIDDEN = ("label", "future", "return_next", "close_cutoff", "excess_return", "overnight_vol")
PRICE_FIELDS = ("open", "high", "low", "close", "vwap")
# Rolling statistics differ by rounding (~1e-7 for R-squared on near-flat windows) when they start
# from a different amount of history; look-ahead changes values far more than this.
TOLERANCE = 1e-6


def config(path: str) -> dict:
    with (ROOT_DIR / path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def compare(stored: pd.DataFrame, fresh: pd.DataFrame, what: str) -> int:
    """Print differing columns between two per-symbol tables; return the number of bad cells."""
    keys = ["date", "symbol"]
    columns = [c for c in stored.columns if c not in keys and c != "index_group" and pd.api.types.is_numeric_dtype(stored[c])]
    merged = stored.merge(fresh, on=keys, how="outer", suffixes=("_stored", "_fresh"), indicator=True)
    only = merged["_merge"].value_counts()
    bad = 0
    if only.get("left_only", 0) or only.get("right_only", 0):
        print(f"  {what}: rows only stored {only.get('left_only', 0)}, only recomputed {only.get('right_only', 0)}")
        bad += int(only.get("left_only", 0) + only.get("right_only", 0))
    both = merged[merged["_merge"] == "both"]
    worst = []
    for column in columns:
        a, b = both[f"{column}_stored"].astype(float), both[f"{column}_fresh"].astype(float)
        differs = ~(((a - b).abs() <= TOLERANCE) | (a.isna() & b.isna()))
        if differs.any():
            bad += int(differs.sum())
            worst.append((int(differs.sum()), column))
    if worst:
        print(f"  {what}: {bad} differing cells in {len(worst)} column(s), e.g. "
              + ", ".join(f"{column} ({count})" for count, column in sorted(worst, reverse=True)[:8]))
    else:
        print(f"  {what}: {len(both)} rows x {len(columns)} columns identical")
    return bad


def check_minutes(days: list[pd.Timestamp]) -> int:
    manager = mfm.MinutesFeatureManager(PATHS["minutes"])
    storage = Storage(manager.config["storage"], ROOT_DIR)
    loader, bad = mfm.load_stored_dbn_bars, 0
    manager._write = lambda *args, **kwargs: None
    for day in days:
        cutoff = mfm.session_bounds(day, manager.cutoff_before_close)[2]

        def live_loader(storage_, paths, adjust=True, day=day, cutoff=cutoff):
            bars = loader(storage_, paths, adjust=adjust)
            local_day = bars["datetime_utc"].dt.tz_convert(mfm.MARKET_TZ).dt.normalize().dt.tz_localize(None)
            return bars[(local_day != day) | (bars["datetime_utc"] < cutoff)]

        mfm.load_stored_dbn_bars = live_loader
        try:
            fresh = manager.calculate_features(day, day)
        finally:
            mfm.load_stored_dbn_bars = loader
        stored = storage.read_parquet(manager.output_dir() / f"{day:%Y-%m-%d}.parquet")
        bad += compare(stored, fresh, f"minutes {day:%Y-%m-%d} (bars cut at {cutoff.tz_convert(mfm.MARKET_TZ):%H:%M})")
    return bad


def check_daily(days: list[pd.Timestamp]) -> int:
    manager = dfm.DailyFeatureManager(PATHS["daily"])
    storage = Storage(manager.config["storage"], ROOT_DIR)
    write, bad = Storage.write_parquet, 0
    Storage.write_parquet = lambda *args, **kwargs: None
    try:
        for day in days:
            fresh = manager.calculate_features(start_date=day, end_date=day)
            stored = storage.read_parquet(manager.output_dir() / f"{day:%Y-%m-%d}.parquet")
            bad += compare(stored, fresh, f"daily   {day:%Y-%m-%d} (session bars up to that day)")
    finally:
        Storage.write_parquet = write
    return bad


def check_combined(days: list[pd.Timestamp]) -> int:
    train = config(PATHS["train"])
    storage = Storage(train["storage"], ROOT_DIR)
    directory = Path(train["storage"]["input"]["features"]) / train["version"]["features"]
    data = pd.concat([storage.read_parquet(directory / f"{day:%Y-%m-%d}.parquet") for day in days], ignore_index=True)
    bad = 0
    expected = previous_sessions(data["date"])
    wrong_day = (data["daily_feature_date"] != expected) | (data["daily_feature_date"] >= data["date"])
    print(f"  combined: {len(data)} rows; daily features from the previous session: {'yes' if not wrong_day.any() else f'NO ({wrong_day.sum()} rows)'}")
    bad += int(wrong_day.sum())
    non_features = {"date", "symbol", "index_group", "label", "daily_feature_date"}
    suspicious = [c for c in data.columns if c not in non_features and any(word in c.lower() for word in FORBIDDEN)]
    print(f"  combined: model inputs named like labels/returns/cutoff price: {suspicious or 'none'}")
    bad += len(suspicious)
    model_path = ROOT_DIR / train["model"]["path"]
    if model_path.exists():
        import joblib

        inputs = list(joblib.load(model_path).feature_names_in_)
        leaked = [c for c in inputs if c in {"label", "daily_feature_date"} or any(word in c.lower() for word in FORBIDDEN)]
        missing = sorted(set(inputs) - set(data.columns))
        print(f"  model: {len(inputs)} inputs; label-like inputs: {leaked or 'none'}; inputs missing from combined data: {missing[:5] or 'none'}")
        bad += len(leaked) + len(missing)
    return bad


def check_scale_invariance() -> int:
    """Formulas must not change when all prices are x2 and volume /2 (a 2-for-1 split)."""
    rng = np.random.default_rng(0)
    n = 200
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))))
    base = {
        "open": close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.005, n)),
        "high": close * 1.01, "low": close * 0.99, "close": close,
        "vwap": close * (1 + rng.normal(0, 0.002, n)),
        "volume": pd.Series(rng.integers(1e5, 1e6, n).astype(float)),
    }
    scaled = {k: (v * 2 if k in PRICE_FIELDS else v / 2) for k, v in base.items()}
    bad = 0
    for name, path, manager_class in (
        ("daily", PATHS["daily"], dfm.DailyFeatureManager),
        ("minute state", PATHS["minutes"], mfm.MinutesFeatureManager),
    ):
        manager = manager_class(ROOT_DIR / path)
        fields, columns = (manager.feature_fields, manager.feature_columns) if name == "daily" else (manager.state_fields, manager.state_columns)
        a = evaluate_expressions(fields, base, "BASE")
        b = evaluate_expressions(fields, scaled, "SCALED")
        # atol: CORR* use Log($volume + 1), whose +1 moves them by ~1e-6 when volume is halved.
        changed = [col for col, x, y in zip(columns, a, b) if not np.allclose(x.astype(float), y.astype(float), rtol=1e-6, atol=1e-5, equal_nan=True)]
        print(f"  {name} formulas: {len(columns)} checked; depend on price/volume level: {changed or 'none'}")
        bad += len(changed)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=5, help="Number of most recent sessions to check")
    parser.add_argument("--profile", default="basic", choices=sorted(PROFILES), help="Pipeline configs to check (src/live.py PROFILES)")
    args = parser.parse_args()
    PATHS.update(PROFILES[args.profile])
    combined = config(PATHS["combined"])
    storage = Storage(combined["storage"], ROOT_DIR)
    directory = Path(combined["storage"]["output"]["features"]) / combined["version"]["combined_features"]
    days = sorted(pd.Timestamp(Path(p).stem) for p in storage.list(directory, "*.parquet"))[-args.days:]
    daily_days = sorted(set(previous_sessions(pd.Series(days))))
    print(f"Sessions: {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}")
    bad = 0
    print("1. Minute features, live-style recompute")
    bad += check_minutes(days)
    print("2. Daily features, recompute without later sessions")
    bad += check_daily(daily_days)
    print("3. Combined rows and model inputs")
    bad += check_combined(days)
    print("4. Price-level independence")
    bad += check_scale_invariance()
    print("No look-ahead found." if bad == 0 else f"Problems found: {bad} (see above).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
