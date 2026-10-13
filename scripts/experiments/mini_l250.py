"""Experiment (2026-10-09): "mini + $250M" — the mini model (EQUS.MINI minute features to 15:58) trained
on the b1542_l250 universe (S&P 500 / Nasdaq-100 + liquid US stocks >= $250M), picks from the current
stocks. Reuses the $250M lists, labels (l50_mini_v1) and daily features (l50_v2); adds EQUS.MINI minute
data for the download lists and mini-style minute features. Nothing live is touched.

New data versions: symbols/l250_v1_pull (history), raw/minutes_equs_mini/l250_v1,
features/minutes/l250_mini_v1, features/combined/l250_mini_v1, evaluation/l250_mini_v1.

    python scripts/experiments/mini_l250.py WORKDIR SECTORS_CSV
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/experiments/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


l50, s2 = load("liquid50"), load("step2")
WORK, S3, log, stage, edit, CONF = l50.WORK, l50.S3, l50.log, l50.stage, l50.edit, l50.CONF
l50.DONE = WORK / "stages_done_mini_l250.json"
START_RAW, START_FEAT, START_COMBINED, END = "2024-07-01", "2024-08-01", "2024-10-01", "2026-10-07"
VARIANT = "weekly_plain_label_from2025-06"


def history_lists() -> None:
    """symbols/l250_v1_pull for the past (the live builder writes it from 2026-10-07 on): the $250M list
    plus stocks at >= $125M (known before the day); sessions before 2024-10-01 use the first lists."""
    bars = pd.concat(pd.read_parquet(p) for p in sorted((WORK / "bars2").glob("*.parquet")))
    types = pd.read_parquet(WORK / "types.parquet").set_index("raw_symbol")["security_type"]
    bars = bars[bars["ticker"].map(types).isin(["C", "A"])].copy()
    bars["dv"] = bars["close"] * bars["volume"]
    dv = bars.pivot_table(index="day", columns="ticker", values="dv", aggfunc="sum").sort_index()
    close = bars.pivot_table(index="day", columns="ticker", values="close", aggfunc="last").reindex_like(dv)
    on_way = (dv.rolling(20, min_periods=1).median().shift(1) >= 125e6) & (close.ffill().shift(1) >= 5)
    out_pull, out_list = WORK / "lists/l250_v1_pull", WORK / "lists/l250_v1"
    out_pull.mkdir(parents=True, exist_ok=True)
    for day in dv.index:
        if not (pd.Timestamp(START_COMBINED) <= day <= pd.Timestamp("2026-10-06")):
            continue
        base = pd.read_csv(out_list / f"{day:%Y-%m-%d}.csv")
        extra = sorted({s.replace(".", "-") for s in on_way.columns[on_way.loc[day].to_numpy()]} - set(base["symbol"]))
        frame = pd.concat([base, pd.DataFrame({"date": f"{day:%Y-%m-%d}", "symbol": extra, "index_group": "liquid", "company_name": "Unknown"})])
        frame.to_csv(out_pull / f"{day:%Y-%m-%d}.csv", index=False)
    first_pull, first_list = pd.read_csv(out_pull / f"{START_COMBINED}.csv"), pd.read_csv(out_list / f"{START_COMBINED}.csv")
    for day in pd.bdate_range(START_RAW, "2024-09-30"):
        first_pull.assign(date=f"{day:%Y-%m-%d}").to_csv(out_pull / f"{day:%Y-%m-%d}.csv", index=False)
        first_list.assign(date=f"{day:%Y-%m-%d}").to_csv(out_list / f"{day:%Y-%m-%d}.csv", index=False)
    for name, folder in (("l250_v1_pull", out_pull), ("l250_v1", out_list)):
        # the live builder's own files (2026-10-07 on) are never overwritten
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", "--exclude", "2026-10-07.csv", "--exclude", "2026-10-08.csv",
                        "--exclude", "2026-10-09.csv", str(folder), f"{S3}/symbols/{name}/"], check=True)


def write_configs() -> None:
    def pull(c):
        c["version"].update(data="l250_v1", symbols="l250_v1_pull")
        c["schedule"].update(start_date=START_RAW, end_date=END)
    def minutes(c):
        c["version"].update(features="l250_mini_v1", raw_data="l250_v1", symbols="l250_v1")
        c["schedule"].update(start_date=START_FEAT, end_date=END)
    def combined(c):
        c["version"].update(daily_features="l50_v2", minute_features="l250_mini_v1", labels="l50_mini_v1", combined_features="l250_mini_v1")
        c["schedule"].update(start_date=START_COMBINED, end_date=END)
    def train(c):
        c["version"].update(features="l250_mini_v1", evaluation="l250_mini_v1")
        c["walk_forward"]["variants"] = [v for v in c["walk_forward"]["variants"] if v["name"] == VARIANT]
        c["live"]["model_path"] = "models/experiments/l250_mini_v1.joblib"
        c["model"]["path"] = "models/experiments/l250_mini_v1_single.joblib"
    edit("config/data/minutes_equs_mini_config.yaml", "minutes_equs_mini_l250_config.yaml", pull)
    edit("config/feature/minutes_feature_config_mini.yaml", "minutes_feature_config_mini_l250.yaml", minutes)
    edit("config/feature/combined_feature_config_mini.yaml", "combined_feature_config_mini_l250.yaml", combined)
    edit("config/train/train_model_mini.yaml", "train_model_mini_l250.yaml", train)


def pull_minutes() -> None:
    """One month per process with a time limit and retries (a Databento connection can hang)."""
    for month in pd.date_range(START_RAW, END, freq="MS"):
        last = min(month + pd.offsets.MonthEnd(0), pd.Timestamp(END))
        for attempt in range(1, 4):
            try:
                subprocess.run([sys.executable, "-W", "ignore", "-m", "src.pull.minutes_pull", "--config",
                                str(CONF / "minutes_equs_mini_l250_config.yaml"), "--start-date", f"{month:%Y-%m-%d}",
                                "--end-date", f"{last:%Y-%m-%d}", "--skip-existing"], cwd=ROOT, check=True, timeout=1800,
                               stdout=subprocess.DEVNULL)
                log(f"  minutes {month:%Y-%m} done")
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                log(f"  minutes {month:%Y-%m} attempt {attempt} failed: {type(error).__name__}")
                if attempt == 3:
                    raise


def minute_features() -> None:
    from src.data.features.minutes_feature_manager import MinutesFeatureManager
    MinutesFeatureManager(str(CONF / "minutes_feature_config_mini_l250.yaml")).calculate_features(START_FEAT, END)


def combined() -> None:
    from src.data.features.combined_feature_manager import CombinedFeatureManager
    CombinedFeatureManager(str(CONF / "combined_feature_config_mini_l250.yaml")).combine(START_COMBINED, END)


def walk_forward() -> None:
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / "train_model_mini_l250.yaml"),
                    "--variant", VARIANT, "--include-holdout"], cwd=ROOT, check=True)


def compare() -> None:
    V = f"{VARIANT}_with_holdout"
    models = {"b1542": ("b1542_v1_rerun20261008", None), "b1542_l250 (max 4/sector)": ("l250_b1542_v1", 4),
              "mini": ("mini_v1_rerun20261008", None), "mini | max 4/sector": ("mini_v1_rerun20261008", 4),
              "mini + $250M": ("l250_mini_v1", None), "mini + $250M | max 4/sector": ("l250_mini_v1", 4),
              "mini + $250M | max 3/sector": ("l250_mini_v1", 3)}
    returns = {name: s2.daily_returns(s2.predictions(version, V), cap) for name, (version, cap) in models.items()}
    days = sorted(set.intersection(*(set(r.index) for r in returns.values())))
    returns = {name: r.reindex(days) for name, r in returns.items()}
    lines = [f"common days: {len(days)} ({pd.Timestamp(days[0]):%Y-%m-%d}..{pd.Timestamp(days[-1]):%Y-%m-%d})"]
    periods = {"independent 2025-06..11": (None, "2025-11-30"), "later 2025-12..2026-07": ("2025-12-01", "2026-07-31"),
               "HOLDOUT 2026-08-01..": ("2026-08-01", None), "whole period": (None, None)}
    for period, (a, b) in periods.items():
        rows = {}
        for name, r in returns.items():
            x = r[(r.index >= pd.Timestamp(a)) if a else slice(None)]
            x = x[(x.index <= pd.Timestamp(b)) if b else slice(None)]
            rows[name] = {**s2.stats(x), "days": len(x)}
        lines += [f"\n--- {period} ---", pd.DataFrame(rows).T.round(2).to_string()]
    for base in ("mini", "b1542_l250 (max 4/sector)"):
        lines.append(f"\n--- gain of mini + $250M over {base}, whole period: mean, 90% range, chance > 0 ---")
        for name in ("mini + $250M", "mini + $250M | max 4/sector", "mini + $250M | max 3/sector"):
            diff = (returns[name] - returns[base]).dropna()
            low, high, chance = s2.bootstrap(diff)
            lines.append(f"{name:30} {diff.mean() * 100:+.3f}%/day  90% range [{low:+.3f}, {high:+.3f}]  chance>0 {chance:.0f}%")
    result = WORK / "mini_l250_result.txt"
    result.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        telegram.send(f"🧪 <b>mini + $250M test finished</b>\nDetails: {result}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("mini_l250 1 history lists", history_lists)
        stage("mini_l250 2 configs", write_configs)
        stage("mini_l250 3 EQUS.MINI download", pull_minutes)
        stage("mini_l250 4 minute features", minute_features)
        stage("mini_l250 5 combined data", combined)
        stage("mini_l250 6 walk-forward", walk_forward)
        stage("mini_l250 7 compare", compare)
        log("ALL DONE mini_l250")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED mini_l250: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ mini + $250M test stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
