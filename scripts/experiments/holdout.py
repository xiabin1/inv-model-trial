"""Final check (2026-10-08): every step-2 model on the untouched holdout 2026-08-01..2026-10-06.

1. Extends the expanded-universe data (liquid50 / liquid250) from 2026-10-02 to 2026-10-07: minute
   download, session bars, daily and minute features, labels (to 2026-10-06: a label needs the next
   open), combined data, $250M filter. 2026-10-07 uses the 2026-10-06 lists.
2. Waits for step2.py, then reruns each model's walk-forward with --include-holdout.
3. Compares all models on the holdout, the earlier periods and the whole period.

    python scripts/experiments/holdout.py WORKDIR SECTORS_CSV
"""

from __future__ import annotations

import importlib.util
import shutil
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


l50, l250, s2 = load("liquid50"), load("liquid250"), load("step2")
WORK, S3, log, stage, CONF = l50.WORK, l50.S3, l50.log, l50.stage, l50.CONF
DONE = WORK / "stages_done_holdout.json"
for module in (l50, l250, s2):
    module.DONE = DONE
NEW_START, NEW_END, LABEL_END = "2026-10-05", "2026-10-07", "2026-10-06"
HOLDOUT = "2026-08-01"
MODELS = {  # name: (config file, variant)
    "baseline current": ("train_model_baseline_from2025-06.yaml", s2.BASE_VARIANT),
    "baseline $250M": ("train_model_l250.yaml", s2.BASE_VARIANT),
    **{f"run {run}": (f"train_step2_{run}.yaml", variant) for run, (_, variant, _) in s2.RUNS.items()},
}


def extend_lists() -> None:
    for name in ("l50_v1", "l50_v1_pull", "l250_v1"):
        folder = WORK / "lists" / name
        source = folder / "2026-10-06.csv"
        target = folder / f"{NEW_END}.csv"
        pd.read_csv(source).assign(date=NEW_END).to_csv(target, index=False)
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", str(target), f"{S3}/symbols/{name}/{target.name}"], check=True)


def extend_data() -> None:
    from src.data.features.combined_feature_manager import CombinedFeatureManager
    from src.data.features.daily_feature_manager import DailyFeatureManager
    from src.data.features.label_manager import LabelManager
    from src.data.features.minutes_feature_manager import MinutesFeatureManager
    from src.pull import session_bars as sb

    subprocess.run([sys.executable, "-W", "ignore", "-m", "src.pull.minutes_pull", "--config", str(CONF / "minutes_config.yaml"),
                    "--start-date", NEW_START, "--end-date", NEW_END, "--skip-existing"], cwd=ROOT, check=True, timeout=3600)
    original = sb.load_config
    sb.load_config = lambda path=None: original(CONF / "session_bars_config.yaml")
    sb.main(NEW_START, NEW_END)
    DailyFeatureManager(str(CONF / "daily_feature_config.yaml")).calculate_features(start_date=NEW_START, end_date=NEW_END)
    MinutesFeatureManager(str(CONF / "minutes_feature_config.yaml")).calculate_features(NEW_START, NEW_END)
    LabelManager(str(CONF / "label_config.yaml")).calculate_labels("2026-10-02", LABEL_END)
    CombinedFeatureManager(str(CONF / "combined_feature_config.yaml")).combine("2026-10-02", NEW_END)


def wait_for_step2() -> None:
    while "step2 compare" not in (WORK / "stages_done_step2.json").read_text():
        time.sleep(60)


def walk_forward(model: str) -> None:
    config, variant = MODELS[model]
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / config), "--variant", variant,
                    "--include-holdout"], cwd=ROOT, check=True)


def compare() -> None:
    import yaml

    returns = {}
    for model, (config, variant) in MODELS.items():
        version = yaml.safe_load((CONF / config).read_text())["version"]["evaluation"]
        p = s2.predictions(version, f"{variant}_with_holdout")
        for cap in (None, 4):
            returns[(model, cap)] = s2.daily_returns(p, cap)
    lines = [f"days: {min(r.index.min() for r in returns.values()):%Y-%m-%d}..{max(r.index.max() for r in returns.values()):%Y-%m-%d}; "
             f"holdout from {HOLDOUT} (not used by any earlier test or choice)"]
    periods = {"HOLDOUT 2026-08-01..2026-10-06": (HOLDOUT, None), "independent 2025-06..11": (None, "2025-11-30"),
               "later 2025-12..2026-07": ("2025-12-01", "2026-07-31"), "whole period": (None, None)}
    for period, (a, b) in periods.items():
        rows = {}
        for (model, cap), r in returns.items():
            if a:
                r = r[r.index >= pd.Timestamp(a)]
            if b:
                r = r[r.index <= pd.Timestamp(b)]
            rows[f"{model}{' | max 4/sector' if cap else ''}"] = {**s2.stats(r), "days": len(r)}
        lines += [f"\n--- {period} ---", pd.DataFrame(rows).T.round(2).to_string()]
    base = returns[("baseline current", None)]
    for period, start in (("holdout", HOLDOUT), ("whole period", None)):
        lines.append(f"\n--- gain over the current baseline (as now), {period}: mean, 90% range, chance > 0 ---")
        for (model, cap), r in returns.items():
            if model == "baseline current" and cap is None:
                continue
            diff = (r - base).dropna()
            if start:
                diff = diff[diff.index >= pd.Timestamp(start)]
            low, high, chance = s2.bootstrap(diff)
            lines.append(f"{model + (' | max 4/sector' if cap else ''):34} gain {diff.mean() * 100:+.3f}%/day  "
                         f"90% range [{low:+.3f}, {high:+.3f}]  chance>0 {chance:.0f}%")
    result = WORK / "holdout_result.txt"
    result.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        telegram.send(f"🧪 <b>Holdout check finished</b> (all models, 2026-08-01..10-06)\nDetails: {result}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("holdout 1 lists to 10-07", extend_lists)
        stage("holdout 2 expanded data to 10-07", extend_data)
        stage("holdout 3 $250M data", l250.filter_combined)
        stage("holdout 4 wait for step 2", wait_for_step2)
        for model in MODELS:
            stage(f"holdout walk-forward {model}", lambda model=model: walk_forward(model))
        stage("holdout compare", compare)
        log("ALL DONE holdout")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED holdout: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ Holdout check stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
