"""Experiment (2026-10-10): is live XNAS.BASIC (~$1,500/month) worth it? Models deciding with XNAS.BASIC
bars up to 15:58 instead of 15:42:
  b1558        current universe (data b1558_v1, extended to 2026-10-09)
  b1558_l250   $250M universe (XNAS.BASIC raw l50_v1, lists l250_v1, labels l50_mini_v1, daily l50_v2)
compared with b1542 / b1542_l250 (15:42, delayed data) and mini / mini_l250 (EQUS.MINI to 15:58, free live).
The decisive comparison is b1558_l250 vs mini_l250: both decide at 15:58, only the data source differs.

    python scripts/experiments/b1558_l250.py WORKDIR SECTORS_CSV
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/experiments/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


l50, s2 = load("liquid50"), load("step2")
WORK, S3, log, stage, edit, CONF = l50.WORK, l50.S3, l50.log, l50.stage, l50.edit, l50.CONF
l50.DONE = WORK / "stages_done_b1558_l250.json"
END = "2026-10-09"
VARIANT = "weekly_plain_label_from2025-06"


def extend_b1558() -> None:
    from src.data.features.combined_feature_manager import CombinedFeatureManager
    from src.data.features.minutes_feature_manager import MinutesFeatureManager
    MinutesFeatureManager(str(ROOT / "config/feature/minutes_feature_config_b1558.yaml")).calculate_features("2026-10-05", END)
    CombinedFeatureManager(str(ROOT / "config/feature/combined_feature_config_b1558.yaml")).combine("2026-10-05", END)


def write_configs() -> None:
    def minutes(c):
        c["version"].update(features="l250_b1558_v1", raw_data="l50_v1", symbols="l250_v1")
        c["schedule"].update(start_date="2024-08-01", end_date=END)
    def combined(c):
        c["version"].update(daily_features="l50_v2", minute_features="l250_b1558_v1", labels="l50_mini_v1", combined_features="l250_b1558_v1")
        c["schedule"].update(start_date="2024-10-01", end_date=END)
    def train(features: str, evaluation: str):
        def change(c):
            c["version"].update(features=features, evaluation=evaluation)
            c["walk_forward"]["variants"] = [v for v in c["walk_forward"]["variants"] if v["name"] == VARIANT] or \
                [{"name": VARIANT, "retrain_every": "week", "train_window": "expanding", "vol_adjust": False, "first_test_date": "2025-06-02"}]
            c["live"]["model_path"] = f"models/experiments/{evaluation}.joblib"
            c["model"]["path"] = f"models/experiments/{evaluation}_single.joblib"
        return change
    edit("config/feature/minutes_feature_config_b1558.yaml", "minutes_feature_config_b1558_l250.yaml", minutes)
    edit("config/feature/combined_feature_config_b1558.yaml", "combined_feature_config_b1558_l250.yaml", combined)
    edit("config/train/train_model_b1558.yaml", "train_model_b1558_l250.yaml", train("l250_b1558_v1", "l250_b1558_v1"))
    edit("config/train/train_model_b1558.yaml", "train_model_b1558_rerun.yaml", train("b1558_v1", "b1558_v1_rerun20261010"))


def l250_features() -> None:
    from src.data.features.combined_feature_manager import CombinedFeatureManager
    from src.data.features.minutes_feature_manager import MinutesFeatureManager
    MinutesFeatureManager(str(CONF / "minutes_feature_config_b1558_l250.yaml")).calculate_features("2024-08-01", END)
    CombinedFeatureManager(str(CONF / "combined_feature_config_b1558_l250.yaml")).combine("2024-10-01", END)


def walk_forward(config: str) -> None:
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / config), "--variant", VARIANT,
                    "--include-holdout"], cwd=ROOT, check=True)


def compare() -> None:
    V = f"{VARIANT}_with_holdout"
    models = {"b1542 (15:42)": ("b1542_v1_rerun20261008", None), "b1558 (XNAS 15:58)": ("b1558_v1_rerun20261010", None),
              "mini (EQUS 15:58)": ("mini_v1_rerun20261008", None),
              "b1542_l250 | max 4": ("l250_b1542_v1", 4), "b1558_l250 | max 4": ("l250_b1558_v1", 4),
              "b1558_l250 | max 3": ("l250_b1558_v1", 3), "b1558_l250 | no cap": ("l250_b1558_v1", None),
              "mini_l250 | max 3": ("l250_mini_v1", 3), "mini_l250 | max 4": ("l250_mini_v1", 4)}
    returns = {name: s2.daily_returns(s2.predictions(version, V), cap) for name, (version, cap) in models.items()}
    days = sorted(set.intersection(*(set(r.index) for r in returns.values())))
    returns = {name: r.reindex(days) for name, r in returns.items()}
    lines = [f"common days: {len(days)} ({pd.Timestamp(days[0]):%Y-%m-%d}..{pd.Timestamp(days[-1]):%Y-%m-%d})"]
    for period, (a, b) in {"independent 2025-06..11": (None, "2025-11-30"), "later 2025-12..2026-07": ("2025-12-01", "2026-07-31"),
                           "HOLDOUT 2026-08-01..": ("2026-08-01", None), "whole period": (None, None)}.items():
        rows = {}
        for name, r in returns.items():
            x = r[(r.index >= pd.Timestamp(a)) if a else slice(None)]
            x = x[(x.index <= pd.Timestamp(b)) if b else slice(None)]
            rows[name] = {**s2.stats(x), "days": len(x)}
        lines += [f"\n--- {period} ---", pd.DataFrame(rows).T.round(2).to_string()]
    lines.append("\n--- value of XNAS.BASIC to 15:58 (whole period): mean daily gain, 90% range, chance > 0 ---")
    for a, b in (("b1558 (XNAS 15:58)", "b1542 (15:42)"), ("b1558 (XNAS 15:58)", "mini (EQUS 15:58)"),
                 ("b1558_l250 | max 4", "b1542_l250 | max 4"), ("b1558_l250 | max 3", "mini_l250 | max 3"),
                 ("b1558_l250 | max 4", "mini_l250 | max 4")):
        diff = (returns[a] - returns[b]).dropna()
        low, high, chance = s2.bootstrap(diff)
        lines.append(f"{a:22} vs {b:20} {diff.mean() * 100:+.3f}%/day  90% range [{low:+.3f}, {high:+.3f}]  chance>0 {chance:.0f}%")
    result = WORK / "b1558_l250_result.txt"
    result.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        telegram.send(f"🧪 <b>XNAS.BASIC 15:58 test finished</b> (is the live plan worth it?)\nDetails: {result}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("b1558 1 extend b1558_v1 to 10-09", extend_b1558)
        stage("b1558 2 configs", write_configs)
        stage("b1558 3 $250M features at 15:58", l250_features)
        stage("b1558 4 walk-forward b1558_l250", lambda: walk_forward("train_model_b1558_l250.yaml"))
        stage("b1558 5 walk-forward b1558 (current universe)", lambda: walk_forward("train_model_b1558_rerun.yaml"))
        stage("b1558 6 compare", compare)
        log("ALL DONE b1558_l250")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED b1558_l250: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ XNAS.BASIC 15:58 test stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
