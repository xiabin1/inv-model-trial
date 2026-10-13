"""Experiment step 2 (2026-10-08): model variants on the 2025-06..2026-07 walk-forward.

Runs (weekly retrain, expanding window, test from 2025-06-02; picks always from the current
S&P 500 / Nasdaq-100 stocks):
  A1-A3  current universe, label as now, randomized trees (80% rows / 80% features), seeds 1-3
  B1-B3  $250M universe (liquid250.py data), same label and randomness, seeds 1-3
  C      current universe, volatility-adjusted label (rank of excess return / typical move)
  D      $250M universe, volatility-adjusted label
Compared with the two existing deterministic baselines (current, $250M), for the pick rules
"as now" and "max 4 per sector", on 2025-06..11 (independent), 2025-12..2026-07 and all 290 days,
plus a block-bootstrap 90% range of the daily gain over the current baseline.

    python scripts/experiments/step2.py WORKDIR SECTORS_CSV
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("liquid50", ROOT / "scripts/experiments/liquid50.py")
l50 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l50)
WORK, S3, log, stage, edit, CONF = l50.WORK, l50.S3, l50.log, l50.stage, l50.edit, l50.CONF
l50.DONE = WORK / "stages_done_step2.json"
SECTORS = pd.read_csv(sys.argv[2]).set_index("symbol")["sector"].to_dict()
BASE_VARIANT = "weekly_plain_label_from2025-06"
VOL_VARIANT = "vol_adjusted_from2025-06"
COST, N = 0.001, 10

RUNS = {  # name: (features version, variant, extra LightGBM params)
    **{f"A{s}": ("b1542_v1", BASE_VARIANT, {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": s}) for s in (1, 2, 3)},
    **{f"B{s}": ("l250_b1542_v1", BASE_VARIANT, {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": s}) for s in (1, 2, 3)},
    "C": ("b1542_v1", VOL_VARIANT, {}),
    "D": ("l250_b1542_v1", VOL_VARIANT, {}),
}
EXISTING = {"baseline current": ("b1542_v1_rerun20261008", BASE_VARIANT), "baseline $250M": ("l250_b1542_v1", BASE_VARIANT)}


def evaluation(run: str) -> str:
    return f"step2_{run}"


def write_configs() -> None:
    for run, (features, variant, params) in RUNS.items():
        def change(c, features=features, variant=variant, params=params, run=run):
            c["version"].update(features=features, evaluation=evaluation(run))
            c["model"]["params"].update(params)
            base = next(v for v in c["walk_forward"]["variants"] if v["name"] == BASE_VARIANT)
            c["walk_forward"]["variants"] = [base] if variant == BASE_VARIANT else [{**base, "name": VOL_VARIANT, "vol_adjust": True}]
            c["live"]["model_path"] = f"models/experiments/{evaluation(run)}.joblib"
            c["model"]["path"] = f"models/experiments/{evaluation(run)}_single.joblib"
        edit("config/train/train_model_b1542.yaml", f"train_step2_{run}.yaml", change)


def walk_forward(run: str) -> None:
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / f"train_step2_{run}.yaml"),
                    "--variant", RUNS[run][1]], cwd=ROOT, check=True)


def predictions(version: str, variant: str) -> pd.DataFrame:
    path = WORK / f"{version}_{variant}.parquet"
    if not path.exists():
        subprocess.run(["aws", "s3", "cp", "--only-show-errors",
                        f"{S3}/evaluation/{version}/walk_forward/{variant}/predictions.parquet", str(path)], check=True)
    p = pd.read_parquet(path).dropna(subset=["future_return"])
    members = {}
    for day in p["date"].unique():
        f = pd.read_csv(WORK / "lists/l250_v1" / f"{pd.Timestamp(day):%Y-%m-%d}.csv")
        members[day] = set(f.loc[f["index_group"] != "liquid", "symbol"])
    return p[[s in members[d] for d, s in zip(p["date"], p["symbol"])]]  # picks from current stocks only


def daily_returns(p: pd.DataFrame, sector_cap: int | None) -> pd.Series:
    out = {}
    for day, g in p.sort_values("prediction", ascending=False).groupby("date"):
        chosen, count = [], {}
        for symbol, ret in zip(g["symbol"], g["future_return"]):
            sector = SECTORS.get(symbol, "?")
            if sector_cap and count.get(sector, 0) >= sector_cap:
                continue
            chosen.append(ret)
            count[sector] = count.get(sector, 0) + 1
            if len(chosen) == N:
                break
        out[day] = np.mean(chosen) - COST
    return pd.Series(out).sort_index()


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    return {"mean%": r.mean() * 100, "sharpe": r.mean() / r.std() * np.sqrt(252), "total%": (eq.iloc[-1] - 1) * 100,
            "maxDD%": (eq / eq.cummax() - 1).min() * 100, "worst%": r.min() * 100, "days<-2%": int((r < -0.02).sum())}


def bootstrap(diff: pd.Series, block: int = 5, reps: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """90% range of the mean daily difference, resampling blocks of `block` consecutive days."""
    rng = np.random.default_rng(seed)
    values = diff.to_numpy()
    starts = np.arange(len(values) - block + 1)
    means = [np.concatenate([values[s:s + block] for s in rng.choice(starts, len(values) // block + 1)])[:len(values)].mean()
             for _ in range(reps)]
    return np.percentile(means, 5) * 100, np.percentile(means, 95) * 100, (np.array(means) > 0).mean() * 100


def compare() -> None:
    sources = {**{name: v for name, v in EXISTING.items()}, **{f"run {r}": (evaluation(r), RUNS[r][1]) for r in RUNS}}
    returns = {(name, cap): daily_returns(predictions(*source), cap) for name, source in sources.items() for cap in (None, 4)}
    lines = []
    for period, (a, b) in {"independent 2025-06..11": (None, "2025-11-30"), "later 2025-12..2026-07": ("2025-12-01", None),
                           "whole 290 days": (None, None)}.items():
        rows = {}
        for (name, cap), r in returns.items():
            r = r[(r.index >= pd.Timestamp(a)) if a else slice(None)]
            r = r[(r.index <= pd.Timestamp(b)) if b else slice(None)]
            rows[f"{name}{' | max 4/sector' if cap else ''}"] = stats(r)
        lines += [f"\n--- {period} ---", pd.DataFrame(rows).T.round(2).to_string()]
    base = returns[("baseline current", None)]
    lines.append("\n--- gain over the current baseline (as now), 290 days: mean, 90% range, chance > 0 ---")
    groups = {"baseline $250M": [("baseline $250M", None)], "A seeds (current, randomized)": [(f"run A{s}", None) for s in (1, 2, 3)],
              "B seeds ($250M, randomized)": [(f"run B{s}", None) for s in (1, 2, 3)], "C (current, vol label)": [("run C", None)],
              "D ($250M, vol label)": [("run D", None)]}
    for label, keys in groups.items():
        for key in keys:
            diff = (returns[key] - base).dropna()
            low, high, chance = bootstrap(diff)
            lines.append(f"{label:32} {key[0]:16} gain {diff.mean() * 100:+.3f}%/day  90% range [{low:+.3f}, {high:+.3f}]  chance>0 {chance:.0f}%")
    result = WORK / "step2_result.txt"
    result.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        telegram.send(f"🧪 <b>Step 2 tests finished</b>\nDetails: {result}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("step2 configs", write_configs)
        for run in RUNS:
            stage(f"step2 walk-forward {run}", lambda run=run: walk_forward(run))
        stage("step2 compare", compare)
        log("ALL DONE step2")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED step2: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ Step 2 tests stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
