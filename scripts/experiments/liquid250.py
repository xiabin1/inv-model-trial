"""Experiment (2026-10-08), follow-up of liquid50.py: universe = S&P 500 / Nasdaq-100 members plus
liquid US stocks with a median 20-day dollar volume >= $250M (leave below $200M; on the >= $125M
list on 15 of the last 20 days; common stock or ADR, price >= $5). The $250M level was chosen from
the liquid50 results (2025-12..2026-07), so the walk-forward starts in 2025-06: 2025-06..2025-11 is
the independent check. The data is the liquid50 data filtered to this universe (no new download).

    python scripts/experiments/liquid250.py WORKDIR      # same WORKDIR as liquid50.py
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
WORK, S3, log, stage, edit, stats = l50.WORK, l50.S3, l50.log, l50.stage, l50.edit, l50.stats
l50.DONE = WORK / "stages_done_l250.json"
CONF = l50.CONF
VARIANT = "weekly_plain_label_from2025-06"


def build_lists() -> None:
    bars = pd.concat(pd.read_parquet(p) for p in sorted((WORK / "bars2").glob("*.parquet")))
    types = pd.read_parquet(WORK / "types.parquet").set_index("raw_symbol")["security_type"]
    bars = bars[bars["ticker"].map(types).isin(["C", "A"])].copy()
    bars["dv"] = bars["close"] * bars["volume"]
    dv = bars.pivot_table(index="day", columns="ticker", values="dv", aggfunc="sum").sort_index()
    close = bars.pivot_table(index="day", columns="ticker", values="close", aggfunc="last").reindex_like(dv)
    med = dv.rolling(20, min_periods=1).median().shift(1)
    last_close = close.ffill().shift(1)
    on_list = ((med >= 125e6) & (last_close >= 5)).astype(int).rolling(20, min_periods=1).sum().shift(1).fillna(0)
    enter = (med >= 250e6) & (last_close >= 5) & (on_list >= 15)
    stay = (med >= 200e6) & (last_close >= 5)
    out = WORK / "lists/l250_v1"
    out.mkdir(parents=True, exist_ok=True)
    state = pd.Series(False, index=dv.columns)
    for day in dv.index:
        state = (state & stay.loc[day]) | enter.loc[day]
        if day < pd.Timestamp(l50.START_COMBINED):
            continue
        base = pd.read_csv(WORK / "lists/l50_v1" / f"{day:%Y-%m-%d}.csv")
        members = base[base["index_group"] != "liquid"]
        liquid = {s.replace(".", "-") for s in state.index[state.to_numpy()]} - set(members["symbol"])
        frame = pd.concat([members, pd.DataFrame({"date": f"{day:%Y-%m-%d}", "symbol": sorted(liquid),
                                                  "index_group": "liquid", "company_name": "Unknown"})])
        frame.to_csv(out / f"{day:%Y-%m-%d}.csv", index=False)
    subprocess.run(["aws", "s3", "sync", "--only-show-errors", str(out), f"{S3}/symbols/l250_v1/"], check=True)


def filter_combined() -> None:
    """features/combined/l250_b1542_v1 = the liquid50 rows of the stocks on each day's $250M list."""
    from src.data.storage import Storage
    import yaml

    config = yaml.safe_load((CONF / "combined_feature_config.yaml").read_text())
    storage = Storage(config["storage"], ROOT)
    source, target = Path("features/combined/l50_b1542_v1"), Path("features/combined/l250_b1542_v1")
    missing = 0
    for path in sorted(storage.list(source, "*.parquet")):
        day = Path(path).stem
        keep = set(pd.read_csv(WORK / "lists/l250_v1" / f"{day}.csv")["symbol"])
        data = storage.read_parquet(source / Path(path).name)
        missing += len(keep - set(data["symbol"]))
        storage.write_parquet(data[data["symbol"].isin(keep)], target / Path(path).name)
    log(f"combined l250: {missing} list entries without liquid50 rows (summed over days)")


def write_configs() -> None:
    def train(features: str, evaluation: str):
        def change(c):
            c["version"].update(features=features, evaluation=evaluation)
            c["walk_forward"]["variants"] = [v for v in c["walk_forward"]["variants"] if v["name"] == VARIANT]
            c["live"]["model_path"] = f"models/experiments/{evaluation}.joblib"
            c["model"]["path"] = f"models/experiments/{evaluation}_single.joblib"
        return change
    edit("config/train/train_model_b1542.yaml", "train_model_l250.yaml", train("l250_b1542_v1", "l250_b1542_v1"))
    edit("config/train/train_model_b1542.yaml", "train_model_baseline_from2025-06.yaml", train("b1542_v1", "b1542_v1_rerun20261008"))


def walk_forward(config: str) -> None:
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / config), "--variant", VARIANT],
                   cwd=ROOT, check=True)


def compare() -> None:
    def load(version):
        path = WORK / f"{version}_{VARIANT}.parquet"
        subprocess.run(["aws", "s3", "cp", "--only-show-errors",
                        f"{S3}/evaluation/{version}/walk_forward/{VARIANT}/predictions.parquet", str(path)], check=True)
        return pd.read_parquet(path).dropna(subset=["future_return"])

    base, wide = load("b1542_v1_rerun20261008"), load("l250_b1542_v1")
    days = sorted(set(base["date"]) & set(wide["date"]))
    base, wide = base[base["date"].isin(days)], wide[wide["date"].isin(days)].copy()
    members = {d: set(pd.read_csv(WORK / "lists/l250_v1" / f"{pd.Timestamp(d):%Y-%m-%d}.csv").query("index_group != 'liquid'")["symbol"])
               for d in days}
    wide["is_new"] = [s not in members[d] for d, s in zip(wide["date"], wide["symbol"])]
    big = wide[wide["future_return"].abs() > 0.4]
    lines = [f"test days: {len(days)} ({pd.Timestamp(days[0]):%Y-%m-%d}..{pd.Timestamp(days[-1]):%Y-%m-%d}); "
             f"independent check = before 2025-12-01 (the $250M level was chosen on 2025-12..2026-07)",
             f"new-stock rows with |overnight| > 40%: {int(big['is_new'].sum())} "
             f"{big.loc[big['is_new'], ['symbol', 'date', 'future_return']].round(3).to_dict('records')}"]
    setups = (("current universe", base.assign(is_new=False)), ("$250M universe", wide),
              ("$250M model, picks from current stocks", wide[~wide["is_new"]]))
    for period, start, end in (("independent check 2025-06..2025-11", None, "2025-11-30"),
                               ("tuning period 2025-12..2026-07", "2025-12-01", None), ("whole period", None, None)):
        rows = {}
        for name, frame in setups:
            f = frame
            if start:
                f = f[f["date"] >= pd.Timestamp(start)]
            if end:
                f = f[f["date"] <= pd.Timestamp(end)]
            top = f.sort_values("prediction", ascending=False).groupby("date").head(10)
            daily = (top["future_return"] - 0.001).groupby(top["date"]).mean()
            rows[name] = {**stats(daily), "days": len(daily), "new picks %": top["is_new"].mean() * 100}
        lines += [f"\n--- {period} ---", pd.DataFrame(rows).T.round(2).to_string()]
    result = WORK / "liquid250_result.txt"
    result.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        telegram.send(f"🧪 <b>$250M universe test finished</b>\nDetails: {result}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("l250 1 lists", build_lists)
        stage("l250 2 combined data", filter_combined)
        stage("l250 3 configs", write_configs)
        stage("l250 4 walk-forward $250M", lambda: walk_forward("train_model_l250.yaml"))
        stage("l250 5 walk-forward baseline from 2025-06", lambda: walk_forward("train_model_baseline_from2025-06.yaml"))
        stage("l250 6 compare", compare)
        log("ALL DONE l250")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED l250: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ $250M universe test stopped: {str(error)[:300]}")
        except Exception:
            pass
        raise
