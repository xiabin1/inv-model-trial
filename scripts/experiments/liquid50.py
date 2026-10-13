"""Experiment (2026-10-08): b1542 model on an expanded universe of liquid US stocks, versus the
current S&P 500 + Nasdaq-100 universe. Runs detached and resumable; nothing live is touched.

Universe rules (each day D, data before D only; EQUS.SUMMARY daily bars, all exchanges):
  stock type common or ADR (Databento security_type C / A), last close >= $5, and
  download list:  median 20-day dollar volume >= $25M
  candidate list: enter >= $50M, leave < $40M, on the download list on >= 15 of the last 20 days,
                  plus all S&P 500 / Nasdaq-100 members of that day.
Data versions (S3, next to the live ones): symbols/l50_v1(_pull), raw/minutes/l50_v1,
raw/session_daily/l50_v1, features/daily/l50_v2, features/minutes/l50_b1542_v1, labels/l50_mini_v1,
features/combined/l50_b1542_v1, evaluation/l50_b1542_v1. The baseline (current universe) is rerun
with today's code into evaluation/b1542_v1_rerun20261008.

    python scripts/experiments/liquid50.py WORKDIR      # WORKDIR holds the EQUS.SUMMARY downloads
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORK = Path(sys.argv[1])
CONF = ROOT / "config/experiments/liquid50"
DONE = WORK / "stages_done.json"
S3 = "s3://portfolio-calc-platform-dev/financial/tp1/data"
START_RAW, START_FEAT, START_COMBINED, END = "2024-07-01", "2024-08-01", "2024-10-01", "2026-10-02"


def log(text: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}", flush=True)


def stage(name: str, run) -> None:
    done = json.loads(DONE.read_text()) if DONE.exists() else []
    if name in done:
        log(f"skip {name} (done)")
        return
    log(f"=== {name}")
    started = time.time()
    run()
    DONE.write_text(json.dumps(done + [name]))
    log(f"=== {name} done in {(time.time() - started) / 60:.0f} min")


# ---------------------------------------------------------------- 1. lists

def build_lists() -> None:
    bars = pd.concat(pd.read_parquet(p) for p in sorted((WORK / "bars2").glob("*.parquet")))
    types = pd.read_parquet(WORK / "types.parquet").set_index("raw_symbol")["security_type"]
    bars = bars[bars["ticker"].map(types).isin(["C", "A"])].copy()
    bars["dv"] = bars["close"] * bars["volume"]
    dv = bars.pivot_table(index="day", columns="ticker", values="dv", aggfunc="sum").sort_index()
    close = bars.pivot_table(index="day", columns="ticker", values="close", aggfunc="last").reindex_like(dv)
    med = dv.rolling(20, min_periods=1).median().shift(1)
    last_close = close.ffill().shift(1)
    download = (med >= 25e6) & (last_close >= 5)
    on_download = download.astype(int).rolling(20, min_periods=1).sum().shift(1).fillna(0)
    enter = (med >= 50e6) & (last_close >= 5) & (on_download >= 15)
    stay = (med >= 40e6) & (last_close >= 5)
    index_files = sorted((WORK / "index").glob("*.csv"))
    index_days = [pd.Timestamp(f.stem) for f in index_files]
    out = WORK / "lists"
    for name in ("l50_v1", "l50_v1_pull"):
        (out / name).mkdir(parents=True, exist_ok=True)
    state = pd.Series(False, index=dv.columns)
    counts = []
    for day in dv.index:
        state = (state & stay.loc[day]) | enter.loc[day]
        if day < pd.Timestamp(START_COMBINED):
            continue
        members = pd.read_csv(index_files[max(i for i, d in enumerate(index_days) if d <= day)])
        members = members.drop_duplicates("symbol").set_index("symbol")["index_group"]
        liquid = {s.replace(".", "-") for s in state.index[state.to_numpy()]}
        pull = {s.replace(".", "-") for s in download.columns[download.loc[day].to_numpy()]}
        for name, symbols in (("l50_v1", liquid | set(members.index)), ("l50_v1_pull", pull | liquid | set(members.index))):
            frame = pd.DataFrame({"date": f"{day:%Y-%m-%d}", "symbol": sorted(symbols)})
            frame["index_group"] = frame["symbol"].map(members).fillna("liquid")
            frame["company_name"] = "Unknown"
            frame.to_csv(out / name / f"{day:%Y-%m-%d}.csv", index=False)
            if day == pd.Timestamp(START_COMBINED):  # earlier sessions (history only) use the first lists
                frame.assign(date=START_RAW).to_csv(out / name / f"{START_RAW}.csv", index=False)
        counts.append({"day": day, "candidates": len(liquid | set(members.index)), "new": len(liquid - set(members.index)),
                       "download": len(pull | liquid | set(members.index))})
    counts = pd.DataFrame(counts).set_index("day")
    counts.to_csv(WORK / "l50_counts.csv")
    log(f"lists: candidates mean {counts.candidates.mean():.0f} (new {counts.new.mean():.0f}), download mean {counts.download.mean():.0f}")
    for name in ("l50_v1", "l50_v1_pull"):
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", str(out / name), f"{S3}/symbols/{name}/"], check=True)


# ---------------------------------------------------------------- 2. configs

def edit(source: str, target: str, change) -> Path:
    config = yaml.safe_load((ROOT / source).read_text())
    change(config)
    CONF.mkdir(parents=True, exist_ok=True)
    path = CONF / target
    path.write_text("# Experiment liquid50 (scripts/experiments/liquid50.py); generated, not used by live jobs\n"
                    + yaml.safe_dump(config, sort_keys=False))
    return path


def write_configs() -> None:
    def pull(c):
        c["version"].update(data="l50_v1", symbols="l50_v1_pull")
        c["schedule"].update(start_date=START_RAW, end_date=END)
    def session(c):
        c["version"].update(raw_minutes="l50_v1", data="l50_v1")
        c["schedule"].update(start_date=START_RAW, end_date=END)
    def daily(c):
        c["version"].update(features="l50_v2", raw_data="l50_v1", symbols="l50_v1")
        c["schedule"].update(start_date=START_FEAT, end_date=END)
    def minutes(c):
        c["version"].update(features="l50_b1542_v1", raw_data="l50_v1", symbols="l50_v1")
        c["schedule"].update(start_date=START_FEAT, end_date=END)
    def labels(c):
        c["version"].update(labels="l50_mini_v1", raw_data="l50_v1")
        c["storage"]["input"]["cutoff_raw"]["version"] = "l50_v1"
        c["schedule"].update(start_date=START_FEAT, end_date="2026-10-01")
    def combined(c):
        c["version"].update(daily_features="l50_v2", minute_features="l50_b1542_v1", labels="l50_mini_v1", combined_features="l50_b1542_v1")
        c["schedule"].update(start_date=START_COMBINED, end_date=END)
    def train(features: str, evaluation: str):
        def change(c):
            c["version"].update(features=features, evaluation=evaluation)
            c["walk_forward"]["variants"] = [v for v in c["walk_forward"]["variants"] if v["name"] == "weekly_plain_label"]
            c["live"]["model_path"] = f"models/experiments/{evaluation}.joblib"  # never the live model
            c["model"]["path"] = f"models/experiments/{evaluation}_single.joblib"
        return change
    edit("config/data/minutes_config.yaml", "minutes_config.yaml", pull)
    edit("config/data/session_bars_config.yaml", "session_bars_config.yaml", session)
    edit("config/feature/daily_feature_config.yaml", "daily_feature_config.yaml", daily)
    edit("config/feature/minutes_feature_config_b1542.yaml", "minutes_feature_config.yaml", minutes)
    edit("config/feature/label_config_mini.yaml", "label_config.yaml", labels)
    edit("config/feature/combined_feature_config_b1542.yaml", "combined_feature_config.yaml", combined)
    edit("config/train/train_model_b1542.yaml", "train_model.yaml", train("l50_b1542_v1", "l50_b1542_v1"))
    edit("config/train/train_model_b1542.yaml", "train_model_baseline.yaml", train("b1542_v1", "b1542_v1_rerun20261008"))


# ---------------------------------------------------------------- 3.. data and models

def pull_minutes() -> None:
    """One month per process with a time limit: a Databento connection can hang without a timeout
    (2026-10-07: all six download threads waited for 2 hours). Saved days are skipped on a retry."""
    for month in pd.date_range(START_RAW, END, freq="MS"):
        last = min(month + pd.offsets.MonthEnd(0), pd.Timestamp(END))
        for attempt in range(1, 4):
            try:
                subprocess.run([sys.executable, "-W", "ignore", "-m", "src.pull.minutes_pull", "--config", str(CONF / "minutes_config.yaml"),
                                "--start-date", f"{month:%Y-%m-%d}", "--end-date", f"{last:%Y-%m-%d}", "--skip-existing"],
                               cwd=ROOT, check=True, timeout=1800, stdout=subprocess.DEVNULL)
                log(f"  minutes {month:%Y-%m} done")
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                log(f"  minutes {month:%Y-%m} attempt {attempt} failed: {type(error).__name__}")
                if attempt == 3:
                    raise


def session_bars() -> None:
    from src.pull import session_bars as sb
    original = sb.load_config
    sb.load_config = lambda path=None: original(CONF / "session_bars_config.yaml")
    sb.main(START_RAW, END)


def daily_features() -> None:
    from src.data.features.daily_feature_manager import DailyFeatureManager
    DailyFeatureManager(str(CONF / "daily_feature_config.yaml")).calculate_features(start_date=START_FEAT, end_date=END)


def minute_features() -> None:
    from src.data.features.minutes_feature_manager import MinutesFeatureManager
    MinutesFeatureManager(str(CONF / "minutes_feature_config.yaml")).calculate_features(START_FEAT, END)


def labels() -> None:
    from src.data.features.label_manager import LabelManager
    LabelManager(str(CONF / "label_config.yaml")).calculate_labels(START_FEAT, "2026-10-01")


def combined() -> None:
    from src.data.features.combined_feature_manager import CombinedFeatureManager
    CombinedFeatureManager(str(CONF / "combined_feature_config.yaml")).combine(START_COMBINED, END)


def walk_forward(config: str) -> None:
    subprocess.run([sys.executable, "-m", "src.walk_forward", "--config", str(CONF / config),
                    "--variant", "weekly_plain_label"], cwd=ROOT, check=True)


# ---------------------------------------------------------------- compare

def stats(daily: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    return {"mean%": daily.mean() * 100, "std%": daily.std() * 100, "sharpe": daily.mean() / daily.std() * np.sqrt(252),
            "total%": (eq.iloc[-1] - 1) * 100, "maxDD%": (eq / eq.cummax() - 1).min() * 100, "worst%": daily.min() * 100,
            "days<-2%": int((daily < -0.02).sum()), "hit%": (daily > 0).mean() * 100}


def compare() -> None:
    def load(version):
        path = WORK / f"{version}.parquet"
        subprocess.run(["aws", "s3", "cp", "--only-show-errors",
                        f"{S3}/evaluation/{version}/walk_forward/weekly_plain_label/predictions.parquet", str(path)], check=True)
        return pd.read_parquet(path).dropna(subset=["future_return"])
    base, wide = load("b1542_v1_rerun20261008"), load("l50_b1542_v1")
    days = sorted(set(base["date"]) & set(wide["date"]))
    base, wide = base[base["date"].isin(days)], wide[wide["date"].isin(days)]
    index_members = {d: set(pd.read_csv(WORK / "lists/l50_v1" / f"{pd.Timestamp(d):%Y-%m-%d}.csv").query("index_group != 'liquid'")["symbol"]) for d in days}
    wide["is_new"] = [s not in index_members[d] for d, s in zip(wide["date"], wide["symbol"])]
    lines = [f"test days: {len(days)} ({pd.Timestamp(days[0]):%Y-%m-%d}..{pd.Timestamp(days[-1]):%Y-%m-%d})"]
    big = wide[wide["future_return"].abs() > 0.4]
    lines.append(f"rows with |overnight| > 40% in the wide test data: {len(big)} (new stocks: {int(big['is_new'].sum())}) "
                 f"- possibly unrecorded splits of new stocks: {big.loc[big['is_new'], ['symbol', 'date', 'future_return']].head(12).to_dict('records')}")
    rows = {}
    for name, frame, new_cost in (("current universe (rerun)", base.assign(is_new=False), 0.001),
                                  ("expanded universe, cost 0.1%", wide, 0.001), ("expanded universe, new stocks cost 0.2%", wide, 0.002),
                                  ("expanded model, picks from current stocks only", wide[~wide["is_new"]], 0.001)):
        for clip in (False, True):
            f = frame[frame["future_return"].abs() <= 0.4] if clip else frame
            top = f.sort_values("prediction", ascending=False).groupby("date").head(10)
            net = top["future_return"] - np.where(top["is_new"], new_cost, 0.001)
            daily = net.groupby(top["date"]).mean()
            rows[name + (" [|move|>40% removed]" if clip else "")] = {**stats(daily), "new picks %": top["is_new"].mean() * 100}
    table = pd.DataFrame(rows).T.round(2)
    lines.append(table.to_string())
    summary = summary_path = WORK / "liquid50_result.txt"
    summary.write_text("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    try:
        from src.notify import telegram
        best = table.iloc[:, :3].to_string()
        telegram.send(f"🧪 <b>Expanded-universe test finished</b> (liquid $50M)\n<pre>{best}</pre>\nDetails: {summary_path}")
    except Exception as error:
        log(f"Telegram: {error}")


if __name__ == "__main__":
    try:
        stage("1 lists", build_lists)
        stage("2 configs", write_configs)
        stage("3 minute download", pull_minutes)
        stage("4 session bars", session_bars)
        stage("5 daily features", daily_features)
        stage("6 minute features", minute_features)
        stage("7 labels", labels)
        stage("8 combined data", combined)
        stage("9 walk-forward expanded", lambda: walk_forward("train_model.yaml"))
        stage("10 walk-forward baseline rerun", lambda: walk_forward("train_model_baseline.yaml"))
        stage("11 compare", compare)
        log("ALL DONE")
    except Exception as error:
        import traceback
        traceback.print_exc()
        log(f"FAILED: {error}")
        try:
            from src.notify import telegram
            telegram.send(f"⚠️ Expanded-universe test stopped: {str(error)[:300]}\nLog: {WORK}/liquid50.log")
        except Exception:
            pass
        raise
