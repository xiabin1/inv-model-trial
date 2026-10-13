"""Walk-forward test: retrain on schedule, score the following period, as in live use.

For each test period (a day, a week, a month, or everything for `never`), a fresh model is trained
on the labels that exist before the period starts. A day's label needs the next session's
open, so at the start of a period whose first day is D, the newest usable label is from
two sessions before D (e.g. retraining over the weekend before Monday uses rows up to
Thursday; Friday's label needs Monday's open). All out-of-sample scores are then joined
and evaluated together.

    python -m src.walk_forward                      # all variants in train_model.yaml
    python -m src.walk_forward --variant weekly_expanding
    python -m src.walk_forward --variant weekly_expanding --include-holdout
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from .data.feature_loader import load_feature_data
from .data.storage import Storage
from .models.evaluate import score_report
from .models.model_trainer import make_labels
from .train import NON_FEATURE_COLUMNS, ROOT_DIR, load_runtime_config

RETRAIN_PERIODS = {"week": "W-FRI", "month": "M"}


def test_periods(test_days: list[pd.Timestamp], retrain_every: str) -> list[list[pd.Timestamp]]:
    """Group test days into the periods that share one model."""
    if retrain_every == "never":
        return [test_days]
    if retrain_every == "day":
        return [[day] for day in test_days]
    if retrain_every not in RETRAIN_PERIODS:
        raise ValueError(f"retrain_every must be day, week, month or never, got {retrain_every!r}")
    keys = pd.DatetimeIndex(test_days).to_period(RETRAIN_PERIODS[retrain_every])
    groups: dict = {}
    for day, key in zip(test_days, keys):
        groups.setdefault(key, []).append(day)
    return list(groups.values())


def training_rows(
    labeled: pd.DataFrame, all_days: list[pd.Timestamp], first_test_day: pd.Timestamp, train_window: str | int
) -> pd.DataFrame:
    """Rows whose label is known before `first_test_day`'s decision, optionally the last N days."""
    position = all_days.index(first_test_day)
    if position < 2:
        raise ValueError(f"No labeled history before {first_test_day.date()}")
    last_known = all_days[position - 2]  # its label ended at the previous session's open
    rows = labeled[labeled["date"] <= last_known]
    if train_window != "expanding":
        window_days = [day for day in all_days if day <= last_known][-int(train_window):]
        rows = rows[rows["date"] >= window_days[0]]
    return rows


# Overnight moves beyond +-40% are almost always a split or spin-off missing from
# config/symbols/corporate_actions.yaml (CTVA read -81%). Such rows are left out of the training
# data only: test days keep every row, so the backtest still sees real crashes and jumps.
MAX_TRAIN_MOVE = 0.40


def trainable(rows: pd.DataFrame, max_move: float = MAX_TRAIN_MOVE) -> pd.DataFrame:
    """Training rows without suspicious overnight returns (|future_return| > max_move)."""
    keep = rows["future_return"].abs() <= max_move
    if not keep.all():
        dropped = rows.loc[~keep, ["symbol", "date", "future_return"]]
        sample = ", ".join(f"{r.symbol} {pd.Timestamp(r.date):%Y-%m-%d} {r.future_return:+.0%}" for r in dropped.head(5).itertuples())
        print(f"  training: left out {len(dropped)} row(s) with an overnight move beyond +-{max_move:.0%} ({sample})")
    return rows[keep]


def variant_label(config: dict, variant: dict) -> dict:
    """The model label settings with the variant's overrides."""
    label_keys = ("method", "top_share", "threshold", "vol_adjust", "vol_window")
    return {**config["model"]["label"], **{k: variant[k] for k in label_keys if k in variant}}


def label_rows(data: pd.DataFrame, label: dict) -> pd.DataFrame:
    """`make_labels` with the settings from `variant_label` (rows without a label are dropped)."""
    return make_labels(
        data,
        method=label.get("method", "top_share"),
        top_share=float(label.get("top_share", 0.2)),
        threshold=float(label.get("threshold", 0.01)),
        vol_adjust=bool(label.get("vol_adjust", False)),
        vol_window=int(label.get("vol_window", 20)),
    )


def run_variant(
    name: str,
    variant: dict,
    data: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    storage: Storage,
    include_holdout: bool,
) -> dict:
    walk = config["walk_forward"]
    label = variant_label(config, variant)
    labeled = label_rows(data, label)
    all_days = sorted(pd.Timestamp(day) for day in data["date"].unique())
    first = pd.Timestamp(variant.get("first_test_date", walk["first_test_date"]))
    holdout = pd.Timestamp(walk["final_holdout_from"])
    last = pd.Timestamp(variant.get("last_test_date", walk.get("last_test_date", all_days[-1])))
    test_days = [day for day in all_days if first <= day <= last and (include_holdout or day < holdout)]
    retrain_every = variant.get("retrain_every", "week")
    train_window = variant.get("train_window", "expanding")
    periods = test_periods(test_days, retrain_every)
    print(
        f"\n=== {name}: retrain {retrain_every}, window {train_window}, "
        f"label {label.get('method', 'top_share')}"
        + (
            f" > {label.get('threshold')}"
            if label.get("method") == "threshold"
            else f" vol_adjust={label.get('vol_adjust', False)}"
        )
        + f"; {len(periods)} model(s), "
        f"{len(test_days)} test days {test_days[0].date()}..{test_days[-1].date()} ==="
    )

    scored_parts, weekly = [], []
    started = time.perf_counter()
    for number, days in enumerate(periods, 1):
        train = trainable(training_rows(labeled, all_days, days[0], train_window),
                          float(config["model"].get("max_train_move", MAX_TRAIN_MOVE)))
        model = LGBMClassifier(**config["model"]["params"])
        model.fit(train[feature_columns], train["label"].astype(int))
        test = labeled[labeled["date"].isin(days)].copy()
        if test.empty:
            continue
        test["prediction"] = model.predict_proba(test[feature_columns])[:, list(model.classes_).index(1)]
        scored_parts.append(test)
        day_ic = test.groupby("date").apply(lambda x: x["prediction"].corr(x["future_return"], method="spearman"))
        top10 = test.sort_values(["date", "prediction"], ascending=[True, False]).groupby("date").head(10)
        weekly.append(
            {
                "period_start": days[0].date(),
                "period_end": days[-1].date(),
                "train_from": train["date"].min().date(),
                "train_to": train["date"].max().date(),
                "train_rows": len(train),
                "test_days": len(days),
                "rank_ic": float(day_ic.mean()),
                "top10_mean_return": float(top10.groupby("date")["future_return"].mean().mean()),
            }
        )
        if number % 25 == 0 or number == len(periods):
            print(f"  model {number}/{len(periods)} ({time.perf_counter() - started:.0f}s)", flush=True)

    scored = pd.concat(scored_parts, ignore_index=True)
    evaluation = config["evaluation"]
    metrics, daily_ic, deciles = score_report(scored, walk.get("top_n", [10, 30, 50]), evaluation["cost_per_trade"])
    metrics.update(
        {
            "variant": name,
            "retrain_every": retrain_every,
            "train_window": str(train_window),
            "label_method": label.get("method", "top_share"),
            "vol_adjust": bool(label.get("vol_adjust", False)),
            "models": len(periods),
            "test_from": str(test_days[0].date()),
            "test_to": str(test_days[-1].date()),
            "includes_holdout": include_holdout,
            "seconds": round(time.perf_counter() - started, 1),
        }
    )
    output = (
        Path(config["storage"]["output"]["evaluation"])
        / config["version"]["evaluation"]
        / "walk_forward"
        / (f"{name}_with_holdout" if include_holdout else name)
    )
    columns = ["date", "symbol", "prediction", "label", "future_return", "excess_return", "overnight_vol"]
    storage.write_parquet(scored[columns], output / "predictions.parquet")
    storage.write_csv(pd.DataFrame(weekly), output / "periods.csv")
    storage.write_csv(daily_ic, output / "daily_ic.csv")
    storage.write_csv(deciles.reset_index(), output / "deciles.csv")
    storage.write_bytes(output / "summary.json", json.dumps(metrics, indent=2).encode(), "application/json")
    print_summary(metrics, deciles, walk.get("top_n", [10, 30, 50]))
    print(f"  saved to {storage.describe(output)}")
    return metrics


def print_summary(metrics: dict, deciles: pd.DataFrame, top_ns: list[int]) -> None:
    print(
        f"  Rank IC {metrics['rank_ic_mean']:.4f} (t {metrics['rank_ic_t_stat']:.2f}, ICIR/yr "
        f"{metrics['rank_ic_ir_annual']:.2f}, positive months {metrics['rank_ic_positive_months']:.0%}), "
        f"AUC {metrics.get('auc', float('nan')):.4f}, decile spread {metrics['decile_spread_excess']:.3%}/day"
    )
    print("  excess return by score decile:", " ".join(f"{v:+.3%}" for v in deciles["mean_excess"]))
    for n in top_ns:
        print(
            f"  top{n}: annual {metrics[f'top{n}_strategy_annual_return']:.1%}, sharpe "
            f"{metrics[f'top{n}_strategy_sharpe_ratio']:.2f}, maxDD {metrics[f'top{n}_strategy_max_drawdown']:.1%}, "
            f"daily mean/median/trimmed {metrics[f'top{n}_strategy_mean_daily']:.3%}/"
            f"{metrics[f'top{n}_strategy_median_daily']:.3%}/{metrics[f'top{n}_strategy_trimmed_mean_daily']:.3%}, "
            f"t {metrics[f'top{n}_strategy_t_stat']:.2f} | vs high-vol: {metrics[f'top{n}_edge_vs_high_vol_mean_daily']:.3%} "
            f"(t {metrics[f'top{n}_edge_vs_high_vol_t_stat']:.2f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward test with scheduled retraining")
    parser.add_argument("--config", default=str(ROOT_DIR / "config/train/train_model.yaml"))
    parser.add_argument("--variant", action="append", help="Run only these variant names (repeatable)")
    parser.add_argument("--include-holdout", action="store_true", help="Also test the final holdout period")
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    walk = config.get("walk_forward")
    if not isinstance(walk, dict) or not walk.get("variants"):
        raise ValueError("train config needs a walk_forward section with variants")
    variants = {variant["name"]: variant for variant in walk["variants"]}
    names = args.variant or list(variants)
    unknown = set(names) - set(variants)
    if unknown:
        raise ValueError(f"Unknown variant(s) {sorted(unknown)}; available: {sorted(variants)}")

    storage = Storage(config["storage"], ROOT_DIR)
    feature_dir = Path(config["storage"]["input"]["features"]) / config["version"]["features"]
    data = load_feature_data(feature_dir, storage)
    feature_columns = [column for column in data.columns if column not in NON_FEATURE_COLUMNS]
    print(f"Loaded {len(data)} rows, {data['date'].nunique()} days, {len(feature_columns)} features")
    results = [
        run_variant(name, variants[name], data, feature_columns, config, storage, args.include_holdout)
        for name in names
    ]
    print("\n=== comparison ===")
    keys = ["rank_ic_mean", "rank_ic_t_stat", "top10_strategy_mean_daily", "top10_strategy_t_stat",
            "top30_strategy_mean_daily", "top30_strategy_t_stat", "top50_strategy_mean_daily", "top50_strategy_t_stat"]
    print(pd.DataFrame(results).set_index("variant")[[k for k in keys if k in results[0]]].round(5).to_string())


if __name__ == "__main__":
    main()
