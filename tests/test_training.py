from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.storage import Storage
from src.models.evaluate import calculate_strategy_metrics, evaluate_model
from src.models.model_trainer import StockModelTrainer, make_labels
from src.models.stock_predictor import StockPredictor


def returns_frame() -> pd.DataFrame:
    # Day 1: market up (every stock > 1%); day 2: market down.
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 5 + ["2026-01-06"] * 5),
            "symbol": list("ABCDE") * 2,
            "label": [0.050, 0.030, 0.020, 0.015, 0.012, -0.010, -0.020, -0.030, -0.040, -0.050],
        }
    )


def test_top_share_labels_best_stocks_each_day():
    labeled = make_labels(returns_frame(), method="top_share", top_share=0.2)
    assert labeled.groupby("date")["label"].sum().tolist() == [1, 1]
    top = labeled[labeled["label"] == 1]
    assert top["symbol"].tolist() == ["A", "A"]  # best of an up day and of a down day
    assert labeled["future_return"].iloc[0] == 0.05
    assert labeled.groupby("date")["excess_return"].median().abs().max() == pytest.approx(0)


def test_threshold_labels_follow_the_market():
    labeled = make_labels(returns_frame(), method="threshold", threshold=0.01)
    assert labeled.groupby("date")["label"].sum().tolist() == [5, 0]


def test_rows_without_return_are_dropped():
    frame = returns_frame()
    frame.loc[0, "label"] = np.nan
    assert len(make_labels(frame)) == 9


def synthetic_features(days: int = 60, symbols: int = 40, seed: int = 0) -> pd.DataFrame:
    """Returns partly driven by `signal`, so a model can learn something."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-05", periods=days)
    frame = pd.DataFrame(
        [(d, f"S{i:02d}") for d in dates for i in range(symbols)], columns=["date", "symbol"]
    )
    frame["signal"] = rng.normal(size=len(frame))
    frame["noise"] = rng.normal(size=len(frame))
    market = frame["date"].map(dict(zip(dates, rng.normal(0, 0.01, days))))
    frame["label"] = market + 0.01 * frame["signal"] + rng.normal(0, 0.01, len(frame))
    frame["index_group"] = "sp500"
    return frame


def manager(data: pd.DataFrame, train_end: str, verify_start: str) -> SimpleNamespace:
    return SimpleNamespace(
        feature_columns=["signal", "noise"],
        label={"method": "top_share", "top_share": 0.2},
        dataset_periods={
            "train": (str(data["date"].min().date()), train_end),
            "verification": (verify_start, str(data["date"].max().date())),
        },
    )


def test_train_split_is_by_date(tmp_path):
    data = synthetic_features()
    fm = manager(data, "2026-02-27", "2026-03-02")
    trainer = StockModelTrainer(model_path=tmp_path / "m.joblib", feature_manager=fm, model_params={"n_estimators": 20})
    metrics = trainer.train_qlib(data, test_share=0.25)
    train_days = data[data["date"] <= "2026-02-27"]["date"].nunique()
    assert metrics["test_rows"] == 40 * (train_days - int(train_days * 0.75))
    assert metrics["auc"] > 0.6  # the model finds `signal`


def test_evaluation_and_top_n_predictions(tmp_path):
    data = synthetic_features()
    fm = manager(data, "2026-02-27", "2026-03-02")
    model_path = tmp_path / "m.joblib"
    StockModelTrainer(model_path=model_path, feature_manager=fm, model_params={"n_estimators": 20}).train_qlib(data)
    storage = Storage({"backend": "local", "local": {"root": str(tmp_path)}})
    metrics = evaluate_model(data, fm, model_path, storage, "evaluation/v1", top_n=5, cost_per_trade=0.001)
    assert metrics["rank_ic_mean"] > 0
    assert metrics["strategy_mean_daily_excess"] > 0  # picks beat the day's median
    assert (tmp_path / "evaluation/v1/ic_daily.csv").exists()
    predictions = StockPredictor(model_path).predict_top(data, "2026-03-02", "2026-03-27", top_n=3)
    assert predictions.groupby("date").size().eq(3).all()
    assert predictions.groupby("date")["rank"].apply(list).map(lambda r: r == [1, 2, 3]).all()


def test_strategy_subtracts_cost_and_compares_market():
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 3 + ["2026-01-06"] * 3),
            "prediction": [0.9, 0.5, 0.1] * 2,
            "future_return": [0.02, 0.0, -0.01, 0.01, 0.0, -0.02],
            "excess_return": [0.02, 0.0, -0.01, 0.01, 0.0, -0.02],
        }
    )
    metrics = calculate_strategy_metrics(scored, top_n=1, cost_per_trade=0.001)
    # Top pick each day: 2% then 1%, minus 0.1% cost each day.
    expected = (1.019 * 1.009) ** (252 / 2) - 1
    assert metrics["strategy_annual_return"] == pytest.approx(expected)
    assert metrics["strategy_hit_rate"] == 1.0
    assert metrics["market_annual_return"] < 0


def test_vol_adjusted_label_does_not_favour_volatile_stocks():
    # CALM moves +/-0.5%, WILD +/-5%, both at random; WILD always has the bigger |return|.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-05", periods=60)
    rows = []
    for d in dates:
        rows.append((d, "CALM", rng.choice([-1, 1]) * 0.005))
        rows.append((d, "WILD", rng.choice([-1, 1]) * 0.05))
        rows.extend((d, f"MID{i}", rng.normal(0, 0.01)) for i in range(8))
    frame = pd.DataFrame(rows, columns=["date", "symbol", "label"])
    plain = make_labels(frame, top_share=0.2)
    adjusted = make_labels(frame, top_share=0.2, vol_adjust=True)
    share = lambda df, s: df.loc[df["symbol"] == s, "label"].mean()
    # Raw ranking: WILD is top-20% on every up day, CALM almost never.
    assert share(plain, "WILD") > 0.4 and share(plain, "CALM") < 0.1
    # Volatility-adjusted: an up day counts the same for both, so their shares are similar.
    assert abs(share(adjusted, "CALM") - share(adjusted, "WILD")) < 0.15
    assert share(adjusted, "CALM") > 0.15


def test_overnight_vol_uses_only_past_returns():
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-05", periods=12), "symbol": "A", "label": [0.01, -0.01] * 5 + [0.5, 0.0]}
    )
    labeled = make_labels(frame, vol_adjust=False)
    # Days 1-10 lack 10 past returns; day 11 uses days 1-10 only, so its own +50% is not in it.
    assert labeled["overnight_vol"].iloc[:10].isna().all()
    assert labeled["overnight_vol"].iloc[10] == pytest.approx(pd.Series([0.01, -0.01] * 5).std())
    assert labeled["overnight_vol"].iloc[11] > 5 * labeled["overnight_vol"].iloc[10]


def test_high_vol_baseline_and_robustness_metrics():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2026-01-05", periods=30)
    scored = pd.DataFrame([(d, f"S{i}") for d in dates for i in range(10)], columns=["date", "symbol"])
    scored["overnight_vol"] = scored["symbol"].str[1:].astype(int)  # S8, S9 = most volatile 20%
    scored["future_return"] = rng.normal(0, 0.01, len(scored)) + np.where(scored["overnight_vol"] >= 8, 0.002, 0)
    scored["excess_return"] = scored["future_return"]
    scored["prediction"] = scored["overnight_vol"]  # the "model" just picks the volatile stocks
    metrics = calculate_strategy_metrics(scored, top_n=2, cost_per_trade=0.0)
    assert metrics["edge_vs_high_vol_mean_daily"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["strategy_mean_daily"] == pytest.approx(metrics["high_vol_mean_daily"])
    assert {"strategy_median_daily", "strategy_trimmed_mean_daily", "strategy_t_stat"} <= set(metrics)
