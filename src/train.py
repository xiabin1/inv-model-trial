import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from .data.feature_loader import load_feature_data
from .data.storage import Storage
from .models.evaluate import evaluate_model
from .models.model_trainer import StockModelTrainer
from .models.stock_predictor import StockPredictor

ROOT_DIR = Path(__file__).resolve().parents[1]
NON_FEATURE_COLUMNS = {"date", "symbol", "index_group", "label", "daily_feature_date"}


def load_runtime_config(path: str | Path = ROOT_DIR / "config/train/train_model.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for section in ("version", "schedule", "model", "prediction", "evaluation", "storage"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"train config must define a '{section}' object")
    storage = config["storage"]
    for section, name in (("input", "features"), ("output", "evaluation")):
        if not storage.get(section, {}).get(name):
            raise ValueError(f"train storage.{section}.{name} is required")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and score the stock classifier")
    parser.add_argument("--config", default=str(ROOT_DIR / "config/train/train_model.yaml"))
    config = load_runtime_config(parser.parse_args().config)
    storage = Storage(config["storage"], ROOT_DIR)
    schedule = config["schedule"]
    model_config = config["model"]
    if model_config.get("type") != "lightgbm":
        raise ValueError("train.model.type must be 'lightgbm'")

    periods = {
        "train": tuple(schedule["train_period"]),
        "verification": tuple(schedule["verification_period"]),
    }
    first = min(start for start, _ in periods.values())
    last = max(end for _, end in periods.values())
    feature_dir = Path(config["storage"]["input"]["features"]) / config["version"]["features"]
    feature_data = load_feature_data(feature_dir, storage, first, last)
    feature_manager = SimpleNamespace(
        feature_columns=[column for column in feature_data.columns if column not in NON_FEATURE_COLUMNS],
        label=model_config["label"],
        dataset_periods=periods,
    )
    label = model_config["label"]
    print(
        f"Loaded {len(feature_data)} rows, {feature_data['symbol'].nunique()} symbols, "
        f"{len(feature_manager.feature_columns)} features; label: {label['method']}"
        + (
            f" (top {label['top_share']:.0%} per day"
            + (", by excess return / overnight volatility)" if label.get("vol_adjust") else ")")
            if label["method"] == "top_share"
            else f" (> {label['threshold']})"
        )
    )

    trainer = StockModelTrainer(
        model_path=ROOT_DIR / model_config["path"],
        feature_manager=feature_manager,
        model_params=model_config["params"],
    )
    metrics = trainer.train_qlib(feature_data, test_share=model_config["test_share"])
    print(
        f"Trained on {int(metrics['train_rows'])} rows; held-out last {model_config['test_share']:.0%} "
        f"of train days ({int(metrics['test_rows'])} rows): auc {metrics['auc']:.4f}, "
        f"precision {metrics['precision']:.2%}, recall {metrics['recall']:.2%}, "
        f"positive share {metrics['positive_share']:.2%}"
    )

    evaluation = config["evaluation"]
    output_dir = Path(config["storage"]["output"]["evaluation"]) / config["version"]["evaluation"]
    verification = evaluate_model(
        feature_data,
        feature_manager,
        ROOT_DIR / model_config["path"],
        storage,
        output_dir,
        top_n=evaluation["top_n"],
        cost_per_trade=evaluation["cost_per_trade"],
    )
    print(
        f"Verification {periods['verification'][0]}..{periods['verification'][1]} "
        f"({int(verification['verification_rows'])} rows, {int(verification['ic_days'])} days): "
        f"auc {verification['auc']:.4f}, mean IC {verification['ic_mean']:.4f}, "
        f"mean Rank IC {verification['rank_ic_mean']:.4f}, ICIR {verification['ic_ir']:.4f}"
    )
    print(
        f"Top {evaluation['top_n']} per day, after {evaluation['cost_per_trade']:.2%} cost: "
        f"annual {verification['strategy_annual_return']:.2%}, "
        f"sharpe {verification['strategy_sharpe_ratio']:.2f}, "
        f"max drawdown {verification['strategy_max_drawdown']:.2%}, "
        f"days positive {verification['strategy_hit_rate']:.2%}, "
        f"mean daily excess {verification['strategy_mean_daily_excess']:.4%}"
    )
    print(
        f"  daily after cost: mean {verification['strategy_mean_daily']:.3%}, "
        f"median {verification['strategy_median_daily']:.3%}, "
        f"without 3 best/worst days {verification['strategy_trimmed_mean_daily']:.3%}, "
        f"t-stat {verification['strategy_t_stat']:.2f}"
    )
    print(
        f"All stocks (no cost): annual {verification['market_annual_return']:.2%}, "
        f"sharpe {verification['market_sharpe_ratio']:.2f}, "
        f"max drawdown {verification['market_max_drawdown']:.2%}"
    )
    if "high_vol_annual_return" in verification:
        print(
            f"Most volatile 20% of stocks, after cost (a model that only finds volatile stocks): "
            f"annual {verification['high_vol_annual_return']:.2%}, "
            f"sharpe {verification['high_vol_sharpe_ratio']:.2f}, "
            f"mean daily {verification['high_vol_mean_daily']:.3%}"
        )
        print(
            f"  model edge over it per day: mean {verification['edge_vs_high_vol_mean_daily']:.3%}, "
            f"median {verification['edge_vs_high_vol_median_daily']:.3%}, "
            f"without 3 best/worst days {verification['edge_vs_high_vol_trimmed_mean_daily']:.3%}, "
            f"t-stat {verification['edge_vs_high_vol_t_stat']:.2f}"
        )

    prediction = config["prediction"]
    start, end = periods[prediction["period"]]
    predictions = StockPredictor(ROOT_DIR / model_config["path"]).predict_top(
        feature_data, start, end, top_n=prediction["top_n"]
    )
    prediction_path = output_dir / "predictions.csv"
    storage.write_csv(predictions, prediction_path)
    print(f"Saved {len(predictions)} predictions to {storage.describe(prediction_path)}")
    if not predictions.empty:
        latest = predictions["date"].max()
        print(f"\nTop {prediction['top_n']} for {latest:%Y-%m-%d}:")
        print(predictions[predictions["date"] == latest].to_string(index=False))


if __name__ == "__main__":
    main()
