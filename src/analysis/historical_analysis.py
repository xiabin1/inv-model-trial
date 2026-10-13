from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from qlib.config import C
from qlib.data.base import Feature
from qlib.data.ops import Operators, register_all_ops
from qlib.utils import parse_field

from ..data.storage import Storage

ROOT_DIR = Path(__file__).resolve().parents[2]


class HistoricalAnalyzer:
    """Analyze configured expressions across the complete local OHLCV history."""

    def __init__(self, config_path: str | Path = "config/analysis/analysis_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.input_storage = Storage(self.config["input_storage"], ROOT_DIR)
        self.output_storage = Storage(self.config["output_storage"], ROOT_DIR)

    def analyze(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_path = self.config["input_storage"]["data_path"]
        if not self.input_storage.exists(data_path):
            raise FileNotFoundError(f"OHLCV data not found: {self.input_storage.describe(data_path)}")
        data = self.input_storage.read_csv(data_path, parse_dates=["date"])
        if data.empty:
            raise ValueError("OHLCV data is empty")
        if "index_group" not in data:
            data["index_group"] = "unknown"

        start_date = pd.Timestamp(self.config["schedule"]["start_date"])
        end_date = pd.Timestamp(self.config["schedule"]["end_date"])
        analysis_config = self.config["analysis"]
        if data["date"].max() < start_date or data["date"].min() > end_date:
            raise ValueError("Configured analysis period contains no OHLCV data")

        register_all_ops(C)
        event_rows = []
        for symbol, symbol_data in data.groupby("symbol"):
            symbol_data = symbol_data.sort_values("date").reset_index(drop=True)
            values = {
                column: pd.to_numeric(symbol_data[column], errors="coerce")
                for column in ("open", "high", "low", "close", "volume", "vwap")
            }
            values["formula_value"] = self._evaluate_expression(
                analysis_config["formula"], values, symbol
            )
            symbol_data["formula_value"] = values["formula_value"].to_numpy()
            matches = self._matches(symbol_data["formula_value"])
            for row in symbol_data.loc[matches, ["date", "formula_value"]].itertuples(
                index=False
            ):
                if not start_date <= row.date <= end_date:
                    continue
                event_rows.append(
                    {
                        "symbol": symbol,
                        "index_group": symbol_data["index_group"].iloc[0],
                        "date": row.date,
                        "formula": analysis_config["formula"],
                        "value": float(row.formula_value),
                        "threshold": analysis_config["threshold"],
                        "comparison": analysis_config["comparison"],
                    }
                )

        detail = pd.DataFrame(event_rows)
        if detail.empty:
            summary = pd.DataFrame(
                columns=[
                    "symbol", "index_group", "occurrences", "first_date",
                    "last_date", "max_value",
                ]
            )
        else:
            summary = (
                detail.groupby("symbol")
                .agg(
                    index_group=("index_group", "first"),
                    occurrences=("date", "size"),
                    first_date=("date", "min"),
                    last_date=("date", "max"),
                    max_value=("value", "max"),
                )
                .reset_index()
            )
            summary = summary[
                summary["occurrences"] >= analysis_config["minimum_occurrences"]
            ].sort_values(
                ["occurrences", "max_value", "symbol"],
                ascending=[False, False, True],
            ).head(analysis_config["top_n"])

        output_dir = self.config["output_storage"].get("output_dir", ".")
        self.output_storage.write_csv(summary, Path(output_dir) / "summary.csv")
        self.output_storage.write_csv(
            detail.sort_values(["symbol", "date"]), Path(output_dir) / "detail.csv"
        )
        return summary, detail

    def _matches(self, values: pd.Series) -> pd.Series:
        threshold = self.config["analysis"]["threshold"]
        comparison = self.config["analysis"]["comparison"]
        if comparison == "greater":
            return values > threshold
        if comparison == "greater_or_equal":
            return values >= threshold
        if comparison == "less":
            return values < threshold
        if comparison == "less_or_equal":
            return values <= threshold
        raise ValueError(f"Unsupported comparison: {comparison}")

    @staticmethod
    def _evaluate_expression(
        formula: str, values: dict[str, pd.Series], symbol: str
    ) -> pd.Series:
        class PandasFeature(Feature):
            def __init__(self, name: str):
                super().__init__(name)

            def _load_internal(self, instrument, start_index, end_index, *args):
                return values[self._name].iloc[start_index : end_index + 1].copy()

        expression = parse_field(formula).replace("Operators.Feature(", "Feature(")
        result = eval(
            expression,
            {"Feature": PandasFeature, "Operators": Operators},
        ).load(symbol, 0, len(next(iter(values.values()))) - 1, "day")
        return pd.Series(result).reset_index(drop=True)

    @staticmethod
    def _load_config(path: str | Path) -> dict:
        with Path(path).open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        required = {"analysis", "schedule", "input_storage", "output_storage"}
        missing = required.difference(config)
        if missing:
            raise ValueError(f"Analysis configuration is missing: {sorted(missing)}")
        try:
            start = date.fromisoformat(str(config["schedule"]["start_date"]))
            end = date.fromisoformat(str(config["schedule"]["end_date"]))
        except ValueError as error:
            raise ValueError("Analysis dates must use YYYY-MM-DD format") from error
        if start > end:
            raise ValueError("start_date must not exceed end_date")
        if config["analysis"]["top_n"] < 1 or config["analysis"]["minimum_occurrences"] < 1:
            raise ValueError("top_n and minimum_occurrences must be at least 1")
        if config["analysis"]["comparison"] not in {
            "greater",
            "greater_or_equal",
            "less",
            "less_or_equal",
        }:
            raise ValueError("comparison must be a supported comparison name")
        return config


def main() -> None:
    summary, _ = HistoricalAnalyzer().analyze()
    if summary.empty:
        print("No matching events found")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()