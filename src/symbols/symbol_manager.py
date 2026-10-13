from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import yaml

from ..data.storage import Storage


IndexName = Literal["sp500", "nasdaq100", "both"]


class SymbolManager:
    """Load current index constituents and select reproducible random samples."""

    INDEX_URLS = {
        "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "nasdaq100": "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
    }

    # Wikipedia pages that maintain a dated addition/removal change log, used to
    # reconstruct point-in-time (as-of) index membership from the current list.
    CHANGE_LOG_URLS = {
        "sp500": "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
        "nasdaq100": "https://en.wikipedia.org/wiki/Historical_components_of_the_Nasdaq-100",
    }

    def __init__(self, random_state: int | None = 42):
        self.random_state = random_state
        self._change_log_cache: dict[str, pd.DataFrame] = {}
        self._current_symbols_cache: dict[str, list[str]] = {}

    @staticmethod
    def load_config(path: str = "config/symbols/symbols_config.yaml") -> dict:
        with open(path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        required = {"version", "selection", "storage"}
        missing = required.difference(config or {})
        if missing:
            raise ValueError(f"symbol configuration is missing: {sorted(missing)}")
        selection = config["selection"]
        if selection["universe"] not in {"sp500", "nasdaq100", "both"}:
            raise ValueError("universe must be sp500, nasdaq100, or both")
        symbol_count = selection.get("symbol_count")
        if symbol_count != "all" and (
            not isinstance(symbol_count, int) or isinstance(symbol_count, bool) or symbol_count < 1
        ):
            raise ValueError("symbol_count must be a positive integer or 'all'")
        if not isinstance(config["version"].get("data"), str) or not config["version"]["data"].strip():
            raise ValueError("version must be a non-empty string")
        return config

    def get_symbols(self, index: Literal["sp500", "nasdaq100"]) -> list[str]:
        """Return today's constituents for an index, cached in memory for this instance."""
        if index in self._current_symbols_cache:
            return self._current_symbols_cache[index]
        if index == "sp500":
            tables = self._read_tables(self.INDEX_URLS[index])
            table = next(
                (candidate for candidate in tables if "Symbol" in candidate.columns),
                None,
            )
            if table is None:
                raise ValueError("Could not find the S&P 500 ticker table")
            symbols = table["Symbol"].tolist()
        else:
            response = self._get_json(self.INDEX_URLS[index])
            rows = response.get("data", {}).get("data", {}).get("rows", [])
            symbols = [row["symbol"] for row in rows if row.get("symbol")]
            if not symbols:
                raise ValueError("Nasdaq API returned no Nasdaq-100 symbols")
        normalized = self._normalize_symbols(symbols)
        self._current_symbols_cache[index] = normalized
        return normalized

    def get_company_names(self, index: IndexName = "both") -> dict[str, str]:
        """Return the current company name for each symbol in the selected index universe."""
        indexes = ("sp500", "nasdaq100") if index == "both" else (index,)
        names: dict[str, str] = {}
        for index_name in indexes:
            if index_name == "sp500":
                tables = self._read_tables(self.INDEX_URLS[index_name])
                table = next(
                    (
                        candidate
                        for candidate in tables
                        if "Symbol" in candidate.columns and "Security" in candidate.columns
                    ),
                    None,
                )
                if table is None:
                    raise ValueError("Could not find S&P 500 company names")
                names.update(
                    {
                        symbol: str(company)
                        for symbol, company in zip(table["Symbol"], table["Security"])
                    }
                )
            else:
                response = self._get_json(self.INDEX_URLS[index_name])
                rows = response.get("data", {}).get("data", {}).get("rows", [])
                names.update(
                    {
                        self._normalize_symbols([row["symbol"]])[0]: str(row["name"])
                        for row in rows
                        if row.get("symbol") and row.get("name")
                    }
                )
        return names

    def get_index_groups(
        self, universe: IndexName = "both", as_of: str | pd.Timestamp | None = None
    ) -> dict[str, str]:
        """Return each symbol's index membership for the selected universe."""
        indexes = ("sp500", "nasdaq100") if universe == "both" else (universe,)
        memberships: dict[str, set[str]] = {}
        for index_name in indexes:
            symbols = self.get_symbols_as_of(index_name, as_of) if as_of else self.get_symbols(index_name)
            for symbol in symbols:
                memberships.setdefault(symbol, set()).add(index_name)
        return {
            symbol: "&".join(
                name for name in ("sp500", "nasdaq100") if name in groups
            )
            for symbol, groups in memberships.items()
        }

    @staticmethod
    def _read_tables(url: str) -> list[pd.DataFrame]:
        return pd.read_html(StringIO(SymbolManager._get_text(url)))

    @staticmethod
    def _get_text(url: str) -> str:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _get_json(url: str) -> dict:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def select_random(
        self,
        n: int | str,
        universe: IndexName = "both",
        as_of: str | pd.Timestamp | None = None,
    ) -> list[str]:
        """Select N symbols from one index or the combined index universe.

        The universe can be one index or the combined S&P 500/Nasdaq-100 list.
        If `as_of` is given, sample from the reconstructed point-in-time
        membership for that date instead of today's constituents.
        """
        universe_symbols = (
            self.get_universe_as_of(as_of, universe) if as_of else self._current_universe(universe)
        )
        if n == "all":
            return sorted(set(universe_symbols))
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer or 'all'")
        return self._sample(universe_symbols, n)

    def _current_universe(self, universe: IndexName) -> list[str]:
        indexes = ("sp500", "nasdaq100") if universe == "both" else (universe,)
        symbols_by_index = {name: self.get_symbols(name) for name in indexes}
        return list(
            dict.fromkeys(symbol for symbols in symbols_by_index.values() for symbol in symbols)
        )

    def get_membership_changes(self, index: Literal["sp500", "nasdaq100"]) -> pd.DataFrame:
        """Return the (date, added, removed) change log for an index, oldest first.

        Results are cached in memory and, once fetched, persisted under
        `data/symbols/membership/<index>_changes.csv` so re-running as-of
        queries does not re-scrape Wikipedia every time.
        """
        if index in self._change_log_cache:
            return self._change_log_cache[index]
        cache_path = Path(f"data/symbols/membership/{index}_changes.csv")
        storage = Storage(root=Path(__file__).resolve().parents[2])
        if storage.exists(cache_path):
            changes = storage.read_csv(cache_path, parse_dates=["date"])
        else:
            tables = self._read_tables(self.CHANGE_LOG_URLS[index])
            changes = self._normalize_change_log(self._find_change_log_table(tables))
            storage.write_csv(changes, cache_path)
        self._change_log_cache[index] = changes
        return changes

    def get_symbols_as_of(self, index: Literal["sp500", "nasdaq100"], as_of: str | pd.Timestamp) -> list[str]:
        """Reconstruct index constituents as of a past date.

        Starts from today's constituents and reverses every change that
        happened after `as_of`: a symbol added after that date is removed
        (it was not yet a member), and a symbol removed after that date is
        added back (it was still a member on that date).
        """
        as_of_ts = pd.Timestamp(as_of)
        current = set(self.get_symbols(index))
        changes = self.get_membership_changes(index)
        later_changes = changes[changes["date"] > as_of_ts].sort_values("date", ascending=False)
        for _, row in later_changes.iterrows():
            added, removed = row["added"], row["removed"]
            if isinstance(added, str) and added:
                current.discard(added)
            if isinstance(removed, str) and removed:
                current.add(removed)
        return sorted(current)

    def get_universe_as_of(
        self, as_of: str | pd.Timestamp, universe: IndexName = "both"
    ) -> list[str]:
        """Return the combined point-in-time constituents for one or both indexes."""
        indexes = ("sp500", "nasdaq100") if universe == "both" else (universe,)
        symbols: set[str] = set()
        for name in indexes:
            symbols.update(self.get_symbols_as_of(name, as_of))
        return sorted(symbols)

    @staticmethod
    def _find_change_log_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
        for table in tables:
            columns = [SymbolManager._flatten_column(col) for col in table.columns]
            if (
                any("date" in col for col in columns)
                and any("added" in col and "ticker" in col for col in columns)
                and any("removed" in col and "ticker" in col for col in columns)
            ):
                table = table.copy()
                table.columns = columns
                return table
        raise ValueError("Could not find an index membership change-log table")

    @staticmethod
    def _flatten_column(column) -> str:
        if isinstance(column, tuple):
            return " ".join(str(part) for part in column).strip().lower()
        return str(column).strip().lower()

    @staticmethod
    def _normalize_change_log(table: pd.DataFrame) -> pd.DataFrame:
        date_col = next(col for col in table.columns if "date" in col)
        added_col = next(
            (col for col in table.columns if "added" in col and "ticker" in col),
            next(col for col in table.columns if "added" in col),
        )
        removed_col = next(
            (col for col in table.columns if "removed" in col and "ticker" in col),
            next(col for col in table.columns if "removed" in col),
        )
        changes = table[[date_col, added_col, removed_col]].copy()
        changes.columns = ["date", "added", "removed"]
        changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
        changes = changes.dropna(subset=["date"])
        changes["added"] = changes["added"].map(SymbolManager._extract_ticker)
        changes["removed"] = changes["removed"].map(SymbolManager._extract_ticker)
        return changes.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _extract_ticker(value: object) -> str | None:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "—", "-", ""}:
            return None
        normalized = SymbolManager._normalize_symbols([text])
        return normalized[0] if normalized else None

    def _sample(self, symbols: list[str], n: int) -> list[str]:
        if n > len(symbols):
            raise ValueError(f"Requested {n} symbols, but only {len(symbols)} are available")
        return (
            pd.Series(symbols)
            .sample(n=n, random_state=self.random_state)
            .sort_values()
            .tolist()
        )

    def generate_symbols(self, config: dict, on_date: str | pd.Timestamp) -> pd.DataFrame:
        """Select and save a deduplicated point-in-time symbol file."""
        selection = config["selection"]
        day = pd.Timestamp(on_date).normalize()
        symbols = self.select_random(
            selection["symbol_count"], selection["universe"], as_of=day
        )
        groups = self.get_index_groups(selection["universe"], as_of=day)
        names = self.get_company_names(selection["universe"])
        result = pd.DataFrame({"date": day.strftime("%Y-%m-%d"), "symbol": symbols})
        result["index_group"] = result["symbol"].map(groups).fillna("unknown")
        result["company_name"] = result["symbol"].map(names).fillna("Unknown")
        result = result.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        storage = Storage(config["storage"], Path(__file__).resolve().parents[2])
        path = self.symbol_snapshot_path(
            self.storage_base_dir(config), config["version"]["data"], day
        )
        storage.write_csv(result, path)
        return result

    @staticmethod
    def load_generated_symbols(config: dict, version: str | None = None) -> pd.DataFrame:
        """Load the immutable symbol metadata for the configured version."""
        symbol_version = version or config["version"]["data"]
        storage = Storage(config["storage"], Path(__file__).resolve().parents[2])
        path = SymbolManager.symbol_file_path(SymbolManager.storage_base_dir(config), symbol_version)
        if not storage.exists(path):
            raise FileNotFoundError(
                f"Symbol file not found: {storage.describe(path)}. Run symbol_manager.py to generate it"
            )
        data = storage.read_csv(path)
        required = {"symbol", "index_group", "company_name"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Symbol file is missing columns: {sorted(missing)}")
        return data

    def generate_symbol_snapshots(
        self, config: dict, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp
    ) -> dict[pd.Timestamp, pd.DataFrame]:
        """Write one deduplicated point-in-time symbol file per business day."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if start > end:
            raise ValueError("start_date must not exceed end_date")
        snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
        for day in pd.bdate_range(start, end):
            snapshots[day] = self.generate_symbols(config, day)
        return snapshots

    @staticmethod
    def load_symbol_snapshot(config: dict, on_date: str | pd.Timestamp, version: str | None = None) -> pd.DataFrame:
        """Load the per-date symbol membership file generated by `generate_symbol_snapshots`."""
        symbol_version = version or config["version"]["data"]
        storage = Storage(config["storage"], Path(__file__).resolve().parents[2])
        path = SymbolManager.symbol_snapshot_path(SymbolManager.storage_base_dir(config), symbol_version, on_date)
        if not storage.exists(path):
            raise FileNotFoundError(
                f"Symbol snapshot not found: {storage.describe(path)}. Run generate_symbol_snapshots first"
            )
        data = storage.read_csv(path)
        required = {"date", "symbol", "index_group", "company_name"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Symbol snapshot is missing columns: {sorted(missing)}"
            )
        return data.drop_duplicates(subset=["symbol"]).reset_index(drop=True)

    @staticmethod
    def load_latest_symbol_snapshot(
        config: dict,
        on_or_before: str | pd.Timestamp,
        version: str | None = None,
        lookback_days: int = 30,
    ) -> tuple[pd.Timestamp, pd.DataFrame]:
        """List recent symbol snapshots and load the newest one on or before the requested date."""
        symbol_version = version or config["version"]["data"]
        storage = Storage(config["storage"], Path(__file__).resolve().parents[2])
        day = pd.Timestamp(on_or_before)
        if day.tzinfo is not None:
            day = day.tz_convert("America/New_York").tz_localize(None)
        day = day.normalize()
        symbols_dir = SymbolManager.storage_base_dir(config)
        snapshot_dir = Path(symbols_dir) / symbol_version
        earliest = day - pd.Timedelta(days=lookback_days)
        candidates: list[tuple[pd.Timestamp, Path]] = []
        for snapshot_path in storage.list(snapshot_dir, "*.csv"):
            snapshot_date = pd.to_datetime(Path(snapshot_path).stem, errors="coerce")
            if pd.isna(snapshot_date):
                continue
            snapshot_day = pd.Timestamp(snapshot_date)
            if snapshot_day.tzinfo is not None:
                snapshot_day = snapshot_day.tz_convert("America/New_York").tz_localize(None)
            snapshot_day = snapshot_day.normalize()
            if earliest <= snapshot_day <= day:
                candidates.append((snapshot_day, Path(snapshot_path)))
        if candidates:
            snapshot_day, snapshot_path = max(candidates, key=lambda item: item[0])
            data = storage.read_csv(snapshot_path)
            required = {"date", "symbol", "index_group", "company_name"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(
                    f"Symbol snapshot is missing columns: {sorted(missing)}"
                )
            return snapshot_day, data.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        raise FileNotFoundError(
            f"No symbol snapshot found on or before {day:%Y-%m-%d} "
            f"within {lookback_days} day(s) under "
            f"{storage.describe(snapshot_dir)}"
        )

    @staticmethod
    def symbol_file_path(symbols_dir: str | Path, version: str) -> Path:
        return Path(symbols_dir) / version / "symbols.csv"

    @staticmethod
    def symbol_snapshot_path(symbols_dir: str | Path, version: str, on_date: str | pd.Timestamp) -> Path:
        return Path(symbols_dir) / version / f"{pd.Timestamp(on_date):%Y-%m-%d}.csv"

    @staticmethod
    def storage_base_dir(config: dict) -> str | Path:
        return config["storage"].get("output_dir", ".")

    @staticmethod
    def _normalize_symbols(symbols: list[str]) -> list[str]:
        return sorted(
            {
                str(symbol).strip().upper().replace(".", "-")
                for symbol in symbols
                if str(symbol).strip()
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point-in-time symbol files")
    parser.add_argument("--config", default="config/symbols/symbols_config.yaml")
    parser.add_argument("--date", help="Generate one file for YYYY-MM-DD")
    parser.add_argument("--start-date", help="Override configured schedule start date")
    parser.add_argument("--end-date", help="Override configured schedule end date")
    args = parser.parse_args()
    config = SymbolManager.load_config(args.config)
    manager = SymbolManager(config["selection"].get("random_state", 42))
    if args.date:
        result = manager.generate_symbols(config, args.date)
        print(
            f"Saved {len(result)} symbols to "
            f"{manager.symbol_snapshot_path(manager.storage_base_dir(config), config['version']['data'], args.date)}"
        )
        return

    schedule = config.get("schedule", {})
    start_date = args.start_date or schedule.get("start_date")
    end_date = args.end_date or schedule.get("end_date")
    if not start_date or not end_date:
        raise ValueError("Provide --date or configure schedule.start_date and schedule.end_date")
    snapshots = manager.generate_symbol_snapshots(config, start_date, end_date)
    print(f"Generated {len(snapshots)} daily symbol files")


if __name__ == "__main__":
    main()