#!/usr/bin/env python3
"""List overnight price jumps that look like splits, reverse splits or spin-offs not yet
recorded in config/symbols/corporate_actions.yaml.

Reads the regular-session daily bars (storage from config/data/session_bars_config.yaml),
applies the actions already recorded, and reports what is left. Verify every candidate against an
official source (e.g. an OCC Information Memo) before adding it to the YAML file.

    python3 scripts/detect_corporate_actions.py
    python3 scripts/detect_corporate_actions.py --start 2026-01-01 --end 2026-09-25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data.raw.corporate_actions import find_candidates  # noqa: E402
from src.data.raw.dbn_store import file_day  # noqa: E402
from src.data.raw.session_bars import FILE_SUFFIX, load_session_bars  # noqa: E402
from src.data.storage import Storage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="First date to scan, e.g. 2025-01-01 (default: all files)")
    parser.add_argument("--end", help="Last date to scan (default: all files)")
    args = parser.parse_args()

    with (ROOT_DIR / "config/data/session_bars_config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    storage = Storage(config["storage"], ROOT_DIR)
    directory = Path(config["storage"]["output"]["session_bars"]) / config["version"]["data"]
    first = pd.Timestamp(args.start) if args.start else pd.Timestamp.min
    last = pd.Timestamp(args.end) if args.end else pd.Timestamp.max
    files = [path for path in storage.list(directory, f"*{FILE_SUFFIX}") if first <= file_day(path) <= last]
    if not files:
        raise SystemExit(f"No session bar files in {storage.describe(directory)}; run python -m src.pull.session_bars")
    print(f"Scanning {len(files)} daily files in {storage.describe(directory)}")
    bars = load_session_bars(storage, files)

    candidates = find_candidates(bars)
    if candidates.empty:
        print("No unrecorded split-like or large overnight moves found.")
        return
    shown = candidates.assign(date=candidates["date"].dt.strftime("%Y-%m-%d"), ratio=candidates["ratio"].round(4))
    print(f"\n{len(candidates)} candidate(s):")
    print(shown.to_string(index=False))

    splits = candidates[candidates["suggested_split"].notna()]
    if not splits.empty:
        print("\nYAML for the split-like ones (verify each, then fill `source` and set verified: true):")
        for row in splits.itertuples():
            split = row.suggested_split
            split_text = f"{int(split)}" if split >= 1 else f"{split:g}"
            print(
                f"  - {{symbol: {row.symbol}, date: {row.date:%Y-%m-%d}, split: {split_text}, "
                f'verified: false, source: ""}}'
            )
    others = candidates[candidates["suggested_split"].isna()]
    if not others.empty:
        print(
            "\nOther large moves: check whether each is a spin-off (add `factor:` from the OCC memo) "
            "or a genuine news move (no entry)."
        )


if __name__ == "__main__":
    main()
