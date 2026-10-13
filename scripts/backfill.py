#!/usr/bin/env python3
"""Backfill runner: pass the step names to run, in order.

Each step's config gets its ``schedule`` dates rewritten before it runs, since
most pipeline modules read dates only from YAML.

    python3 scripts/backfill.py symbols daily_pull minutes_pull
    python3 scripts/backfill.py            # runs every step in the listed order
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
LOG_DIR = ROOT_DIR / "logs"

RAW_START = "2025-01-01"
FEATURE_START = "2025-06-01"

# step name -> (module, config path relative to config/, date range: raw|feature)
STEPS: dict[str, tuple[str, str, str]] = {
    "symbols": ("src.symbol_generate", "symbols/symbols_config.yaml", "raw"),
    "daily_pull": ("src.pull.daily_pull", "data/daily_config.yaml", "raw"),
    "minutes_pull": ("src.pull.minutes_pull", "data/minutes_config.yaml", "raw"),
    "session_bars": ("src.pull.session_bars", "data/session_bars_config.yaml", "raw"),
    "daily_features": ("src.daily_feature_process", "feature/daily_feature_config.yaml", "feature"),
    "minutes_features": ("src.minutes_feature_process", "feature/minutes_feature_config.yaml", "feature"),
    "labels": ("src.data.features.label_manager", "feature/label_config.yaml", "feature"),
    "combined_features": ("src.combined_feature_process", "feature/combined_feature_config.yaml", "feature"),
}
NEEDS_API_KEY = {"daily_pull", "minutes_pull"}

SCHEDULE_KEY = re.compile(r"^(\s{2})(start_date|end_date):\s*(.*)$")


def previous_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def update_schedule(path: Path, updates: dict[str, str]) -> list[str]:
    """Replace direct children of the top-level ``schedule:`` block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    in_schedule = False
    applied: list[str] = []
    for index, line in enumerate(lines):
        if line and not line[0].isspace():
            in_schedule = line.split("#", 1)[0].strip() == "schedule:"
            continue
        if not in_schedule:
            continue
        match = SCHEDULE_KEY.match(line)
        if not match:
            continue
        indent, key, old_value = match.groups()
        if key not in updates or old_value.strip() == updates[key]:
            continue
        lines[index] = f"{indent}{key}: {updates[key]}"
        applied.append(f"{key}: {old_value.strip()} -> {updates[key]}")
    if applied:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return applied


def apply_config(step: str, args: argparse.Namespace, end: str) -> None:
    _, relative_path, range_kind = STEPS[step]
    start = args.raw_start if range_kind == "raw" else args.feature_start
    changes = update_schedule(CONFIG_DIR / relative_path, {"start_date": start, "end_date": end})
    print(f"[config] {relative_path}: {'; '.join(changes) if changes else 'already up to date'}")


def run_step(step: str) -> None:
    module = STEPS[step][0]
    log_path = LOG_DIR / f"backfill_{step}.log"
    print(f"\n=== {step} ({module}) -> {log_path} ===", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", module],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise SystemExit(f"step '{step}' failed with exit code {returncode}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "steps",
        nargs="*",
        metavar="STEP",
        help=f"Step names to run in order (default: all). Available: {', '.join(STEPS)}",
    )
    parser.add_argument("--raw-start", default=RAW_START, help="Raw data backfill start date")
    parser.add_argument("--feature-start", default=FEATURE_START, help="Feature backfill start date")
    parser.add_argument("--end-date", help="Backfill end date (default: latest weekday)")
    parser.add_argument("--config-only", action="store_true", help="Only update configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = args.steps or list(STEPS)
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s): {', '.join(unknown)}; available: {', '.join(STEPS)}")
    end = args.end_date or previous_weekday(date.today()).isoformat()
    if not (args.raw_start <= args.feature_start < end):
        raise SystemExit("expected raw-start <= feature-start < end-date")

    print(f"Steps    : {' -> '.join(steps)}")
    print(f"Raw      : {args.raw_start} -> {end}")
    print(f"Feature  : {args.feature_start} -> {end}\n")

    if not args.config_only and set(steps) & NEEDS_API_KEY and not os.environ.get("DATABENTO_API_KEY", "").strip():
        raise SystemExit("DATABENTO_API_KEY must be set for the data pull steps")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for step in steps:
        apply_config(step, args, end)
        if not args.config_only:
            run_step(step)
    print("\nBackfill finished successfully.")


if __name__ == "__main__":
    main()
