"""AWS Glue Python Shell entry point for SymbolManager.

Packaging:
  - Zip the `src/` package (must contain `src/__init__.py`) and upload it to S3,
    then reference it with the Glue job's `--extra-py-files` parameter so
    `import src...` resolves.
  - Upload `symbols_config.yaml` to S3 and reference it with `--extra-files`
    (Glue downloads it into the job's local working directory), then pass
    `--config symbols_config.yaml` (just the filename, not a path).

Job arguments:
  --JOB_NAME     (required, provided by Glue automatically)
  --config       (required) path/filename of the symbols config yaml
  --date         (optional) generate a single snapshot for YYYY-MM-DD
  --start_date   (optional) backfill range start, used with --end_date
  --end_date     (optional) backfill range end, used with --start_date
  --s3_bucket    (optional) overrides storage.s3.bucket from the config
  --s3_prefix    (optional) overrides storage.s3.prefix from the config

Use --date for the daily run and --start_date/--end_date for the one-time
historical backfill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from awsglue.utils import getResolvedOptions

sys.path.insert(0, str(Path(__file__).resolve().parent))

from symbols.symbol_manager import SymbolManager

REQUIRED_ARGS = ["JOB_NAME", "config"]
OPTIONAL_ARGS = ["date", "start_date", "end_date", "s3_bucket", "s3_prefix"]


def _resolved_args() -> dict:
    options = REQUIRED_ARGS + [opt for opt in OPTIONAL_ARGS if f"--{opt}" in sys.argv]
    return getResolvedOptions(sys.argv, options)


def main() -> None:
    args = _resolved_args()
    config = SymbolManager.load_config(args["config"])

    if args.get("s3_bucket"):
        config["storage"]["backend"] = "s3"
        config["storage"].setdefault("s3", {})["bucket"] = args["s3_bucket"]
        if args.get("s3_prefix"):
            config["storage"]["s3"]["prefix"] = args["s3_prefix"]

    manager = SymbolManager(config["selection"].get("random_state", 42))

    start_date, end_date = args.get("start_date"), args.get("end_date")
    # No date args at all -> daily trigger call, generate today's snapshot only.
    date_arg = args.get("date") or (pd.Timestamp.today().strftime("%Y-%m-%d") if not start_date and not end_date else None)
    if date_arg:
        result = manager.generate_symbols(config, date_arg)
        print(f"Generated {len(result)} symbols for {date_arg}")
        return

    if not start_date or not end_date:
        raise ValueError("Provide --date, or both --start_date and --end_date")
    snapshots = manager.generate_symbol_snapshots(config, start_date, end_date)
    print(f"Generated {len(snapshots)} daily symbol files from {start_date} to {end_date}")


if __name__ == "__main__":
    main()
