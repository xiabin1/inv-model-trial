#!/usr/bin/env python3
"""Print the first (or last, with --tail) rows of a Parquet file from S3 or the local disk.

    python3 scripts/read_parquet.py s3://my-bucket/path/features/combined/v1/2026-09-24.parquet -n 5
    python3 scripts/read_parquet.py s3://my-bucket/path/2026-09-24.parquet -n 5 --tail
    python3 scripts/read_parquet.py data/features/minutes/v1/2026-09-24.parquet --columns "date,symbol,min_rvol,daily_MA*"
    python3 scripts/read_parquet.py data/features/minutes/v1/2026-09-24.parquet --list-columns
"""

from __future__ import annotations

import argparse
import fnmatch
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq


def download_s3_object(uri: str, destination: Path, region: str | None) -> Path:
    import boto3

    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if not parsed.netloc or not key:
        raise SystemExit(f"invalid S3 URI: {uri}")
    boto3.client("s3", region_name=region).download_file(parsed.netloc, key, str(destination))
    return destination


def select_columns(available: list[str], patterns: str | None) -> list[str]:
    """Columns matching comma-separated names or wildcards, in file order."""
    if not patterns:
        return available
    wanted = [pattern.strip() for pattern in patterns.split(",") if pattern.strip()]
    selected = [column for column in available if any(fnmatch.fnmatch(column, p) for p in wanted)]
    if not selected:
        raise SystemExit(f"no columns match {patterns!r}; use --list-columns to see them")
    return selected


def show(path: Path, label: str, args: argparse.Namespace) -> None:
    parquet = pq.ParquetFile(path)
    available = parquet.schema_arrow.names
    total_rows = parquet.metadata.num_rows
    print(f"{label}\ntotal rows: {total_rows}  |  columns: {len(available)}")
    if args.list_columns:
        for number, column in enumerate(available, 1):
            print(f"{number:4}  {column}")
        return

    columns = select_columns(available, args.columns)
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.tail(args.rows) if args.tail else frame.head(args.rows)
    print(f"showing: {len(frame)} rows x {len(columns)} columns ({'last' if args.tail else 'first'} rows)")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.T if args.transpose else frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="s3://bucket/key.parquet or a local file path")
    parser.add_argument("-n", "--rows", type=int, default=10, help="Number of rows to print")
    parser.add_argument("-t", "--tail", action="store_true", help="Print the last rows instead of the first")
    parser.add_argument("-c", "--columns", help='Comma-separated names or wildcards, e.g. "date,symbol,min_*"')
    parser.add_argument("-T", "--transpose", action="store_true", help="Print rows as columns (for wide files)")
    parser.add_argument("-l", "--list-columns", action="store_true", help="List column names and exit")
    parser.add_argument("--region", help="AWS region for the S3 bucket")
    args = parser.parse_args()

    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    if args.path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as temporary_file:
            show(download_s3_object(args.path, Path(temporary_file.name), args.region), args.path, args)
    else:
        local_path = Path(args.path)
        if not local_path.exists():
            raise SystemExit(f"file not found: {local_path}")
        show(local_path, args.path, args)


if __name__ == "__main__":
    main()
