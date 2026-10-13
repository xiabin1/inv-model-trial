#!/usr/bin/env python3
"""Print the first (or last, with --tail) rows of a DBN (.dbn.zst) file from S3 or the local disk.

    python3 scripts/read_dbn.py s3://my-bucket/path/file.ohlcv-1d.dbn.zst -n 10
    python3 scripts/read_dbn.py s3://my-bucket/path/file.ohlcv-1d.dbn.zst -n 10 --tail
    python3 scripts/read_dbn.py data/raw/daily/v1/20260901.ohlcv-1d.dbn.zst
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import databento as db
import pandas as pd


def download_s3_object(uri: str, destination: Path, region: str | None) -> Path:
    import boto3

    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if not parsed.netloc or not key:
        raise SystemExit(f"invalid S3 URI: {uri}")
    boto3.client("s3", region_name=region).download_file(parsed.netloc, key, str(destination))
    return destination


def head_frame(path: Path, rows: int, schema: str | None) -> pd.DataFrame:
    store = db.DBNStore.from_file(path)
    frames = store.to_df(schema=schema, count=rows)
    return next(iter(frames), pd.DataFrame())


def tail_frame(path: Path, rows: int, schema: str | None) -> pd.DataFrame:
    """Stream the file in chunks, keeping only the last rows in memory."""
    store = db.DBNStore.from_file(path)
    tail = pd.DataFrame()
    for chunk in store.to_df(schema=schema, count=max(rows, 100_000)):
        tail = pd.concat([tail, chunk]).tail(rows)
    return tail


def count_rows(path: Path) -> int:
    """Count every record in the file; requires decoding the whole stream."""
    return sum(1 for _ in db.DBNStore.from_file(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="s3://bucket/key.dbn.zst or a local file path")
    parser.add_argument("-n", "--rows", type=int, default=10, help="Number of rows to print")
    parser.add_argument("-t", "--tail", action="store_true", help="Print the last rows instead of the first")
    parser.add_argument("--schema", help="Schema name, required only for mixed-schema files")
    parser.add_argument("--region", help="AWS region for the S3 bucket")
    args = parser.parse_args()

    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")
    read_frame = tail_frame if args.tail else head_frame

    if args.path.startswith("s3://"):
        with tempfile.NamedTemporaryFile(suffix=".dbn.zst") as temporary_file:
            local_path = download_s3_object(args.path, Path(temporary_file.name), args.region)
            frame = read_frame(local_path, args.rows, args.schema)
            total_rows = count_rows(local_path)
    else:
        local_path = Path(args.path)
        if not local_path.exists():
            raise SystemExit(f"file not found: {local_path}")
        frame = read_frame(local_path, args.rows, args.schema)
        total_rows = count_rows(local_path)

    print(
        f"{args.path}\n"
        f"total rows: {total_rows}  |  columns: {len(frame.columns)}  |  showing: {len(frame)}"
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame)


if __name__ == "__main__":
    main()
