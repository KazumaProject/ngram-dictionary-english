#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterator, List, Tuple


class SqliteCounter:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))

    def iter_bucket(self, bucket: str) -> Iterator[Tuple[str, str, int]]:
        cur = self.conn.execute(
            """
            SELECT ngram, pos, count
            FROM counts
            WHERE bucket = ?
            ORDER BY count DESC, ngram ASC, pos ASC
            """,
            (bucket,),
        )
        yield from cur

    def max_count(self, bucket: str) -> int:
        cur = self.conn.execute(
            "SELECT COALESCE(MAX(count), 0) FROM counts WHERE bucket = ?",
            (bucket,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def sum_count(self, bucket: str) -> int:
        cur = self.conn.execute(
            "SELECT COALESCE(SUM(count), 0) FROM counts WHERE bucket = ?",
            (bucket,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_buckets(self, prefix: str | None = None) -> List[str]:
        if prefix is None:
            cur = self.conn.execute(
                "SELECT DISTINCT bucket FROM counts ORDER BY bucket ASC"
            )
        else:
            cur = self.conn.execute(
                "SELECT DISTINCT bucket FROM counts WHERE bucket LIKE ? ORDER BY bucket ASC",
                (f"{prefix}%",),
            )
        return [row[0] for row in cur]

    def close(self) -> None:
        self.conn.close()


def write_bucket(
    counter: SqliteCounter,
    bucket: str,
    out_path: Path,
    fmt: str,
    output_mode: str,
) -> None:
    total_count = counter.sum_count(bucket)
    max_count = counter.max_count(bucket)

    if total_count <= 0 or max_count <= 0:
        print(f"skip: {bucket} (empty)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ngram, pos, count in counter.iter_bucket(bucket):
            prob = count / total_count
            cost = -math.log(prob)
            rank_score = max_count - count + 1

            if fmt == "tsv":
                if output_mode == "rank_score":
                    f.write(f"{ngram}\t{pos}\t{rank_score}\n")
                elif output_mode == "prob":
                    f.write(f"{ngram}\t{pos}\t{prob:.12f}\n")
                elif output_mode == "cost":
                    f.write(f"{ngram}\t{pos}\t{cost:.12f}\n")
                elif output_mode == "all":
                    f.write(
                        f"{ngram}\t{pos}\t{count}\t{prob:.12f}\t{cost:.12f}\t{rank_score}\n"
                    )
                else:
                    raise ValueError(f"Unsupported output mode: {output_mode}")
            elif fmt == "jsonl":
                row = {
                    "ngram": ngram,
                    "pos": pos,
                }

                if output_mode == "rank_score":
                    row["score"] = rank_score
                elif output_mode == "prob":
                    row["prob"] = prob
                elif output_mode == "cost":
                    row["cost"] = cost
                elif output_mode == "all":
                    row["count"] = count
                    row["prob"] = prob
                    row["cost"] = cost
                    row["rank_score"] = rank_score
                else:
                    raise ValueError(f"Unsupported output mode: {output_mode}")

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                raise ValueError(f"Unsupported format: {fmt}")

    print(f"wrote: {out_path}")


def bucket_to_filename(bucket: str, fmt: str) -> str:
    if bucket.startswith("title_"):
        n = bucket.split("_", 1)[1]
        return f"title_{n}_gram.{fmt}"
    if bucket.startswith("text_"):
        n = bucket.split("_", 1)[1]
        return f"text_{n}_gram.{fmt}"
    safe = bucket.replace("/", "_")
    return f"{safe}.{fmt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write n-gram output files from existing sqlite counts.")
    parser.add_argument("--db", default="out/ngram_counts.sqlite3", help="path to sqlite db")
    parser.add_argument("--output-dir", default="out", help="output directory")
    parser.add_argument("--format", choices=["tsv", "jsonl"], default="tsv")
    parser.add_argument(
        "--output-mode",
        choices=["rank_score", "prob", "cost", "all"],
        default="cost",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=[],
        help="specific bucket(s) to write, e.g. --bucket text_4 --bucket title_2",
    )
    parser.add_argument(
        "--prefix",
        choices=["title", "text", "all"],
        default="all",
        help="write only title_* or text_* buckets, or all",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"sqlite not found: {db_path}")

    output_dir = Path(args.output_dir)
    counter = SqliteCounter(db_path)

    if args.bucket:
        buckets = args.bucket
    else:
        if args.prefix == "title":
            buckets = counter.list_buckets("title_")
        elif args.prefix == "text":
            buckets = counter.list_buckets("text_")
        else:
            buckets = counter.list_buckets()

    if not buckets:
        raise SystemExit("no buckets found")

    for bucket in buckets:
        out_path = output_dir / bucket_to_filename(bucket, args.format)
        write_bucket(
            counter=counter,
            bucket=bucket,
            out_path=out_path,
            fmt=args.format,
            output_mode=args.output_mode,
        )

    counter.close()


if __name__ == "__main__":
    main()
