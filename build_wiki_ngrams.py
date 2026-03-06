#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Tuple

import spacy
from datasets import load_dataset
from tqdm import tqdm


PAREN_PATTERNS = [
    re.compile(r"\([^()]*\)"),
    re.compile(r"（[^（）]*）"),
]

ASCII_WORD_RE = re.compile(r"^[A-Za-z]+(?:[.-][A-Za-z]+)*$")
HAS_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "NORP",
    "EVENT",
    "WORK_OF_ART",
    "PRODUCT",
    "LAW",
    "LANGUAGE",
)


def strip_parenthesized(text: str) -> str:
    if not text:
        return ""
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        for pat in PAREN_PATTERNS:
            cur = pat.sub(" ", cur)
    return re.sub(r"\s+", " ", cur).strip()


def normalize_token_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_valid_token(token) -> bool:
    text = normalize_token_text(token.text)

    if not text:
        return False

    if token.is_space or token.is_punct or token.like_url:
        return False

    if token.like_num:
        return False

    if not HAS_ASCII_ALPHA_RE.search(text):
        return False

    if not ASCII_WORD_RE.fullmatch(text):
        return False

    if len(text) == 1 and text.lower() not in {"a", "i"}:
        return False

    return True


def is_valid_ngram_window(window) -> bool:
    texts = [normalize_token_text(t.text) for t in window]

    if not all(ASCII_WORD_RE.fullmatch(x) for x in texts):
        return False

    if all(t.is_stop for t in window):
        return False

    if not any(not t.is_stop for t in window):
        return False

    return True


def title_entry(nlp, title: str) -> Tuple[int, str, str] | None:
    cleaned = strip_parenthesized(title)
    if not cleaned:
        return None
    doc = nlp(cleaned)
    toks = [t for t in doc if is_valid_token(t)]
    if not toks:
        return None
    if not is_valid_ngram_window(toks):
        return None
    n = len(toks)
    ngram = " ".join(normalize_token_text(t.text) for t in toks)
    pos = ",".join(t.pos_ for t in toks)
    return n, ngram, pos


def parse_entity_labels(value: str) -> set[str] | None:
    value = value.strip()
    if value.lower() in {"all", "*"}:
        return None
    labels = {part.strip() for part in value.split(",") if part.strip()}
    return labels or set(DEFAULT_ENTITY_LABELS)


def iter_text_windows(
    nlp,
    text: str,
    n: int,
    entity_labels: set[str] | None,
    include_entities: bool,
) -> Iterator[Tuple[str, str]]:
    cleaned = strip_parenthesized(text)
    if not cleaned:
        return

    doc = nlp(cleaned)

    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
    for sent in sentences:
        toks = [t for t in sent if is_valid_token(t)]
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            window = toks[i : i + n]

            if not is_valid_ngram_window(window):
                continue

            ngram = " ".join(normalize_token_text(t.text) for t in window)
            pos = ",".join(t.pos_ for t in window)
            yield ngram, pos

    if not include_entities:
        return

    for ent in doc.ents:
        if entity_labels is not None and ent.label_ not in entity_labels:
            continue
        toks = [t for t in ent if is_valid_token(t)]
        if len(toks) != n:
            continue

        if not is_valid_ngram_window(toks):
            continue

        ngram = " ".join(normalize_token_text(t.text) for t in toks)
        pos = ",".join(t.pos_ for t in toks)
        yield ngram, pos


class SqliteCounter:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS counts (bucket TEXT NOT NULL, ngram TEXT NOT NULL, pos TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(bucket, ngram, pos))"
        )
        self.conn.commit()

    def iter_bucket(self, bucket: str) -> Iterator[Tuple[str, str, int]]:
        cur = self.conn.execute(
            "SELECT ngram, pos, count FROM counts WHERE bucket = ? ORDER BY count DESC, ngram ASC, pos ASC",
            (bucket,),
        )
        yield from cur

    def max_count(self, bucket: str) -> int:
        cur = self.conn.execute("SELECT COALESCE(MAX(count), 0) FROM counts WHERE bucket = ?", (bucket,))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_buckets(self, prefix: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT DISTINCT bucket FROM counts WHERE bucket LIKE ? ORDER BY bucket ASC",
            (f"{prefix}%",),
        )
        return [row[0] for row in cur]

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def write_bucket(counter: SqliteCounter, bucket: str, out_path: Path, fmt: str) -> None:
    max_count = counter.max_count(bucket)
    if max_count <= 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ngram, pos, count in counter.iter_bucket(bucket):
            score = max_count - count + 1
            if fmt == "tsv":
                f.write(f"{ngram}\t{pos}\t{score}\n")
            else:
                f.write(json.dumps({"ngram": ngram, "pos": pos, "score": score}, ensure_ascii=False) + "\n")


def load_spacy(model_name: str):
    try:
        return spacy.load(model_name, disable=["lemmatizer"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
        ) from exc


def get_dataset(config: str, streaming: bool):
    return load_dataset("wikimedia/wikipedia", config, split="train", streaming=streaming)


def flush_batches(counter: SqliteCounter, title_batch, text_batch) -> None:
    if title_batch:
        grouped = Counter(title_batch)
        rows = []
        for (bucket, item), c in grouped.items():
            rows.extend([(bucket, item[0], item[1])] * c)
        counter.conn.executemany(
            "INSERT INTO counts(bucket, ngram, pos, count) VALUES(?, ?, ?, 1) "
            "ON CONFLICT(bucket, ngram, pos) DO UPDATE SET count = count + 1",
            rows,
        )
        title_batch.clear()

    if text_batch:
        grouped = Counter(text_batch)
        rows = []
        for (bucket, item), c in grouped.items():
            rows.extend([(bucket, item[0], item[1])] * c)
        counter.conn.executemany(
            "INSERT INTO counts(bucket, ngram, pos, count) VALUES(?, ?, ?, 1) "
            "ON CONFLICT(bucket, ngram, pos) DO UPDATE SET count = count + 1",
            rows,
        )
        text_batch.clear()

    counter.conn.commit()


def process_dataset(args) -> None:
    nlp = load_spacy(args.spacy_model)
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    entity_labels = parse_entity_labels(args.entity_labels)
    dataset = get_dataset(args.dataset_config, args.streaming)
    if args.limit is not None:
        dataset = dataset.take(args.limit) if args.streaming else dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "ngram_counts.sqlite3"
    if db_path.exists() and not args.resume:
        db_path.unlink()
    counter = SqliteCounter(db_path)

    title_batch: List[Tuple[str, Tuple[str, str]]] = []
    text_batch: List[Tuple[str, Tuple[str, str]]] = []

    total = None if args.streaming else len(dataset)
    iterator = tqdm(dataset, total=total, desc="Processing Wikipedia", unit="row")

    for row in iterator:
        title = row.get("title", "")
        text = row.get("text", "")

        if not args.text_only:
            t = title_entry(nlp, title)
            if t is not None:
                n, ngram, pos = t
                if args.title_max_n is None or n <= args.title_max_n:
                    title_batch.append((f"title_{n}", (ngram, pos)))

        if not args.title_only:
            for ngram, pos in iter_text_windows(
                nlp=nlp,
                text=text,
                n=args.text_n,
                entity_labels=entity_labels,
                include_entities=not args.no_entities,
            ):
                text_batch.append((f"text_{args.text_n}", (ngram, pos)))

        if len(title_batch) + len(text_batch) >= args.flush_every:
            flush_batches(counter, title_batch, text_batch)

    flush_batches(counter, title_batch, text_batch)

    ext = args.format
    if not args.text_only:
        title_buckets = counter.list_buckets("title_")
        for bucket in tqdm(title_buckets, desc="Writing title files", unit="file"):
            n = bucket.split("_")[1]
            write_bucket(counter, bucket, output_dir / f"title_{n}_gram.{ext}", args.format)

    if not args.title_only:
        text_bucket = f"text_{args.text_n}"
        if counter.max_count(text_bucket) > 0:
            write_bucket(counter, text_bucket, output_dir / f"text_{args.text_n}_gram.{ext}", args.format)

    counter.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build title/text n-gram dictionaries from wikimedia/wikipedia.")
    p.add_argument("--dataset-config", default="20231101.en", help="Hugging Face dataset config, e.g. 20231101.en")
    p.add_argument("--text-n", type=int, default=4, help="n for text n-gram output")
    p.add_argument("--title-max-n", type=int, default=8, help="maximum title token length to emit")
    p.add_argument("--format", choices=["tsv", "jsonl"], default="tsv", help="output format")
    p.add_argument("--output-dir", default="out", help="output directory")
    p.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy English model")
    p.add_argument("--streaming", action="store_true", help="use Hugging Face streaming mode")
    p.add_argument("--limit", type=int, default=None, help="limit number of processed rows")
    p.add_argument("--flush-every", type=int, default=1000, help="flush batch size to sqlite")
    p.add_argument("--resume", action="store_true", help="reuse existing sqlite db instead of replacing it")
    p.add_argument(
        "--entity-labels",
        default=",".join(DEFAULT_ENTITY_LABELS),
        help=(
            "comma-separated spaCy NER labels to also register when token length matches --text-n; "
            "use 'all' or '*' for every entity label"
        ),
    )
    p.add_argument("--no-entities", action="store_true", help="disable named-entity registration for text")
    p.add_argument("--title-only", action="store_true", help="build only title dictionaries")
    p.add_argument("--text-only", action="store_true", help="build only text dictionary")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.title_only and args.text_only:
        raise SystemExit("--title-only and --text-only cannot be used together")
    if not args.title_only and args.text_n <= 0:
        raise SystemExit("--text-n must be >= 1")
    if args.title_max_n is not None and args.title_max_n <= 0:
        raise SystemExit("--title-max-n must be >= 1")
    process_dataset(args)


if __name__ == "__main__":
    main()