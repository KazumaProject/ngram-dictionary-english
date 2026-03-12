#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import Counter
from pathlib import Path

import spacy
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_POS_LABELS: tuple[str, ...] = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "SCONJ",
    "VERB",
    "X",
)

SPECIAL_LABELS: tuple[str, ...] = ("BOS", "EOS")


def load_spacy(model_name: str):
    try:
        return spacy.load(model_name, disable=["lemmatizer"])
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
        ) from exc


def get_dataset(config: str, streaming: bool):
    return load_dataset("wikimedia/wikipedia", config, split="train", streaming=streaming)


def build_label_maps(include_all_observed: bool) -> tuple[dict[str, int], list[str]]:
    labels: list[str] = list(SPECIAL_LABELS)
    if not include_all_observed:
        labels.extend(DEFAULT_POS_LABELS)

    label_to_id = {label: i for i, label in enumerate(labels)}
    return label_to_id, labels


def normalize_pos_label(pos_label: str, include_all_observed: bool) -> str:
    if include_all_observed:
        return pos_label or "X"
    return pos_label if pos_label in DEFAULT_POS_LABELS else "X"


def iter_pos_sequences(
    nlp,
    text: str,
    include_all_observed: bool,
    keep_num: bool,
):
    if not text or not text.strip():
        return

    doc = nlp(text)
    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]

    for sent in sentences:
        seq = ["BOS"]

        for token in sent:
            if token.is_space or token.is_punct or token.like_url:
                continue

            if not keep_num and token.like_num:
                continue

            pos_label = normalize_pos_label(token.pos_, include_all_observed)
            seq.append(pos_label)

        seq.append("EOS")

        if len(seq) >= 3:
            yield seq


class SqliteConnectionCounter:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                prev_id INTEGER NOT NULL,
                next_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY(prev_id, next_id)
            )
            """
        )
        self.conn.commit()

    def flush_counts(self, batch: Counter[tuple[int, int]]) -> None:
        if not batch:
            return

        rows = [(prev_id, next_id, count) for (prev_id, next_id), count in batch.items()]
        self.conn.executemany(
            """
            INSERT INTO transitions(prev_id, next_id, count)
            VALUES(?, ?, ?)
            ON CONFLICT(prev_id, next_id) DO UPDATE SET count = count + excluded.count
            """,
            rows,
        )
        self.conn.commit()
        batch.clear()

    def iter_all(self):
        cur = self.conn.execute(
            "SELECT prev_id, next_id, count FROM transitions ORDER BY prev_id ASC, next_id ASC"
        )
        yield from cur

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def write_index_file(out_path: Path, labels: list[str]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for idx, label in enumerate(labels):
            f.write(f"{idx}\t{label}\n")


def write_transition_counts(
    counter: SqliteConnectionCounter,
    out_path: Path,
) -> Counter[int]:
    row_sum: Counter[int] = Counter()

    with out_path.open("w", encoding="utf-8") as f:
        for prev_id, next_id, count in counter.iter_all():
            row_sum[prev_id] += count
            f.write(f"{prev_id}\t{next_id}\t{count}\n")

    return row_sum


def write_connection_costs(
    counter: SqliteConnectionCounter,
    out_path: Path,
    num_labels: int,
    row_sum: Counter[int],
    alpha: float,
) -> None:
    observed_counts = {(prev_id, next_id): count for prev_id, next_id, count in counter.iter_all()}

    with out_path.open("w", encoding="utf-8") as f:
        for prev_id in range(num_labels):
            denom = row_sum[prev_id] + alpha * num_labels

            for next_id in range(num_labels):
                count = observed_counts.get((prev_id, next_id), 0)
                prob = (count + alpha) / denom if denom > 0 else 1.0 / num_labels
                cost = int(round(-1000.0 * math.log(prob)))
                f.write(f"{prev_id}\t{next_id}\t{count}\t{cost}\n")


def process_dataset(args) -> None:
    nlp = load_spacy(args.spacy_model)
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    dataset = get_dataset(args.dataset_config, args.streaming)
    if args.limit is not None:
        dataset = dataset.take(args.limit) if args.streaming else dataset.select(range(min(args.limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "pos_connection_counts.sqlite3"
    if db_path.exists() and not args.resume:
        db_path.unlink()

    counter = SqliteConnectionCounter(db_path)
    label_to_id, id_to_label = build_label_maps(args.include_all_observed_pos)
    batch: Counter[tuple[int, int]] = Counter()

    total = None if args.streaming else len(dataset)
    iterator = tqdm(dataset, total=total, desc="Processing Wikipedia POS transitions", unit="row")

    for row in iterator:
        text = row.get("text", "")

        for seq in iter_pos_sequences(
            nlp=nlp,
            text=text,
            include_all_observed=args.include_all_observed_pos,
            keep_num=args.keep_num,
        ):
            for prev_label, next_label in zip(seq, seq[1:]):
                if prev_label not in label_to_id:
                    label_to_id[prev_label] = len(id_to_label)
                    id_to_label.append(prev_label)
                if next_label not in label_to_id:
                    label_to_id[next_label] = len(id_to_label)
                    id_to_label.append(next_label)

                prev_id = label_to_id[prev_label]
                next_id = label_to_id[next_label]
                batch[(prev_id, next_id)] += 1

        if sum(batch.values()) >= args.flush_every:
            counter.flush_counts(batch)

    counter.flush_counts(batch)

    write_index_file(output_dir / "pos_index.tsv", id_to_label)
    row_sum = write_transition_counts(counter, output_dir / "pos_transition_counts.tsv")
    write_connection_costs(
        counter=counter,
        out_path=output_dir / "pos_connection_costs.tsv",
        num_labels=len(id_to_label),
        row_sum=row_sum,
        alpha=args.alpha,
    )

    counter.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build POS transition counts and connection costs from wikimedia/wikipedia."
    )
    p.add_argument("--dataset-config", default="20231101.en", help="Hugging Face dataset config, e.g. 20231101.en")
    p.add_argument("--output-dir", default="out_pos_connection", help="output directory")
    p.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy English model")
    p.add_argument("--streaming", action="store_true", help="use Hugging Face streaming mode")
    p.add_argument("--limit", type=int, default=None, help="limit number of processed rows")
    p.add_argument("--flush-every", type=int, default=100000, help="flush batch size in number of transitions")
    p.add_argument("--resume", action="store_true", help="reuse existing SQLite DB instead of replacing it")
    p.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="additive smoothing constant for connection cost calculation",
    )
    p.add_argument(
        "--include-all-observed-pos",
        action="store_true",
        help="register every observed spaCy POS label instead of using the default fixed label set",
    )
    p.add_argument(
        "--keep-num",
        action="store_true",
        help="keep tokens recognized as numbers instead of skipping them",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.alpha <= 0:
        raise SystemExit("--alpha must be > 0")
    process_dataset(args)


if __name__ == "__main__":
    main()