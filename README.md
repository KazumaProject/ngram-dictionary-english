# wiki_ngram_pipeline

Build title and text n-gram dictionaries from `wikimedia/wikipedia` on Hugging Face.

## Features

- Removes text inside `()` and `（）`
- Saves **title** entries as full cleaned titles, automatically routed to `title_<n>_gram.*` based on token count
- Saves **text** entries as sliding-window n-grams for the specified `--text-n`
- Can also register spaCy named entities into the text dictionary when the entity token length matches `--text-n`
- Uses spaCy POS tags as-is
- Supports both `tsv` and `jsonl`
- Shows progress with `tqdm`
- Uses SQLite for scalable counting
- Supports `--title-only` and `--text-only`

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Output format

### TSV

```text
I am going	PRON,AUX,VERB	100
```

Columns:

1. n-gram
2. POS sequence
3. score

### JSONL

```json
{"ngram":"I am going","pos":"PRON,AUX,VERB","score":100}
```

## Score

Score is computed per output bucket as:

```text
score = max_count - count + 1
```

So more frequent entries get lower scores, and rarer entries get higher scores.

## Examples

### Build title and text together

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --text-n 4 --format tsv --output-dir out --streaming
```

Outputs may include:

- `out/title_1_gram.tsv`
- `out/title_2_gram.tsv`
- `out/title_3_gram.tsv`
- `out/text_4_gram.tsv`

### Build title only

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --title-only --format tsv --output-dir out --streaming
```

This writes only:

- `out/title_<n>_gram.tsv`

`--text-n` is not needed for `--title-only` because no text dictionary is generated.

### Build text only

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --text-only --text-n 4 --format jsonl --output-dir out --streaming
```

This writes only:

- `out/text_4_gram.jsonl`

### Disable entity registration

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --text-n 4 --no-entities --format tsv --output-dir out --streaming
```

### Register all entity labels

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --text-n 4 --entity-labels all --format tsv --output-dir out --streaming
```

### Limit rows for testing

```bash
python build_wiki_ngrams.py --dataset-config 20231101.en --text-n 4 --limit 1000 --format tsv --output-dir out
```

## Main options

- `--dataset-config`: dataset config such as `20231101.en`
- `--text-n`: n for text n-gram generation
- `--title-max-n`: maximum token length of titles to emit
- `--format`: `tsv` or `jsonl`
- `--output-dir`: output directory
- `--spacy-model`: spaCy model name
- `--streaming`: use streaming mode
- `--limit`: process only a subset of rows
- `--flush-every`: batch size before flushing to SQLite
- `--resume`: reuse existing SQLite DB
- `--entity-labels`: comma-separated entity labels, or `all`
- `--no-entities`: disable entity-based registration for text
- `--title-only`: generate only title dictionaries
- `--text-only`: generate only text dictionary

## Notes

- `--title-only` and `--text-only` cannot be used together.
- Titles are **not** generated as sliding-window n-grams. Each cleaned title is stored as one entry in the corresponding `title_<n>_gram.*` bucket.
- Text entity registration only adds entities whose token length exactly matches `--text-n`.

## POS connection cost builder

What it builds

This script counts sentence-level POS transitions from Wikipedia text and exports:
	•	pos_index.tsv: maps 0-based integer IDs to POS labels
	•	pos_transition_counts.tsv: raw transition counts
	•	pos_connection_costs.tsv: smoothed connection costs

The builder inserts BOS at the start of each sentence and EOS at the end.

Output format

pos_index.tsv
```bash

0	BOS
1	EOS
2	ADJ
3	ADP
4	ADV
5	AUX
6	CCONJ
7	DET
8	INTJ
9	NOUN
10	NUM
11	PART
12	PRON
13	PROPN
14	SCONJ
15	VERB
16	X
```

Columns:
	1.	integer ID
	2.	POS label

pos_transition_counts.tsv

0	7	15320
7	9	892013
9	1	230991

Columns:
	1.	previous POS ID
	2.	next POS ID
	3.	raw transition count

pos_connection_costs.tsv

0	7	15320	120
7	9	892013	40
9	1	230991	181

Columns:
	1.	previous POS ID
	2.	next POS ID
	3.	raw transition count
	4.	connection cost

Cost formula

Connection cost is computed from additively smoothed transition probabilities:

P(next | prev) = (count(prev,next) + alpha) / (row_sum(prev) + alpha * num_labels)
cost = round(-1000 * log(P(next | prev)))

More frequent transitions get lower costs. Rare or unseen transitions get higher costs.

Examples

Build POS connection costs with streaming

python build_wiki_pos_connection.py --dataset-config 20231101.en --output-dir out_pos_connection --streaming

Outputs:
	•	out_pos_connection/pos_index.tsv
	•	out_pos_connection/pos_transition_counts.tsv
	•	out_pos_connection/pos_connection_costs.tsv
	•	out_pos_connection/pos_connection_counts.sqlite3

Limit rows for testing

python build_wiki_pos_connection.py --dataset-config 20231101.en --output-dir out_pos_connection --limit 1000

Keep numeric tokens

python build_wiki_pos_connection.py --dataset-config 20231101.en --output-dir out_pos_connection --streaming --keep-num

Register all observed POS labels dynamically

python build_wiki_pos_connection.py --dataset-config 20231101.en --output-dir out_pos_connection --streaming --include-all-observed-pos

Change smoothing strength

python build_wiki_pos_connection.py --dataset-config 20231101.en --output-dir out_pos_connection --streaming --alpha 1.0

Main options
	•	--dataset-config: dataset config such as 20231101.en
	•	--output-dir: output directory
	•	--spacy-model: spaCy model name
	•	--streaming: use streaming mode
	•	--limit: process only a subset of rows
	•	--flush-every: flush threshold in number of transitions
	•	--resume: reuse existing SQLite DB
	•	--alpha: additive smoothing constant used for connection costs
	•	--include-all-observed-pos: dynamically add every observed spaCy POS label
	•	--keep-num: keep number-like tokens instead of skipping them