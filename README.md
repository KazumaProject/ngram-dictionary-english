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
