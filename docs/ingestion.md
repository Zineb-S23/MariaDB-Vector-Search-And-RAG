# Ingestion: Wikipedia acquisition and chunking

## Seed titles

`data/seed_titles.csv` defines which Wikipedia articles make up the corpus. It is a UTF-8
CSV file with the header `topic,title`:

- `topic` is a broad label such as `physics`. It later becomes the `topic` of the article
  in the database and can be used to filter searches.
- `title` is the English Wikipedia article title. A title that contains a comma must be
  written in quotes.

`load_seed_titles` in `src/wikirag/ingestion/wikipedia.py` reads and checks the file. It
returns `(topic, title)` pairs in file order. While reading, it:

- removes extra spaces and turns underscores into spaces (`Black_hole` becomes `Black hole`),
- skips blank lines,
- stops with a `ValidationError` that names the line number when a row is incomplete, when
  the topic or title is empty, or when the same article appears twice (an article can have
  only one topic),
- stops with a `ValidationError` when the file is missing, has the wrong header or has no rows.

The list currently holds a small starter set. It will grow once the team has agreed on the
final topics.

## Running the tests

```bash
uv run --extra dev python -m pytest tests/unit/test_wikipedia_seed.py -q
```