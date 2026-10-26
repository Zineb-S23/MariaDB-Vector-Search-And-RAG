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


## Wikipedia source

Articles come from the public MediaWiki Action API of English Wikipedia
(`https://en.wikipedia.org/w/api.php`). `fetch_article` in
`src/wikirag/ingestion/wikipedia.py` asks for one article at a time:

- the plain text of the article (no markup),
- the latest revision id and its timestamp,
- the visible categories (hidden maintenance categories are left out),
- the titles of the linked articles from the main namespace.

Redirects are followed. Wikipedia returns long category and link lists in several parts, so
the function keeps asking until nothing is left, with a safety limit of 20 requests per
article. Every request carries a descriptive `User-Agent` and a 30 second timeout, and
`build_snapshot` waits 0.2 seconds between articles. Network and server problems are
reported as `WikiRagError`; a missing or incomplete article is a `ValidationError`.



## Snapshot

Wikipedia articles change over time, so the team downloads them once and shares the result
as a snapshot file: `data/snapshot/wikipedia_snapshot.jsonl`. Everyone who ingests from the
same file works with the same text.

The file has one JSON object per line and one line per article, with these fields:
`article_id`, `title`, `url`, `revision_id`, `revised_at`, `fetched_at`, `topic`,
`categories`, `linked_titles` and `text`. Times are UTC. Keys are sorted and non-ASCII
characters are written as they are, so the same data always produces the same file.

- `build_snapshot(seed_path, out_path, delay_s=0.2)` reads the seed list, downloads every
  article and returns how many were written. An article that cannot be downloaded is skipped
  with a warning. A second seed title that leads to the same article (for example through a
  redirect) is skipped too. The file is written under a temporary name first, so an existing
  snapshot is only replaced when at least one article was downloaded.
- `load_snapshot(path)` returns the articles one by one. A damaged line raises a
  `ValidationError` that names the line number.

Downloading again later can give different text, because Wikipedia keeps changing. The
`revision_id` and `fetched_at` fields in the file record which version was used.




## Running the tests

```bash
uv run --extra dev python -m pytest tests/unit -q
```

The Wikipedia tests use a fake API and need no internet access.
