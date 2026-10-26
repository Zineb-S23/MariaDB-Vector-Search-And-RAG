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



## Cleaning

`clean_text` in `src/wikirag/ingestion/cleaning.py` tidies the plain text of an article. The
same input always gives the same output, which matters because the article fingerprint is
computed from the cleaned text. If a rule changes, the fingerprints change and the affected
articles are embedded again.

Rules, applied in this order:

1. Unicode NFC normalization, so that `é` has one representation.
2. Line endings become `\n`.
3. Invisible characters are removed: zero-width space, soft hyphen, word joiner and
   byte-order mark. The zero-width non-joiner is kept, because some languages need it
   inside words.
4. Every run of spaces, tabs and non-breaking spaces becomes one space.
5. Spaces at the start and end of each line are removed.
6. Three or more line breaks become one blank line, so paragraph breaks survive.
7. The whole text is stripped.

The wording of the text is never changed. The function raises `ValidationError` if its input
is not a string.



## Chunking

`split_into_chunks` in `src/wikirag/ingestion/chunking.py` cuts cleaned text into pieces
that the embedding model can read. The defaults match the shared contract: at most 1000
characters per chunk and an overlap of 150 characters. Both values are counted in characters.

How a chunk is cut:

- A chunk is at most `max_chars` long. If the text is longer, the chunk ends at the last
  paragraph break, line break, sentence end (`. `) or space found in the second half of its
  window, in that order of preference. If none exists it is cut at `max_chars`.
- The next chunk starts about `overlap_chars` before the end of the previous one, moved
  forward to the beginning of a word. A sentence that is cut in half therefore appears whole
  in one of the two chunks.
- Text without any whitespace cannot be overlapped cleanly and is cut without overlap.
- Chunks have no leading or trailing whitespace, and chunk numbers run from 0 without gaps.
- The same text and settings always give the same chunks. Blank text gives no chunks.

The result is a list of `ChunkDraft` objects (`chunk_index`, `text` and an empty
`section_title`). The embedding pipeline turns them into shared `ChunkRecord` objects by
adding the embedding. Invalid settings, such as an overlap that is not smaller than
`max_chars`, raise `ValidationError`.

The chunk size and overlap are only starting values. The planned experiment compares
other settings on the same questions before the final values are chosen.
## Building the snapshot

The committed snapshot `data/snapshot/wikipedia_snapshot.jsonl` holds the 18 articles of
`data/seed_titles.csv`, downloaded on 2026-10-26. To rebuild it (this needs internet access
and gives the current Wikipedia text, so the result can differ):

```bash
uv run python -c "from pathlib import Path; from wikirag.ingestion.wikipedia import build_snapshot; print(build_snapshot(Path('data/seed_titles.csv'), Path('data/snapshot/wikipedia_snapshot.jsonl')))"
```

The `revision_id` and `fetched_at` fields of every article record which Wikipedia version
was used.
