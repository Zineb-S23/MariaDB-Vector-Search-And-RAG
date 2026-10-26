"""Seed an isolated MariaDB database for HNSW cache experiments."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from datetime import datetime

from wikirag.config import EMBEDDING_DIM, load_settings
from wikirag.db.repository import WikiRepository
from wikirag.models import ArticleRecord, ChunkRecord

ARTICLE_ID = 8800001
CHUNK_COUNT = 6000
RANDOM_SEED = 20261026


def generate_chunks(count: int) -> tuple[ChunkRecord, ...]:
    """Generate deterministic normalized synthetic vectors.

    Args:
        count: Number of chunks to generate.

    Returns:
        tuple[ChunkRecord, ...]: Ordered chunks containing normalized
            384-dimensional vectors.

    Raises:
        ValueError: If count is not a positive integer.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")

    generator = random.Random(RANDOM_SEED)
    chunks: list[ChunkRecord] = []

    for index in range(count):
        values = [generator.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]

        norm = math.sqrt(sum(value * value for value in values))

        vector = tuple(value / norm for value in values)

        chunks.append(
            ChunkRecord(
                chunk_index=index,
                text=f"Synthetic cache benchmark chunk {index}",
                embedding=vector,
            )
        )

    return tuple(chunks)


def main() -> None:
    """Populate only the dedicated Experiment E database.

    Refuse to seed a database containing existing records.

    Returns:
        None: The synthetic dataset is committed and verified.

    Raises:
        RuntimeError: If the database configuration is unsafe or
            the target database already contains application data.
    """
    settings = load_settings()

    config = replace(
        settings.db,
        host="127.0.0.1",
        port=13308,
        database="wikirag_cache_test",
    )

    # Refuse to operate outside the dedicated experiment database.
    if (
        config.host != "127.0.0.1"
        or config.port != 13308
        or config.database != "wikirag_cache_test"
        or config.database == settings.db.database
    ):
        raise RuntimeError("Unsafe cache experiment database")

    fingerprint = hashlib.sha256(f"cache-v1:{CHUNK_COUNT}:{RANDOM_SEED}".encode()).hexdigest()

    with WikiRepository(config) as repo:
        # Never overwrite an existing experimental or application corpus.
        if repo.count_articles() != 0 or repo.count_chunks() != 0:
            raise RuntimeError(
                "The experiment database already contains data. Refusing to overwrite it."
            )

        repo.ensure_embedding_meta(
            "synthetic-cache-experiment-v1",
            EMBEDDING_DIM,
        )

        print(f"Generating {CHUNK_COUNT} synthetic vectors...")

        chunks = generate_chunks(CHUNK_COUNT)

        timestamp = datetime(2026, 1, 1)

        article = ArticleRecord(
            article_id=ARTICLE_ID,
            title="Synthetic Cache Experiment",
            url="https://example.invalid/cache-experiment",
            revision_id=1,
            fingerprint=fingerprint,
            revised_at=timestamp,
            fetched_at=timestamp,
            length_chars=sum(len(chunk.text) for chunk in chunks),
            topic="synthetic",
            categories=("Cache experiment",),
            linked_titles=(),
        )

        outcome = repo.ingest_article(article, chunks)

        assert outcome == "inserted"
        assert repo.count_articles() == 1
        assert repo.count_chunks() == CHUNK_COUNT

        # Verify the native search index can retrieve stored vectors.
        hits = repo.search_chunks(
            chunks[0].embedding,
            top_k=5,
        )

        assert hits, "Indexed search returned no results"

        print("Database:", config.database)
        print("Articles:", repo.count_articles())
        print("Chunks:", repo.count_chunks())
        print("Indexed search hits:", len(hits))
        print("Synthetic dataset initialized successfully")


if __name__ == "__main__":
    main()
