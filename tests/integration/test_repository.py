"""Integration tests for the MariaDB-backed WikiRepository."""

import json
from contextlib import closing
from dataclasses import replace
from datetime import datetime

import pytest

from wikirag.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    DbConfig,
    load_settings,
)
from wikirag.db.connection import get_connection
from wikirag.db.repository import WikiRepository
from wikirag.db.schema import init_schema
from wikirag.errors import (
    DatabaseConnectionError,
    DatabaseError,
    EmbeddingDimensionError,
    SchemaMismatchError,
    ValidationError,
)
from wikirag.models import ArticleRecord, ChunkRecord, SearchFilters


@pytest.fixture
def test_db_config() -> DbConfig:
    """Initialize an isolated MariaDB database for one test.

    Returns:
        DbConfig: Connection settings for the test database.

    Raises:
        ValidationError: If test database configuration is unsafe.
    """
    settings = load_settings()

    if not settings.db_name_test.endswith("_test") or settings.db_name_test == settings.db.database:
        raise ValidationError("Integration tests require a separate test database")

    config = replace(settings.db, database=settings.db_name_test)

    # Only the dedicated test database is rebuilt.
    with closing(get_connection(config)) as conn:
        init_schema(conn, recreate=True)

    return config


def test_repository_starts_empty(test_db_config: DbConfig) -> None:
    """Verify repository counts and missing fingerprints.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 0
        assert repo.count_chunks() == 0
        assert repo.get_fingerprint(123456) is None


def test_embedding_metadata_is_idempotent(
    test_db_config: DbConfig,
) -> None:
    """Store embedding metadata without duplicating it.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

        # Calling again with identical values must be safe.
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT model_name, embedding_dim FROM embedding_meta")
        rows = cursor.fetchall()

    assert len(rows) == 1
    assert rows[0]["model_name"] == EMBEDDING_MODEL_NAME
    assert rows[0]["embedding_dim"] == EMBEDDING_DIM


def test_embedding_metadata_rejects_conflicts(
    test_db_config: DbConfig,
) -> None:
    """Reject incompatible embedding models without overwriting metadata.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        repo.ensure_embedding_meta(
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIM,
        )

        with pytest.raises(SchemaMismatchError):
            repo.ensure_embedding_meta(
                "different-embedding-model",
                EMBEDDING_DIM,
            )

        with pytest.raises(SchemaMismatchError):
            repo.ensure_embedding_meta(
                EMBEDDING_MODEL_NAME,
                768,
            )

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT model_name, embedding_dim FROM embedding_meta WHERE id = 1")
        row = cursor.fetchone()

    # Rejected calls must leave the original settings unchanged.
    assert row["model_name"] == EMBEDDING_MODEL_NAME
    assert row["embedding_dim"] == EMBEDDING_DIM


def test_embedding_metadata_rejects_blank_name(
    test_db_config: DbConfig,
) -> None:
    """Reject invalid model names before writing to MariaDB.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        with pytest.raises(ValidationError):
            repo.ensure_embedding_meta("   ", EMBEDDING_DIM)

        assert repo.count_articles() == 0

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS total FROM embedding_meta")
        row = cursor.fetchone()

    assert row["total"] == 0


def test_repository_reads_stored_article(
    test_db_config: DbConfig,
) -> None:
    """Count a real article and retrieve its SHA-256 fingerprint.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article_id = 987654
    fingerprint = "a" * 64
    timestamp = datetime(2025, 1, 1)

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO articles (
                article_id, title, url, revision_id,
                revised_at, length_chars, fingerprint,
                topic, language, fetched_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                article_id,
                "Integration Test Article",
                "https://en.wikipedia.org/wiki/Integration_Test_Article",
                123,
                timestamp,
                100,
                fingerprint,
                "testing",
                "en",
                timestamp,
            ),
        )
        conn.commit()

    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 1
        assert repo.count_chunks() == 0
        assert repo.get_fingerprint(article_id) == fingerprint
        assert repo.get_fingerprint(123456) is None


def test_repository_rejects_calls_after_close(
    test_db_config: DbConfig,
) -> None:
    """Prevent database operations after the repository is closed.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    with WikiRepository(test_db_config) as repo:
        assert repo.count_articles() == 0

    # Closing repeatedly should be harmless.
    repo.close()

    with pytest.raises(DatabaseConnectionError):
        repo.count_articles()

    with pytest.raises(DatabaseConnectionError):
        repo.get_fingerprint(123456)


def _example_article() -> ArticleRecord:
    """Build deterministic metadata for an integration-test article.

    Returns:
        ArticleRecord: Example article with categories and links.
    """
    return ArticleRecord(
        article_id=990001,
        title="Test Physics Article",
        url="https://en.wikipedia.org/wiki/Test_Physics_Article",
        revision_id=10,
        fingerprint="a" * 64,
        revised_at=datetime(2026, 1, 1),
        fetched_at=datetime(2026, 1, 2),
        length_chars=80,
        topic="physics",
        categories=("Physics", "Science"),
        linked_titles=("Relativity", "Quantum mechanics"),
    )


def _example_chunks() -> tuple[ChunkRecord, ...]:
    """Create two chunks containing valid 384-dimensional vectors.

    Returns:
        tuple[ChunkRecord, ...]: Two ordered test chunks.
    """
    first = (1.0,) + (0.0,) * 383
    second = (0.0, 1.0) + (0.0,) * 382

    return (
        ChunkRecord(
            chunk_index=0,
            text="Physics explains motion.",
            embedding=first,
        ),
        ChunkRecord(
            chunk_index=1,
            text="Energy and matter interact.",
            embedding=second,
        ),
    )


def test_ingest_inserts_all_related_records(
    test_db_config: DbConfig,
) -> None:
    """Store article metadata, native vectors, categories and links.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    with WikiRepository(test_db_config) as repo:
        assert repo.ingest_article(article, _example_chunks()) == "inserted"
        assert repo.count_articles() == 1
        assert repo.count_chunks() == 2
        assert repo.get_fingerprint(article.article_id) == article.fingerprint

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT chunk_index, text, char_count, "
            "VEC_ToText(embedding) AS vector_json "
            "FROM chunks WHERE article_id = %s ORDER BY chunk_index",
            (article.article_id,),
        )
        rows = cursor.fetchall()

        assert [row["chunk_index"] for row in rows] == [0, 1]
        assert rows[0]["char_count"] == len(_example_chunks()[0].text)

        # Read the actual native VECTOR value back from MariaDB.
        vector = json.loads(rows[0]["vector_json"])
        assert len(vector) == 384
        assert vector[:3] == [1.0, 0.0, 0.0]

        cursor.execute(
            "SELECT c.name FROM categories AS c "
            "JOIN article_categories AS ac "
            "ON ac.category_id = c.category_id "
            "WHERE ac.article_id = %s ORDER BY c.name",
            (article.article_id,),
        )
        assert [row["name"] for row in cursor.fetchall()] == [
            "Physics",
            "Science",
        ]

        cursor.execute(
            "SELECT target_title, target_article_id FROM links "
            "WHERE source_article_id = %s ORDER BY target_title",
            (article.article_id,),
        )
        links = cursor.fetchall()

        assert [row["target_title"] for row in links] == [
            "Quantum mechanics",
            "Relativity",
        ]
        assert all(row["target_article_id"] is None for row in links)


def test_ingest_unchanged_fingerprint_skips_writes(
    test_db_config: DbConfig,
) -> None:
    """Preserve records when the incoming fingerprint is unchanged.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    with WikiRepository(test_db_config) as repo:
        assert repo.ingest_article(article, _example_chunks()) == "inserted"

        unchanged = replace(
            article,
            title="Should Not Replace Original Title",
        )

        assert repo.ingest_article(unchanged, _example_chunks()) == "skipped"
        assert repo.count_chunks() == 2

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT title FROM articles WHERE article_id = %s",
            (article.article_id,),
        )
        assert cursor.fetchone()["title"] == article.title


def test_ingest_changed_fingerprint_replaces_associations(
    test_db_config: DbConfig,
) -> None:
    """Replace chunks, categories and links when article content changes.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    changed = replace(
        article,
        title="Revised Physics Article",
        fingerprint="b" * 64,
        revision_id=11,
        categories=("New category",),
        linked_titles=("New destination",),
    )

    replacement = (replace(_example_chunks()[1], chunk_index=0),)

    with WikiRepository(test_db_config) as repo:
        assert repo.ingest_article(article, _example_chunks()) == "inserted"
        assert repo.ingest_article(changed, replacement) == "updated"

        assert repo.get_fingerprint(article.article_id) == changed.fingerprint
        assert repo.count_articles() == 1
        assert repo.count_chunks() == 1

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT title FROM articles WHERE article_id = %s",
            (article.article_id,),
        )
        assert cursor.fetchone()["title"] == changed.title

        cursor.execute(
            "SELECT text FROM chunks WHERE article_id = %s",
            (article.article_id,),
        )
        assert [row["text"] for row in cursor.fetchall()] == [
            replacement[0].text,
        ]

        cursor.execute(
            "SELECT c.name FROM categories AS c "
            "JOIN article_categories AS ac "
            "ON ac.category_id = c.category_id "
            "WHERE ac.article_id = %s",
            (article.article_id,),
        )
        assert [row["name"] for row in cursor.fetchall()] == [
            "New category",
        ]

        cursor.execute(
            "SELECT target_title FROM links WHERE source_article_id = %s",
            (article.article_id,),
        )
        assert [row["target_title"] for row in cursor.fetchall()] == [
            "New destination",
        ]


def test_ingest_invalid_embedding_leaves_database_empty(
    test_db_config: DbConfig,
) -> None:
    """Reject invalid vector dimensions before changing the database.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    invalid = replace(
        _example_chunks()[0],
        embedding=(1.0,) * 383,
    )

    with WikiRepository(test_db_config) as repo:
        with pytest.raises(EmbeddingDimensionError):
            repo.ingest_article(_example_article(), (invalid,))

        assert repo.count_articles() == 0
        assert repo.count_chunks() == 0


def test_ingest_database_error_rolls_back_update(
    test_db_config: DbConfig,
) -> None:
    """Preserve existing data when a later database write fails.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    updated = replace(
        article,
        fingerprint="b" * 64,
        categories=("Different category",),
        linked_titles=("New destination",),
    )

    with WikiRepository(test_db_config) as repo:
        assert repo.ingest_article(article, _example_chunks()) == "inserted"

    # Deliberately fail an INSERT into links after earlier writes.
    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "CREATE TRIGGER fail_wikirag_link_insert "
            "BEFORE INSERT ON links "
            "FOR EACH ROW SIGNAL SQLSTATE '45000' "
            "SET MESSAGE_TEXT = 'Intentional integration-test failure'"
        )

    with WikiRepository(test_db_config) as repo:
        with pytest.raises((DatabaseError, DatabaseConnectionError)):
            repo.ingest_article(updated, _example_chunks()[:1])

        # The failed update must not overwrite the original fingerprint.
        assert repo.get_fingerprint(article.article_id) == article.fingerprint
        assert repo.count_chunks() == 2

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT c.name FROM categories AS c "
            "JOIN article_categories AS ac "
            "ON ac.category_id = c.category_id "
            "WHERE ac.article_id = %s ORDER BY c.name",
            (article.article_id,),
        )
        assert [row["name"] for row in cursor.fetchall()] == [
            "Physics",
            "Science",
        ]

        cursor.execute(
            "SELECT target_title FROM links WHERE source_article_id = %s ORDER BY target_title",
            (article.article_id,),
        )
        assert [row["target_title"] for row in cursor.fetchall()] == [
            "Quantum mechanics",
            "Relativity",
        ]


def test_get_article_reconstructs_metadata(
    test_db_config: DbConfig,
) -> None:
    """Read stored article metadata, categories and outgoing links.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(article, _example_chunks())

        stored = repo.get_article(article.article_id)

        assert stored is not None
        assert stored.article_id == article.article_id
        assert stored.title == article.title
        assert stored.fingerprint == article.fingerprint
        assert stored.revision_id == article.revision_id
        assert stored.topic == article.topic
        assert set(stored.categories) == set(article.categories)
        assert set(stored.linked_titles) == set(article.linked_titles)

        assert repo.get_article(999999) is None


def test_get_article_by_title_matches_exact_case(
    test_db_config: DbConfig,
) -> None:
    """Verify article title lookup uses exact case-sensitive matching.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(article, _example_chunks())

        stored = repo.get_article_by_title("Test Physics Article")

        assert stored is not None
        assert stored.article_id == article.article_id
        assert repo.get_article_by_title("test physics article") is None
        assert repo.get_article_by_title("Nonexistent article") is None


def test_case_distinct_titles_and_links(
    test_db_config: DbConfig,
) -> None:
    """Verify case-distinct titles remain separate during link resolution.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    base = _example_article()

    first = replace(
        base,
        title="Apple",
        linked_titles=(),
    )

    second = replace(
        base,
        article_id=990002,
        title="APPLE",
        fingerprint="b" * 64,
        linked_titles=(),
    )

    source = replace(
        base,
        article_id=990003,
        title="Source Article",
        fingerprint="c" * 64,
        linked_titles=("Apple", "APPLE"),
    )

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(first, _example_chunks())
        repo.ingest_article(second, _example_chunks())
        repo.ingest_article(source, _example_chunks())

        assert repo.get_article_by_title("Apple").article_id == 990001
        assert repo.get_article_by_title("APPLE").article_id == 990002
        assert repo.get_article_by_title("apple") is None

        assert repo.resolve_links() == 2

        linked = repo.get_linked_articles(source.article_id)
        assert [article.article_id for article in linked] == [990001, 990002]


def test_get_chunk_and_neighbors(
    test_db_config: DbConfig,
) -> None:
    """Read a chunk and its surrounding positions in order.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()

    third = ChunkRecord(
        chunk_index=2,
        text="The third paragraph.",
        embedding=(0.0, 0.0, 1.0) + (0.0,) * 381,
    )
    chunks = (*_example_chunks(), third)

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(article, chunks)

        # Obtain the actual auto-generated chunk identifiers.
        with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
            cursor.execute(
                "SELECT chunk_id FROM chunks WHERE article_id = %s ORDER BY chunk_index",
                (article.article_id,),
            )
            ids = [row["chunk_id"] for row in cursor.fetchall()]

        middle = repo.get_chunk(ids[1])

        assert middle is not None
        assert middle.chunk_index == 1
        assert middle.article_title == article.title
        assert middle.text == chunks[1].text

        neighbors = repo.get_neighbor_chunks(ids[1], window=1)
        assert [chunk.chunk_index for chunk in neighbors] == [0, 1, 2]

        center_only = repo.get_neighbor_chunks(ids[1], window=0)
        assert [chunk.chunk_index for chunk in center_only] == [1]

        assert repo.get_chunk(999999) is None
        assert repo.get_neighbor_chunks(999999) == []

        with pytest.raises(ValidationError):
            repo.get_neighbor_chunks(ids[1], window=-1)


def test_list_topics_and_categories(
    test_db_config: DbConfig,
) -> None:
    """Filter category names by article count and list corpus topics.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    first = _example_article()

    second = replace(
        first,
        article_id=990002,
        title="Second Physics Article",
        url="https://en.wikipedia.org/wiki/Second_Physics_Article",
        fingerprint="b" * 64,
        topic="science",
        categories=("Physics",),
        linked_titles=(),
    )

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(first, _example_chunks())
        repo.ingest_article(second, _example_chunks())

        assert repo.list_topics() == ["physics", "science"]
        assert repo.list_categories() == ["Physics", "Science"]
        assert repo.list_categories(min_articles=2) == ["Physics"]
        assert repo.list_categories(min_articles=3) == []

        with pytest.raises(ValidationError):
            repo.list_categories(min_articles=0)


def _linked_target_article() -> ArticleRecord:
    """Create a target article referenced by our existing test article.

    Returns:
        ArticleRecord: Another article in the test corpus.
    """
    return replace(
        _example_article(),
        article_id=990002,
        title="Relativity",
        url="https://en.wikipedia.org/wiki/Relativity",
        fingerprint="b" * 64,
        linked_titles=(),
    )


def test_resolve_links_matches_stored_articles(
    test_db_config: DbConfig,
) -> None:
    """Resolve known targets while leaving unknown titles unresolved.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    source = _example_article()
    target = _linked_target_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())
        repo.ingest_article(target, _example_chunks())

        assert repo.resolve_links() == 1
        assert repo.resolve_links() == 1

        linked = repo.get_linked_articles(source.article_id)

        assert [article.article_id for article in linked] == [target.article_id]
        assert repo.get_linked_articles(target.article_id) == []

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT target_title, target_article_id "
            "FROM links WHERE source_article_id = %s "
            "ORDER BY target_title",
            (source.article_id,),
        )
        rows = cursor.fetchall()

    assert len(rows) == 2
    assert rows[0]["target_title"] == "Quantum mechanics"
    assert rows[0]["target_article_id"] is None
    assert rows[1]["target_title"] == "Relativity"
    assert rows[1]["target_article_id"] == target.article_id


def test_links_resolve_after_later_ingestion(
    test_db_config: DbConfig,
) -> None:
    """Resolve links after their target articles are ingested.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    source = _example_article()
    target = _linked_target_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())

        assert repo.resolve_links() == 0
        assert repo.get_linked_articles(source.article_id) == []

        repo.ingest_article(target, _example_chunks())

        assert repo.resolve_links() == 1
        assert len(repo.get_linked_articles(source.article_id)) == 1


def test_delete_article_cascades_and_unresolves_incoming_links(
    test_db_config: DbConfig,
) -> None:
    """Delete dependent records and clear incoming target references.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    source = _example_article()
    target = _linked_target_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())
        repo.ingest_article(target, _example_chunks())
        repo.resolve_links()

        assert repo.count_articles() == 2
        assert repo.count_chunks() == 4

        assert repo.delete_article(target.article_id) is True

        assert repo.count_articles() == 1
        assert repo.count_chunks() == 2
        assert repo.get_article(target.article_id) is None
        assert repo.get_linked_articles(source.article_id) == []

        # Deleting an already absent article returns False.
        assert repo.delete_article(target.article_id) is False

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT target_article_id FROM links "
            "WHERE source_article_id = %s AND target_title = %s",
            (source.article_id, "Relativity"),
        )
        row = cursor.fetchone()

    assert row is not None
    assert row["target_article_id"] is None


def test_delete_source_removes_outgoing_links(
    test_db_config: DbConfig,
) -> None:
    """Delete an article and its outgoing links and chunks.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    source = _example_article()

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())

        assert repo.delete_article(source.article_id) is True
        assert repo.count_articles() == 0
        assert repo.count_chunks() == 0

    with closing(get_connection(test_db_config)) as conn, conn.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) AS total FROM links WHERE source_article_id = %s",
            (source.article_id,),
        )
        row = cursor.fetchone()

    assert row["total"] == 0


def test_semantic_search_returns_ranked_hits(
    test_db_config: DbConfig,
) -> None:
    """Verify MariaDB returns correctly ranked native vector results.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    article = _example_article()
    query_embedding = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(article, _example_chunks())

        hits = repo.search_chunks(query_embedding, top_k=2)

        assert len(hits) == 2
        assert [hit.chunk_index for hit in hits] == [0, 1]

        assert hits[0].article_id == article.article_id
        assert hits[0].article_title == article.title
        assert hits[0].article_url == article.url
        assert hits[0].text == "Physics explains motion."

        assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

        assert hits[1].distance == pytest.approx(1.0, abs=1e-5)

        limited = repo.search_chunks(query_embedding, top_k=1)

        assert len(limited) == 1
        assert limited[0].chunk_index == 0


def test_semantic_search_returns_empty_list(
    test_db_config: DbConfig,
) -> None:
    """Return no matches when the database contains no chunks.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    query_embedding = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        assert repo.search_chunks(query_embedding) == []


def test_semantic_search_rejects_invalid_inputs(
    test_db_config: DbConfig,
) -> None:
    """Reject invalid query vectors, limits, and unsupported filters.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    valid_vector = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        with pytest.raises(EmbeddingDimensionError):
            repo.search_chunks((1.0,))

        with pytest.raises(EmbeddingDimensionError):
            repo.search_chunks((0.0,) * EMBEDDING_DIM)

        with pytest.raises(ValidationError):
            repo.search_chunks(valid_vector, top_k=0)

        with pytest.raises(ValidationError):
            repo.search_chunks(valid_vector, top_k=101)

        with pytest.raises(ValidationError):
            repo.search_chunks(valid_vector, top_k=True)


def test_hybrid_search_filters_before_top_k(
    test_db_config: DbConfig,
) -> None:
    """Return eligible chunks even when closer chunks fail a filter.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    physics = _example_article()

    biology = replace(
        _example_article(),
        article_id=990002,
        title="Test Biology Article",
        url="https://en.wikipedia.org/wiki/Test_Biology_Article",
        fingerprint="b" * 64,
        topic="biology",
        categories=("Biology",),
        linked_titles=(),
    )

    biology_chunk = ChunkRecord(
        chunk_index=0,
        text="Cells contain genetic material.",
        embedding=(0.8, 0.6) + (0.0,) * (EMBEDDING_DIM - 2),
    )

    query_vector = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(physics, _example_chunks())
        repo.ingest_article(biology, (biology_chunk,))

        # The closest unfiltered result belongs to Physics.
        unfiltered = repo.search_chunks(query_vector, top_k=1)

        assert len(unfiltered) == 1
        assert unfiltered[0].article_id == physics.article_id

        # Biology must still be returned, despite being farther away.
        filtered = repo.search_chunks(
            query_vector,
            top_k=1,
            filters=SearchFilters(categories=("Biology",)),
        )

        assert len(filtered) == 1
        assert filtered[0].article_id == biology.article_id
        assert filtered[0].text == "Cells contain genetic material."
        assert filtered[0].distance == pytest.approx(0.2, abs=1e-5)


def test_hybrid_search_combines_relational_filters(
    test_db_config: DbConfig,
) -> None:
    """Apply metadata, category, link, and distance filters together.

    Args:
        test_db_config: Isolated MariaDB test configuration.
    """
    source = replace(
        _example_article(),
        linked_titles=("Test Biology Article",),
    )

    biology = replace(
        _example_article(),
        article_id=990002,
        title="Test Biology Article",
        url="https://en.wikipedia.org/wiki/Test_Biology_Article",
        fingerprint="b" * 64,
        topic="biology",
        categories=("Biology",),
        length_chars=150,
        revised_at=datetime(2026, 4, 1),
        linked_titles=(),
    )

    biology_chunk = ChunkRecord(
        chunk_index=0,
        text="Biology studies living organisms.",
        embedding=(0.8, 0.6) + (0.0,) * (EMBEDDING_DIM - 2),
    )

    query_vector = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())
        repo.ingest_article(biology, (biology_chunk,))

        assert repo.resolve_links() == 1

        filters = SearchFilters(
            topics=("BIOLOGY",),
            categories=("biology",),
            article_ids=(990002,),
            exclude_article_ids=(990001,),
            min_length_chars=100,
            edited_since=datetime(2026, 2, 1),
            linked_from_article_id=990001,
            max_distance=0.3,
        )

        hits = repo.search_chunks(
            query_vector,
            top_k=5,
            filters=filters,
        )

        assert len(hits) == 1
        assert hits[0].article_id == biology.article_id
        assert hits[0].distance == pytest.approx(0.2, abs=1e-5)

        # A stricter distance threshold excludes the same result.
        stricter = replace(filters, max_distance=0.1)

        assert (
            repo.search_chunks(
                query_vector,
                filters=stricter,
            )
            == []
        )


@pytest.mark.parametrize(
    ("filters", "expected_article_ids"),
    (
        pytest.param(
            SearchFilters(topics=("BIOLOGY",)),
            [990002],
            id="topic-case-insensitive",
        ),
        pytest.param(
            SearchFilters(categories=("biology",)),
            [990002],
            id="category-case-insensitive",
        ),
        pytest.param(
            SearchFilters(article_ids=(990002,)),
            [990002],
            id="include-article",
        ),
        pytest.param(
            SearchFilters(exclude_article_ids=(990001,)),
            [990002],
            id="exclude-article",
        ),
        pytest.param(
            SearchFilters(min_length_chars=100),
            [990002],
            id="minimum-length",
        ),
        pytest.param(
            SearchFilters(edited_since=datetime(2026, 2, 1)),
            [990002],
            id="revision-date",
        ),
        pytest.param(
            SearchFilters(linked_from_article_id=990001),
            [990002],
            id="linked-from-article",
        ),
        pytest.param(
            SearchFilters(max_distance=0.3),
            [990001, 990002],
            id="maximum-distance",
        ),
        pytest.param(
            SearchFilters(min_length_chars=150),
            [],
            id="strict-length-boundary",
        ),
        pytest.param(
            SearchFilters(
                topics=("biology",),
                categories=("Physics",),
            ),
            [],
            id="conflicting-filters",
        ),
    ),
)
def test_hybrid_search_individual_filters(
    test_db_config: DbConfig,
    filters: SearchFilters,
    expected_article_ids: list[int],
) -> None:
    """Verify each hybrid filter against real MariaDB data.

    Args:
        test_db_config: Isolated MariaDB test configuration.
        filters: Search constraints for the current test case.
        expected_article_ids: Expected article IDs in ranked order.
    """
    source = replace(
        _example_article(),
        linked_titles=("Test Biology Article",),
    )

    biology = replace(
        _example_article(),
        article_id=990002,
        title="Test Biology Article",
        url="https://en.wikipedia.org/wiki/Test_Biology_Article",
        fingerprint="b" * 64,
        topic="biology",
        categories=("Biology",),
        length_chars=150,
        revised_at=datetime(2026, 4, 1),
        linked_titles=(),
    )

    biology_chunk = ChunkRecord(
        chunk_index=0,
        text="Biology studies living organisms.",
        embedding=(0.8, 0.6) + (0.0,) * (EMBEDDING_DIM - 2),
    )

    query_vector = (1.0,) + (0.0,) * (EMBEDDING_DIM - 1)

    with WikiRepository(test_db_config) as repo:
        repo.ingest_article(source, _example_chunks())
        repo.ingest_article(biology, (biology_chunk,))

        assert repo.resolve_links() == 1

        hits = repo.search_chunks(
            query_vector,
            top_k=5,
            filters=filters,
        )

        assert [hit.article_id for hit in hits] == expected_article_ids

        # Every returned hit must preserve the shared retrieval contract.
        assert all(hit.score == pytest.approx(1.0 - hit.distance) for hit in hits)
