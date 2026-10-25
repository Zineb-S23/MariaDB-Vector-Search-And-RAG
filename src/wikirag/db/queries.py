"""SQL queries used by the MariaDB WikiRAG repository."""

from __future__ import annotations

from wikirag.models import SearchFilters

# Read the existing embedding model configuration.
GET_EMBEDDING_META = """
SELECT model_name, embedding_dim
FROM embedding_meta
WHERE id = 1
"""

# Insert metadata once without overwriting an existing model.
ENSURE_EMBEDDING_META = """
INSERT INTO embedding_meta (id, model_name, embedding_dim)
VALUES (1, %s, %s)
ON DUPLICATE KEY UPDATE id = id
"""

COUNT_ARTICLES = """
SELECT COUNT(*) AS total
FROM articles
"""

COUNT_CHUNKS = """
SELECT COUNT(*) AS total
FROM chunks
"""

GET_FINGERPRINT = """
SELECT fingerprint
FROM articles
WHERE article_id = %s
"""

# Lock an existing article while deciding whether to replace its data.
GET_FINGERPRINT_FOR_UPDATE = """
SELECT fingerprint
FROM articles
WHERE article_id = %s
FOR UPDATE
"""

INSERT_ARTICLE = """
INSERT INTO articles (
    article_id, title, url, revision_id, revised_at,
    length_chars, fingerprint, topic, language, fetched_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

UPDATE_ARTICLE = """
UPDATE articles
SET title = %s,
    url = %s,
    revision_id = %s,
    revised_at = %s,
    length_chars = %s,
    fingerprint = %s,
    topic = %s,
    language = %s,
    fetched_at = %s
WHERE article_id = %s
"""

DELETE_ARTICLE_CHUNKS = """
DELETE FROM chunks WHERE article_id = %s
"""

DELETE_ARTICLE_CATEGORIES = """
DELETE FROM article_categories WHERE article_id = %s
"""

DELETE_OUTGOING_LINKS = """
DELETE FROM links WHERE source_article_id = %s
"""

INSERT_CHUNK = """
INSERT INTO chunks (
    article_id, chunk_index, section_title,
    text, char_count, embedding
)
VALUES (%s, %s, %s, %s, %s, VEC_FromText(%s))
"""

UPSERT_CATEGORY = """
INSERT INTO categories (name)
VALUES (%s)
ON DUPLICATE KEY UPDATE category_id = category_id
"""

GET_CATEGORY_ID = """
SELECT category_id FROM categories WHERE name = %s
"""

INSERT_ARTICLE_CATEGORY = """
INSERT INTO article_categories (article_id, category_id)
VALUES (%s, %s)
"""

INSERT_OUTGOING_LINK = """
INSERT INTO links (source_article_id, target_title, target_article_id)
VALUES (%s, %s, NULL)
"""

GET_ARTICLE_BY_ID = """
SELECT article_id, title, url, revision_id, revised_at,
       length_chars, fingerprint, topic, language, fetched_at
FROM articles
WHERE article_id = %s
"""

# Match titles exactly using the schema's binary collation.
GET_ARTICLE_BY_TITLE = """
SELECT article_id, title, url, revision_id, revised_at,
       length_chars, fingerprint, topic, language, fetched_at
FROM articles
WHERE title = %s
LIMIT 1
"""

GET_ARTICLE_CATEGORIES = """
SELECT c.name
FROM categories AS c
JOIN article_categories AS ac
    ON ac.category_id = c.category_id
WHERE ac.article_id = %s
ORDER BY c.name
"""

GET_ARTICLE_LINK_TITLES = """
SELECT target_title
FROM links
WHERE source_article_id = %s
ORDER BY target_title
"""

GET_CHUNK = """
SELECT c.chunk_id, c.article_id, a.title AS article_title,
       c.chunk_index, c.section_title, c.text
FROM chunks AS c
JOIN articles AS a ON a.article_id = c.article_id
WHERE c.chunk_id = %s
"""

GET_NEIGHBOR_CHUNKS = """
SELECT c.chunk_id, c.article_id, a.title AS article_title,
       c.chunk_index, c.section_title, c.text
FROM chunks AS c
JOIN articles AS a ON a.article_id = c.article_id
WHERE c.article_id = %s
  AND c.chunk_index BETWEEN %s AND %s
ORDER BY c.chunk_index
"""

LIST_TOPICS = """
SELECT DISTINCT topic
FROM articles
WHERE topic IS NOT NULL AND topic <> ''
ORDER BY topic
"""

LIST_CATEGORIES = """
SELECT c.name
FROM categories AS c
JOIN article_categories AS ac
    ON ac.category_id = c.category_id
GROUP BY c.category_id, c.name
HAVING COUNT(DISTINCT ac.article_id) >= %s
ORDER BY c.name
"""

# Resolve outgoing links only to articles with exactly matching titles.
RESOLVE_LINK_TARGETS = """
UPDATE links AS l
LEFT JOIN articles AS a
    ON l.target_title = a.title
SET l.target_article_id = a.article_id
"""

COUNT_RESOLVED_LINKS = """
SELECT COUNT(*) AS total
FROM links
WHERE target_article_id IS NOT NULL
"""

GET_LINKED_ARTICLES = """
SELECT DISTINCT
    a.article_id, a.title, a.url, a.revision_id,
    a.revised_at, a.length_chars, a.fingerprint,
    a.topic, a.language, a.fetched_at
FROM links AS l
JOIN articles AS a
    ON a.article_id = l.target_article_id
WHERE l.source_article_id = %s
ORDER BY a.article_id
"""

DELETE_ARTICLE_BY_ID = """
DELETE FROM articles
WHERE article_id = %s
"""


# Rank Wikipedia chunks using MariaDB's native cosine vector distance.

# Rank chunks with the native vector index before joining article metadata.
SEARCH_CHUNKS_SEMANTIC = """
SELECT
    ranked.chunk_id,
    ranked.article_id,
    a.title AS article_title,
    a.url AS article_url,
    ranked.chunk_index,
    ranked.section_title,
    ranked.text,
    ranked.distance
FROM (
    SELECT
        c.chunk_id,
        c.article_id,
        c.chunk_index,
        c.section_title,
        c.text,
        VEC_DISTANCE_COSINE(
            c.embedding,
            VEC_FromText(%s)
        ) AS distance
    FROM chunks AS c
    ORDER BY VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(%s)
    ) ASC
    LIMIT %s
) AS ranked
JOIN articles AS a
    ON a.article_id = ranked.article_id
ORDER BY
    ranked.distance ASC,
    ranked.article_id ASC,
    ranked.chunk_index ASC,
    ranked.chunk_id ASC
"""


def build_hybrid_search_query(
    filters: SearchFilters,
    *,
    vector_json: str,
    top_k: int,
) -> tuple[str, tuple[object, ...]]:
    """Build an exact cosine search with relational SQL filters.

    Apply all relational constraints before selecting the nearest
    chunks. The vector index is excluded to avoid selective-filter
    under-fill in this initial implementation.

    Args:
        filters: Validated search constraints.
        vector_json: Validated query embedding serialized as JSON.
        top_k: Validated maximum number of results.

    Returns:
        tuple[str, tuple[object, ...]]: Parameterized SQL and values.
    """
    conditions: list[str] = []
    params: list[object] = [vector_json]

    if filters.topics:
        placeholders = ", ".join("%s" for _ in filters.topics)
        conditions.append(f"LOWER(a.topic) IN ({placeholders})")
        params.extend(topic.casefold() for topic in filters.topics)

    if filters.categories:
        placeholders = ", ".join("%s" for _ in filters.categories)
        conditions.append(
            "EXISTS ("
            "SELECT 1 FROM article_categories AS ac "
            "JOIN categories AS cat ON cat.category_id = ac.category_id "
            "WHERE ac.article_id = a.article_id "
            f"AND LOWER(cat.name) IN ({placeholders})"
            ")"
        )
        params.extend(category.casefold() for category in filters.categories)

    if filters.article_ids:
        placeholders = ", ".join("%s" for _ in filters.article_ids)
        conditions.append(f"a.article_id IN ({placeholders})")
        params.extend(filters.article_ids)

    if filters.exclude_article_ids:
        placeholders = ", ".join("%s" for _ in filters.exclude_article_ids)
        conditions.append(f"a.article_id NOT IN ({placeholders})")
        params.extend(filters.exclude_article_ids)

    if filters.min_length_chars is not None:
        conditions.append("a.length_chars > %s")
        params.append(filters.min_length_chars)

    if filters.edited_since is not None:
        conditions.append("a.revised_at >= %s")
        params.append(filters.edited_since)

    if filters.linked_from_article_id is not None:
        conditions.append(
            "EXISTS ("
            "SELECT 1 FROM links AS l "
            "WHERE l.source_article_id = %s "
            "AND l.target_article_id = a.article_id"
            ")"
        )
        params.append(filters.linked_from_article_id)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    distance_clause = ""
    if filters.max_distance is not None:
        distance_clause = "WHERE ranked.distance <= %s"
        params.append(filters.max_distance)

    params.append(top_k)

    # Preserve parameterized SQL; f-strings are prohibited by our SQL hygiene test.
    statement = "\n".join(  # noqa: FLY002
        (
            """
            SELECT
                ranked.chunk_id,
                ranked.article_id,
                ranked.article_title,
                ranked.article_url,
                ranked.chunk_index,
                ranked.section_title,
                ranked.text,
                ranked.distance
            FROM (
                SELECT
                    c.chunk_id,
                    c.article_id,
                    a.title AS article_title,
                    a.url AS article_url,
                    c.chunk_index,
                    c.section_title,
                    c.text,
                    VEC_DISTANCE_COSINE(
                        c.embedding,
                        VEC_FromText(%s)
                    ) AS distance
                FROM chunks AS c IGNORE INDEX (embedding)
                JOIN articles AS a
                    ON a.article_id = c.article_id
            """,
            where_clause,
            """
            ) AS ranked
            """,
            distance_clause,
            """
            ORDER BY
                ranked.distance ASC,
                ranked.article_id ASC,
                ranked.chunk_index ASC,
                ranked.chunk_id ASC
            LIMIT %s
            """,
        )
    )

    return statement, tuple(params)

    return statement, tuple(params)
