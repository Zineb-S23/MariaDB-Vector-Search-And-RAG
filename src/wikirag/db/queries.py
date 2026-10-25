"""SQL queries used by the MariaDB WikiRAG repository."""

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
