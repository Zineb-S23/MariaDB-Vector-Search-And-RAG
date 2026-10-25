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
