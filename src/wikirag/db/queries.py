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
