
-- MariaDB WikiRAG: native vector and relational SQL demonstrations.
-- Based on the Area 4 blueprint, Section 6.5 (Q1-Q6).
--
-- Prerequisites:
-- 1. The MariaDB 11.8 schema in sql/001_schema.sql is initialized.
-- 2. At least one article with an embedded chunk has been ingested.
-- 3. Q2, Q3, and Q5 may return no rows until matching Wikipedia
--    metadata and resolved links exist.
--
-- Demonstrations use SQL session variables for readability.
-- Application code must use bound parameters for user inputs.
--
-- Q1-Q5 are read-only. Q6 is commented out because it requires
-- two separate database sessions and an explicit rollback.

-- Use one stored vector as a reproducible example query vector.
SET @q = (
    SELECT VEC_ToText(c.embedding)
    FROM chunks AS c
    ORDER BY c.chunk_id
    LIMIT 1
);

-- Change these values to match the ingested Wikipedia corpus.
SET @topic = 'physics';
SET @category = 'Quantum mechanics';
SET @min_length_chars = 5000;
SET @edited_since = '2025-01-01 00:00:00';

-- Choose a source article having at least one resolved link.
SET @hub_article_id = (
    SELECT l.source_article_id
    FROM links AS l
    WHERE l.target_article_id IS NOT NULL
    ORDER BY l.source_article_id
    LIMIT 1
);

-- Q1: Pure semantic top-5 using the native VECTOR index.
-- Rank chunks before joining relational article metadata.
SELECT
    ranked.chunk_id,
    ranked.article_id,
    a.title,
    a.url,
    ranked.distance
FROM (
    SELECT
        c.chunk_id,
        c.article_id,
        VEC_DISTANCE_COSINE(
            c.embedding,
            VEC_FromText(@q)
        ) AS distance
    FROM chunks AS c
    ORDER BY VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(@q)
    ) ASC
    LIMIT 5
) AS ranked
JOIN articles AS a
    ON a.article_id = ranked.article_id
ORDER BY
    ranked.distance ASC,
    ranked.article_id ASC,
    ranked.chunk_id ASC;

-- Q2: Semantic ranking constrained by a relational topic.
-- An exact scan ensures the topic filter applies before LIMIT.
SELECT
    c.chunk_id,
    a.title,
    a.topic,
    VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(@q)
    ) AS distance
FROM chunks AS c IGNORE INDEX (embedding)
JOIN articles AS a
    ON a.article_id = c.article_id
WHERE LOWER(a.topic) = LOWER(@topic)
ORDER BY
    distance ASC,
    a.article_id ASC,
    c.chunk_index ASC,
    c.chunk_id ASC
LIMIT 5;

-- Q3: Semantic ranking constrained by a many-to-many category.
-- Article-category relationships remain inside MariaDB.
SELECT
    c.chunk_id,
    a.title,
    VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(@q)
    ) AS distance
FROM chunks AS c IGNORE INDEX (embedding)
JOIN articles AS a
    ON a.article_id = c.article_id
WHERE EXISTS (
    SELECT 1
    FROM article_categories AS ac
    JOIN categories AS cat
        ON cat.category_id = ac.category_id
    WHERE ac.article_id = a.article_id
      AND LOWER(cat.name) = LOWER(@category)
)
ORDER BY
    distance ASC,
    a.article_id ASC,
    c.chunk_index ASC,
    c.chunk_id ASC
LIMIT 5;

-- Q4: Expand semantic results through the Wikipedia link graph.
-- First retrieve nearby chunks, then join resolved article links.
WITH top_chunks AS (
    SELECT
        c.chunk_id,
        c.article_id
    FROM chunks AS c
    ORDER BY VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(@q)
    ) ASC
    LIMIT 5
)
SELECT DISTINCT
    src.article_id AS source_article_id,
    src.title AS found_in,
    dst.article_id AS linked_article_id,
    dst.title AS linked_article
FROM top_chunks AS t
JOIN articles AS src
    ON src.article_id = t.article_id
JOIN links AS l
    ON l.source_article_id = t.article_id
JOIN articles AS dst
    ON dst.article_id = l.target_article_id
ORDER BY
    src.article_id ASC,
    dst.article_id ASC;

-- Q5: Combine vector similarity with multiple relational filters.
-- Eligible articles are filtered before exact top-K ranking.
SELECT
    c.chunk_id,
    a.article_id,
    a.title,
    a.length_chars,
    a.revised_at,
    VEC_DISTANCE_COSINE(
        c.embedding,
        VEC_FromText(@q)
    ) AS distance
FROM chunks AS c IGNORE INDEX (embedding)
JOIN articles AS a
    ON a.article_id = c.article_id
WHERE a.length_chars > @min_length_chars
  AND a.revised_at >= @edited_since
  AND EXISTS (
      SELECT 1
      FROM article_categories AS ac
      JOIN categories AS cat
          ON cat.category_id = ac.category_id
      WHERE ac.article_id = a.article_id
        AND LOWER(cat.name) = LOWER(@category)
  )
  AND EXISTS (
      SELECT 1
      FROM links AS l
      WHERE l.source_article_id = @hub_article_id
        AND l.target_article_id = a.article_id
  )
ORDER BY
    distance ASC,
    a.article_id ASC,
    c.chunk_index ASC,
    c.chunk_id ASC
LIMIT 5;

-- Q6: Demonstrate transactional consistency across two sessions.
-- This example is deliberately not executed by this SQL file.
--
-- Prerequisite: At least one stored article with an embedding.
-- Use separate MariaDB client sessions A and B.
--
-- SESSION A:
-- Select an existing article and stored vector.
--
-- SET @demo_article_id = (
--     SELECT article_id FROM chunks ORDER BY chunk_id LIMIT 1
-- );
-- SET @demo_vector = (
--     SELECT VEC_ToText(embedding)
--     FROM chunks ORDER BY chunk_id LIMIT 1
-- );
-- SET @demo_chunk_index = (
--     SELECT COALESCE(MAX(chunk_index), -1) + 1
--     FROM chunks WHERE article_id = @demo_article_id
-- );
--
-- START TRANSACTION;
--
-- INSERT INTO chunks (
--     article_id, chunk_index, section_title,
--     text, char_count, embedding
-- )
-- VALUES (
--     @demo_article_id,
--     @demo_chunk_index,
--     NULL,
--     'WikiRAG uncommitted visibility demo',
--     CHAR_LENGTH('WikiRAG uncommitted visibility demo'),
--     VEC_FromText(@demo_vector)
-- );
--
-- Keep transaction A open without committing.
--
-- SESSION B:
-- Run the following SELECT in a separate session:
--
-- SELECT COUNT(*) AS visible_chunks
-- FROM chunks
-- WHERE text = 'WikiRAG uncommitted visibility demo';
--
-- Expected: 0, because session A has not committed.
--
-- SESSION A:
-- ROLLBACK;
--
-- SESSION B:
-- Run the same SELECT again.
--
-- Expected: 0, because the inserted chunk was rolled back.
--
-- This demonstrates visibility of transactional vector storage.
-- Repeat with an explicit COMMIT in an isolated test environment
-- to demonstrate visibility after a successful transaction.
