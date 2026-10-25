
-- Record the embedding model and vector dimension used by this database.
CREATE TABLE embedding_meta (
    id TINYINT UNSIGNED NOT NULL PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    embedding_dim SMALLINT UNSIGNED NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_embedding_meta_single_row CHECK (id = 1)
) ENGINE=InnoDB;

-- Store Wikipedia article metadata alongside relational search attributes.
CREATE TABLE articles (
    article_id BIGINT UNSIGNED NOT NULL PRIMARY KEY,
    title VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    url VARCHAR(512) NOT NULL,
    revision_id BIGINT UNSIGNED NOT NULL,
    revised_at DATETIME NOT NULL,
    length_chars INT UNSIGNED NOT NULL,
    fingerprint CHAR(64) NOT NULL,
    topic VARCHAR(64) NULL,
    language CHAR(2) NOT NULL DEFAULT 'en',
    fetched_at DATETIME NOT NULL,
    ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_articles_title (title),
    KEY ix_articles_topic (topic),
    KEY ix_articles_revised (revised_at),
    KEY ix_articles_length (length_chars)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Store article chunks with MariaDB's native cosine HNSW vector index.
CREATE TABLE chunks (
    chunk_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    article_id BIGINT UNSIGNED NOT NULL,
    chunk_index INT UNSIGNED NOT NULL,
    section_title VARCHAR(255) NULL,
    `text` TEXT NOT NULL,
    char_count INT UNSIGNED NOT NULL,
    embedding VECTOR(384) NOT NULL,

    UNIQUE KEY uq_chunks_article_idx (article_id, chunk_index),

    CONSTRAINT fk_chunks_article
        FOREIGN KEY (article_id)
        REFERENCES articles (article_id)
        ON DELETE CASCADE,

    VECTOR INDEX (embedding) M=16 DISTANCE=cosine
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Keep category names in a separate relational table.
CREATE TABLE categories (
    category_id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,

    UNIQUE KEY uq_categories_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Associate articles with multiple Wikipedia categories.
CREATE TABLE article_categories (
    article_id BIGINT UNSIGNED NOT NULL,
    category_id INT UNSIGNED NOT NULL,

    PRIMARY KEY (article_id, category_id),
    KEY ix_ac_category (category_id),

    CONSTRAINT fk_ac_article
        FOREIGN KEY (article_id)
        REFERENCES articles (article_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_ac_category
        FOREIGN KEY (category_id)
        REFERENCES categories (category_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- Preserve outgoing article links even when targets are outside the corpus.
CREATE TABLE links (
    source_article_id BIGINT UNSIGNED NOT NULL,
    target_title VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    target_article_id BIGINT UNSIGNED NULL,

    PRIMARY KEY (source_article_id, target_title),
    KEY ix_links_target (target_article_id),

    CONSTRAINT fk_links_source
        FOREIGN KEY (source_article_id)
        REFERENCES articles (article_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_links_target
        FOREIGN KEY (target_article_id)
        REFERENCES articles (article_id)
        ON DELETE SET NULL
) ENGINE=InnoDB;
