
"""
Shared data models for the MariaDB WikiRAG project.

These models define the contracts between Wikipedia ingestion,
database storage, semantic retrieval, RAG, and evaluation.

All dataclasses use frozen=True and slots=True to prevent accidental
field reassignment and reduce memory overhead.

The public field names and types must remain consistent across modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Shared operation and retrieval mode names.
IngestOutcome = Literal["inserted", "updated", "skipped"]
RetrievalMode = Literal["semantic", "hybrid"]


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    """
    Represent Wikipedia article metadata passed from ingestion to database.

    Attributes:
        article_id (int): Unique Wikipedia page identifier.
        title (str): Wikipedia article title.
        url (str): Full URL of the article.
        revision_id (int): Wikipedia revision identifier.
        fingerprint (str): SHA-256 fingerprint, represented as 64 hex characters.
        revised_at (datetime): Article revision timestamp in naive UTC.
        length_chars (int): Character count of the cleaned article text.
        fetched_at (datetime): Timestamp when the article was fetched.
        topic (str | None): Optional corpus topic assigned to the article.
        language (str): Wikipedia language code, defaulting to English.
        categories (tuple[str, ...]): Categories associated with the article.
        linked_titles (tuple[str, ...]): Titles of linked Wikipedia articles.
    """

    article_id: int
    title: str
    url: str
    revision_id: int
    fingerprint: str
    revised_at: datetime
    length_chars: int
    fetched_at: datetime
    topic: str | None = None
    language: str = "en"
    categories: tuple[str, ...] = ()
    linked_titles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """
    Represent one article text chunk and its embedding before storage.

    Attributes:
        chunk_index (int): Zero-based position of the chunk in its article.
        text (str): Cleaned text belonging to the chunk.
        embedding (tuple[float, ...]): Embedding vector with 384 finite values.
        section_title (str | None): Optional section containing the chunk.
    """

    chunk_index: int
    text: str
    embedding: tuple[float, ...]
    section_title: str | None = None


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """
    Represent an article chunk retrieved from MariaDB.

    Attributes:
        chunk_id (int): Database-generated unique chunk identifier.
        article_id (int): Identifier of the parent Wikipedia article.
        article_title (str): Title of the parent article.
        chunk_index (int): Position of the chunk within its article.
        section_title (str | None): Optional section heading.
        text (str): Stored chunk text.
    """

    chunk_id: int
    article_id: int
    article_title: str
    chunk_index: int
    section_title: str | None
    text: str


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """
    Represent optional constraints for database-native hybrid retrieval.

    Empty tuple fields mean that the corresponding filter is disabled.
    Optional scalar fields are disabled when set to None.

    Attributes:
        topics (tuple[str, ...]): Allowed corpus topics.
        categories (tuple[str, ...]): Allowed Wikipedia categories.
        article_ids (tuple[int, ...]): Article identifiers to include.
        exclude_article_ids (tuple[int, ...]): Article identifiers to exclude.
        min_length_chars (int | None): Minimum cleaned article length.
        edited_since (datetime | None): Earliest allowed revision timestamp.
        linked_from_article_id (int | None): Restrict results to articles
            linked from the specified article.
        max_distance (float | None): Maximum allowed vector distance.
    """

    topics: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    article_ids: tuple[int, ...] = ()
    exclude_article_ids: tuple[int, ...] = ()
    min_length_chars: int | None = None
    edited_since: datetime | None = None
    linked_from_article_id: int | None = None
    max_distance: float | None = None

    def is_empty(self) -> bool:
        """
        Determine whether all retrieval filters are disabled.

        Returns:
            bool: True if every filter is unset, otherwise False.
        """

        # Check scalar values against None so max_distance=0.0 remains active.
        return (
            not self.topics
            and not self.categories
            and not self.article_ids
            and not self.exclude_article_ids
            and self.min_length_chars is None
            and self.edited_since is None
            and self.linked_from_article_id is None
            and self.max_distance is None
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """
    Represent one ranked result returned by database vector search.

    Attributes:
        chunk_id (int): Unique identifier of the matching chunk.
        article_id (int): Wikipedia identifier of the source article.
        article_title (str): Title of the source article.
        article_url (str): URL used for source attribution.
        chunk_index (int): Position of the matching chunk.
        section_title (str | None): Optional source section heading.
        text (str): Retrieved chunk text.
        distance (float): Vector distance between query and chunk.
        score (float): Similarity score derived from the distance.
    """

    chunk_id: int
    article_id: int
    article_title: str
    article_url: str
    chunk_index: int
    section_title: str | None
    text: str
    distance: float
    score: float


@dataclass(frozen=True, slots=True)
class IngestReport:
    """
    Summarize the outcome of an article ingestion run.

    Attributes:
        inserted (int): Number of newly inserted articles.
        updated (int): Number of existing articles updated.
        skipped (int): Number of unchanged articles skipped.
        failed (tuple[tuple[str, str], ...]): Failed article titles
            paired with their error messages.
    """

    inserted: int
    updated: int
    skipped: int
    failed: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Store the complete result of one retrieval operation.

    Attributes:
        query (str): Original user query.
        mode (RetrievalMode): Semantic or hybrid retrieval mode.
        top_k (int): Maximum number of requested search results.
        filters (SearchFilters | None): Applied retrieval filters.
        hits (tuple[SearchHit, ...]): Ranked retrieved chunks.
        embed_ms (float): Query embedding time in milliseconds.
        search_ms (float): Database search time in milliseconds.
        latency_ms (float): Total retrieval latency in milliseconds.
    """

    query: str
    mode: RetrievalMode
    top_k: int
    filters: SearchFilters | None
    hits: tuple[SearchHit, ...]
    embed_ms: float
    search_ms: float
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """
    Store formatted source context provided to a RAG generator.

    Attributes:
        text (str): Numbered source passages with titles and citations.
        sources (tuple[SearchHit, ...]): Sources in citation order.
        truncated (bool): Whether the context size limit excluded content.
    """

    text: str
    sources: tuple[SearchHit, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """
    Represent an answer produced by the RAG pipeline.

    Attributes:
        question (str): Original user question.
        answer (str): Generated or extractive answer text.
        context (ContextBundle): Source context provided to the generator.
        retrieval (RetrievalResult): Retrieval output used for the answer.
        generator_name (str): Name of the generator implementation.
        latency_ms (float): Total RAG pipeline latency in milliseconds.
    """

    question: str
    answer: str
    context: ContextBundle
    retrieval: RetrievalResult
    generator_name: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """
    Represent one labeled retrieval evaluation question.

    Attributes:
        query_id (str): Unique identifier of the evaluation question.
        text (str): Natural-language query.
        relevant_article_ids (frozenset[int]): Non-empty collection
            of correct source article identifiers.
        filters (SearchFilters | None): Optional hybrid retrieval filters.
        notes (str): Optional notes about the expected results.
    """

    query_id: str
    text: str
    relevant_article_ids: frozenset[int]
    filters: SearchFilters | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    """
    Store aggregated evaluation metrics for one retrieval mode.

    Attributes:
        n_queries (int): Number of evaluated queries.
        recall_at_k (dict[int, float]): Recall for each evaluated k.
        precision_at_k (dict[int, float]): Precision for each evaluated k.
        mrr (float): Mean reciprocal rank.
        latency_ms_p50 (float): Median retrieval latency in milliseconds.
        latency_ms_p95 (float): 95th-percentile retrieval latency.
    """

    n_queries: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    latency_ms_p50: float
    latency_ms_p95: float


@dataclass(frozen=True, slots=True)
class EvalReport:
    """
    Store corpus information and complete retrieval evaluation results.

    Attributes:
        embedding_model (str): Name of the model used for evaluation.
        corpus_articles (int): Number of articles in the evaluated corpus.
        corpus_chunks (int): Number of stored chunks in the corpus.
        ks (tuple[int, ...]): Values of k used for retrieval evaluation.
        modes (dict[str, ModeMetrics]): Aggregated metrics by mode.
        per_query (tuple[dict, ...]): Per-query evaluation measurements,
            including ranks, reciprocal rank, and latency.
    """

    embedding_model: str
    corpus_articles: int
    corpus_chunks: int
    ks: tuple[int, ...]
    modes: dict[str, ModeMetrics]
    per_query: tuple[dict, ...]
