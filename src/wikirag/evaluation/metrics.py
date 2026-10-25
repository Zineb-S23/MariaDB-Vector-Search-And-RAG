"""Evaluation metrics for ranked article retrieval."""

from collections.abc import Collection, Sequence

from wikirag.errors import ValidationError


def _validate_relevant(relevant: Collection[int]) -> set[int]:
    """Validate and normalize ground-truth relevant article IDs.

    Args:
        relevant: Ground-truth relevant article IDs.

    Returns:
        The unique relevant article IDs.

    Raises:
        ValidationError: If no relevant article IDs are provided.
    """
    relevant_ids = set(relevant)
    if not relevant_ids:
        raise ValidationError("relevant must not be empty")
    return relevant_ids


def recall_at_k(
    retrieved: Sequence[int],
    relevant: Collection[int],
    k: int,
) -> float:
    """Calculate Recall@K for ranked article retrieval.

    Args:
        retrieved: Ranked article IDs returned by retrieval.
        relevant: Ground-truth relevant article IDs.
        k: Number of top-ranked unique article IDs to consider.

    Returns:
        The fraction of unique relevant article IDs retrieved in the top K.

    Raises:
        ValidationError: If relevant is empty or k is not a positive integer.
    """
    relevant_ids = _validate_relevant(relevant)

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValidationError("k must be a positive integer")

    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_relevant = set(unique_retrieved[:k]) & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)


def precision_at_k(
    retrieved: Sequence[int],
    relevant: Collection[int],
    k: int,
) -> float:
    """Calculate Precision@K for ranked article retrieval.

    Args:
        retrieved: Ranked article IDs returned by retrieval.
        relevant: Ground-truth relevant article IDs.
        k: Number of top-ranked unique article IDs to consider.

    Returns:
        The number of relevant article IDs in the top K divided by K.

    Raises:
        ValidationError: If relevant is empty or k is not a positive integer.
    """
    relevant_ids = _validate_relevant(relevant)

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValidationError("k must be a positive integer")

    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_relevant = set(unique_retrieved[:k]) & relevant_ids
    return len(retrieved_relevant) / k


def reciprocal_rank(
    retrieved: Sequence[int],
    relevant: Collection[int],
) -> float:
    """Calculate the reciprocal rank of the first relevant article.

    Args:
        retrieved: Ranked article IDs returned by retrieval.
        relevant: Ground-truth relevant article IDs.

    Returns:
        The reciprocal rank of the first relevant article, or 0.0 if none
        is retrieved.

    Raises:
        ValidationError: If relevant is empty.
    """
    relevant_ids = _validate_relevant(relevant)

    unique_retrieved = dict.fromkeys(retrieved)
    for rank, article_id in enumerate(unique_retrieved, start=1):
        if article_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean(values: Sequence[float]) -> float:
    """Calculate the arithmetic mean of a non-empty sequence of values.

    Args:
        values: Numeric values to average.

    Returns:
        The arithmetic mean of the values.

    Raises:
        ValidationError: If values is empty.
    """
    if not values:
        raise ValidationError("values must not be empty")

    return sum(values) / len(values)
