"""Evaluation metrics for ranked article retrieval."""

from collections.abc import Collection, Sequence

from wikirag.errors import ValidationError


def _validate_inputs(
    retrieved: Sequence[int],
    relevant: Collection[int],
) -> None:
    """Validate inputs shared by the retrieval metrics."""
    if not relevant:
        raise ValidationError("relevant must not be empty")


def recall_at_k(
    retrieved: Sequence[int],
    relevant: Collection[int],
    k: int,
) -> float:
    """Return the fraction of relevant articles retrieved within the first k results."""
    _validate_inputs(retrieved, relevant)

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValidationError("k must be a positive integer")

    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_relevant = set(unique_retrieved[:k]) & set(relevant)
    return len(retrieved_relevant) / len(relevant)


def precision_at_k(
    retrieved: Sequence[int],
    relevant: Collection[int],
    k: int,
) -> float:
    """Return the fraction of the first k results that are relevant."""
    _validate_inputs(retrieved, relevant)

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValidationError("k must be a positive integer")

    unique_retrieved = list(dict.fromkeys(retrieved))
    retrieved_relevant = set(unique_retrieved[:k]) & set(relevant)
    return len(retrieved_relevant) / k


def reciprocal_rank(
    retrieved: Sequence[int],
    relevant: Collection[int],
) -> float:
    """Return the reciprocal rank of the first relevant result, or zero."""
    _validate_inputs(retrieved, relevant)

    unique_retrieved = dict.fromkeys(retrieved)
    for rank, article_id in enumerate(unique_retrieved, start=1):
        if article_id in relevant:
            return 1.0 / rank

    return 0.0


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence of values."""
    if not values:
        raise ValidationError("values must not be empty")

    return sum(values) / len(values)
