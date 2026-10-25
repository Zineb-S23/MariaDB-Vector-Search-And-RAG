"""Unit tests for retrieval evaluation metrics."""

import pytest

from wikirag.errors import ValidationError
from wikirag.evaluation.metrics import (
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.mark.unit
def test_recall_at_k_counts_relevant_articles_in_top_k() -> None:
    assert recall_at_k([10, 20, 30], {20, 30, 40}, 2) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_recall_at_k_returns_zero_when_no_relevant_article_is_retrieved() -> None:
    assert recall_at_k([10, 20], {30}, 2) == 0.0


@pytest.mark.unit
def test_recall_at_k_uses_available_results_when_fewer_than_k() -> None:
    assert recall_at_k([10, 20], {20, 30}, 5) == pytest.approx(1 / 2)


@pytest.mark.unit
def test_precision_at_k_divides_by_k_even_when_fewer_results_exist() -> None:
    assert precision_at_k([10, 20], {20, 30}, 5) == pytest.approx(1 / 5)


@pytest.mark.unit
def test_precision_at_k_counts_relevant_articles_in_top_k() -> None:
    assert precision_at_k([10, 20, 30], {20, 30}, 2) == pytest.approx(1 / 2)


@pytest.mark.unit
def test_reciprocal_rank_uses_first_relevant_result() -> None:
    assert reciprocal_rank([10, 20, 30], {20, 30}) == pytest.approx(1 / 2)


@pytest.mark.unit
def test_reciprocal_rank_returns_zero_when_no_relevant_result_exists() -> None:
    assert reciprocal_rank([10, 20], {30}) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_recall_at_k_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValidationError):
        recall_at_k([10], {10}, k)


@pytest.mark.unit
@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_precision_at_k_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValidationError):
        precision_at_k([10], {10}, k)


@pytest.mark.unit
def test_retrieval_metrics_reject_empty_relevant_collection() -> None:
    with pytest.raises(ValidationError):
        recall_at_k([10], set(), 1)

    with pytest.raises(ValidationError):
        precision_at_k([10], set(), 1)

    with pytest.raises(ValidationError):
        reciprocal_rank([10], set())


@pytest.mark.unit
def test_mean_returns_arithmetic_mean() -> None:
    assert mean([1.0, 2.0, 6.0]) == pytest.approx(3.0)


@pytest.mark.unit
def test_mean_accepts_negative_and_fractional_values() -> None:
    assert mean([-1.0, 0.5, 2.0]) == pytest.approx(0.5)


@pytest.mark.unit
def test_mean_rejects_empty_sequence() -> None:
    with pytest.raises(ValidationError):
        mean([])


@pytest.mark.unit
def test_precision_at_k_deduplicates_article_ids_preserving_rank_order() -> None:
    assert precision_at_k([10, 10, 20], {20}, 2) == pytest.approx(1 / 2)


@pytest.mark.unit
def test_reciprocal_rank_deduplicates_article_ids_preserving_rank_order() -> None:
    assert reciprocal_rank([10, 10, 20], {20}) == pytest.approx(1 / 2)
