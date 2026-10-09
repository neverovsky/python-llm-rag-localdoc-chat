
import pytest

from src.embeddings import EmbeddingService


@pytest.mark.integration
def test_embedding_model_returns_vectors():
    service = EmbeddingService()

    vectors = service.embed([
        "The server returns HTTP 500.",
        "The database connection failed.",
    ])

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])
    assert all(
        isinstance(value, (int, float))
        for vector in vectors
        for value in vector
    )
