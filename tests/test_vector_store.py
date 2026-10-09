
import pytest

from src.chunking import Chunk
from src.vector_store import VectorStore


def test_upsert_and_search(tmp_path):
    store = VectorStore(path=tmp_path)

    try:
        chunks = [
            Chunk(
                text="The database connection failed.",
                source="manual.pdf",
                page=1,
            ),
            Chunk(
                text="The server returned HTTP 500.",
                source="manual.pdf",
                page=2,
            ),
        ]

        # Test vectors only. Real embeddings come from Ollama.
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        store.ensure_collection(vector_size=3)
        store.upsert(chunks, vectors)

        results = store.search(
            query_vector=[0.9, 0.1, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0]["text"] == chunks[0].text
        assert results[0]["source"] == "manual.pdf"
        assert results[0]["page"] == 1
        assert "score" in results[0]

    finally:
        store.close()


def test_upsert_rejects_mismatched_vectors(tmp_path):
    store = VectorStore(path=tmp_path)

    try:
        chunks = [
            Chunk(
                text="Some documentation",
                source="manual.pdf",
                page=1,
            )
        ]

        with pytest.raises(ValueError):
            store.upsert(chunks, [])

    finally:
        store.close()


def test_search_returns_empty_for_missing_collection(tmp_path):
    store = VectorStore(path=tmp_path)

    try:
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    finally:
        store.close()
