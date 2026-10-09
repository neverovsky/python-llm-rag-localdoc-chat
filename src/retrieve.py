
from pathlib import Path

from src.embeddings import EmbeddingService
from src.vector_store import VectorStore
from time import perf_counter


def retrieve(
    question: str,
    qdrant_path: str | Path = "data/qdrant",
    limit: int = 5,
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question must not be empty")

    embedding_service = EmbeddingService()
    vector_store = VectorStore(path=qdrant_path)

    try:
        start = perf_counter()
        vectors = embedding_service.embed([question])
        embedding_time = perf_counter() - start

        if not vectors:
            return []

        start = perf_counter()
        results = vector_store.search(
            query_vector=vectors[0],
            limit=limit,
        )
        search_time = perf_counter() - start

        print(f"[Timing] Embedding: {embedding_time:.2f}s")
        print(f"[Timing] Qdrant search: {search_time:.2f}s")

        return results
    finally:
        vector_store.close()

if __name__ == "__main__":
    question = input("Ask a question: ")

    results = retrieve(question)

    if not results:
        print("No matching documents found.")
    else:
        for index, result in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Source: {result.get('source')}")
            print(f"Page: {result.get('page')}")
            print(f"Score: {result.get('score', 0):.4f}")
            print(result.get("text", ""))
