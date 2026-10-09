
from pathlib import Path

from src.embeddings import EmbeddingService
from src.pdf_loader import load_pdf_directory
from src.vector_store import VectorStore


def ingest_directory(
    pdf_directory: str | Path = "data/pdfs",
    qdrant_path: str | Path = "data/qdrant",
    batch_size: int = 32,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    chunks = load_pdf_directory(pdf_directory)

    if not chunks:
        print("No PDF text found.")
        return 0

    embedding_service = EmbeddingService()
    vector_store = VectorStore(path=qdrant_path)

    try:
        # Create vectors in batches to limit memory usage.
        first_vectors = embedding_service.embed(
            [chunk.text for chunk in chunks[:batch_size]]
        )

        if not first_vectors:
            print("No embeddings were created.")
            return 0

        vector_store.ensure_collection(
            vector_size=len(first_vectors[0])
        )

        vector_store.upsert(
            chunks[:batch_size],
            first_vectors,
        )

        for start in range(batch_size, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embedding_service.embed(
                [chunk.text for chunk in batch]
            )
            vector_store.upsert(batch, vectors)

        print(f"Indexed {len(chunks)} chunks.")
        return len(chunks)

    finally:
        vector_store.close()


if __name__ == "__main__":
    ingest_directory()
