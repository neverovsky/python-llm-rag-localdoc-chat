
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.chunking import Chunk


class VectorStore:
    def __init__(
        self,
        path: str | Path = "data/qdrant",
        collection_name: str = "localdocs",
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(path=str(path))

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have one vector")

        if not chunks:
            return

        points = []

        for index, (chunk, vector) in enumerate(
            zip(chunks, vectors)
        ):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk.source}:{chunk.page}:{index}:{chunk.text}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "chunk_index": index,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        if not self.client.collection_exists(self.collection_name):
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                **(point.payload or {}),
                "score": point.score,
            }
            for point in response.points
        ]

    def close(self) -> None:
        self.client.close()
