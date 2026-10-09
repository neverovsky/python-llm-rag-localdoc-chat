from src.vector_store import VectorStore

store = VectorStore(path="data/qdrant")

try:
    info = store.client.get_collection("localdocs")

    print(f"Points: {info.points_count}")
    print(f"Status: {info.status}")

    points, _ = store.client.scroll(
        collection_name="localdocs",
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload or {}

        print("\n---")
        print(f"Source: {payload.get('source')}")
        print(f"Page: {payload.get('page')}")
        print(f"Text: {payload.get('text')}")

finally:
    store.close()