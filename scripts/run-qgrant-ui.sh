# not used yet, will be for next step with remote qgrant server
# right now qgrant is local file, use python -m scripts.inspect_qdrant instead
docker stop localdocs-qdrant
docker rm localdocs-qdrant
docker run -d \
  --name localdocs-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/data/qdrant-server:/qdrant/storage" \
  qdrant/qdrant

open http://localhost:6333/dashboard