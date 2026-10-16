# localdocs-rag

A console chatbot that answers questions about your PDF files. Everything runs locally: PyMuPDF for text extraction, Ollama for embeddings and answers, Qdrant (local mode) as the vector store.

This is a sample project, created as the companion code for the article {url}. It is a learning project, not production software: one folder of PDFs, a console interface, and small local models. The same story is also in the repo at [docs/article.md](docs/article.md).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) running on `http://localhost:11434`
- About 1.2 GB of disk for the small models (more if you use `qwen3:latest`, 5.2 GB)

## Install

```bash
git clone <this-repo-url>
cd localdocs-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pull the models:

```bash
ollama pull qwen3-embedding:0.6b   # embeddings
ollama pull qwen3:0.6b             # answers (fast, less accurate)
# optional, better answers but slower:
# ollama pull qwen3:latest
```

Check that Ollama is up:

```bash
ollama list
```

## Usage

1. Put your PDFs into `data/pdfs/`.
2. Build the index:

   ```bash
   python -m src.ingest
   ```

3. Ask questions:

   ```bash
   python -m src.chat
   ```

   Type `exit` or `quit` to leave.

Other entry points:

```bash
python -m src.retrieve             # search only, prints the top chunks
python -m scripts.inspect_qdrant   # look at what is stored in the index
```

Run commands from the repo root and use `python -m ...`, since the code imports `src.*`.

## Configuration

There is no config file yet; the settings are plain defaults in the code.

| What | Where |
|---|---|
| Answer model (`qwen3:0.6b`) | `src/chat.py` |
| Embedding model, Ollama host | `src/embeddings.py` |
| Chunk size / overlap (1200 / 200) | `src/chunking.py`, `src/pdf_loader.py` |
| Index folder (`data/qdrant`), collection name | `src/vector_store.py` |
| Number of retrieved chunks (5) | `src/retrieve.py` |

## Tests

```bash
python -m pytest                    # everything, needs Ollama running
python -m pytest -m "not integration"   # skip tests that need Ollama
```

## Troubleshooting

- **`Connection refused` on ingest or chat:** Ollama isn't running. Start it with `ollama serve`.
- **Qdrant "already accessed by another instance":** local mode allows only one process on `data/qdrant`. Close the other script (for example the inspector) and retry.
- **Changed the embedding model:** vector size changes, so delete `data/qdrant` and run `python -m src.ingest` again.
- **Removed or changed a PDF:** ingest only adds and overwrites points, it never deletes. To drop stale data, delete `data/qdrant` and re-ingest.
