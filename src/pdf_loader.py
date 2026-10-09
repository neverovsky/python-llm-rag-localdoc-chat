
from pathlib import Path

import pymupdf

from src.chunking import Chunk, split_text


def load_pdf(
    pdf_path: str | Path,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Extract text from a PDF and split each page into chunks."""
    path = Path(pdf_path)
    chunks: list[Chunk] = []

    with pymupdf.open(path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text")

            page_chunks = split_text(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk_text in page_chunks:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=path.name,
                        page=page_index + 1,
                    )
                )

    return chunks


def load_pdf_directory(
    directory: str | Path,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[Chunk]:
    """Load all PDF files from a directory."""
    pdf_directory = Path(directory)

    chunks: list[Chunk] = []

    for pdf_path in sorted(pdf_directory.glob("*.pdf")):
        chunks.extend(
            load_pdf(
                pdf_path,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return chunks
