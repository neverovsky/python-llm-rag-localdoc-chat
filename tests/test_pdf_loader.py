
import pymupdf

from src.pdf_loader import load_pdf, load_pdf_directory


def create_test_pdf(path):
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "RAG retrieves relevant documents.")

        page = document.new_page()
        page.insert_text((72, 72), "Qdrant stores vectors.")

        document.save(path)


def test_load_pdf_extracts_pages(tmp_path):
    pdf_path = tmp_path / "manual.pdf"
    create_test_pdf(pdf_path)

    chunks = load_pdf(pdf_path, chunk_size=100, overlap=10)

    assert len(chunks) == 2
    assert chunks[0].source == "manual.pdf"
    assert chunks[0].page == 1
    assert "RAG retrieves" in chunks[0].text
    assert chunks[1].page == 2
    assert "Qdrant stores vectors" in chunks[1].text


def test_load_pdf_directory(tmp_path):
    create_test_pdf(tmp_path / "manual.pdf")

    chunks = load_pdf_directory(tmp_path)

    assert len(chunks) == 2
    assert {chunk.source for chunk in chunks} == {"manual.pdf"}


def test_empty_directory(tmp_path):
    assert load_pdf_directory(tmp_path) == []
