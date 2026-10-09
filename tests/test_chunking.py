
import pytest

from src.chunking import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   \n ") == []


def test_short_text_returns_one_chunk():
    assert split_text("Hello RAG") == ["Hello RAG"]


def test_long_text_is_split():
    text = "a" * 2500

    chunks = split_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) == 3
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_chunks_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_text(text, chunk_size=10, overlap=3)

    assert chunks[0][-3:] == chunks[1][:3]


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("test", chunk_size=0)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("test", chunk_size=10, overlap=10)
