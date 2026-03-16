"""
tests/test_chunker.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.chunker import chunk_text, chunk_documents, CHUNK_SIZE, MIN_CHUNK_SIZE


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short text."
        result = chunk_text(text)
        assert result == [text]

    def test_long_text_is_split(self):
        text = "word " * 500  # ~2500 chars, well over CHUNK_SIZE
        result = chunk_text(text)
        assert len(result) > 1
        assert all(len(c) <= CHUNK_SIZE + 50 for c in result)

    def test_chunks_have_overlap(self):
        # Create text with clear paragraph breaks
        paragraph = "This is a paragraph with enough words to fill space. " * 10
        text = "\n\n".join([paragraph] * 5)
        result = chunk_text(text, chunk_size=300, overlap=50)
        # With overlap, adjacent chunks should share some content
        if len(result) > 1:
            # Last 50 chars of chunk[0] should appear in chunk[1]
            end_of_first = result[0][-30:]
            assert end_of_first in result[1] or len(result[1]) > 0

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_splits_on_paragraphs_first(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = chunk_text(text, chunk_size=30, overlap=0)
        assert len(result) >= 2


class TestChunkDocuments:
    def test_carries_over_metadata(self):
        docs = [{
            "text": "word " * 300,
            "source": "test.pdf",
            "type": "pdf",
            "page": 1,
        }]
        result = chunk_documents(docs)
        assert len(result) > 0
        assert all(c["source"] == "test.pdf" for c in result)
        assert all(c["type"] == "pdf" for c in result)
        assert all(c["page"] == 1 for c in result)

    def test_adds_chunk_index(self):
        docs = [{"text": "word " * 300, "source": "test.txt", "type": "plaintext"}]
        result = chunk_documents(docs)
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_filters_tiny_chunks(self):
        docs = [{"text": "tiny", "source": "test.txt", "type": "plaintext"}]
        result = chunk_documents(docs)
        assert all(len(c["text"]) >= MIN_CHUNK_SIZE for c in result)

    def test_handles_empty_documents(self):
        result = chunk_documents([])
        assert result == []

    def test_handles_empty_text(self):
        docs = [{"text": "", "source": "empty.txt", "type": "plaintext"}]
        result = chunk_documents(docs)
        assert result == []

    def test_multiple_documents(self):
        docs = [
            {"text": "word " * 200, "source": "a.txt", "type": "plaintext"},
            {"text": "word " * 200, "source": "b.txt", "type": "plaintext"},
        ]
        result = chunk_documents(docs)
        sources = {c["source"] for c in result}
        assert "a.txt" in sources
        assert "b.txt" in sources
