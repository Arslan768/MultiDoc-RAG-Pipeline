"""
tests/test_pipeline.py

Integration tests for retriever and the full generate() pipeline.
All external calls (ChromaDB, Gemini) are mocked.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestRetriever:
    def test_returns_empty_when_no_collection(self):
        from rag.retriever import retrieve

        with patch("rag.retriever.chromadb.PersistentClient") as MockClient:
            MockClient.return_value.get_collection.side_effect = Exception("No collection")
            result = retrieve("test query")

        assert result == []

    def test_returns_empty_when_collection_empty(self):
        from rag.retriever import retrieve

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch("rag.retriever.chromadb.PersistentClient") as MockClient:
            MockClient.return_value.get_collection.return_value = mock_collection
            result = retrieve("test query")

        assert result == []

    def test_returns_ranked_results(self):
        from rag.retriever import retrieve

        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "documents": [["chunk one text", "chunk two text"]],
            "metadatas": [[
                {"source": "a.pdf", "type": "pdf"},
                {"source": "b.md", "type": "markdown"},
            ]],
            "distances": [[0.1, 0.4]],  # lower = more similar
        }

        with patch("rag.retriever.chromadb.PersistentClient") as MockClient:
            MockClient.return_value.get_collection.return_value = mock_collection
            with patch("rag.retriever.embed_query", return_value=[0.1] * 768):
                result = retrieve("test query", top_k=2)

        assert len(result) == 2
        # First result should have higher score (distance 0.1 → score 0.9)
        assert result[0]["score"] > result[1]["score"]
        assert result[0]["source"] == "a.pdf"

    def test_score_is_one_minus_distance(self):
        from rag.retriever import retrieve

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["some text"]],
            "metadatas": [[{"source": "test.pdf"}]],
            "distances": [[0.25]],
        }

        with patch("rag.retriever.chromadb.PersistentClient") as MockClient:
            MockClient.return_value.get_collection.return_value = mock_collection
            with patch("rag.retriever.embed_query", return_value=[0.1] * 768):
                result = retrieve("query")

        assert result[0]["score"] == round(1 - 0.25, 4)


class TestGenerator:
    def test_returns_no_docs_message_when_empty(self):
        from rag.generator import generate

        with patch("rag.generator.retrieve", return_value=[]):
            result = generate("What is X?")

        assert "No relevant documents" in result["answer"]
        assert result["sources"] == []
        assert result["chunks"] == []

    def test_full_pipeline_returns_answer(self):
        from rag.generator import generate

        mock_chunks = [
            {"text": "Paris is the capital of France.", "source": "geography.txt", "score": 0.9, "metadata": {}},
            {"text": "France is in Western Europe.", "source": "geography.txt", "score": 0.85, "metadata": {}},
        ]

        mock_response = MagicMock()
        mock_response.text = "Paris is the capital of France. Sources: [Source 1]"

        with patch("rag.generator.retrieve", return_value=mock_chunks):
            with patch("rag.generator.genai.Client") as MockClient:
                MockClient.return_value.models.generate_content.return_value = mock_response
                with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                    result = generate("What is the capital of France?")

        assert "Paris" in result["answer"]
        assert "geography.txt" in result["sources"]
        assert len(result["chunks"]) == 2

    def test_deduplicates_sources(self):
        from rag.generator import generate

        mock_chunks = [
            {"text": "chunk 1", "source": "doc.pdf", "score": 0.9, "metadata": {}},
            {"text": "chunk 2", "source": "doc.pdf", "score": 0.8, "metadata": {}},
            {"text": "chunk 3", "source": "other.txt", "score": 0.7, "metadata": {}},
        ]

        mock_response = MagicMock()
        mock_response.text = "Answer here."

        with patch("rag.generator.retrieve", return_value=mock_chunks):
            with patch("rag.generator.genai.Client") as MockClient:
                MockClient.return_value.models.generate_content.return_value = mock_response
                with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                    result = generate("question")

        # doc.pdf appears twice in chunks but once in sources
        assert result["sources"].count("doc.pdf") == 1
        assert len(result["sources"]) == 2

    def test_returns_error_without_api_key(self):
        from rag.generator import generate

        mock_chunks = [{"text": "some text", "source": "doc.txt", "score": 0.9, "metadata": {}}]

        with patch("rag.generator.retrieve", return_value=mock_chunks):
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            with patch.dict("os.environ", {}, clear=True):
                result = generate("question")

        assert "Error" in result["answer"]
