"""
tests/test_loaders.py

Unit tests for all four document loaders.
All tests run offline — no network calls, no API keys.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPDFLoader:
    def test_extracts_text_from_pdf(self, tmp_path):
        from rag.loaders.pdf import load_pdf
        import PyPDF2

        # Create a minimal PDF-like mock
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page one content."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("rag.loaders.pdf.PyPDF2.PdfReader", return_value=mock_reader):
            fake_pdf = tmp_path / "test.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake")
            result = load_pdf(fake_pdf)

        assert len(result) == 1
        assert result[0]["text"] == "This is page one content."
        assert result[0]["page"] == 1
        assert result[0]["type"] == "pdf"
        assert result[0]["source"] == "test.pdf"

    def test_returns_empty_on_missing_file(self):
        from rag.loaders.pdf import load_pdf
        result = load_pdf("/nonexistent/path/file.pdf")
        assert result == []

    def test_skips_empty_pages(self, tmp_path):
        from rag.loaders.pdf import load_pdf

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "   "  # whitespace only
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Real content here."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]

        with patch("rag.loaders.pdf.PyPDF2.PdfReader", return_value=mock_reader):
            fake_pdf = tmp_path / "test.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake")
            result = load_pdf(fake_pdf)

        assert len(result) == 1
        assert result[0]["text"] == "Real content here."


class TestMarkdownLoader:
    def test_splits_on_headings(self, tmp_path):
        from rag.loaders.markdown import load_markdown

        content = "# Introduction\nThis is the intro.\n\n## Details\nHere are details."
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        result = load_markdown(md_file)
        assert len(result) == 2
        assert result[0]["section"] == "Introduction"
        assert result[1]["section"] == "Details"
        assert all(r["type"] == "markdown" for r in result)

    def test_handles_no_headings(self, tmp_path):
        from rag.loaders.markdown import load_markdown

        content = "Just plain text with no headings at all."
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        result = load_markdown(md_file)
        assert len(result) == 1
        assert result[0]["section"] == "full"

    def test_captures_intro_before_first_heading(self, tmp_path):
        from rag.loaders.markdown import load_markdown

        content = "Intro text before heading.\n\n# Section One\nContent here."
        md_file = tmp_path / "test.md"
        md_file.write_text(content)

        result = load_markdown(md_file)
        assert result[0]["section"] == "intro"
        assert "Intro text" in result[0]["text"]

    def test_returns_empty_on_missing_file(self):
        from rag.loaders.markdown import load_markdown
        result = load_markdown("/nonexistent/file.md")
        assert result == []


class TestPlaintextLoader:
    def test_loads_text_file(self, tmp_path):
        from rag.loaders.plaintext import load_plaintext

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world. This is a test file.")

        result = load_plaintext(txt_file)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world. This is a test file."
        assert result[0]["type"] == "plaintext"
        assert result[0]["source"] == "test.txt"

    def test_returns_empty_for_blank_file(self, tmp_path):
        from rag.loaders.plaintext import load_plaintext

        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("   \n\n  ")
        result = load_plaintext(txt_file)
        assert result == []


class TestURLLoader:
    def test_extracts_text_from_html(self):
        from rag.loaders.url import load_url

        fake_html = """<html><head><title>Test Page</title></head>
        <body><article><h1>Article Title</h1><p>Main content here.</p></article>
        <script>remove this</script><nav>remove nav</nav></body></html>"""

        mock_response = MagicMock()
        mock_response.text = fake_html
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("rag.loaders.url.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.return_value = mock_response
            result = load_url("https://example.com/test")

        assert len(result) == 1
        assert "Article Title" in result[0]["text"]
        assert "Main content" in result[0]["text"]
        assert "remove this" not in result[0]["text"]
        assert result[0]["type"] == "url"
        assert result[0]["title"] == "Test Page"

    def test_handles_http_error(self):
        import httpx
        from rag.loaders.url import load_url

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("rag.loaders.url.httpx.Client") as MockClient:
            MockClient.return_value.__enter__.return_value.get.side_effect = (
                httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_resp)
            )
            result = load_url("https://example.com/missing")

        assert result == []


class TestAutoLoader:
    def test_detects_pdf(self, tmp_path):
        from rag.loaders import load

        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"fake")

        with patch("rag.loaders.pdf.load_pdf", return_value=[{"text": "pdf content", "source": "doc.pdf", "type": "pdf"}]) as mock:
            result = load(str(fake_pdf))
        mock.assert_called_once()

    def test_detects_url(self):
        from rag.loaders import load
        with patch("rag.loaders.url.load_url", return_value=[]) as mock:
            load("https://example.com")
        mock.assert_called_once_with("https://example.com")

    def test_rejects_unsupported_extension(self, tmp_path):
        from rag.loaders import load
        f = tmp_path / "file.xyz"
        f.write_text("content")
        result = load(str(f))
        assert result == []
