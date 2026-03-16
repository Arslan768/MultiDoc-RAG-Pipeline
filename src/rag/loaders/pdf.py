"""
loaders/pdf.py

Extracts text from PDF files page by page.
Returns a list of dicts — one per page — with text + metadata.
"""

from pathlib import Path
import PyPDF2


def load_pdf(path: str | Path) -> list[dict]:
    """
    Load a PDF and return a list of page dicts.

    Each dict has:
        text     — raw text of the page
        source   — filename
        page     — page number (1-indexed)
        type     — "pdf"

    Returns empty list on failure (never raises).
    """
    path = Path(path)
    pages = []

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({
                        "text": text,
                        "source": path.name,
                        "page": i + 1,
                        "type": "pdf",
                    })
    except Exception as e:
        print(f"[pdf loader] Error reading {path.name}: {e}")

    return pages
