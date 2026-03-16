"""
loaders/__init__.py

Single entry point: load(source) figures out the file type
and calls the right loader automatically.
"""

from pathlib import Path
from .pdf import load_pdf
from .markdown import load_markdown
from .plaintext import load_plaintext
from .url import load_url

LOADERS = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_plaintext,
}


def load(source: str) -> list[dict]:
    """
    Load any supported source — file path or URL.

    Automatically detects type from extension or URL prefix.
    Returns a list of dicts with 'text' and metadata fields.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)

    path = Path(source)
    if not path.exists():
        print(f"[loader] File not found: {source}")
        return []

    suffix = path.suffix.lower()
    loader = LOADERS.get(suffix)

    if not loader:
        print(f"[loader] Unsupported file type: {suffix}. Supported: {list(LOADERS.keys())}")
        return []

    return loader(path)


__all__ = ["load", "load_pdf", "load_markdown", "load_plaintext", "load_url"]
