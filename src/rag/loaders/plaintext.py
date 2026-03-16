"""
loaders/plaintext.py

Loads plain .txt files. Simple — just read and return.
"""

from pathlib import Path


def load_plaintext(path: str | Path) -> list[dict]:
    """
    Load a plain text file and return a single-item list.

    Each dict has:
        text     — full file content
        source   — filename
        type     — "plaintext"
    """
    path = Path(path)

    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[plaintext loader] Error reading {path.name}: {e}")
        return []

    if not text:
        return []

    return [{
        "text": text,
        "source": path.name,
        "type": "plaintext",
    }]
