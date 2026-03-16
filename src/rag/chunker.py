"""
chunker.py

Splits loaded documents into smaller chunks suitable for embedding.

Why chunking matters:
- Embedding models have token limits (~2048 tokens for text-embedding-004)
- Smaller chunks = more precise retrieval
- Overlap ensures context isn't lost at chunk boundaries

Strategy: recursive character splitting
  Try to split on paragraphs first (\n\n), then sentences (\n),
  then words ( ). This keeps semantically related text together.
"""


CHUNK_SIZE = 1000      # characters per chunk (not tokens)
CHUNK_OVERLAP = 200    # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 100   # discard chunks smaller than this


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a string into overlapping chunks using recursive splitting.

    Args:
        text:       The text to split
        chunk_size: Target size of each chunk in characters
        overlap:    How many characters to repeat between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Separators tried in order — prefer splitting on larger units first
    separators = ["\n\n", "\n", ". ", " ", ""]

    for separator in separators:
        if separator in text:
            chunks = _split_with_separator(text, separator, chunk_size, overlap)
            if chunks:
                return chunks

    # Fallback: hard split
    return _hard_split(text, chunk_size, overlap)


def _split_with_separator(
    text: str,
    separator: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text on a separator then merge small pieces into chunks."""
    parts = text.split(separator)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + (separator if current else "") + part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start new chunk with overlap from the end of current
            if overlap > 0 and current:
                overlap_text = current[-overlap:]
                current = overlap_text + (separator if overlap_text else "") + part
            else:
                current = part

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last resort: split at exact character positions."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Take a list of loaded document dicts and return a list of chunk dicts.

    Each input dict has at minimum: text, source, type
    Each output dict has: text, source, type, chunk_index,
                          plus any extra metadata from the input
    """
    chunks = []

    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue

        text_chunks = chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            if len(chunk) < MIN_CHUNK_SIZE:
                continue

            chunk_doc = {
                **doc,           # carry over all metadata (source, type, page, etc.)
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            }
            chunks.append(chunk_doc)

    return chunks
