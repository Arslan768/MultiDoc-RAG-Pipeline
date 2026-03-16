"""
retriever.py

The "read" side of RAG — given a user question,
find the most relevant chunks stored in ChromaDB.

Flow:
  1. Embed the query with task_type=RETRIEVAL_QUERY
  2. Search ChromaDB using cosine similarity
  3. Return top-k results with their text and metadata
"""

import os
import chromadb
from dotenv import load_dotenv

from .embedder import embed_query

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "documents"
DEFAULT_TOP_K = 5


def retrieve(query: str, top_k: int = DEFAULT_TOP_K, source_filter: str | None = None) -> list[dict]:
    """
    Find the most relevant chunks for a query.

    Args:
        query:         The user's question
        top_k:         How many chunks to return
        source_filter: Optional — only search within a specific source file/URL

    Returns:
        List of result dicts, each with:
            text       — the chunk text
            source     — where it came from
            score      — similarity score (0-1, higher = more similar)
            metadata   — all stored metadata
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = embed_query(query)

    # Build optional where filter
    where = None
    if source_filter:
        where = {"source": source_filter}

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns distances (lower = more similar for cosine)
    # Convert to similarity scores (higher = more similar)
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        # Cosine distance → similarity: similarity = 1 - distance
        similarity = round(1 - distance, 4)

        chunks.append({
            "text": doc,
            "score": similarity,
            "metadata": results["metadatas"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
        })

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


def list_sources() -> list[str]:
    """Return all unique sources currently stored in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    return sorted(list({m["source"] for m in results["metadatas"]}))
