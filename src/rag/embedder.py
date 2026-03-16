"""
embedder.py

Wraps Gemini's text-embedding-004 model.

Why text-embedding-004:
- Free tier with generous limits
- 768-dimensional vectors
- Supports task_type which improves retrieval quality:
    RETRIEVAL_DOCUMENT  — when embedding chunks for storage
    RETRIEVAL_QUERY     — when embedding a user question

Always use the right task_type — it meaningfully improves results.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in .env file.")
    return genai.Client(api_key=api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document chunks for storage in ChromaDB.

    Uses task_type=RETRIEVAL_DOCUMENT which tells the model
    these are passages that will be retrieved later.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of 768 floats)
    """
    client = get_client()
    embeddings = []

    # Gemini embedding API processes one text at a time
    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        embeddings.append(result.embeddings[0].values)

    return embeddings


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string for retrieval.

    Uses task_type=RETRIEVAL_QUERY — different from document
    embedding, optimized for the asymmetric search use case.

    Args:
        text: The user's question

    Returns:
        Single embedding vector (list of 768 floats)
    """
    client = get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    return result.embeddings[0].values
