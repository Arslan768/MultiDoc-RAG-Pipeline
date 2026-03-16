"""
generator.py

Takes retrieved chunks and generates a grounded answer using Gemini.

The key prompt engineering principle here is "grounding":
tell the model to answer ONLY from the provided context,
not from its general training knowledge. This prevents
hallucination and ensures answers are traceable to sources.
"""

import os
from google import genai
from dotenv import load_dotenv

from .retriever import retrieve

load_dotenv()

GENERATION_MODEL = "gemini-2.0-flash"
DEFAULT_TOP_K = 5


def _build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build the RAG prompt by stuffing retrieved chunks into context.

    Format:
        [Context]
        Source: filename.pdf (chunk 1)
        <text>

        Source: another.md
        <text>

        [Question]
        <query>
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source = chunk["source"]
        metadata = chunk.get("metadata", {})

        # Build a human-readable source label
        label = source
        if metadata.get("page"):
            label += f" (page {metadata['page']})"
        elif metadata.get("section"):
            label += f" — {metadata['section']}"

        context_parts.append(f"[Source {i}: {label}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question using ONLY the information in the context above
- If the context does not contain enough information to answer, say so clearly
- At the end of your answer, list the sources you used as "Sources: [Source 1], [Source 2]..."
- Be concise but complete

QUESTION: {query}

ANSWER:"""

    return prompt


def generate(query: str, top_k: int = DEFAULT_TOP_K, source_filter: str | None = None) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate an answer.

    Args:
        query:         The user's question
        top_k:         How many chunks to retrieve
        source_filter: Only search within a specific source

    Returns:
        Dict with:
            answer   — the generated answer string
            sources  — list of source strings used
            chunks   — the raw retrieved chunks
            query    — the original question
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY not set.",
            "sources": [],
            "chunks": [],
            "query": query,
        }

    # ── RETRIEVE ──────────────────────────────────────────────
    chunks = retrieve(query, top_k=top_k, source_filter=source_filter)

    if not chunks:
        return {
            "answer": "No relevant documents found. Please ingest some documents first using the `ingest` command.",
            "sources": [],
            "chunks": [],
            "query": query,
        }

    # ── GENERATE ──────────────────────────────────────────────
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(query, chunks)

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    answer = response.text or "No answer generated."

    # Collect unique sources
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "query": query,
    }
