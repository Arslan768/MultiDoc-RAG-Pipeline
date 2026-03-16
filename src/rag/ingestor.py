"""
ingestor.py

Orchestrates the full ingestion pipeline:
  load → chunk → embed → store in ChromaDB

This is the "write" side of the RAG system.
You run this once per document (or batch of documents).
ChromaDB persists to disk so you don't re-embed on every query.
"""

import os
import uuid
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .loaders import load
from .chunker import chunk_documents
from .embedder import embed_documents

load_dotenv()

console = Console()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "documents"


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # We provide our own embeddings so tell ChromaDB not to embed
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _source_hash(source: str) -> str:
    """Create a short hash of the source path/URL for deduplication."""
    return hashlib.md5(source.encode()).hexdigest()[:8]


def ingest(source: str, force: bool = False) -> dict:
    """
    Ingest a single source (file path or URL) into ChromaDB.

    Pipeline:
        1. Load   — extract raw text + metadata
        2. Chunk  — split into overlapping chunks
        3. Embed  — get Gemini embeddings for each chunk
        4. Store  — upsert into ChromaDB with metadata

    Args:
        source: File path or URL to ingest
        force:  If True, re-ingest even if already present

    Returns:
        Dict with stats: chunks_added, chunks_skipped, source
    """
    collection = get_collection()

    # Check if already ingested
    source_id = _source_hash(source)
    existing = collection.get(where={"source_hash": source_id})

    if existing["ids"] and not force:
        console.print(f"[yellow]Skipping[/yellow] {source} — already ingested ({len(existing['ids'])} chunks). Use --force to re-ingest.")
        return {"chunks_added": 0, "chunks_skipped": len(existing["ids"]), "source": source}

    # If force, delete existing chunks for this source first
    if existing["ids"] and force:
        collection.delete(where={"source_hash": source_id})
        console.print(f"[yellow]Re-ingesting[/yellow] {source}...")

    console.print(f"[cyan]Loading[/cyan] {source}...")

    # ── 1. LOAD ───────────────────────────────────────────────
    documents = load(source)
    if not documents:
        console.print(f"[red]No content extracted from[/red] {source}")
        return {"chunks_added": 0, "chunks_skipped": 0, "source": source}

    console.print(f"  Loaded {len(documents)} section(s)")

    # ── 2. CHUNK ──────────────────────────────────────────────
    chunks = chunk_documents(documents)
    if not chunks:
        console.print(f"[red]No chunks produced from[/red] {source}")
        return {"chunks_added": 0, "chunks_skipped": 0, "source": source}

    console.print(f"  Split into {len(chunks)} chunks")

    # ── 3. EMBED ──────────────────────────────────────────────
    texts = [c["text"] for c in chunks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Embedding {task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"{len(texts)} chunks...", total=len(texts))
        embeddings = []
        # Embed in small batches to show progress
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embed_documents(batch)
            embeddings.extend(batch_embeddings)
            progress.advance(task, len(batch))

    # ── 4. STORE ──────────────────────────────────────────────
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_id}_{i}_{uuid.uuid4().hex[:6]}"
        ids.append(chunk_id)

        # ChromaDB metadata must be str/int/float/bool only — no nested dicts
        metadata = {
            "source": chunk.get("source", source),
            "source_hash": source_id,
            "type": chunk.get("type", "unknown"),
            "chunk_index": chunk.get("chunk_index", i),
        }
        # Add optional fields if present
        if "page" in chunk:
            metadata["page"] = chunk["page"]
        if "section" in chunk:
            metadata["section"] = chunk["section"]
        if "title" in chunk:
            metadata["title"] = chunk["title"]

        metadatas.append(metadata)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    console.print(f"[green]Stored {len(chunks)} chunks[/green] from {source}")
    return {"chunks_added": len(chunks), "chunks_skipped": 0, "source": source}


def ingest_directory(directory: str, extensions: list[str] | None = None, force: bool = False) -> list[dict]:
    """
    Ingest all supported files in a directory.

    Args:
        directory:  Path to the folder
        extensions: List of extensions to include e.g. ['.pdf', '.md']
                    If None, ingests all supported types
        force:      Re-ingest even if already present

    Returns:
        List of result dicts, one per file
    """
    supported = {".pdf", ".md", ".markdown", ".txt"}
    if extensions:
        supported = {e if e.startswith(".") else f".{e}" for e in extensions}

    directory = Path(directory)
    if not directory.exists():
        console.print(f"[red]Directory not found:[/red] {directory}")
        return []

    files = [f for f in directory.rglob("*") if f.suffix.lower() in supported]

    if not files:
        console.print(f"[yellow]No supported files found in[/yellow] {directory}")
        return []

    console.print(f"\nFound {len(files)} file(s) in {directory}\n")

    results = []
    for file in files:
        result = ingest(str(file), force=force)
        results.append(result)

    total_added = sum(r["chunks_added"] for r in results)
    console.print(f"\n[bold green]Done.[/bold green] {total_added} chunks added across {len(files)} file(s).")
    return results


def get_stats() -> dict:
    """Return stats about what's currently stored in ChromaDB."""
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"total_chunks": 0, "sources": []}

    # Get all unique sources
    results = collection.get(include=["metadatas"])
    sources = list({m["source"] for m in results["metadatas"]})

    return {
        "total_chunks": count,
        "sources": sources,
        "source_count": len(sources),
    }
