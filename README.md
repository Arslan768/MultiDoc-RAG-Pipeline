# Multi-doc RAG Pipeline

A retrieval-augmented generation (RAG) system that lets you ingest your own documents — PDFs, Markdown, plain text, and web URLs — and ask questions about them. Answers are grounded in your documents with source citations.

Built as Project 02 of the [Agentic AI Engineer roadmap](https://github.com/yourusername).

## Demo

![Demo](DEMO.gif)

## Features

- Ingest PDFs, Markdown files, plain text, and web URLs
- Semantic search with Gemini embeddings (`text-embedding-004`)
- Grounded answers with source citations via `gemini-2.0-flash`
- ChromaDB vector store — runs locally, persists to disk
- CLI interface + Gradio web UI

## Architecture

```
Document → Loader → Chunker → Embedder → ChromaDB
                                               ↓
Question → Embedder → ChromaDB search → Generator → Answer + Sources
```

```
src/rag/
├── loaders/        # PDF, Markdown, TXT, URL → raw text
├── chunker.py      # recursive text splitting with overlap
├── embedder.py     # Gemini text-embedding-004 wrapper
├── ingestor.py     # orchestrates load → chunk → embed → store
├── retriever.py    # semantic search over ChromaDB
└── generator.py    # builds prompt + calls Gemini
```

## Install

```bash
git clone https://github.com/yourusername/rag-pipeline
cd rag-pipeline

uv sync
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

## Usage

### CLI

```bash
# Ingest a single file
uv run python cli/main.py ingest-cmd ./docs/paper.pdf

# Ingest a URL
uv run python cli/main.py ingest-cmd https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

# Ingest an entire folder
uv run python cli/main.py ingest-dir ./docs

# Ask a question
uv run python cli/main.py query "What is the main argument of the paper?"

# Ask with source chunks visible
uv run python cli/main.py query "Explain attention mechanism" --show-chunks

# List all ingested sources
uv run python cli/main.py sources

# Show database stats
uv run python cli/main.py stats

# Clear all data
uv run python cli/main.py clear --yes
```

### Web UI

```bash
uv run python ui/app.py
# Open http://localhost:7860
```

Upload files or paste a URL in the left panel, then ask questions in the chat.

## Run tests

```bash
uv run pytest tests/ -v
```

Tests use mocks — no API calls, no credits consumed.

## License

MIT
