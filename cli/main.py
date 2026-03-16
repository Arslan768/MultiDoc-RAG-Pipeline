"""
cli/main.py

CLI commands:
    rag ingest <path>        ingest a file or URL
    rag ingest-dir <folder>  ingest all files in a folder
    rag query <question>     ask a question
    rag sources              list all ingested sources
    rag stats                show database stats
    rag clear                delete all ingested data
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.ingestor import ingest, ingest_directory, get_stats
from rag.generator import generate
from rag.retriever import list_sources

app = typer.Typer(
    name="rag",
    help="Multi-document RAG pipeline — ingest docs and query them.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest_cmd(
    source: str = typer.Argument(..., help="File path or URL to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already present"),
):
    """Ingest a single file (PDF, Markdown, TXT) or URL into the vector database."""
    result = ingest(source, force=force)
    if result["chunks_added"] > 0:
        console.print(f"\n[bold green]Done.[/bold green] Added {result['chunks_added']} chunks.")


@app.command()
def ingest_dir(
    directory: str = typer.Argument(..., help="Directory to ingest"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest existing sources"),
):
    """Ingest all supported files in a directory."""
    ingest_directory(directory, force=force)


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Show retrieved chunks"),
):
    """Ask a question and get an answer grounded in your documents."""
    console.print()
    console.print(Panel(question, title="[bold cyan]Query[/bold cyan]", border_style="cyan"))
    console.print()

    with console.status("[cyan]Retrieving and generating...[/cyan]"):
        result = generate(question, top_k=top_k)

    # Print answer
    console.print(Markdown(result["answer"]))

    # Print sources
    if result["sources"]:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for source in result["sources"]:
            console.print(f"  [dim]•[/dim] {source}")

    # Optionally show retrieved chunks
    if show_chunks and result["chunks"]:
        console.print()
        console.print("[bold]Retrieved chunks:[/bold]")
        for i, chunk in enumerate(result["chunks"], 1):
            score = chunk["score"]
            source = chunk["source"]
            preview = chunk["text"][:200].replace("\n", " ")
            console.print(f"\n  [cyan][{i}][/cyan] Score: [green]{score}[/green] — {source}")
            console.print(f"  [dim]{preview}...[/dim]")


@app.command()
def sources():
    """List all documents currently ingested."""
    srcs = list_sources()
    if not srcs:
        console.print("[yellow]No documents ingested yet.[/yellow] Run `rag ingest <file>` first.")
        return

    table = Table(title="Ingested Sources", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Source")

    for i, src in enumerate(srcs, 1):
        table.add_row(str(i), src)

    console.print(table)


@app.command()
def stats():
    """Show database statistics."""
    s = get_stats()
    console.print(f"\n[bold]ChromaDB Stats[/bold]")
    console.print(f"  Total chunks : [green]{s['total_chunks']}[/green]")
    console.print(f"  Sources      : [green]{s.get('source_count', 0)}[/green]")
    if s.get("sources"):
        for src in s["sources"]:
            console.print(f"    [dim]•[/dim] {src}")
    console.print()


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete all ingested data from ChromaDB."""
    if not confirm:
        confirmed = typer.confirm("This will delete all ingested data. Continue?")
        if not confirmed:
            console.print("Cancelled.")
            raise typer.Exit()

    import chromadb, os
    from dotenv import load_dotenv
    load_dotenv()
    path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=path)
    client.delete_collection("documents")
    console.print("[green]Cleared all ingested data.[/green]")


if __name__ == "__main__":
    app()
