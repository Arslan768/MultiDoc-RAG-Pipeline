"""
ui/app.py

Gradio web interface for the RAG pipeline.
Provides:
  - File upload (PDF, MD, TXT)
  - URL ingestion
  - Chat interface with source citations
  - Sidebar showing ingested documents

Run with: uv run python ui/app.py
Then open: http://localhost:7860
"""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
from rag.ingestor import ingest, get_stats
from rag.generator import generate
from rag.retriever import list_sources


def ingest_file(files) -> str:
    """Handle file upload and ingestion."""
    if not files:
        return "No files selected."

    results = []
    for file in files:
        result = ingest(file.name)
        if result["chunks_added"] > 0:
            results.append(f"✓ {Path(file.name).name} — {result['chunks_added']} chunks added")
        elif result["chunks_skipped"] > 0:
            results.append(f"↷ {Path(file.name).name} — already ingested ({result['chunks_skipped']} chunks)")
        else:
            results.append(f"✗ {Path(file.name).name} — failed to ingest")

    return "\n".join(results)


def ingest_url(url: str) -> str:
    """Handle URL ingestion."""
    if not url.strip():
        return "Please enter a URL."
    result = ingest(url.strip())
    if result["chunks_added"] > 0:
        return f"✓ Ingested {result['chunks_added']} chunks from {url}"
    elif result["chunks_skipped"] > 0:
        return f"↷ Already ingested ({result['chunks_skipped']} chunks)"
    else:
        return f"✗ Failed to ingest {url}"


def get_sources_text() -> str:
    """Return current sources as a formatted string."""
    sources = list_sources()
    if not sources:
        return "No documents ingested yet."
    return "\n".join(f"• {s}" for s in sources)


def chat(message: str, history: list) -> tuple[str, list]:
    """Process a chat message and return the answer with sources."""
    if not message.strip():
        return "", history

    result = generate(message, top_k=5)
    answer = result["answer"]

    # Append source citations to the answer
    if result["sources"]:
        sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in result["sources"])
        answer = answer + sources_text

    history.append((message, answer))
    return "", history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Pipeline", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Multi-doc RAG Pipeline")
        gr.Markdown("Ingest your documents, then ask questions about them.")

        with gr.Row():

            # ── LEFT COLUMN: Ingestion ────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Ingest Documents")

                with gr.Tab("Upload Files"):
                    file_input = gr.File(
                        label="Upload PDF, Markdown, or TXT files",
                        file_types=[".pdf", ".md", ".txt"],
                        file_count="multiple",
                    )
                    upload_btn = gr.Button("Ingest Files", variant="primary")
                    upload_status = gr.Textbox(label="Status", lines=4, interactive=False)

                    upload_btn.click(
                        fn=ingest_file,
                        inputs=[file_input],
                        outputs=[upload_status],
                    )

                with gr.Tab("Ingest URL"):
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="https://example.com/article",
                    )
                    url_btn = gr.Button("Ingest URL", variant="primary")
                    url_status = gr.Textbox(label="Status", lines=2, interactive=False)

                    url_btn.click(
                        fn=ingest_url,
                        inputs=[url_input],
                        outputs=[url_status],
                    )

                gr.Markdown("### Ingested Documents")
                sources_display = gr.Textbox(
                    label="",
                    lines=8,
                    interactive=False,
                    value=get_sources_text,
                    every=5,  # refresh every 5 seconds
                )

            # ── RIGHT COLUMN: Chat ────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions")
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    show_label=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                clear_btn = gr.Button("Clear Chat", variant="secondary")

                send_btn.click(
                    fn=chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
                msg_input.submit(
                    fn=chat,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                )
                clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
