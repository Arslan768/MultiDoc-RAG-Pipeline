"""
loaders/markdown.py

Loads Markdown files and splits them on headings.
Each heading section becomes its own chunk with metadata.
"""

from pathlib import Path
import re


def load_markdown(path: str | Path) -> list[dict]:
    """
    Load a Markdown file and return a list of section dicts.

    Splits on # headings so each section stays semantically together.
    Each dict has:
        text     — section text including the heading
        source   — filename
        section  — heading title (or "intro" for pre-heading content)
        type     — "markdown"
    """
    path = Path(path)

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[markdown loader] Error reading {path.name}: {e}")
        return []

    # Split on any heading level (# ## ###)
    heading_pattern = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
    headings = list(heading_pattern.finditer(content))

    sections = []

    if not headings:
        # No headings — return the whole file as one chunk
        if content.strip():
            sections.append({
                "text": content.strip(),
                "source": path.name,
                "section": "full",
                "type": "markdown",
            })
        return sections

    # Content before the first heading
    intro = content[:headings[0].start()].strip()
    if intro:
        sections.append({
            "text": intro,
            "source": path.name,
            "section": "intro",
            "type": "markdown",
        })

    # Each heading + its content until the next heading
    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        section_text = content[start:end].strip()
        heading_title = match.group().lstrip("#").strip()

        if section_text:
            sections.append({
                "text": section_text,
                "source": path.name,
                "section": heading_title,
                "type": "markdown",
            })

    return sections
