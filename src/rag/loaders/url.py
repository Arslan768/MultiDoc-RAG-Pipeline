"""
loaders/url.py

Fetches a URL and extracts clean text.
Reuses the same approach as Project 01's url_reader tool.
"""

import httpx
from bs4 import BeautifulSoup

SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}
MAX_CHARS = 20000  # higher than Project 01 since we chunk afterwards


def load_url(url: str) -> list[dict]:
    """
    Fetch a URL and return a single-item list with extracted text.

    Each dict has:
        text     — clean page text
        source   — the URL
        title    — page <title> if found
        type     — "url"
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Remove unwanted tags
        for tag in soup(SKIP_TAGS):
            tag.decompose()

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(id="main")
            or soup.find(class_="content")
            or soup.body
        )

        if not main:
            return []

        text = main.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        clean = "\n".join(lines)[:MAX_CHARS]

        if not clean:
            return []

        return [{
            "text": clean,
            "source": url,
            "title": title,
            "type": "url",
        }]

    except Exception as e:
        print(f"[url loader] Error fetching {url}: {e}")
        return []
