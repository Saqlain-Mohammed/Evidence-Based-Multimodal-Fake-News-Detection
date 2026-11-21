"""URL article extraction (newspaper3k + BeautifulSoup fallback)."""
import requests
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (research fact-check prototype)"}


def extract_url(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"headline": "", "content": "", "source_url": "", "method": "none"}
    if not re.match(r"^https?://", url, re.I):
        url = "http://" + url
    # Try newspaper3k
    try:
        from newspaper import Article
        a = Article(url)
        a.download(); a.parse()
        if (a.text or "").strip():
            return {"headline": (a.title or "")[:300], "content": (a.text or "")[:8000],
                    "source_url": url, "method": "newspaper3k",
                    "top_image": a.top_image or ""}
    except Exception:
        pass
    # BeautifulSoup fallback
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")[:300]
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n".join(p for p in paras if len(p.split()) > 5)[:8000]
        og = soup.find("meta", property="og:image")
        img = (og.get("content") if og else "") or ""
        return {"headline": title, "content": text, "source_url": url,
                "method": "beautifulsoup", "top_image": img}
    except Exception as e:
        return {"headline": "", "content": "", "source_url": url,
                "method": f"failed: {e}"}
