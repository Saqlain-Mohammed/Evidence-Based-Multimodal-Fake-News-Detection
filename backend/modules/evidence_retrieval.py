"""Evidence retrieval: GDELT DOC API (free, no key) + Google News RSS fallback.
Never fabricates: returns [] with status 'insufficient' when offline/empty.
Ranks by semantic similarity (sentence-transformers when available, else TF-IDF)."""
import re
import html
import requests
import feedparser
from datetime import datetime

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "FakeNewsDetector/1.0 (research prototype)"}
TRUSTED_HINTS = ("reuters", "apnews", "bbc", "guardian", "npr", "pbs",
                 "aljazeera", "dw.com", "france24", "ndtv", "thehindu",
                 "indianexpress", "hindustantimes", "cnn", "nbcnews")

_st_model = None


def _query_text(headline: str, content: str) -> str:
    base = (headline or "").strip() or (content or "")[:300]
    # keep first ~12 meaningful words
    words = re.findall(r"[A-Za-z]{3,}", base)
    stop = {"the", "and", "for", "with", "that", "this", "from", "have", "has",
            "are", "was", "were", "will", "would", "about", "after", "news"}
    kept = [w for w in words if w.lower() not in stop][:12]
    return " ".join(kept) if kept else "news"


def _gdelt(query: str, n: int = 5):
    try:
        r = requests.get(GDELT_URL, params={
            "query": query, "mode": "artlist", "maxrecords": n,
            "format": "json", "sort": "datedesc",
        }, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        arts = (r.json() or {}).get("articles", [])
        out = []
        for a in arts:
            out.append({
                "source": a.get("sourceCommonName") or a.get("domain") or "GDELT indexed outlet",
                "title": a.get("title") or "",
                "url": a.get("url") or "",
                "date": (a.get("seendate") or "")[:8],
                "snippet": a.get("title") or "",
                "origin": "GDELT DOC 2.1 (global news index)",
            })
        return out
    except Exception:
        return []


def _google_news_rss(query: str, n: int = 5):
    try:
        import urllib.parse
        import socket
        q = urllib.parse.quote(query)
        old = socket.getdefaulttimeout()
        socket.setdefaulttimeout(15)
        try:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en",
                                    request_headers=HEADERS)
        finally:
            socket.setdefaulttimeout(old)
        out = []
        for e in (feed.entries or [])[:n]:
            out.append({
                "source": getattr(e, "source", {}).get("title", "Google News outlet") if hasattr(e, "source") else "Google News outlet",
                "title": html.unescape(getattr(e, "title", "")),
                "url": getattr(e, "link", ""),
                "date": getattr(e, "published", ""),
                "snippet": html.unescape(re.sub(r"<[^>]+>", "", getattr(e, "summary", ""))[:400]),
                "origin": "Google News RSS",
            })
        return out
    except Exception:
        return []


def _fetch_body(url: str, timeout: int = 10) -> str:
    """Fetch first ~600 chars of an article body. '' on any failure (never fabricated)."""
    if not url:
        return ""
    try:
        from newspaper import Article
        a = Article(url, request_timeout=10)
        a.download()
        a.parse()
        if (a.text or "").strip():
            return a.text.strip()[:600]
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paras if len(p.split()) > 5)[:600]
        return text
    except Exception:
        return ""


def _similarity(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    try:
        global _st_model
        if _st_model is None:
            # Local cache only: never block a request on a model download.
            # Pre-download once to enable: SentenceTransformer("all-MiniLM-L6-v2")
            import os
            prev = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev
        qe = _st_model.encode([query])[0]
        de = _st_model.encode(docs)
        import numpy as np
        qn = qe / (np.linalg.norm(qe) + 1e-9)
        sims = []
        for v in de:
            vn = v / (np.linalg.norm(v) + 1e-9)
            sims.append(float(np.dot(qn, vn)))
        return sims
    except Exception:
        pass
    # TF-IDF cosine fallback
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer().fit_transform([query] + docs)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten().tolist()
        return [float(s) for s in sims]
    except Exception:
        return [0.0] * len(docs)


def retrieve_evidence(headline: str, content: str, n: int = 5) -> dict:
    query = _query_text(headline, content)
    if not (headline or content or "").strip():
        return {"query": "", "status": "insufficient",
                "message": "No claim text supplied — evidence retrieval skipped.",
                "items": [], "similarity_model": "none", "evidence_match": 0.0}
    items = _gdelt(query, n)
    if len(items) < 3:
        items += _google_news_rss(query, n)
    # dedupe by URL
    seen, uniq = set(), []
    for it in items:
        if it.get("url") in seen:
            continue
        seen.add(it.get("url"))
        uniq.append(it)
    items = uniq[:n]
    if not items:
        return {"query": query, "status": "insufficient",
                "message": "No external evidence could be retrieved (offline or no matches). Shown as Insufficient evidence — nothing fabricated.",
                "items": [], "similarity_model": "none", "evidence_match": 0.0}
    docs = [f"{i.get('title','')} {i.get('snippet','')}" for i in items]
    sims = _similarity(f"{headline} {content}"[:1000], docs)
    for i, s in zip(items, sims):
        i["similarity"] = round(float(s), 3)
        low = (i.get("source") or "").lower()
        i["trusted"] = any(h in low for h in TRUSTED_HINTS)
    items.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    avg = round(float(sum(sims) / len(sims)), 3) if sims else 0.0
    model = "sentence-transformers all-MiniLM-L6-v2" if _st_model is not None else "TF-IDF cosine fallback"

    # Enrich top-3 with fetched article bodies + per-source stance (concurrent, timeout-guarded).
    claim_text = f"{headline} {content}"[:1000]
    try:
        from modules.contradiction import _lexical as _stance_fn
    except Exception:
        _stance_fn = None
    import concurrent.futures

    def enrich(it):
        body = _fetch_body(it.get("url", ""))
        if body:
            it["body_snippet"] = body
        compare = (it.get("title", "") + " " + (body or it.get("snippet", "")))[:800]
        if _stance_fn is not None and compare.strip():
            try:
                label, conf = _stance_fn(claim_text, compare)
                it["stance"] = label
                it["stance_confidence"] = conf
            except Exception:
                pass
        return it

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        top = list(ex.map(enrich, items[:3], timeout=45))
    items = top + items[3:]
    return {"query": query, "status": "ok", "message": f"{len(items)} items retrieved.",
            "items": items, "similarity_model": model, "evidence_match": avg}
