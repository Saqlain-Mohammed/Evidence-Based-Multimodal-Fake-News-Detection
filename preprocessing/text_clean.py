"""Text preprocessing shared by training and inference (single source of truth)."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean(text: str, max_chars: int = 2000) -> str:
    text = (text or "")
    text = _URL.sub(" [URL] ", text)
    text = _WS.sub(" ", text).strip()
    return text[:max_chars]


def title_body(title: str, text: str) -> str:
    return clean(f"{title or ''} {text or ''}")
