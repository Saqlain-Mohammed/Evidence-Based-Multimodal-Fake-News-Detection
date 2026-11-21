"""Text analysis: BERT-based contextual analysis with graceful fallback."""
import os
import re
import hashlib
import numpy as np

_tokenizer = None
_bert_model = None
_model_name = "bert-base-uncased"

SENSATIONAL = [
    "shocking", "unbelievable", "miracle", "secret", "exposed", "breaking",
    "viral", "you won't believe", "mind-blowing", "emergency", "alert",
    "hoax", "conspiracy", "cover-up", "coverup", "urgent", "must watch",
    "100%", "guaranteed", "cure", "instant", "$$$", "free money",
]
CLICKBAIT_PATTERNS = [
    r"you won'?t believe", r"\d+ (things|reasons|secrets|tricks)",
    r"(shocking|unbelievable|amazing|incredible)\b",
    r"[!]{2,}", r"\bBREAKING\b",
]


def _try_load_bert():
    global _tokenizer, _bert_model
    if _tokenizer is not None and _bert_model is not None:
        return True
    # Local cache only: never block an HTTP request on a ~440MB model download.
    # To enable full BERT, pre-download once:
    #   python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased')"
    import os
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _bert_model = AutoModel.from_pretrained(_model_name)
        _bert_model.eval()
        return True
    except Exception:
        _tokenizer = None
        _bert_model = None
        return False
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def bert_features(text: str) -> np.ndarray:
    """768-d contextual embedding. Real BERT when available, else hashed fallback."""
    if _try_load_bert():
        try:
            import torch
            toks = _tokenizer(text[:2000], return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                out = _bert_model(**toks)
            vec = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
            return vec.astype(float)
        except Exception:
            pass
    # Deterministic hashed embedding fallback (documented in API response)
    vec = np.zeros(768, dtype=float)
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % 768] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


_baseline = None
_baseline_tried = False


def _try_load_baseline():
    """New trained baseline (TF-IDF LogReg on leakage-free data/realistic).

    Preferred over the legacy demo transformer when present; cached.
    """
    global _baseline, _baseline_tried
    if _baseline_tried:
        return _baseline
    _baseline_tried = True
    try:
        import os
        import pickle
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p = os.path.join(root, "models", "baselines", "tfidf_logreg.pkl")
        if os.path.exists(p):
            _baseline = pickle.load(open(p, "rb"))
    except Exception:
        _baseline = None
    return _baseline


def analyze_text(headline: str, content: str) -> dict:
    text = f"{headline or ''} {content or ''}".strip()
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    n_chars = len(text)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(re.findall(r"[A-Za-z]", text)))
    excl = text.count("!") + text.count("?")
    sens_hits = [w for w in SENSATIONAL if w.lower() in text.lower()]
    clickbait_hits = []
    for p in CLICKBAIT_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            clickbait_hits.append(p)

    # BERT embedding (contextual, not keyword matching — keywords below are only
    # interpretable signals layered on top for the explanation panel)
    emb = bert_features(text if text else "empty news input")
    emb_norm = float(np.linalg.norm(emb))

    # Heuristic fake-signal score in [0,1] used as one fusion input.
    # Documented methodology: sensational language + clickbait + caps + too-short content.
    score = 0.0
    score += min(0.35, 0.07 * len(sens_hits))
    score += min(0.25, 0.12 * len(clickbait_hits))
    score += min(0.15, max(0.0, caps_ratio - 0.08) * 1.2)
    score += min(0.10, excl * 0.02)
    if 0 < n_words < 40:
        score += 0.15  # very thin articles are a weak signal
    if n_words == 0:
        score = 0.5  # no text -> neutral
    fake_signal = float(min(1.0, max(0.0, score)))
    bert_used = _tokenizer is not None and _bert_model is not None

    # Fine-tuned head: prefer the retrained leakage-free baseline (models/);
    # fall back to the legacy demo transformer artifact only when absent.
    tuned_p, tuned_used, tuned_engine = None, False, ""
    try:
        clf = _try_load_baseline()
        if clf is not None:
            import sys as _sys
            sys_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))), "preprocessing")
            if sys_path not in _sys.path:
                _sys.path.insert(0, sys_path)
            from text_clean import title_body as _tb
            p_real = float(clf.predict_proba([_tb(headline, content) or "empty"])[0][1])
            tuned_p = 1.0 - p_real
            tuned_used = True
            tuned_engine = "tfidf-logreg (retrained, leakage-free v2)"
            fake_signal = float(min(1.0, max(0.0, 0.7 * tuned_p + 0.3 * fake_signal)))
    except Exception:
        pass
    if not tuned_used:
        try:
            from modules.trained_loader import text_fake_prob
            tuned_p = text_fake_prob(text if text else "empty news input")
            if tuned_p is not None:
                tuned_used = True
                tuned_engine = "legacy demo transformer (v1, trivial-data — superseded)"
                fake_signal = float(min(1.0, max(0.0, 0.65 * tuned_p + 0.35 * fake_signal)))
        except Exception:
            pass

    return {
        "word_count": n_words,
        "char_count": n_chars,
        "caps_ratio": round(float(caps_ratio), 3),
        "sensational_hits": sens_hits[:10],
        "clickbait_patterns": len(clickbait_hits),
        "fake_signal": round(fake_signal, 3),
        "text_confidence": round(float(0.55 + 0.4 * min(1.0, n_words / 300)), 3),
        "embedding_norm": round(emb_norm, 3),
        "embedding_dim": int(emb.shape[0]),
        "model": "bert-base-uncased (HuggingFace Transformers)" if bert_used else "hashed-embedding fallback (BERT unavailable offline)",
        "finetuned_model_used": tuned_used,
        "finetuned_p_fake": round(float(tuned_p), 3) if tuned_p is not None else None,
        "tuned_engine": tuned_engine,
        "features": emb[:32].tolist(),  # truncated vector for fusion/debug display
    }
