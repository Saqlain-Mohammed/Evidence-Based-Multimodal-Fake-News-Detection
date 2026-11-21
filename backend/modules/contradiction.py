"""Contradiction detection: NLI (BART-MNLI) with lexical fallback.
Labels each claim-evidence pair SUPPORTS / CONTRADICTS / NOT ENOUGH INFO."""
import re

_nli = None


def _try_load_nli():
    global _nli
    if _nli is not None:
        return True
    # Local cache only: never block an HTTP request on a ~1.6GB model download.
    # To enable the full NLI model, pre-download it once:
    #   python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
    import os
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import pipeline
        _nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return True
    except Exception:
        _nli = None
        return False
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


NEGATIONS = {"not", "no", "never", "false", "fake", "denied", "denies", "debunked",
             "refuted", "contradicts", "contrary", "disproved", "hoax", "rumor", "rumour"}
NUM_RE = re.compile(r"\d+(?:,\d+)*(?:\.\d+)?%?")


def _lexical(claim: str, ev: str) -> tuple[str, float]:
    cw = set(re.findall(r"[a-z]+", claim.lower()))
    ew = set(re.findall(r"[a-z]+", ev.lower()))
    overlap = len(cw & ew) / max(1, len(cw | ew))
    ev_neg = any(w in ew for w in NEGATIONS)
    cn, en = set(NUM_RE.findall(claim)), set(NUM_RE.findall(ev))
    num_mismatch = bool(cn and en and cn.isdisjoint(en))
    if (ev_neg and overlap > 0.12) or num_mismatch:
        return "CONTRADICTS", round(min(0.9, 0.55 + overlap), 2)
    if overlap > 0.18:
        return "SUPPORTS", round(min(0.9, 0.5 + overlap), 2)
    return "NOT ENOUGH INFO", round(0.4 + overlap, 2)


def check_claim(headline: str, content: str, evidence_items: list[dict]) -> dict:
    claim = (headline or "").strip() or (content or "")[:500].strip()
    if not claim:
        return {"claim": "", "status": "insufficient", "overall": "NOT ENOUGH INFO",
                "confidence": 0.0, "fake_signal": 0.5,
                "pairs": [], "model": "NLI (no claim text)"}
    if not evidence_items:
        return {"claim": claim, "status": "insufficient", "overall": "NOT ENOUGH INFO",
                "confidence": 0.0, "fake_signal": 0.5,
                "pairs": [], "model": "NLI skipped — no evidence retrieved"}
    use_nli = _try_load_nli()
    pairs = []
    for ev in evidence_items[:5]:
        ev_text = f"{ev.get('title','')} {ev.get('snippet','')}".strip()[:800]
        if use_nli:
            try:
                r = _nli(ev_text, candidate_labels=["supports", "contradicts", "neutral"],
                         hypothesis_template="This text {} the claim: " + claim[:300])
                top, score = r["labels"][0], float(r["scores"][0])
                label = {"supports": "SUPPORTS", "contradicts": "CONTRADICTS"}.get(top, "NOT ENOUGH INFO")
                pairs.append({"evidence_title": ev.get("title", "")[:160],
                              "evidence_source": ev.get("source", ""),
                              "label": label, "confidence": round(score, 3),
                              "similarity": ev.get("similarity")})
                continue
            except Exception:
                pass
        label, conf = _lexical(claim, ev_text)
        pairs.append({"evidence_title": ev.get("title", "")[:160],
                      "evidence_source": ev.get("source", ""),
                      "label": label, "confidence": conf,
                      "similarity": ev.get("similarity")})
    # Aggregate: contradiction dominates (evidence-based design)
    n_c = sum(1 for p in pairs if p["label"] == "CONTRADICTS")
    n_s = sum(1 for p in pairs if p["label"] == "SUPPORTS")
    if n_c > 0 and n_c >= n_s:
        overall, fake_signal = "CONTRADICTS", 0.85
    elif n_s > n_c:
        overall, fake_signal = "SUPPORTS", 0.2
    else:
        overall, fake_signal = "NOT ENOUGH INFO", 0.5
    conf = round(float(sum(p["confidence"] for p in pairs) / len(pairs)), 3) if pairs else 0.0
    return {"claim": claim[:500], "status": "ok", "overall": overall,
            "confidence": conf, "fake_signal": fake_signal, "pairs": pairs,
            "model": "facebook/bart-large-mnli (NLI)" if use_nli and _nli is not None
                     else "lexical NLI fallback (overlap + negation + numeric mismatch)"}
