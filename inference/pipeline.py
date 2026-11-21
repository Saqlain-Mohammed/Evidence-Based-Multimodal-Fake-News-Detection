"""Unified inference pipeline (single entrypoint for backend + scripts).

Runs: fake-news text -> AI-text -> image manipulation -> AI image ->
evidence/consistency -> calibrated combined result with explicit
INSUFFICIENT EVIDENCE / NEEDS HUMAN VERIFICATION states.

All chart values come from these live outputs (no static demo data).
"""
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "backend"))
sys.path.insert(0, os.path.join(BASE, "preprocessing"))
from text_clean import title_body  # noqa: E402


def _load_baseline():
    import pickle
    p = os.path.join(BASE, "models", "baselines", "tfidf_logreg.pkl")
    if os.path.exists(p):
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            return None
    return None


_BASE = None


def fake_news_score(title, text):
    """Returns (p_fake, source) with learned baseline when available."""
    global _BASE
    if _BASE is None:
        _BASE = _load_baseline()
    t = title_body(title, text)
    if not t.strip():
        return 0.5, "no-text-neutral"
    if _BASE is not None:
        try:
            p_real = float(_BASE.predict_proba([t])[0][1])
            return round(1.0 - p_real, 3), "tfidf-logreg (trained, balanced)"
        except Exception:
            pass
    from modules.text_analysis import analyze_text
    r = analyze_text(title, text)
    return r["fake_signal"], "heuristic-fallback"


def ai_text_score(title, text):
    import pickle
    t = title_body(title, text)
    if not t.strip():
        return None
    p = os.path.join(BASE, "models", "ai_text.pkl")
    if os.path.exists(p):
        try:
            clf = pickle.load(open(p, "rb"))["pipeline"]
            return round(float(clf.predict_proba([t])[0][1]), 3)
        except Exception:
            pass
    return None


def full_analysis(title="", text="", image_bytes=None, url=""):
    from modules.image_analysis import analyze_image
    from modules.source_credibility import score_source
    from modules.evidence_retrieval import retrieve_evidence
    from modules.contradiction import check_claim
    from modules.ai_content import analyze_ai_image, combine_ai

    p_fake_text, text_src = fake_news_score(title, text)
    p_ai_learned = ai_text_score(title, text)
    img_r = analyze_image(image_bytes)
    src_r = score_source(url, text or "")
    ev_r = retrieve_evidence(title, text)
    contra_r = check_claim(title, text, ev_r.get("items", []))
    im_ai = analyze_ai_image(image_bytes)

    has_text = bool((title_body(title, text)).strip())
    has_image = bool(img_r.get("present"))
    ev_ok = ev_r.get("status") != "insufficient"

    # evidence-conditioned confidence: cap when nothing supports the verdict
    evidence_level = float(ev_r.get("evidence_match", 0.0)) if ev_ok else 0.0
    base_conf = 0.5 + 0.3 * (1.0 - abs(p_fake_text - (0.5 if not has_image else img_r.get("fake_signal", 0.5)))) \
        + 0.2 * evidence_level
    confidence = round(max(0.05, min(0.95, base_conf if (has_text or has_image) else 0.05)), 3)

    if not has_text and not has_image:
        overall, detail = "INSUFFICIENT EVIDENCE", "No text or image supplied — nothing to analyze."
    elif not has_text and has_image and not ev_ok:
        overall, detail = "NEEDS HUMAN VERIFICATION", "Image-only input with no retrievable evidence."
    elif len((text or "").split()) < 12 and not has_image and not ev_ok:
        overall, detail = "NEEDS HUMAN VERIFICATION", "Text too short and no supporting evidence retrieved."
    elif p_fake_text >= 0.65 and not ev_ok:
        overall, detail = "NEEDS HUMAN VERIFICATION", \
            "Text looks suspicious but external evidence is insufficient to confirm."
    else:
        overall = "LIKELY FAKE" if p_fake_text >= 0.65 else (
            "LIKELY REAL" if p_fake_text <= 0.45 else "UNCERTAIN — MIXED SIGNALS")
        detail = "Combined text, visual and evidence signals."

    p_fake_img = img_r.get("fake_signal", 0.5) if has_image else None
    chart = {
        "fake_vs_real": {"fake": round(p_fake_text * 100, 1),
                         "real": round((1 - p_fake_text) * 100, 1)},
        "ai_vs_human": ({"ai": round(p_ai_learned * 100, 1),
                          "human": round((1 - p_ai_learned) * 100, 1)}
                         if p_ai_learned is not None else None),
        "image_manipulation": ({"suspicious": round(p_fake_img * 100, 1),
                                "clean": round((1 - p_fake_img) * 100, 1)}
                               if p_fake_img is not None else None),
        "evidence_support": round(evidence_level * 100, 1),
        "model_confidence": round(confidence * 100, 1),
    }
    return {
        "fake_news": {"verdict": overall if "LIKELY" in overall or "UNCERTAIN" in overall else
                      ("LIKELY FAKE" if p_fake_text >= 0.65 else ("LIKELY REAL" if p_fake_text <= 0.35 else "UNCERTAIN")),
                      "p_fake": p_fake_text, "engine": text_src},
        "ai_text": {"p_ai": p_ai_learned,
                    "verdict": ("LIKELY AI" if (p_ai_learned or 0) >= 0.65 else
                                ("LIKELY HUMAN" if (p_ai_learned or 0) <= 0.45 else "UNCERTAIN"))
                    if p_ai_learned is not None else "NO TEXT"},
        "image": {"present": has_image,
                  "p_suspicious": p_fake_img,
                  "p_ai": im_ai.get("p_ai"),
                  "notes": (img_r.get("risk_reasons", []) if has_image else ["No image supplied."])},
        "evidence": {"status": ev_r.get("status"), "match": evidence_level,
                     "contradiction": contra_r.get("overall"),
                     "items": len(ev_r.get("items", []))},
        "overall_status": overall, "overall_detail": detail,
        "confidence": confidence,
        "recommendation": ("Verify with reliable sources before treating the claim as factual."
                           if overall != "LIKELY REAL" else
                           "No red flags found, but routine source-checks still apply."),
        "chart": chart,
    }
