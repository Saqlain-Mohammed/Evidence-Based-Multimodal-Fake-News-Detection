"""AI CONTENT DETECTION workflow (separate from fake-news detection).

Determines whether TEXT and/or IMAGE look AI-generated (LLM-written prose,
diffusion-model imagery) vs human-made. Two engines per modality:

- Text: HuggingFace `openai-community/roberta-base-openai-detector`
  (cache-only; never downloads inside a request) blended with a documented
  stylometric heuristic (burstiness, lexical diversity, generic-phrasing,
  personal-voice markers). Engine used is always reported.
- Image: forensic heuristics (EXIF/generator metadata, ELA uniformity,
  noise/smoothness, saturation spikes) with documented indicators.

Returns probabilities + indicators + explanations + confidence. Never
fabricates: every indicator states the measured value behind it.
"""
import io
import math
import os
import re

AI_TEXT_MODEL = "openai-community/roberta-base-openai-detector"
_ai_text_clf = None

COMMON_WORDS = set("""the be to of and a in that have it for not on with he as you do at
this but his by from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time just him know take people into year your good some
could them see other than then now look only come its over think also back after use two how our work
first well way even new want because any these give day most us is are was were has had been will would
can could should may might must shall""".split())


def _try_load_ai_text():
    global _ai_text_clf
    if _ai_text_clf is not None:
        return _ai_text_clf or None
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"  # cache-only: never block on download
    try:
        from transformers import pipeline
        _ai_text_clf = pipeline("text-classification", model=AI_TEXT_MODEL)
        return _ai_text_clf
    except Exception:
        _ai_text_clf = False
        return None
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def _sentences(text):
    return [s.strip() for s in re.split(r"[.!?…\n]+", text) if len(s.split()) >= 3]


def stylometric_scores(text):
    """Heuristic P(AI) in [0,1] + human-readable indicators with measured values."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = len(words)
    indicators = []
    if n < 12:
        return 0.5, ["Text too short (<12 words) — stylometric signals unreliable; neutral score assigned."], {"words": n}
    sents = _sentences(text)
    lens = [len(s.split()) for s in sents] or [n]
    mean_l = sum(lens) / len(lens)
    var_l = sum((x - mean_l) ** 2 for x in lens) / len(lens)
    cv = math.sqrt(var_l) / max(1.0, mean_l)  # burstiness: humans vary sentence length
    bigrams = list(zip(words, words[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))  # lexical diversity
    common_ratio = sum(1 for w in words if w in COMMON_WORDS) / n  # generic phrasing
    personal = len(re.findall(r"\b(i|me|my|mine|we|us|our|you're|don't|can't|won't|it's|gonna|yeah|haha|omg|uh|um)\b", text.lower())) / n
    emdashes = text.count("—") + text.count("–")
    stock = len(re.findall(r"\b(delve|delves|tapestry|furthermore|moreover|additionally|it is important to note|in conclusion|overall,|landscape|realm|testament|pivotal|leverage|utilize|utilizing|showcase|underscores?|nestled|bustling|vibrant|boasts?|in today's fast-paced)\b", text, re.I))
    typo_like = len(re.findall(r"(.)\1{2,}|[!?]{2,}|\b(uh|um|hmm)\b", text.lower()))

    score = 0.5
    if cv < 0.45:
        score += 0.14
        indicators.append((f"+ AI-leaning: uniform sentence rhythm (burstiness CV={cv:.2f}; humans typically vary more)."))
    else:
        score -= 0.10
        indicators.append((f"− Human-leaning: varied sentence rhythm (CV={cv:.2f})."))
    if distinct2 < 0.75:
        score += 0.10
        indicators.append((f"+ AI-leaning: low bigram diversity ({distinct2:.0%} unique; repetitive phrasing)."))
    else:
        score -= 0.08
        indicators.append((f"− Human-leaning: rich vocabulary (bigram diversity {distinct2:.0%})."))
    if common_ratio > 0.60:
        score += 0.08
        indicators.append((f"+ AI-leaning: highly generic diction ({common_ratio:.0%} common words)."))
    if personal > 0.015 or typo_like > 0:
        score -= 0.14
        indicators.append((f"− Human-leaning: personal voice/casual markers present ({personal:.1%}, {typo_like} informal hits)."))
    if stock >= 2 or emdashes >= 3:
        score += 0.12
        indicators.append((f"+ AI-leaning: LLM-typical stock phrasing ({stock} tell-words, {emdashes} em-dashes)."))
    elif stock == 0 and emdashes == 0:
        score -= 0.05
        indicators.append(("− Human-leaning: no LLM stock phrases or em-dash styling."))
    stats = {"words": n, "sentences": len(sents), "burstiness_cv": round(cv, 3),
             "bigram_diversity": round(distinct2, 3), "common_word_ratio": round(common_ratio, 3),
             "stock_tells": stock}
    return float(min(0.95, max(0.05, score))), indicators, stats


def analyze_ai_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"present": False, "p_ai": None, "confidence": 0.0,
                "indicators": ["No text supplied."], "engine": "none"}
    h_score, indicators, stats = stylometric_scores(text)
    engine = "stylometric heuristic engine (burstiness, diversity, diction, voice)"
    model_p = None
    clf = _try_load_ai_text()
    if clf is not None:
        try:
            r = clf(text[:2000], truncation=True)[0]
            lab, sc = r["label"].lower(), float(r["score"])
            # openai detector labels: "fake" = machine-generated
            model_p = sc if "fake" in lab else 1.0 - sc
            engine = AI_TEXT_MODEL + " (fine-tuned detector) blended 60/40 with stylometrics"
        except Exception:
            model_p = None
    p_ai = float(min(0.97, max(0.03, (0.6 * model_p + 0.4 * h_score) if model_p is not None else h_score)))
    # Documented decisive rule: dense LLM stock vocabulary (≥8 tell-words) almost
    # never co-occurs in human news prose — floor the score so it can't be
    # washed out by a weak detector output on short texts.
    if stats.get("stock_tells", 0) >= 8:
        if p_ai < 0.66:
            p_ai = 0.66
        indicators.append(f"+ Strong AI signal: dense LLM stock vocabulary ({stats['stock_tells']} tell-words) — decisive floor applied.")
    conf = round(float(0.55 + 0.35 * min(1.0, stats["words"] / 300) + (0.05 if model_p is not None else 0.0)), 3)
    return {"present": True, "p_ai": round(p_ai, 3), "ai_pct": round(p_ai * 100, 1),
            "confidence": conf, "indicators": indicators, "stats": stats,
            "model_p_ai": round(model_p, 3) if model_p is not None else None, "engine": engine}


GENERATOR_TAGS = ("stable diffusion", "midjourney", "dall", "firefly", "sdxl", "comfyui",
                  "automatic1111", "novelai", "leonardo", "ideogram", "civitai", "tensor.art")


def analyze_ai_image(raw: bytes | None) -> dict:
    if not raw:
        return {"present": False, "p_ai": None, "confidence": 0.0,
                "indicators": ["No image supplied — image AI check skipped."], "engine": "none"}
    from PIL import Image
    import numpy as np
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return {"present": True, "p_ai": 0.5, "confidence": 0.0,
                "indicators": [f"Could not decode image: {e}"], "engine": "forensic heuristic engine"}
    w, h = img.size
    indicators = []
    score = 0.4  # slight human prior for real uploads

    # 1. Generator metadata (strongest signal when present)
    meta_hit = None
    try:
        exif = img.getexif() or {}
        blob = " ".join(str(v) for v in exif.values()) + " " + str(img.info)
        low = blob.lower()
        for tag in GENERATOR_TAGS:
            if tag in low:
                meta_hit = tag
                break
    except Exception:
        pass
    if meta_hit:
        score += 0.45
        indicators.append(f"+ Strong AI signal: generator tag '{meta_hit}' found in image metadata.")
    else:
        indicators.append("− No AI-generator tags in metadata (neutral; most uploads strip metadata).")

    # 2. ELA uniformity: recompress at q90, measure difference spread
    try:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        diff = np.asarray(img, dtype=float) - np.asarray(recomp, dtype=float)
        ela_std = float(np.abs(diff).std())
        ela_mean = float(np.abs(diff).mean())
        if ela_std < 6.0:
            score += 0.12
            indicators.append(f"+ AI-leaning: unusually uniform recompression response (ELA std={ela_std:.1f}, mean={ela_mean:.1f}).")
        else:
            score -= 0.08
            indicators.append(f"− Human-leaning: natural recompression texture (ELA std={ela_std:.1f}).")
    except Exception:
        ela_std, ela_mean = None, None

    # 3. Noise/smoothness via Laplacian variance
    try:
        import cv2
        gray = np.array(img.convert("L"))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 30:
            score += 0.10
            indicators.append(f"+ AI-leaning: over-smooth detail (sharpness var={lap_var:.1f}; diffusion outputs are often plastic-smooth).")
        else:
            score -= 0.08
            indicators.append(f"− Human-leaning: natural sensor detail (sharpness var={lap_var:.1f}).")
    except Exception:
        lap_var = None

    # 4. Saturation spikes (flat synthetic color regions)
    try:
        import cv2
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(float)
        spike = float(((sat > 250).mean() + (sat < 3).mean()))
        if spike > 0.35:
            score += 0.06
            indicators.append(f"+ Weak AI-leaning: flat synthetic-looking color zones ({spike:.0%} extreme saturation).")
    except Exception:
        pass

    p_ai = float(min(0.97, max(0.03, score)))
    res = max(1, w * h)
    conf = round(float(0.55 + 0.3 * min(1.0, res / (800 * 800)) + (0.1 if meta_hit else 0.0)), 3)
    return {"present": True, "p_ai": round(p_ai, 3), "ai_pct": round(p_ai * 100, 1),
            "confidence": conf, "indicators": indicators,
            "stats": {"width": w, "height": h, "ela_std": ela_std, "ela_mean": ela_mean,
                      "sharpness_var": lap_var, "metadata_hit": meta_hit},
            "engine": "forensic heuristic engine (metadata + ELA + noise + color)"}


def label_ai(p: float) -> str:
    if p >= 0.65:
        return "LIKELY AI-GENERATED"
    if p >= 0.45:
        return "UNCERTAIN — MIXED SIGNALS"
    return "LIKELY HUMAN-MADE"


# ---------- reference-panel features: score breakdown, key indicators, methods ----------

STOCK_RE = re.compile(r"\b(delve|delves|tapestry|furthermore|moreover|additionally|it is important to note|in conclusion|overall,|landscape|realm|testament|pivotal|leverage|utilize|utilizing|showcase|underscores?|nestled|bustling|vibrant|boasts?|in today's fast-paced)\b", re.I)


def _consecutive_similarity(sents):
    """Mean cosine similarity of consecutive sentences (TF-IDF). High = highly-structured AI-like flow."""
    if len(sents) < 2:
        return 0.5, 0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        v = TfidfVectorizer().fit_transform(sents)
        sims = [float(cosine_similarity(v[i:i + 1], v[i + 1:i + 2])[0, 0]) for i in range(len(sents) - 1)]
        return float(sum(sims) / len(sims)), len(sims)
    except Exception:
        return 0.5, 0


def score_breakdown(text: str, text_r: dict, image_r: dict) -> dict:
    """Five reference-panel sub-scores (0-100), each documented."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    n = len(words)
    sents = _sentences(text or "")
    stock = len(STOCK_RE.findall(text or ""))
    common_ratio = sum(1 for w in words if w in COMMON_WORDS) / max(1, n)

    linguistic = (min(95, max(5, int(round(
        15 + min(50, stock * 9) + common_ratio * 25 + (1 - (text_r.get('stats', {}).get('bigram_diversity', 0.8))) * 20
    )))) if n >= 12 else 50)
    sim, _ = _consecutive_similarity(sents)
    semantic = min(95, max(5, int(round(sim * 100)))) if len(sents) >= 2 else 50
    # perplexity PROXY (documented): rare-word + long-sentence predictability estimate
    rare_ratio = 1.0 - common_ratio
    avg_len = (sum(len(s.split()) for s in sents) / max(1, len(sents)))
    perplexity_proxy = min(95, max(5, int(round(100 - (rare_ratio * 70 + min(1.0, avg_len / 25) * 30))))) if n >= 12 else 50
    stylometric = int(round((text_r.get("p_ai", 0.5)) * 100)) if text_r.get("present") else 50
    forensics = int(round((image_r.get("p_ai", 0.5)) * 100)) if image_r.get("present") else None
    return {
        "linguistic_patterns": linguistic,
        "semantic_consistency": semantic,
        "perplexity_analysis": perplexity_proxy,
        "perplexity_note": "proxy estimate (rare-word + sentence-length model), not true LM perplexity",
        "stylometric_features": stylometric,
        "image_forensics": forensics,
    }


def key_indicators(text: str, text_r: dict, image_r: dict) -> list:
    """Four named Key-AI-Indicator cards like the reference panel."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    n = len(words)
    sents = _sentences(text or "")
    lens = [len(s.split()) for s in sents] or [0]
    cv = (sum((x - sum(lens) / max(1, len(lens))) ** 2 for x in lens) / max(1, len(lens))) ** 0.5 / max(1.0, (sum(lens) / max(1, len(lens))))
    bigrams = list(zip(words, words[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))
    personal = len(re.findall(r"\b(i|me|my|mine|we|us|our|you're|don't|can't|won't|it's|gonna|yeah|haha|omg)\b", (text or "").lower()))
    out = [
        {"name": "Low Perplexity Score",
         "desc": "Text is too predictable and consistent." if cv < 0.5 else "Sentence rhythm varies naturally.",
         "ai_leaning": bool(cv < 0.5 and n >= 12)},
        {"name": "Generic Content",
         "desc": "Uses generic and non-specific information." if distinct2 < 0.8 and n >= 12 else "Content carries specific, varied detail.",
         "ai_leaning": bool(distinct2 < 0.8 and n >= 12)},
        {"name": "Repetitive Patterns",
         "desc": "Similar sentence structures detected." if cv < 0.45 and len(sents) >= 2 else "No strong structural repetition.",
         "ai_leaning": bool(cv < 0.45 and len(sents) >= 2)},
        {"name": "Formal and Neutral Tone",
         "desc": "Very balanced and non-emotional language." if personal == 0 and n >= 12 else "Personal voice and emotion present.",
         "ai_leaning": bool(personal == 0 and n >= 12)},
    ]
    if image_r.get("present"):
        out.append({"name": "Synthetic Visual Texture",
                    "desc": image_r["indicators"][1] if len(image_r["indicators"]) > 1 else "Forensic image check completed.",
                    "ai_leaning": bool((image_r.get("p_ai", 0) or 0) >= 0.5)})
    return out


def detection_methods(text_r: dict, image_r: dict) -> list:
    methods = [
        {"name": "Transformer-based classifier",
         "detail": AI_TEXT_MODEL + (" (active)" if text_r.get("model_p_ai") is not None else " (not cached — stylometrics used)")},
        {"name": "Perplexity scoring", "detail": "Proxy estimate via rare-word + sentence-length model."},
        {"name": "Stylometric analysis", "detail": "Burstiness, bigram diversity, diction, personal voice."},
        {"name": "Semantic coherence check", "detail": "Consecutive-sentence TF-IDF similarity."},
    ]
    if image_r.get("present"):
        methods.append({"name": "Vision model (for images)", "detail": "Metadata + ELA + noise + color forensics."})
    return methods


def combine_ai(text_r: dict, image_r: dict) -> dict:
    parts, weights = [], []
    if text_r.get("present") and text_r.get("p_ai") is not None:
        parts.append(("text", text_r["p_ai"], text_r["confidence"]))
    if image_r.get("present") and image_r.get("p_ai") is not None:
        parts.append(("image", image_r["p_ai"], image_r["confidence"]))
    if not parts:
        return {"p_ai": 0.5, "label": "NO INPUT", "confidence": 0.0, "basis": "nothing to analyze"}
    tot_c = sum(c for _, _, c in parts) or 1.0
    p_ai = round(float(sum(p * c for _, p, c in parts) / tot_c), 3)
    conf = round(float(sum(c for _, _, c in parts) / len(parts)), 3)
    return {"p_ai": p_ai, "ai_pct": round(p_ai * 100, 1), "human_pct": round((1 - p_ai) * 100, 1),
            "label": label_ai(p_ai), "confidence": conf,
            "basis": "+".join(n for n, _, _ in parts) + " (confidence-weighted)"}
