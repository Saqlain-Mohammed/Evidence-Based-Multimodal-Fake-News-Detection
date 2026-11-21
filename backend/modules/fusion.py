"""Multimodal late-fusion classifier.
Fuses text + image + source + evidence + contradiction signals, then a
small logistic head produces the final REAL/FAKE decision with explanation."""

WEIGHTS = {"text": 0.30, "image": 0.15, "source": 0.20, "evidence": 0.25, "contradiction": 0.10}


def label_for(p: float) -> str:
    if p >= 0.65:
        return "LIKELY FAKE"
    elif p >= 0.45:
        return "UNCERTAIN / NEEDS REVIEW"
    return "LIKELY REAL"


def fuse(text_sig: float, image_sig: float | None, source_sig: float,
         evidence_match: float, contra_sig: float, has_image: bool) -> dict:
    img = image_sig if (has_image and image_sig is not None) else 0.5
    ev_sig = 1.0 - max(0.0, min(1.0, evidence_match))  # low match -> higher fake signal
    w = dict(WEIGHTS)
    if not has_image:
        # redistribute image weight to text + evidence
        w = {"text": 0.36, "image": 0.0, "source": 0.22, "evidence": 0.30, "contradiction": 0.12}
    fused = (w["text"] * text_sig + w["image"] * img + w["source"] * source_sig
             + w["evidence"] * ev_sig + w["contradiction"] * contra_sig)
    # logistic calibration (documented): sharpen mid-range
    import math
    logit = (fused - 0.5) * 6.0
    fake_prob = 1.0 / (1.0 + math.exp(-logit))
    fake_prob = round(float(fake_prob), 3)
    credibility = round(float((1.0 - fake_prob) * 100), 1)
    label = label_for(fake_prob)
    contribs = {
        "TEXT ANALYSIS": round(w["text"] * text_sig, 3),
        "IMAGE ANALYSIS": round(w["image"] * img, 3),
        "SOURCE CREDIBILITY": round(w["source"] * source_sig, 3),
        "EVIDENCE MATCH": round(w["evidence"] * ev_sig, 3),
        "CONTRADICTION": round(w["contradiction"] * contra_sig, 3),
    }
    per_mod = {
        "TEXT ANALYSIS": round(text_sig * 100, 1),
        "IMAGE ANALYSIS": round(img * 100, 1),
        "SOURCE CREDIBILITY": round(source_sig * 100, 1),
        "EVIDENCE MATCH": round(ev_sig * 100, 1),
    }
    return {"fake_probability": fake_prob, "fake_pct": round(fake_prob * 100, 1),
            "credibility_score": credibility, "label": label,
            "weights": w, "contributions": contribs, "per_modality_fake_pct": per_mod,
            "architecture": "late fusion: modality fake-signals -> weighted concat -> logistic head"}
