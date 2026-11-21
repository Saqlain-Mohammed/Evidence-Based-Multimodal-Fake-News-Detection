"""Loads trained artifacts from backend/artifacts/ (written by training/).
Everything is lazy + optional: any missing artifact -> (None, False) and the
caller keeps the pre-trained/fallback behavior. Thread-safe-ish via locks.
"""
import json
import os
import threading

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(BASE, "artifacts")

_lock = threading.Lock()
_text = None      # (tokenizer, model, device) or False when unavailable
_image = None     # torch model or False
_fusion = None    # sklearn pipeline dict or False
_metrics_cache = None


def artifacts_dir():
    return ART


def metrics():
    global _metrics_cache
    mp = os.path.join(ART, "metrics.json")
    if not os.path.exists(mp):
        return {"trained": False}
    try:
        d = json.load(open(mp, encoding="utf-8"))
        d["trained"] = True
        return d
    except Exception:
        return {"trained": False}


def get_text_model():
    """Returns (tokenizer, model, device, info) or None."""
    global _text
    with _lock:
        if _text is not None:
            return _text or None
        d = os.path.join(ART, "text_model")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
            info = {}
            try:
                info = json.load(open(os.path.join(d, "model_info.json"), encoding="utf-8"))
            except Exception:
                pass
            _text = (tok, model, device, info)
            return _text
        except Exception:
            _text = False
            return None


def text_fake_prob(text: str):
    """P(fake) from the fine-tuned text model, or None if unavailable."""
    tm = get_text_model()
    if tm is None:
        return None
    tok, model, device, _ = tm
    try:
        import torch
        with torch.no_grad():
            e = tok((text or "")[:2000], truncation=True, max_length=128,
                    padding=True, return_tensors="pt")
            e = {k: v.to(device) for k, v in e.items()}
            probs = torch.nn.Softmax(dim=-1)(model(**e).logits)[0].tolist()
        return float(probs[0])  # index 0 = fake
    except Exception:
        return None


def get_image_model():
    global _image
    with _lock:
        if _image is not None:
            return _image or None
        pt = os.path.join(ART, "image_model.pt")
        try:
            import torch
            import torchvision.models as models
            blob = torch.load(pt, map_location="cpu")
            m = models.resnet50(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 2)
            m.load_state_dict(blob["state_dict"])
            m.eval()
            _image = m
            return m
        except Exception:
            _image = False
            return None


def image_fake_prob(pil_image):
    m = get_image_model()
    if m is None:
        return None
    try:
        import torch
        import torchvision.transforms as T
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with torch.no_grad():
            probs = torch.nn.Softmax(dim=-1)(m(tf(pil_image).unsqueeze(0)))[0].tolist()
        return float(probs[0])
    except Exception:
        return None


def get_fusion():
    """Returns sklearn fusion pipeline or None."""
    global _fusion
    with _lock:
        if _fusion is not None:
            return _fusion or None
        try:
            import pickle
            blob = pickle.load(open(os.path.join(ART, "fusion.pkl"), "rb"))
            _fusion = blob["pipeline"]
            return _fusion
        except Exception:
            _fusion = False
            return None


def fusion_fake_prob(features):
    """features: [p_fake_text, p_fake_image, source_sig, evidence_match, contra_sig, agreement].

    Returns P(fake)=P(class 0). NOTE (audit fix Sep-2026): this previously
    returned index 1 = P(real), inverting every verdict. Class order is looked
    up from clf.classes_ so retrained heads stay correct.
    """
    clf = get_fusion()
    if clf is None:
        return None
    try:
        import numpy as np
        proba = clf.predict_proba(np.array([features], dtype=float))[0]
        classes = list(getattr(clf, "classes_", [0, 1]))
        if 0 in classes:
            return float(proba[classes.index(0)])
        return float(proba[0])
    except Exception:
        return None
