"""PHASE 4 — Train the multimodal fusion head (sklearn MLP).

Features per item (6 dims):
  [p_fake_text, p_fake_image, source_fake_signal, evidence_match,
   contradiction_fake_signal, text_image_agreement]

- p_fake_* come from the fine-tuned models (neutral 0.5 if a model is missing,
  recorded in metrics so it is never silent).
- source_fake_signal uses the backend's documented source-credibility rules.
- evidence_match / contradiction at TRAIN time use cheap offline proxies
  (TF-IDF title↔body consistency; negation-based contradiction proxy), because
  live retrieval over thousands of rows is infeasible. Documented in metrics
  as "training proxies"; inference uses live values. See README §Methodology.

Saves artifacts/fusion.pkl (StandardScaler + MLPClassifier pipeline).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from common import read_jsonl, set_seed, save_metrics, prf

NEG = {"not", "no", "never", "false", "fake", "denied", "denies", "debunked",
       "refuted", "hoax", "rumor", "rumour", "scam"}


def tfidf_proxy(a, b):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        if not a.strip() or not b.strip():
            return 0.0
        v = TfidfVectorizer().fit_transform([a, b])
        return float(cosine_similarity(v[0:1], v[1:])[0, 0])
    except Exception:
        return 0.0


def contra_proxy(title, text):
    tw = set(re.findall(r"[a-z]+", title.lower()))
    ew = set(re.findall(r"[a-z]+", text.lower()))
    if not tw:
        return 0.5
    neg = any(w in ew for w in NEG)
    overlap = len(tw & ew) / max(1, len(tw))
    if neg and overlap > 0.1:
        return 0.8
    if overlap > 0.25:
        return 0.2
    return 0.5


def source_proxy(source):
    try:
        from modules.source_credibility import score_source
        return score_source(source if "." in (source or "") else "", "")["fake_signal"]
    except Exception:
        return 0.5


def text_probs(rows, text_dir, batch=16):
    """Return {id: p_fake} using fine-tuned text model, or all-neutral."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        tok = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(text_dir, local_files_only=True)
    except Exception as e:
        print(f"text model unavailable ({e}) -> neutral 0.5 (recorded in metrics)")
        return {r["id"]: 0.5 for r in rows}, False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out = {}
    sm = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            texts = [((r.get("title") or "") + " " + (r.get("text") or ""))[:2000] for r in chunk]
            e = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
            e = {k: v.to(device) for k, v in e.items()}
            probs = sm(model(**e).logits).cpu().tolist()
            for r, p in zip(chunk, probs):
                out[r["id"]] = float(p[0])  # index 0 = fake
    return out, True


def image_probs(rows, image_pt, batch=16):
    import torch
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    if not (image_pt and os.path.exists(image_pt)):
        print("image model unavailable -> neutral 0.5 (recorded in metrics)")
        return {r["id"]: 0.5 for r in rows}, False
    try:
        blob = torch.load(image_pt, map_location="cpu")
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(blob["state_dict"])
    except Exception as e:
        print(f"image model unloadable ({e}) -> neutral 0.5")
        return {r["id"]: 0.5 for r in rows}, False
    m.eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    sm = torch.nn.Softmax(dim=-1)
    out = {}
    with torch.no_grad():
        items = [r for r in rows if r.get("image") and os.path.exists(r["image"])]
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            imgs = []
            for r in chunk:
                try:
                    imgs.append(tf(Image.open(r["image"]).convert("RGB")))
                except Exception:
                    imgs.append(tf(Image.new("RGB", (224, 224), (128, 128, 128))))
            probs = sm(m(torch.stack(imgs))).tolist()
            for r, p in zip(chunk, probs):
                out[r["id"]] = float(p[0])
    for r in rows:
        out.setdefault(r["id"], 0.5)
    return out, True


def featurize(rows, tp, ip):
    import numpy as np
    X, y = [], []
    for r in rows:
        t, b = r.get("title", ""), r.get("text", "")
        ev = tfidf_proxy(t, b)
        contra = contra_proxy(t, b)
        src = source_proxy(r.get("source", ""))
        pt, pi = tp.get(r["id"], 0.5), ip.get(r["id"], 0.5)
        agree = 1.0 - abs(pt - pi)
        X.append([pt, pi, src, ev, contra, agree])
        y.append(r["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def main(args):
    set_seed(42)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import pickle

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(args.data, "val.jsonl"))
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    tp, text_ok = text_probs(train_rows + val_rows, args.text_model)
    ip, img_ok = image_probs(train_rows + val_rows, args.image_model)
    Xtr, ytr = featurize(train_rows, tp, ip)
    Xva, yva = featurize(val_rows, tp, ip)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=42))
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)
    m = prf(yva.tolist(), preds.tolist())
    print(f"fusion val: {m}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump({"pipeline": clf,
                 "features": ["p_fake_text", "p_fake_image", "source_fake_signal",
                              "evidence_match_proxy", "contradiction_proxy", "text_image_agreement"],
                 "proxies": "evidence/contradiction are offline training proxies (TF-IDF consistency, negation); inference uses live values."},
                open(args.out, "wb"))
    art = os.path.dirname(os.path.abspath(args.out))
    save_metrics(art, {"fusion": {"val": m, "text_model_used": text_ok,
                                  "image_model_used": img_ok,
                                  "note": "MLP on 6 modality features; evidence/contra are training proxies."}})
    print(f"saved fusion -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fakeddit")
    ap.add_argument("--text-model", default="../backend/artifacts/text_model")
    ap.add_argument("--image-model", default="../backend/artifacts/image_model.pt")
    ap.add_argument("--out", default="../backend/artifacts/fusion.pkl")
    ap.add_argument("--max-train", type=int, default=0)
    args, _ = ap.parse_known_args()
    main(args)
