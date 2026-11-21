"""Fusion v2: retrain multimodal head on leakage-free data/realistic.

Features (6): [p_fake_text(baseline TF-IDF), p_fake_image(image_resnet),
source_fake_signal, evidence_match_proxy, contradiction_proxy, agreement].
Fixes audited bugs: correct P(fake)=P(class 0) convention, validation-based
selection (MLP vs LogReg), class_weight balanced, proxies documented.
Saves backend/artifacts/fusion.pkl (backs up v1 to fusion.v1.pkl).
"""
import argparse, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "preprocessing"))
from common import read_jsonl, set_seed  # noqa: E402
from text_clean import title_body  # noqa: E402
from train_fusion import (tfidf_proxy, contra_proxy, source_proxy,  # noqa: E402
                          image_probs as _img_probs)


def text_probs_baseline(rows, model_path):
    clf = pickle.load(open(model_path, "rb"))
    texts = [title_body(r.get("title"), r.get("text")) or "empty" for r in rows]
    p_real = clf.predict_proba(texts)[:, 1]
    return {r["id"]: float(1.0 - p) for r, p in zip(rows, p_real)}


def image_probs_v2(rows, image_pt, data_dir, batch=32):
    import torch
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    tf = T.Compose([T.Resize((160, 160)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def resolve(p):
        if p and os.path.exists(p):
            return p
        if p and os.path.exists(os.path.join(data_dir, p)):
            return os.path.join(data_dir, p)
        return ""

    blob = torch.load(image_pt, map_location="cpu")
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    sm = torch.nn.Softmax(dim=-1)
    out = {}
    items = [(r, resolve(r.get("image", ""))) for r in rows]
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            imgs = []
            for _, p in chunk:
                try:
                    imgs.append(tf(Image.open(p).convert("RGB")))
                except Exception:
                    imgs.append(tf(Image.new("RGB", (160, 160), (128, 128, 128))))
            probs = sm(m(torch.stack(imgs))).tolist()
            for (r, _), pr in zip(chunk, probs):
                out[r["id"]] = float(pr[0])  # index 0 = fake
    return out


def featurize(rows, tp, ip):
    import numpy as np
    X, y = [], []
    for r in rows:
        t, b = r.get("title", ""), r.get("text", "")
        ev = tfidf_proxy(t, b)
        contra = contra_proxy(t, b)
        src = source_proxy(r.get("source", ""))
        pt, pi = tp.get(r["id"], 0.5), ip.get(r["id"], 0.5)
        X.append([pt, pi, src, ev, contra, 1.0 - abs(pt - pi)])
        y.append(r["label"])
    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/realistic")
    ap.add_argument("--text-baseline", default="../models/baselines/tfidf_logreg.pkl")
    ap.add_argument("--image-model", default="../models/image_resnet.pt")
    ap.add_argument("--out", default="../backend/artifacts/fusion.pkl")
    args = ap.parse_args()
    set_seed(42)
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(args.data, "val.jsonl"))
    tp = text_probs_baseline(train_rows + val_rows, args.text_baseline)
    ip = image_probs_v2(train_rows + val_rows, args.image_model,
                        os.path.abspath(args.data))
    Xtr, ytr = featurize(train_rows, tp, ip)
    Xva, yva = featurize(val_rows, tp, ip)
    cands = {
        "mlp32": make_pipeline(StandardScaler(),
                               MLPClassifier(hidden_layer_sizes=(32,), max_iter=800,
                                             random_state=42)),
        "logreg_bal": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=1000,
                                                       class_weight="balanced")),
    }
    best, best_name, best_f1 = None, "", -1
    for name, clf in cands.items():
        clf.fit(Xtr, ytr)
        f1m = float(f1_score(yva, clf.predict(Xva), average="macro", zero_division=0))
        print(f"{name}: val_macro_f1={f1m:.4f}")
        if f1m > best_f1:
            best, best_name, best_f1 = clf, name, f1m
    print(f"selected: {best_name} ({best_f1:.4f})")
    if os.path.exists(args.out):
        os.replace(args.out, args.out + ".v1bak")
        print("v1 fusion backed up to fusion.pkl.v1bak")
    pickle.dump({"pipeline": best, "selected": best_name,
                 "val_macro_f1": round(best_f1, 4),
                 "features": ["p_fake_text", "p_fake_image", "source_fake_signal",
                              "evidence_match_proxy", "contradiction_proxy",
                              "text_image_agreement"],
                 "proxies": "evidence/contra offline proxies; inference uses live values.",
                 "text_engine": "tfidf-logreg v2", "image_engine": "image_resnet v2"},
                open(args.out, "wb"))
    print(f"saved fusion v2 -> {args.out}")


if __name__ == "__main__":
    main()
