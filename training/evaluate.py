"""PHASE 11 — Evaluate trained artifacts on held-out test splits
(and optional cross-dataset files in the same jsonl format, e.g. VERITE /
FineFake converted with scripts in training/converters/).

Writes artifacts/metrics.json {text_model.test, image_model.test,
fusion.test, cross: {...}} and prints a report.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, save_metrics, prf
from train_fusion import text_probs, image_probs, featurize


def eval_text(rows, text_dir):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(text_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    preds, gold = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            chunk = rows[i:i + 16]
            texts = [((r.get("title") or "") + " " + (r.get("text") or ""))[:2000] for r in chunk]
            e = tok(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
            e = {k: v.to(device) for k, v in e.items()}
            preds += model(**e).logits.argmax(-1).cpu().tolist()
            gold += [r["label"] for r in chunk]
    return prf(gold, preds)


def eval_image(rows, image_pt):
    import torch
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.models as models
    blob = torch.load(image_pt, map_location="cpu")
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    items = [r for r in rows if r.get("image") and os.path.exists(r["image"])]
    preds, gold = [], []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            imgs = []
            for r in chunk:
                try:
                    imgs.append(tf(Image.open(r["image"]).convert("RGB")))
                except Exception:
                    imgs.append(tf(Image.new("RGB", (224, 224), (128, 128, 128))))
            preds += m(torch.stack(imgs)).argmax(-1).tolist()
            gold += [r["label"] for r in chunk]
    return prf(gold, preds)


def eval_fusion(rows, args):
    import pickle
    tp, _ = text_probs(rows, args.text_model)
    ip, _ = image_probs(rows, args.image_model)
    X, y = featurize(rows, tp, ip)
    clf = pickle.load(open(args.fusion, "rb"))["pipeline"]
    return prf(y.tolist(), clf.predict(X).tolist())


def main(args):
    art = os.path.abspath(args.artifacts)
    test_rows = read_jsonl(os.path.join(args.data, "test.jsonl"))
    report = {}
    if os.path.isdir(args.text_model):
        m = eval_text(test_rows, args.text_model)
        report.setdefault("text_model", {})["test"] = m
        print("TEXT :", m)
    if os.path.exists(args.image_model):
        m = eval_image(test_rows, args.image_model)
        report.setdefault("image_model", {})["test"] = m
        print("IMAGE:", m)
    if os.path.exists(args.fusion):
        m = eval_fusion(test_rows, args)
        report.setdefault("fusion", {})["test"] = m
        print("FUSION:", m)
    cross = {}
    if args.cross:
        for f in args.cross.split(","):
            name = os.path.splitext(os.path.basename(f))[0]
            rows = read_jsonl(f)
            if os.path.exists(args.fusion):
                cross[name] = eval_fusion(rows, args)
                print(f"CROSS[{name}]:", cross[name])
    report["cross"] = cross
    save_metrics(art, report)
    print(f"metrics -> {os.path.join(art, 'metrics.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fakeddit")
    ap.add_argument("--artifacts", default="../backend/artifacts")
    ap.add_argument("--text-model", default="../backend/artifacts/text_model")
    ap.add_argument("--image-model", default="../backend/artifacts/image_model.pt")
    ap.add_argument("--fusion", default="../backend/artifacts/fusion.pkl")
    ap.add_argument("--cross", default="", help="comma-separated jsonl files for cross-dataset eval")
    args, _ = ap.parse_known_args()
    main(args)
