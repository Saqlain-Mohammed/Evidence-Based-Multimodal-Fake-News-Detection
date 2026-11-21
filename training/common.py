"""Shared helpers for training scripts."""
import json
import os
import random


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def save_metrics(artifacts_dir, fragment: dict):
    """Merge a metrics fragment into artifacts/metrics.json.
    One-level deep merge so nested sections (e.g. text_model.test)
    don't wipe sibling keys like val/epochs."""
    os.makedirs(artifacts_dir, exist_ok=True)
    mp = os.path.join(artifacts_dir, "metrics.json")
    data = {}
    if os.path.exists(mp):
        try:
            data = json.load(open(mp, encoding="utf-8"))
        except Exception:
            data = {}
    for k, v in fragment.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    json.dump(data, open(mp, "w", encoding="utf-8"), indent=2)
    return mp


def prf(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {"accuracy": round(acc, 4), "precision": round(float(p), 4),
            "recall": round(float(r), 4), "f1": round(float(f), 4),
            "confusion_matrix": cm, "n": len(y_true)}
