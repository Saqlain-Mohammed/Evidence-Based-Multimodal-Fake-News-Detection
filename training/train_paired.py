"""Train a joint title/image classifier on the public Fakeddit paired subset."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.paired_features import image_features, paired_features
from training.train_public import metrics, normalize, save_report


def read_splits(folder):
    # Reserve test examples first, then validation; exclude repeated title, URL or image across all splits.
    seen = set()
    parts = {}
    with zipfile.ZipFile(folder / "image_cache.zip") as archive:
        members = {Path(s).name: s for s in archive.namelist() if not s.endswith("/") and "__MACOSX/" not in s}
        for split in ("test", "val", "train"):
            rows = []
            with (folder / f"{split}_balanced.csv").open(encoding="utf-8-sig", newline="") as f:
                for record in csv.DictReader(f):
                    name = Path(record["image_path"]).name
                    if name not in members: continue
                    raw = archive.read(members[name])
                    title = record["text"].strip()
                    if not title: continue
                    keys = {("title", normalize(title)), ("image", hashlib.sha256(raw).hexdigest())}
                    if record["original_url"]: keys.add(("url", record["original_url"]))
                    if keys & seen: continue
                    try:
                        image = image_features(raw)
                    except OSError:
                        continue
                    seen.update(keys)
                    rows.append((title, image, int(record["6_way_label"]) != 0))
            if len({r[2] for r in rows}) != 2: raise ValueError(f"Missing paired classes in {split}")
            parts[split] = rows
            print(split, len(rows), flush=True)
    return parts


def train():
    folder = ROOT / "data/public/fakeddit_pairs"
    parts = read_splits(folder)
    model = {"text": make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                    TruncatedSVD(n_components=96, random_state=42))}
    model["text"].fit([r[0] for r in parts["train"]])
    x, y = {}, {}
    for name, rows in parts.items():
        titles, images, labels = zip(*rows)
        x[name] = paired_features(model, titles, images)
        y[name] = np.asarray(labels, dtype=int)
    ablations = {}
    for view, columns in (("title_only", slice(None, 96)), ("image_only", slice(96, None)), ("joint", slice(None))):
        classifier = HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, class_weight="balanced", random_state=42)
        classifier.fit(x["train"][:, columns], y["train"])
        calibration = LogisticRegression().fit(classifier.decision_function(x["val"][:, columns]).reshape(-1, 1), y["val"])
        scores = calibration.predict_proba(classifier.decision_function(x["test"][:, columns]).reshape(-1, 1))[:, 1]
        ablations[view] = metrics(y["test"], scores)
        if view == "joint": model.update(classifier=classifier, calibration=calibration)
    result = save_report("paired_news", {"dataset": "Fakeddit balanced public subset", "architecture": "TF-IDF/SVD and spatial RGB statistics with joint gradient boosting",
        "counts": {("validation" if k == "val" else k):len(v) for k,v in parts.items()}, "test": ablations["joint"],
        "modality_baselines": ablations, "class_counts": {k:np.bincount(v).tolist() for k,v in y.items()},
        "evaluation_scope": "mirror split with duplicate titles, image bytes and source URLs removed",
        "limitations": "Predicts Fakeddit class 0 versus all other community-derived classes, including satire. Historical Reddit benchmark. Requires a title and image together. No independent fact-checking or video/audio fusion."})
    joblib.dump(model, result / "model.joblib", compress=3)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
