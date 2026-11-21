"""Train a three-way MultiNLI baseline using the official training split."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.stance_features import LABELS, encode


def batches(split):
    path = ROOT / "data/public/multinli" / f"{split}.parquet"
    source = pq.ParquetFile(path)
    metadata = json.loads(source.schema_arrow.metadata[b"huggingface"])
    assert tuple(metadata["info"]["features"]["label"]["names"]) == LABELS
    for batch in source.iter_batches(batch_size=2048, columns=["premise", "hypothesis", "label"]):
        data = batch.to_pydict()
        ids = [i for i, label in enumerate(data["label"]) if label in (0, 1, 2)]
        yield encode((data["premise"][i], data["hypothesis"][i]) for i in ids), np.array(data["label"])[ids]


def evaluate(model, split):
    actual, predictions = [], []
    for features, labels in batches(split):
        actual.extend(labels)
        predictions.extend(model.predict(features))
    return {"accuracy": accuracy_score(actual, predictions), "macro_f1": f1_score(actual, predictions, average="macro"),
            "confusion_matrix": confusion_matrix(actual, predictions).tolist(), "n": len(actual)}


if __name__ == "__main__":
    model = SGDClassifier(loss="log_loss", alpha=0.00004, random_state=42, average=True)
    count = 0
    for features, labels in batches("train"):
        model.partial_fit(features, labels, classes=np.arange(3))
        count += len(labels)
    folder = ROOT / "models/public/evidence_stance"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"task": "evidence_stance", "dataset": "MultiNLI", "architecture": "Hashed paired lexical features and averaged SGD",
              "trained_from_scratch": True, "synthetic": False, "labels": LABELS, "training_pairs": count,
              "matched_development": evaluate(model, "validation_matched"),
              "mismatched_development": evaluate(model, "validation_mismatched"),
              "limitations": "Development splits only. Sentence relations are experimental and do not verify factual accuracy or source reliability."}
    print(report, flush=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(model, folder / "model.joblib", compress=3)
