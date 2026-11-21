"""Fit the official ASVspoof LA protocol and score the held-out evaluation once."""
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.audio_features import spectral_features
from training.train_public import metrics, save_report


def load_part(name):
    folder = ROOT / "data/prepared/asvspoof"
    manifest = folder / f"{name}.jsonl"
    while not manifest.exists():
        print("Waiting for official", name, "manifest", flush=True)
        time.sleep(45)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    tensors = np.load(rows[0]["tensor_path"], mmap_mode="r")
    batches = []
    for i in range(0,len(rows),64):
        indices = [r["tensor_index"] for r in rows[i:i+64]]
        batches.append(spectral_features(tensors[indices]))
    return np.concatenate(batches), np.array([r["label"] for r in rows]), rows


def check_split(rows, previous):
    for field in ("group", "sha256"):
        if {r[field] for r in rows} & {r[field] for r in previous}:
            raise ValueError("Official partitions overlap on " + field)


def run():
    train_x, train_y, train_rows = load_part("train")
    val_x, val_y, val_rows = load_part("val")
    check_split(val_rows, train_rows)
    # Validation chooses model complexity. The evaluation partition is read after this choice.
    best_f1 = -1
    for depth in (7, 15):
        clf = HistGradientBoostingClassifier(max_leaf_nodes=depth,max_iter=180,class_weight="balanced",random_state=42)
        clf.fit(train_x, train_y)
        raw = clf.decision_function(val_x).reshape(-1,1)
        calibration = LogisticRegression().fit(raw,val_y)
        score = metrics(val_y,calibration.predict_proba(raw)[:,1])["macro_f1"]
        print("dev macro F1",depth,score,flush=True)
        if score > best_f1:
            best_f1, best = score, {"classifier":clf,"calibration":calibration}
    del train_x,val_x
    # Save the validated candidate locally so evaluation can resume after interruption.
    candidate = ROOT / "data/prepared/asvspoof/official-candidate.joblib"
    joblib.dump(best,candidate)
    test_x,test_y,test_rows = load_part("test")
    check_split(test_rows,train_rows+val_rows)
    scores = best["calibration"].predict_proba(best["classifier"].decision_function(test_x).reshape(-1,1))[:,1]
    result = metrics(test_y,scores)
    fpr,tpr,_ = roc_curve(test_y,scores)
    index = np.argmin(np.abs(fpr-(1-tpr)))
    result["eer"] = float((fpr[index]+1-tpr[index])/2)
    folder = save_report("audio",{"dataset":"ASVspoof 2019 LA official partitions",
        "architecture":"Spectral band statistics and gradient boosting","artifact_format":"joblib",
        "counts":{"train":len(train_y),"validation":len(val_y),"test":len(test_y)},"test":result,
        "validation_macro_f1":best_f1,"evaluation_scope":"official-train-dev-eval",
        "limitations":"First six seconds of English LA recordings. Generalization to calls, music, newer generators or speaker identity has not been established."})
    shutil.copyfile(candidate,folder/"model.joblib")
    shutil.copyfile(ROOT/"data/public/asvspoof/provenance.json",folder/"dataset_provenance.json")


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        run()
