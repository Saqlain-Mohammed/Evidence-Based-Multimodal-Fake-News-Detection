"""Baselines for fake-news text: TF-IDF + LogReg / Linear SVM (stratified, weighted).

Compares against the transformer in the eval report; selection is by
validation macro-F1, never test.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from common import read_jsonl  # noqa: E402
from text_clean import title_body  # noqa: E402


def load(data):
    out = {}
    for split in ("train", "val"):
        rows = read_jsonl(os.path.join(data, split + ".jsonl"))
        out[split] = ([title_body(r.get("title"), r.get("text")) for r in rows],
                      [r["label"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/realistic")
    ap.add_argument("--out", default="../models/baselines")
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score
    import pickle

    d = load(args.data)
    Xtr, ytr = d["train"]
    Xva, yva = d["val"]
    cands = {
        "tfidf_logreg": make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
                                      LogisticRegression(max_iter=1000, class_weight="balanced")),
        "tfidf_svm": make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
                                   LinearSVC(class_weight="balanced")),
    }
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for name, clf in cands.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        f1m = float(f1_score(yva, pred, average="macro", zero_division=0))
        res[name] = {"val_macro_f1": round(f1m, 4)}
        pickle.dump(clf, open(os.path.join(args.out, name + ".pkl"), "wb"))
        print(name, res[name])
    json.dump(res, open(os.path.join(args.out, "baselines.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
