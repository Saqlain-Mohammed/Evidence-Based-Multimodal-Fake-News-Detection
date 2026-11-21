"""AI-text classifier training (human vs AI-generated), TF-IDF + LogReg.

Separate task from fake-news (2x2 documented in README). No stock-phrase floor,
no hard-coded rules — purely learned, validated on held-out story groups.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from common import read_jsonl  # noqa: E402
from text_clean import title_body  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/realistic")
    ap.add_argument("--out", default="../models/ai_text.pkl")
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score
    import pickle

    def xy(split):
        rows = read_jsonl(os.path.join(args.data, split + ".jsonl"))
        return ([title_body(r.get("title"), r.get("text")) for r in rows],
                [r["ai_label"] for r in rows])

    Xtr, ytr = xy("train")
    Xva, yva = xy("val")
    clf = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    f1m = float(f1_score(yva, pred, average="macro", zero_division=0))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pickle.dump({"pipeline": clf, "task": "ai_text",
                 "labels": {0: "human", 1: "ai"}}, open(args.out, "wb"))
    print({"val_macro_f1": round(f1m, 4), "n_train": len(ytr), "n_val": len(yva)})


if __name__ == "__main__":
    main()
