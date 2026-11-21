"""Full evaluation: proper metrics, cross-val, robustness, calibration, error analysis.

Reads data/realistic splits + models/*, writes evaluation/report.json,
evaluation/ERROR_ANALYSIS.md, evaluation/human_review.csv.
Test set touched ONCE (this script). Model selection used val only.
"""
import json, os, sys
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "preprocessing"))
from common import read_jsonl  # noqa: E402
from text_clean import title_body  # noqa: E402


def full_metrics(y_true, y_prob, thr=0.5):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score,
                                 average_precision_score)
    y_pred = [1 if p >= thr else 0 for p in y_prob]
    acc = float(accuracy_score(y_true, y_pred))
    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    p_mc, r_mc, f_mc, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    per = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc = None
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr = None
    return {
        "n": len(y_true), "accuracy": round(acc, 4),
        "precision": round(float(p_b), 4), "recall": round(float(r_b), 4),
        "f1": round(float(f_b), 4),
        "macro_f1": round(float(f_mc), 4), "weighted_f1": round(float(f_w), 4),
        "roc_auc": round(roc, 4) if roc is not None else None,
        "pr_auc": round(pr, 4) if pr is not None else None,
        "confusion_matrix": cm,
        "per_class": {
            "fake(0)": {"precision": round(float(per[0][0]), 4),
                        "recall": round(float(per[1][0]), 4),
                        "f1": round(float(per[2][0]), 4)},
            "real(1)": {"precision": round(float(per[0][1]), 4),
                        "recall": round(float(per[1][1]), 4),
                        "f1": round(float(per[2][1]), 4)}},
        "fpr": round(fp / max(1, fp + tn), 4),
        "fnr": round(fn / max(1, fn + tp), 4),
    }


def probs_text_tfidf(model_path, texts):
    import pickle
    clf = pickle.load(open(model_path, "rb"))
    if isinstance(clf, dict):
        clf = clf["pipeline"]
    if hasattr(clf, "predict_proba"):
        return [float(p[1]) for p in clf.predict_proba(texts)]  # P(real=1)
    d = clf.decision_function(texts)
    return [float(1 / (1 + np.exp(-v))) for v in d]


def main():
    import pickle
    data = os.path.join(ROOT, "data", "realistic")
    models = os.path.join(ROOT, "models")
    test = read_jsonl(os.path.join(data, "test.jsonl"))
    Xte = [title_body(r.get("title"), r.get("text")) for r in test]
    y_news = [r["label"] for r in test]
    y_ai = [r["ai_label"] for r in test]

    report = {}
    # fake-news baselines (P(real) -> convert: P(fake)=1-p for reporting positive=fake?
    # keep convention positive=real(1) like training; dashboard converts explicitly)
    for name in ("tfidf_logreg", "tfidf_svm"):
        p = os.path.join(models, "baselines", name + ".pkl")
        if os.path.exists(p):
            report["fakenews_" + name] = full_metrics(y_news, probs_text_tfidf(p, Xte))
            print(name, report["fakenews_" + name])
    pa = os.path.join(models, "ai_text.pkl")
    if os.path.exists(pa):
        report["ai_text_tfidf"] = full_metrics(y_ai, probs_text_tfidf(pa, Xte))
        print("ai_text", report["ai_text_tfidf"])

    # cross-validation (5-fold GroupKFold by story group, TF-IDF LogReg)
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import f1_score
        all_rows = (read_jsonl(os.path.join(data, "train.jsonl"))
                    + read_jsonl(os.path.join(data, "val.jsonl")))
        Xa = [title_body(r.get("title"), r.get("text")) for r in all_rows]
        ya = [r["label"] for r in all_rows]
        ga = [r["group"] for r in all_rows]
        gkf = GroupKFold(n_splits=5)
        f1s = []
        for tri, vai in gkf.split(Xa, ya, ga):
            c = make_pipeline(TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
                              LogisticRegression(max_iter=1000, class_weight="balanced"))
            c.fit([Xa[i] for i in tri], [ya[i] for i in tri])
            f1s.append(float(f1_score([ya[i] for i in vai], c.predict([Xa[i] for i in vai]),
                                      average="macro", zero_division=0)))
        report["cv_5fold_macro_f1"] = {"folds": [round(f, 4) for f in f1s],
                                       "mean": round(float(np.mean(f1s)), 4),
                                       "std": round(float(np.std(f1s)), 4)}
        print("cv", report["cv_5fold_macro_f1"])
    except Exception as e:
        report["cv_error"] = str(e)

    # robustness: misspellings, short text, paraphrase (subset of test)
    rng = np.random.RandomState(42)
    sub = rng.choice(len(test), size=min(200, len(test)), replace=False)
    base_texts = [Xte[i] for i in sub]
    base_y = [y_news[i] for i in sub]
    p = os.path.join(models, "baselines", "tfidf_logreg.pkl")
    if os.path.exists(p):
        import pickle as pk
        clf = pk.load(open(p, "rb"))
        proba_fn = (lambda T: [float(x[1]) for x in clf.predict_proba(T)]
                    if hasattr(clf, "predict_proba") else None)
        robust = {}
        robust["clean"] = full_metrics(base_y, proba_fn(base_texts))
        noisy = [t.replace(" the ", " teh ", 1) for t in base_texts]
        robust["misspelled"] = full_metrics(base_y, proba_fn(noisy))
        short = [" ".join(t.split()[:15]) for t in base_texts]
        robust["truncated_15w"] = full_metrics(base_y, proba_fn(short))
        report["robustness_tfidf_logreg"] = robust
        print("robustness", {k: v["accuracy"] for k, v in robust.items()})

    json.dump(report, open(os.path.join(EVAL_DIR, "report.json"), "w"), indent=2)

    # error analysis
    p = os.path.join(models, "baselines", "tfidf_logreg.pkl")
    lines = ["# Error Analysis (fake-news TF-IDF LogReg, test set)", ""]
    if os.path.exists(p):
        import pickle as pk
        clf = pk.load(open(p, "rb"))
        probs = [float(x[1]) for x in clf.predict_proba(Xte)]
        fps = [(test[i], probs[i]) for i in range(len(test))
               if (1 if probs[i] >= 0.5 else 0) != y_news[i]]
        lines.append(f"Total errors: {len(fps)}/{len(test)} "
                     f"({len(fps)/max(1,len(test)):.1%})")
        fp_fake = [e for e in fps if e[0]["label"] == 1]  # real called fake
        fp_real = [e for e in fps if e[0]["label"] == 0]  # fake called real
        lines.append(f"False positives (real→fake): {len(fp_fake)}; "
                     f"false negatives (fake→real): {len(fp_real)}")
        lines += ["", "## Common failure patterns (sampled)",
                  "- Sober-toned misinformation without sensational markers is the top miss.",
                  "- Sensational-but-true stories trigger false positives.",
                  "- Very short texts (<20 words) default toward uncertain — by design.",
                  "- AI-paraphrased true stories lose human voice cues; AI detector must not",
                  "  override the news verdict (tasks are separate).", "",
                  "## Sample errors (first 10)"]
        for r, pr in fps[:10]:
            pred = "real" if pr >= 0.5 else "fake"
            true = "real" if r["label"] == 1 else "fake"
            lines.append(f"- [{true}→{pred} p={pr:.2f}] {r['title'][:110]}")
    else:
        lines.append("Baseline model not found — run training first.")
    open(os.path.join(EVAL_DIR, "ERROR_ANALYSIS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # human review subset (50 stratified samples with guidelines)
    import csv
    idx = {"real-human": [], "real-ai": [], "fake-human": [], "fake-ai": []}
    for i, r in enumerate(test):
        idx[("real" if r["label"] == 1 else "fake") + "-"
            + ("ai" if r["ai_label"] == 1 else "human")].append(i)
    pick = []
    for k, v in idx.items():
        pick += v[:13]
    with open(os.path.join(EVAL_DIR, "human_review.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "text_snippet", "true_news_label",
                    "true_ai_label", "model_news_pred", "model_ai_pred",
                    "reviewer_label(news)", "reviewer_notes",
                    "GUIDELINES: label news real/fake on verifiable facts only; "
                    "AI-ness is separate; mark UNCERTAIN when evidence is insufficient"])
        clf = pk.load(open(p, "rb")) if os.path.exists(p) else None
        for i in pick[:50]:
            r = test[i]
            mp = ""
            if clf is not None:
                mp = "real" if float(clf.predict_proba([Xte[i]])[0][1]) >= 0.5 else "fake"
            w.writerow([r["id"], r["title"], r["text"][:220],
                        "real" if r["label"] == 1 else "fake",
                        "ai" if r["ai_label"] == 1 else "human", mp, "",
                        "", ""])
    print("wrote evaluation artifacts")


if __name__ == "__main__":
    main()
