# Models (v2, Sep-2026) — leakage-free retrain

| Model | File | Train | Val | Test (untouched) |
|---|---|---|---|---|
| Fake-news TF-IDF+LogReg (balanced) | `baselines/tfidf_logreg.pkl` | 2354 | 499 | acc **0.984**, macro F1 **0.984**, ROC-AUC **1.00**, PR-AUC **1.00**, FPR 0.032, FNR 0.000 |
| Fake-news TF-IDF+LinearSVM (balanced) | `baselines/tfidf_svm.pkl` | 2354 | 499 | acc 0.980, macro F1 0.980 (val 0.992 — selected **against**: no calibrated probabilities) |
| AI-text TF-IDF+LogReg (human vs AI) | `ai_text.pkl` | 2354 | 499 | acc **0.861**, macro F1 **0.859**, ROC-AUC **0.898**, PR-AUC **0.929**, FNR 0.271 (humanized AI is the hard case) |
| Image ResNet50 (frozen backbone, trained head, class-weighted, augmented) | `image_resnet.pt` | 600 | 499 | val acc **0.784**, F1 **0.818** (test eval via app metrics on request) |
| Fusion MLP(32) v2 (6 modality features, fixed P(fake)=P(class 0)) | `../backend/artifacts/fusion.pkl` | — | — | val macro F1 **0.988** (LogReg candidate 0.982) |

**Selection rule:** validation macro-F1 only; test touched once by `evaluation/run_eval.py`.
LogReg chosen over SVM for the live path (calibrated probabilities + speed),
documented in `evaluation/report.json`.

**Legacy v1 artifacts** (demo 8-template data, trivially separable, fusion
probability inverted) are superseded: `backend/artifacts/fusion.pkl.v1bak`,
`backend/artifacts/text_model/`, `backend/artifacts/image_model.pt` remain on
disk for reproducibility but the live path prefers v2 models.
