# AUDIT — Existing Fake-News + AI-Content Pipeline (Sept 2026)

Method: full code read of `backend/`, `backend/modules/`, `training/`, `frontend/`, artifacts + live probing.
No accuracy numbers were changed; all findings are evidence-backed.

## A. Critical prediction/inference bugs (accuracy-facing)

1. **Fusion probability inverted (CRITICAL).** `backend/modules/trained_loader.py:fusion_fake_prob`
   returns `predict_proba(...)[0][1]`. The fusion MLP is trained with labels
   `0=fake, 1=real` (`training/train_fusion.py:141`), so index 1 is **P(real)**,
   not P(fake). Live probe: fake-like features `[0.9,0.9,0.8,0.1,0.8,0.9]` →
   `proba=[0.95, 0.05], pred=0 (fake)` yet the API would report
   `fake_probability=0.05 → LIKELY REAL`. Fake inputs are scored as real and
   vice-versa whenever the head is confident. **Fix: return P(class 0).**
2. **Fusion head collapsed to one class.** Same probe: both fake-like AND
   real-like feature vectors predict class 0 with ~0.95+ confidence. A head that
   predicts one class cannot have 1.0 val/test accuracy on balanced data —
   the reported `metrics.json` (1.0 everywhere) is therefore **not the
   performance of this artifact on these features** (stale/mismatched metrics or
   train/eval feature mismatch). Proxies differ between train (`evidence_match`
   = title↔body TF-IDF, `contradiction` = negation heuristic) and inference
   (live retrieval/NLI), so the head operates out-of-distribution in production.
3. **Trivially separable demo data → fake 1.0 accuracy.** `prepare_fakeddit.py`
   demo mode: 8 repeating titles, 2 repeating bodies, images = red-diagonal
   (fake) vs blue-grid (real). Text and image are each ~100% separable by one
   keyword/pixel rule. Any model gets 1.0 without learning anything real.
   `metrics.json` 1.0/1.0/1.0 on all splits + cross-dataset "1.0" on the
   synthetic FakeNewsNet split proves leakage-by-construction (same 8 templates
   in train/val/test/cross), not generalization.
4. **Train/val/test leakage by construction.** Demo rows cycle
   `REAL_TITLES[(start+i) % 8]` with identical bodies; splits are contiguous
   slices of the same cycle, so every val/test sample is a near-exact duplicate
   of many train samples. No dedup, no group-aware split.
5. **Label semantics:** text model `model_info.json` uses `{"0":"fake","1":"real"}`
   while some scripts/docs describe "1=true". Consistent internally, but the
   fusion inversion (bug 1) flips it in production. No label-noise audit exists.
6. **AI-text decisive rule is a hard-coded override.** `ai_content.py:129`: if
   `stock_tells >= 8` force `p_ai >= 0.66`. This is a hand-written floor on top
   of the model, not a learned behavior — it inflates AI-detection "accuracy" on
   purple-prose probes while hiding the weak underlying detector (GPT-2-era
   RoBERTa detector gives P(AI)=0.53 on blatant LLM prose). Documented, but it
   is exactly the kind of artificial boost the spec forbids as an evaluation
   claim. Kept as a *flagged heuristic indicator*, removed from the trained
   AI-text model's score path (new `train_ai_text.py` has no such rule).
7. **Confidence scores are not calibrated.** Text `0.55+0.4*min(1,n/300)`,
   AI `0.55+0.35*...`, image resolution-based, fusion logistic sharpening —
   all hand-set, none fit on held-out data. A 95% "confidence" has no measured
   meaning. New pipeline fits a calibration note + caps confidence when
   evidence is insufficient.
8. **No "Insufficient Evidence" state.** Old `label_for` forces REAL/UNCERTAIN/
   FAKE on every input, even empty text + no image + no evidence. New pipeline
   adds explicit `INSUFFICIENT EVIDENCE / NEEDS HUMAN VERIFICATION`.

## B. Dataset problems

- 600/150/150 synthetic rows, 8 templates, 2 bodies, 1 source ("demo"),
  1 time period, no topics/publishers/styles/lengths variation.
- Images: solid-color + geometric cue; no manipulation/context-mismatch/AI-image
  distinctions; every edited-looking image = fake by construction.
- No AI-text / AI-image training datasets at all (only heuristics at inference).
- No multimodal consistency labels (image-caption mismatch untested).
- No missing-value/corruption/conflicting-label handling; no class-imbalance
  handling (overtly balanced, hiding real-world skew).
- Real-data path (`--real`) exists but was never run here; label-column
  auto-detection unverified against actual Fakeddit release.

## C. Preprocessing / feature mismatches (train vs production)

- Text train tokenizes `title+text [:2000]`, max_len 128 (demo: 64); inference
  `text_fake_prob` uses max_len 128 — matches, OK. But `bert_features`
  (512-token BERT or hashed fallback) is a *different* representation never used
  in training; the explanation panel cites signals the classifier never saw.
- Image train uses Resize256→CenterCrop224 + ImageNet norm; inference
  `image_fake_prob` identical — OK. But `analyze_image` risk heuristics
  (sharpness/resolution/contrast) are blended 35% with the tuned model with no
  validation of the blend weight.
- Fusion train features 3–6 are offline proxies; inference substitutes live
  values with different scales/semantics (evidence_match train = title↔body
  similarity; inference = retrieval similarity). Distribution shift, unmeasured.

## D. Model selection / training process

- Single config each (DistilBERT 1 epoch; ResNet50 frozen 2 epochs; MLP(32,)):
  no baselines (TF-IDF+LogReg/SVM), no hyperparameter search on val, no early
  stopping discipline (saves on `>=` so last-epoch ties win), no class weights,
  no LR schedule, no seeds reported per-run (fixed 42, fine), no cross-val.
- `train_text.py --offline` flag exists but `run_all.py --demo` never passes it;
  demo fine-tune silently downloads DistilBERT if uncached (non-reproducible
  offline). New runner passes `--offline` with graceful TF-IDF fallback.
- Image training silently drops rows without image files; demo always has them,
  real mode may train on a biased subset without warning beyond a print.

## E. Evaluation methodology

- `common.prf` reports accuracy/precision/recall/F1 (binary-positive=1=real)
  + confusion matrix only. Missing: macro F1, ROC-AUC, PR-AUC, per-class P/R/F1,
  FPR/FNR, calibration, cross-val, robustness, error analysis.
- `e2e test_all.py` (24 checks) asserts *behaviors* (label present, separation
  direction) not metric thresholds — good — and keeps passing after fixes.
- No leakage checks (duplicates, near-dups, story/source overlap, filename
  leakage), no human-review subset, no failure-pattern analysis.

## F. API/backend integration

- `/api/analyze` (fake-news only) and `/api/analyze_ai` (AI only); no combined
  endpoint producing the required structured CONTENT ANALYSIS. Added:
  `/api/analyze_full`.
- Chart endpoints (`/api/explore`, `/api/stats`) return live values — OK — but
  the Enter-Content card had no chart section; added with live-only data.
- `fusion_fake_prob` inversion (bug 1) affects every `/api/analyze` call when
  artifacts exist (i.e. always in this deployment).

## G. Frontend

- Verdict/gauge/breakdown/why-box render live values — OK. Missing (added):
  in-card Chart-Type section bound to the last result (no static demo data),
  structured multi-task result block, Insufficient-Evidence banner, "Needs Human
  Verification" recommendation, per-modality contribution chart from the real
  `contributions` object.

## H. What this rebuild changes (traceability)

| Audit item | Fix location |
|---|---|
| Fusion inversion + collapse | `backend/modules/trained_loader.py`, `training/train_fusion.py`, `evaluation/` |
| Trivial data + leakage | `data/` + `preprocessing/` + `training/prepare_realistic.py` (group-aware splits, dedup, leakage report) |
| No baselines / no selection | `training/train_baselines.py`, `training/train_ai_text.py`, validation-based selection |
| Thin metrics | `evaluation/run_eval.py` (acc, prec, rec, F1, macro F1, ROC/PR-AUC, CM, per-class, FPR/FNR, CV, calibration, robustness) |
| No human review | `evaluation/human_review.csv` + guidelines |
| No uncertainty state | backend `label_for` + `/api/analyze_full` + frontend banner |
| Uncalibrated confidence | capped/evidence-conditioned confidence, documented as assistive |
| Missing combined task view | `/api/analyze_full` + frontend structured block + charts from live outputs |
| Hard-coded AI floor in eval path | removed from trained model path; kept only as labeled heuristic indicator |
| Reproducibility | `requirements.txt`, seeds, `models/VERSIONS.md`, `README` rerun instructions |
