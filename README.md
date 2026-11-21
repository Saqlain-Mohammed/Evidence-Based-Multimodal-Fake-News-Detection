# Veritas content verification workspace

A redesigned research application for examining news credibility, AI writing,
image generation, image manipulation, video deepfakes and synthetic audio.
Independent detectors share a FastAPI backend, a responsive website and a local
SQLite report history. No cloud model API is required.

## Run locally

The included checkpoints were created with Python 3.14, PyTorch 2.14 CPU and
scikit-learn 1.9.1. Use the recorded environment in `requirements-lock.txt` to
reproduce their inference. For a fresh environment, install `requirements.txt`.
Python package and Torch wheel availability varies by Python version.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001>. Interactive API documentation is at `/docs`.
`start.ps1` is an alternative launcher, using port 8000 by default.
This is a local research service without authentication. Do not expose it to the
public internet. Reports are shared by clients of the same server, not private
accounts. Large raw datasets, uploaded files, report history and the virtual
environment are intentionally excluded from Git.

## What is actually trained

Independent real-data checkpoints are included. Their
parameters/features are learned from scratch: no pretrained BERT, ImageNet
weights, or hosted LLM inference is used by this workspace.

- **News credibility:** word and character TF-IDF with logistic regression,
  sigmoid-calibrated on held-out validation data. 60,920 cleaned WELFake
  examples; 48,736 train, 6,090 validation, 6,094 test. Held-out accuracy
  **98.88%**, macro F1 **0.9886**. This learns article patterns, not facts.
- **AI text:** a separate TF-IDF/logistic model, calibrated independently.
  75,185 HC3 English answers; 60,069 train, 7,526 validation, 7,590 test.
  Held-out accuracy **99.30%**, macro F1 **0.9924**. All answers to a question
  remain in the same split. The historical ChatGPT benchmark is not a guarantee
  about current generators, edited prose or student authorship.
- **AI image:** a randomly initialized three-block CNN trained for five epochs
  on CIFAKE. 88,831 training images and 9,871 validation images after pixel
  deduplication and removal of training/test collisions; original 20,000-image
  test retained. Accuracy **91.80%**, macro F1 **0.9179**. Positive-class Grad-CAM
  shows model attention, not a proven tamper mask.

**The cross-domain result is much weaker:** the WELFake model scores **55.69%**
accuracy and **0.4191** macro F1 on 1,002 LIAR official-test statements after the
documented binary mapping. This is shown in the Model Lab. Do not present the
WELFake score as general fact-checking accuracy. The API abstains from scoring
news passages shorter than 80 words. AI text requires at least 20 words.

- **Deepfake video:** the from-scratch CNN/GRU scored **50%** (5/10) on a source-disjoint UADFV test. This is an experimental clip score with insufficient evidence of reliability.

The model registry reads actual checkpoints and `report.json` files. The image
manipulation, video and audio architectures and training pipelines are separate;
their status stays **Training required** until a real-data checkpoint exists.
Ongoing additional acquisitions/training are documented in `IMPLEMENTATION.md`.
Legacy synthetic models in ignored local folders are never loaded by the new API.

## Reproduce public-data training

Downloads use provider releases or documented mirrors, never execute dataset
Python scripts, and record source URLs, byte counts and SHA-256 digests.
All data and caches stay on the project drive.

```powershell
.\.venv\Scripts\python.exe training/download_public.py welfake hc3 cifake liar
.\.venv\Scripts\python.exe training/train_public.py --task all --epochs 5
```

Sources and label mappings:

- [WELFake official release](https://zenodo.org/records/4561253): the release CSV uses
  `0=real, 1=fake`. The record description reverses those labels. Its 35,028 real and 37,106 fake article totals match the file with this correction. In the CSV, 21,127 of 21,129 articles with an opening Reuters dateline have label 0. Empty/short rows and exact body
  duplicates are removed. Article-prefix groups are split before feature fitting.
  Paraphrased near duplicates and publisher/topic shortcuts can remain.
- [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3): human=0, ChatGPT=1;
  duplicate answers are removed; the question determines the split. CC BY-SA 4.0
  and any stricter underlying corpus terms apply; cite Guo et al. (2023).
- [CIFAKE](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images):
  120K 32×32 CIFAR-10/Stable Diffusion 1.4 images. The
  [HF mirror](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset)
  preserves official splits and `0=FAKE, 1=REAL`, remapped internally. Cite Bird
  and Lotfi (2024) and the original CIFAR-10 work. This is not a modern-generator
  or manipulation benchmark.
- [LIAR original paper and dataset](https://huggingface.co/datasets/ucsbnlp/liar):
  official test is used only for transfer evaluation, never calibration.
  False/barely-true/pants-fire are risk; true/mostly-true are non-risk; half-true
  is excluded rather than relabelled as entirely true or false.

For large archives behind buffering proxies, use the resumable range downloader:

```powershell
.\.venv\Scripts\python.exe training/ranged_download.py asvspoof
.\.venv\Scripts\python.exe training/ranged_download.py casia
.\.venv\Scripts\python.exe training/ranged_download.py deepfaketimit
```

## Train the remaining independent media models

`training/scratch_models.py` defines an image manipulation CNN, a frame-CNN plus
bidirectional GRU for video, and a spectrogram CNN for audio. Their checkpoints
are independent of the AI-image model. A scratch text Transformer is also
provided as an experimental architecture; the measured text checkpoints use the
CPU baselines described above.

Prepare a provider protocol CSV with `path,label,group,split`. Labels are
0=authentic, 1=risk. `group` must identify the original image and derivatives,
video source/identity, or audio speaker. Preserve official splits when available.
Never split a video’s frames into train and test.

```powershell
.\.venv\Scripts\python.exe training/prepare_media.py --csv data/provider.csv --out data/prepared/audio
.\.venv\Scripts\python.exe training/train_media.py --task audio --data data/prepared/audio --dataset-name "ASVspoof 2019 LA" --epochs 15 --batch 32
```

Choose `--task image_manipulation`, `video` or `audio` with the corresponding
manifest directory. The importer rejects cross-split source/hash overlap.
Training selects checkpoints on validation F1, calibrates on validation only,
then evaluates the test set; audio additionally reports an approximate ROC EER.
Only create a checkpoint after a completed evaluation. Binary labels do not
justify claims about cloning attribution, editing subtype or tamper timestamps.

## Website and clients

The website has six detector tabs, drag/drop, TXT/DOCX/PDF extraction, image
previews, per-model signals, Grad-CAM attention, online related reporting,
reference-scoped overlap, history, JSON report export, a Model Lab, dataset
sources, responsive phone navigation and a web-app manifest.

`extension/` contains a Manifest V3 Chrome/Edge client for selected text and page
text. `python scripts/package_extension.py` creates Chromium and Firefox ZIPs;
Firefox uses its background-script manifest. Load unpacked in a development
browser. It only sends content when Analyze is clicked and defaults to local
inference with online evidence disabled. The extension has not been installed or
published to extension stores. Image/video context menus are not implemented yet.

`mobile/` contains an Expo React Native client with file selection, photo capture,
detector modes and report sharing. Run `npm install`, then `npx expo install --fix`
and `npm start` there to align the SDK’s native dependencies. Android emulator
uses `http://10.0.2.2:8001`. Native Android/iOS builds, inbound native share-sheet
integration and voice recording still need platform testing and implementation.
Camera permission is requested only on the user’s capture action.

## API and verification

- `GET /api/health` — current readiness; never loads synthetic heads.
- `GET /api/v2/models` — checkpoint readiness, measured metrics, transfer results.
- `GET /api/v2/datasets` — corpus catalog and access notes.
- `POST /api/v2/analyze` — multipart `mode`, `text`, `url`, `file`,
  `reference_text`, `live_evidence`.
- `GET /api/v2/history`, `GET /api/v2/reports/{id}`,
  `DELETE /api/v2/reports/{id}` — shared local report storage.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests cover task separation, missing-checkpoint abstention, short input guards,
report history, reference-overlap scope, private URL rejection, upload validation,
network-independent model inference, real image attention and media split leakage.

## Changes from the project proposal

The supplied report is treated as a requirements reference. Its declarations of
completion and proposed performance are not taken as evidence of implemented
features. `IMPLEMENTATION.md` tracks the objectives and remaining gaps.

The implementation keeps FastAPI rather than changing a working backend to
Flask. Scratch CPU baselines establish reproducible measured results before
expensive Transformer/ViT training. AI generation and truth are separate tasks.
Related headlines are not labelled as support/contradiction by keyword overlap;
the old lexical entailment fallback is excluded. HTTPS and publisher reputation
are not assigned arbitrary factual credibility percentages.

The overall report uses an explicit review policy, not an average of incompatible
probabilities. A learned multimodal truth-fusion model needs jointly labelled
examples and missing-modality validation; proxy/synthetic labels would create
misleading confidence. A validated NLI model, primary-source retrieval, modern
generator benchmarks and source-disjoint external tests remain necessary for a
stronger real-world system.
