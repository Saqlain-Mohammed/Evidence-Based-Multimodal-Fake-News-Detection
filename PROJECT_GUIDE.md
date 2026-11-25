# Veritas content verification workspace

A redesigned research application for examining news credibility, AI writing,
image generation, image manipulation, video deepfakes and synthetic audio.
Independent detectors share a FastAPI backend, a responsive website and a local
SQLite report history. No cloud model API is required.

## Run locally

On Windows, double-click `Start-Veritas.cmd` in the project folder. It starts
the local server and opens the redesigned Verification Studio, or opens the
studio if it is already running. Keep the server window open during use.

The included checkpoints were created with Python 3.14, PyTorch 2.14 CPU and
scikit-learn 1.9.1. Use the recorded environment in `requirements-lock.txt` to
reproduce their inference. For a fresh environment, install `requirements.txt`.
Python package and Torch wheel availability varies by Python version.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is at `/docs`.
`start.ps1` is an alternative launcher, using port 8000 by default.
The default loopback service shares local history. For per-account access, set
`VERITAS_ACCOUNTS_FILE` to a private SQLite path outside Git, provision each named user
with `python backend/accounts.py USERNAME`, and start the server in the same environment.
Each command prints a new bearer token once. Enter it in the website, mobile client
or extension. Use HTTPS when serving other devices. Account mode isolates report
listing, reading and deletion. The inference queue allows two concurrent analyses. Large raw datasets, uploaded files, report history and the virtual
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

- **Video frame deletion:** trained and evaluated on 152 clips from 25 CSVTED scenes, keeping related clips together. Among 26 held-out clips, the default threshold labelled all 26 edited (21 edited, five originals), matching majority accuracy at 80.77% with macro F1 of 0.447. Of 27 candidate timestamps, at least one fell within two frames of the deletion in 17 of the 21 edited clips. Use the timestamps for manual inspection.

- **Title and image together:** trained on 4,693 deduplicated Fakeddit pairs. On 1,429 test posts, joint macro F1 was 0.578 versus 0.555 for title alone and 0.445 for image alone. The binary target groups satire and other community-labelled categories together. Majority-class accuracy was 80.2%, versus 81.2% for the joint model.
- **Evidence comparison:** trained on 392,702 MultiNLI pairs. Accuracy was 59.66% on matched development and 60.84% on mismatched development. The app compares the opening claim with a selected reference sentence or news headline and exposes uncertainty.
- **Audio:** a from-scratch spectral gradient-boosting classifier used all 25,380 ASVspoof LA training utterances and 24,844 official development utterances. Official evaluation on 71,237 recordings yielded 82.61% accuracy, 0.690 macro F1, 0.889 ROC AUC and 18.48% equal-error rate. Most evaluation recordings are spoofs, so accuracy alone is insufficient.
- **Partial speech editing:** trained on 11,137 recordings and calibrated on 5,096 development recordings from the PartialSpoof v1.2 subset. On 14,710 evaluation recordings, accuracy was **76.91%**, clip macro F1 **0.759** and edited-segment F1 **0.567** over 320,864 windows. The output lists candidate intervals in seconds.
- **Voice conversion:** a separate classifier scored 84.57% accuracy and 0.760 macro F1 on 56,495 official evaluation utterances after excluding hybrid TTS/VC attacks. It detected 7,033 of 14,742 human-source conversions. It cannot identify the target speaker.
- **Image manipulation and masks:** source-connected CASIA v2 partitions contain 6,967 training, 816 validation and 4,581 test images. The forensic classifier scored 78.63% accuracy and 0.768 macro F1, up from 53.22% accuracy for the CNN. Model selection used validation macro F1 (0.834 versus 0.725). The independent pixel model achieved 4.66% mean tampered-image IoU. The interface labels the localization experimental.

The Model Lab reads each installed detector's held-out metrics. Each checkpoint folder also contains its machine-readable report. Legacy synthetic weights are excluded from inference.

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

## Reproduce joint modeling, speech and media training

Run the following commands from the repository root using the project Python environment.

```powershell
python training/ranged_download.py multinli fakeddit_pairs
python training/train_stance.py
python training/train_paired.py
python training/install_transcription.py
.\training\resume_media.ps1
```

The optional English transcription component uses Vosk's pretrained small English model, downloaded by the installer. It processes up to two minutes locally and passes the transcript through the news detector's ordinary length check. The independent spoof detector examines the first six seconds.

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

The website has six detector tabs, multi-file drag/drop with an individual report for each file, TXT/DOCX/PDF extraction, image
previews, per-model signals, Grad-CAM attention, Google News headlines and FactCheck.org excerpts, CRED-1 publisher context,
reference-scoped overlap, history, JSON report export, a Model Lab, dataset
sources, responsive phone navigation and a web-app manifest.

`extension/` contains Chrome/Edge and Firefox clients with context menus for selected text, images and videos, plus whole-page capture. Media from sites that restrict cross-origin reads can be supplied as local files. Build the two extension ZIPs using `python scripts/package_extension.py`.

`mobile/` contains an Expo client with camera input, file selection, microphone recording, inbound sharing and report export. Install the locked dependencies with `npm ci`. Use `npm run android` or `npm run ios` in a configured native development environment. The Android emulator defaults to `http://10.0.2.2:8000`. The Android development APK includes ARM64 and x86-64 libraries. An x86-64 emulator verified inbound text sharing, analysis and report creation against the local backend. To reproduce it with the Android SDK and JDK installed, run `npx expo prebuild --platform android` from `mobile`, then `./gradlew assembleRelease "-PreactNativeArchitectures=arm64-v8a,x86_64"` from `mobile/android` (use `gradlew.bat` on Windows). The APK uses a development signing key. The iOS JavaScript bundle was exported. Native iOS compilation, store signing and physical-device testing remain open.

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

Optional source reputation context is provided by Alexander Loth’s CRED-1 compact dataset under CC BY 4.0. The pinned release, original license, and source attributions are in `backend/resources/cred1/`. A match describes a domain rather than an individual article.

The implementation keeps FastAPI rather than changing a working backend to
Flask. Scratch CPU baselines establish reproducible measured results before
expensive Transformer/ViT training. AI generation and truth are separate tasks.
Related headlines are not labelled as support/contradiction by keyword overlap;
the old lexical entailment fallback is excluded. HTTPS and publisher reputation
are not assigned arbitrary factual credibility percentages.

The overall report uses an explicit review policy, not an average of incompatible
probabilities. The separate joint model requires both an image and a title and uses actual paired examples. Its target is a historical Reddit label. A validated NLI model, primary-source retrieval, modern
generator benchmarks and source-disjoint external tests remain necessary for a
stronger real-world system.
