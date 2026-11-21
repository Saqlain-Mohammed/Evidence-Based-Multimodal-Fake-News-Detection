# Implementation status and project objectives

The supplied project document proposes five analysis areas and three clients.
This record distinguishes code, completed training, measured performance and
features that still need work. It does not claim every proposed capability is
finished merely because an upload form or model architecture exists.

## Completed and measured

- Full responsive website redesign with forest-green visual identity, editorial
  headings, independent detector navigation, keyboard controls, file previews,
  honest empty/error/loading states, model readiness and dataset provenance.
- Shared local FastAPI API, SQLite history, report retrieval/deletion and JSON
  export. Document input supports TXT, DOCX and text-based PDF.
- Independently trained scratch WELFake news-pattern, HC3 AI-text and CIFAKE
  AI-image models. Held-out calibration and real test reports are included in
  `models/public/`. Original LIAR test measures cross-domain failure openly.
- CIFAKE model Grad-CAM attention, explicitly distinct from a tamper mask.
- Public URL text extraction, related Google News RSS reporting with TF-IDF
  relevance, publisher identity context and clear insufficient-evidence states.
- Reference-scoped five-word overlap. No fabricated internet-wide plagiarism
  percentage or AI-authorship certainty.
- Separate scratch image manipulation CNN, video CNN/GRU and audio spectrogram
  CNN, with protocol imports, source/identity/speaker grouping, hash-leakage
  rejection, validation selection, calibration and held-out evaluation.
- Chrome/Edge and Firefox development extension packages for selected/page text.
- Expo mobile client source for file selection, camera capture and report sharing.
- New automated API/model tests. Legacy synthetic models are excluded from serving.

## Additional data acquisition in progress

The user requested acquisition from the beginning rather than supplying datasets.
Resumable, byte-range-validated downloads are being acquired under `data/public/`:

- Full ASVspoof 2019 LA train/development/evaluation via the documented Bisher
  mirror, preserving official speaker IDs, labels and partition sizes.
- CASIA v2 image/mask archive via KhangTruong’s mirror. The mirror’s `cross_test`
  name is retained in provenance; using it to train requires a newly declared
  source-grouped internal split and cannot be reported as the MVSS cross-test
  protocol. Donor/host connections must remain in one partition.
- Official DeepfakeTIMIT archive from Idiap/Zenodo. Authentic VidTIMIT sequences
  must be paired with corresponding fake sequences; fake videos alone are not a
  binary training dataset. It is a small historical baseline, not a replacement
  for large modern FF++/DF40/Celeb-DF evaluation.

Provider sources:
[ASVspoof](https://www.asvspoof.org/index2019.html),
[ASVspoof mirror](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA),
[CASIA naming and corrections](https://github.com/namtpham/casia2groundtruth),
[CASIA mirror](https://huggingface.co/datasets/KhangTruong/Casia-v2.0),
[DeepfakeTIMIT](https://zenodo.org/records/4068245),
[VidTIMIT](https://conradsanderson.id.au/vidtimit/).

## Objectives still requiring completion or validation

- Reliable evidence-based factual support/contradiction: current related
  reporting is not factual entailment. No new NLI checkpoint is trained yet.
- Jointly trained multimodal truth fusion: current report uses an explainable
  review policy. There is no validated cross-modal attention fusion checkpoint.
- Remaining media checkpoints: architecture availability is not completed
  real-data training. The live registry is the authoritative readiness record.
- Manipulation subtype/localization, image context misuse, video tamper timestamps,
  speaker clone attribution and splice localization require suitable supervision
  and external evaluation; binary models cannot establish these claims.
- Audio transcription and transcript-to-news verification are not yet connected.
- Bulk analysis, authenticated user accounts and account-private cross-device
  history are not implemented. The local database is shared by all clients.
- Native mobile share-target/voice recording, signed APK/iOS builds and device
  verification; extension image/video context menus and store distribution.
- Modern and non-English generalization: HC3/CIFAKE are narrow older benchmarks;
  WELFake does not transfer reliably to LIAR. M4, GenImage, DF40 and real-world
  independent datasets are recommended expansion, not claimed completed training.

## Design improvements over the proposed score fusion

AI-written text can be factual and human text can be false. The backend therefore
keeps their probabilities separate. A missing model produces no percentage.
Related headlines are labelled unverified. A source URL is context, not proof.
The model lab shows held-out counts, class mappings, data limits and the weak LIAR
transfer result alongside high in-domain scores. These choices reduce misleading
confidence while preserving a unified report and clear human-review recommendation.
