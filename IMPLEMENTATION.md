# Project objectives and validation

## Verification workflow

The website, extension and mobile source use the same FastAPI analysis and report endpoints. The website supports files, text, article URLs, batch uploads, report export and history. Local mode shares history. Token mode restricts report access to the issuing account. Two analyses can execute concurrently per server process.

The detectors answer separate questions. WELFake models article patterns, HC3 models writing origin, and CIFAKE models historical image generation. CASIA supplies manipulation labels and pixel annotations. UADFV supplies paired original and deepfake clips. ASVspoof 2019 LA supplies bona fide and synthesized or converted speech. Two separate classifiers estimate general spoof risk and human-source voice conversion. A third detector uses PartialSpoof segment annotations to locate inserted synthetic speech in 160 ms windows. Each trained detector and its evaluation report are in `models/public/`.

Text and image inputs provided together invoke a separate Fakeddit classifier. Its macro F1 exceeded title-only and image-only comparisons on the mirror's deduplicated split. It predicts a community-derived benchmark label that includes satire. The final review policy preserves the individual scores instead of interpreting them all as probabilities of false news.

## Evidence and explanations

The system retrieves Google News headlines and FactCheck.org excerpts. A scratch MultiNLI classifier compares the opening claim with retrieved or supplied passages. The interface exposes the source, passage, experimental relation and retrieval limits. CRED-1 supplies attributed publisher-level context for matching domains. Five-word overlap is measured against the user's reference. An English Vosk model can transcribe the first two minutes of audio locally and pass the transcript to the news detector. Vosk is an auxiliary pretrained component.

The AI-image CNN provides Grad-CAM attention. The CASIA mask model uses actual pixel annotations and reports mean intersection over union. Its low overlap limits the conclusions that can be drawn from highlighted regions. The video CNN/GRU uses temporal information across sampled frames and exposes their timestamps. Its ten-clip held-out test reached chance accuracy. A competing temporal residual classifier performed worse on validation and was not selected. A separate CSVTED frame-deletion detector found a candidate within two frames of the annotated deletion in 17 of 21 edited test clips. Its default clip threshold also flagged all five originals. The application labels these signals experimental.

## Client validation

The web workflow has been exercised on desktop and mobile-width layouts, including two-file submission, opening each report and the model registry. The Edge extension was loaded in a temporary profile and submitted a real analysis with a report link. Chrome/Edge and Firefox packages support selection, whole-page, image and video inputs. The Expo app supports camera, file inputs, microphone recordings, incoming shares, expandable evidence and report sharing. An Android release build, signed with a development key and containing ARM64 and x86-64 libraries, was installed on an x86-64 emulator. The inbound text share reached the form and produced a real server-side report. The iOS JavaScript bundle was exported. Native iOS and physical-device tests remain open.

## Open validation and research work

- Reliable real-world factual adjudication and a jointly labelled four-input text, image, publisher and evidence model. The paired model handles title and image.
- Corpus-wide plagiarism and paraphrase detection. The implemented comparison is reference-scoped.
- Reliable localization and manipulation subtypes across external image datasets, independently validated video tamper intervals, generalization to other audio editing techniques and speaker-specific clone attribution.
- Larger independent video and contemporary generator benchmarks, non-English validation, and external assessments of robustness to recompression and editing.
- Signed native distribution, iOS compilation on macOS and full device/browser compatibility testing.

## Reproduction

Use the acquisition and training commands in the README. Media importers retain source groups, reject cross-split byte collisions and preserve the official ASVspoof partitions. Raw corpora, user submissions, local credentials and report history stay outside the repository. The model reports contain split sizes, metrics and protocol-specific caveats.
