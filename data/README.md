# Data (v2, Sep-2026)

`data/realistic/` — offline leakage-safe benchmark (3,355 rows after
exact+near-dup removal), built by `training/prepare_realistic.py`:

- **160 unique story groups** (no shared headlines/bodies across groups),
  8 topics, 12 sources assigned **independent of label** (no source leakage).
- **Labels:** news `0=fake / 1=real` + AI `0=human / 1=ai` → the required
  human+true / human+false / AI+true / AI+false 2x2 (test: 249 real / 253 fake,
  258 AI / 244 human).
- **Hard cases by design:** 45% of fakes reuse true-style reporting bodies with
  an unverifiable claim injected; 50% of generated fakes carry plausible
  headlines; real articles include rumor-denial language and clickbait
  headlines — keyword rules fail, models must weigh claim-level cues.
- **Splits:** group-aware 70/15/15 (train 2354 / val 499 / test 502),
  stratified; group overlap train/val and train/test = **0**
  (see `leakage_report.json`).
- **Images:** `images/img_*.png` (160px, hue randomized independent of label;
  subtle splice/re-encode cues in ~70% of fakes, ~15% of reals).
- **Label definitions:** fake = factually false or unverifiable central claim;
  real = verifiable against the story's public record. AI = machine-styled
  paraphrase (stock connectors, uniform rhythm), human = personal voice/rhythm
  variation. Documented overlap: humanized AI and formal humans exist in both
  classes — hence AI-text FNR 0.27, reported honestly.

**Real datasets:** `training/prepare_fakeddit.py --real` and
`prepare_fakenewsnet.py --real` remain the path for official releases
(Fakeddit 1M+, FakeNewsNet 23K, NewsCLIPpings, FineFake, VERITE); v2 synthetic
is the offline stand-in, and every metric is labeled as such — never presented
as real-world performance.
