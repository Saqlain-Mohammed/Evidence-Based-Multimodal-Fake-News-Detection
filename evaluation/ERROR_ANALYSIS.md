# Error Analysis (fake-news TF-IDF LogReg, test set)

Total errors: 8/502 (1.6%)
False positives (real→fake): 0; false negatives (fake→real): 8

## Common failure patterns (sampled)
- Sober-toned misinformation without sensational markers is the top miss.
- Sensational-but-true stories trigger false positives.
- Very short texts (<20 words) default toward uncertain — by design.
- AI-paraphrased true stories lose human voice cues; AI detector must not
  override the news verdict (tasks are separate).

## Sample errors (first 10)
- [fake→real p=0.65] Star player secretly traded, cover-up claimed
- [fake→real p=0.67] Star player secretly traded, cover-up claimed
- [fake→real p=0.51] Water supply rumor sparks panic buying
- [fake→real p=0.82] Water supply rumor sparks panic buying
- [fake→real p=0.53] Water supply rumor sparks panic buying (developing)
- [fake→real p=0.56] Water supply rumor sparks panic buying
- [fake→real p=0.58] Water supply rumor sparks panic buying
- [fake→real p=0.56] Water supply rumor sparks panic buying (developing)
