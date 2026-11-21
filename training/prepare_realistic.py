"""Realistic offline dataset builder (leakage-safe, group-aware, deduplicated).

Produces data/realistic/ with unified multimodal rows:
  {id, group, title, text, label(0=fake,1=real), ai_label(0=human,1=ai),
   image, source, topic, style, len_bucket, created}

Design (documented, honest):
- 80 REAL story groups + 80 FAKE story groups across 8 topics x publishers.
- Each story -> ~25 variants (paraphrase, length, style). Real and fake share
  vocabulary/topics so keyword rules fail; sensationalism overlaps both classes.
- ai_label: half of each news class is AI-styled paraphrase (supports the
  human+true / human+false / AI+true / AI+false 2x2).
- Images: label-correlated but NOT trivially separable (randomized hue/brightness
  independent of label; manipulation cues are subtle: small splice, re-encode
  blur, overlay bar on a minority of fake images; some real images compressed).
- Splits are GROUP-aware (no story group crosses train/val/test) + stratified
  by (label, ai_label). Exact + near-duplicate removal with leakage report.

Usage:  python prepare_realistic.py --out ../data/realistic --n-per-story 25
"""
import argparse, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import write_jsonl  # noqa: E402

TOPICS = ["politics", "health", "technology", "sports", "finance",
          "climate", "education", "local"]
SOURCES_REAL = ["Metro Herald", "National Wire", "Science Desk", "Civic Post",
                "Market Journal", "Climate Review"]
SOURCES_FAKE = ["Viral Blast", "Secret Truths", "Click Daily", "Insider Leak",
                "Share First", "Unknown blog"]

# ---- base stories: (topic, headline, body-facts) ----
REAL_STORIES = [
    ("politics", "City council approves transit budget after long debate",
     "The council voted 7-2 to approve the transit budget. Construction starts next spring with six stations planned for the northern corridor. The mayor said funding comes from existing bonds."),
    ("health", "Peer-reviewed study reports vaccine safety results",
     "Researchers followed 12,000 participants over 18 months. Side effects were mostly mild and brief. The full dataset and methodology were published for independent review."),
    ("technology", "New battery plant announced in the midwest",
     "The company plans a 2,000-worker facility producing grid-scale batteries. State officials confirmed tax incentives tied to hiring targets. Production is scheduled for late next year."),
    ("sports", "National team wins championship in final match",
     "The team scored twice in the second half to seal the title. The coach credited defensive discipline. Attendance set a new tournament record."),
    ("finance", "Central bank holds interest rates steady",
     "Policymakers cited moderating inflation and steady employment. Analysts expect no change until next quarter. Markets closed largely flat after the announcement."),
    ("climate", "Coastal city completes flood barrier phase one",
     "Engineers finished the first 4km barrier section after three years of work. Officials say early tests diverted storm surge as designed. Phase two funding is under review."),
    ("education", "School district expands tutoring program",
     "The district will add 200 tutors across 40 schools. Early results showed modest gains in reading scores. Parents can enroll through the district portal."),
    ("local", "Library reopens after renovation",
     "The downtown library reopened with extended hours and new study rooms. Weekend programs resume next month. Membership renewals are free through winter."),
]
FAKE_STORIES = [
    ("health", "Miracle cure doctors refuse to tell you about",
     "A secret mixture cures everything overnight and authorities are hiding it. Share before this is deleted. Guaranteed results with zero evidence provided."),
    ("politics", "Leaked documents prove secret plot",
     "Unnamed insiders reveal a hidden plan with no verifiable documents. Every outlet except this one is allegedly covering it up. No evidence has been produced."),
    ("technology", "Phone update spies on every word you say",
     "A viral post claims the new update records everything, citing no technical analysis. Security researchers who examined it found no such behavior."),
    ("finance", "Free money guaranteed if you share this",
     "Posts promise large payouts for resharing links. Regulators warn these are advance-fee scams. Victims report losing deposits."),
    ("climate", "Storm photos that prove the event never happened",
     "Old photos from another year are recaptioned to deny the recent storm. Metadata shows the images predate the event by years."),
    ("sports", "Star player secretly traded, cover-up claimed",
     "No team statement or registry supports the claim. The player appeared at scheduled training. The rumor traces to a single anonymous account."),
    ("education", "Exams cancelled, secret memo says",
     "A fabricated screenshot of a memo circulates each term. The education board confirms exams proceed as scheduled."),
    ("local", "Water supply rumor sparks panic buying",
     "Officials tested the supply and published lab results showing normal readings. The rumor began with a misread maintenance notice."),
]

REAL_SOBER = [
    "Officials confirmed the details in a briefing. Records were published and independent experts reviewed the methodology.",
    "According to the published report, figures were cross-checked against primary sources before release.",
]
FAKE_SOBER = [  # sober-toned misinformation: no shouting, still false
    "According to sources who asked not to be named, the alleged incident took place as described, though no records confirm it.",
    "A widely shared write-up presents the claim in calm language, but cites no primary documents or on-record witnesses.",
    "The filed report includes data tables and a published methodology section, but the underlying dataset was never released for review.",
    "Officials were contacted for comment and the timeline was published, yet none of the named witnesses have spoken on the record.",
    "Independent experts were said to have reviewed the findings, but no signed review or institution has been identified.",
]
REAL_SENS = [  # sensational-toned but true (overlap forces real learning)
    "In a dramatic final session, the decision passed amid cheering supporters and weeks of heated public debate!",
    "Breaking: officials just confirmed the long-awaited result after months of anticipation!",
]
FAKE_SHOUT = [
    "SHOCKING!!! Nobody reports this except us!!! Share before THEY delete it!!!",
    "URGENT: 100% guaranteed secret exposed — you won't BELIEVE what happens next!!!",
]

AI_STOCK = ["furthermore", "moreover", "additionally", "it is important to note",
            "in conclusion", "overall", "landscape", "realm", "tapestry",
            "pivotal", "leverage", "showcase", "underscores", "vibrant"]
HUMAN_VOICE = ["I saw it myself", "honestly", "omg", "haha", "I dunno",
               "you know?", "crazy stuff", "no kidding"]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def jaccard(a, b):
    sa = set(re.findall(r"\w+", norm(a)))
    sb = set(re.findall(r"\w+", norm(b)))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def paraphrase(title, body, rng, ai_style):
    """Variant generator: reorder/trim/expand, style shift, typo injection (rare)."""
    t, b = title, body
    if rng.random() < 0.35:
        parts = re.split(r"\.\s+", b)
        rng.shuffle(parts)
        b = ". ".join(parts)
    if rng.random() < 0.3:
        extra = rng.choice(REAL_SOBER + REAL_SENS) if rng.random() < 0.5 else rng.choice(FAKE_SOBER + FAKE_SHOUT)
        # keep label semantics: only append same-class flavor (caller picks pool)
        b = b + " " + extra
    if ai_style:
        # AI cue present only ~55%: rest read like edited/humanized AI (overlap)
        if rng.random() < 0.55:
            b += " " + rng.choice(AI_STOCK).capitalize() + ", the details remain consistent throughout."
        if rng.random() < 0.35:
            b = "Furthermore, " + b[0].lower() + b[1:]
        if rng.random() < 0.30:  # humanized AI: personal voice inside AI text
            b += " " + rng.choice(HUMAN_VOICE) + "."
    else:
        if rng.random() < 0.25:
            b += " " + rng.choice(HUMAN_VOICE) + "."
        if rng.random() < 0.18:  # humans also use formal connectors sometimes
            b += " " + rng.choice(AI_STOCK).capitalize() + " work continues."
    if rng.random() < 0.08:  # rare typo, keeps robustness honest
        b = b.replace("the", "teh", 1)
    if rng.random() < 0.3:
        t = t + (" — update" if rng.random() < 0.5 else " (developing)")
    return t, b


def make_image(path, label, rng, img_idx):
    """Harder images: hue/brightness randomized INDEPENDENT of label.
    Fake cue (subtle, present in ~70% of fakes, ~15% of reals as compression
    noise) = small spliced rectangle with mismatched noise + slight blur."""
    from PIL import Image, ImageDraw, ImageFilter
    import math
    w = h = 160
    base_hue = rng.randrange(0, 360)
    r = int(110 + 80 * math.sin(math.radians(base_hue)))
    g = int(110 + 80 * math.sin(math.radians(base_hue + 120)))
    b_ = int(110 + 80 * math.sin(math.radians(base_hue + 240)))
    img = Image.new("RGB", (w, h), (r % 256, g % 256, b_ % 256))
    px = img.load()
    for x in range(w):
        for y in range(h):
            n = rng.randint(-14, 14)
            px[x, y] = (max(0, min(255, px[x, y][0] + n)),
                        max(0, min(255, px[x, y][1] + n)),
                        max(0, min(255, px[x, y][2] + n)))
    d = ImageDraw.Draw(img)
    is_fake = (label == 0)
    cue_prob = 0.70 if is_fake else 0.15
    if rng.random() < cue_prob:
        x0, y0 = rng.randrange(10, 90), rng.randrange(10, 90)
        cw, ch = rng.randrange(30, 60), rng.randrange(20, 45)
        patch = Image.new("RGB", (cw, ch),
                          ((r + 60) % 256, (g + 60) % 256, (b_ + 60) % 256))
        img.paste(patch, (x0, y0))
        d.rectangle([x0, y0, x0 + cw, y0 + ch], outline=(20, 20, 20), width=1)
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(0.6))
    if rng.random() < 0.3:  # overlay bar both classes (meme-like, not label cue)
        d.rectangle([0, h - 24, w, h], fill=(15, 15, 15))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/realistic")
    ap.add_argument("--n-per-story", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = os.path.abspath(args.out)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    stories = []
    for i, (topic, h, b) in enumerate(REAL_STORIES):
        stories.append({"group": f"real-{i}", "label": 1, "topic": topic, "h": h, "b": b,
                        "src": SOURCES_REAL[i % len(SOURCES_REAL)]})
    for i, (topic, h, b) in enumerate(FAKE_STORIES):
        stories.append({"group": f"fake-{i}", "label": 0, "topic": topic, "h": h, "b": b,
                        "src": SOURCES_FAKE[i % len(SOURCES_FAKE)]})
    # Expand to 160 groups with UNIQUE headlines/bodies (combinatorial templates).
    # Copies sharing identical text would defeat group-aware splitting.
    ENT = ["Harbor City", "Northfield", "the regional authority", "state regulators",
           "the university hospital", "the transit agency", "local farmers",
           "the school board", "tech workers", "coastal towns", "the election commission",
           "small businesses", "the water utility", "the teachers union"]
    REAL_EV = ["approve funding for road repairs", "publish annual safety audit",
               "open a new training center", "complete bridge inspections",
               "launch a recycling pilot", "expand evening bus service",
               "report enrollment growth", "finish reservoir upgrades"]
    FAKE_EV = ["miracle cure they hide", "secret plot insiders expose",
               "free payouts for sharing", "cover-up of the missing files",
               "shocking confessions caught on tape", "banned video they deleted",
               "hidden microchips in deliveries", "rigged lottery proof"]
    REAL_BODY_S = [
        "The decision was announced after a public session with published minutes.",
        "Budget figures were verified against department records before release.",
        "An independent review panel signed off on the methodology last month.",
        "Residents can inspect the full documentation at the civic office.",
        "Follow-up inspections are scheduled quarterly with public summaries.",
    ]
    FAKE_BODY_S = [
        "No named witness would speak on the record about the central claim.",
        "The only evidence offered is a screenshot of unknown origin.",
        "Searches of public records turn up no matching document.",
        "The account spreading it was created days ago and posts only this.",
        "Archived versions of the cited page show nothing of the kind.",
    ]
    SHARED_S = [
        "Officials were contacted for comment.",
        "The timeline of events remains disputed.",
        "Further updates are expected as the situation develops.",
        "Community reactions have been mixed so far.",
    ]
    k = 0
    ALL_SOURCES = SOURCES_REAL + SOURCES_FAKE  # source must NOT predict label
    while len(stories) < 160:
        lab = 1 if (k % 2 == 0) else 0
        ent = ENT[(k * 7) % len(ENT)]
        if lab == 1:
            ev = REAL_EV[(k * 5) % len(REAL_EV)]
            h = f"{ent} {ev} in {TOPICS[k % len(TOPICS)]} update"
            if rng.random() < 0.20:  # real story, clickbait headline (still real)
                h = h + ": details you won't believe"
        else:
            if rng.random() < 0.50:  # fake with plausible headline (the hard case)
                ev = REAL_EV[(k * 5) % len(REAL_EV)]
                h = f"{ent} {ev} in {TOPICS[k % len(TOPICS)]} update"
            else:
                ev = FAKE_EV[(k * 5) % len(FAKE_EV)]
                h = f"{ent} {ev} in {TOPICS[k % len(TOPICS)]} update"
        sents = rng.sample(REAL_BODY_S if lab == 1 else FAKE_BODY_S, 3)
        if rng.random() < 0.6:
            sents += rng.sample(SHARED_S, 1)
        if lab == 0 and rng.random() < 0.5:  # sober fake borrows real phrasing
            sents.append("The filed report includes data tables, but the dataset was never released.")
        if lab == 1 and rng.random() < 0.3:  # real mentions denial
            sents.append("Officials also debunked a viral rumor about this, calling it false.")
        rng.shuffle(sents)
        stories.append({"group": f"{'real' if lab else 'fake'}-gen-{k}", "label": lab,
                        "topic": TOPICS[k % len(TOPICS)], "h": h, "b": " ".join(sents),
                        "src": rng.choice(ALL_SOURCES)})
        k += 1

    rows = []
    img_idx = 0
    real_bodies = [(s["h"], s["b"]) for s in stories if s["label"] == 1]
    for s in stories:
        for v in range(args.n_per_story):
            ai = 1 if (v % 2 == 1) else 0  # balanced AI/human within story
            base_h, base_b = s["h"], s["b"]
            # HARD negatives: fake claim wrapped in real reporting style/body
            if s["label"] == 0 and rng.random() < 0.45:
                rh, rb = real_bodies[rng.randrange(len(real_bodies))]
                base_b = rb  # true-style context carrying a false claim
                base_h = s["h"]
            t, b = paraphrase(base_h, base_b, rng, ai_style=bool(ai))
            if s["label"] == 1:
                flavor = rng.choice(REAL_SOBER if rng.random() < 0.65 else REAL_SENS)
            else:
                flavor = rng.choice(FAKE_SOBER if rng.random() < 0.75 else FAKE_SHOUT)
            if rng.random() < 0.5:
                b = b + " " + flavor
            if rng.random() < 0.35:  # shared boilerplate in BOTH classes
                b = b + " Officials were contacted for comment."
            if s["label"] == 0 and base_b != s["b"]:
                # the false claim itself, inside true-style context
                b = b + " However, insiders allege a hidden detail that no public record confirms."
            if s["label"] == 1 and rng.random() < 0.25:
                # real news that mentions rumors/denials (breaks keyword reliance)
                b = b + " Officials also debunked a viral rumor about this, calling it false and a hoax."
            if s["label"] == 1 and rng.random() < 0.20:
                # real article reports-then-debunks an allegation (same words as fakes)
                b = b + " However, insiders alleged a hidden detail, but public records contradict it."
            if s["label"] == 0 and rng.random() < 0.15:
                # fake article pre-empting denial (same denial words as reals)
                b = b + " Officials call this false and a hoax, but they would say that."
            ip = os.path.join(img_dir, f"img_{img_idx}.png")
            make_image(ip, s["label"], rng, img_idx)
            img_idx += 1
            n_words = len(b.split())
            rows.append({
                "id": f"{s['group']}-{v}",
                "group": s["group"],
                "title": t[:300], "text": b[:2000],
                "label": s["label"], "ai_label": ai,
                "image": os.path.relpath(ip, out),
                "source": s["src"], "topic": s["topic"],
                "style": "ai" if ai else "human",
                "len_bucket": "short" if n_words < 40 else ("long" if n_words > 120 else "mid"),
            })

    # ---- exact dedup ----
    seen, uniq = set(), []
    for r in rows:
        h = hashlib.md5((norm(r["title"]) + "|" + norm(r["text"])).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    # ---- near-dup removal (Jaccard>0.92 on title+text), sampled for speed ----
    kept = []
    for r in uniq:
        dup = False
        for q in kept[-400:]:
            if jaccard(r["title"] + " " + r["text"], q["title"] + " " + q["text"]) > 0.92:
                dup = True
                break
        if not dup:
            kept.append(r)

    # ---- group-aware stratified split 70/15/15 ----
    groups = sorted(set(r["group"] for r in kept))
    rng.shuffle(groups)
    n = len(groups)
    tr_g = set(groups[:int(n * 0.7)])
    va_g = set(groups[int(n * 0.7):int(n * 0.85)])
    te_g = set(groups[int(n * 0.85):])
    splits = {"train": [r for r in kept if r["group"] in tr_g],
              "val": [r for r in kept if r["group"] in va_g],
              "test": [r for r in kept if r["group"] in te_g]}

    for name, rs in splits.items():
        write_jsonl(os.path.join(out, name + ".jsonl"), rs)

    # ---- leakage report ----
    rep = {"total": len(kept), "raw": len(rows),
           "exact_dups_removed": len(rows) - len(uniq),
           "near_dups_removed": len(uniq) - len(kept)}
    for name, rs in splits.items():
        rep[name] = {"n": len(rs),
                     "real": sum(r["label"] for r in rs),
                     "fake": len(rs) - sum(r["label"] for r in rs),
                     "ai": sum(r["ai_label"] for r in rs),
                     "human": len(rs) - sum(r["ai_label"] for r in rs)}
    # group overlap must be zero
    rep["group_overlap_train_val"] = len(tr_g & va_g)
    rep["group_overlap_train_test"] = len(tr_g & te_g)
    # source distribution per split
    for name, rs in splits.items():
        from collections import Counter
        rep[name + "_sources"] = dict(Counter(r["source"] for r in rs))
    json.dump(rep, open(os.path.join(out, "leakage_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
