"""PHASE 1 — Prepare FakeNewsNet (GossipCop / PolitiFact article folders).

Accepts the standard FakeNewsNet checkout layout:
  <raw>/gossipcop/fake/<id>/news content.json ...
  <raw>/politifact/real/<id>/...

The parser is deliberately flexible: it walks for *.json files and extracts
title/text from common keys (title/headline, text/body/content/article) and
images from common keys (images/top_img/image) or sibling image files.
Rows: {id, title, text, image, label (0=fake, 1=real), source (domain or outlet)}

--demo generates synthetic article-style rows so the pipeline runs offline.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_jsonl, set_seed

TITLE_KEYS = ["title", "headline", "headline_title"]
TEXT_KEYS = ["text", "body", "content", "article", "body_text", "news_text"]
IMG_KEYS = ["images", "image", "top_img", "top_image", "main_image"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def pick(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            return v[0] if isinstance(v[0], str) else ""
    return ""


def parse_real(raw, max_per_split=0):
    rows = []
    for dataset in ("gossipcop", "politifact"):
        for label_name, label in (("fake", 0), ("real", 1)):
            base = os.path.join(raw, dataset, label_name)
            if not os.path.isdir(base):
                continue
            for entry in sorted(os.listdir(base)):
                ed = os.path.join(base, entry)
                if not os.path.isdir(ed):
                    continue
                data = {}
                for jf in glob.glob(os.path.join(ed, "*.json")):
                    try:
                        data.update(json.load(open(jf, encoding="utf-8")))
                    except Exception:
                        pass
                title = pick(data, TITLE_KEYS)
                text = pick(data, TEXT_KEYS)
                if not (title or text):
                    continue
                img = pick(data, IMG_KEYS)
                if not img or not os.path.exists(img):
                    img = ""
                    for f in glob.glob(os.path.join(ed, "*")):
                        if f.lower().endswith(IMG_EXTS):
                            img = f
                            break
                domain = ""
                for k in ("source", "publisher", "domain", "outlet", "site"):
                    if isinstance(data.get(k), str) and data[k].strip():
                        domain = data[k].strip()
                        break
                rows.append({"id": f"fnn-{dataset}-{entry}", "title": title[:500],
                             "text": text[:6000], "image": img, "label": label,
                             "source": domain or dataset})
                if max_per_split and sum(1 for r in rows if r["label"] == label) >= max_per_split * 2:
                    pass
    return rows


def prepare_demo(out, n=300):
    from PIL import Image, ImageDraw
    set_seed(7)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    topics = [("Election commission releases final voter turnout data", 1),
              ("Health ministry issues seasonal advisory for citizens", 1),
              ("Scientists confirm water found in new deep-sea survey", 1),
              ("SHOCKING celebrity secret the media HIDES from you!!!", 0),
              ("Miracle wealth method banned in 12 countries?!", 0)]
    rows = []
    for i in range(n):
        t, lab = topics[i % len(topics)]
        img = Image.new("RGB", (224, 224), (200, 200, 210) if lab else (210, 180, 180))
        d = ImageDraw.Draw(img)
        d.ellipse([40, 40, 184, 184], outline=(40, 120, 40) if lab else (200, 40, 40), width=6)
        p = os.path.join(img_dir, f"fnn_demo_{i}.png")
        img.save(p)
        body = ("The article reports verified statements from officials with named sources and "
                "published data. " * 6) if lab else ("Unverified claims with no named sources!!! "
                "Share urgently before it is deleted!!! " * 6)
        rows.append({"id": f"fnn-demo-{i}", "title": t, "text": body, "image": p,
                     "label": lab, "source": "demo-outlet"})
    # stratified split 70/15/15
    real = [r for r in rows if r["label"] == 1]
    fake = [r for r in rows if r["label"] == 0]

    def split(part, fr):
        n0, n1 = int(len(part) * fr[0]), int(len(part) * fr[1])
        return part[:n0], part[n0:n0 + n1], part[n0 + n1:]

    tr_r, va_r, te_r = split(real, (0.7, 0.15))
    tr_f, va_f, te_f = split(fake, (0.7, 0.15))
    write_jsonl(os.path.join(out, "train.jsonl"), tr_r + tr_f)
    write_jsonl(os.path.join(out, "val.jsonl"), va_r + va_f)
    write_jsonl(os.path.join(out, "test.jsonl"), te_r + te_f)
    print(f"demo FakeNewsNet-style data -> {out} "
          f"(train={len(tr_r+tr_f)}, val={len(va_r+va_f)}, test={len(te_r+te_f)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="")
    ap.add_argument("--out", default="data/fakenewsnet")
    args, _ = ap.parse_known_args()
    demo = "--demo" in sys.argv
    set_seed(42)
    if demo:
        prepare_demo(args.out)
    else:
        if not args.raw:
            raise SystemExit("Pass --raw <FakeNewsNet checkout> or use --demo.")
        rows = parse_real(args.raw)
        if not rows:
            raise SystemExit("No articles parsed — check --raw layout (see README).")
        import random
        random.seed(42)
        random.shuffle(rows)
        n = len(rows)
        write_jsonl(os.path.join(args.out, "train.jsonl"), rows[:int(n * 0.7)])
        write_jsonl(os.path.join(args.out, "val.jsonl"), rows[int(n * 0.7):int(n * 0.85)])
        write_jsonl(os.path.join(args.out, "test.jsonl"), rows[int(n * 0.85):])
        print(f"parsed {n} articles -> {args.out}")
