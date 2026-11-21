"""PHASE 1 — Prepare Fakeddit (multimodal split).

Real mode: point --raw at the official Fakeddit release folder
(github.com/entitize/fakeddit), which contains train.tsv / validate.tsv /
test.tsv plus an images/ directory. Image download from image_url is
attempted for rows whose file is missing (timeout-guarded, failures skipped).

Demo mode (--demo): generates a deterministic synthetic dataset (balanced
fake/real, text + solid-pattern images) so the whole pipeline runs offline.

Output: train.jsonl / val.jsonl / test.jsonl with rows:
  {id, title, text, image (local path or ""), label (0=fake, 1=real), source}

Fakeddit 2-way convention: 1 = true/real, 0 = fake. Override with
--true-values / --false-values if your copy differs (script prints the
mapping it used — always verify against the paper for your report).
"""
import argparse
import io
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_jsonl, set_seed

LABEL_CANDIDATES = ["2_way_label", "binary_label", "label", "2way_label"]


def detect_label_col(df, override=None):
    if override and override in df.columns:
        return override
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(f"No label column found. Columns: {list(df.columns)[:20]}. "
                     f"Pass --label-col explicitly.")


def map_labels(series, true_values, false_values):
    tv = {str(x).lower() for x in true_values}
    fv = {str(x).lower() for x in false_values}
    out = []
    for v in series:
        s = str(v).strip().lower()
        if s in tv:
            out.append(1)
        elif s in fv:
            out.append(0)
        else:
            out.append(None)
    return out


def download_images(rows, img_dir, workers=8, timeout=15):
    import concurrent.futures
    import requests
    os.makedirs(img_dir, exist_ok=True)

    def one(row):
        url = (row.get("image_url") or "").strip()
        if not url:
            return None
        name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        dest = os.path.join(img_dir, name)
        if os.path.exists(dest):
            return dest
        try:
            r = requests.get(url, timeout=timeout,
                             headers={"User-Agent": "FakeNewsResearch/1.0"})
            if r.status_code == 200 and len(r.content) > 2000:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(one, rows))
    n = sum(1 for p in paths if p)
    print(f"images resolved: {n}/{len(rows)}")
    return paths


def prepare_real(raw, out, args):
    import pandas as pd
    files = {"train": "train.tsv", "val": "validate.tsv", "test": "test.tsv"}
    for split, fn in files.items():
        p = os.path.join(raw, fn)
        if not os.path.exists(p):
            alt = os.path.join(raw, "data", fn)
            p = alt if os.path.exists(alt) else p
        if not os.path.exists(p):
            raise SystemExit(f"Missing {fn} in --raw {raw} (also tried data/). "
                             f"Download the official Fakeddit release first.")
        df = pd.read_csv(p, sep="\t")
        if args.max and len(df) > args.max:
            df = df.sample(args.max, random_state=42)
        lcol = detect_label_col(df, args.label_col)
        text_col = "title" if "title" in df.columns else df.columns[0]
        body_col = "text" if "text" in df.columns else None
        labels = map_labels(df[lcol], args.true_values.split(","), args.false_values.split(","))
        rows = []
        for i, (_, r) in enumerate(df.iterrows()):
            if labels[i] is None:
                continue
            rows.append({
                "id": f"fakeddit-{split}-{i}",
                "title": str(r.get(text_col, ""))[:500],
                "text": str(r.get(body_col, "") if body_col else "")[:2000],
                "image_url": str(r.get("image_url", "") or ""),
                "label": labels[i],
                "source": str(r.get("domain", "") or r.get("subreddit", "") or "fakeddit"),
            })
        print(f"{split}: label col '{lcol}', kept {len(rows)} rows "
              f"(real={sum(x['label'] for x in rows)}, fake={len(rows)-sum(x['label'] for x in rows)})")
        if not args.skip_images:
            paths = download_images(rows, os.path.join(out, "images"))
            for row, lp in zip(rows, paths):
                row["image"] = lp or ""
                row.pop("image_url", None)
        else:
            for row in rows:
                row["image"] = ""
        # multimodal split: keep rows that actually have an image file
        multi = [x for x in rows if x.get("image") and os.path.exists(x["image"])]
        print(f"{split}: multimodal rows with image file: {len(multi)}/{len(rows)}")
        write_jsonl(os.path.join(out, f"{split}.jsonl"), multi if multi else rows)


REAL_TITLES = [
    ("City council approves new public transit funding", 1),
    ("Researchers publish peer-reviewed vaccine safety study", 1),
    ("Central bank holds interest rates steady amid growth", 1),
    ("National team wins championship after final match", 1),
    ("SHOCKING miracle cure doctors HATE this one trick!!!", 0),
    ("You won't BELIEVE what this celebrity did — secret exposed", 0),
    ("URGENT: free money guaranteed, share before deleted!!", 0),
    ("Breaking: alien cover-up finally revealed by insider?!", 0),
]
REAL_BODIES = {
    1: "Officials confirmed the details in a press conference today. The full report with data tables was published and independent experts reviewed the methodology before release.",
    0: "Nobody is reporting this except us!!! Share share share before THEY delete it!!! One hundred percent guaranteed unbelievable secret!!!",
}


def prepare_demo(out, n_train=600, n_val=150, n_test=150):
    """Deterministic synthetic data: varied templates + generated images."""
    from PIL import Image, ImageDraw
    import random
    set_seed(42)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    def make_image(label, idx):
        img = Image.new("RGB", (224, 224),
                        (30 + (idx * 37) % 180, 60 + (idx * 53) % 150, 90 + (idx * 29) % 120))
        d = ImageDraw.Draw(img)
        if label == 0:  # fake: harsh diagonal + block (manipulation-like cue)
            d.line([(0, 0), (224, 224)], fill=(255, 40, 40), width=9)
            d.rectangle([150, 20, 210, 80], fill=(250, 250, 250))
        else:  # real: calm grid
            for x in range(0, 224, 28):
                d.line([(x, 0), (x, 224)], fill=(120, 200, 255), width=2)
        p = os.path.join(img_dir, f"demo_{idx}.png")
        img.save(p)
        return p

    def rows(n, start):
        out_rows = []
        for i in range(n):
            t, lab = REAL_TITLES[(start + i) % len(REAL_TITLES)]
            vary = f" (update {start + i})" if (start + i) % 3 == 0 else ""
            out_rows.append({
                "id": f"demo-{start + i}",
                "title": t + vary,
                "text": REAL_BODIES[lab],
                "image": make_image(lab, start + i),
                "label": lab,
                "source": "demo",
            })
        return out_rows

    write_jsonl(os.path.join(out, "train.jsonl"), rows(n_train, 0))
    write_jsonl(os.path.join(out, "val.jsonl"), rows(n_val, n_train))
    write_jsonl(os.path.join(out, "test.jsonl"), rows(n_test, n_train + n_val))
    print(f"demo data written to {out} (train={n_train}, val={n_val}, test={n_test})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="", help="official Fakeddit release folder")
    ap.add_argument("--out", default="data/fakeddit")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="synthetic offline data")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--label-col", default="")
    ap.add_argument("--true-values", default="1,true,real")
    ap.add_argument("--false-values", default="0,false,fake")
    args = ap.parse_args()
    # NOTE: '--demo' flag parsed manually to keep file editing simple
    demo = args.demo or "--demo" in sys.argv
    set_seed(42)
    if demo:
        prepare_demo(args.out)
    else:
        if not args.raw:
            raise SystemExit("Pass --raw <fakeddit release folder> or use --demo.")
        prepare_real(args.raw, args.out, args)
