"""Download public research corpora with atomic writes and recorded hashes.

All cache/data paths are on the project drive, never the system drive.
No dataset code is executed. See data/catalog.json for access/usage notes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "multinli": {f"{s}.parquet": f"https://huggingface.co/datasets/nyu-mll/multi_nli/resolve/da70db2af9d09693783c3320c4249840212ee221/data/{s}-00000-of-00001.parquet" for s in ("train","validation_matched","validation_mismatched")},
    "welfake": {"WELFake_Dataset.csv": "https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1"},
    "hc3": {"all.jsonl": "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"},
    "cifake": {f"{s}.parquet": f"https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/data/{s}-00000-of-00001.parquet" for s in ("train", "test")},
    "liar": {"liar.zip": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"},
    "asvspoof": {f"{s}.parquet": f"https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/resolve/aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c/data/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")},
    "deepfaketimit": {"DeepfakeTIMIT.tar.gz": "https://zenodo.org/api/records/4068245/files/DeepfakeTIMIT.tar.gz/content"},
    "casia": {"all.parquet": "https://huggingface.co/datasets/KhangTruong/Casia-v2.0/resolve/main/data/cross_test-00000-of-00001.parquet"},
    "uadfv": {f"{s}.zip": f"https://huggingface.co/datasets/scarlettss/UADFV/resolve/main/{s}.zip" for s in ('real','fake')},
}

def download(name):
    folder = ROOT / "data" / "public" / name
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in SOURCES[name].items():
        dest = folder / filename
        if not dest.exists():
            print(f"Downloading {name}/{filename}", flush=True)
            partial = dest.with_suffix(dest.suffix + ".partial")
            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=(20, 90)) as r:
                        r.raise_for_status()
                        with partial.open("wb") as f:
                            for chunk in r.iter_content(1024 * 1024):
                                f.write(chunk)
                    partial.replace(dest)
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
        h = hashlib.sha256()
        with dest.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        records.append({"file": filename, "url": url, "bytes": dest.stat().st_size, "sha256": h.hexdigest()})
        print(f"Ready {name}/{filename}: {dest.stat().st_size:,} bytes", flush=True)
    (folder / "provenance.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["welfake", "hc3", "cifake", "liar"], choices=list(SOURCES))
    args = ap.parse_args()
    for name in args.datasets:
        download(name)
