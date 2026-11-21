"""PHASE runner: executes the full pipeline.

Demo (offline, default): synthetic data + small training on CPU.
  python run_all.py --demo

Real (needs downloads + GPU recommended):
  python run_all.py --real --fakeddit-raw <official release> --fakenewsnet-raw <checkout>
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd):
    print("\n=== " + " ".join(cmd) + " ===")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--fakeddit-raw", default="")
    ap.add_argument("--fakenewsnet-raw", default="")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    demo = args.demo or not args.real
    flag = ["--demo"] if demo else []

    # PHASE 1 — data
    sh([sys.executable, "prepare_fakeddit.py", "--out", "data/fakeddit"] + flag)
    sh([sys.executable, "prepare_fakenewsnet.py", "--out", "data/fakenewsnet"] + flag)
    # PHASE 2 — text
    sh([sys.executable, "train_text.py", "--data", "data/fakeddit",
        "--out", "../backend/artifacts/text_model"] + flag)
    # PHASE 3 — image
    sh([sys.executable, "train_image.py", "--data", "data/fakeddit",
        "--out", "../backend/artifacts/image_model.pt"] + flag)
    # PHASE 4 — fusion (blend both datasets' train rows for robustness)
    sh([sys.executable, "train_fusion.py", "--data", "data/fakeddit",
        "--text-model", "../backend/artifacts/text_model",
        "--image-model", "../backend/artifacts/image_model.pt",
        "--out", "../backend/artifacts/fusion.pkl"] + flag)
    # PHASE 11 — evaluation
    if not args.skip_eval:
        sh([sys.executable, "evaluate.py", "--data", "data/fakeddit",
            "--artifacts", "../backend/artifacts",
            "--text-model", "../backend/artifacts/text_model",
            "--image-model", "../backend/artifacts/image_model.pt",
            "--fusion", "../backend/artifacts/fusion.pkl",
            "--cross", "data/fakenewsnet/test.jsonl"])
    print("\nALL PHASES COMPLETE. Artifacts -> backend/artifacts/")


if __name__ == "__main__":
    main()
