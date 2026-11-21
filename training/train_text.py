"""PHASE 2 — Fine-tune a text classifier (DistilBERT default, BERT optional).

Manual torch loop (no accelerate needed). Saves artifacts/text_model/
(model + tokenizer + label map) and merges metrics into metrics.json.
Demo defaults are small so the full pipeline finishes on CPU in minutes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, set_seed, save_metrics, prf


def build_loaders(train_rows, val_rows, tok, max_len, batch):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class TD(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            t = ((r.get("title") or "") + " " + (r.get("text") or ""))[:2000]
            e = tok(t, truncation=True, max_length=max_len, padding="max_length",
                    return_tensors="pt")
            return {k: v.squeeze(0) for k, v in e.items()}, torch.tensor(r["label"])

    def collate(b):
        xs, ys = zip(*b)
        out = {k: torch.stack([x[k] for x in xs]) for k in xs[0]}
        return out, torch.stack(list(ys))

    return (DataLoader(TD(train_rows), batch_size=batch, shuffle=True, collate_fn=collate),
            DataLoader(TD(val_rows), batch_size=batch, collate_fn=collate))


def run_epoch(model, loader, opt, device, train=True):
    import torch
    model.train(train)
    tot_loss, preds, gold = 0.0, [], []
    for x, y in loader:
        x = {k: v.to(device) for k, v in x.items()}
        y = y.to(device)
        if train:
            opt.zero_grad()
        out = model(**x, labels=y)
        loss = out.loss
        if train:
            loss.backward()
            opt.step()
        tot_loss += loss.item() * len(y)
        preds += out.logits.argmax(-1).detach().cpu().tolist()
        gold += y.cpu().tolist()
    return tot_loss / max(1, len(gold)), preds, gold


def main(args):
    set_seed(42)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(args.data, "val.jsonl"))
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    print(f"text train={len(train_rows)} val={len(val_rows)} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=args.offline)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tr_loader, va_loader = build_loaders(train_rows, val_rows, tok, args.max_len, args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_f1, best_state = -1, None
    for ep in range(args.epochs):
        tr_loss, _, _ = run_epoch(model, tr_loader, opt, device, True)
        va_loss, preds, gold = run_epoch(model, va_loader, opt, device, False)
        m = prf(gold, preds)
        print(f"epoch {ep+1}/{args.epochs} train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} acc={m['accuracy']} f1={m['f1']}")
        if m["f1"] >= best_f1:
            best_f1 = m["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = m
    model.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    json.dump({"labels": {"0": "fake", "1": "real"}, "base": args.model},
              open(os.path.join(args.out, "model_info.json"), "w"))
    save_metrics(os.path.dirname(os.path.abspath(args.out)) if args.out.endswith("text_model")
                 else args.out, {})
    art = os.path.normpath(args.out + "/..") if os.path.basename(os.path.normpath(args.out)) == "text_model" else args.out
    save_metrics(art, {"text_model": {"base": args.model, "epochs": args.epochs,
                                      "val": best_metrics}})
    print(f"saved text model -> {args.out}  best val: {best_metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fakeddit")
    ap.add_argument("--out", default="../backend/artifacts/text_model")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--offline", action="store_true",
                    help="use cached HF weights only (no download)")
    args, _ = ap.parse_known_args()
    if "--demo" in sys.argv:
        args.epochs = min(args.epochs, 1)
        args.batch = 8
        args.max_len = 64
        if not args.max_train:
            args.max_train = 400
    main(args)
