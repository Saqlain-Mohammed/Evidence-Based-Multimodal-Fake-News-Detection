"""PHASE 3 — Fine-tune ResNet50 image classifier.

Freezes the backbone by default (fast, CPU-friendly); unfreeze with
--unfreeze. Saves artifacts/image_model.pt (state dict + meta) and merges
image metrics into metrics.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl, set_seed, save_metrics, prf


def make_model(num_classes=2, pretrained=True, freeze=True):
    import torchvision.models as models
    try:
        from torchvision.models import ResNet50_Weights
        w = ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
    except Exception:
        m = models.resnet50(pretrained=pretrained)
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    import torch
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


def loaders(train_rows, val_rows, batch, size=160, data_dir="", augment=True):
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    def resolve(p):
        if p and os.path.exists(p):
            return p
        if p and data_dir and os.path.exists(os.path.join(data_dir, p)):
            return os.path.join(data_dir, p)
        return ""

    if augment:
        tf_train = T.Compose([T.Resize((size, size)),
                              T.RandomHorizontalFlip(p=0.5),
                              T.ColorJitter(brightness=0.15, contrast=0.15),
                              T.ToTensor(),
                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        tf_train = None
    tf_eval = T.Compose([T.Resize((size, size)), T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    class ID(Dataset):
        def __init__(self, rows, train):
            self.paths = []
            for r in rows:
                p = resolve(r.get("image", ""))
                if p:
                    self.paths.append((p, r["label"]))
            self.tf = tf_train if (train and tf_train is not None) else tf_eval
            print(f"rows with usable image: {len(self.paths)}/{len(rows)}")

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            import torch
            p, y = self.paths[i]
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (size, size), (128, 128, 128))
            return self.tf(img), torch.tensor(y)

    return (DataLoader(ID(train_rows, True), batch_size=batch, shuffle=True),
            DataLoader(ID(val_rows, False), batch_size=batch))


def run_epoch(model, loader, opt, device, train, weight=None):
    import torch
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    model.train(train)
    tot, preds, gold = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        if train:
            loss.backward()
            opt.step()
        tot += loss.item() * len(y)
        preds += logits.argmax(-1).detach().cpu().tolist()
        gold += y.cpu().tolist()
    return tot / max(1, len(gold)), preds, gold


def main(args):
    set_seed(42)
    import torch
    train_rows = read_jsonl(os.path.join(args.data, "train.jsonl"))
    val_rows = read_jsonl(os.path.join(args.data, "val.jsonl"))
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    model = make_model(pretrained=not args.no_pretrained, freeze=not args.unfreeze)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tr_loader, va_loader = loaders(train_rows, val_rows, args.batch,
                                   size=args.size, data_dir=os.path.abspath(args.data))
    # class-weighted loss (handles imbalance honestly, no resampling leakage)
    from collections import Counter
    cnt = Counter(int(r["label"]) for r in train_rows)
    w = torch.tensor([len(train_rows) / max(1, cnt[0]), len(train_rows) / max(1, cnt[1])],
                     dtype=torch.float)
    print(f"class weights (fake,real): {w.tolist()}")
    import torch.nn as _nn
    _orig = run_epoch  # keep signature; inject weights via closure below
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_f1, best_state, best_metrics = -1, None, None
    for ep in range(args.epochs):
        tr_loss, _, _ = run_epoch(model, tr_loader, opt, device, True, w.to(device))
        va_loss, preds, gold = run_epoch(model, va_loader, opt, device, False)
        m = prf(gold, preds)
        print(f"epoch {ep+1}/{args.epochs} train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} acc={m['accuracy']} f1={m['f1']}")
        if m["f1"] >= best_f1:
            best_f1, best_metrics = m["f1"], m
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"state_dict": best_state, "labels": {"0": "fake", "1": "real"},
                "backbone": "resnet50", "unfrozen": bool(args.unfreeze)}, out)
    art = os.path.dirname(os.path.abspath(out))
    save_metrics(art, {"image_model": {"backbone": "resnet50", "epochs": args.epochs,
                                       "val": best_metrics}})
    print(f"saved image model -> {out}  best val: {best_metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fakeddit")
    ap.add_argument("--out", default="../backend/artifacts/image_model.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--size", type=int, default=160)
    args, _ = ap.parse_known_args()
    if "--demo" in sys.argv:
        args.epochs = min(args.epochs, 2)
        args.batch = 8
        if not args.max_train:
            args.max_train = 400
    main(args)
