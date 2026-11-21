"""Train independent CPU models from scratch on real public data.

python training/train_public.py --task all --epochs 5
No synthetic data, pretrained embeddings, or test-set model selection.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache"))
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix

def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()

def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()

def splits(groups):
    idx = np.arange(len(groups))
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(idx, groups=groups))
    va, te = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=43).split(rest, groups=np.array(groups)[rest]))
    return tr, rest[va], rest[te]

def metrics(y, p):
    return {"accuracy": float(accuracy_score(y, p >= .5)), "macro_f1": float(f1_score(y, p >= .5, average="macro")),
        "roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, p >= .5, labels=[0, 1]).tolist(), "n": len(y)}

def save_report(task, report):
    folder = ROOT / "models" / "public" / task
    folder.mkdir(parents=True, exist_ok=True)
    report.update({"task": task, "synthetic": False, "trained_from_scratch": True, "positive_label": 1,
        "created_utc": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(task, json.dumps(report), flush=True)
    return folder

def text_task(task):
    import pandas as pd
    import joblib
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    texts, labels, groups = [], [], []
    seen = set()
    if task == "fake_news":
        frame = pd.read_csv(ROOT / "data/public/welfake/WELFake_Dataset.csv").dropna(subset=["title", "text", "label"])
        for r in frame.to_dict("records"):
            body = str(r['text'])
            if len(body.split()) < 30:
                continue
            key = digest(body)
            if key in seen:
                continue
            seen.add(key)
            texts.append((str(r['title']) + ". " + body)[:18000])
            labels.append(int(r['label']))  # Released CSV uses 1=fake, verified against source counts and Reuters datelines.
            groups.append(digest(body[:500]))
        dataset, caveat = "WELFake", "Historical English articles; publisher/topic artifacts remain. Not factual verification. Near-duplicate paraphrases may remain."
    else:
        for line in (ROOT / "data/public/hc3/all.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            group = digest(r['question'])
            for field, label in [('human_answers', 0), ('chatgpt_answers', 1)]:
                for answer in r.get(field, []):
                    key = digest(answer)
                    if key in seen or len(answer.split()) < 20:
                        continue
                    seen.add(key)
                    texts.append(answer[:18000]); labels.append(label); groups.append(group)
        dataset, caveat = "HC3 English", "ChatGPT-era answer detection; does not establish authorship or generalize automatically to newer generators, edited text, or other languages."
    y = np.asarray(labels)
    tr, va, te = splits(groups)
    print(f"{task}: {len(y):,} examples; train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    # Fit vocabulary exclusively on train. Calibrate on validation; test is untouched.
    model = Pipeline([('features', FeatureUnion([
        ('word', TfidfVectorizer(max_features=65000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)),
        ('char', TfidfVectorizer(max_features=35000, analyzer='char_wb', ngram_range=(3, 5), min_df=5, sublinear_tf=True, dtype=np.float32))])),
        ('classifier', LogisticRegression(C=4, max_iter=350, solver='liblinear', class_weight='balanced', random_state=42))])
    model.fit([texts[i] for i in tr], y[tr])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
    calibrated.fit([texts[i] for i in va], y[va])
    prob = calibrated.predict_proba([texts[i] for i in te])[:, list(calibrated.classes_).index(1)]
    report = {"dataset": dataset, "architecture": "word + character TF-IDF / logistic regression / held-out sigmoid calibration",
        "counts": {"train": len(tr), "validation": len(va), "test": len(te)}, "test": metrics(y[te], prob),
        "split": "grouped 80/10/10; article prefix or HC3 question; exact full-text duplicates removed before splitting",
        "group_overlap": 0, "limitations": caveat}
    folder = save_report(task, report)
    joblib.dump({"model": calibrated, "task": task, "positive_label": 1}, folder / "model.joblib")
    (folder / 'split_manifest.json').write_text(json.dumps({s: [groups[i] for i in indices] for s, indices in [('train',tr),('validation',va),('test',te)]}), encoding='utf-8')
    if task == 'fake_news':
        evaluate_liar(calibrated, folder)

def evaluate_liar(model, folder):
    import csv, zipfile
    file = ROOT / 'data/public/liar/liar.zip'
    if not file.exists(): return
    with zipfile.ZipFile(file) as z:
        rows = list(csv.reader(io.StringIO(z.read('test.tsv').decode()), delimiter='\t'))
    # Keep the original six labels in the mapping documentation. Half-true excluded.
    rows = [r for r in rows if r[1] != 'half-true']
    y = np.array([int(r[1] in ('false','barely-true','pants-fire')) for r in rows])
    p = model.predict_proba([r[2] for r in rows])[:,list(model.classes_).index(1)]
    result = {'dataset':'LIAR official test (zero adaptation)', 'test':metrics(y,p),
        'mapping':'false/barely-true/pants-fire=1; mostly-true/true=0; half-true excluded',
        'purpose':'Cross-domain stress test. Low performance must be reported.'}
    (folder / 'cross_domain.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('LIAR cross-domain', result, flush=True)

def image_task(epochs):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from PIL import Image
    import pyarrow.parquet as pq
    from training.scratch_models import ImageCNN
    torch.manual_seed(42); torch.set_num_threads(6)
    def load(split):
        rows = pq.read_table(ROOT / f'data/public/cifake/{split}.parquet').to_pylist()
        xs, ys, keys = [], [], []
        for row in rows:
            im = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB').resize((32,32))
            a = np.asarray(im).copy()
            xs.append(a); ys.append(1 - row['label']); keys.append(hashlib.sha256(a.tobytes()).hexdigest())
        return np.stack(xs), np.array(ys), keys
    x,y,keys = load('train'); xt,yt,kt = load('test')
    # Retain original test; remove duplicates and any train/test pixel collisions.
    seen = set(kt); keep=[]
    for i,k in enumerate(keys):
        if k not in seen: keep.append(i); seen.add(k)
    x,y = x[keep],y[keep]; groups=[keys[i] for i in keep]
    tr, va = next(GroupShuffleSplit(n_splits=1,test_size=.1,random_state=42).split(x,groups=groups))
    def loader(a,b,shuffle=False):
        return DataLoader(TensorDataset(torch.from_numpy(a).permute(0,3,1,2),torch.tensor(b,dtype=torch.float32)),batch_size=256,shuffle=shuffle)
    train, val, test = loader(x[tr],y[tr],True),loader(x[va],y[va]),loader(xt,yt)
    model=ImageCNN(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    criterion=torch.nn.BCEWithLogitsLoss(); best=-1; state=None
    def predict(dl):
        model.eval(); ps=[]
        with torch.inference_mode():
            for a,b in dl: ps.extend(model(a.float()/255).sigmoid().tolist())
        return np.array(ps)
    for epoch in range(epochs):
        model.train(); total=0
        for a,b in train:
            a=a.float()/255
            if torch.rand(1).item()<.5: a=a.flip(-1)
            opt.zero_grad(); loss=criterion(model(a),b); loss.backward(); opt.step(); total+=float(loss.detach())
        p=predict(val); score=f1_score(y[va],p>=.5,average='macro')
        print(f'image_ai epoch {epoch+1}/{epochs} loss={total/len(train):.4f} val_f1={score:.4f}',flush=True)
        if score>best:
            best=score; state={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    # Temperature scaling on validation, not test.
    model.eval()
    val_logits=torch.tensor(np.log(np.clip(predict(val),1e-6,1-1e-6)/(1-np.clip(predict(val),1e-6,1-1e-6))))
    log_t=torch.nn.Parameter(torch.zeros(())); optimizer=torch.optim.LBFGS([log_t],max_iter=40)
    def closure():
        optimizer.zero_grad(); loss=criterion(val_logits/log_t.exp(),torch.tensor(y[va],dtype=torch.float32)); loss.backward(); return loss
    optimizer.step(closure); temperature=float(log_t.exp().detach().clamp(.1,10))
    p=predict(test); p=1/(1+np.exp(-np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))/temperature))
    folder=save_report('image_ai',{'dataset':'CIFAKE (original split, HF mirror)', 'architecture':'randomly initialized three-block CNN',
        'epochs':epochs,'counts':{'train':len(tr),'validation':len(va),'test':len(yt)},'test':metrics(yt,p),
        'temperature':temperature,'split':'official test; training pixel duplicates and train/test collisions removed; validation held out from train',
        'limitations':'32x32 CIFAR-10 versus Stable Diffusion 1.4. Not a manipulation detector; no claim of accuracy on modern generators or arbitrary photos.'})
    torch.save({'state_dict':state,'task':'image_ai','positive_label':1,'size':32,'temperature':temperature},folder/'model.pt')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['all','fake_news','ai_text','image_ai'],default='all'); ap.add_argument('--epochs',type=int,default=5)
    args=ap.parse_args()
    for task in (['fake_news','ai_text','image_ai'] if args.task=='all' else [args.task]):
        if task=='image_ai': image_task(args.epochs)
        else: text_task(task)
