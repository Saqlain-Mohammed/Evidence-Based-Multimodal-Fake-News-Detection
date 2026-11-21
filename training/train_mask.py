"""Train an independent pixel classifier using CASIA's actual ground-truth masks."""
import io
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.mask_features import pixels


def main():
    prepared = ROOT / "data/prepared/casia"
    masks = {}
    parquet = pq.ParquetFile(ROOT / "data/public/casia/all.parquet")
    for batch in parquet.iter_batches(batch_size=32):
        for record in batch.to_pylist():
            name = Path(record["image"]["path"]).name
            with Image.open(io.BytesIO(record["mask"]["bytes"])) as image:
                masks[name] = np.asarray(image.convert("L").resize((64,64),Image.Resampling.NEAREST)).ravel()>0
    def samples(split):
        for line in (prepared/f"{split}.jsonl").read_text().splitlines():
            row=json.loads(line);path=Path(row["path"])
            yield pixels(path.read_bytes()),masks[path.name],bool(row["label"])
    features,targets=[],[]
    rng=np.random.default_rng(42)
    for x,mask,_ in samples("train"):
        for flag in (True,False):
            indices=np.flatnonzero(mask==flag)
            if len(indices):
                take=rng.choice(indices,min(32,len(indices)),replace=False)
                features.append(x[take]);targets.append(mask[take])
    model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,class_weight="balanced",random_state=42)
    model.fit(np.concatenate(features),np.concatenate(targets))
    def evaluate(split,thresholds):
        iou=np.zeros(len(thresholds));clean_area=np.zeros(len(thresholds));edited=clean=0
        pending=[]
        def score(chunk):
            nonlocal edited,clean
            probs=model.predict_proba(np.concatenate([row[0] for row in chunk]))[:,1].reshape(len(chunk),4096)
            for scores,(_,mask,tampered) in zip(probs,chunk):
                pred=scores[:,None]>=thresholds
                if tampered:
                    iou[:]+=np.logical_and(pred,mask[:,None]).sum(0)/np.maximum(1,np.logical_or(pred,mask[:,None]).sum(0));edited+=1
                else:clean_area[:]+=pred.mean(0);clean+=1
        for sample in samples(split):
            pending.append(sample)
            if len(pending)==16:score(pending);pending=[]
        if pending:score(pending)
        return iou/edited,clean_area/clean,edited,clean
    grid=np.arange(.3,.91,.05)
    iou,area,_,_=evaluate("val",grid)
    # Penalize excessive highlights in authentic images when selecting an operating point.
    threshold=float(grid[np.argmax(iou-area)])
    iou,area,edited,clean=evaluate("test",np.array([threshold]))
    folder=ROOT/"models/public/tamper_mask";folder.mkdir(parents=True,exist_ok=True)
    report={"dataset":"CASIA v2","architecture":"Pixel-wise boosted color and local residual features",
            "trained_from_scratch":True,"threshold":threshold,"tampered_test_images":edited,"authentic_test_images":clean,
            "mean_tampered_image_iou":float(iou[0]),"mean_authentic_highlight_fraction":float(area[0]),
            "limitations":"64x64 localization estimate. IoU is averaged on tampered test images. Authentic highlighted area is averaged separately."}
    joblib.dump({"model":model,"threshold":threshold},folder/"model.joblib",compress=3)
    (folder/"report.json").write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=3): main()
