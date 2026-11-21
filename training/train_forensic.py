"""Compare a compression-residual classifier to the CASIA CNN by validation macro F1."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.forensic_features import features
from training.train_public import metrics,save_report


def split(name):
    folder=ROOT/"data/prepared/casia"
    rows=[json.loads(line) for line in (folder/f"{name}.jsonl").read_text().splitlines()]
    return np.stack([features(Path(r["path"]).read_bytes()) for r in rows]),np.array([r["label"] for r in rows])


if __name__=="__main__":
    with threadpool_limits(limits=2):
        x,y=split("train")
        model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,class_weight="balanced",random_state=42).fit(x,y)
        val,vy=split("val")
        raw=model.decision_function(val).reshape(-1,1)
        calib=LogisticRegression().fit(raw,vy)
        val_f1=metrics(vy,calib.predict_proba(raw)[:,1])["macro_f1"]
        print("CASIA forensic validation",val_f1,flush=True)
        folder=ROOT/"models/public/image_manipulation"
        cnn=json.loads((folder/"report.json").read_text())
        if cnn["architecture"]!="ImageCNN":raise ValueError("Run the CNN comparison first.")
        import torch
        from training.scratch_models import ImageCNN
        from training.train_media import Media
        torch.set_num_threads(2)
        net=ImageCNN();checkpoint=torch.load(folder/"model.pt",map_location="cpu",weights_only=True)
        net.load_state_dict(checkpoint["state_dict"]);net.eval()
        rows=[json.loads(line) for line in (ROOT/"data/prepared/casia/val.jsonl").read_text().splitlines()]
        scores=[]
        with torch.inference_mode():
            for a,b in torch.utils.data.DataLoader(Media(rows,"image_manipulation"),batch_size=64):
                scores.extend((net(a)/checkpoint["temperature"]).sigmoid().tolist())
        baseline=metrics(vy,np.array(scores))["macro_f1"]
        if val_f1>baseline:
            test,ty=split("test")
            scores=calib.predict_proba(model.decision_function(test).reshape(-1,1))[:,1]
            payload={"classifier":model,"calibration":calib}
            joblib.dump(payload,folder/"model.joblib",compress=3)
            save_report("image_manipulation",{"dataset":cnn["dataset"],"architecture":"Compression and residual features with gradient boosting",
                "artifact_format":"joblib","counts":cnn["counts"],"test":metrics(ty,scores),"cnn_baseline_test":cnn["test"],
                "validation_macro_f1":{"cnn":baseline,"forensic":val_f1},"evaluation_scope":"internal-source-connected-split",
                "limitations":"CASIA acquisition and compression signatures can predict historical labels. Source components create a substantial class shift. This is an exploratory comparison on one corpus."})
