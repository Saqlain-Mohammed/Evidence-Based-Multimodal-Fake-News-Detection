"""Select a UADFV clip classifier on source-disjoint validation data."""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"backend"))
from training.temporal_features import describe
from training.train_public import metrics,save_report
from training.scratch_models import VideoCNN
from workbench import decode_video


def run():
    torch.set_num_threads(2)
    folder=ROOT/"models/public/video"
    original=json.loads((folder/"report.json").read_text())
    if original["architecture"]!="VideoCNN":raise ValueError("Train the CNN baseline before comparison.")
    network=VideoCNN()
    blob=torch.load(folder/"model.pt",weights_only=True,map_location="cpu")
    network.load_state_dict(blob["state_dict"]);network.eval()
    def data(split):
        x,y,base=[],[],[]
        for line in (ROOT/f"data/prepared/uadfv/{split}.jsonl").read_text().splitlines():
            row=json.loads(line);frames,_=decode_video(row["path"])
            x.append(describe(frames));y.append(row["label"])
            if split=="val":
                with torch.inference_mode():base.append(float((network(frames[None])/blob["temperature"]).sigmoid().item()))
        return np.stack(x),np.array(y),np.array(base)
    x,y,_=data("train");v,vy,p=data("val")
    benchmark=metrics(vy,p)["macro_f1"]
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight="balanced",random_state=42)).fit(x,y)
    score=metrics(vy,clf.predict_proba(v)[:,1])["macro_f1"]
    print("Validation macro F1",score,"CNN",benchmark,flush=True)
    (folder/"validation_comparison.json").write_text(json.dumps({"metric":"validation macro F1","cnn":benchmark,"temporal_residual":score,"validation_clips":len(vy)},indent=2))
    if score>benchmark:
        t,ty,_=data("test")
        joblib.dump(clf,folder/"model.joblib")
        save_report("video",{"dataset":original["dataset"],"architecture":"Temporal residual features and regularized logistic regression",
            "artifact_format":"joblib","counts":original["counts"],"test":metrics(ty,clf.predict_proba(t)[:,1]),
            "cnn_baseline_test":original["test"],"validation_macro_f1":{"cnn":benchmark,"temporal":score},
            "evaluation_scope":original["evaluation_scope"],"limitations":original["limitations"]})


if __name__=="__main__":run()
