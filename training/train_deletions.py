"""Independent temporal video model trained on annotated CSVTED frame deletions."""
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.deletion_features import transitions,candidates
from training.train_public import metrics,save_report


def run():
    folder=ROOT/"data/public/csvted"
    while not (folder/"manifest.json").exists():time.sleep(15)
    manifest=json.loads((folder/"manifest.json").read_text())
    rows=manifest["records"]
    groups=sorted({r["group"] for r in rows})
    rng=np.random.default_rng(42);rng.shuffle(groups)
    a,b=int(len(groups)*.6),int(len(groups)*.8)
    assignments={g:split for split,part in (('train',groups[:a]),('val',groups[a:b]),('test',groups[b:])) for g in part}
    parts={s:[] for s in ('train','val','test')}
    for row in rows:
        x,fps=transitions(folder/row['file'])
        start=row['removed_start_frame']
        # Feature index zero measures frame 0 -> frame 1. Provider frames are 1-based.
        boundary=None if start is None else start-2
        if boundary is not None and not 0<=boundary<len(x):continue
        y=np.zeros(len(x),dtype=int)
        if boundary is not None:y[max(0,boundary-1):boundary+2]=1
        parts[assignments[row['group']]].append((x,y,boundary,fps))
    train=parts['train']
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,class_weight='balanced',random_state=42)
    model.fit(np.concatenate([r[0] for r in train]),np.concatenate([r[1] for r in train]))
    def predict(part):
        probs=[model.predict_proba(r[0])[:,1] for r in part]
        return probs,np.asarray([[p.max()] for p in probs]),np.asarray([r[2] is not None for r in part])
    dev=parts['val'];valid,vx,vy=predict(dev)
    labels=np.concatenate([r[1] for r in dev]);prob=np.concatenate(valid)
    cut=max(np.linspace(.1,.99,40),key=lambda t:f1_score(labels,prob>=t))
    cal=LogisticRegression().fit(vx,vy)
    test=parts['test'];scores,tx,ty=predict(test)
    found=0;predicted=0;forged=0
    for (_,_,boundary,fps),ps in zip(test,scores):
        times=candidates(ps,cut,fps);predicted+=len(times)
        if boundary is not None:
            forged+=1
            found+=int(any(abs(t*fps-1-boundary)<=2.1 for t in times))
    folder=save_report('video_edits',{'dataset':'CSVTED matched source clips and frame deletions','architecture':'Local frame discontinuities and gradient boosting',
        'counts':{'train':len(train),'validation':len(dev),'test':len(test)},'test':metrics(ty,cal.predict_proba(tx)[:,1]),
        'localization':{'within_two_frames':found,'edited_test_clips':forged,'candidate_count':predicted,'candidate_threshold':float(cut)},
        'evaluation_scope':'source-scene-disjoint internal 60/20/20 split','limitations':'Exploratory frame-deletion detector. Natural cuts can trigger false positives. Checks the first 120 seconds or 3,600 frames. Clip-level and boundary localization results are measured separately.'})
    joblib.dump({'classifier':model,'calibration':cal,'threshold':cut},folder/'model.joblib',compress=3)
    (folder/'dataset_provenance.json').write_text(json.dumps(dict(manifest,source_groups=assignments),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
