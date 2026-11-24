"""Learn partial speech edits using the provider's 160 ms segment annotations."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import zipfile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.splice_features import windows,clip_score
from training.train_public import metrics,save_report


class LabelReader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('collections','defaultdict'):collections.defaultdict,('builtins','list'):list}
        if (module,name) not in allowed:raise ValueError(f'Unexpected class in pinned label file: {module}.{name}')
        return allowed[module,name]


def read_split(split, hashes):
    folder=ROOT/f"data/public/partialspoof_{split}"
    while not (folder/'provenance.json').exists():
        print("Waiting for PartialSpoof",split,flush=True);time.sleep(45)
    paths=folder/f"{split}_seglab_0.16.npy"
    with paths.open("rb") as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError("Unexpected label file version")
        shape,_,dtype=np.lib.format.read_array_header_1_0(f)
        if shape!=() or dtype!=np.dtype("O"):raise ValueError("Unexpected label array")
        annotations=LabelReader(f).load().item()
    with (folder/f"{split}_labels.csv").open(newline='') as f:
        labels={r['utt_id']:int(r['label_str']=='spoof') for r in csv.DictReader(f)}
    x=[];y=[];lengths=[];truth=[];skipped=0
    for name in ('bonafide','spoofed'):
        with zipfile.ZipFile(folder/(name+'.zip')) as z:
            for member in z.infolist():
                key=Path(member.filename).stem
                if key not in labels:continue
                raw=z.read(member);digest=hashlib.sha256(raw).digest()
                if digest in hashes:
                    skipped+=1;continue
                frames,_=windows(raw)
                mask=np.asarray(annotations[key]).ravel().astype(str)
                if not np.isin(mask,['0','1']).all():raise ValueError('Unexpected frame label.')
                if abs(len(frames)-len(mask))>2:skipped+=1;continue
                n=min(len(frames),len(mask))
                if n==0:continue
                mask=mask[:n]=='0'
                hashes.add(digest);x.append(frames[:n]);y.append(mask)
                lengths.append(n);truth.append(labels[key])
    if not x:raise ValueError("No labelled audio found")
    print(split,len(x),"recordings",sum(lengths),"windows",skipped,"omitted",flush=True)
    return np.concatenate(x),np.concatenate(y),lengths,np.array(truth),skipped


def aggregate(scores,lengths):
    chunks=np.split(scores,np.cumsum(lengths)[:-1])
    return np.array([clip_score(values) for values in chunks])


def run():
    hashes=set()
    train,y,counts,_,sk1=read_split('train',hashes)
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,class_weight='balanced',random_state=42)
    model.fit(train,y);del train
    val,vy,vc,vt,sk2=read_split('dev',hashes)
    vs=model.predict_proba(val)[:,1]
    threshold=max(np.arange(.15,.86,.05),key=lambda cut:f1_score(vy,vs>=cut))
    cal=LogisticRegression().fit(aggregate(vs,vc).reshape(-1,1),vt);del val
    test,ty,tc,tt,sk3=read_split('eval',hashes)
    ps=model.predict_proba(test)[:,1]
    clip=cal.predict_proba(aggregate(ps,tc).reshape(-1,1))[:,1]
    folder=save_report('audio_edits',{'dataset':'PartialSpoof v1.2 sampled subset','architecture':'160 ms spectral and transition features with gradient boosting',
        'counts':{'train':len(counts),'validation':len(vc),'test':len(tc)},'test':metrics(tt,clip),
        'segment_f1':float(f1_score(ty,ps>=threshold)),'segment_threshold':float(threshold),
        'evaluated_windows':len(ty),'omitted_duplicates_or_misaligned':{'train':sk1,'validation':sk2,'test':sk3},
        'evaluation_scope':'supplied train/dev/eval partitions; 160 ms frame annotations',
        'limitations':'Synthetic speech segments inserted into real speech. Estimates candidate intervals; does not cover every edit, ordinary cuts, music, or all recording conditions.'})
    joblib.dump({'classifier':model,'calibration':cal,'threshold':threshold},folder/'model.joblib',compress=3)
    provenance={s:json.loads((ROOT/f'data/public/partialspoof_{s}/provenance.json').read_text()) for s in ('train','dev','eval')}
    (folder/'dataset_provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=3):run()
