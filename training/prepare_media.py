"""Create disjoint media manifests from provider protocols, never filenames.

Input CSV columns: path,label,group,split (split is optional).
label: 0=authentic, 1=manipulated/spoof. group must identify the original
image/clip/person/speaker and ALL derivatives; never use the row ID as group.
Preserve official train/val/test assignments where supplied.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def prepare(source, out):
    source=Path(source).resolve(); out=Path(out).resolve()
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    if not rows: raise ValueError('Empty protocol.')
    seen={}; cleaned=[]
    for row in rows:
        if not row.get('group'): raise ValueError('Provider source/identity/speaker group is required.')
        p=Path(row['path'])
        if not p.is_absolute(): p=source.parent/p
        if not p.is_file(): raise ValueError(f'Media missing: {p}')
        label=int(row['label'])
        if label not in (0,1): raise ValueError('Labels must be 0=authentic, 1=risk.')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        digest=h.hexdigest()
        record={'path':str(p.resolve()),'label':label,'group':row['group'],'sha256':digest,'split':row.get('split','')}
        if digest in seen:
            prev=seen[digest]
            if prev['label']!=label or (prev['split'] and record['split'] and prev['split']!=record['split']):
                raise ValueError('Identical media has conflicting labels or official splits.')
            continue
        seen[digest]=record; cleaned.append(record)
    official=any(r['split'] for r in cleaned)
    if official:
        aliases={'dev':'val','validation':'val','eval':'test'}
        for r in cleaned:
            r['split']=aliases.get(r['split'],r['split'])
            if r['split'] not in ('train','val','test'): raise ValueError('Every official row needs train/val/test.')
    else:
        groups=np.array([r['group'] for r in cleaned]); index=np.arange(len(cleaned))
        train,rest=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(index,groups=groups))
        va,te=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=43).split(rest,groups=groups[rest]))
        for name,indices in [('train',train),('val',rest[va]),('test',rest[te])]:
            for i in indices: cleaned[i]['split']=name
    assigned={}
    for r in cleaned:
        prior=assigned.setdefault(r['group'],r['split'])
        if prior!=r['split']: raise ValueError(f"Group leakage across splits: {r['group']}")
    out.mkdir(parents=True,exist_ok=True)
    counts={}
    for name in ('train','val','test'):
        part=[r for r in cleaned if r['split']==name]
        if len({r['label'] for r in part})!=2: raise ValueError(f'{name} must contain both classes.')
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in part),encoding='utf-8')
        counts[name]=dict(Counter(r['label'] for r in part))
    (out/'provenance.json').write_text(json.dumps({'source_protocol':str(source),'protocol_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'official_splits':official,'counts':counts,'group_overlap':0,'deduplicated':len(rows)-len(cleaned)},indent=2),encoding='utf-8')
    print(counts)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--out',required=True); a=ap.parse_args();prepare(a.csv,a.out)
