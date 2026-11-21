"""Pair all real/fake UADFV source clips for a small historical video baseline.

Not a modern deepfake benchmark. Source-video groups prevent clip derivatives
crossing partitions. Inserted identities may still overlap across the corpus.
"""
import csv
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.prepare_media import prepare

def main():
    folder=ROOT/'data/prepared/uadfv';folder.mkdir(parents=True,exist_ok=True)
    rows=[];counts={}
    for label,name in enumerate(['real','fake']):
        archive=ROOT/f'data/public/uadfv/{name}.zip'
        if not archive.exists():raise FileNotFoundError(archive)
        dest=folder/name;dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.is_dir() or not entry.filename.lower().endswith(('.mp4','.avi','.mov')):continue
                filename=Path(entry.filename).name
                if '__MACOSX' in Path(entry.filename).parts or filename.startswith('._'):continue
                path=dest/filename
                if not path.exists():
                    with z.open(entry) as source,path.open('wb') as target:
                        import shutil
                        shutil.copyfileobj(source,target)
                key=path.stem.lower().replace('fake','').replace('real','').strip('_- ')
                rows.append({'path':str(path),'label':label,'group':key});counts[name]=counts.get(name,0)+1
    real={r['group'] for r in rows if r['label']==0};fake={r['group'] for r in rows if r['label']==1}
    if real!=fake:raise ValueError('Cannot pair original and fake UADFV source IDs. Inspect archive names before training.')
    protocol=folder/'provider.csv'
    with protocol.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','label','group']);writer.writeheader();writer.writerows(rows)
    prepare(protocol,folder)
    print('UADFV paired:',counts,'source groups:',len(real),flush=True)

if __name__=='__main__':main()
