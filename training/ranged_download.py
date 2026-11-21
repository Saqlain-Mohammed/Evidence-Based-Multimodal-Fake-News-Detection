"""Resumable range downloads for large public archives on buffering proxies.

Each completed 8 MiB range is persisted. Exact Content-Range validation prevents
silently writing an HTML error or a cached wrong chunk into a dataset archive.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import requests
from download_public import ROOT,SOURCES

def transfer(url,path,workers=4):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return
    with requests.get(url,headers={'Range':'bytes=0-0'},timeout=(15,30),stream=True) as r:
        r.raise_for_status();match=re.fullmatch(r'bytes 0-0/(\d+)',r.headers.get('Content-Range',''))
        if not match:raise ValueError('Server must support verified byte ranges.')
        total=int(match[1])
    block=8*1024*1024;count=(total+block-1)//block
    part=path.with_suffix(path.suffix+'.ranges');journal=path.with_suffix(path.suffix+'.ranges.json')
    done=set(json.loads(journal.read_text()) if journal.exists() and part.exists() else [])
    if not part.exists():part.touch()
    def fetch(index):
        start=index*block;end=min(total-1,start+block-1)
        for retry in range(4):
            try:
                r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,60))
                r.raise_for_status()
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{total}' or len(r.content)!=end-start+1:
                    raise ValueError('Range response differs from requested bytes.')
                return index,r.content
            except (requests.RequestException,ValueError):
                if retry==3:raise
                time.sleep(min(4,retry+1))
    print(f'Fetching {path.name}: {total/1e6:.1f} MB; {len(done)}/{count} ranges already saved',flush=True)
    with part.open('r+b') as f,ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,i) for i in range(count) if i not in done]
        for future in as_completed(futures):
            index,data=future.result();f.seek(index*block);f.write(data);f.flush();done.add(index)
            journal.write_text(json.dumps(sorted(done)))
            if len(done)%8==0 or len(done)==count:print(f'{path.name}: {len(done)}/{count} ranges ({len(done)*100/count:.0f}%)',flush=True)
    if part.stat().st_size!=total:raise ValueError('Final size mismatch.')
    part.replace(path)

def dataset(name):
    folder=ROOT/'data/public'/name;records=[]
    for filename,url in SOURCES[name].items():
        path=folder/filename;transfer(url,path)
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        records.append({'file':filename,'url':url,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        print('Ready',name,filename,flush=True)
    (folder/'provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('datasets',nargs='+',choices=list(SOURCES));a=ap.parse_args()
    for name in a.datasets:dataset(name)
