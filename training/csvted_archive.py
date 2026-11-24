"""Read pinned CSVTED ZIP members with bounded, verified HTTP range requests."""
import io
import json
from pathlib import Path
import time
import zipfile
import requests

REPO="CardiffVisualComputing/CSVTED"
REVISION="4ea2fdeb792c69d100367a61d1190f0e1cd1587e"
BASE=f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"


class RemoteArchive(io.RawIOBase):
    def __init__(self,name,size):
        self.url=BASE+name
        self.size=size
        self.pos=0

    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=max(0,offset+(self.pos if whence==1 else self.size if whence==2 else 0))
        return self.pos

    def read(self,n=-1):
        end=min(self.size,self.pos+n) if n>=0 else self.size
        if end<=self.pos:return b""
        start=self.pos
        for attempt in range(4):
            try:
                response=requests.get(self.url,headers={"Range":f"bytes={start}-{end-1}"},timeout=(10,60))
                response.raise_for_status()
                expected=f"bytes {start}-{end-1}/{self.size}"
                if response.headers.get("Content-Range")!=expected or len(response.content)!=end-start:
                    raise ValueError("Dataset range mismatch")
                self.pos=end
                return response.content
            except requests.RequestException:
                if attempt==3:raise
                time.sleep(attempt+1)


def sizes():
    response=requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}",timeout=15)
    response.raise_for_status()
    return {row['path']:row['size'] for row in response.json()}
import concurrent.futures
import csv
import hashlib
import io
from pathlib import PurePosixPath
import re
import struct
import zlib


def acquire():
    root=Path(__file__).resolve().parents[1]/"data/public/csvted"
    root.mkdir(parents=True,exist_ok=True)
    archive_sizes=sizes()
    tables={}
    for name in ("Original.zip","Frame-Deletion.zip"):
        with zipfile.ZipFile(RemoteArchive(name,archive_sizes[name])) as archive:
            tables[name]={PurePosixPath(i.filename).stem.lower():i for i in archive.infolist() if not i.is_dir()}
    response=requests.get(BASE+"Deletion.csv",timeout=30);response.raise_for_status()
    annotations={r["Video_Name"].lower():int(r["Start of Tamper"]) for r in csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))}
    items=[];scenes=set()
    originals=tables["Original.zip"]
    for name,info in tables["Frame-Deletion.zip"].items():
        match=re.fullmatch(r"\d+_del_\d+_\d+_(.+)",name)
        if not match or name not in annotations:continue
        scene=match[1]
        if scene not in originals or max(originals[scene].file_size,info.file_size)>30*1024**2:continue
        if scene not in scenes:items.append(("Original.zip",originals[scene],scene,None))
        scenes.add(scene);items.append(("Frame-Deletion.zip",info,scene,annotations[name]))
    def fetch(item):
        name,info,scene,boundary=item
        filename=PurePosixPath(info.filename).name
        destination=root/filename
        if not destination.exists():
            archive=RemoteArchive(name,archive_sizes[name])
            archive.seek(info.header_offset)
            packet=archive.read(30+4096+info.compress_size)
            if packet[:4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP member header')
            n,extra=struct.unpack_from('<HH',packet,26)
            start=30+n+extra
            if start>4126:raise ValueError('Unexpected ZIP header length')
            data=packet[start:start+info.compress_size]
            if info.compress_type==zipfile.ZIP_DEFLATED:data=zlib.decompress(data,-15)
            elif info.compress_type!=zipfile.ZIP_STORED:raise ValueError('Unsupported ZIP compression')
            if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:raise ValueError('ZIP member integrity mismatch')
            tmp=destination.with_suffix('.partial');tmp.write_bytes(data);tmp.replace(destination)
        data=destination.read_bytes()
        if zlib.crc32(data)!=info.CRC:raise ValueError('Existing video CRC mismatch')
        return dict(file=filename,group=scene,removed_start_frame=boundary,archive=name,member=info.filename,sha256=hashlib.sha256(data).hexdigest())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows=[]
        for r in pool.map(fetch,items):
            rows.append(r)
            if len(rows)%20==0:print("CSVTED",len(rows),"/",len(items),flush=True)
    (root/"manifest.json").write_text(json.dumps({"source":BASE,"selection":"All exact-name original/deletion pairs fitting 30 MiB each, annotated in Deletion.csv.","records":rows},indent=2))
    print("Acquired",len(rows),"videos from",len(scenes),"source scenes",flush=True)


if __name__=="__main__":acquire()
