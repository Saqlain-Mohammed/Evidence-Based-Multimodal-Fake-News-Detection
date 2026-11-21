"""Import acquired real media with provenance and reusable preprocessing caches."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

def wait_file(path,wait):
    while not path.exists():
        if not wait:raise FileNotFoundError(path)
        print('Waiting for acquisition:',path.name,flush=True);time.sleep(30)

def evaluation_spectrograms(batch):
    import soundfile as sf
    import torch
    waveforms = []
    for row in batch:
        samples, sr = sf.read(io.BytesIO(row['audio']['bytes']), frames=96000, always_2d=True, dtype='float32')
        if sr != 16000 or not len(samples):
            raise ValueError('Unexpected ASVspoof evaluation audio format.')
        waveforms.append(np.pad(samples.mean(1), (0, 96000-len(samples))))
    x = torch.from_numpy(np.stack(waveforms))
    x = torch.stft(x, n_fft=512, hop_length=160, window=torch.hann_window(512), return_complex=True).abs().log1p()
    x = (x-x.mean(dim=(1,2),keepdim=True))/(x.std(dim=(1,2),keepdim=True)+1e-6)
    return torch.nn.functional.interpolate(x.unsqueeze(1), size=(128,192), mode='bilinear', align_corners=False).numpy()


def audio(wait):
    from workbench import decode_audio
    import torch
    torch.set_num_threads(2)
    folder=ROOT/'data/prepared/asvspoof';folder.mkdir(parents=True,exist_ok=True)
    speakers={};hashes={}
    for source,split in [('train','train'),('validation','val'),('test','test')]:
        dest=folder/f'{split}.jsonl'
        if dest.exists():
            rows=[json.loads(l) for l in dest.read_text().splitlines()]
            for r in rows:speakers[r['group']]=split;hashes[r['sha256']]=split
            print('Already imported',split,len(rows),flush=True);continue
        path=ROOT/f'data/public/asvspoof/{source}.parquet';wait_file(path,wait)
        parquet=pq.ParquetFile(path);n=parquet.metadata.num_rows
        media=folder/'flac'/split;media.mkdir(parents=True,exist_ok=True)
        tensor_path=folder/f'{split}-spectrograms.npy'
        cache=np.lib.format.open_memmap(tensor_path,mode='w+',dtype=np.float16,shape=(n,1,128,192))
        rows=[]
        for batch in parquet.iter_batches(batch_size=32 if split=='test' else 128):
            batch_rows = batch.to_pylist()
            evaluation = evaluation_spectrograms(batch_rows) if split=='test' else None
            for offset,row in enumerate(batch_rows):
                raw=row['audio']['bytes'];digest=hashlib.sha256(raw).hexdigest();speaker=row['speaker_id']
                if speaker in speakers and speakers[speaker]!=split:raise ValueError('Official speaker split overlap.')
                if digest in hashes and hashes[digest]!=split:raise ValueError('Audio byte overlap across official splits.')
                speakers[speaker]=split;hashes[digest]=split
                index=len(rows)
                if evaluation is None:
                    path=media/(row['audio_file_name']+'.flac')
                    if not path.exists():path.write_bytes(raw)
                    x,_=decode_audio(path);cache[index]=x.numpy()
                else:
                    path=ROOT/f'data/public/asvspoof/{source}.parquet'
                    cache[index]=evaluation[offset]
                rows.append({'path':str(path),'label':int(row['key']),'group':speaker,'sha256':digest,'split':split,
                    'source_row':index,'utterance':row['audio_file_name'],'attack':row['system_id'],'tensor_path':str(tensor_path),'tensor_index':index})
            if len(rows)%1024==0:print('Imported audio',split,len(rows),'/',n,flush=True)
        cache.flush();del cache
        temporary=dest.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        temporary.replace(dest)
        print('Audio ready',split,len(rows),flush=True)
    (folder/'provenance.json').write_text(json.dumps({'source':'Bisher/ASVspoof_2019_LA at aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c',
        'partitions':'official train/development/evaluation','speaker_overlap':0,'byte_overlap':0,
        'preprocessing':'first 6 seconds at 16k mono; whitened log-STFT 512/160 resized 128x192; no pretrained encoder'},indent=2),encoding='utf-8')

def casia(wait):
    from training.prepare_media import prepare
    from workbench import image_tensor
    path=ROOT/'data/public/casia/all.parquet';wait_file(path,wait)
    folder=ROOT/'data/prepared/casia';folder.mkdir(parents=True,exist_ok=True)
    media=folder/'images';media.mkdir(exist_ok=True)
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def union(a,b):parents[find(a)]=find(b)
    rows=[];parquet=pq.ParquetFile(path)
    import csv
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            item=row['image'];filename=Path(item.get('path') or '').name
            if not filename or not filename.lower().startswith(('au_','tp_')):
                raise ValueError('CASIA mirror lacks source filenames; cannot establish donor/host groups.')
            lower=filename.lower();raw=item['bytes'];label=int(lower.startswith('tp_'))
            mask=np.asarray(Image.open(io.BytesIO(row['mask']['bytes'])).convert('L'))
            if bool(mask.max()>0)!=bool(label):raise ValueError('Filename class disagrees with tamper mask.')
            if label:
                ids=re.findall(r'[a-z]{3}\d{5}',lower)
                if len(ids)!=2:raise ValueError('CASIA donor/host identifiers missing: '+filename)
                union(ids[0],ids[1]);source=ids[0]
            else:
                match=re.search(r'au_([a-z]{3})_(\d{5})',lower)
                if not match:raise ValueError('Unexpected authentic source ID: '+filename)
                source=match[1]+match[2];find(source)
            imagepath=media/filename
            if not imagepath.exists():imagepath.write_bytes(raw)
            # Reject bad images before recording a protocol.
            with Image.open(imagepath) as im:im.verify()
            rows.append({'path':str(imagepath),'label':label,'source':source})
        if len(rows)%1024==0:print('Imported CASIA',len(rows),flush=True)
    protocol=folder/'provider.csv'
    with protocol.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['path','label','group']);writer.writeheader()
        for row in rows:writer.writerow({'path':row['path'],'label':row['label'],'group':find(row['source'])})
    prepare(protocol,folder)
    print('CASIA source-connected groups:',len({find(r['source']) for r in rows}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['audio','casia'],required=True);ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    (audio if a.task=='audio' else casia)(a.wait)
