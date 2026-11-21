"""Train a separate scratch checkpoint for image manipulation, video or audio.

Use provider data converted by prepare_media.py. Never trains on demo images.
Example: python training/train_media.py --task audio --data data/asvspoof --epochs 15
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from training.scratch_models import ImageCNN,VideoCNN,AudioCNN
from training.train_public import metrics,save_report
from workbench import decode_audio,decode_video

from functools import lru_cache

@lru_cache(maxsize=128)
def video_frames(path):
    # Reuse uniformly sampled frames across epochs. The bound also suits larger corpora.
    frames, _ = decode_video(path)
    return frames

class Media(Dataset):
    def __init__(self,rows,task,train=False):
        self.rows,self.task,self.train=rows,task,train
        self.cache={}
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];p=r['path']
        if r.get('tensor_path'):
            cachepath=r['tensor_path']
            if cachepath not in self.cache:self.cache[cachepath]=np.load(cachepath,mmap_mode='r')
            x=torch.from_numpy(np.array(self.cache[cachepath][r['tensor_index']],dtype=np.float32))
        elif self.task=='audio': x,_=decode_audio(p)
        elif self.task=='video': x=video_frames(p)
        else:
            with Image.open(p) as im: a=np.asarray(im.convert('RGB').resize((64,64))).copy()
            x=torch.from_numpy(a).permute(2,0,1).float()/255
        if self.train and self.task!='audio' and torch.rand(1)<.5: x=x.flip(-1)
        return x,torch.tensor(r['label'],dtype=torch.float32)

def train(a):
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(6)
    data=Path(a.data);parts={name:[json.loads(l) for l in (data/f'{name}.jsonl').read_text(encoding='utf-8').splitlines() if l] for name in ('train','val','test')}
    for left,right in [('train','val'),('train','test'),('val','test')]:
        for field in ('group','sha256'):
            if set(r[field] for r in parts[left])&set(r[field] for r in parts[right]): raise ValueError(f'{field} leakage: {left}/{right}')
    for name,rows in parts.items():
        if len({r['label'] for r in rows})!=2: raise ValueError(f'{name} needs both classes.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=({'image_manipulation':ImageCNN,'video':VideoCNN,'audio':AudioCNN}[a.task])().to(device)
    loaders={name:DataLoader(Media(rows,a.task,name=='train'),batch_size=a.batch,shuffle=name=='train',num_workers=0) for name,rows in parts.items()}
    positive=sum(r['label'] for r in parts['train']);negative=len(parts['train'])-positive
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/positive,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    def predict(loader):
        model.eval();ys=[];logits=[]
        with torch.inference_mode():
            for x,y in loader: logits.extend(model(x.to(device)).cpu().tolist());ys.extend(y.tolist())
        return np.array(ys),np.array(logits)
    best=-1;wait=0;state=None
    for epoch in range(a.epochs):
        model.train();total=0
        for x,y in loaders['train']:
            opt.zero_grad();loss=criterion(model(x.to(device)),y.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()
        y,z=predict(loaders['val']);p=1/(1+np.exp(-np.clip(z,-30,30)));score=metrics(y,p)['macro_f1']
        print(f'epoch={epoch+1} loss={total/len(loaders["train"]):.4f} val_f1={score:.4f}',flush=True)
        if score>best: best=score;wait=0;state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait>=4: break
    model.load_state_dict(state)
    y,z=predict(loaders['val']);z=torch.tensor(z,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32)
    log_t=torch.nn.Parameter(torch.zeros(()));cal=torch.optim.LBFGS([log_t],max_iter=40)
    def closure():
        cal.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(z/log_t.exp(),y);loss.backward();return loss
    cal.step(closure);temperature=float(log_t.exp().detach().clamp(.1,10))
    y,z=predict(loaders['test']);p=1/(1+np.exp(-np.clip(z/temperature,-30,30)));result=metrics(y,p)
    if a.task=='audio':
        from sklearn.metrics import roc_curve
        fpr,tpr,_=roc_curve(y,p);fnr=1-tpr;idx=np.argmin(np.abs(fpr-fnr));result['eer']=float((fpr[idx]+fnr[idx])/2)
    folder=save_report(a.task,{'dataset':a.dataset_name,'architecture':type(model).__name__,'epochs':epoch+1,
        'counts':{'train':len(parts['train']),'validation':len(parts['val']),'test':len(parts['test'])},'test':result,
        'temperature':temperature,'evaluation_scope':a.evaluation_scope,
        'limitations':a.limitations or 'Binary classification only; no validated tamper localization, clone attribution, or editing subtype. Evaluate unseen sources and attacks before deployment.'})
    torch.save({'state_dict':state,'task':a.task,'positive_label':1,'size':64,'frames':16,'temperature':temperature},folder/'model.pt')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['image_manipulation','video','audio'],required=True);ap.add_argument('--data',required=True);ap.add_argument('--dataset-name',required=True);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--lr',type=float,default=.0005);ap.add_argument('--evaluation-scope',default='provider/group-disjoint');ap.add_argument('--limitations',default='');train(ap.parse_args())
