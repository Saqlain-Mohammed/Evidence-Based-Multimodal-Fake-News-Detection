"""Honest, modality-specific research workbench API. Legacy demo heads are excluded."""
from accounts import OWNER
import asyncio
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import ipaddress
import json
from pathlib import Path
import re
import socket
import sqlite3
import tempfile
import time
from urllib.parse import urlparse, quote
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
import requests

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
router = APIRouter(prefix="/api/v2")
JOBLIB_TASKS = {'fake_news', 'ai_text', 'paired_news'}
TASKS = {
    'paired_news': ('Title and image context', 'Fakeddit paired subset', 'text + image'),
    'fake_news': ('News credibility', 'WELFake + LIAR cross-domain evaluation', 'text'),
    'ai_text': ('AI text', 'HC3 English', 'text'),
    'image_ai': ('AI image', 'CIFAKE', 'image'),
    'image_manipulation': ('Image manipulation', 'CASIA v2 / IMD2020', 'image'),
    'video': ('Deepfake video', 'FaceForensics++ / Celeb-DF v2', 'video'),
    'audio': ('Synthetic audio', 'ASVspoof 2019 LA / 2021 DF', 'audio'),
}
MAX_BYTES = 30 * 1024 * 1024
MEDIA_LOCK = threading.Lock()
ANALYSIS_SLOTS = asyncio.Semaphore(2)
DB = ROOT / 'data' / 'workbench.sqlite3'

def connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute('CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, created TEXT, mode TEXT, title TEXT, result TEXT)')
    c.execute("CREATE TABLE IF NOT EXISTS report_owners(id TEXT PRIMARY KEY, owner TEXT NOT NULL)")
    return c

def uses_joblib(task):
    if task in {'audio','image_manipulation'}:
        path = ROOT/'models/public'/task/'report.json'
        return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('artifact_format') == 'joblib'
    return task in JOBLIB_TASKS


def registry():
    rows = []
    for key, (name, data, modality) in TASKS.items():
        folder = ROOT / 'models/public' / key
        artifact = folder / ('model.joblib' if uses_joblib(key) else 'model.pt')
        report = None
        if (folder / 'report.json').exists():
            report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
        active = artifact.exists() and report is not None
        cross = folder / 'cross_domain.json'
        rows.append({'id': key, 'name': name, 'modality': modality, 'dataset': report.get('dataset',data) if active else data,
            'status': 'ready' if active else 'training_required', 'report': report if active else None,
            'cross_domain': json.loads(cross.read_text(encoding='utf-8')) if active and cross.exists() else None})
    return rows

@lru_cache(maxsize=16)
def load_artifact(task, modified):
    folder = ROOT / 'models/public' / task
    if uses_joblib(task):
        import joblib
        return joblib.load(folder / 'model.joblib')
    import torch
    from training.scratch_models import ImageCNN, AudioCNN, VideoCNN
    torch.set_num_threads(4)
    blob = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    cls = AudioCNN if task == 'audio' else VideoCNN if task == 'video' else ImageCNN
    model = cls(); model.load_state_dict(blob['state_dict']); model.eval()
    return model, blob

def artifact(task):
    folder = ROOT / 'models/public' / task
    p = folder / ('model.joblib' if uses_joblib(task) else 'model.pt')
    if not p.exists() or not (folder / 'report.json').exists():
        return None
    return load_artifact(task, p.stat().st_mtime_ns)

def unavailable(task, reason=None):
    return {'id':task, 'name':TASKS[task][0], 'status':'unavailable', 'probability':None,
        'verdict':'Training required', 'explanation':reason or f'No real-data checkpoint for {TASKS[task][0]}. Run the separate training pipeline with the documented dataset.'}

def signal(task, probability, explanation, **extra):
    p = float(probability)
    verdict = 'Higher risk' if p >= .65 else 'Lower risk' if p <= .35 else 'Inconclusive'
    return {'id':task, 'name':TASKS[task][0], 'status':'analyzed', 'probability':round(p,4),
        'verdict':verdict, 'explanation':explanation, **extra}

def text_signal(task, text):
    blob = artifact(task)
    if blob is None: return unavailable(task)
    minimum = 80 if task == 'fake_news' else 20
    if len(text.split()) < minimum:
        return {'id':task,'name':TASKS[task][0],'status':'insufficient_input','probability':None,
            'verdict':'More text needed','explanation':f'Provide at least {minimum} words. This model was trained on {"articles, not short factual claims" if task == "fake_news" else "passages"}.'}
    model = blob['model']
    p = model.predict_proba([text])[0][list(model.classes_).index(1)]
    note = ('A learned article-pattern score. It does not prove that a claim is false.' if task=='fake_news' else
        'A learned writing-pattern score from HC3. AI authorship is uncertain; this is not evidence of misconduct.')
    return signal(task, p, note)

def image_tensor(raw, size):
    import torch
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        a = np.asarray(im.convert('RGB').resize((size,size))).copy()
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).float()/255

def image_signal(task, raw):
    # Grad-CAM installs hooks on the shared model. Serialize to keep requests isolated.
    with MEDIA_LOCK:
        return _image_signal(task, raw)

def _image_signal(task, raw):
    loaded=artifact(task)
    if loaded is None: return unavailable(task)
    import torch
    if isinstance(loaded,dict):
        from training.forensic_features import features
        decision=loaded['classifier'].decision_function(features(raw).reshape(1,-1)).reshape(-1,1)
        p=loaded['calibration'].predict_proba(decision)[0,1]
        return signal(task,p,'Experimental CASIA compression and residual score. Historical file processing can influence this result.',verdict='Experimental score',**mask_overlay(task,raw))
    model, blob = loaded
    x = image_tensor(raw,blob.get('size',32))
    with torch.inference_mode():
        p = (model(x)/blob.get('temperature',1.)).sigmoid().item()
    # Grad-CAM of the positive-class logit, never labelled a tamper mask.
    capture={}
    def hook(module, inputs, output):
        capture['features']=output
        output.register_hook(lambda grad: capture.update(grad=grad))
    handle=model.features.register_forward_hook(hook)
    try:
        model.zero_grad(); model(x.requires_grad_(True)).sum().backward()
        f,g=capture['features'],capture['grad']
        cam=(f*g.mean((2,3),keepdim=True)).sum(1).relu()[0].detach().numpy()
        cam=cam/(cam.max()+1e-8)
        original=Image.open(io.BytesIO(raw)).convert('RGB'); original.thumbnail((640,640))
        heat=np.asarray(Image.fromarray(np.uint8(cam*255)).resize(original.size))/255
        pixels=np.asarray(original).astype(float)
        overlay=np.stack([np.full_like(heat,245),170*(1-heat),30*(1-heat)],axis=-1)
        alpha=heat[...,None]*.5
        output=Image.fromarray(np.uint8(pixels*(1-alpha)+overlay*alpha)); buf=io.BytesIO(); output.save(buf,format='PNG')
        encoded='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
    finally:
        handle.remove(); model.zero_grad()
    localization = mask_overlay(task, raw)
    return signal(task,p,'Model attention is shown below. It is not a localized proof of manipulation. CIFAKE AI detection is limited to its historical 32×32 benchmark.' if task=='image_ai' else 'Binary manipulation risk; attention is not a forensic tamper mask.',
        heatmap=encoded, heatmap_label='Positive-class Grad-CAM attention',**({'verdict':'Experimental score'} if task=='image_manipulation' else {}),**localization)

def mask_overlay(task, raw):
    localization = {}
    mask_path = ROOT/'models/public/tamper_mask/model.joblib'
    if task == 'image_manipulation' and mask_path.exists():
        from training.mask_features import pixels as mask_pixels
        mask_model = load_pixel_model(mask_path.stat().st_mtime_ns)
        scores = mask_model['model'].predict_proba(mask_pixels(raw))[:,1]
        region = scores.reshape(64,64) >= mask_model['threshold']
        with Image.open(io.BytesIO(raw)) as im:
            display=im.convert('RGB')
        display.thumbnail((640,640))
        alpha = Image.fromarray(np.uint8(region)*130).resize(display.size,Image.Resampling.NEAREST)
        display.paste((210,60,65),(0,0,display.width,display.height),alpha)
        image_bytes=io.BytesIO();display.save(image_bytes,format='JPEG',quality=82)
        localization={'mask_overlay':'data:image/jpeg;base64,'+base64.b64encode(image_bytes.getvalue()).decode(),
                      'highlighted_fraction':float(region.mean()),
                      'mask_note':'Experimental mask trained on CASIA pixel annotations. Mean overlap with true regions was 4.66% on CASIA test images. Use only as an exploratory visualization.'}
    return localization


@lru_cache(maxsize=1)
def load_pixel_model(modified):
    import joblib
    return joblib.load(ROOT/'models/public/tamper_mask/model.joblib')


def paired_news_signal(text, raw):
    from training.paired_features import image_features, paired_features
    model = artifact('paired_news')
    if model is None:
        return unavailable('paired_news')
    title = re.split(r'(?<=[.!?])\s+|\n+', text.strip(), maxsplit=1)[0][:1000]
    features = paired_features(model, [title], [image_features(raw)])
    decision = model['classifier'].decision_function(features).reshape(-1, 1)
    score = model['calibration'].predict_proba(decision)[0, 1]
    return signal('paired_news', score, 'Joint title/image model trained on historical Reddit categories including satire. This estimates a dataset label, not factual truth.', compared_title=title)


def video_signal(path):
    loaded=artifact('video')
    if loaded is None: return unavailable('video')
    import torch
    model,blob=loaded
    frames,times=decode_video(path,blob.get('frames',16),blob.get('size',64))
    with torch.inference_mode():
        p=(model(frames.unsqueeze(0))/blob.get('temperature',1)).sigmoid().item()
    return signal('video',p,'Experimental video model. The UADFV baseline classified 5 of 10 held-out clips correctly. Sampled times describe coverage, not proven tampering timestamps.',sampled_seconds=times,verdict='Experimental score')

def decode_video(path, count=16, size=64):
    import cv2, torch
    cap=cv2.VideoCapture(str(path))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
        if n<2 or fps<=0: raise ValueError('Video has no decodable frame sequence.')
        positions=np.linspace(0,n-1,count,dtype=int); frames=[]; times=[]
        for index in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(index)); ok,frame=cap.read()
            if not ok: raise ValueError('Unable to decode the selected video frames.')
            frame=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(size,size))
            frames.append(torch.from_numpy(frame.copy()).permute(2,0,1).float()/255); times.append(round(float(index/fps),2))
        return torch.stack(frames),times
    finally: cap.release()

def decode_audio(path, seconds=6):
    import soundfile as sf, torch
    from scipy.signal import resample_poly
    from math import gcd
    info=sf.info(str(path))
    a,sr=sf.read(str(path),frames=min(info.frames,info.samplerate*seconds),dtype='float32',always_2d=True)
    a=a.mean(1)
    if not len(a): raise ValueError('Audio is empty.')
    if sr!=16000:
        factor=gcd(sr,16000); a=resample_poly(a,16000//factor,sr//factor)
    length=16000*seconds; a=np.pad(a[:length],(0,max(0,length-len(a))))
    t=torch.from_numpy(a.copy()); spec=torch.stft(t,n_fft=512,hop_length=160,window=torch.hann_window(512),return_complex=True).abs().log1p()
    spec=((spec-spec.mean())/(spec.std()+1e-6)).unsqueeze(0).unsqueeze(0)
    spec=torch.nn.functional.interpolate(spec,size=(128,192),mode='bilinear',align_corners=False)[0]
    return spec, min(info.duration,seconds)

def audio_signal(path):
    from transcription import transcript
    loaded=artifact('audio')
    if loaded is None: return unavailable('audio')
    import torch
    a,duration=decode_audio(path)
    if isinstance(loaded, dict):
        from training.audio_features import spectral_features
        features = spectral_features(a.unsqueeze(0).numpy())
        z = loaded['classifier'].decision_function(features).reshape(-1,1)
        p = loaded['calibration'].predict_proba(z)[0,1]
    else:
        model,blob=loaded
        with torch.inference_mode(): p=(model(a.unsqueeze(0))/blob.get('temperature',1)).sigmoid().item()
    report=json.loads((ROOT/'models/public/audio/report.json').read_text(encoding='utf-8'))
    scope=report.get('evaluation_scope','unknown')
    scope_note=' Development model: only an internal speaker split of the original training partition has been evaluated; official unseen-attack evaluation is pending.' if scope=='internal-training-partition-only' else ''
    return signal('audio',p,'Bona-fide versus spoof risk for the first six seconds. Binary training cannot identify a speaker, distinguish cloning from synthesis, or prove a splice.'+scope_note,analyzed_seconds=duration,evaluation_scope=scope,
        transcript=transcript(path))

def safe_url(url):
    p=urlparse(url)
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password:
        raise ValueError('Use a public http or https URL without credentials.')
    for entry in socket.getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(entry[4][0]).is_global:
            raise ValueError('Only public internet URLs are supported.')
    return url

def public_get(url, limit=2*1024*1024):
    for _ in range(4):
        safe_url(url)
        with requests.get(url,timeout=(5,8),stream=True,allow_redirects=False,headers={'User-Agent':'VeritasResearch/3.0'}) as r:
            if r.is_redirect:
                from urllib.parse import urljoin
                url=urljoin(url,r.headers['Location']); continue
            r.raise_for_status(); parts=[]; size=0
            for b in r.iter_content(65536):
                size+=len(b)
                if size>limit: raise ValueError('Remote document exceeds the size limit.')
                parts.append(b)
            return b''.join(parts),url
    raise ValueError('Too many URL redirects.')

def article(url):
    from bs4 import BeautifulSoup
    data,final=public_get(url)
    soup=BeautifulSoup(data,'html.parser')
    for t in soup(['script','style','nav','header','footer']): t.decompose()
    return (' '.join(p.get_text(' ',strip=True) for p in soup.select('article p, main p') or soup.find_all('p')))[:50000],final

@lru_cache(maxsize=1)
def load_stance_model(modified):
    import joblib
    return joblib.load(ROOT / 'models/public/evidence_stance/model.joblib')


def compare_claim(text, passages):
    """Compare the opening claim with one relevant sentence and return both."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from training.stance_features import LABELS, encode
    claim = re.split(r'(?<=[.!?])\s+|\n+', text.strip(), maxsplit=1)[0][:1000]
    sentences = [s.strip()[:1000] for s in passages if s.strip()][:160]
    path = ROOT / 'models/public/evidence_stance/model.joblib'
    if not sentences or not claim or not path.exists():
        return None
    try:
        vectors = TfidfVectorizer(stop_words='english').fit_transform([claim, *sentences])
    except ValueError:
        return None
    similarity = (vectors[1:] @ vectors[0].T).toarray().ravel()
    index = int(similarity.argmax())
    result = {'claim': claim, 'passage': sentences[index], 'assessment': 'Insufficient overlap'}
    if similarity[index] >= 0.12:
        model = load_stance_model(path.stat().st_mtime_ns)
        scores = model.predict_proba(encode([(sentences[index], claim)]))[0]
        result['assessment'] = LABELS[int(model.classes_[scores.argmax()])] if scores.max() >= .65 else 'Inconclusive'
    return result


def google_headlines(query):
    raw,_=public_get('https://news.google.com/rss/search?q='+quote(query)+'&hl=en&gl=US&ceid=US:en')
    return [{'title':row.findtext('title',''), 'url':row.findtext('link',''),'source':row.findtext('source',''),
             'published':row.findtext('pubDate',''),'kind':'Headline'}
            for row in ET.fromstring(raw).findall('./channel/item')[:6]]


def published_fact_checks(query):
    from bs4 import BeautifulSoup
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    terms=[word for word in re.findall(r"[a-zA-Z]{4,}",query.lower()) if word not in ENGLISH_STOP_WORDS]
    if not terms:return []
    url='https://www.factcheck.org/wp-json/wp/v2/posts?per_page=3&_fields=title,excerpt,link,date&search='+quote(max(terms[:8],key=len))
    raw,_=public_get(url)
    rows=[]
    for post in json.loads(raw):
        title=BeautifulSoup(post['title']['rendered'],'html.parser').get_text(' ',strip=True)
        excerpt=BeautifulSoup(post['excerpt']['rendered'],'html.parser').get_text(' ',strip=True)
        rows.append({'title':title,'passage':excerpt[:1200],'url':post['link'],'source':'FactCheck.org',
                     'published':post['date'],'kind':'Published fact-check excerpt'})
    return rows


def evidence(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    query=' '.join(re.findall(r"[\w'-]+",text)[:18])
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        searches=[pool.submit(provider,query) for provider in (google_headlines,published_fact_checks)]
        for search in searches:
            try: rows.extend(search.result())
            except (requests.RequestException,ValueError,KeyError,TypeError,ET.ParseError,OSError): continue
    if rows:
        documents=[r.get('passage') or r['title'].removesuffix(' - '+r['source']) for r in rows]
        try:
            vectors=TfidfVectorizer(stop_words='english').fit_transform([text[:3000],*documents])
            relevance=cosine_similarity(vectors[:1],vectors[1:])[0]
        except ValueError:
            relevance=np.zeros(len(rows))
        for row,passage,score in zip(rows,documents,relevance):
            row['relevance']=round(float(score),3)
            row['stance']='Unverified'
            if score<.08:continue
            comparison=compare_claim(text,re.split(r'(?<=[.!?])\s+',passage))
            if comparison:
                row['comparison']=comparison
                row['stance']='Experimental comparison: '+comparison['assessment']
        rows=sorted((row for row in rows if row['relevance']>=.08),key=lambda row:row['relevance'],reverse=True)
    return {'status':'retrieved' if rows else 'insufficient','items':rows,'query':query,
            'note':'Searches Google News headlines and FactCheck.org publication excerpts. Sentence relations are experimental. Read the linked source to establish what it reviewed.'}


def plagiarism(text, references):
    # Passage shingles allow an interpretable overlap score with a declared corpus.
    def tokens(t): return re.findall(r'\w+',t.lower())
    def shingles(ts): return {tuple(ts[i:i+5]) for i in range(max(0,len(ts)-4))}
    ts=tokens(text); covered=set(); matches=[]
    for ref in references[:20]:
        reftext=ref.get('text','')[:100000]; grams=shingles(tokens(reftext))
        positions=[i for i in range(max(0,len(ts)-4)) if tuple(ts[i:i+5]) in grams]
        if positions:
            for i in positions: covered.update(range(i,i+5))
            matches.append({'title':ref.get('title','Reference'), 'url':ref.get('url',''),
                'matching_shingles':len(positions),'excerpt':' '.join(ts[positions[0]:positions[0]+24])})
    return {'status':'compared' if references else 'no_corpus','overlap_pct':round(100*len(covered)/max(1,len(ts)),1) if references else None,
        'references_checked':len(references),'matches':matches,'note':'Five-word overlap against the supplied references only. Not an internet-wide plagiarism percentage or a judgment about attribution.'}

@lru_cache(maxsize=1)
def source_directory():
    return json.loads((ROOT/'backend/resources/cred1/domains.json').read_text(encoding='utf-8'))


def source_info(url):
    host=(urlparse(url).hostname or '').lower().rstrip('.')
    record=source_directory().get(host.removeprefix('www.')) if host else None
    info={'domain':host or None,'status':'Matched directory' if record else 'Unlisted' if host else 'Not supplied',
          'note':'No matching domain rating in the CRED-1 snapshot.' if host else 'No source rating available.'}
    if record:
        info.update(directory='CRED-1',source_score=record['s'],
                    attribution_url='https://github.com/aloth/cred-1/tree/f1594c04dfb805373932def7f8746089f7371322',
                    note=f"CRED-1 snapshot domain score {record['s']:.2f} on a 0–1 scale. Publisher-level context only. An individual claim still needs evidence.")
    return info


def execute(mode,text,url,raw,filename,reference_text,live_evidence):
    started=time.monotonic(); signals=[]; ev={'status':'not_requested','items':[]}; refs=[]
    if url and not text:
        text,url=article(url)
        if not text.strip(): raise ValueError('No article text could be extracted. Paste the article text instead.')
    if filename.lower().endswith(('.txt','.md','.docx','.pdf')) and raw:
        text=(text+'\n'+extract_document(raw,filename)).strip(); raw=None
    if reference_text.strip(): refs.append({'title':'Your comparison reference','text':reference_text})
    if mode in ('news','text','all') and text:
        if mode in ('news','all'): signals.append(text_signal('fake_news',text))
        if mode in ('text','all'): signals.append(text_signal('ai_text',text))
        if live_evidence:
            ev=evidence(text)
            # Headlines are not a full-document plagiarism corpus.
        if mode in ('text','all'): overlap=plagiarism(text,refs)
        else: overlap=None
    else: overlap=None
    if raw:
        suffix=Path(filename).suffix.lower()
        if suffix in ('.png','.jpg','.jpeg','.webp'):
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>25_000_000: raise ValueError('Image exceeds 25 megapixels.')
                im.verify()
            if mode not in ('image','all','news'): raise ValueError('Select image or combined analysis for an image file.')
            signals.extend([image_signal('image_ai',raw),image_signal('image_manipulation',raw)])
            if text and mode in ('all','news'):
                signals.append(paired_news_signal(text,raw))
        elif suffix in ('.m4a','.wav','.flac','.ogg','.mp4','.webm','.avi','.mov'):
            task='audio' if suffix in ('.m4a','.wav','.flac','.ogg') else 'video'
            if mode not in (task,'all'): raise ValueError(f'Select {task} or combined analysis for this file.')
            with tempfile.TemporaryDirectory(prefix='veritas-',dir=ROOT/'data') as temp:
                path=Path(temp)/('input'+suffix); path.write_bytes(raw)
                if suffix == '.m4a':
                    from transcription import decode_phone_recording
                    path = decode_phone_recording(path)
                media_result = audio_signal(path) if task=='audio' else video_signal(path)
                signals.append(media_result)
                spoken = media_result.get('transcript', {}).get('text')
                if spoken:
                    news = text_signal('fake_news', spoken)
                    news['name'] = 'Transcript news credibility'
                    signals.append(news)
                    if live_evidence and not text:
                        ev = evidence(spoken)
        else: raise ValueError('Unsupported file. Use TXT, DOCX, PDF, JPG, PNG, WEBP, M4A, WAV, FLAC, OGG, MP4, WEBM or MOV.')
    if not signals and overlap is None: raise ValueError('Add text, a supported file or a public article URL.')
    active=[s for s in signals if s['status']=='analyzed']
    risky=[s['name'] for s in active if s['probability']>=.65 and s['verdict']!='Experimental score']
    # These are different targets; averaging authorship and truth probabilities is invalid.
    verdict='Review recommended' if risky else 'Inconclusive' if active else 'Analysis unavailable'
    result={'id':uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),'mode':mode,
        'title':(text[:110] or filename or url),'verdict':verdict,'signals':signals,'evidence':ev,'plagiarism':overlap,
        'reference_comparison':compare_claim(text, re.split(r'(?<=[.!?])\s+|\n+',reference_text)) if reference_text and mode in ('news','all','text') else None,
        'source':source_info(url),'summary':('Higher-risk signals: '+', '.join(risky)+'.' if risky else 'No conclusion about authenticity can be established from the available signals.'),
        'fusion':{'method':'explainable decision policy','probability':None,
            'note':'Scores measure different tasks and are kept separate. A title and image supplied together receive a separate learned Fakeddit score. Other detector outputs retain their individual meanings.'},
        'limitations':['Automated scores are research signals and can be wrong.','AI-generated content may be factual; human-written content may be false.',
            'Evidence headlines are not proof of support or contradiction.'],
        'elapsed_seconds':round(time.monotonic()-started,2)}
    with connection() as c:
        c.execute('INSERT INTO reports VALUES (?,?,?,?,?)',(result['id'],result['created_at'],mode,result['title'],json.dumps(result)))
        c.execute('INSERT INTO report_owners VALUES (?,?)',(result['id'],OWNER.get()))
    return result

def extract_document(raw,filename):
    if filename.lower().endswith(('.txt','.md')): return raw.decode('utf-8-sig')[:50000]
    if filename.lower().endswith('.pdf'):
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(raw))
        if len(reader.pages)>100: raise ValueError('PDF limit is 100 pages.')
        return '\n'.join(p.extract_text() or '' for p in reader.pages)[:50000]
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        info=z.getinfo('word/document.xml')
        if info.file_size>5*1024*1024: raise ValueError('Document text is too large.')
        tree=ET.fromstring(z.read(info))
        return '\n'.join(''.join(p.itertext()) for p in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))[:50000]

@router.get('/models')
def models():
    auxiliary = []
    for key in ('evidence_stance', 'tamper_mask'):
        folder = ROOT / 'models/public' / key
        if (folder/'report.json').exists() and (folder/'model.joblib').exists():
            auxiliary.append({'id':key, 'report':json.loads((folder/'report.json').read_text(encoding='utf-8'))})
    return {'models':registry(),'auxiliary':auxiliary,'device':'CPU','legacy_synthetic_models_used':False}

@router.get('/datasets')
def datasets():
    p=ROOT/'data/catalog.json'
    return json.loads(p.read_text(encoding='utf-8'))

@router.post('/analyze')
async def analyze(mode:str=Form('all'),text:str=Form(''),url:str=Form(''),reference_text:str=Form(''),
                  live_evidence:bool=Form(True),file:UploadFile|None=File(None)):
    if mode not in ('news','text','image','video','audio','all'): raise HTTPException(400,'Unknown detector mode.')
    if len(text)>50000 or len(reference_text)>100000: raise HTTPException(413,'Text exceeds the supported length.')
    raw=await file.read(MAX_BYTES+1) if file else None
    if raw and len(raw)>MAX_BYTES: raise HTTPException(413,'File limit is 30 MB.')
    try:
        async with ANALYSIS_SLOTS:
            return await run_in_threadpool(execute,mode,text.strip(),url.strip(),raw,file.filename if file else '',reference_text,live_evidence)
    except (ValueError,UnidentifiedImageError,OSError,ET.ParseError) as e:
        raise HTTPException(400,str(e)) from e

@router.get('/history')
def history():
    with connection() as c:
        rows=c.execute("SELECT r.id,created,mode,title FROM reports r LEFT JOIN report_owners o ON r.id=o.id WHERE coalesce(o.owner,'')=? ORDER BY created DESC LIMIT 100",(OWNER.get(),)).fetchall()
    return {'items':[dict(zip(['id','created_at','mode','title'],r)) for r in rows]}

@router.get('/reports/{report_id}')
def report(report_id:str):
    with connection() as c: row=c.execute("SELECT result FROM reports r LEFT JOIN report_owners o ON r.id=o.id WHERE r.id=? AND coalesce(o.owner,'')=?",(report_id,OWNER.get())).fetchone()
    if row is None: raise HTTPException(404,'Report not found.')
    return json.loads(row[0])

@router.delete('/reports/{report_id}')
def delete_report(report_id:str):
    with connection() as c:
        removed = c.execute("DELETE FROM reports WHERE id=? AND coalesce((SELECT owner FROM report_owners WHERE id=reports.id),'')=?",(report_id,OWNER.get()))
        if removed.rowcount == 0: raise HTTPException(404,'Report not found.')
        c.execute('DELETE FROM report_owners WHERE id=?',(report_id,))
    return {'deleted':True}
