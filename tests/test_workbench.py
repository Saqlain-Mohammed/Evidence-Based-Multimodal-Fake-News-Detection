import io
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'));sys.path.insert(0,str(ROOT))
import workbench as w
from app import app

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'DB',tmp_path/'history.sqlite3')
    return TestClient(app)

def test_registry_excludes_legacy_synthetic_models(client):
    r=client.get('/api/v2/models').json()
    assert r['legacy_synthetic_models_used'] is False
    assert {m['id'] for m in r['models']}==set(w.TASKS)
    for m in r['models']:
        if m['status']=='ready':
            assert m['report']['trained_from_scratch'] is True
            assert m['report']['synthetic'] is False

def test_empty_submission_and_wrong_mode(client):
    assert client.post('/api/v2/analyze',data={'mode':'all','live_evidence':'false'}).status_code==400
    assert client.post('/api/v2/analyze',data={'mode':'unknown'}).status_code==400

def test_history_round_trip_does_not_invent_fusion(client,monkeypatch):
    monkeypatch.setattr(w,'text_signal',lambda task,text:w.signal(task,.85,'Fixture score for API contract test.'))
    text='A passage for verifying that model results and history are stored consistently. '*10
    r=client.post('/api/v2/analyze',data={'mode':'all','text':text,'live_evidence':'false'})
    assert r.status_code==200,r.text
    data=r.json();assert data['fusion']['probability'] is None
    assert data['verdict']=='Review recommended'
    assert data['evidence']['items']==[]
    assert client.get('/api/v2/history').json()['items'][0]['id']==data['id']
    assert client.get('/api/v2/reports/'+data['id']).json()==data
    assert client.delete('/api/v2/reports/'+data['id']).status_code==200
    assert client.get('/api/v2/reports/'+data['id']).status_code==404

def test_missing_audio_has_no_fabricated_score(client,monkeypatch):
    monkeypatch.setattr(w,'artifact',lambda task:None)
    r=client.post('/api/v2/analyze',data={'mode':'audio','live_evidence':'false'},files={'file':('test.wav',b'test','audio/wav')})
    assert r.status_code==200
    s=r.json()['signals'][0];assert s['status']=='unavailable';assert s['probability'] is None

def test_plagiarism_declares_corpus_and_matches():
    text='This is a specific original passage with enough words to identify overlap.'
    assert w.plagiarism(text,[])['overlap_pct'] is None
    result=w.plagiarism(text,[{'text':text,'title':'Original'}])
    assert result['overlap_pct']==100
    assert result['references_checked']==1 and result['matches']
    assert w.plagiarism(text,[{'text':'Entirely different words with no resemblance at all'}])['overlap_pct']==0

@pytest.mark.parametrize('url',['file:///etc/passwd','http://127.0.0.1:8000','http://localhost/','http://169.254.169.254/','http://[::1]/','https://user:secret@example.com'])
def test_private_and_non_http_urls_rejected(url):
    with pytest.raises(ValueError): w.safe_url(url)

def test_bad_image_rejected_even_without_checkpoint(client):
    r=client.post('/api/v2/analyze',data={'mode':'image'},files={'file':('broken.png',b'not an image','image/png')})
    assert r.status_code==400

def test_invalid_file_mode_rejected(client):
    b=io.BytesIO();Image.new('RGB',(20,20)).save(b,format='PNG')
    r=client.post('/api/v2/analyze',data={'mode':'audio'},files={'file':('image.png',b.getvalue(),'image/png')})
    assert r.status_code==400

def test_each_scratch_network_has_independent_gradients():
    import torch
    from training.scratch_models import ImageCNN,AudioCNN,VideoCNN,TextTransformer
    torch.set_num_threads(2)
    for model,x in [(ImageCNN(),torch.rand(2,3,32,32)),(AudioCNN(),torch.rand(2,1,257,60)),
                    (VideoCNN(),torch.rand(2,4,3,32,32)),(TextTransformer(vocab_size=100),torch.randint(1,100,(2,16)))]:
        out=model(x);assert out.shape==(2,)
        out.sum().backward();assert model.head.weight.grad is not None if hasattr(model.head,'weight') else model.head[-1].weight.grad is not None

def test_checkpoint_prediction_and_gradcam():
    if w.artifact('image_ai') is None: pytest.skip('No real CIFAKE checkpoint')
    buf=io.BytesIO();Image.new('RGB',(64,64),(100,140,180)).save(buf,format='PNG')
    result=w.image_signal('image_ai',buf.getvalue())
    assert 0<=result['probability']<=1
    assert result['heatmap'].startswith('data:image/png;base64,')
    assert 'attention' in result['heatmap_label']

def test_official_media_group_leakage_is_rejected(tmp_path):
    from training.prepare_media import prepare
    files=[]
    for i in range(6):
        p=tmp_path/f'{i}.wav';p.write_bytes(bytes([i])*20);files.append(p)
    protocol=tmp_path/'labels.csv'
    protocol.write_text('path,label,group,split\n'+''.join(f'{p.name},{i%2},same,\"{["train","val","test"][i//2]}\"\n' for i,p in enumerate(files)))
    with pytest.raises(ValueError,match='Group leakage'):prepare(protocol,tmp_path/'out')

def test_real_text_scores_and_short_input_guard():
    if w.artifact('ai_text') is None:pytest.skip('No HC3 checkpoint')
    assert w.text_signal('ai_text','too short')['probability'] is None
    r=w.text_signal('ai_text','The council held a meeting to discuss the new transport budget and invited residents to respond to the proposal before the deadline.')
    assert 0<=r['probability']<=1
    assert w.text_signal('fake_news','A short factual claim cannot be verified by an article classifier.')['probability'] is None
