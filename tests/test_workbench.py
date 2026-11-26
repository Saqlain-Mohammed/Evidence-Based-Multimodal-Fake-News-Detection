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

def test_workspace_entry_points_open_current_interface(client):
    for path in ('/', '/index.html', '/frontend/index.html', '/static/index.html'):
        response = client.get(path)
        assert response.status_code == 200
        assert response.url.path == '/'
        assert response.headers['cache-control'] == 'no-cache'
        assert 'data-veritas-workspace="current"' in response.text
        assert 'Verification studio' in response.text


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

def test_reference_comparison_exposes_both_sentences(client):
    text = 'The city council approved the bridge repairs.'
    reference = 'The city council approved the bridge repairs. Rain is expected tonight.'
    result = client.post('/api/v2/analyze', data={'mode':'news', 'text':text,
                         'reference_text':reference, 'live_evidence':'false'}).json()
    comparison = result['reference_comparison']
    assert comparison['claim'] == comparison['passage'] == text
    assert comparison['assessment'] in {'entailment','neutral','contradiction','Inconclusive'}
    assert result['fusion']['probability'] is None


def test_unrelated_reference_abstains():
    comparison = w.compare_claim('The bridge needs repairs.', ['Lemons grow on citrus trees.'])
    assert comparison['assessment'] == 'Insufficient overlap'


def test_welfake_probability_direction():
    import joblib
    payload = joblib.load(ROOT/'models/public/fake_news/model.joblib')
    assert payload['csv_label_mapping'] == '0=real, 1=fake'
    sample = ('WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday to approve the measure, '
              'according to a statement released by the committee. ')
    assert payload['model'].predict_proba([sample * 6])[0,1] < .5

def test_account_histories_and_legacy_rows_are_isolated(client,tmp_path,monkeypatch):
    from accounts import provision
    db=tmp_path/'accounts.sqlite3'
    users=[{'Authorization':'Bearer '+provision(db,name)} for name in ('one','two')]
    legacy=client.post('/api/v2/analyze',data={'mode':'news','text':'test','live_evidence':'false'}).json()['id']
    monkeypatch.setenv('VERITAS_ACCOUNTS_FILE',str(db))
    assert client.get('/api/v2/history').status_code==401
    created=client.post('/api/v2/analyze',headers=users[0],data={'mode':'news','text':'short claim','live_evidence':'false'})
    assert created.status_code==200, created.text
    item=created.json()['id']
    assert client.get('/api/v2/history',headers=users[1]).json()['items']==[]
    assert client.get('/api/v2/reports/'+legacy,headers=users[0]).status_code==404
    assert client.delete('/api/v2/reports/'+item,headers=users[1]).status_code==404
    assert client.get('/api/v2/reports/'+item,headers=users[0]).status_code==200

def test_paired_analysis_keeps_localization_and_joint_score_distinct(client):
    image=io.BytesIO()
    Image.new('RGB',(40,40),(63,70,80)).save(image,format='PNG')
    result=client.post('/api/v2/analyze',data={'mode':'all','text':'An image of the public meeting.',
                       'live_evidence':'false'},files={'file':('image.png',image.getvalue())})
    assert result.status_code==200,result.text[:200]
    signals={s['id']:s for s in result.json()['signals']}
    assert signals['image_manipulation']['mask_overlay'].startswith('data:image/jpeg;base64,')
    assert signals['image_ai']['heatmap'].startswith('data:image/png;base64,')
    assert signals['paired_news']['compared_title']=='An image of the public meeting.'


def test_publisher_context_is_scoped_to_matched_domain():
    domain=next(iter(w.source_directory()))
    assert w.source_info('https://www.'+domain)['source_score']==w.source_directory()[domain]['s']
    assert 'source_score' not in w.source_info('https://unlisted.example')


def test_primary_evidence_survives_news_feed_failure(monkeypatch):
    def primary(query):
        return [{'title':'City council approves repairs', 'passage':'The city council approved the bridge repairs.',
                 'url':'https://example.org/article','source':'Test publication','published':'','kind':'Excerpt'}]
    def fail(query):raise OSError('Offline')
    monkeypatch.setattr(w,'published_fact_checks',primary)
    monkeypatch.setattr(w,'google_headlines',fail)
    result=w.evidence('The city council approved the bridge repairs.')
    assert result['status']=='retrieved'
    assert result['items'][0]['comparison']['passage']==result['items'][0]['passage']


def test_splice_intervals_use_half_open_clip_bounded_times():
    from training.splice_features import intervals
    assert intervals(np.array([.2,.9,.9,.1,.8]),.7,.75)==[[.16,.48],[.64,.75]]
    assert intervals(np.array([.2]),.7,.03)==[]


def test_video_boundaries_merge_adjacent_candidates():
    from training.deletion_features import candidates
    assert candidates(np.array([.02,.83,.95,.79,.1,.2,.87]),.8,25)==[.12,.28]

def test_audio_edit_checkpoint_upload_returns_bounded_intervals(client):
    import soundfile as sf
    audio=io.BytesIO()
    sf.write(audio,np.zeros(8000,dtype=np.float32),16000,format='WAV')
    response=client.post('/api/v2/analyze',data={'mode':'audio','live_evidence':'false'},
                         files={'file':('test.wav',audio.getvalue(),'audio/wav')})
    assert response.status_code==200,response.text[:200]
    signals={s['id']:s for s in response.json()['signals']}
    assert signals['audio_edits']['status']=='analyzed'
    assert all(0<=start<end<=.5 for start,end in signals['audio_edits']['candidate_intervals'])


def test_video_edits_are_reported_independently(client,tmp_path):
    import cv2
    path=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(64,64))
    assert writer.isOpened()
    for idx in range(24):writer.write(np.full((64,64,3),idx*10,dtype=np.uint8))
    writer.release()
    response=client.post('/api/v2/analyze',data={'mode':'video','live_evidence':'false'},
                         files={'file':('fixture.avi',path.read_bytes(),'video/x-msvideo')})
    assert response.status_code==200,response.text[:200]
    signals={s['id']:s for s in response.json()['signals']}
    assert signals['video']['status']==signals['video_edits']['status']=='analyzed'
    assert all(0<=t<2 for t in signals['video_edits']['candidate_seconds'])


@pytest.mark.parametrize(("mode", "text_file", "media", "expected"), [
    ("news", "news.txt", None, {"fake_news"}),
    ("text", "text.txt", None, {"ai_text"}),
    ("image", None, "image.png", {"image_ai", "image_manipulation"}),
    ("video", None, "video.avi", {"video", "video_edits"}),
    ("audio", None, "audio.wav", {"audio", "audio_edits", "voice_conversion"}),
    ("all", "news.txt", "image.png", {
        "fake_news", "ai_text", "image_ai", "image_manipulation", "paired_news",
    }),
])
def test_public_samples(client, monkeypatch, mode, text_file, media, expected):
    samples = ROOT / "tests" / "samples"
    data = {"mode": mode}
    if text_file:
        data["text"] = (samples / text_file).read_text(encoding="utf-8")
    files = {"file": (media, (samples / media).read_bytes())} if media else None

    def unsolicited_evidence(text):
        pytest.fail("Evidence search should require opt-in")

    monkeypatch.setattr(w, "evidence", unsolicited_evidence)
    response = client.post("/api/v2/analyze", data=data, files=files)
    assert response.status_code == 200, response.text[:200]
    report = response.json()
    assert report["evidence"]["status"] == "not_requested"
    signals = {item["id"]: item for item in report["signals"]}
    assert set(signals) == expected
    for item in signals.values():
        assert item["status"] == "analyzed"
        assert 0 <= item["probability"] <= 1
