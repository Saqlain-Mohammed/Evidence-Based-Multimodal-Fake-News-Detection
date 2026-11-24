'use strict';
const $ = (id) => document.getElementById(id);
const paths = {
 scan:'M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10m-7-3 3 3-3 3',
 history:'M3 11a9 9 0 1 1 2.6 7M3 4v7h7m2-4v5l3 2',
 layers:'m12 3 9 5-9 5-9-5 9-5m-9 9 9 5 9-5M3 16l9 5 9-5',
 database:'M21 6c0 2-4 3-9 3S3 8 3 6s4-3 9-3 9 1 9 3ZM3 6v12c0 2 4 3 9 3s9-1 9-3V6M3 12c0 2 4 3 9 3s9-1 9-3',
 book:'M12 5c-3-2-7-2-10-1v15c3-1 7-1 10 1m0-15c3-2 7-2 10-1v15c-3-1-7-1-10 1V5',
 shield:'m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3m-4 9 3 3 5-6',
 lock:'M6 10h12v11H6V10m3 0V6a3 3 0 0 1 6 0v4m-3 4v3',
 external:'M14 3h7v7m0-7L10 14M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5',
 spark:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3',
 news:'M5 4h15v15a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8h3m0-4v15a2 2 0 0 1-2 2M9 8h7m-7 4h7m-7 4h7',
 text:'M4 5V3h16v2M12 3v18m-4 0h8M4 10h3M4 14h3',
 image:'M3 3h18v18H3V3m0 14 6-6 5 5 3-3 4 4M16 7h.01',
 video:'M3 5h13v14H3V5m13 5 5-3v10l-5-3',
 audio:'M3 10v4m4-8v12m5-16v20m5-16v12m4-8v4',
 link:'m10 13 4-4m-6 7-1 1a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m0 2 1-1a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0',
 upload:'M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5',
 download:'M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4',
 compass:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-6-3-2 5-4 1 2-5 4-1',
 check:'m5 12 4 4 10-10',
 bulb:'M9 18h6m-5 3h4M8 15a6 6 0 1 1 8 0v3H8v-3',
 info:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM12 11v6m0-10v.01',
 menu:'M4 6h16M4 12h16M4 18h16'
};
function icon(name) { return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.scan}"/></svg>`; }
function icons(root=document) { root.querySelectorAll('[data-icon]').forEach(el=>{el.innerHTML=icon(el.dataset.icon);}); }
function esc(value) { const e=document.createElement('span'); e.textContent=value ?? ''; return e.innerHTML; }
function safeLink(value) { try {const u=new URL(value); return ['https:','http:'].includes(u.protocol)?esc(u.href):'#';} catch{return '#';} }
let mode='all', selectedFile=null, previewUrl=null, currentReport=null, controller=null;
const titles={workspace:'Verification studio',history:'Analysis history',models:'Model lab',datasets:'Dataset library',method:'How it works'};
const descriptions={all:'News, writing patterns and media — one report, separate signals.',news:'Check article patterns and find related reporting.',text:'Examine AI writing patterns and compare source overlap.',image:'Separate checks for AI generation and image manipulation.',video:'A dedicated temporal model for video deepfake signals.',audio:'A dedicated spectrogram model for synthetic speech signals.'};
const accepts={all:'.txt,.md,.pdf,.docx,.jpg,.jpeg,.png,.webp,.mp4,.webm,.mov,.avi,.m4a,.wav,.flac,.ogg',news:'.txt,.md,.pdf,.docx,.jpg,.jpeg,.png,.webp',text:'.txt,.md,.pdf,.docx',image:'.jpg,.jpeg,.png,.webp',video:'.mp4,.webm,.mov,.avi',audio:'.m4a,.wav,.flac,.ogg'};
const fmt=(p)=>`${(p*100).toFixed(1)}%`;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,3500);}
let accessToken='';
$('account-form').addEventListener('submit',async event=>{
  event.preventDefault();accessToken=$('account-key').value.trim();
  try{await api('/models');$('account-key').value='';$('account-form').hidden=true;refreshModels();loadHistory();if(location.hash==='#datasets')loadDatasets();if(location.hash.startsWith('#report/'))navigate();}
  catch(e){$('account-message').textContent=e.message;}
});
async function api(path,options={}){const headers=new Headers(options.headers);if(accessToken)headers.set('Authorization','Bearer '+accessToken);
const r=await fetch('/api/v2'+path,{...options,headers});if(r.status===401)$('account-form').hidden=false;let d;try{d=await r.json();}catch{throw Error('The server returned an unreadable response.');}if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:d.error||'The request failed.');return d;}
function error(message){$('form-error').textContent=message;$('form-error').hidden=false;}
function updateCount(){$('char-count').textContent=`${$('content').value.length.toLocaleString()} / 50,000`;}
function setMode(next){mode=next;document.querySelectorAll('.detector').forEach(b=>{b.classList.toggle('active',b.dataset.mode===mode);b.setAttribute('aria-selected',String(b.dataset.mode===mode));b.tabIndex=b.dataset.mode===mode?0:-1;});$('mode-description').textContent=descriptions[mode];$('file-input').accept=accepts[mode];$('content').placeholder=['image','video','audio'].includes(mode)?'Optional context for your media. Add a file below to run this detector.':'Paste an article, a claim, or a passage you’d like to examine…';$('sample-button').hidden=['image','video','audio'].includes(mode);$('source-url').disabled=['image','video','audio'].includes(mode);$('file-hint').textContent=`${{all:'Documents, images, video or audio',news:'Documents or images',text:'TXT, PDF or DOCX',image:'JPG, PNG or WEBP',video:'MP4, WEBM, MOV or AVI',audio:'M4A, WAV, FLAC or OGG'}[mode]} · up to 30 MB`;$('form-error').hidden=true;}
let selectedFiles=[];
function setFile(file){if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;selectedFile=null;$('file-preview').hidden=true;$('preview-image').hidden=true;if(!file)return;if(file.size>30*1024*1024){error('Choose a file smaller than 30 MB.');$('file-input').value='';return;}selectedFile=file;$('selected-file-name').textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;$('file-preview').hidden=false;if(file.type.startsWith('image/')){previewUrl=URL.createObjectURL(file);$('preview-image').src=previewUrl;$('preview-image').hidden=false;}}
function setFiles(files){
  selectedFiles=Array.from(files);
  if(selectedFiles.some(file=>file.size>30*1024*1024)){
    selectedFiles=[];setFile(null);$('file-input').value='';
    error('Each file must be 30 MB or smaller.');return;
  }
  setFile(selectedFiles[0]);
  if(selectedFiles.length>1){
    $('selected-file-name').textContent=selectedFiles.length+' files · one report per file';
    $('preview-image').hidden=true;
  }
}
async function refreshModels(){try{const data=await api('/models');let ready=0;$('engine-list').innerHTML=data.models.map(m=>{if(m.status==='ready')ready++;return `<div class="engine-row"><span class="engine-icon">${icon(m.modality==='text'?'text':m.modality)}</span><span class="engine-label">${esc(m.name)}</span><span class="engine-status ${m.status==='ready'?'ready':''}">${m.status==='ready'?'READY':'UNTRAINED'}</span></div>`;}).join('');$('connection-label').textContent='Local engine connected';$('connection-dot').classList.add('ready');$('ready-label').textContent=`${ready} of ${data.models.length} model checkpoints ready`;
 $('model-cards').innerHTML=data.models.map(m=>{const r=m.report;return `<article class="panel resource-card"><span class="status-pill ${r?'ready':''}">${r?'Trained from scratch':'Training required'}</span><h2>${esc(m.name)}</h2><p>${esc(m.dataset)}</p>${r?`<div class="metrics-row"><div class="metric"><b>${fmt(r.test.accuracy)}</b><span>Held-out accuracy</span></div><div class="metric"><b>${r.test.macro_f1.toFixed(3)}</b><span>Macro F1</span></div><div class="metric"><b>${r.test.roc_auc.toFixed(3)}</b><span>ROC AUC</span></div></div><p>${esc(r.architecture)}</p>${r.evaluation_scope?`<p><strong>Evaluation protocol:</strong> ${esc(r.evaluation_scope)}</p>`:``}<p>${Object.entries(r.counts).map(([k,v])=>`${esc(k)}: ${v.toLocaleString()}`).join(' · ')}</p><p>${esc(r.limitations)}</p>${m.cross_domain?`<p><strong>Cross-domain stress test:</strong> ${esc(m.cross_domain.dataset)} accuracy ${fmt(m.cross_domain.test.accuracy)}. This measures transfer beyond the training dataset.</p>`:''}`:'<p>No checkpoint is active. The separate training pipeline is implemented; the provider dataset and a completed training run are required.</p>'}</article>`;}).join('');
 for(const entry of data.auxiliary||[]){
   const r=entry.report, card=document.createElement('article');
   const stance=entry.id==='evidence_stance';
   const measurement=stance
     ? `${r.training_pairs.toLocaleString()} training pairs · ${fmt(r.mismatched_development.accuracy)} mismatched development accuracy`
     : `${fmt(r.mean_tampered_image_iou)} mean intersection over union on tampered images · ${fmt(r.mean_authentic_highlight_fraction)} mean highlighted area on authentic images`;
   card.className='panel resource-card';
   card.innerHTML=`<span class="status-pill ready">Trained from scratch</span><h2>${stance?'Evidence comparison':'Tamper localization'}</h2><p>${esc(r.dataset)}</p><p>${esc(measurement)}</p><p>${esc(r.limitations)}</p>`;
   $('model-cards').append(card);
 }
 }catch(e){$('connection-label').textContent='Engine unavailable';$('ready-label').textContent='Start the local backend to analyze';$('connection-dot').classList.remove('ready');$('engine-list').innerHTML='<p>Cannot reach the model registry. Refresh after starting the backend.</p>';$('model-cards').textContent=e.message;}}
async function loadDatasets(){try{const d=await api('/datasets');$('dataset-cards').innerHTML=d.datasets.map(r=>`<article class="panel resource-card"><span class="section-kicker">${esc(r.task)}</span><h2>${esc(r.name)}</h2><p><strong>${esc(r.size)}</strong></p><p>${esc(r.access)}</p><p>${esc(r.notes)}</p><a href="${safeLink(r.url)}" target="_blank" rel="noopener">Visit the dataset source</a></article>`).join('');}catch(e){$('dataset-cards').textContent=e.message;}}
async function loadHistory(){try{const d=await api('/history');$('history-count').textContent=d.items.length;$('history-list').innerHTML=d.items.length?d.items.map(r=>`<article class="panel history-item"><span class="engine-icon">${icon('history')}</span><div class="history-description"><h2>${esc(r.title)}</h2><p>${esc(new Date(r.created_at).toLocaleString())} · ${esc(r.mode)} analysis</p></div><button class="secondary-button open-report" data-id="${esc(r.id)}">Open report</button><button class="text-button delete-report" data-id="${esc(r.id)}" aria-label="Delete report">Delete</button></article>`).join(''):'<div class="panel result-empty"><h2>No analyses yet.</h2><p>Your first verification report will appear here.</p></div>';}catch(e){$('history-list').textContent=e.message;}}
async function navigate(){const hash=location.hash.slice(1);
if(/^report\/[0-9a-f]{32}$/.test(hash)){
  const requested=hash.slice(7);
  document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!=='page-workspace');
  try{const result=await api('/reports/'+requested);if(location.hash==='#'+hash){renderReport(result);$('results-panel').scrollIntoView({block:'start'});}}
  catch(e){error(e.message);}
  return;
}const page=Object.hasOwn(titles,hash)?hash:'workspace';document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==`page-${page}`);document.querySelectorAll('[data-page]').forEach(a=>{a.classList.toggle('active',a.dataset.page===page);if(a.dataset.page===page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});$('breadcrumb-page').textContent=titles[page];$('sidebar').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');if(page==='history')loadHistory();if(page==='models')refreshModels();if(page==='datasets')loadDatasets();}
function renderReport(r){currentReport=r;$('result-empty').hidden=true;$('loading-state').hidden=true;$('result-content').hidden=false;$('download-report').hidden=false;$('result-title').textContent='Your verification report';
 const signals=r.signals.map(s=>{const p=s.probability,cls=p===null?'warn':p>=.65?'risk':p>.35?'warn':'';return `<article class="signal-card"><div class="signal-top"><h4>${esc(s.name)}</h4><span class="signal-number">${p===null?'—':fmt(p)}</span></div><div class="signal-verdict ${cls}">${esc(s.verdict)}</div>${p===null?'':`<div class="meter" role="meter" aria-label="${esc(s.name)} risk score" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${(p*100).toFixed(1)}"><span style="width:${(p*100).toFixed(1)}%"></span></div>`}<p>${esc(s.explanation)}</p>${s.heatmap&&s.heatmap.startsWith('data:image/png;base64,')?`<img class="heatmap" src="${s.heatmap}" alt="Positive-class model attention"><div class="heatmap-caption">${esc(s.heatmap_label)}</div>`:''}${s.mask_overlay?.startsWith('data:image/jpeg;base64,')?`<img class="heatmap" src="${s.mask_overlay}" alt="Estimated tamper region highlighted in red"><p>${esc(s.mask_note)} Highlighted area: ${fmt(s.highlighted_fraction)}.</p>`:''}${s.transcript?`<p><strong>English transcript</strong> · ${esc(s.transcript.status)}</p><p>${esc(s.transcript.text)}</p>`:''}${s.sampled_seconds?`<p>Sampled seconds: ${s.sampled_seconds.join(', ')}</p>`:''}${s.visual_changes?`<p>Largest visual changes between samples: ${s.visual_changes.map(v=>v.from_seconds+'–'+v.to_seconds+' s').join(', ')}. Scene changes can be ordinary cuts.</p>`:''}${s.analyzed_seconds?`<p>Analyzed: first ${s.analyzed_seconds.toFixed(1)} seconds</p>`:''}</article>`;}).join('');
 const ev=r.evidence,pl=r.plagiarism,comparison=r.reference_comparison;
 $('result-content').innerHTML=`<div class="result-body"><div class="verdict-banner"><div class="result-meta">${esc(new Date(r.created_at).toLocaleString())} · ${r.elapsed_seconds}s</div><h3>${esc(r.verdict)}</h3><p>${esc(r.summary)}</p></div><div class="signal-grid">${signals}</div>${comparison?`<section class="report-section"><h3>Reference comparison</h3><p><strong>Opening claim</strong> · ${esc(comparison.claim)}</p><p><strong>Reference passage</strong> · ${esc(comparison.passage)}</p><p>Experimental sentence relation · ${esc(comparison.assessment)}</p></section>`:''}${pl?`<section class="report-section"><h3>Reference overlap ${pl.overlap_pct===null?'':`· ${pl.overlap_pct}%`}</h3><p>${esc(pl.note)}</p>${pl.matches.map(m=>`<p><strong>${esc(m.title)}</strong> · ${m.matching_shingles} matching phrases<br>“${esc(m.excerpt)}”</p>`).join('')}</section>`:''}<section class="report-section"><h3>Related reporting · ${ev.items.length} sources</h3><p>${esc(ev.note||'Online evidence search was not requested for this analysis.')}</p>${ev.items.map(e=>`<a class="evidence-item" href="${safeLink(e.url)}" target="_blank" rel="noopener"><strong>${esc(e.title)}</strong>${e.comparison?`<p>Compared passage · ${esc(e.comparison.passage)}</p>`:''}<small>${esc(e.source)} · ${esc(e.stance)} · relevance ${(e.relevance*100).toFixed(0)}%</small></a>`).join('')}</section><section class="report-section"><h3>Source context</h3><p>${esc(r.source.domain||'No publisher URL supplied.')} ${esc(r.source.note)}</p>${r.source.attribution_url?`<a href="${safeLink(r.source.attribution_url)}" target="_blank" rel="noopener">CRED-1 source and methodology</a>`:''}</section><section class="report-section"><h3>How the signals are combined</h3><p>${esc(r.fusion.note)}</p><ul>${r.limitations.map(l=>`<li>${esc(l)}</li>`).join('')}</ul></section></div>`;
}
function downloadReport(){if(!currentReport)return;const blob=new Blob([JSON.stringify(currentReport,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`veritas-report-${currentReport.id.slice(0,8)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('analysis-form').addEventListener('submit',async event=>{
  event.preventDefault();
  if(controller)return;
  $('form-error').hidden=true;
  const text=$('content').value.trim();
  const url=$('source-url').disabled?'':$('source-url').value.trim();
  if(['image','video','audio'].includes(mode)&&!selectedFiles.length){error('Add a '+mode+' file.');return;}
  if(!text&&!url&&!selectedFiles.length){error('Add text, a public URL, or a file.');return;}
  const files=selectedFiles.length?[...selectedFiles]:[null];
  const batch=files.length>1, outcome=$('batch-outcomes');
  outcome.replaceChildren();outcome.hidden=!batch;
  const data=new FormData();
  data.set('mode',mode);data.set('text',text);data.set('url',url);
  data.set('reference_text',$('reference-text').value);
  data.set('live_evidence',String($('live-evidence').checked));
  controller=new AbortController();
  const job=controller;
  $('analyze-button').disabled=true;
  $('result-empty').hidden=true;$('result-content').hidden=true;$('loading-state').hidden=false;
  $('download-report').hidden=true;
  try{
    for(const [index,file] of files.entries()){
      if(job.signal.aborted)break;
      $('analyze-button-label').textContent=batch?`Analyzing ${index+1} of ${files.length}`:'Analyzing…';
      if(file)data.set('file',file);
      try{
        const result=await api('/analyze',{method:'POST',body:data,
          signal:AbortSignal.any([job.signal,AbortSignal.timeout(180000)])});
        renderReport(result);
        if(batch){
          if(index+1<files.length)$('loading-state').hidden=false;
          const button=document.createElement('button');button.type='button';button.className='text-button';
          button.textContent=file.name+' · '+result.verdict;button.onclick=()=>renderReport(result);
          const row=document.createElement('p');row.append(button);outcome.append(row);
        }
      }catch(err){
        const message=job.signal.aborted?'Stopped waiting. Check analysis history for completed reports.':err.message;
        if(!batch||job.signal.aborted){error(message);break;}
        const row=document.createElement('p');row.textContent=file.name+' · '+message;outcome.append(row);
      }
    }
    await loadHistory();refreshModels();
  }finally{
    $('loading-state').hidden=true;
    if(currentReport)renderReport(currentReport);else $('result-empty').hidden=false;
    controller=null;$('analyze-button').disabled=false;$('analyze-button-label').textContent='Analyze content';
  }
});
$('cancel-button').addEventListener('click',()=>controller?.abort());
$('content').addEventListener('input',updateCount);
$('sample-button').addEventListener('click',()=>{$('content').value='The city council voted on Tuesday to expand evening bus services after a three-month public consultation. The transport department said the first additional routes would begin next month, subject to a final budget review. Residents can view the proposed timetable on the council website.';updateCount();$('content').focus();toast('Sample added. Run an analysis to see actual model results.');});
document.querySelectorAll('.detector').forEach(b=>{b.addEventListener('click',()=>setMode(b.dataset.mode));b.addEventListener('keydown',e=>{const tabs=[...document.querySelectorAll('.detector')];let index=tabs.indexOf(b);if(e.key==='ArrowRight')index=(index+1)%tabs.length;else if(e.key==='ArrowLeft')index=(index+tabs.length-1)%tabs.length;else if(e.key==='Home')index=0;else if(e.key==='End')index=tabs.length-1;else return;e.preventDefault();setMode(tabs[index].dataset.mode);tabs[index].focus();});});
$('file-input').addEventListener('change',e=>setFiles(e.target.files));$('remove-file').addEventListener('click',()=>{setFiles([]);$('file-input').value='';});
['dragenter','dragover'].forEach(name=>$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.add('dragover');}));['dragleave','drop'].forEach(name=>$('dropzone').addEventListener(name,e=>{e.preventDefault();$('dropzone').classList.remove('dragover');if(name==='drop'&&e.dataTransfer.files.length)setFiles(e.dataTransfer.files);}));
$('download-report').addEventListener('click',downloadReport);
$('menu-toggle').addEventListener('click',()=>{const open=$('sidebar').classList.toggle('open');$('menu-toggle').setAttribute('aria-expanded',String(open));});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('sidebar').classList.remove('open');$('menu-toggle').setAttribute('aria-expanded','false');}});
$('history-list').addEventListener('click',async e=>{const open=e.target.closest('.open-report'),del=e.target.closest('.delete-report');try{if(open){const r=await api('/reports/'+open.dataset.id);location.hash='workspace';renderReport(r);$('results-panel').scrollIntoView({block:'start'});}if(del){await api('/reports/'+del.dataset.id,{method:'DELETE'});await loadHistory();toast('Report deleted.');}}catch(err){toast(err.message);}});
window.addEventListener('hashchange',navigate);
function syncNavigation(){const narrow=matchMedia('(max-width: 850px)').matches;$('sidebar').inert=narrow&&!$('sidebar').classList.contains('open');}
$('menu-toggle').addEventListener('click',syncNavigation);window.addEventListener('resize',syncNavigation);window.addEventListener('hashchange',()=>{syncNavigation();$('main').focus({preventScroll:true});});document.addEventListener('keydown',e=>{if(e.key==='Escape')syncNavigation();});
icons();setMode('all');navigate();syncNavigation();refreshModels();loadHistory();setInterval(refreshModels,60000);
