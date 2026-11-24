'use strict';
const api=globalThis.browser||chrome;
const $=id=>document.getElementById(id);
function base(){const u=new URL($('server').value);if(u.protocol!=='http:'||!['localhost','127.0.0.1'].includes(u.hostname))throw Error('Use a local http://localhost or http://127.0.0.1 server.');return u.origin;}
const token = location.hash.slice(1);
let selectedMedia;
api.storage.local.get(token ? [token, 'server'] : ['server']).then(async saved => {
  if (saved.server) $('server').value = saved.server;
  const item = saved[token];
  if (item) {
    $('text').value = item.text || '';
    if (item.media) {
      selectedMedia = item;
      $('media-source').textContent = 'Selected ' + item.mode + ': ' + item.media;
    }
    await api.storage.local.remove(token);
  }
});

async function selectedFile() {
  if ($('file').files[0]) return $('file').files[0];
  if (!selectedMedia) return null;
  const url = new URL(selectedMedia.media);
  if (!['https:', 'http:'].includes(url.protocol)) throw Error('Save the media and choose its file.');
  const response = await fetch(url, {credentials: 'omit', signal: AbortSignal.timeout(30000)})
    .catch(() => { throw Error('This site restricts media access. Save the media and choose its file.'); });
  if (!response.ok || !response.body) throw Error('Media unavailable. Choose a local copy.');
  const reader = response.body.getReader(), chunks = [];
  let bytes = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    bytes += value.length;
    if (bytes > 30 * 1024 * 1024) { await reader.cancel(); throw Error('File exceeds 30 MB.'); }
    chunks.push(value);
  }
  const type = (response.headers.get('content-type') || '').split(';')[0];
  const ext = {'image/jpeg':'jpg','image/png':'png','image/webp':'webp','video/mp4':'mp4','video/webm':'webm','video/quicktime':'mov','video/x-msvideo':'avi'}[type];
  if (!ext) throw Error('Unsupported media type. Choose a supported local file.');
  return new File(chunks, 'selection.' + ext, {type});
}
$('capture').addEventListener('click',async()=>{try{const [tab]=await api.tabs.query({active:true,currentWindow:true});const results=await api.scripting.executeScript({target:{tabId:tab.id},func:()=>String(window.getSelection())||(document.querySelector('article')||document.querySelector('main')||document.body).innerText.slice(0,50000)});$('text').value=results[0].result;$('result').textContent='Page text added. Review it before sending.';}catch{$('result').textContent='Cannot capture this page. Paste the passage manually.';}});
$('analyze').addEventListener('click',async()=>{try{const server=base();$('analyze').disabled=true;$('result').textContent='Examining the passage…';await api.storage.local.set({server});const data=new FormData();const file=await selectedFile();if(!file&&!$('text').value.trim())throw Error('Add text or choose a file.');if(file){if(file.size>30*1024*1024)throw Error('File exceeds 30 MB.');data.set('file',file);}data.set('mode','all');data.set('text',$('text').value);data.set('reference_text',$('reference').value);data.set('live_evidence',String($('evidence').checked));const r=await fetch(server+'/api/v2/analyze',{method:'POST',headers:$('token').value?{Authorization:'Bearer '+$('token').value}:{},body:data,signal:AbortSignal.timeout(180000)});const d=await r.json();if(!r.ok)throw Error(d.detail||'Analysis failed.');$('result').textContent=d.verdict+'\n\n'+d.signals.map(s=>s.name+': '+(s.probability===null?s.verdict:(s.probability*100).toFixed(1)+'% risk')).join('\n');$('studio').href=server+'/#report/'+d.id;}catch(e){$('result').textContent=e.message;}finally{$('analyze').disabled=false;}});
