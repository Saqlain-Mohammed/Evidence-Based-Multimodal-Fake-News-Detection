import React, {useEffect,useState} from 'react';
import {ActivityIndicator, Alert, Linking, Pressable, SafeAreaView, ScrollView, Share, StyleSheet, Switch, Text, TextInput, View} from 'react-native';
import {RecordingPresets, requestRecordingPermissionsAsync, setAudioModeAsync, useAudioRecorder, useAudioRecorderState} from 'expo-audio';
import {useIncomingShare} from 'expo-sharing';
import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';

const MODES=['all','news','text','image','video','audio'];
export default function App(){
  const [showSources,setShowSources]=useState(false);
  const [token,setToken]=useState('');
  const [server,setServer]=useState('http://10.0.2.2:8000');
  const [text,setText]=useState(''),[mode,setMode]=useState('all'),[file,setFile]=useState(null),[evidence,setEvidence]=useState(false),[busy,setBusy]=useState(false),[result,setResult]=useState(null);
  const {resolvedSharedPayloads,clearSharedPayloads}=useIncomingShare();
  useEffect(()=>{
    const item=resolvedSharedPayloads[0];
    if(!item)return;
    setResult(null);
    if(['text','url'].includes(item.shareType)){
      setText(item.value);setFile(null);setMode('all');
    }else if(item.contentUri){
      const name=item.originalName||item.contentUri.split('/').pop();
      setFile({uri:item.contentUri,name,mimeType:item.contentMimeType,size:item.contentSize});
      setText('');setMode(['image','audio','video'].includes(item.contentType)?item.contentType:'all');
    }
  },[resolvedSharedPayloads]);
  const recorder=useAudioRecorder(RecordingPresets.HIGH_QUALITY);
  const recording=useAudioRecorderState(recorder);
  const [changingRecording,setChangingRecording]=useState(false);
  async function recordVoice(){
    setChangingRecording(true);
    try {
      if(recording.isRecording){
        await recorder.stop();
        await setAudioModeAsync({allowsRecording:false});
        if(recorder.uri){
          setFile({uri:recorder.uri,name:'voice.m4a',mimeType:'audio/mp4'});
          setMode('audio');
        }
      } else {
        const permission=await requestRecordingPermissionsAsync();
        if(!permission.granted){Alert.alert('Microphone access is needed to record.');return;}
        await setAudioModeAsync({allowsRecording:true,playsInSilentMode:true});
        await recorder.prepareToRecordAsync();
        recorder.record();
      }
    } catch(e){Alert.alert('Recording unavailable',e.message);}
    finally{setChangingRecording(false);}
  }
  async function choose(){try{const r=await DocumentPicker.getDocumentAsync({copyToCacheDirectory:true,multiple:false});if(!r.canceled)setFile(r.assets[0]);}catch(e){Alert.alert('Cannot open file',e.message);}}
  async function camera(){try{const permission=await ImagePicker.requestCameraPermissionsAsync();if(!permission.granted){Alert.alert('Camera permission needed','You can choose a file instead.');return;}const r=await ImagePicker.launchCameraAsync({quality:.8});if(!r.canceled){const a=r.assets[0];setFile({...a,name:a.fileName||'capture.jpg',mimeType:a.mimeType||'image/jpeg'});setMode('image');}}catch(e){Alert.alert('Cannot capture photo',e.message);}}
  async function analyze(){try{if(!text.trim()&&!file)throw Error('Add text or a file first.');if(file?.size>30*1024*1024)throw Error('File limit is 30 MB.');const base=new URL(server);if(!['http:','https:'].includes(base.protocol))throw Error('Use an http or https server URL.');setBusy(true);const data=new FormData();data.append('mode',mode);if(/^https?:\/\/\S+$/i.test(text.trim()))data.append('url',text.trim());else data.append('text',text);data.append('live_evidence',String(evidence));if(file)data.append('file',{uri:file.uri,name:file.name||'upload',type:file.mimeType||'application/octet-stream'});const r=await fetch(base.origin+'/api/v2/analyze',{method:'POST',headers:token?{Authorization:'Bearer '+token}:{},body:data,signal:AbortSignal.timeout(180000)});const d=await r.json();if(!r.ok)throw Error(d.detail||'Analysis failed.');setResult(d);setShowSources(false);clearSharedPayloads();}catch(e){Alert.alert('Analysis unavailable',e.message);}finally{setBusy(false);}}
  return <SafeAreaView style={s.safe}><ScrollView contentContainerStyle={s.body} keyboardShouldPersistTaps="handled"><Text style={s.brand}>veritas.</Text><Text style={s.kicker}>CONTENT VERIFICATION</Text><Text style={s.title}>A closer look at{`\n`}what’s real.</Text><Text style={s.description}>Independent model signals. Evidence with context.</Text><View style={s.card}><Text style={s.label}>Your research server</Text><TextInput accessibilityLabel="Research server URL" autoCapitalize="none" keyboardType="url" style={s.input} value={server} onChangeText={setServer}/><TextInput accessibilityLabel="Workspace access token" secureTextEntry autoCapitalize="none" placeholder="Access token (protected servers)" style={s.input} value={token} onChangeText={setToken}/><Text style={s.note}>Android emulator: 10.0.2.2. A physical phone needs a reachable development server. Use token access and HTTPS for a shared server.</Text><View style={s.tabs}>{MODES.map(m=><Pressable key={m} accessibilityRole="tab" accessibilityState={{selected:m===mode}} onPress={()=>setMode(m)} style={[s.tab,m===mode&&s.activeTab]}><Text style={{color:m===mode?'#195b46':'#6c7d70'}}>{m==='all'?'Combined':m[0].toUpperCase()+m.slice(1)}</Text></Pressable>)}</View><Text style={s.label}>Your content</Text><TextInput accessibilityLabel="Content to analyze" style={[s.input,s.textarea]} multiline maxLength={50000} textAlignVertical="top" placeholder="Paste a passage to examine…" value={text} onChangeText={setText}/><View style={s.actions}><Pressable style={s.secondary} onPress={choose}><Text style={s.green}>Choose file</Text></Pressable><Pressable style={s.secondary} onPress={camera}><Text style={s.green}>Take photo</Text></Pressable></View><Pressable disabled={changingRecording||busy} style={s.secondary} onPress={recordVoice}><Text style={s.green}>{recording.isRecording?'Stop and attach recording':'Record voice'}</Text></Pressable>{file&&<View style={s.row}><Text style={s.note}>{file.name}</Text><Pressable onPress={()=>setFile(null)}><Text style={s.green}>Remove</Text></Pressable></View>}<View style={s.row}><Text style={s.description}>Find related reporting online</Text><Switch value={evidence} onValueChange={setEvidence} trackColor={{true:'#195b46'}}/></View><Text style={s.note}>Online search sends a short query externally. Reports are stored on your server.</Text><Pressable style={[s.primary,busy&&{opacity:.5}]} onPress={analyze} disabled={busy||recording.isRecording}>{busy?<ActivityIndicator color="white"/>:<Text style={s.white}>Analyze content</Text>}</Pressable></View>{result&&<View style={s.card}><Text style={s.kicker}>ANALYSIS REPORT</Text><Text style={s.resultTitle}>{result.verdict}</Text><Text style={s.description}>{result.summary}</Text>{result.signals.map(r=><View style={s.signal} key={r.name}><View style={s.row}><Text style={s.label}>{r.name}</Text><Text style={s.score}>{r.probability===null?'—':(r.probability*100).toFixed(1)+'%'}</Text></View><Text style={s.green}>{r.verdict}</Text><Text style={s.description}>{r.explanation}</Text>{r.transcript?.text&&<Text style={s.description}>{r.transcript.text}</Text>}</View>)}<Pressable style={s.secondary} onPress={()=>setShowSources(!showSources)}><Text style={s.green}>{showSources?'Hide':'Show'} evidence ({result.evidence.items.length})</Text></Pressable>{showSources&&<View><Text style={s.note}>{result.evidence.note}</Text>{result.evidence.items.map((item,i)=><Pressable key={i} style={s.signal} onPress={()=>/^https?:\/\//i.test(item.url)&&Linking.openURL(item.url)}><Text style={s.label}>{item.title}</Text><Text style={s.note}>{item.source} · {item.stance}</Text></Pressable>)}</View>}<Text style={s.description}>{result.fusion.note}</Text><Pressable style={s.secondary} onPress={()=>Share.share({message:JSON.stringify(result,null,2),title:'Veritas report'})}><Text style={s.green}>Share report</Text></Pressable></View>}<Text style={s.note}>Research signals, with room for human judgment.</Text></ScrollView></SafeAreaView>;
}
const s=StyleSheet.create({safe:{flex:1,backgroundColor:'#f6f8f7'},body:{padding:24,gap:16,paddingBottom:60},brand:{fontSize:32,fontWeight:'700',letterSpacing:-1.5,color:'#195b46'},kicker:{fontSize:10,letterSpacing:1.6,color:'#657d69',marginTop:10},title:{fontFamily:'serif',fontSize:40,lineHeight:46,color:'#22362a',letterSpacing:-1},description:{fontSize:14,lineHeight:23,color:'#65796a'},card:{padding:20,borderWidth:1,borderColor:'#e4e9e6',borderRadius:14,backgroundColor:'white',gap:12},label:{fontSize:14,fontWeight:'600',color:'#22362a'},input:{borderWidth:1,borderColor:'#dfe6e0',borderRadius:8,padding:12,fontSize:16,color:'#22362a',backgroundColor:'#fcfdfc'},textarea:{minHeight:150},note:{fontSize:11,color:'#74866f',lineHeight:18,flexShrink:1},tabs:{flexDirection:'row',flexWrap:'wrap',gap:5},tab:{paddingVertical:10,paddingHorizontal:8,borderBottomWidth:2,borderBottomColor:'transparent'},activeTab:{borderBottomColor:'#195b46',backgroundColor:'#eaf3ee'},actions:{flexDirection:'row',gap:10},secondary:{padding:12,borderWidth:1,borderColor:'#dce5dd',borderRadius:7,alignItems:'center',flexGrow:1},green:{color:'#195b46',fontSize:13},row:{flexDirection:'row',alignItems:'center',justifyContent:'space-between',gap:10},primary:{padding:15,backgroundColor:'#195b46',borderRadius:8,alignItems:'center',marginTop:8},white:{color:'white',fontSize:15,fontWeight:'600'},resultTitle:{fontSize:25,color:'#745d2f'},signal:{borderTopWidth:1,borderTopColor:'#e4e9e6',paddingTop:15,gap:8},score:{fontSize:23,color:'#22362a'}});
