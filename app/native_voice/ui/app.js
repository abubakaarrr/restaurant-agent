import {pcm16,VoiceActivity} from './audio.mjs';
const $=id=>document.getElementById(id);
let socket,context,stream,node,source,gain,vad,playback;
let live=false,connected=false,ready=false,muted=false,interrupting=false,pending=null,turns=0;
let startedAt=0,lastSpeechEnd=0,currentToken='',playbackStarted=0,timer;
const message=(role,text)=>{if(!text)return;const empty=$('messages').querySelector('.empty');if(empty)empty.remove();const d=document.createElement('div');d.className='message '+role;const label=document.createElement('small');label.textContent=role==='user'?'YOU':role==='agent'?'RESTAURANT AGENT':'CALL NOTE';d.append(label,document.createTextNode(text));$('messages').append(d);d.scrollIntoView({block:'nearest'});};
function status(text,hint=''){ $('status').textContent=text;$('hint').textContent=hint; }
function send(data){if(socket?.readyState===WebSocket.OPEN)socket.send(typeof data==='string'||data instanceof ArrayBuffer?data:JSON.stringify(data));}
function stopAudio(){if(playback){playback.onended=null;try{playback.stop();}catch{}playback.disconnect();playback=null;}$('orb').classList.remove('speaking');currentToken='';}
function interrupt(){
  if(!live||interrupting||ready)return;
  stopAudio();interrupting=true;send({type:'interrupt'});status('Listening…','Go ahead. The previous response was interrupted.');
}
function submit(audio){ready=false;lastSpeechEnd=audio.end;send(audio.pcm);status('Thinking…','Your turn is complete. Waiting for a verified reply.');$('stop').disabled=false;}
function capture(samples){
  if(!live||!connected||muted)return;
  const result=vad.feed(samples,{silenceMs:Number($('silence').value),threshold:playback ? .025 : .018});
  $('level').value=Math.min(1,result.rms*8);
  if(result.started){if(!ready)interrupt();status('Listening…','Keep speaking. A pause sends your turn automatically.');}
  if(result.chunks){
    const audio={pcm:pcm16(result.chunks,context.sampleRate),end:performance.now()-result.quietMs};
    if(ready)submit(audio);else pending=audio;
  }
}
async function play(result){
  const binary=atob(result.audio), bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
  const view=new DataView(bytes.buffer), buffer=context.createBuffer(1,bytes.length/2,24000), data=buffer.getChannelData(0);
  for(let i=0;i<data.length;i++)data[i]=view.getInt16(i*2,true)/32768;
  await context.resume();if(!live||interrupting)return;
  currentToken=result.token;playback=context.createBufferSource();playback.buffer=buffer;playback.connect(context.destination);
  const token=currentToken;playback.onended=()=>{playback=null;$('orb').classList.remove('speaking');if(live&&token===currentToken){send({type:'played',token});currentToken='';}};
  playbackStarted=performance.now();$('latency').textContent=((playbackStarted-lastSpeechEnd)/1000).toFixed(2)+' s';
  playback.start();$('orb').classList.add('speaking');status('Agent speaking','You can interrupt by speaking.');
}
async function receive(event){
  if(!live)return;const data=JSON.parse(event.data);
  if(data.type==='ready'){
    connected=true;ready=true;interrupting=false;$('stop').disabled=true;
    if(pending){const audio=pending;pending=null;submit(audio);}else if(!vad.active)status(muted?'Microphone muted':'Listening…',muted?'Unmute to continue.':'Say hello, ask a question, or place a test order.');
  }else if(data.type==='result'){
    if(interrupting)return;
    turns++;$('turns').textContent=turns+' turns';message('user',data.caller);
    $('state').textContent=JSON.stringify({processing_ms:data.elapsed_ms,order:data.state,tools:data.tools},null,2);
    if(data.allowed&&data.audio){message('agent',data.assistant);await play(data);}
    else message('note','The reply was withheld: '+data.reasons.join(', ')+'. Please clarify or try another question.');
  }else if(data.type==='error'){message('note',data.message);await endCall(data.message);}
  else if(data.type==='notice'){$('notice').textContent=data.message;}
}
async function startCall(){
  $('start').disabled=true;$('notice').textContent='';status('Connecting…','Allow microphone access when your browser asks.');
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error('Microphone access requires localhost in Chrome or Edge.');
    stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
    context=new AudioContext();await context.resume();await context.audioWorklet.addModule('/assets/capture.js');
    vad=new VoiceActivity(context.sampleRate);source=context.createMediaStreamSource(stream);node=new AudioWorkletNode(context,'microphone-capture');gain=context.createGain();gain.gain.value=0;source.connect(node);node.connect(gain);gain.connect(context.destination);node.port.onmessage=e=>capture(e.data);
    socket=new WebSocket('ws://'+location.host+'/voice');live=true;connected=false;ready=false;interrupting=false;pending=null;muted=false;turns=0;startedAt=Date.now();
    $('messages').replaceChildren();$('state').textContent='Waiting for your first turn.';$('turns').textContent='0 turns';$('latency').textContent='—';$('mute').textContent='Mute';
    socket.onmessage=e=>receive(e).catch(()=>endCall('Audio playback failed. Check your output device and start a new call.'));
    socket.onerror=()=>{$('notice').textContent='Could not connect to the local voice server.';};
    socket.onclose=()=>{if(live)endCall('Call disconnected. Start a new call to reconnect.');};
    $('end').disabled=false;$('mute').disabled=false;$('orb').classList.add('active');
    timer=setInterval(()=>{const sec=Math.floor((Date.now()-startedAt)/1000);$('duration').textContent=String(Math.floor(sec/60)).padStart(2,'0')+':'+String(sec%60).padStart(2,'0');},1000);
    status('Connecting to agent…','Your microphone is ready.');
  }catch(error){await endCall(error.name==='NotAllowedError'?'Microphone permission was denied. Allow it in the browser, then start again.':error.message);}
}
async function endCall(reason='Call ended'){
  live=false;connected=false;ready=false;pending=null;stopAudio();clearInterval(timer);send({type:'end'});socket?.close();
  stream?.getTracks().forEach(t=>t.stop());node?.disconnect();source?.disconnect();gain?.disconnect();if(context&&context.state!=='closed')await context.close();
  $('start').disabled=false;$('end').disabled=true;$('mute').disabled=true;$('stop').disabled=true;$('level').value=0;$('orb').classList.remove('active');status('Call ended','Start another call for a fresh conversation.');$('notice').textContent=reason==='Call ended'?'':reason;
}
$('start').onclick=startCall;$('end').onclick=()=>endCall();$('stop').onclick=interrupt;
$('mute').onclick=()=>{muted=!muted;stream?.getAudioTracks().forEach(t=>t.enabled=!muted);vad.reset();$('mute').textContent=muted?'Unmute':'Mute';if(ready)status(muted?'Microphone muted':'Listening…');};
window.addEventListener('pagehide',()=>{stream?.getTracks().forEach(t=>t.stop());socket?.close();});
fetch('/info').then(r=>r.json()).then(info=>{$('restaurant').textContent=info.restaurant;$('clock').textContent='Restaurant clock: '+info.clock;}).catch(()=>{$('notice').textContent='Local server is unavailable.';});
