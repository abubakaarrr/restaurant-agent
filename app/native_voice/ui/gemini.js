const $ = id => document.getElementById(id);
let socket, context, stream, node, source, gain, timer;
let live = false, connected = false, muted = false, startedAt = 0, turns = 0;
let epoch = 0, token = '', sources = new Set(), nextAudioAt = 0, audioFinished = false;
let lastSpeechAt = 0, lastEndpointAt = 0, firstAudio = true, activeAgent, diagnostics = {};
const transcriptNodes = new Map();
let microphoneCheckBusy = false, microphoneCheckUrl = '';
let endAfterPlayback = false;
function microphoneConstraints() {
  const device = $('inputDevice').value;
  return {audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    ...(device ? {deviceId: {exact: device}} : {})}};
}
async function showMicrophones(track) {
  const chosen = track?.getSettings().deviceId || $('inputDevice').value;
  const inputs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput');
  $('inputDevice').replaceChildren(new Option('Browser default microphone', ''));
  for (const [i, input] of inputs.entries()) $('inputDevice').add(new Option(input.label || 'Microphone ' + (i + 1), input.deviceId));
  if ([...$('inputDevice').options].some(o => o.value === chosen)) $('inputDevice').value = chosen;
  if (track) {
    const settings = track.getSettings();
    $('inputStatus').textContent = 'Using: ' + (track.label || 'Unnamed microphone') +
      ' · echo cancellation ' + (settings.echoCancellation === true ? 'enabled' : 'not confirmed') +
      ' · noise suppression ' + (settings.noiseSuppression === true ? 'enabled' : 'not confirmed');
  }
}
function microphoneWav(chunks) {
  const count = chunks.reduce((n, c) => n + c.length, 0);
  const data = new DataView(new ArrayBuffer(44 + count * 2));
  const str = (at, text) => { for (let i = 0; i < text.length; i++) data.setUint8(at + i, text.charCodeAt(i)); };
  str(0, 'RIFF'); data.setUint32(4, 36 + count * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  data.setUint32(16, 16, true); data.setUint16(20, 1, true); data.setUint16(22, 1, true);
  data.setUint32(24, 16000, true); data.setUint32(28, 32000, true); data.setUint16(32, 2, true); data.setUint16(34, 16, true);
  str(36, 'data'); data.setUint32(40, count * 2, true);
  let at = 44;
  for (const chunk of chunks) for (const sample of chunk) { const v = Math.max(-1, Math.min(1, sample)); data.setInt16(at, Math.round(v * (v < 0 ? 32768 : 32767)), true); at += 2; }
  return new Blob([data.buffer], {type: 'audio/wav'});
}
async function checkMicrophone() {
  if (live || microphoneCheckBusy) return;
  microphoneCheckBusy = true; $('start').disabled = true; $('checkMicrophone').disabled = true; $('inputDevice').disabled = true;
  let checkStream, checkContext, checkSource, checkNode, silent;
  const chunks = []; let count = 0, squares = 0, clipped = 0;
  $('microphonePlayback').pause(); $('microphonePlayback').hidden = true;
  if (microphoneCheckUrl) { URL.revokeObjectURL(microphoneCheckUrl); microphoneCheckUrl = ''; }
  try {
    checkStream = await navigator.mediaDevices.getUserMedia(microphoneConstraints());
    await showMicrophones(checkStream.getAudioTracks()[0]);
    checkContext = new AudioContext({sampleRate: 16000, latencyHint: 'interactive'});
    if (checkContext.sampleRate !== 16000) throw new Error('This browser did not provide 16 kHz audio. Try Chrome or Edge.');
    await checkContext.resume(); await checkContext.audioWorklet.addModule('/assets/capture.js');
    checkSource = checkContext.createMediaStreamSource(checkStream); checkNode = new AudioWorkletNode(checkContext, 'microphone-capture');
    silent = checkContext.createGain(); silent.gain.value = 0;
    checkSource.connect(checkNode); checkNode.connect(silent); silent.connect(checkContext.destination);
    $('microphoneCheckResult').textContent = 'Recording eight seconds locally. Say the first sentence of your request now; the agent will stay silent.';
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('The browser did not deliver microphone audio. Check its microphone permission and selected input.')), 12000);
      checkNode.port.onmessage = e => {
        const samples = e.data.slice(0, 8 * 16000 - count);
        chunks.push(samples); count += samples.length;
        let energy = 0;
        for (const v of samples) { energy += v*v; if (Math.abs(v) > .99) clipped++; }
        squares += energy; $('level').value = Math.min(1, Math.sqrt(energy / Math.max(samples.length, 1)) * 8);
        if (count >= 8 * 16000) { checkNode.port.onmessage = null; clearTimeout(timeout); resolve(); }
      };
    });
    microphoneCheckUrl = URL.createObjectURL(microphoneWav(chunks));
    $('microphonePlayback').src = microphoneCheckUrl; $('microphonePlayback').hidden = false;
    const db = Math.round(20 * Math.log10(Math.max(Math.sqrt(squares / Math.max(count, 1)), 1e-6)));
    $('microphoneCheckResult').textContent = 'Recorded input: ' + db + ' dBFS average; ' + (100 * clipped / Math.max(count, 1)).toFixed(1) + '% clipped. Press play and check whether these are your actual words. This sample stays in this page and was not sent to Gemini.';
  } catch (e) {
    $('microphoneCheckResult').textContent = e.name === 'NotAllowedError' ? 'Microphone permission was denied. Allow it and try again.' : e.message;
  } finally {
    checkStream?.getTracks().forEach(t => t.stop()); checkNode?.disconnect(); checkSource?.disconnect(); silent?.disconnect();
    if (checkContext && checkContext.state !== 'closed') await checkContext.close();
    microphoneCheckBusy = false; $('start').disabled = false; $('checkMicrophone').disabled = false; $('inputDevice').disabled = false; $('level').value = 0;
  }
}
function message(role, text, id) {
  if (!text && !id) return;
  $('messages').querySelector('.empty')?.remove();
  let row = id && transcriptNodes.get(id);
  if (!row) {
    row = document.createElement('div'); row.className = 'message ' + role;
    const label = document.createElement('small'); label.textContent = role === 'user' ? 'YOU' : role === 'agent' ? 'RESTAURANT AGENT · AI VOICE' : 'CALL NOTE';
    const body = document.createElement('span'); row.append(label, body); $('messages').append(row);
    if (id) transcriptNodes.set(id, row);
  }
  row.querySelector('span').textContent = text;
  row.scrollIntoView({block: 'nearest'}); return row;
}
function status(text, hint = '') { $('status').textContent = text; $('hint').textContent = hint; }
function send(data) { if (socket?.readyState === WebSocket.OPEN) socket.send(data instanceof ArrayBuffer ? data : JSON.stringify(data)); }
function evidence() { $('state').textContent = JSON.stringify(diagnostics, null, 2); }
function stopAudio() {
  for (const s of sources) { s.onended = null; try { s.stop(); } catch {} s.disconnect(); }
  sources.clear(); nextAudioAt = 0; token = ''; audioFinished = false;
  $('orb').classList.remove('speaking');
  if (activeAgent?.dataset.playback === 'pending') activeAgent.dataset.playback = 'interrupted';
}
function interrupt() { stopAudio(); send({type: 'interrupt'}); status('Listening…', 'Go ahead.'); }
function capture(samples) {
  if (!live || !connected) return;
  let energy = 0, clipping = 0;
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const v = muted ? 0 : samples[i]; energy += v * v; if (Math.abs(v) > .99) clipping++;
    pcm[i] = Math.round(Math.max(-1, Math.min(1, v)) * (v < 0 ? 32768 : 32767));
  }
  const rms = Math.sqrt(energy / samples.length); $('level').value = Math.min(1, rms * 8);
  // This is timing telemetry only. It NEVER commits a turn or cancels a response.
  if (rms > .018 && !sources.size) lastSpeechAt = performance.now();
  if (clipping > samples.length * .08) $('notice').textContent = 'Microphone is clipping. Move slightly farther from it or lower its input level.';
  if (socket.bufferedAmount > 160000) { endCall('The connection cannot keep up with the microphone. Reconnect once the network is stable.'); return; }
  send(pcm.buffer);
}
function maybePlayed() {
  if (endAfterPlayback && !sources.size && (!token || audioFinished)) { endCall(); return; }
  if (!token || !audioFinished || sources.size) return;
  const finished = token; token = ''; $('orb').classList.remove('speaking');
  if (activeAgent) activeAgent.dataset.playback = 'complete';
  send({type: 'played', token: finished}); status(muted ? 'Microphone muted' : 'Listening…', 'Ask another question or continue your order.');
}
async function playChunk(data) {
  if (data.epoch !== epoch || data.token !== token) return;
  await context.resume();
  if (!live || data.epoch !== epoch || data.token !== token) return;
  const raw = atob(data.audio), view = new DataView(new ArrayBuffer(raw.length));
  for (let i = 0; i < raw.length; i++) view.setUint8(i, raw.charCodeAt(i));
  const buffer = context.createBuffer(1, raw.length / 2, 24000), pcm = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) pcm[i] = view.getInt16(i * 2, true) / 32768;
  const s = context.createBufferSource(); s.buffer = buffer; s.connect(context.destination);
  const at = Math.max(context.currentTime + .035, nextAudioAt); nextAudioAt = at + buffer.duration;
  sources.add(s); s.onended = () => { sources.delete(s); s.disconnect(); maybePlayed(); };
  if (firstAudio) {
    firstAudio = false;
    const now = performance.now() + (at - context.currentTime) * 1000;
    $('latency').textContent = lastSpeechAt ? ((now - lastSpeechAt) / 1000).toFixed(2) + ' s' : '—';
    diagnostics.browser_endpoint_to_playback_ms = lastEndpointAt ? Math.round(now - lastEndpointAt) : null;
    diagnostics.browser_speech_end_to_playback_estimate_ms = lastSpeechAt ? Math.round(now - lastSpeechAt) : null; evidence();
  }
  s.start(at); $('orb').classList.add('speaking'); $('stop').disabled = false;
  status('Agent speaking', 'You can interrupt naturally.');
}
async function receive(data) {
  if (!live) return;
  if (data.type === 'call_closing') {
    connected = false;
    stream?.getTracks().forEach(track => { track.onended = null; track.onmute = null; track.stop(); });
    $('mute').disabled = true; $('stop').disabled = true;
  } else if (data.type === 'call_end_after_playback') {
    endAfterPlayback = true; maybePlayed();
  } else if (data.type === 'ready') {
    connected = true; diagnostics.session_id = data.session_id; diagnostics.pipeline = data.pipeline; $('model').textContent = data.model || 'Gemini 3.8 Live'; evidence();
    status('Listening…', 'Say hello, ask about the restaurant, or place a test order.');
  } else if (data.type === 'interrupted') { stopAudio(); epoch = data.epoch; }
  else if (data.type === 'speech_started') { status('Listening…', 'Keep speaking. Your audio is streamed continuously.'); }
  else if (data.type === 'speech_stopped') { lastEndpointAt = performance.now(); status('Thinking…', 'Preparing your reply.'); }
  else if (data.type === 'transcript_delta') {
    const old = transcriptNodes.get(data.item_id)?.querySelector('span').textContent || '';
    message('user', old + data.delta, data.item_id);
  } else if (data.type === 'transcript') {
    message('user', data.text, data.item_id); turns++; $('turns').textContent = turns + ' turns';
  } else if (data.type === 'assistant') {
    if (data.epoch !== epoch) return;
    token = data.token; firstAudio = true; audioFinished = false; nextAudioAt = 0;
    activeAgent = message('agent', data.text, data.token); activeAgent.dataset.playback = 'pending';
  } else if (data.type === 'assistant_delta') {
    if (data.epoch === epoch && data.token === token) message('agent', data.text, data.token);
  } else if (data.type === 'audio') { await playChunk(data); }
  else if (data.type === 'audio_done') { if (data.token === token) { audioFinished = true; maybePlayed(); } }
  else if (data.type === 'state') { diagnostics.order = data.order; diagnostics.tentative_request = data.draft; evidence(); }
  else if (data.type === 'tool') { (diagnostics.tools ||= []).push(data); evidence(); window.dispatchEvent(new Event('native-voice-state-changed')); }
  else if (data.type === 'timing') { diagnostics.timing = data; evidence(); }
  else if (data.type === 'diagnostic') {
    (diagnostics.events ||= []).push(data); evidence();
    if (data.stage === 'playback_ack_too_early') {
      const retryToken = data.token, retryEpoch = data.epoch;
      setTimeout(() => { if (live && epoch === retryEpoch) send({type: 'played', token: retryToken}); },
                 Math.max(50, Number(data.retry_after_ms) + 60));
    }
  }
  else if (data.type === 'notice') { message('note', data.message); }
  else if (data.type === 'error') { await endCall(data.message); }
}
async function startCall() {
  if (microphoneCheckBusy) return;
  endAfterPlayback = false;
  $('microphonePlayback').pause(); $('checkMicrophone').disabled = true; $('inputDevice').disabled = true;
  $('start').disabled = true; $('language').disabled = true; $('pace').disabled = true;
  $('notice').textContent = ''; status('Connecting…', 'Allow microphone access when asked.');
  try {
    stream = await navigator.mediaDevices.getUserMedia(microphoneConstraints());
    await showMicrophones(stream.getAudioTracks()[0]);
    stream.getAudioTracks()[0].onmute = () => { if (live) $('notice').textContent = 'The browser microphone track has stopped supplying audio. Check your selected headset and connection.'; };
    stream.getAudioTracks()[0].onended = () => { if (live) endCall('The selected microphone disconnected. Select your headset and reconnect.'); };
    context = new AudioContext({sampleRate: 16000, latencyHint: 'interactive'});
    if (context.sampleRate !== 16000) throw new Error('This browser does not support the required audio format. Please use current Chrome or Edge.');
    await context.resume(); await context.audioWorklet.addModule('/assets/capture.js');
    source = context.createMediaStreamSource(stream); node = new AudioWorkletNode(context, 'microphone-capture'); gain = context.createGain(); gain.gain.value = 0;
    source.connect(node); node.connect(gain); gain.connect(context.destination); node.port.onmessage = e => capture(e.data);
    socket = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/voice-gemini?pace=' + $('pace').value);
    live = true; connected = false; muted = false; epoch = 0; turns = 0; lastSpeechAt = 0; lastEndpointAt = 0;
    diagnostics = {microphone: stream.getAudioTracks()[0].getSettings(), microphone_name: stream.getAudioTracks()[0].label, audio_sample_rate: context.sampleRate};
    transcriptNodes.clear(); $('messages').replaceChildren(); $('latency').textContent = '—'; $('turns').textContent = '0 turns'; $('mute').textContent = 'Mute';
    let incoming = Promise.resolve(); socket.onmessage = e => { incoming = incoming.then(() => receive(JSON.parse(e.data))).catch(() => endCall('Audio processing failed. Reconnect to try again.')); };
    socket.onerror = () => { $('notice').textContent = 'Could not connect to the local voice server.'; };
    socket.onclose = () => { if (live) endCall('The call disconnected. Reconnect to start a fresh session.'); };
    $('end').disabled = false; $('mute').disabled = false; $('stop').disabled = false; $('orb').classList.add('active');
    startedAt = Date.now(); timer = setInterval(() => { const s = Math.floor((Date.now() - startedAt) / 1000); $('duration').textContent = String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); }, 1000);
    evidence();
  } catch (e) { await endCall(e.name === 'NotAllowedError' ? 'Allow microphone access, then start again.' : e.message); }
}
async function endCall(reason = 'Call ended') {
  const wasLive = live; live = false; connected = false; stopAudio(); clearInterval(timer);
  if (wasLive) send({type: 'end'}); socket?.close(); stream?.getTracks().forEach(t => t.stop());
  node?.disconnect(); source?.disconnect(); gain?.disconnect();
  if (context && context.state !== 'closed') await context.close();
  for (const id of ['start', 'language', 'pace', 'inputDevice', 'checkMicrophone']) $(id).disabled = false;
  for (const id of ['end', 'mute', 'stop']) $(id).disabled = true;
  $('level').value = 0; $('orb').classList.remove('active'); status('Call ended', 'Start another call for a new session.'); $('notice').textContent = reason === 'Call ended' ? '' : reason;
}
$('start').onclick = startCall; $('end').onclick = () => endCall(); $('stop').onclick = interrupt;
$('checkMicrophone').onclick = checkMicrophone;
showMicrophones().catch(() => {});
$('mute').onclick = () => { muted = !muted; stream?.getAudioTracks().forEach(t => t.enabled = !muted); $('mute').textContent = muted ? 'Unmute' : 'Mute'; status(muted ? 'Microphone muted' : 'Listening…'); };
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(t => t.stop()); socket?.close(); });
fetch('/info').then(r => { if (!r.ok) throw new Error('info_unavailable'); return r.json(); }).then(info => { if ($('restaurant')) $('restaurant').textContent = info.restaurant; $('clock').textContent = 'Restaurant clock: ' + info.clock; }).catch(() => { $('notice').textContent = 'Local voice server is unavailable.'; });
