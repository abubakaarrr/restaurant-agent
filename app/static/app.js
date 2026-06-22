// Restaurant AI Receptionist — browser demo
// Server injects config into window.APP_CONFIG before this script runs.

const { vapiPubKey, vapiAssistantId, restaurantName, vapiReady } = window.APP_CONFIG;

let sessionId = 'demo-' + Math.random().toString(36).slice(2, 10);
window.__callActive = false;
window.__vapi = null;

// ── Voice call ────────────────────────────────────────────────

async function toggleCall() {
  if (!vapiReady) {
    alert('Add VAPI_PUBLIC_KEY and VAPI_ASSISTANT_ID to your .env file to enable voice calls.');
    return;
  }
  const vapi = window.__vapi;
  if (!vapi) {
    document.getElementById('callStatus').textContent = '● SDK loading... try again in 2 seconds';
    document.getElementById('callStatus').classList.add('visible');
    return;
  }
  if (window.__callActive) {
    vapi.stop();
  } else {
    document.getElementById('callBtn').className = 'call-btn connecting';
    document.getElementById('callBtn').textContent = '⏳';
    document.getElementById('callStatus').textContent = '● Allow microphone when prompted...';
    document.getElementById('callStatus').classList.add('visible');
    try {
      await vapi.start(vapiAssistantId);
    } catch (e) {
      console.error('vapi.start error:', e);
      document.getElementById('callStatus').textContent = '● ' + (e.message || 'Call failed');
      document.getElementById('callBtn').className = 'call-btn idle';
      document.getElementById('callBtn').textContent = '📞';
    }
  }
}

// ── Retell voice call ─────────────────────────────────────────

async function toggleRetellCall() {
  if (!window.APP_CONFIG.retellReady) {
    alert('Add RETELL_API_KEY and RETELL_AGENT_ID to your .env file to enable Retell calls.');
    return;
  }
  const client = window.__retell;
  if (!client) {
    document.getElementById('retellStatus').textContent = '● SDK loading... try again in 2 seconds';
    document.getElementById('retellStatus').classList.add('visible');
    return;
  }

  if (window.__retellActive) {
    client.stopCall();
    return;
  }

  const btn = document.getElementById('retellBtn');
  const status = document.getElementById('retellStatus');
  retellShownTurns = 0;
  btn.className = 'call-btn connecting';
  btn.textContent = '⏳';
  status.textContent = '● Connecting...';
  status.classList.add('visible');

  try {
    const res = await fetch('/api/retell/web-call', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    status.textContent = '● Allow microphone when prompted...';
    await client.startCall({ accessToken: data.access_token });
  } catch (e) {
    console.error('Retell start error:', e);
    status.textContent = '● ' + (e.message || 'Call failed');
    btn.className = 'call-btn idle';
    btn.textContent = '📞';
  }
}

// Live transcript from Retell: re-render only the finalized turns into the
// chat panel. Retell sends the whole transcript each update, so we track how
// many complete turns we've already shown to avoid duplicates.
let retellShownTurns = 0;

window.renderRetellTranscript = function (transcript) {
  // All turns except the last are considered final; the last one is still
  // being spoken, so we wait until a newer turn appears before locking it in.
  const finalCount = Math.max(0, transcript.length - 1);
  for (let i = retellShownTurns; i < finalCount; i++) {
    const turn = transcript[i];
    const role = turn.role === 'agent' ? 'agent' : 'user';
    addMessage(role, turn.content || '');
  }
  retellShownTurns = finalCount;
};

// ── Text chat helpers ─────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function addMessage(role, text) {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  const av = role === 'agent' ? '🤖' : '👤';
  d.innerHTML = `<div class="msg-avatar">${av}</div><div class="bubble">${escapeHtml(text)}</div>`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function showTyping() {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg agent';
  d.id = 'typing';
  d.innerHTML = '<div class="msg-avatar">🤖</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById('typing');
  if (el) el.remove();
}

function escapeHtml(t) {
  return t
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

async function sendMessage() {
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;

  addMessage('user', text);
  showTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId, caller_phone: '+1555000001' }),
    });
    const data = await res.json();
    hideTyping();
    addMessage('agent', data.reply || 'Sorry, I had trouble responding.');
  } catch (e) {
    hideTyping();
    addMessage('agent', 'Connection error. Is the server running?');
  }

  document.getElementById('sendBtn').disabled = false;
  document.getElementById('input').focus();
}

function resetChat() {
  sessionId = 'demo-' + Math.random().toString(36).slice(2, 10);
  document.getElementById('messages').innerHTML = '';
  addMessage('agent', `Hello! Thank you for calling ${restaurantName}. I'm Sana, your AI receptionist. How can I help you today?`);
}

// ── Tab switching ─────────────────────────────────────────────

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

  document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
  document.getElementById(`tab-${tabName}`).classList.add('active');

  if (tabName === 'voice-studio') {
    loadVoices();
    checkTTSHealth();
  }
  if (tabName === 'history') {
    loadHistory();
  }
}

// ── History ───────────────────────────────────────────────────

function fmtDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, {
    weekday: 'short', day: 'numeric', month: 'short',
    hour: 'numeric', minute: '2-digit',
  });
}

function renderOrderBlock(order, title) {
  if (!order) return '';
  const lines = (order.items || []).map(it => `
    <div class="hist-order-line">
      <span><span class="qty">${it.quantity}×</span> ${escapeHtml(it.item_name)}${it.notes ? ` <span class="note">(${escapeHtml(it.notes)})</span>` : ''}</span>
      <span class="price">$${it.subtotal.toFixed(2)}</span>
    </div>
  `).join('');
  return `
    <div class="hist-order">
      <div class="hist-order-title">${title} · #${order.order_id} · ${escapeHtml(order.status)}</div>
      <div class="hist-order-items">${lines || '<div class="hist-order-line">No items</div>'}</div>
      <div class="hist-order-total"><span>Total</span><span>$${order.total_amount.toFixed(2)}</span></div>
    </div>
  `;
}

function renderReservation(r) {
  const tableInfo = r.table_number
    ? `Table ${r.table_number}${r.location ? ` · ${escapeHtml(r.location)}` : ''}`
    : 'No table';
  const preorderBadge = r.preorder ? '<span class="hist-badge preorder">Pre-order</span>' : '';
  const reason = (r.status === 'cancelled' && r.cancellation_reason)
    ? `<div class="hist-reason">Reason: ${escapeHtml(r.cancellation_reason)}</div>`
    : '';
  return `
    <div class="hist-item">
      <div class="hist-item-top">
        <span class="hist-item-name">${escapeHtml(r.customer_name)}</span>
        <div class="hist-badges">
          <span class="hist-badge table">${tableInfo}</span>
          ${preorderBadge}
          <span class="hist-badge ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span>
        </div>
      </div>
      <div class="hist-item-meta">
        <span><strong>${r.party_size}</strong> ${r.party_size === 1 ? 'guest' : 'guests'}</span>
        <span>${fmtDateTime(r.booked_at)}</span>
        <span>Booking #${r.booking_id}</span>
        ${r.customer_phone ? `<span>${escapeHtml(r.customer_phone)}</span>` : ''}
      </div>
      ${reason}
      ${renderOrderBlock(r.preorder, 'Pre-order')}
    </div>
  `;
}

function renderPickup(o) {
  return `
    <div class="hist-item">
      <div class="hist-item-top">
        <span class="hist-item-name">${escapeHtml(o.customer_name || 'Guest')}</span>
        <div class="hist-badges">
          <span class="hist-badge ${escapeHtml(o.status)}">${escapeHtml(o.status)}</span>
        </div>
      </div>
      <div class="hist-item-meta">
        <span>Order #${o.order_id}</span>
        <span>${fmtDateTime(o.created_at)}</span>
        ${o.customer_phone ? `<span>${escapeHtml(o.customer_phone)}</span>` : ''}
      </div>
      ${renderOrderBlock(o, 'Items')}
    </div>
  `;
}

async function loadHistory() {
  const resList = document.getElementById('reservationsList');
  const pickList = document.getElementById('pickupList');
  resList.innerHTML = '<div class="vs-empty">Loading reservations...</div>';
  pickList.innerHTML = '<div class="vs-empty">Loading orders...</div>';

  try {
    const res = await fetch('/api/history');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const reservations = data.reservations || [];
    const pickups = data.pickup_orders || [];

    document.getElementById('statReservations').textContent = reservations.length;
    document.getElementById('statPreorders').textContent =
      reservations.filter(r => r.preorder).length;
    document.getElementById('statPickups').textContent = pickups.length;

    resList.innerHTML = reservations.length
      ? reservations.map(renderReservation).join('')
      : '<div class="vs-empty">No reservations yet.</div>';

    pickList.innerHTML = pickups.length
      ? pickups.map(renderPickup).join('')
      : '<div class="vs-empty">No pickup orders yet.</div>';
  } catch (e) {
    resList.innerHTML = `<div class="vs-empty">Failed to load history: ${escapeHtml(e.message)}</div>`;
    pickList.innerHTML = '';
  }
}

// ── Voice Studio ──────────────────────────────────────────────

let selectedFile = null;

async function checkTTSHealth() {
  const dot = document.getElementById('ttsStatusDot');
  const txt = document.getElementById('ttsStatusText');
  try {
    const res = await fetch('/api/tts/health');
    const data = await res.json();
    if (data.status === 'ok') {
      dot.className = 'vs-status-dot online';
      txt.textContent = `TTS service online (${data.device})`;
    } else {
      dot.className = 'vs-status-dot offline';
      txt.textContent = 'TTS service offline — start the tts container';
    }
  } catch {
    dot.className = 'vs-status-dot offline';
    txt.textContent = 'TTS service unreachable';
  }
}

async function loadVoices() {
  const grid = document.getElementById('voiceGrid');
  const select = document.getElementById('ttsVoiceSelect');
  try {
    const res = await fetch('/api/voices');
    const data = await res.json();
    const voices = data.voices || [];

    if (voices.length === 0) {
      grid.innerHTML = '<div class="vs-empty">No voices yet — upload one above!</div>';
    } else {
      grid.innerHTML = voices.map(v => `
        <div class="vs-voice-card">
          <div class="vs-voice-icon">${v.category === 'system' ? '🔊' : '🎙️'}</div>
          <div class="vs-voice-info">
            <strong>${escapeHtml(v.name)}</strong>
            <span class="vs-voice-meta">${v.category} · ${v.size_kb} KB</span>
          </div>
          <div class="vs-voice-actions">
            <button class="vs-btn-small" onclick="previewVoice('${v.voice_key}')">Play</button>
            ${v.category === 'custom' ? `<button class="vs-btn-small vs-btn-danger" onclick="deleteVoice('${v.id}')">Del</button>` : ''}
          </div>
        </div>
      `).join('');
    }

    select.innerHTML = '<option value="">Select a voice...</option>' +
      voices.map(v => `<option value="${v.voice_key}">${v.name} (${v.category})</option>`).join('');
  } catch {
    grid.innerHTML = '<div class="vs-empty">Failed to load voices</div>';
  }
}

// Upload area drag-and-drop + click
document.addEventListener('DOMContentLoaded', () => {
  const area = document.getElementById('uploadArea');
  const fileInput = document.getElementById('voiceFileInput');
  if (!area || !fileInput) return;

  area.addEventListener('click', () => fileInput.click());
  area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('dragover'); });
  area.addEventListener('dragleave', () => area.classList.remove('dragover'));
  area.addEventListener('drop', e => {
    e.preventDefault();
    area.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
  });
});

function handleFileSelect(file) {
  selectedFile = file;
  document.getElementById('uploadArea').style.display = 'none';
  document.getElementById('uploadForm').style.display = 'block';
  document.getElementById('fileName').textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  const baseName = file.name.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9-_ ]/g, '');
  document.getElementById('voiceName').value = baseName;
}

function clearUpload() {
  selectedFile = null;
  document.getElementById('uploadArea').style.display = '';
  document.getElementById('uploadForm').style.display = 'none';
  document.getElementById('voiceFileInput').value = '';
  document.getElementById('uploadStatus').textContent = '';
}

async function uploadVoice() {
  if (!selectedFile) return;
  const name = document.getElementById('voiceName').value.trim();
  if (!name) { alert('Please enter a voice name'); return; }

  const btn = document.getElementById('uploadBtn');
  const status = document.getElementById('uploadStatus');
  btn.disabled = true;
  btn.textContent = 'Uploading...';
  status.textContent = '';

  const form = new FormData();
  form.append('file', selectedFile);
  form.append('name', name);

  try {
    const res = await fetch('/api/voices/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok) {
      status.className = 'vs-upload-status success';
      status.textContent = `Voice "${data.voice.name}" uploaded successfully!`;
      clearUpload();
      loadVoices();
    } else {
      status.className = 'vs-upload-status error';
      status.textContent = data.detail || 'Upload failed';
    }
  } catch (e) {
    status.className = 'vs-upload-status error';
    status.textContent = 'Network error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Upload Voice';
}

function previewVoice(voiceKey) {
  const audio = new Audio(`/api/voices/${voiceKey}`);
  audio.play().catch(e => alert('Cannot play: ' + e.message));
}

async function deleteVoice(voiceId) {
  if (!confirm(`Delete voice "${voiceId}"?`)) return;
  try {
    await fetch(`/api/voices/${voiceId}`, { method: 'DELETE' });
    loadVoices();
  } catch (e) {
    alert('Delete failed: ' + e.message);
  }
}

async function generateTTS() {
  const voiceKey = document.getElementById('ttsVoiceSelect').value;
  const text = document.getElementById('ttsText').value.trim();
  if (!voiceKey) { alert('Select a voice first'); return; }
  if (!text) { alert('Enter some text'); return; }

  const btn = document.getElementById('generateBtn');
  const status = document.getElementById('ttsGenStatus');
  const container = document.getElementById('audioPlayerContainer');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  status.textContent = 'Sending to TTS service — this may take a few seconds...';
  container.style.display = 'none';

  try {
    const res = await fetch('/api/tts/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        voice_key: voiceKey,
        temperature: parseFloat(document.getElementById('ttsTemp').value),
        top_p: parseFloat(document.getElementById('ttsTopP').value),
        repetition_penalty: parseFloat(document.getElementById('ttsRep').value),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const player = document.getElementById('audioPlayer');
    player.src = url;
    document.getElementById('audioDownload').href = url;
    container.style.display = 'flex';
    status.textContent = 'Done!';
    player.play();
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Generate Speech';
}

// ── Init ──────────────────────────────────────────────────────

document.getElementById('input').focus();
