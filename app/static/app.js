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

// ── Init ──────────────────────────────────────────────────────

document.getElementById('input').focus();
