// Restaurant AI Receptionist — browser demo
// Server injects config into window.APP_CONFIG before this script runs.

const { vapiPubKey, vapiAssistantId, restaurantName, agentName, vapiReady } = window.APP_CONFIG;

let sessionId = 'demo-' + Math.random().toString(36).slice(2, 10);
window.__callActive = false;
window.__vapi = null;

// ── Page Navigation ───────────────────────────────────────────

let _currentPage = 'call-chat';
const _pageLoaded = {};

function switchPage(pageName) {
  if (pageName === _currentPage) return;

  document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.page-content').forEach(p => p.classList.remove('active'));

  const btn = document.querySelector(`.nav-link[onclick="switchPage('${pageName}')"]`);
  if (btn) btn.classList.add('active');
  const page = document.getElementById(`page-${pageName}`);
  if (page) page.classList.add('active');

  _currentPage = pageName;

  if (pageName === 'dashboard' && !_pageLoaded.dashboard) {
    _pageLoaded.dashboard = true;
    loadDashboard();
  }
  if (pageName === 'reservations' && !_pageLoaded.reservations) {
    _pageLoaded.reservations = true;
    loadReservations();
  }
  if (pageName === 'orders' && !_pageLoaded.orders) {
    _pageLoaded.orders = true;
    loadOrders();
  }
  if (pageName === 'menu' && !_pageLoaded.menu) {
    _pageLoaded.menu = true;
    loadMenu();
  }
  if (pageName === 'knowledge' && !_pageLoaded.knowledge) {
    _pageLoaded.knowledge = true;
    loadKnowledge();
  }
  if (pageName === 'settings' && !_pageLoaded.settings) {
    _pageLoaded.settings = true;
    loadSettings();
  }

  // Close sidebar on mobile
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('open');
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarOverlay').classList.toggle('open');
}

// ── Voice call (Vapi) ─────────────────────────────────────────

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
    const res = await fetch('/api/retell/web-call', {
      method: 'POST',
      headers: authHeaders(),
    });
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

let retellShownTurns = 0;

window.renderRetellTranscript = function (transcript) {
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
  const av = role === 'agent' ? '🍽️' : '👤';
  d.innerHTML = `<div class="msg-avatar">${av}</div><div class="bubble">${escapeHtml(text)}</div>`;
  c.appendChild(d);
  c.scrollTop = c.scrollHeight;
}

function showTyping() {
  const c = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg agent';
  d.id = 'typing';
  d.innerHTML = '<div class="msg-avatar">🍽️</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
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
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
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
  addMessage('agent', `Hi, you've reached ${restaurantName}. This is ${agentName}. How can I help you today?`);
}

// ── Utilities ─────────────────────────────────────────────────

function fmtDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return d.toLocaleString(undefined, {
    weekday: 'short', day: 'numeric', month: 'short',
    hour: 'numeric', minute: '2-digit',
  });
}

function fmtCurrency(val) {
  return '$' + Number(val || 0).toFixed(2);
}

function authHeaders() {
  const token = window.APP_CONFIG.csrfToken;
  return token ? { 'X-CSRF-Token': token } : {};
}

// ── Dashboard ─────────────────────────────────────────────────

let _historyCache = null;

async function fetchHistory() {
  if (_historyCache) return _historyCache;
  const res = await fetch('/api/history', { headers: authHeaders() });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  _historyCache = await res.json();
  return _historyCache;
}

function invalidateCache() {
  _historyCache = null;
}

async function loadDashboard() {
  try {
    const [statsRes, history] = await Promise.all([
      fetch('/api/stats', { headers: authHeaders() }).then(r => r.ok ? r.json() : null).catch(() => null),
      fetchHistory(),
    ]);

    const reservations = history.reservations || [];
    const pickups = history.pickup_orders || [];
    const allOrders = [
      ...reservations.filter(r => r.preorder).map(r => r.preorder),
      ...pickups,
    ];

    document.getElementById('statTotalCalls').textContent =
      reservations.length + pickups.length;
    document.getElementById('statTotalReservations').textContent =
      statsRes ? statsRes.total_reservations : reservations.length;
    document.getElementById('statTotalOrders').textContent =
      statsRes ? statsRes.total_orders : allOrders.length;
    document.getElementById('statTotalRevenue').textContent =
      fmtCurrency(statsRes ? statsRes.total_revenue : allOrders.reduce((s, o) => s + (o.total_amount || 0), 0));

    // Recent reservations (top 5)
    const recentRes = reservations.slice(0, 5);
    const resContainer = document.getElementById('dashRecentReservations');
    if (recentRes.length === 0) {
      resContainer.innerHTML = '<div class="empty-state">No reservations yet</div>';
    } else {
      resContainer.innerHTML = recentRes.map(r => `
        <div class="dash-item">
          <div class="dash-item-info">
            <div class="dash-item-name">${escapeHtml(r.customer_name)}</div>
            <div class="dash-item-meta">${fmtDateTime(r.booked_at)} · ${r.party_size} guests</div>
          </div>
          <span class="status-badge ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span>
        </div>
      `).join('');
    }

    // Recent orders (top 5)
    const recentOrders = allOrders.slice(0, 5);
    const ordContainer = document.getElementById('dashRecentOrders');
    if (recentOrders.length === 0) {
      ordContainer.innerHTML = '<div class="empty-state">No orders yet</div>';
    } else {
      ordContainer.innerHTML = recentOrders.map(o => `
        <div class="dash-item">
          <div class="dash-item-info">
            <div class="dash-item-name">${escapeHtml(o.customer_name || 'Guest')}</div>
            <div class="dash-item-meta">${fmtDateTime(o.created_at)} · ${(o.items || []).length} items</div>
          </div>
          <span class="td-price">${fmtCurrency(o.total_amount)}</span>
        </div>
      `).join('');
    }
  } catch (e) {
    console.error('Dashboard load error:', e);
    document.getElementById('dashRecentReservations').innerHTML =
      `<div class="empty-state">Failed to load: ${escapeHtml(e.message)}</div>`;
  }
}

// ── Reservations page ─────────────────────────────────────────

async function loadReservations() {
  const tbody = document.getElementById('reservationsBody');
  tbody.innerHTML = '<tr><td colspan="7" class="empty-state">Loading...</td></tr>';

  try {
    invalidateCache();
    const history = await fetchHistory();
    const reservations = history.reservations || [];

    if (reservations.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No reservations yet</td></tr>';
      return;
    }

    tbody.innerHTML = reservations.map(r => {
      const tableInfo = r.table_number
        ? `Table ${r.table_number}<br><small style="color:var(--text-dim)">${escapeHtml(r.location || '')}</small>`
        : '—';
      const notes = r.notes
        ? `<span class="td-notes">${escapeHtml(r.notes)}</span>`
        : '<span style="color:var(--text-dim)">—</span>';
      return `
        <tr>
          <td>${fmtDateTime(r.booked_at)}</td>
          <td class="td-name">${escapeHtml(r.customer_name)}</td>
          <td>${escapeHtml(r.customer_phone || '—')}</td>
          <td>${r.party_size}</td>
          <td>${tableInfo}</td>
          <td><span class="status-badge ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span></td>
          <td>${notes}</td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
  }
}

// ── Knowledge page ────────────────────────────────────────────

async function loadKnowledge() {
  const gapsBody = document.getElementById('knowledgeGapsBody');
  const faqBody = document.getElementById('knowledgeFaqBody');
  gapsBody.innerHTML = '<tr><td colspan="4" class="empty-state">Loading...</td></tr>';
  faqBody.innerHTML = '<tr><td colspan="3" class="empty-state">Loading...</td></tr>';

  try {
    const res = await fetch('/api/knowledge/gaps', { headers: authHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const unresolved = (data.gaps || []).filter(g => g.status === 'unresolved');
    const faq = data.knowledge || [];

    if (unresolved.length === 0) {
      gapsBody.innerHTML = '<tr><td colspan="4" class="empty-state">No unanswered questions</td></tr>';
    } else {
      gapsBody.innerHTML = unresolved.map(gap => `
        <tr>
          <td>${fmtDateTime(gap.created_at)}</td>
          <td class="td-name">${escapeHtml(gap.question)}</td>
          <td><span class="td-notes">${escapeHtml(gap.context_excerpt || '—')}</span></td>
          <td>
            <form class="knowledge-resolve" onsubmit="return resolveKnowledgeGap(event, ${gap.id})">
              <textarea name="answer" rows="2" required placeholder="Answer the host should use next time"></textarea>
              <button type="submit" class="btn-small">Save answer</button>
            </form>
          </td>
        </tr>
      `).join('');
    }

    if (faq.length === 0) {
      faqBody.innerHTML = '<tr><td colspan="3" class="empty-state">No saved answers yet</td></tr>';
    } else {
      faqBody.innerHTML = faq.map(item => `
        <tr>
          <td class="td-name">${escapeHtml(item.question)}</td>
          <td>${escapeHtml(item.answer)}</td>
          <td>${fmtDateTime(item.updated_at || item.created_at)}</td>
        </tr>
      `).join('');
    }
  } catch (e) {
    gapsBody.innerHTML = `<tr><td colspan="4" class="empty-state">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
    faqBody.innerHTML = `<tr><td colspan="3" class="empty-state">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
  }
}

async function resolveKnowledgeGap(event, gapId) {
  event.preventDefault();
  const answer = event.target.answer.value.trim();
  if (!answer) return false;
  const button = event.target.querySelector('button');
  button.disabled = true;
  try {
    const res = await fetch(`/api/knowledge/gaps/${gapId}/resolve`, {
      method: 'POST',
      headers: { ...authHeaders(), 'Content-Type': 'application/json' },
      body: JSON.stringify({ answer }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    await loadKnowledge();
  } catch (e) {
    button.disabled = false;
    alert('Could not save answer: ' + e.message);
  }
  return false;
}

// ── Orders page ───────────────────────────────────────────────

async function loadOrders() {
  const tbody = document.getElementById('ordersBody');
  tbody.innerHTML = '<tr><td colspan="7" class="empty-state">Loading...</td></tr>';

  try {
    invalidateCache();
    const history = await fetchHistory();
    const reservations = history.reservations || [];
    const pickups = history.pickup_orders || [];

    const allOrders = [];

    reservations.forEach(r => {
      if (r.preorder) {
        allOrders.push({
          ...r.preorder,
          orderType: 'dine-in',
          customer_name: r.preorder.customer_name || r.customer_name,
          customer_phone: r.preorder.customer_phone || r.customer_phone,
        });
      }
    });
    pickups.forEach(o => {
      allOrders.push({ ...o, orderType: 'takeaway' });
    });

    if (allOrders.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No orders yet</td></tr>';
      return;
    }

    tbody.innerHTML = allOrders.map(o => {
      const items = (o.items || []);
      const itemsHtml = items.length > 0
        ? `<div class="td-items-list">${items.map(it =>
            `<span><span class="item-qty">${it.quantity}×</span> ${escapeHtml(it.item_name)}</span>`
          ).join('')}</div>`
        : '<span style="color:var(--text-dim)">No items</span>';

      const typeCls = o.orderType === 'dine-in' ? 'dine-in' : 'takeaway';

      return `
        <tr>
          <td class="td-name">${escapeHtml(o.customer_name || 'Guest')}</td>
          <td>${escapeHtml(o.customer_phone || '—')}</td>
          <td><span class="type-badge ${typeCls}">${o.orderType === 'dine-in' ? 'Dine-in' : 'Takeaway'}</span></td>
          <td class="td-items">${itemsHtml}</td>
          <td class="td-price">${fmtCurrency(o.total_amount)}</td>
          <td><span class="status-badge ${escapeHtml(o.status)}">${escapeHtml(o.status)}</span></td>
          <td>${fmtDateTime(o.created_at)}</td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">Failed to load: ${escapeHtml(e.message)}</td></tr>`;
  }
}

// ── Menu page ─────────────────────────────────────────────────

const CATEGORY_ICONS = {
  starter: '🥗',
  main: '🍖',
  dessert: '🍰',
  drink: '☕',
  special: '⭐',
};

const CATEGORY_LABELS = {
  starter: 'Starters',
  main: 'Main Courses',
  dessert: 'Desserts',
  drink: 'Drinks',
  special: 'Specials',
};

let _menuManageMode = false;

function toggleMenuManageMode(on) {
  _menuManageMode = on;
  const display = document.getElementById('menuDisplay');
  const page = document.getElementById('page-menu');
  if (on) {
    display.classList.add('menu-manage-mode');
    page.classList.add('menu-manage-active');
  } else {
    display.classList.remove('menu-manage-mode');
    page.classList.remove('menu-manage-active');
  }
}

async function toggleItemAvailability(itemId, checkbox) {
  const available = checkbox.checked;
  const label = checkbox.closest('.menu-item-actions').querySelector('.avail-label');
  const menuItem = checkbox.closest('.menu-item');

  try {
    const res = await fetch(`/api/menu/${itemId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ available }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    if (label) {
      label.textContent = available ? 'Available' : 'Unavailable';
      label.className = 'avail-label ' + (available ? 'on' : 'off');
    }
    if (menuItem) {
      menuItem.classList.toggle('menu-item-unavailable', !available);
    }
  } catch (e) {
    console.error('Toggle availability error:', e);
    checkbox.checked = !available;
  }
}

async function loadMenu() {
  const container = document.getElementById('menuDisplay');
  container.innerHTML = '<div class="empty-state">Loading menu...</div>';

  try {
    const res = await fetch('/api/menu');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const categories = data.categories || {};

    const order = ['starter', 'main', 'dessert', 'drink', 'special'];
    const sortedKeys = Object.keys(categories).sort(
      (a, b) => (order.indexOf(a) === -1 ? 99 : order.indexOf(a)) - (order.indexOf(b) === -1 ? 99 : order.indexOf(b))
    );

    if (sortedKeys.length === 0) {
      container.innerHTML = '<div class="empty-state">No menu items found</div>';
      return;
    }

    container.innerHTML = sortedKeys.map(cat => {
      const items = categories[cat];
      const icon = CATEGORY_ICONS[cat] || '🍴';
      const label = CATEGORY_LABELS[cat] || cat.charAt(0).toUpperCase() + cat.slice(1);
      return `
        <div class="menu-category">
          <div class="menu-category-header">
            <span style="font-size:24px">${icon}</span>
            <h2>${label}</h2>
            <span class="menu-category-count">${items.length} items</span>
          </div>
          <div class="menu-items-grid">
            ${items.map(item => {
              const tags = (item.dietary || []).map(d =>
                `<span class="dietary-tag">${escapeHtml(d)}</span>`
              ).join('');
              const unavailableClass = item.available ? '' : ' menu-item-unavailable';
              const checkedAttr = item.available ? 'checked' : '';
              const availText = item.available ? 'Available' : 'Unavailable';
              const availCls = item.available ? 'on' : 'off';
              return `
                <div class="menu-item${unavailableClass}">
                  <div class="menu-item-info">
                    <div class="menu-item-name">${escapeHtml(item.name)}</div>
                    <div class="menu-item-desc">${escapeHtml(item.description)}</div>
                    ${tags ? `<div class="menu-item-tags">${tags}</div>` : ''}
                  </div>
                  <div class="menu-item-price">${fmtCurrency(item.price)}</div>
                  <div class="menu-item-actions">
                    <label class="avail-switch">
                      <input type="checkbox" ${checkedAttr} onchange="toggleItemAvailability(${item.id}, this)">
                      <span class="avail-switch-track"></span>
                    </label>
                    <span class="avail-label ${availCls}">${availText}</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    }).join('');

    if (_menuManageMode) {
      container.classList.add('menu-manage-mode');
    }
  } catch (e) {
    container.innerHTML = `<div class="empty-state">Failed to load menu: ${escapeHtml(e.message)}</div>`;
  }
}

// ── Add new menu item ─────────────────────────────────────────

async function addNewMenuItem() {
  const name = document.getElementById('newItemName').value.trim();
  const category = document.getElementById('newItemCategory').value;
  const description = document.getElementById('newItemDesc').value.trim();
  const price = parseFloat(document.getElementById('newItemPrice').value);
  const dietaryRaw = document.getElementById('newItemDietary').value.trim();
  const dietary = dietaryRaw ? dietaryRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
  const status = document.getElementById('addItemStatus');

  if (!name) { status.style.color = 'var(--red)'; status.textContent = 'Item name is required'; return; }
  if (!price || price <= 0) { status.style.color = 'var(--red)'; status.textContent = 'Valid price is required'; return; }

  status.style.color = 'var(--text-muted)';
  status.textContent = 'Adding...';

  try {
    const res = await fetch('/api/menu', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ name, category, description, price, dietary }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    status.style.color = 'var(--green)';
    status.textContent = `"${name}" added successfully!`;

    document.getElementById('newItemName').value = '';
    document.getElementById('newItemDesc').value = '';
    document.getElementById('newItemPrice').value = '';
    document.getElementById('newItemDietary').value = '';

    _pageLoaded.menu = false;
    loadMenu();
    _pageLoaded.menu = true;

    setTimeout(() => { status.textContent = ''; }, 3000);
  } catch (e) {
    status.style.color = 'var(--red)';
    status.textContent = 'Failed: ' + e.message;
  }
}

// ── Settings page ─────────────────────────────────────────────

const LANG_CHECKBOXES = {
  English: 'langEn',
  French: 'langFr',
  Spanish: 'langEs',
  Arabic: 'langAr',
  Urdu: 'langUr',
};

const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];

async function loadSettings() {
  try {
    const res = await fetch('/api/settings', { headers: authHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();

    document.getElementById('setName').value = s.restaurant_name || '';
    document.getElementById('setPhone').value = s.phone_number || '';
    document.getElementById('setTimezone').value = s.timezone || 'UTC';
    document.getElementById('setCapacity').value = s.seating_capacity || '';
    document.getElementById('setAddress').value = s.street_address || '';
    document.getElementById('setCity').value = s.city || '';
    document.getElementById('setAgentName').value = s.ai_agent_name || 'Clough';

    const langs = s.languages || ['English'];
    Object.entries(LANG_CHECKBOXES).forEach(([lang, id]) => {
      const cb = document.getElementById(id);
      if (cb) cb.checked = langs.includes(lang);
    });

    const hours = s.opening_hours || {};
    DAYS.forEach(day => {
      const h = hours[day] || {};
      const openEl = document.getElementById(`hours-${day}-open`);
      const closeEl = document.getElementById(`hours-${day}-close`);
      if (openEl && h.open) openEl.value = h.open;
      if (closeEl && h.close) closeEl.value = h.close;
    });
  } catch (e) {
    console.error('Load settings error:', e);
    document.getElementById('settingsStatus').textContent = 'Failed to load settings';
    document.getElementById('settingsStatus').style.color = 'var(--red)';
  }
}

async function saveSettings() {
  const btn = document.querySelector('.btn-save');
  const status = document.getElementById('settingsStatus');
  btn.disabled = true;
  status.textContent = '';

  const languages = [];
  Object.entries(LANG_CHECKBOXES).forEach(([lang, id]) => {
    const cb = document.getElementById(id);
    if (cb && cb.checked) languages.push(lang);
  });

  const opening_hours = {};
  DAYS.forEach(day => {
    const open = document.getElementById(`hours-${day}-open`).value;
    const close = document.getElementById(`hours-${day}-close`).value;
    if (open && close) {
      opening_hours[day] = { open, close };
    }
  });

  const payload = {
    restaurant_name: document.getElementById('setName').value.trim(),
    phone_number: document.getElementById('setPhone').value.trim(),
    timezone: document.getElementById('setTimezone').value,
    seating_capacity: parseInt(document.getElementById('setCapacity').value) || 60,
    street_address: document.getElementById('setAddress').value.trim(),
    city: document.getElementById('setCity').value.trim(),
    ai_agent_name: document.getElementById('setAgentName').value.trim() || 'Clough',
    languages,
    opening_hours,
    hours_unconfirmed: Object.keys(opening_hours).length === 0,
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    status.style.color = 'var(--green)';
    status.textContent = 'Settings saved successfully';
    setTimeout(() => { status.textContent = ''; }, 3000);
  } catch (e) {
    status.style.color = 'var(--red)';
    status.textContent = 'Failed to save: ' + e.message;
  }

  btn.disabled = false;
}

// ── Init ──────────────────────────────────────────────────────

document.getElementById('input').focus();
