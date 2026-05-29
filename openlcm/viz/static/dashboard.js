/* OpenLCM Dashboard — Vanilla JS, no dependencies */

const API = '';
const state = {
  connected: false,
  engine: null,   // latest /api/status
  dag: null,      // latest /api/dag
  messages: [],   // recent messages
  events: [],     // timeline events
  sessions: [],   // session list
  pressure: { prompt: 0, threshold: 0, max: 0, ratio: 0 },
};

let eventSource = null;

// ── Initialise ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  poll();
  setInterval(poll, 8000);

  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => runSearch(searchInput.value.trim()), 400);
    });
  }
});

// ── SSE connection ─────────────────────────────────────────────────────────

function connectSSE() {
  if (eventSource) { eventSource.close(); }
  eventSource = new EventSource(`${API}/events`);

  eventSource.onopen = () => {
    state.connected = true;
    renderConnectionStatus(true);
  };

  eventSource.onerror = () => {
    state.connected = false;
    renderConnectionStatus(false);
    setTimeout(connectSSE, 4000);
  };

  eventSource.onmessage = (e) => {
    try {
      const evt = JSON.parse(e.data);
      if (evt.type === 'ping') return;
      handleEvent(evt);
    } catch (_) {}
  };
}

function handleEvent(evt) {
  const { type, data, ts } = evt;

  // Push to timeline (newest first, cap at 200)
  state.events.unshift({ type, data, ts });
  if (state.events.length > 200) state.events.pop();
  renderTimeline();

  switch (type) {
    case 'token_pressure':
      state.pressure = {
        prompt: data.prompt_tokens || 0,
        threshold: data.threshold_tokens || 0,
        max: data.context_length || 0,
        ratio: data.ratio || 0,
      };
      renderGauge();
      break;

    case 'message_ingested':
      // Refresh feed from API to get content
      fetchMessages();
      break;

    case 'compaction_end':
      // Refresh everything after compaction
      fetchStatus();
      fetchDAG();
      fetchMessages();
      break;

    case 'node_added':
    case 'node_condensed':
      fetchDAG();
      break;

    case 'session_bound':
      fetchStatus();
      fetchDAG();
      fetchMessages();
      fetchSessions();
      break;
  }
}

// ── Polling ────────────────────────────────────────────────────────────────

async function poll() {
  await Promise.allSettled([fetchStatus(), fetchDAG(), fetchMessages(), fetchSessions()]);
}

async function fetchStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    if (!r.ok) return;
    state.engine = await r.json();
    if (state.engine.last_prompt_tokens && state.engine.threshold_tokens) {
      state.pressure.prompt = state.engine.last_prompt_tokens;
      state.pressure.threshold = state.engine.threshold_tokens;
      state.pressure.max = state.engine.context_length || 0;
      state.pressure.ratio = state.pressure.max > 0 ? state.pressure.prompt / state.pressure.max : 0;
    }
    renderGauge();
    renderSessionInfo();
  } catch (_) {}
}

async function fetchDAG() {
  try {
    const r = await fetch(`${API}/api/dag`);
    if (!r.ok) return;
    state.dag = await r.json();
    renderDAG();
  } catch (_) {}
}

async function fetchMessages() {
  try {
    const r = await fetch(`${API}/api/messages?limit=60`);
    if (!r.ok) return;
    const data = await r.json();
    state.messages = data.messages || [];
    renderFeed();
  } catch (_) {}
}

async function fetchSessions() {
  try {
    const r = await fetch(`${API}/api/sessions`);
    if (!r.ok) return;
    const data = await r.json();
    state.sessions = data.sessions || [];
    renderSessions();
  } catch (_) {}
}

// ── Gauge ──────────────────────────────────────────────────────────────────

function renderGauge() {
  const { prompt, threshold, max, ratio } = state.pressure;

  // SVG arc: total circumference of half-circle (r=80, 0..180 deg)
  const r = 72, cx = 100, cy = 100;
  const circum = Math.PI * r; // half circle arc length
  const fillPct = Math.min(ratio, 1);
  const fill = fillPct * circum;

  const color = ratio >= 0.9 ? 'var(--red)' : ratio >= 0.65 ? 'var(--yellow)' : 'var(--green)';

  // Arc path: start at 180deg (left), end at 0deg (right)
  const gaugeEl = document.getElementById('gauge-fill');
  if (gaugeEl) {
    gaugeEl.style.strokeDasharray = `${fill} ${circum}`;
    gaugeEl.style.stroke = color;
  }

  setText('gauge-pct', `${(ratio * 100).toFixed(1)}%`);
  setText('gauge-pct', `${(ratio * 100).toFixed(1)}%`);
  setTextColor('gauge-pct', color);
  setText('gauge-prompt', fmt(prompt));
  setText('gauge-threshold', fmt(threshold));
  setText('gauge-max', fmt(max));
}

// ── DAG ───────────────────────────────────────────────────────────────────

function renderDAG() {
  const el = document.getElementById('dag-container');
  if (!el) return;
  const nodes = (state.dag && state.dag.nodes) || [];

  if (!nodes.length) {
    el.innerHTML = '<div class="dag-empty">No summary nodes yet.<br>Nodes appear after first compaction.</div>';
    return;
  }

  // Group by depth
  const byDepth = {};
  for (const n of nodes) {
    if (!byDepth[n.depth]) byDepth[n.depth] = [];
    byDepth[n.depth].push(n);
  }

  const depthLabels = { 0: 'D0 — Leaf (Recent)', 1: 'D1 — Arc (Session)', 2: 'D2 — Durable' };
  const depths = Object.keys(byDepth).map(Number).sort((a, b) => b - a);

  el.innerHTML = depths.map(d => {
    const ns = byDepth[d].sort((a, b) => a.node_id - b.node_id);
    const label = depthLabels[d] || `D${d} — Depth ${d}`;
    const cards = ns.map(n => `
      <div class="dag-node" data-node-id="${n.node_id}"
           onmouseenter="showNodePreview(event,${n.node_id})"
           onmouseleave="hideNodePreview()">
        <div class="dag-node-id">#${n.node_id}</div>
        <div class="dag-node-tokens">${fmt(n.token_count)} tok <span style="color:var(--text-dim);font-size:10px">/ ${fmt(n.source_token_count)} src</span></div>
        <div class="dag-node-hint">${esc(n.expand_hint || '')}</div>
      </div>
    `).join('');
    return `<div class="dag-depth"><div class="dag-depth-label">${esc(label)}</div><div class="dag-nodes">${cards}</div></div>`;
  }).join('');
}

function showNodePreview(event, nodeId) {
  const node = (state.dag && state.dag.nodes || []).find(n => n.node_id === nodeId);
  if (!node) return;
  const el = document.getElementById('node-preview');
  if (!el) return;
  el.innerHTML = `<b>#${node.node_id} D${node.depth}</b><br>${esc(node.summary_preview || '')}`;
  el.style.left = (event.clientX + 16) + 'px';
  el.style.top = (event.clientY - 20) + 'px';
  el.classList.add('visible');
}

function hideNodePreview() {
  const el = document.getElementById('node-preview');
  if (el) el.classList.remove('visible');
}

// Flash a newly added node
function flashNode(nodeId) {
  const el = document.querySelector(`[data-node-id="${nodeId}"]`);
  if (el) { el.classList.add('flash'); setTimeout(() => el.classList.remove('flash'), 700); }
}

// ── Feed ──────────────────────────────────────────────────────────────────

function renderFeed() {
  const el = document.getElementById('feed-container');
  if (!el) return;
  const msgs = [...state.messages].reverse(); // newest first
  if (!msgs.length) {
    el.innerHTML = '<div style="color:var(--text-dim);text-align:center;padding:40px 0">No messages yet</div>';
    return;
  }
  el.innerHTML = msgs.map((m, i) => {
    const role = m.role || 'unknown';
    const preview = (m.content_preview || '').trim().slice(0, 200);
    const time = m.timestamp ? new Date(m.timestamp * 1000).toLocaleTimeString() : '';
    const tokens = m.token_estimate ? `${m.token_estimate}t` : '';
    return `
      <div class="feed-msg ${role} ${i === 0 ? 'new-msg' : ''}">
        <div class="feed-msg-header">
          <span class="feed-msg-role ${role}">${esc(role)}</span>
          <span class="feed-msg-meta">${esc(time)} ${esc(tokens)}</span>
        </div>
        <div class="feed-msg-content">${esc(preview)}${preview.length >= 200 ? '…' : ''}</div>
      </div>`;
  }).join('');
}

// ── Timeline ──────────────────────────────────────────────────────────────

function renderTimeline() {
  const el = document.getElementById('timeline-container');
  if (!el) return;
  if (!state.events.length) {
    el.innerHTML = '<div class="timeline-empty">Waiting for events…</div>';
    return;
  }
  el.innerHTML = state.events.slice(0, 80).map(evt => {
    const time = evt.ts ? new Date(evt.ts * 1000).toLocaleTimeString() : '';
    const detail = formatEventDetail(evt.type, evt.data);
    return `
      <div class="timeline-event ${evt.type}">
        <div class="timeline-event-header">
          <span class="timeline-event-type">${esc(evt.type.replace(/_/g, ' '))}</span>
          <span class="timeline-event-time">${esc(time)}</span>
        </div>
        <div class="timeline-event-detail">${detail}</div>
      </div>`;
  }).join('');
}

function formatEventDetail(type, data) {
  if (!data) return '';
  switch (type) {
    case 'compaction_end':
      return `${data.messages_before} → ${data.messages_after} msgs &nbsp;·&nbsp; ${fmt(data.tokens_before)} → ${fmt(data.tokens_after)} tokens &nbsp;·&nbsp; ${data.dag_nodes} DAG nodes`;
    case 'node_added':
      return `Node #${data.node_id} D${data.depth} &nbsp;·&nbsp; ${fmt(data.token_count)} tokens (from ${fmt(data.source_token_count)})`;
    case 'node_condensed':
      return `D${data.depth} &nbsp;·&nbsp; ${(data.input_node_ids || []).length} → 1 node (#${data.output_node_id})`;
    case 'token_pressure':
      return `${fmt(data.prompt_tokens)} / ${fmt(data.threshold_tokens)} threshold &nbsp;·&nbsp; ${(data.ratio * 100).toFixed(1)}%`;
    case 'session_bound':
      return `${data.session_id || ''} (${data.platform || 'unknown'})`;
    case 'compaction_start':
      return `${data.messages_count} messages &nbsp;·&nbsp; ${fmt(data.prompt_tokens)} tokens`;
    case 'message_ingested':
      return `${data.role} #${data.store_id} &nbsp;·&nbsp; ${data.token_estimate}t`;
    default:
      return JSON.stringify(data).slice(0, 100);
  }
}

// ── Session info ──────────────────────────────────────────────────────────

function renderSessionInfo() {
  const s = state.engine;
  if (!s) return;
  setText('si-session', s.session_id || '(none)');
  setText('si-platform', s.platform || '—');
  setText('si-compressions', s.compression_count ?? 0);
  setText('si-messages', s.store_messages ?? 0);
  setText('si-nodes', s.dag_nodes ?? 0);
  setText('si-threshold', s.threshold_percent ? `${(s.threshold_percent * 100).toFixed(0)}%` : '75%');
  setText('si-tail', s.fresh_tail_count ?? 64);
  setText('si-model', s.summary_model || 'auto');
  setText('si-db', s.db_path ? s.db_path.split('/').pop() : '—');
  setText('si-status', s.last_compression_status || 'idle');
}

function renderSessions() {
  const el = document.getElementById('sessions-list');
  if (!el) return;
  const current = state.engine && state.engine.session_id;
  el.innerHTML = state.sessions.map(s => `
    <div class="session-item ${s.session_id === current ? 'active' : ''}"
         onclick="selectSession('${esc(s.session_id)}')">
      <div class="session-item-id">${esc(s.session_id.slice(0, 30))}${s.session_id.length > 30 ? '…' : ''}</div>
      <div class="session-item-meta">${s.message_count} msgs &nbsp;·&nbsp; ${s.last_at ? new Date(s.last_at * 1000).toLocaleDateString() : ''}</div>
    </div>`).join('');
}

function selectSession(sessionId) {
  // Switch viewed session — just updates UI view (does not change engine binding)
  state.engine = state.engine || {};
  state.engine.session_id = sessionId;
  fetchMessages();
  fetchDAG();
  renderSessions();
}

// ── Search ────────────────────────────────────────────────────────────────

async function runSearch(query) {
  const el = document.getElementById('search-results');
  if (!el) return;
  if (!query) { el.innerHTML = ''; return; }
  try {
    const r = await fetch(`${API}/api/grep`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 8 }),
    });
    const data = await r.json();
    const hits = data.results || [];
    if (!hits.length) { el.innerHTML = '<div style="color:var(--text-dim);font-size:11px;padding:4px">No results</div>'; return; }
    el.innerHTML = hits.map(h => `
      <div class="search-hit">
        <div class="search-hit-role">${esc(h.role || 'unknown')}</div>
        <div class="search-hit-snippet">${highlight(h.snippet || '', query)}</div>
      </div>`).join('');
  } catch (_) { el.innerHTML = ''; }
}

// ── Connection status ─────────────────────────────────────────────────────

function renderConnectionStatus(connected) {
  const dot = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  if (dot) dot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
  if (label) label.textContent = connected ? 'Live' : 'Disconnected';
}

// ── Utilities ─────────────────────────────────────────────────────────────

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setTextColor(id, color) {
  const el = document.getElementById(id);
  if (el) el.style.color = color;
}

function fmt(n) {
  if (n == null || n === '') return '—';
  const num = Number(n);
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
  if (num >= 1_000) return (num / 1_000).toFixed(1) + 'K';
  return String(num);
}

function esc(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function highlight(text, query) {
  const escaped = esc(text);
  if (!query) return escaped;
  const terms = query.split(/\s+/).filter(Boolean).map(t => esc(t));
  let result = escaped;
  for (const term of terms) {
    result = result.replace(new RegExp(term, 'gi'), m => `<mark>${m}</mark>`);
  }
  return result;
}
