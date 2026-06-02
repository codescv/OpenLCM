/* OpenLCM Dashboard — multi-session SPA */

// ── Global state ─────────────────────────────────────────────────────────────
const S = {
  page: 'overview',
  sessionId: null,
  sessions: [],
  totals: {},
  engine: null,
  dag: null,
  messages: [],
  facts: [],
  selNode: null,
  selectedNodeData: null,
  pressure: { prompt: 0, threshold: 0, max: 0, ratio: 0 },
  events: [],
  feedCount: 0,
};
let _sse = null;
let _ovInterval = null;
let _sessInterval = null;

// ── Router ────────────────────────────────────────────────────────────────────

function navigate(page, sessionId) {
  if (page === 'overview') {
    window.location.hash = 'overview';
  } else if (page === 'session' && sessionId) {
    window.location.hash = `session/${sessionId}`;
  }
}

function navigateSessionTab() {
  if (S.sessionId) navigate('session', S.sessionId);
}

function resolveRoute() {
  const hash = window.location.hash.replace('#', '') || 'overview';
  if (hash.startsWith('session/')) {
    const sid = hash.slice('session/'.length);
    showPage('session', sid);
  } else {
    showPage('overview');
  }
}

function showPage(page, sessionId) {
  S.page = page;
  S.sessionId = sessionId || null;

  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nb-tab').forEach(t => t.classList.remove('active'));

  clearInterval(_ovInterval);
  clearInterval(_sessInterval);

  if (page === 'overview') {
    document.getElementById('page-overview').classList.add('active');
    document.querySelector('.nb-tab[data-page="overview"]').classList.add('active');
    document.getElementById('nb-tab-session').style.display = 'none';
    startOverview();
  } else if (page === 'session') {
    document.getElementById('page-session').classList.add('active');
    const tab = document.getElementById('nb-tab-session');
    tab.style.display = '';
    tab.classList.add('active');
    document.querySelector('.nb-tab[data-page="overview"]').classList.remove('active');
    document.getElementById('nb-session-label').textContent = sessionId || 'Session';
    startSessionDetail(sessionId);
  }
}

window.addEventListener('hashchange', resolveRoute);

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  resolveRoute();

  const si = document.getElementById('search-input');
  if (si) {
    let t;
    si.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(() => runSearch(si.value.trim()), 400);
    });
  }

  document.getElementById('nd-close').addEventListener('click', closeNodeDetail);
});

function forceRefresh() {
  if (S.page === 'overview') pollOverview();
  else pollSession();
}

// ── SSE ───────────────────────────────────────────────────────────────────────

function connectSSE() {
  if (_sse) _sse.close();
  _sse = new EventSource('/events');
  _sse.onerror = () => { setTimeout(connectSSE, 4000); };
  _sse.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      if (ev.type !== 'ping') handleEvent(ev);
    } catch (_) {}
  };
}

function handleEvent(ev) {
  const { type, data, ts } = ev;
  S.events.unshift({ type, data, ts });
  if (S.events.length > 500) S.events.pop();

  // Feed (overview)
  appendFeed(ev);

  // Session detail log
  if (S.page === 'session') appendLog(ev);

  // Reactive refreshes
  switch (type) {
    case 'token_pressure':
      S.pressure = {
        prompt: data.prompt_tokens || 0,
        threshold: data.threshold_tokens || 0,
        max: data.context_length || 0,
        ratio: data.ratio || 0,
      };
      if (S.page === 'session') renderPressure();
      break;
    case 'message_ingested':
      if (S.page === 'session') fetchMessages();
      break;
    case 'compaction_end':
    case 'node_added':
    case 'node_condensed':
    case 'node_deleted':
    case 'dag_cleared':
      if (S.page === 'session') { fetchDAG(); fetchStatus(); }
      if (S.page === 'overview') pollOverview();
      break;
    case 'session_deleted':
    case 'messages_cleared':
      if (S.page === 'overview') pollOverview();
      if (S.page === 'session') { fetchMessages(); fetchDAG(); fetchStatus(); }
      break;
    case 'session_bound':
      if (S.page === 'overview') pollOverview();
      if (S.page === 'session') { fetchStatus(); fetchDAG(); fetchMessages(); }
      break;
    case 'fact_stored':
    case 'fact_deleted':
      if (S.page === 'session') fetchFacts();
      if (S.page === 'overview') pollOverview();
      break;
  }
}

// ── Overview page ─────────────────────────────────────────────────────────────

function startOverview() {
  pollOverview();
  loadFeedHistory();
  _ovInterval = setInterval(pollOverview, 6000);
}

async function loadFeedHistory() {
  try {
    const r = await fetch('/api/events/history');
    if (!r.ok) return;
    const data = await r.json();
    const events = (data.events || []).slice(-60); // last 60 events
    if (!events.length) return;
    for (const ev of events) appendFeed(ev);
  } catch (_) {}
}

async function pollOverview() {
  try {
    const r = await fetch('/api/overview');
    if (!r.ok) return;
    const data = await r.json();
    S.sessions = data.sessions || [];
    S.totals = data.totals || {};
    renderOverview();
  } catch (_) {}
}

function renderOverview() {
  const t = S.totals;
  const sessions = S.sessions;
  const activeCount = sessions.filter(s => s.is_active).length;

  set('ov-sessions', sessions.length);
  set('ov-active', activeCount);
  set('ov-messages', fmt(t.messages || 0));
  set('ov-compressions', fmt(t.compressions || 0));
  set('ov-freed', fmt(t.tokens_freed || 0));
  set('ov-cost-saved', `≈ ${fmtCost(t.tokens_freed || 0)} saved`);
  set('ov-dag-nodes', fmt(t.dag_nodes || 0));
  set('ov-facts', fmt(t.facts || 0));
  set('ov-session-count', `${sessions.length} session${sessions.length !== 1 ? 's' : ''}`);

  const tbody = document.getElementById('sessions-tbody');
  if (!tbody) return;

  if (!sessions.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="tbl-empty">No sessions yet — start an agent to see data here.</td></tr>';
    return;
  }

  tbody.innerHTML = sessions.map(s => {
    const sid = s.session_id;
    const isActive = s.is_active;
    const ratio = (s.context_length > 0 && s.last_prompt_tokens > 0)
      ? Math.min(s.last_prompt_tokens / s.context_length, 1) : 0;
    const barColor = ratio > 0.85 ? 'var(--red)' : ratio > 0.6 ? 'var(--amber)' : 'var(--green)';
    const lastSeen = s.last_at ? timeAgo(s.last_at) : '—';
    const statusText = isActive ? (s.last_compression_status || 'active') : 'idle';
    const statusCls = isActive ? 'active' : 'idle';
    const rowCls = isActive ? 'row-active' : '';
    const dotHtml = isActive
      ? '<span class="dot-active"></span>'
      : '<span class="dot-idle"></span>';

    return `<tr class="${rowCls}" onclick="navigate('session','${esc(sid)}')" title="Open session detail">
      <td><div class="sess-id-cell">${dotHtml}${esc(sid.length > 34 ? sid.slice(0,31) + '…' : sid)}</div></td>
      <td><span class="status-chip ${statusCls}">${esc(statusText)}</span></td>
      <td>
        <div class="ctx-bar-wrap">
          <div class="ctx-bar"><div class="ctx-bar-fill" style="width:${(ratio*100).toFixed(1)}%;background:${barColor}"></div></div>
        </div>
      </td>
      <td class="tbl-mono">${fmtN(s.message_count)}</td>
      <td class="tbl-mono">${s.dag_nodes}</td>
      <td class="tbl-mono">${s.compressions}</td>
      <td class="tbl-green">${fmt(s.tokens_freed)}</td>
      <td class="tbl-time">${lastSeen}</td>
      <td>
        <div class="tbl-actions">
          <button class="tbl-btn" onclick="event.stopPropagation();navigate('session','${esc(sid)}')" title="Open">View →</button>
          <button class="tbl-btn tbl-btn-danger" onclick="event.stopPropagation();ovDeleteSession('${esc(sid)}')" title="Delete session">Delete</button>
        </div>
      </td>
    </tr>`;
  }).join('');
}

function ovDeleteSession(sid) {
  showModal({
    icon: '⚠',
    title: 'Delete Session',
    body: `Permanently delete session <strong style="font-family:var(--mono)">${esc(sid)}</strong>?<br>All messages and DAG nodes will be removed.`,
    confirmText: 'Delete',
    onConfirm: async () => {
      await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}`);
      toast('success', `Session ${sid} deleted`);
      pollOverview();
    },
  });
}

// ── Session detail page ───────────────────────────────────────────────────────

function startSessionDetail(sid) {
  // Reset state
  S.engine = null;
  S.dag = null;
  S.messages = [];
  S.facts = [];
  S.selNode = null;
  S.selectedNodeData = null;
  S.pressure = { prompt: 0, threshold: 0, max: 0, ratio: 0 };

  document.getElementById('sess-id-label').textContent = sid;
  document.getElementById('elog').innerHTML = '<div class="elog-wait">Waiting…</div>';
  document.getElementById('nd-close').addEventListener('click', closeNodeDetail);

  // Wire close
  document.getElementById('nd-close').onclick = closeNodeDetail;

  Promise.all([fetchStatus(), fetchDAG(), fetchMessages(), fetchFacts(), loadEventHistory()]);
  _sessInterval = setInterval(pollSession, 8000);
}

function pollSession() {
  if (!S.sessionId) return;
  Promise.all([fetchStatus(), fetchDAG(), fetchMessages(), fetchFacts()]);
}

async function fetchStatus() {
  try {
    const r = await fetch(`/api/status?session_id=${encodeURIComponent(S.sessionId || '')}`);
    if (!r.ok) return;
    S.engine = await r.json();
    const e = S.engine;
    if (e.last_prompt_tokens && (e.threshold_tokens || e.context_length)) {
      S.pressure.prompt    = e.last_prompt_tokens;
      S.pressure.threshold = e.threshold_tokens || 0;
      S.pressure.max       = e.context_length || 0;
      S.pressure.ratio     = S.pressure.max > 0 ? S.pressure.prompt / S.pressure.max : 0;
    }
    renderPressure();
    renderSessionInfo();
  } catch (_) {}
}

async function fetchDAG() {
  try {
    const r = await fetch(`/api/dag?session_id=${encodeURIComponent(S.sessionId || '')}`);
    if (!r.ok) return;
    S.dag = await r.json();
    renderGraph();
    renderSessionInfo();
    if (S.selNode !== null) renderNodeDetail(S.selNode);
  } catch (_) {}
}

async function fetchMessages() {
  try {
    const r = await fetch(`/api/messages?limit=500&session_id=${encodeURIComponent(S.sessionId || '')}`);
    if (!r.ok) return;
    S.messages = (await r.json()).messages || [];
    renderStore();
  } catch (_) {}
}

// ── Pressure ──────────────────────────────────────────────────────────────────

function renderPressure() {
  const { prompt, threshold, max, ratio } = S.pressure;
  const pct = ratio * 100;
  const cls = ratio >= 0.9 ? 'crit' : ratio >= 0.65 ? 'warn' : '';

  const fill = $('pbar-fill');
  if (fill) { fill.style.width = `${Math.min(pct, 100)}%`; fill.className = `pbar-fill ${cls}`; }
  const pctEl = $('pbar-pct');
  if (pctEl) { pctEl.textContent = prompt > 0 ? `${pct.toFixed(1)}%` : '—'; pctEl.className = `pbar-pct ${cls}`; }
  const marker = $('pbar-marker');
  if (marker && max > 0 && threshold > 0) marker.style.left = `${Math.min((threshold / max) * 100, 100)}%`;
  set('ps-prompt', fmt(prompt));
  set('ps-threshold', fmt(threshold));
  set('ps-max', fmt(max));
}

// ── Session info ──────────────────────────────────────────────────────────────

function renderSessionInfo() {
  const e = S.engine;
  const nodes = (S.dag && S.dag.nodes) || [];
  const freed = nodes.reduce((s, n) => s + Math.max(0, (n.source_token_count || 0) - (n.token_count || 0)), 0);

  set('sc-freed', fmt(freed));
  set('sc-cost', `≈ ${fmtCost(freed)}`);

  if (!e) return;
  set('sc-compressions', e.compression_count ?? (e.compressions ?? 0));
  set('sc-messages', e.message_count ?? e.store_messages ?? 0);
  set('sc-nodes', e.dag_nodes ?? nodes.length);

  const badge = $('sess-status-badge');
  if (badge) {
    const st = e.last_compression_status || (e.is_active ? 'active' : 'idle');
    badge.textContent = st;
    badge.className = `sess-status-badge ${st}`;
  }

  const meta = $('sess-meta');
  if (meta) {
    const parts = [];
    if (e.db_path) parts.push(`db: <span>${esc(e.db_path.split('/').pop())}</span>`);
    if (e.threshold_percent) parts.push(`threshold: <span>${esc(String((e.threshold_percent * 100).toFixed(0)))}</span>%`);
    if (e.fresh_tail_count) parts.push(`tail: <span>${esc(String(e.fresh_tail_count))}</span> msgs`);
    if (e.summary_model) parts.push(`model: <span>${esc(e.summary_model || 'auto')}</span>`);
    meta.innerHTML = parts.join('<br>');
  }
}

// ── Event log ─────────────────────────────────────────────────────────────────

function appendLog(ev) {
  const el = $('elog'); if (!el) return;
  const w = el.querySelector('.elog-wait'); if (w) w.remove();
  while (el.children.length >= 120) el.removeChild(el.lastChild);
  const time = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
  const det = logDet(ev.type, ev.data);
  const d = document.createElement('div');
  d.className = 'log-row';
  d.innerHTML = `<span class="log-t">${esc(time)}</span><span><span class="log-type ${ev.type}">${esc(ev.type.replace(/_/g, ' '))}</span>${det ? `<span class="log-det"> ${det}</span>` : ''}</span>`;
  el.insertBefore(d, el.firstChild);
}

function logDet(type, d) {
  if (!d) return '';
  switch (type) {
    case 'compaction_end':   return `${d.messages_before}→${d.messages_after} msgs · ${fmtN(d.tokens_before)}→${fmtN(d.tokens_after)}t`;
    case 'node_added':       return `#${d.node_id} D${d.depth} · ${fmtN(d.token_count)}t`;
    case 'node_condensed':   return `${(d.input_node_ids || []).length}→1 D${d.depth}`;
    case 'token_pressure':   return `${fmtN(d.prompt_tokens)}/${fmtN(d.threshold_tokens)} (${(d.ratio * 100).toFixed(0)}%)`;
    case 'message_ingested': return `${d.role} #${d.store_id}`;
    case 'compaction_start': return `${fmtN(d.prompt_tokens)}t`;
    case 'fact_stored':  return `${d.key} [${d.category}] scope:${d.scope}`;
    case 'fact_deleted': return `${d.key} scope:${d.scope}`;
    default: return JSON.stringify(d).slice(0, 60);
  }
}

async function loadEventHistory() {
  const el = $('elog'); if (!el) return;
  try {
    const r = await fetch('/api/events/history');
    if (!r.ok) return;
    const data = await r.json();
    const events = data.events || [];
    if (!events.length) return;
    const sid = S.sessionId;
    const relevant = events.filter(ev =>
      !ev.data || !ev.data.session_id || ev.data.session_id === sid
    );
    if (!relevant.length) return;
    const w = el.querySelector('.elog-wait'); if (w) w.remove();
    // events are oldest-first; appendLog inserts at top → newest ends up at top
    for (const ev of relevant) appendLog(ev);
  } catch (_) {}
}

// ── Feed (overview) ───────────────────────────────────────────────────────────

function appendFeed(ev) {
  const el = $('feed-list'); if (!el) return;
  const w = el.querySelector('.feed-wait'); if (w) w.remove();
  while (el.children.length >= 200) el.removeChild(el.lastChild);

  S.feedCount++;
  const badge = $('feed-badge');
  if (badge) badge.textContent = S.feedCount > 99 ? '99+' : S.feedCount;

  const time = ev.ts ? new Date(ev.ts * 1000).toLocaleTimeString('en-US', { hour12: false }) : '';
  const sid = ev.data?.session_id || '';
  const det = logDet(ev.type, ev.data);
  const d = document.createElement('div');
  d.className = 'feed-item';
  d.innerHTML = `<div class="feed-time">${esc(time)}</div><div><span class="feed-type ${ev.type}">${esc(ev.type.replace(/_/g, ' '))}</span>${det ? `<span class="feed-det"> — ${esc(det)}</span>` : ''}</div>${sid ? `<div class="feed-sid">${esc(sid)}</div>` : ''}`;
  el.insertBefore(d, el.firstChild);
}

// ── DAG Graph SVG ─────────────────────────────────────────────────────────────

function renderGraph() {
  const wrap = $('dag-svg-wrap');
  const empty = $('graph-empty');
  if (!wrap) return;
  const nodes = (S.dag && S.dag.nodes) || [];
  if (!nodes.length) {
    wrap.innerHTML = '';
    if (empty) empty.style.display = 'block';
    return;
  }
  if (empty) empty.style.display = 'none';
  wrap.innerHTML = buildSVG(nodes);
}

function buildSVG(nodes) {
  const byD = {};
  for (const n of nodes) (byD[n.depth] = byD[n.depth] || []).push(n);
  const depths = Object.keys(byD).map(Number).sort((a, b) => b - a);
  const maxD = depths[0] || 0;

  const NW = 196, NH = 86, STEP_X = 272, STEP_Y = NH + 22, PAD_X = 28, PAD_Y = 44;
  const pos = {};

  const d0s = (byD[0] || []).slice().sort((a, b) => a.node_id - b.node_id);
  d0s.forEach((n, i) => { pos[n.node_id] = { x: PAD_X + maxD * STEP_X, y: PAD_Y + i * STEP_Y }; });

  for (const d of [...depths].filter(d => d > 0).sort((a, b) => a - b)) {
    (byD[d] || []).slice().sort((a, b) => a.node_id - b.node_id).forEach((n, i) => {
      const childIds = (n.source_ids || []).filter(id => pos[id]);
      let y;
      if (childIds.length) {
        const ys = childIds.map(id => pos[id].y + NH / 2);
        y = (Math.min(...ys) + Math.max(...ys)) / 2 - NH / 2;
      } else { y = PAD_Y + i * STEP_Y; }
      pos[n.node_id] = { x: PAD_X + (maxD - d) * STEP_X, y };
    });
  }

  const dbX = PAD_X + (maxD + 1) * STEP_X + 28;
  const allP = Object.values(pos);
  const svgW = dbX + 160;
  const svgH = allP.reduce((m, p) => Math.max(m, p.y + NH), 0) + PAD_Y;

  const DM = {
    0: { label: 'D0  LEAF',    col: '#0c9e72', bg: '#f0fdf8', tc: '#065f45', bar: 'rgba(12,158,114,0.25)' },
    1: { label: 'D1  ARC',     col: '#c47c08', bg: '#fffbeb', tc: '#7a4e05', bar: 'rgba(196,124,8,0.25)' },
    2: { label: 'D2  DURABLE', col: '#2574e8', bg: '#eff6ff', tc: '#1448a8', bar: 'rgba(37,116,232,0.25)' },
  };

  const P = [];
  P.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${svgW}" height="${svgH}">`);
  P.push(`<defs>
    <filter id="nshadow"><feDropShadow dx="0" dy="1" stdDeviation="3" flood-color="#000" flood-opacity="0.08"/></filter>
    <filter id="nshadow-sel"><feDropShadow dx="0" dy="3" stdDeviation="8" flood-color="#000" flood-opacity="0.14"/></filter>
    <marker id="am-dag" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1.5 L0,6.5 L7,4z" fill="#94adc8"/></marker>
    <marker id="am-db"  markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1.5 L0,6.5 L7,4z" fill="#6a3ed4" opacity="0.45"/></marker>
  </defs>`);

  // Guide lines + column labels
  for (const d of depths) {
    const dm = DM[d] || { label: `D${d}`, col: '#94adc8' };
    const cx = PAD_X + (maxD - d) * STEP_X + NW / 2;
    P.push(`<line x1="${cx}" y1="${PAD_Y - 10}" x2="${cx}" y2="${svgH - 12}" stroke="${dm.col}" stroke-width="1" stroke-dasharray="2,20" opacity="0.2"/>`);
    P.push(`<text x="${cx}" y="${PAD_Y - 14}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="${dm.col}" letter-spacing="2" opacity="0.6" font-weight="700">${dm.label}</text>`);
  }
  const dbCx = dbX + 56;
  P.push(`<line x1="${dbCx}" y1="${PAD_Y - 10}" x2="${dbCx}" y2="${svgH - 12}" stroke="#6a3ed4" stroke-width="1" stroke-dasharray="2,20" opacity="0.15"/>`);
  P.push(`<text x="${dbCx}" y="${PAD_Y - 14}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#6a3ed4" letter-spacing="2" opacity="0.55" font-weight="700">SQLITE</text>`);

  // Parent → child edges
  for (const d of depths.filter(d => d > 0)) {
    for (const n of byD[d]) {
      const p = pos[n.node_id]; if (!p) continue;
      for (const cid of (n.source_ids || []).filter(id => pos[id])) {
        const cp = pos[cid];
        const x1 = p.x + NW, y1 = p.y + NH / 2, x2 = cp.x, y2 = cp.y + NH / 2, mx = (x1 + x2) / 2;
        P.push(`<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="#c8d6ea" stroke-width="1.5" marker-end="url(#am-dag)"/>`);
      }
    }
  }

  // D0 → DB dashed lines
  for (const n of d0s) {
    const p = pos[n.node_id]; if (!p) continue;
    const ids = n.source_ids || [];
    const y1 = p.y + NH / 2;
    const label = ids.length === 0 ? '' : ids.length <= 4 ? ids.map(i => `#${i}`).join(' ') : `#${ids[0]}–#${ids[ids.length - 1]} (${ids.length})`;
    P.push(`<line x1="${p.x + NW}" y1="${y1}" x2="${dbX - 4}" y2="${y1}" stroke="#6a3ed4" stroke-width="1" stroke-dasharray="5,4" marker-end="url(#am-db)" opacity="0.4"/>`);
    if (label) P.push(`<text x="${dbX + 5}" y="${y1 + 4}" font-family="JetBrains Mono,monospace" font-size="9.5" fill="#6a3ed4" opacity="0.6">${esc(label)}</text>`);
  }

  // Nodes
  for (const d of depths) {
    const dm = DM[d] || { label: `D${d}`, col: '#4d6380', bg: 'rgba(77,99,128,0.06)', tc: '#4d6380', bar: 'rgba(77,99,128,0.2)' };
    for (const n of byD[d]) {
      const p = pos[n.node_id]; if (!p) continue;
      const isSel = S.selNode === n.node_id;
      const sw = isSel ? '2' : '1', so = isSel ? '0.9' : '0.3';
      const filter = isSel ? 'url(#nshadow-sel)' : 'url(#nshadow)';
      const ratio = n.source_token_count > 0 ? Math.min((n.token_count || 0) / n.source_token_count, 1) : null;
      const pctFree = ratio !== null ? Math.round((1 - ratio) * 100) : null;
      const barW = NW - 24, barFill = ratio !== null ? barW * ratio : 0;
      const hint = (n.summary_preview || '').slice(0, 38);
      const tokLine = n.source_token_count > 0 ? `${fmtN(n.token_count)}t ← ${fmtN(n.source_token_count)}t` : `${fmtN(n.token_count)}t`;
      const srcCount = (n.source_ids || []).length;
      const srcLabel = n.source_type === 'nodes' ? `↑ ${srcCount} nodes condensed` : `⊞ ${srcCount} store msgs`;

      P.push(`<g class="dag-node-g" data-nid="${n.node_id}" onclick="selectNode(${n.node_id})" onmouseenter="showTip(event,${n.node_id})" onmouseleave="hideTip()">`);
      P.push(`<rect x="${p.x}" y="${p.y}" width="${NW}" height="${NH}" rx="7" fill="${dm.bg}" stroke="${dm.col}" stroke-width="${sw}" stroke-opacity="${so}" filter="${filter}" class="node-box"/>`);
      if (isSel) P.push(`<rect x="${p.x - 3}" y="${p.y - 3}" width="${NW + 6}" height="${NH + 6}" rx="10" fill="none" stroke="${dm.col}" stroke-width="1.5" stroke-opacity="0.2"/>`);

      P.push(`<text x="${p.x + 10}" y="${p.y + 17}" font-family="JetBrains Mono,monospace" font-size="11" fill="${dm.col}" font-weight="700">#${n.node_id}</text>`);
      P.push(`<rect x="${p.x + NW - 32}" y="${p.y + 5}" width="26" height="15" rx="3" fill="${dm.col}" fill-opacity="0.08" stroke="${dm.col}" stroke-width="0.5" stroke-opacity="0.3"/>`);
      P.push(`<text x="${p.x + NW - 19}" y="${p.y + 15}" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="${dm.col}" font-weight="600" opacity="0.7">D${d}</text>`);
      P.push(`<text x="${p.x + 10}" y="${p.y + 34}" font-family="JetBrains Mono,monospace" font-size="11" fill="${dm.tc}" font-weight="600">${esc(tokLine)}</text>`);
      if (pctFree !== null && pctFree > 0) P.push(`<text x="${p.x + NW - 8}" y="${p.y + 34}" text-anchor="end" font-family="JetBrains Mono,monospace" font-size="10" fill="${dm.col}" opacity="0.45">−${pctFree}%</text>`);
      P.push(`<text x="${p.x + 10}" y="${p.y + 50}" font-family="JetBrains Mono,monospace" font-size="9" fill="#7a9bbf">${esc(hint)}${hint.length >= 38 ? '…' : ''}</text>`);
      P.push(`<rect x="${p.x + 10}" y="${p.y + NH - 20}" width="${barW}" height="3" rx="1.5" fill="${dm.bar}" opacity="0.5"/>`);
      if (barFill > 0) P.push(`<rect x="${p.x + 10}" y="${p.y + NH - 20}" width="${barFill.toFixed(1)}" height="3" rx="1.5" fill="${dm.col}" opacity="0.6"/>`);
      P.push(`<text x="${p.x + 10}" y="${p.y + NH - 6}" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#94adc8">${esc(srcLabel)}</text>`);
      P.push(`</g>`);
    }
  }

  P.push('</svg>');
  return P.join('');
}

// ── Node detail panel ─────────────────────────────────────────────────────────

function selectNode(nodeId) {
  S.selNode = S.selNode === nodeId ? null : nodeId;
  renderGraph();
  renderStore();
  renderNodeDetail(S.selNode);
}

function closeNodeDetail() {
  S.selNode = null;
  S.selectedNodeData = null;
  renderGraph();
  renderStore();
  const nd = $('node-detail');
  if (nd) nd.classList.remove('open');
}

function renderNodeDetail(nodeId) {
  const nd = $('node-detail');
  if (!nodeId || !S.dag) { if (nd) nd.classList.remove('open'); return; }
  const n = S.dag.nodes.find(x => x.node_id === nodeId);
  if (!n) { if (nd) nd.classList.remove('open'); return; }
  S.selectedNodeData = n;

  const freed = Math.max(0, (n.source_token_count || 0) - (n.token_count || 0));
  const pctF = n.source_token_count > 0 ? Math.round((freed / n.source_token_count) * 100) : 0;
  const dLabel = ['D0 Leaf', 'D1 Arc', 'D2 Durable'][n.depth] || `D${n.depth}`;
  const bClass = ['nd-badge-d0', 'nd-badge-d1', 'nd-badge-d2'][n.depth] || '';

  set('nd-title', `#${n.node_id} · ${fmtN(n.token_count)}t summary ← ${fmtN(n.source_token_count)}t source  (−${pctF}% freed)`);
  const badge = $('nd-badge');
  if (badge) { badge.textContent = dLabel; badge.className = `nd-badge ${bClass}`; }

  const sumEl = $('nd-summary');
  if (sumEl) sumEl.textContent = n.summary || n.summary_preview || '(no summary)';

  const srcIds = n.source_ids || [];
  const isMsgs = n.source_type === 'messages';
  const explainEl = $('nd-ref-explain');
  if (explainEl) {
    explainEl.innerHTML = isMsgs
      ? `<strong>${srcIds.length} SQLite store messages</strong> were compressed into this node. Click a tag to highlight that row.`
      : `<strong>${srcIds.length} lower-level DAG nodes</strong> were condensed. These nodes still exist in the graph.`;
  }

  const tagsEl = $('nd-ref-tags');
  if (tagsEl) {
    const cls = isMsgs ? 'store-ref' : 'node-ref';
    tagsEl.innerHTML = srcIds.map(id => {
      const cb = isMsgs ? `onclick="highlightStoreRef(${id})"` : '';
      return `<span class="nd-ref-tag ${cls}" ${cb}>${isMsgs ? '#' : 'node#'}${id}</span>`;
    }).join('');
  }

  const jsonEl = $('nd-json');
  if (jsonEl) jsonEl.innerHTML = colorizeJSON(n);

  if (nd) nd.classList.add('open');
}

function highlightStoreRef(storeId) {
  const tr = document.querySelector(`#store-tbody tr[data-sid="${storeId}"]`);
  if (!tr) return;
  tr.scrollIntoView({ behavior: 'smooth', block: 'center' });
  tr.style.outline = '2px solid var(--purple)';
  setTimeout(() => { tr.style.outline = ''; }, 1600);
}

function deleteCurrentNode() {
  if (!S.selectedNodeData) return;
  const n = S.selectedNodeData;
  showModal({
    icon: '⬡',
    title: 'Delete DAG Node',
    body: `Delete node <strong style="font-family:var(--mono)">#${n.node_id} (D${n.depth})</strong>?<br>The compressed summary will be removed. Source messages remain in the SQLite store.`,
    confirmText: 'Delete Node',
    onConfirm: async () => {
      const sid = S.sessionId;
      await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}/dag/${n.node_id}`);
      toast('success', `Node #${n.node_id} deleted`);
      closeNodeDetail();
      fetchDAG();
    },
  });
}

// ── Store table ───────────────────────────────────────────────────────────────

function renderStore() {
  const tbody = $('store-tbody');
  const cnt = $('store-count');
  if (!tbody) return;
  const msgs = S.messages;
  if (cnt) cnt.textContent = `${msgs.length} rows`;
  if (!msgs.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="store-empty">No messages yet</td></tr>';
    return;
  }
  const hlIds = getHighlightIds(S.selNode);
  tbody.innerHTML = msgs.map(m => {
    const role = m.role || 'unknown';
    const id   = m.store_id || '?';
    const tok  = m.token_estimate || 0;
    const txt  = (m.content_preview || '').trim().slice(0, 80).replace(/\n/g, ' ');
    const hl   = hlIds.has(String(id)) ? 'hl' : '';
    return `<tr class="${hl}" data-sid="${id}" onclick="openMsgViewer(${id})" title="Click to read full message" style="cursor:pointer">
      <td class="col-id">${id}</td>
      <td><span class="rbadge r-${role}">${esc(role.slice(0, 4))}</span></td>
      <td class="col-tok">${fmtN(tok)}t</td>
      <td class="col-cnt">${esc(txt)}</td>
      <td class="col-del"><button class="del-msg-btn" onclick="deleteMessage(event,${id})" title="Delete message">✕</button></td>
    </tr>`;
  }).join('');
}

function getHighlightIds(nodeId) {
  if (!nodeId || !S.dag) return new Set();
  const n = S.dag.nodes.find(x => x.node_id === nodeId);
  if (!n || n.source_type !== 'messages') return new Set();
  return new Set((n.source_ids || []).map(String));
}

function deleteMessage(event, storeId) {
  event.stopPropagation();
  showModal({
    icon: '✕',
    title: 'Delete Message',
    body: `Remove message <strong style="font-family:var(--mono)">#${storeId}</strong> from the store?<br>DAG summaries that reference it are not affected.`,
    confirmText: 'Delete',
    onConfirm: async () => {
      const sid = S.sessionId;
      await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}/messages/${storeId}`);
      toast('success', `Message #${storeId} deleted`);
      fetchMessages();
    },
  });
}

// ── Session-level CRUD ────────────────────────────────────────────────────────

function confirmAction(action) {
  const sid = S.sessionId;
  if (!sid) return;
  const configs = {
    'clear-messages': {
      icon: '⊡', title: 'Clear All Messages',
      body: `Delete all stored messages for <strong style="font-family:var(--mono)">${esc(sid)}</strong>?<br>DAG summary nodes will remain intact.`,
      confirmText: 'Clear Messages',
      onConfirm: async () => {
        await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}/messages`);
        toast('success', 'Messages cleared');
        fetchMessages(); fetchStatus();
      },
    },
    'clear-dag': {
      icon: '◈', title: 'Clear DAG',
      body: `Delete all summary nodes for <strong style="font-family:var(--mono)">${esc(sid)}</strong>?<br>Raw messages in the store will remain.`,
      confirmText: 'Clear DAG',
      onConfirm: async () => {
        await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}/dag`);
        toast('success', 'DAG cleared');
        fetchDAG(); fetchStatus();
      },
    },
    'delete-session': {
      icon: '⚠', title: 'Delete Session',
      body: `Permanently delete session <strong style="font-family:var(--mono)">${esc(sid)}</strong>?<br>All messages and DAG nodes will be removed. This cannot be undone.`,
      confirmText: 'Delete Session',
      onConfirm: async () => {
        await apiFetch(`DELETE /api/sessions/${encodeURIComponent(sid)}`);
        toast('success', `Session ${sid} deleted`);
        navigate('overview');
      },
    },
  };
  const cfg = configs[action];
  if (cfg) showModal(cfg);
}

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTip(event, nodeId) {
  const n = (S.dag && S.dag.nodes || []).find(x => x.node_id === nodeId);
  if (!n) return;
  const el = $('node-tooltip'); if (!el) return;
  const dLabel = ['Leaf (D0)', 'Arc (D1)', 'Durable (D2)'][n.depth] || `D${n.depth}`;
  const freed = Math.max(0, (n.source_token_count || 0) - (n.token_count || 0));
  const pctF = n.source_token_count > 0 ? Math.round((freed / n.source_token_count) * 100) : 0;
  const srcIds = n.source_ids || [];
  const refDesc = n.source_type === 'nodes'
    ? `${srcIds.length} child nodes → [${srcIds.slice(0, 4).map(i => '#' + i).join(', ')}${srcIds.length > 4 ? ', …' : ''}]`
    : `${srcIds.length} store msgs → [${srcIds.slice(0, 4).map(i => '#' + i).join(', ')}${srcIds.length > 4 ? ', …' : ''}]`;

  el.innerHTML = `<div class="tt-hd">#${n.node_id} · ${dLabel}</div>`
    + `<div class="tt-sum">${esc((n.summary_preview || '').slice(0, 130))}${(n.summary_preview || '').length > 130 ? '…' : ''}</div>`
    + `<div class="tt-meta">${fmtN(n.token_count)}t summary ← ${fmtN(n.source_token_count)}t source<br>`
    + `freed: ${fmtN(freed)}t (${pctF}%)<br>${esc(refDesc)}<br>`
    + `<span style="opacity:0.5">click to expand</span></div>`;

  el.style.left = `${Math.min(event.clientX + 18, window.innerWidth - 320)}px`;
  el.style.top = `${Math.max(event.clientY - 16, 8)}px`;
  el.classList.add('show');

  if (n.source_type === 'messages') {
    const ids = new Set((n.source_ids || []).map(String));
    document.querySelectorAll('#store-tbody tr[data-sid]').forEach(tr => tr.classList.toggle('hl', ids.has(tr.dataset.sid)));
  }
}

function hideTip() {
  const el = $('node-tooltip'); if (el) el.classList.remove('show');
  if (!S.selNode) document.querySelectorAll('#store-tbody tr.hl').forEach(tr => tr.classList.remove('hl'));
}

// ── Message viewer ────────────────────────────────────────────────────────────

function openMsgViewer(storeId) {
  const m = S.messages.find(x => x.store_id == storeId);
  if (!m) return;

  const role   = m.role || 'unknown';
  const tok    = m.token_estimate || 0;
  const ts     = m.created_at ? new Date(m.created_at * 1000).toLocaleString() : '—';
  const content = m.content_full || m.content_preview || '(empty)';

  // Badge
  const badge = $('mv-badge');
  if (badge) {
    badge.textContent = role.slice(0, 9).toUpperCase();
    badge.className = `mv-badge rbadge r-${role}`;
  }
  set('mv-id',     `#${storeId}`);
  set('mv-tokens', `${fmtN(tok)} tokens`);
  set('mv-time',   ts);

  // Body — detect JSON / tool calls
  const body = $('mv-body');
  if (body) {
    const looksJson = content.trimStart().startsWith('{') || content.trimStart().startsWith('[');
    body.className = `mv-body${looksJson ? ' is-json' : ''}`;
    if (looksJson) {
      try {
        body.innerHTML = colorizeJSON(JSON.parse(content));
      } catch (_) {
        body.textContent = content;
      }
    } else {
      body.textContent = content;
    }
  }

  $('msg-viewer-overlay').classList.add('open');
}

function closeMsgViewer(event) {
  if (event && event.target !== $('msg-viewer-overlay')) return;
  $('msg-viewer-overlay').classList.remove('open');
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    $('msg-viewer-overlay')?.classList.remove('open');
    closeModal();
  }
});

// ── Search ────────────────────────────────────────────────────────────────────

async function runSearch(query) {
  const el = $('search-results'); if (!el) return;
  if (!query) { el.innerHTML = ''; return; }
  try {
    const r = await fetch('/api/grep', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 8, session_id: S.sessionId || '' }),
    });
    const data = await r.json();
    const hits = data.results || [];
    if (!hits.length) { el.innerHTML = '<div style="color:var(--text-dim);font-size:10px;padding:4px">No results</div>'; return; }
    el.innerHTML = hits.map(h => `<div class="sh-hit"><div class="sh-role">${esc(h.role || '?')} #${h.store_id || ''}</div><div class="sh-snip">${hlSearch(h.snippet || '', query)}</div></div>`).join('');
  } catch (_) { el.innerHTML = ''; }
}

// ── Modal system ──────────────────────────────────────────────────────────────

let _pendingAction = null;

function showModal({ icon, title, body, confirmText, confirmClass, onConfirm }) {
  _pendingAction = onConfirm;
  const overlay = $('modal-overlay');
  $('modal-icon').textContent = icon || '⚠';
  $('modal-title').textContent = title || 'Confirm';
  $('modal-body').innerHTML = body || '';
  const btn = $('modal-confirm-btn');
  btn.textContent = confirmText || 'Confirm';
  btn.className = `modal-confirm ${confirmClass || ''}`;
  overlay.classList.add('open');
}

function closeModal() {
  _pendingAction = null;
  $('modal-overlay').classList.remove('open');
}

async function executeConfirmedAction() {
  closeModal();
  if (_pendingAction) {
    try { await _pendingAction(); } catch (e) { toast('error', 'Action failed: ' + e.message); }
  }
}

// ── Toast system ──────────────────────────────────────────────────────────────

function toast(type, message) {
  const container = $('toasts'); if (!container) return;
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span><span>${esc(message)}</span>`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ── JSON colorizer ────────────────────────────────────────────────────────────

function colorizeJSON(obj) {
  const raw = JSON.stringify(obj, null, 2);
  return raw.split('\n').map(line => {
    const e = esc(line);
    const m = e.match(/^(\s*)(&quot;[\w_]+&quot;)(\s*:\s*)(.*)$/);
    if (!m) return e;
    const [, ws, key, sep, rest] = m;
    let val = rest;
    if (rest.startsWith('&quot;')) val = `<span class="jvs">${rest}</span>`;
    else if (/^-?\d/.test(rest)) val = `<span class="jvn">${rest}</span>`;
    else if (/^(true|false|null)/.test(rest)) val = `<span class="jvb">${rest}</span>`;
    return `${ws}<span class="jk">${key}</span>${sep}${val}`;
  }).join('\n');
}

// ── API helper ────────────────────────────────────────────────────────────────

async function apiFetch(descriptor, body) {
  const [method, url] = descriptor.split(' ');
  const opts = { method };
  if (body !== undefined) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }
function set(id, v) { const e = $(id); if (e) e.textContent = v; }

function fmt(n) {
  if (n == null || n === '') return '—';
  const x = Number(n); if (isNaN(x)) return '—';
  if (x >= 1_000_000) return (x / 1_000_000).toFixed(1) + 'M';
  if (x >= 10_000) return (x / 1_000).toFixed(1) + 'K';
  if (x >= 1_000) return (x / 1_000).toFixed(2) + 'K';
  return String(x);
}

function fmtN(n) {
  if (n == null) return '0';
  const x = Number(n);
  if (x >= 1_000) return (x / 1_000).toFixed(1) + 'K';
  return String(x);
}

function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function hlSearch(text, query) {
  let r = esc(text);
  for (const t of query.split(/\s+/).filter(Boolean))
    r = r.replace(new RegExp(esc(t), 'gi'), m => `<mark>${m}</mark>`);
  return r;
}

function fmtCost(tokens) {
  const d = (tokens / 1_000_000) * 3;
  if (d < 0.001) return '< $0.001';
  if (d < 1) return '$' + d.toFixed(3);
  return '$' + d.toFixed(2);
}

function timeAgo(ts) {
  if (!ts) return '—';
  const diff = (Date.now() / 1000) - ts;
  if (diff < 5) return 'just now';
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

// ── Persistent Memory (Fact Store) ────────────────────────────────────────────

async function fetchFacts() {
  try {
    const r = await fetch('/api/facts');
    if (!r.ok) return;
    S.facts = (await r.json()).facts || [];
    renderFacts();
  } catch (_) {}
}

function renderFacts() {
  const el = $('fact-list');
  const countEl = $('fact-count');
  if (!el) return;

  const query = (($('fact-search') || {}).value || '').toLowerCase();
  let facts = S.facts;
  if (query) facts = facts.filter(f =>
    (f.key || '').toLowerCase().includes(query) ||
    (f.value || '').toLowerCase().includes(query) ||
    (f.category || '').toLowerCase().includes(query)
  );

  if (countEl) countEl.textContent = S.facts.length;

  if (!facts.length) {
    el.innerHTML = query
      ? `<div class="fact-empty">No facts match "${esc(query)}"</div>`
      : '<div class="fact-empty">No facts yet — agents use <code>lcm_remember</code> or click ＋</div>';
    return;
  }

  el.innerHTML = facts.map(f => {
    const scopeLabel = f.scope === 'global' ? 'global' : 'session';
    const scopeCls   = f.scope === 'global'  ? 'fact-scope-global' : 'fact-scope-session';
    const catCls     = `fact-cat-${f.category || 'fact'}`;
    const safeScope  = esc(f.scope || 'global');
    const safeKey    = esc(f.key || '');
    const safeVal    = esc((f.value || '').slice(0, 180));
    const updated    = f.updated_at ? timeAgo(f.updated_at) : '';
    const tags       = Array.isArray(f.tags) ? f.tags : [];
    const related    = Array.isArray(f.related_keys) ? f.related_keys : [];
    const tagChips   = tags.map(t => `<span class="fact-tag-chip">${esc(t)}</span>`).join('');
    const relHint    = related.length ? `<span class="fact-rel-hint" title="${esc(related.join(', '))}">⟳ ${related.length} link${related.length > 1 ? 's' : ''}</span>` : '';
    return `<div class="fact-row" title="${safeKey}">
      <div class="fact-header">
        <span class="fact-key">${safeKey}</span>
        <div class="fact-badges">
          <span class="fact-badge ${scopeCls}">${esc(scopeLabel)}</span>
          <span class="fact-badge ${catCls}">${esc(f.category || 'fact')}</span>
        </div>
        <button class="fact-del-btn" onclick="deleteFact('${safeScope}','${safeKey.replace(/'/g, "\\'")}')" title="Delete fact">✕</button>
      </div>
      <div class="fact-value">${safeVal}${(f.value || '').length > 180 ? '…' : ''}</div>
      ${(tagChips || relHint) ? `<div class="fact-meta-row">${tagChips}${relHint}</div>` : ''}
      ${updated ? `<div style="font-size:9px;color:var(--text-dim);margin-top:2px">${esc(updated)}</div>` : ''}
    </div>`;
  }).join('');
}

function showAddFactModal() {
  const overlay = $('add-fact-overlay');
  if (!overlay) return;
  const keyEl = $('af-key'); if (keyEl) keyEl.value = '';
  const valEl = $('af-value'); if (valEl) valEl.value = '';
  const catEl = $('af-category'); if (catEl) catEl.value = 'fact';
  const scopeEl = $('af-scope'); if (scopeEl) scopeEl.value = 'global';
  const tagsEl = $('af-tags'); if (tagsEl) tagsEl.value = '';
  overlay.style.display = 'flex';
  setTimeout(() => { if (keyEl) keyEl.focus(); }, 50);
}

function closeAddFactModal(event) {
  if (event && event.target !== $('add-fact-overlay')) return;
  const overlay = $('add-fact-overlay');
  if (overlay) overlay.style.display = 'none';
}

async function submitAddFact() {
  const key      = (($('af-key') || {}).value || '').trim();
  const value    = (($('af-value') || {}).value || '').trim();
  const category = ($('af-category') || {}).value || 'fact';
  const scope    = ($('af-scope') || {}).value || 'global';
  const rawTags  = (($('af-tags') || {}).value || '').trim();
  const tags     = rawTags ? rawTags.split(',').map(t => t.trim()).filter(Boolean) : [];

  if (!key || !value) { toast('error', 'Key and value are required'); return; }

  const resolvedScope = scope === 'current'
    ? (S.sessionId || 'global')
    : scope;

  const body = { key, value, category, scope: resolvedScope };
  if (tags.length) body.tags = tags;

  try {
    const r = await apiFetch('POST /api/facts', body);
    if (r && r.fact_id) {
      toast('success', `Stored: ${key}`);
      closeAddFactModal();
      fetchFacts();
    }
  } catch (e) {
    toast('error', 'Failed to store fact');
  }
}

function deleteFact(scope, key) {
  showModal({
    icon: '◆',
    title: 'Delete Fact',
    body: `Delete fact <strong style="font-family:var(--mono)">${esc(key)}</strong> (scope: ${esc(scope)})?`,
    confirmText: 'Delete',
    onConfirm: async () => {
      await apiFetch('DELETE /api/facts', { key, scope });
      toast('success', `Fact deleted: ${key}`);
      fetchFacts();
    },
  });
}
