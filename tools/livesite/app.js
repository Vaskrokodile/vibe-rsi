import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------- api ----------
const api = {
  sessions: () => fetch('/api/sessions').then(r => r.json()),
  events: (id, off) => fetch(`/api/events?id=${encodeURIComponent(id)}&off=${off}`).then(r => r.json()),
  stats: () => fetch('/api/stats').then(r => r.json()),
};

const COLORS = {
  user: 0x4ade80, assistant: 0x60a5fa, reasoning: 0xa78bfa,
  tool_call: 0xfb923c, tool_result: 0xfacc15, skill: 0xf472b6,
  subagent: 0x22d3ee, usage: 0x64748b, turn_end: 0x334155,
  session_start: 0xe2e8f0, improve: 0xfbbf24,
};
const LABELS = {
  user: 'prompt', assistant: 'assistant', reasoning: 'reasoning',
  tool_call: 'tool call', tool_result: 'tool result', skill: 'skill',
  subagent: 'subagent', usage: 'usage', turn_end: 'turn end',
  session_start: 'session start', improve: 'harness² improve',
};
const NODE_TYPES = ['session_start','user','reasoning','assistant','tool_call','tool_result','skill','subagent','turn_end'];

// ---------- dom ----------
const $ = s => document.querySelector(s);
const sesslistEl = $('#sesslist'), detailEl = $('#detail'), statsEl = $('#stats');
const toastEl = $('#toast');
let toastTimer = 0;
function toast(msg) {
  toastEl.textContent = msg; toastEl.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.style.display = 'none', 3200);
}

// legend
$('#legend').innerHTML = NODE_TYPES.map(t =>
  `<div class="li"><div class="dot" style="background:#${COLORS[t].toString(16).padStart(6,'0')}"></div>${LABELS[t]}</div>`
).join('') + `<div class="li"><div class="dot" style="background:#fbbf24"></div>harness² improve (center)</div>`;

// ---------- three scene ----------
const canvas = $('#c3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07090f, 0.016);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 2000);
camera.position.set(14, 10, 20);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.autoRotate = true; controls.autoRotateSpeed = 0.35;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.PointLight(0x88ccff, 60); key.position.set(30, 40, 20); scene.add(key);
const fill = new THREE.PointLight(0xff88cc, 25); fill.position.set(-25, -10, -20); scene.add(fill);

// subtle starfield
{
  const g = new THREE.BufferGeometry();
  const n = 900, pos = new Float32Array(n * 3);
  for (let i = 0; i < n * 3; i++) pos[i] = (Math.random() - 0.5) * 240;
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x334155, size: 0.35, transparent: true, opacity: 0.7 })));
}

const sphereGeo = new THREE.SphereGeometry(1, 18, 14);
const haloGeo = new THREE.SphereGeometry(1, 14, 10);
const materials = {};
function mat(color) {
  if (!materials[color]) materials[color] = new THREE.MeshStandardMaterial({
    color, emissive: color, emissiveIntensity: 0.35, roughness: 0.35, metalness: 0.15 });
  return materials[color];
}
function haloMat(color) {
  const k = 'h' + color;
  if (!materials[k]) materials[k] = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.10, blending: THREE.AdditiveBlending, depthWrite: false });
  return materials[k];
}

// ---------- session graph state ----------
const sessions = new Map();   // id -> {meta, off, nodes:[], edges, center:Vector3, group, lastTs, line}
const pickables = [];
let ringRadius = 14;

function sessionCenter(idx, total) {
  ringRadius = Math.max(13, total * 4.5);
  const a = idx * Math.PI * 2 / Math.max(1, total);
  return new THREE.Vector3(Math.cos(a) * ringRadius, 0, Math.sin(a) * ringRadius);
}
function relayoutCenters() {
  const ids = [...sessions.keys()];
  ids.forEach((id, i) => {
    const s = sessions.get(id);
    s.center = sessionCenter(i, ids.length);
    s.group.position.copy(s.center);
  });
}

function nodePos(s, ev, i) {
  // helix along local Y; reasoning orbits wider, tools mid, subagents flare out
  const a = i * 0.62;
  let r = 2.0, y = -i * 0.42;
  const t = ev.type;
  if (t === 'reasoning') r = 3.1;
  else if (t === 'tool_call' || t === 'tool_result') r = 2.6;
  else if (t === 'skill') r = 3.5;
  else if (t === 'subagent') r = 3.2;
  else if (t === 'turn_end' || t === 'session_start') r = 2.3;
  const d = (ev.depth || 0) * 1.1;
  r += d;
  // slight per-type angular offset keeps overlapping indices readable
  return new THREE.Vector3(Math.cos(a) * r, y - d * 0.6, Math.sin(a) * r);
}
const SIZES = { session_start: .30, user: .24, assistant: .28, reasoning: .15,
  tool_call: .19, tool_result: .14, skill: .26, subagent: .32, usage: .1, turn_end: .11 };

function addEvent(sid, ev) {
  let s = sessions.get(sid);
  if (!s) {
    s = {
      meta: {}, off: 0, nodes: [], lastTs: 0,
      group: new THREE.Group(),
      lineGeo: new THREE.BufferGeometry(),
      line: null, n: 0,
    };
    s.line = new THREE.Line(s.lineGeo, new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.35 }));
    s.group.add(s.line);
    scene.add(s.group);
    sessions.set(sid, s);
    relayoutCenters();
    toast(`◈ new session ${sid}`);
  }
  if (ev.type === 'usage') { s.lastTs = ev.ts || s.lastTs; return; } // tracked via stats, not a node
  if (ev.type === 'session_start') s.meta = { model: ev.model, cwd: ev.cwd, name: ev.name };

  const i = s.n++;
  const pos = nodePos(s, ev, i);
  const size = SIZES[ev.type] ?? 0.14;
  const color = COLORS[ev.type] ?? 0x94a3b8;
  const mesh = new THREE.Mesh(sphereGeo, mat(color));
  mesh.position.copy(pos);
  mesh.scale.setScalar(0.001);
  mesh.userData = { sid, ev, target: size };
  const halo = new THREE.Mesh(haloGeo, haloMat(color));
  halo.scale.setScalar(0.001);
  halo.userData = { target: size * 2.6 };
  mesh.add(halo);
  s.group.add(mesh);
  s.nodes.push(mesh);
  pickables.push(mesh);
  s.lastTs = ev.ts || s.lastTs;

  // rebuild edges (consecutive chain)
  const n = s.nodes.length;
  const arr = new Float32Array(n * 3);
  s.nodes.forEach((m, j) => { arr[j*3] = m.position.x; arr[j*3+1] = m.position.y; arr[j*3+2] = m.position.z; });
  s.lineGeo.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  s.lineGeo.computeBoundingSphere();
}

// improvements cluster — gold neurons orbiting the origin
const impGroup = new THREE.Group(); scene.add(impGroup);
function setImprovements(count) {
  while (impGroup.children.length < count) {
    const m = new THREE.Mesh(sphereGeo, mat(COLORS.improve));
    const i = impGroup.children.length;
    const a = i * 2.4, r = 3 + (i % 5) * 0.9;
    m.position.set(Math.cos(a) * r, 2 + (i * 0.5) % 4, Math.sin(a) * r);
    m.scale.setScalar(0.3);
    const halo = new THREE.Mesh(haloGeo, haloMat(COLORS.improve));
    halo.scale.setScalar(3.2); m.add(halo);
    impGroup.add(m);
  }
}

// ---------- interaction ----------
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
canvas.addEventListener('click', e => {
  const r = canvas.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(pickables, false)[0];
  if (hit) showDetail(hit.object.userData.sid, hit.object.userData.ev);
});
canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  canvas.style.cursor = ray.intersectObjects(pickables, false)[0] ? 'pointer' : 'grab';
});

function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
function fmtTs(ts) { return ts ? new Date(ts).toLocaleTimeString() : ''; }

function showDetail(sid, ev) {
  const hex = '#' + (COLORS[ev.type] ?? 0x94a3b8).toString(16).padStart(6, '0');
  let body = '', meta = '';
  const t = ev.type;
  if (t === 'user') { body = ev.text; meta = `author: ${esc(ev.author || 'you')} · session ${esc(sid)}`; }
  else if (t === 'assistant') {
    body = ev.text || '(no text)';
    if (ev.tools?.length) meta = `called: ${esc(ev.tools.join(', '))}`;
  }
  else if (t === 'reasoning') body = ev.text;
  else if (t === 'tool_call') {
    body = `${ev.name}(${ev.summary || ''})`;
    try { meta = 'args: ' + JSON.stringify(JSON.parse(ev.args || '{}'), null, 1); }
    catch { meta = 'args: ' + esc(ev.args); }
  }
  else if (t === 'tool_result') { body = `${ev.name}: ${ev.preview || ''}`; meta = ev.ok ? 'ok' : 'error'; }
  else if (t === 'skill') { body = `skill: ${ev.name}`; meta = 'loaded from harness skills/'; }
  else if (t === 'subagent') { body = `subagent: ${ev.agent || '(default)'}`; meta = esc(ev.title || ''); }
  else if (t === 'turn_end') { body = `stop: ${ev.stop}`; meta = `iters ${ev.iters} · tokens in ${ev.prompt} out ${ev.completion}`; }
  else if (t === 'session_start') { body = `model ${ev.model}`; meta = `${esc(ev.cwd)}\nharness: ${esc(ev.harness)}`; }
  else body = JSON.stringify(ev, null, 1);
  detailEl.style.display = 'block';
  detailEl.innerHTML = `
    <div class="dhead">
      <span class="dtype" style="background:${hex}22;color:${hex};border:1px solid ${hex}55">${LABELS[t] || t}</span>
      <span class="dtime">${fmtTs(ev.ts)} · ${esc(sid)}</span>
      <span class="dclose" onclick="this.parentElement.parentElement.style.display='none'">✕</span>
    </div>
    <pre>${esc(body)}</pre>
    ${meta ? `<div class="dmeta">${meta}</div>` : ''}`;
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') detailEl.style.display = 'none'; });

// ---------- sidebar ----------
let knownSids = new Set();
async function pollSessions() {
  let list;
  try { list = await api.sessions(); } catch { return; }
  list.sort((a, b) => (b.last_ts || 0) - (a.last_ts || 0));
  sesslistEl.innerHTML = list.map(s => {
    const live = Date.now() - (s.last_ts || 0) < 90_000;
    return `<div class="sess ${live ? 'live' : ''}" data-sid="${esc(s.id)}">
      <div class="sid">${esc(s.id)}</div>
      <div class="smeta">${esc(s.model || '?')} · ${s.events} ev</div>
      <div class="smeta">${esc((s.name ? s.name + ' · ' : '') + (s.cwd || ''))}</div>
    </div>`;
  }).join('') || '<div class="smeta" style="padding:8px">no sessions yet — run xera-rsi</div>';
  sesslistEl.querySelectorAll('.sess').forEach(el => {
    el.onclick = () => {
      sesslistEl.querySelectorAll('.sess').forEach(x => x.classList.remove('on'));
      el.classList.add('on');
      focusSession(el.dataset.sid);
    };
  });
  for (const s of list) {
    if (!knownSids.has(s.id)) { knownSids.add(s.id); tailSession(s.id); }
  }
}
function focusSession(sid) {
  const s = sessions.get(sid);
  if (!s) return;
  const target = s.center.clone();
  controls.target.lerp(target, 1);
  camera.position.copy(target.clone().add(new THREE.Vector3(6, 7, 10)));
}

// per-session tail loop
async function tailSession(sid) {
  let off = 0;
  while (true) {
    try {
      const res = await api.events(sid, off);
      if (res && res.events) {
        off = res.off;
        for (const ev of res.events) addEvent(sid, ev);
      }
    } catch { /* server down — retry */ }
    await new Promise(r => setTimeout(r, 1400));
  }
}

// ---------- stats view ----------
let statsTimer = 0;
async function renderStats() {
  let st;
  try { st = await api.stats(); } catch {
    statsEl.innerHTML = '<p style="color:var(--dim)">server unreachable — is serve.py running?</p>';
    return;
  }
  const t = st.totals, a = st.averages, im = st.improvements;
  const cards = [
    ['sessions', t.sessions], ['turns (prompts)', t.turns],
    ['tool calls', t.tool_calls], ['avg tools / session', a.tool_calls_per_session],
    ['skills loaded', t.skills_used], ['avg skills / session', a.skills_per_session],
    ['subagents', t.subagents], ['prompt tokens', t.prompt_tokens.toLocaleString()],
    ['completion tokens', t.completion_tokens.toLocaleString()],
    ['avg tokens / session', (a.prompt_tokens_per_session + a.completion_tokens_per_session).toLocaleString()],
    ['reasoning blocks', t.reasoning_blocks],
    ['harness² improves', im.composes, 'gold'],
  ];
  const maxTool = Math.max(1, ...st.tool_histogram.map(x => x[1]));
  const maxSkill = Math.max(1, ...st.skill_histogram.map(x => x[1]));
  statsEl.innerHTML = `
    <h2>overview</h2>
    <div class="cards">${cards.map(([k, v, cls]) =>
      `<div class="card ${cls || ''}"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`).join('')}</div>

    <h2>tool call histogram</h2>
    ${st.tool_histogram.length ? st.tool_histogram.map(([n, c]) =>
      `<div class="bar-wrap"><div class="blabel">${esc(n)}</div>
       <div class="bar" style="width:${(c / maxTool * 40 + 2)}em;background:var(--orange)"></div>${c}</div>`).join('')
      : '<div style="color:var(--dim)">none yet</div>'}

    <h2>skills used</h2>
    ${st.skill_histogram.length ? st.skill_histogram.map(([n, c]) =>
      `<div class="bar-wrap"><div class="blabel">${esc(n)}</div>
       <div class="bar" style="width:${(c / maxSkill * 40 + 2)}em;background:var(--pink)"></div>${c}</div>`).join('')
      : '<div style="color:var(--dim)">none yet — skills load via the skill_load tool</div>'}

    <h2>harness² self-improvement</h2>
    <div class="cards">
      <div class="card gold"><div class="v">${im.runs_started}</div><div class="k">improve runs</div></div>
      <div class="card gold"><div class="v">${im.composes}</div><div class="k">composes (self-edits)</div></div>
      <div class="card gold"><div class="v">${im.dry_runs}</div><div class="k">dry runs</div></div>
      <div class="card gold"><div class="v">${im.lessons.length}</div><div class="k">lessons adopted</div></div>
    </div>
    ${im.lessons.length ? im.lessons.map(l => `<div class="lesson">${esc(l)}</div>`).join('')
      : '<div style="color:var(--dim)">no lessons yet — run <b>xera-rsi improve "task"</b></div>'}
    <div style="margin-top:10px;font-size:11px;color:var(--dim)">recent improve events:</div>
    <table><tr><th>time</th><th>phase</th><th>adopt</th><th>rationale / task</th></tr>
    ${im.events.slice().reverse().map(e => `<tr>
      <td>${e.ts ? new Date(e.ts).toLocaleString() : ''}</td>
      <td>${esc(e.phase || '')}</td><td>${esc(e.adopt || '')}</td>
      <td>${esc(e.rationale || e.task || '')}</td></tr>`).join('')}</table>

    <h2>sessions</h2>
    <table>
      <tr><th>id</th><th>model</th><th>started</th><th>turns</th><th>tools</th>
          <th>skills</th><th>subagents</th><th>tokens in</th><th>tokens out</th><th>authors</th></tr>
      ${st.sessions.slice().reverse().map(s => `<tr>
        <td>${esc(s.id)}</td><td>${esc(s.model)}</td>
        <td>${s.first_ts ? new Date(s.first_ts).toLocaleString() : ''}</td>
        <td>${s.turns}</td><td>${s.tool_calls}</td>
        <td>${s.skills.length ? esc(s.skills.join(', ')) : '—'}</td>
        <td>${s.subagents}</td>
        <td>${s.prompt_tokens.toLocaleString()}</td><td>${s.completion_tokens.toLocaleString()}</td>
        <td>${esc(Object.entries(s.authors).map(([n, c]) => `${n}(${c})`).join(' '))}</td>
      </tr>`).join('')}
    </table>
    <p style="color:var(--dim);font-size:11px;margin:16px 0">
      live dir: ${esc(st.live_dir)} · updated ${new Date(st.generated_ms).toLocaleTimeString()}
    </p>`;
}

// ---------- tabs ----------
const tabNet = $('#tab-net'), tabStats = $('#tab-stats');
tabNet.onclick = () => {
  tabNet.classList.add('on'); tabStats.classList.remove('on');
  statsEl.style.display = 'none'; canvas.style.display = 'block';
  $('#legend').style.display = 'block'; $('#hint').style.display = 'block';
  clearInterval(statsTimer);
};
tabStats.onclick = () => {
  tabStats.classList.add('on'); tabNet.classList.remove('on');
  statsEl.style.display = 'block'; canvas.style.display = 'none';
  $('#legend').style.display = 'none'; $('#hint').style.display = 'none';
  detailEl.style.display = 'none';
  renderStats();
  clearInterval(statsTimer);
  statsTimer = setInterval(renderStats, 8000);
};

// ---------- resize + render loop ----------
function resize() {
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

const clock = new THREE.Clock();
function frame() {
  requestAnimationFrame(frame);
  const t = clock.getElapsedTime();
  // grow-in animation
  for (const m of pickables) {
    const ud = m.userData;
    if (m.scale.x < ud.target) {
      const s = Math.min(ud.target, m.scale.x + (ud.target - m.scale.x) * 0.12 + 0.01);
      m.scale.setScalar(s);
      if (m.children[0]) m.children[0].scale.setScalar(2.6);
    }
  }
  // gentle helix shimmer on the newest node of each session
  sessions.forEach(s => {
    const last = s.nodes[s.nodes.length - 1];
    if (last) last.rotation.y = t * 2;
  });
  impGroup.rotation.y = t * 0.15;
  controls.update();
  renderer.render(scene, camera);
}
frame();

// boot
pollSessions();
setInterval(pollSessions, 4000);
api.stats().then(st => setImprovements(st.improvements.composes)).catch(() => {});
setInterval(() => api.stats().then(st => setImprovements(st.improvements.composes)).catch(() => {}), 9000);
$('#livetext').textContent = 'live · polling ~/.xera-rsi/live';
