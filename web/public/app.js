const STORAGE_KEY = 'xfy_public_v2_store';
const LEGACY_STORAGE_KEY = 'xfy_public_v1_store';

const QUICK_MODES = [
  { id: 'pure', label: '元气日常', u: 0, v: 0 },
  { id: 'neko', label: '甜系软萌', u: -0.82, v: 0.82 },
  { id: 'tease', label: '知性姐姐', u: 0.82, v: 0.82 },
  { id: 'fragile', label: '安静慢热', u: -0.82, v: -0.82 },
  { id: 'brat', label: '傲娇毒舌', u: 0.82, v: -0.82 },
];

const CORNERS = [
  { key: 'neko_clingy', short: '软萌', light: '微甜软萌', full: '甜系软萌', dx: -Math.SQRT1_2, dy: Math.SQRT1_2, color: '#C87D62' },
  { key: 'older_tease', short: '姐姐', light: '轻熟知性', full: '知性姐姐', dx: Math.SQRT1_2, dy: Math.SQRT1_2, color: '#8E5258' },
  { key: 'fragile_porcelain', short: '慢热', light: '安静轻柔', full: '安静慢热', dx: -Math.SQRT1_2, dy: -Math.SQRT1_2, color: '#587A82' },
  { key: 'brat_mesugaki', short: '傲娇', light: '微辣傲娇', full: '傲娇毒舌', dx: Math.SQRT1_2, dy: -Math.SQRT1_2, color: '#BE8B32' },
];

const STARTER_PROMPTS = [
  '刚加完班出地铁，风好大',
  '帮我写个 Python 异步限流器顺便夸夸我',
  '今晚怎么这么安静，是不是偷偷想我了',
];

function round3(n) {
  return Math.round(n * 1000) / 1000;
}

function clampUnit(n) {
  const num = Number(n);
  if (!Number.isFinite(num)) return 0;
  return Math.max(-1, Math.min(1, num));
}

function randomShortCode(len = 4) {
  const alphabet = 'abcdefghjkmnpqrstuvwxyz23456789';
  let out = '';
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const arr = new Uint8Array(len);
    crypto.getRandomValues(arr);
    for (let i = 0; i < len; i += 1) {
      out += alphabet[arr[i] % alphabet.length];
    }
    return out;
  }
  for (let i = 0; i < len; i += 1) {
    out += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return out;
}

function makeId(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}_${crypto.randomUUID().slice(0, 12)}`;
  }
  return `${prefix}_${Date.now().toString(36)}_${randomShortCode(4)}`;
}

function orbToMoodState(rawU, rawV) {
  const u = clampUnit(rawU);
  const v = clampUnit(rawV);
  const r = Math.min(1, Math.max(Math.abs(u), Math.abs(v)));

  if (r < 0.08) {
    return {
      u: 0,
      v: 0,
      r: 0,
      weights: {
        xiaofeiyu_v4: 1.0,
        neko_clingy: 0,
        older_tease: 0,
        brat_mesugaki: 0,
        fragile_porcelain: 0,
      },
      label: '元气日常',
      accent: '#2D6A4F',
      matchedModeId: 'pure',
    };
  }

  const norm = Math.hypot(u, v) || 1;
  const ux = u / norm;
  const uy = v / norm;

  const sims = CORNERS.map((c) => Math.max(0, ux * c.dx + uy * c.dy));
  const sqSum = sims.reduce((acc, s) => acc + s * s, 0) || 1;
  const totalBudget = Math.min(0.95, (r / 0.82) * 0.85);
  const baseWeight = r <= 0.82 ? 1.0 : round3(1.0 - ((r - 0.82) / 0.18) * 0.1);

  const weights = {
    xiaofeiyu_v4: baseWeight,
    neko_clingy: 0,
    older_tease: 0,
    brat_mesugaki: 0,
    fragile_porcelain: 0,
  };

  const shares = CORNERS.map((c, idx) => {
    const share = (sims[idx] * sims[idx]) / sqSum;
    weights[c.key] = round3(share * totalBudget);
    return { corner: c, share };
  }).sort((a, b) => b.share - a.share);

  let label = '元气日常';
  const top = shares[0];
  const second = shares[1];
  if (top && top.share >= 0.8) {
    label = r < 0.45 ? top.corner.light : top.corner.full;
  } else if (top && second && second.share >= 0.15) {
    label = `${top.corner.short} × ${second.corner.short}混调`;
  } else if (top) {
    label = top.corner.full;
  }

  const matched = QUICK_MODES.find(
    (m) => m.id !== 'pure' && Math.abs(m.u - u) < 0.06 && Math.abs(m.v - v) < 0.06
  );

  return {
    u: round3(u),
    v: round3(v),
    r: round3(r),
    weights,
    label,
    accent: top ? top.corner.color : '#2D6A4F',
    matchedModeId: matched ? matched.id : null,
  };
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function createSeedProfile(customNickname = '', confirmed = false) {
  const now = Date.now();
  const code = randomShortCode(4);
  const profileId = makeId('p');
  const nickname = customNickname ? customNickname.trim().slice(0, 24) : `访客·${code}`;
  const freshConvId = makeId('conv');

  return {
    profileId,
    nickname,
    nicknameConfirmed: Boolean(confirmed),
    orb: { u: 0, v: 0 },
    activeConvId: freshConvId,
    conversations: [
      {
        id: freshConvId,
        title: '新对话',
        isSample: false,
        orb: { u: 0, v: 0 },
        updatedAt: now,
        messages: [],
      },
    ],
  };
}

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.profiles) && parsed.profiles.length > 0) {
        if (!parsed.visitorId) parsed.visitorId = makeId('v');
        parsed.profiles.forEach((p) => {
          if (!p.profileId) p.profileId = p.visitorId || makeId('p');
          if (p.nicknameConfirmed === undefined) {
            p.nicknameConfirmed = Boolean(p.nickname && !p.nickname.startsWith('访客·'));
          }
        });
        return parsed;
      }
    }
  } catch {
    // ignore corrupted localStorage and re-seed cleanly
  }

  try {
    const legacyRaw = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacyRaw) {
      const legacy = JSON.parse(legacyRaw);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      if (legacy && Array.isArray(legacy.profiles) && legacy.profiles.length > 0) {
        const activeOld =
          legacy.profiles.find((p) => p.profileId === legacy.activeProfileId) || legacy.profiles[0];
        const migrated = createSeedProfile(activeOld.nickname || '', Boolean(activeOld.nicknameConfirmed));
        migrated.profileId = activeOld.profileId || migrated.profileId;
        const migratedStore = {
          version: 2,
          visitorId: legacy.visitorId || makeId('v'),
          activeProfileId: migrated.profileId,
          profiles: [migrated],
        };
        saveStore(migratedStore);
        return migratedStore;
      }
    }
  } catch {
    // ignore legacy migration error
  }

  const first = createSeedProfile('', false);
  const store = {
    version: 2,
    visitorId: makeId('v'),
    activeProfileId: first.profileId,
    profiles: [first],
  };
  saveStore(store);
  return store;
}

function saveStore(store) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch {
    // ignore storage quota overflow
  }
}

function splitBubbles(text) {
  const normalized = String(text || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const lines = normalized.split('\n');
  const bubbles = [];
  let inFence = false;
  let fenceLines = [];

  for (const line of lines) {
    if (!inFence) {
      const openMatch = line.match(/^\s*```([^\r\n`]*)$/);
      if (openMatch) {
        inFence = true;
        fenceLines = [`\`\`\`${openMatch[1].trim()}`];
      } else {
        const trimmed = line.trim();
        if (trimmed.length > 0) bubbles.push(trimmed);
      }
    } else if (/^\s*```\s*$/.test(line)) {
      fenceLines.push('```');
      bubbles.push(fenceLines.join('\n'));
      inFence = false;
      fenceLines = [];
    } else {
      fenceLines.push(line);
    }
  }

  if (inFence && fenceLines.length > 0) {
    bubbles.push(fenceLines.join('\n'));
  }

  return bubbles;
}

function parseCodeFenceBubble(text) {
  if (typeof text !== 'string' || !text.startsWith('```')) return null;
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const withoutClose = normalized.endsWith('\n```')
    ? normalized.slice(0, -4)
    : normalized === '```'
      ? ''
      : normalized;
  const nlIdx = withoutClose.indexOf('\n');
  if (nlIdx === -1) {
    return { language: withoutClose.slice(3).trim(), content: '' };
  }
  return {
    language: withoutClose.slice(3, nlIdx).trim(),
    content: withoutClose.slice(nlIdx + 1),
  };
}

function populateBubbleContent(bubbleEl, text) {
  const fence = parseCodeFenceBubble(text);
  if (!fence) {
    bubbleEl.textContent = text;
    return;
  }
  bubbleEl.classList.add('is-code-bubble');

  const header = document.createElement('div');
  header.className = 'code-bubble-header';

  const langEl = document.createElement('span');
  langEl.className = 'code-bubble-lang tabular';
  langEl.textContent = fence.language || 'text';
  header.appendChild(langEl);

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'code-copy-btn';
  copyBtn.textContent = '复制';
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(fence.content);
      copyBtn.textContent = '已复制';
      setTimeout(() => {
        copyBtn.textContent = '复制';
      }, 1400);
    } catch {
      copyBtn.textContent = '复制失败';
    }
  });
  header.appendChild(copyBtn);
  bubbleEl.appendChild(header);

  const preEl = document.createElement('pre');
  preEl.className = 'code-bubble-pre';
  const normLang = String(fence.language || '').trim().toLowerCase();
  if (!normLang || ['text', 'txt', 'plain', 'plaintext', 'markdown', 'md'].includes(normLang)) {
    preEl.style.whiteSpace = 'pre-wrap';
    preEl.style.wordBreak = 'break-word';
  }
  const codeEl = document.createElement('code');
  codeEl.textContent = fence.content;
  preEl.appendChild(codeEl);
  bubbleEl.appendChild(preEl);
}

function formatClockTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function formatDayLabel(ts) {
  const d = new Date(ts);
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  return sameDay ? '今天' : `${d.getMonth() + 1}月${d.getDate()}日`;
}

const store = loadStore();
const runtime = {
  isSending: false,
  streamingBubbles: [],
  streamingPartial: '',
  pendingError: '',
  lastFailedPrompt: '',
  renamingConvId: null,
  animateMsgId: null,
  lastHitNodeId: '',
  clusterPopoverOpen: false,
  clusterPollTimer: null,
  serviceStatus: { ready: false, checking: true, cluster: null },
};

const el = {
  sidebar: document.getElementById('sidebar'),
  drawerBackdrop: document.getElementById('drawer-backdrop'),
  mobileMenuBtn: document.getElementById('mobile-menu-btn'),
  profileBadgeBtn: document.getElementById('profile-badge-btn'),
  profileAvatarInitial: document.getElementById('profile-avatar-initial'),
  profileBadgeName: document.getElementById('profile-badge-name'),
  resetMoodBtn: document.getElementById('reset-mood-btn'),
  orbPad: document.getElementById('orb-pad'),
  orbAmbient: document.getElementById('orb-ambient'),
  orbPuck: document.getElementById('orb-puck'),
  moodReadoutPill: document.getElementById('mood-readout-pill'),
  modePills: document.getElementById('mode-pills'),
  conversationsList: document.getElementById('conversations-list'),
  clearChatsBtn: document.getElementById('clear-chats-btn'),
  newChatBtn: document.getElementById('new-chat-btn'),
  mixChip: document.getElementById('mix-chip'),
  convSubtitle: document.getElementById('conv-subtitle'),
  serviceStatusPill: document.getElementById('service-status-pill'),
  serviceStatusText: document.getElementById('service-status-text'),
  clusterPopover: document.getElementById('cluster-popover'),
  clusterPopoverSummary: document.getElementById('cluster-popover-summary'),
  clusterPopoverList: document.getElementById('cluster-popover-list'),
  chatScroll: document.getElementById('chat-scroll'),
  chatTimeline: document.getElementById('chat-timeline'),
  composerForm: document.getElementById('composer-form'),
  composerInput: document.getElementById('composer-input'),
  sendBtn: document.getElementById('send-btn'),

  welcomeModal: document.getElementById('welcome-modal'),
  welcomeForm: document.getElementById('welcome-form'),
  welcomeNicknameInput: document.getElementById('welcome-nickname-input'),

  profileModal: document.getElementById('profile-modal'),
  closeProfileModal: document.getElementById('close-profile-modal'),
  profileRenameForm: document.getElementById('profile-rename-form'),
  profileNicknameInput: document.getElementById('profile-nickname-input'),
  createProfileBtn: document.getElementById('create-profile-btn'),
  profileList: document.getElementById('profile-list'),

  renameModal: document.getElementById('rename-modal'),
  closeRenameModal: document.getElementById('close-rename-modal'),
  cancelRenameBtn: document.getElementById('cancel-rename-btn'),
  renameForm: document.getElementById('rename-form'),
  renameInput: document.getElementById('rename-input'),
};

function getActiveProfile() {
  const found = store.profiles.find((p) => p.profileId === store.activeProfileId);
  if (found) return found;
  store.activeProfileId = store.profiles[0].profileId;
  saveStore(store);
  return store.profiles[0];
}

function getActiveConversation() {
  const prof = getActiveProfile();
  return prof.conversations.find((c) => c.id === prof.activeConvId) || prof.conversations[0] || null;
}

function toggleMobileDrawer(open) {
  if (open) {
    el.sidebar.classList.add('is-open');
    el.drawerBackdrop.hidden = false;
  } else {
    el.sidebar.classList.remove('is-open');
    el.drawerBackdrop.hidden = true;
  }
}

function renderProfileBadge() {
  const prof = getActiveProfile();
  const initial = (prof.nickname || '访').trim().slice(0, 1) || '访';
  el.profileAvatarInitial.textContent = initial;
  el.profileBadgeName.textContent = prof.nickname;
}

function renderModePills(matchedModeId) {
  el.modePills.innerHTML = '';
  QUICK_MODES.forEach((mode) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `mode-pill-btn${matchedModeId === mode.id ? ' is-active' : ''}`;
    btn.dataset.modeId = mode.id;
    btn.textContent = mode.label;
    btn.addEventListener('click', () => {
      applyOrbCoords(mode.u, mode.v);
    });
    el.modePills.appendChild(btn);
  });
}

function applyOrbCoords(u, v) {
  const prof = getActiveProfile();
  const conv = getActiveConversation();
  const mood = orbToMoodState(u, v);
  prof.orb = { u: mood.u, v: mood.v };
  if (conv && !conv.isSample) {
    conv.orb = { u: mood.u, v: mood.v };
  }
  saveStore(store);
  renderMoodCompass();
}

function renderMoodCompass() {
  const prof = getActiveProfile();
  const conv = getActiveConversation();
  const orb = (conv && conv.orb) ? conv.orb : (prof.orb || { u: 0, v: 0 });
  const mood = orbToMoodState(orb.u, orb.v);

  const leftPct = 50 + mood.u * 32;
  const topPct = 50 - mood.v * 32;
  el.orbPuck.style.left = `${leftPct}%`;
  el.orbPuck.style.top = `${topPct}%`;
  el.orbPuck.style.setProperty('--orb-accent', mood.accent);
  el.orbPuck.style.setProperty('--orb-halo', hexToRgba(mood.accent, 0.16));

  el.orbAmbient.style.background = `radial-gradient(circle at ${leftPct}% ${topPct}%, ${hexToRgba(mood.accent, 0.22)} 0%, ${hexToRgba(mood.accent, 0)} 62%)`;
  el.orbPad.setAttribute('aria-valuetext', mood.label);

  el.moodReadoutPill.textContent = `当前：${mood.label}`;
  el.mixChip.textContent = mood.label;
  renderModePills(mood.matchedModeId);
}

function formatPublicNodeTitle(rawId) {
  const s = String(rawId || '').trim();
  const m = s.match(/^([A-Za-z0-9]+)-(\d+)$/);
  if (m) return `${m[1]} #${m[2]}`;
  return s || '节点';
}

function renderClusterPopover() {
  if (!el.clusterPopover || !el.clusterPopoverList || !el.clusterPopoverSummary) return;

  const cluster = runtime.serviceStatus.cluster;
  const items = Array.isArray(cluster?.items) ? cluster.items : [];
  const totalCap = Number(cluster?.totalCapacity ?? 0);
  el.clusterPopoverSummary.textContent = totalCap > 0 ? `${totalCap} 并发槽位` : '未配置';

  el.clusterPopoverList.innerHTML = '';
  if (items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'cluster-popover-empty';
    empty.textContent = '暂无在线节点';
    el.clusterPopoverList.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'cluster-node-row';

    const left = document.createElement('div');
    left.className = 'cluster-node-left';

    const dot = document.createElement('span');
    const isFull = item.healthy && item.available <= 0;
    dot.className = `cluster-node-dot${!item.healthy ? ' is-cooling' : isFull ? ' is-full' : ''}`;
    left.appendChild(dot);

    const name = document.createElement('span');
    name.className = 'cluster-node-name tabular';
    name.textContent = formatPublicNodeTitle(item.id);
    left.appendChild(name);

    if (runtime.lastHitNodeId && runtime.lastHitNodeId === item.id) {
      const hitTag = document.createElement('span');
      hitTag.className = 'cluster-hit-tag';
      hitTag.textContent = '本轮命中';
      left.appendChild(hitTag);
    }

    const right = document.createElement('div');
    right.className = 'cluster-node-right';

    const cap = Math.max(1, Number(item.capacity || 4));
    const inFlight = Math.max(0, Number(item.inFlight || 0));
    const pct = Math.min(100, Math.round((inFlight / cap) * 100));

    const bar = document.createElement('div');
    bar.className = 'cluster-slot-bar';
    const fill = document.createElement('div');
    fill.className = `cluster-slot-fill${isFull ? ' is-full' : ''}`;
    fill.style.width = `${pct}%`;
    bar.appendChild(fill);
    right.appendChild(bar);

    const slots = document.createElement('span');
    slots.className = 'cluster-node-slots tabular';
    slots.textContent = `${inFlight} / ${cap} 并发`;
    right.appendChild(slots);

    const state = document.createElement('span');
    state.className = `cluster-node-state${!item.healthy ? ' is-cooling' : ''}`;
    state.textContent = !item.healthy ? '冷却切换中' : isFull ? '满载' : '就绪';
    right.appendChild(state);

    row.appendChild(left);
    row.appendChild(right);
    el.clusterPopoverList.appendChild(row);
  });
}

function setClusterPopoverOpen(open) {
  if (!el.clusterPopover || !el.serviceStatusPill) return;
  const next = Boolean(open);
  runtime.clusterPopoverOpen = next;
  el.clusterPopover.hidden = !next;
  el.serviceStatusPill.setAttribute('aria-expanded', next ? 'true' : 'false');

  if (runtime.clusterPollTimer) {
    clearInterval(runtime.clusterPollTimer);
    runtime.clusterPollTimer = null;
  }

  if (next) {
    renderClusterPopover();
    refreshServiceStatus();
    runtime.clusterPollTimer = setInterval(refreshServiceStatus, 10_000);
  }
}

function renderServiceStatus() {
  el.serviceStatusPill.className = 'status-pill';
  if (runtime.serviceStatus.checking) {
    el.serviceStatusPill.classList.add('status-checking');
    el.serviceStatusText.textContent = '连接中';
    renderClusterPopover();
    return;
  }

  const cluster = runtime.serviceStatus.cluster;
  if (runtime.serviceStatus.ready && cluster && cluster.nodes > 0) {
    const summary = cluster.gpuSummary || `${cluster.nodes}节点`;
    const inFlight = Number(cluster.totalInFlight ?? 0);
    const totalCap = Number(cluster.totalCapacity ?? 0);
    const healthyNodes = Number(cluster.healthyNodes ?? 0);

    if (healthyNodes === 0) {
      el.serviceStatusPill.classList.add('status-busy');
      el.serviceStatusText.textContent = `${summary} · 唤醒中`;
    } else if (totalCap > 0 && inFlight >= totalCap) {
      el.serviceStatusPill.classList.add('status-busy');
      el.serviceStatusText.textContent = `${summary} · ${inFlight}/${totalCap} 满载`;
    } else {
      el.serviceStatusPill.classList.add('status-ready');
      el.serviceStatusText.textContent = `${summary} · ${inFlight}/${totalCap} 并发`;
    }
  } else if (runtime.serviceStatus.ready) {
    el.serviceStatusPill.classList.add('status-ready');
    el.serviceStatusText.textContent = '在线 · 可对话';
  } else {
    el.serviceStatusPill.classList.add('status-offline');
    el.serviceStatusText.textContent = '暂忙 · 离线';
  }
  renderClusterPopover();
}

async function refreshServiceStatus() {
  try {
    const res = await fetch('/api/status', { cache: 'no-store' });
    const data = await res.json().catch(() => ({}));
    runtime.serviceStatus = {
      checking: false,
      ready: Boolean(res.ok && data.ready),
      cluster: data?.cluster || null,
    };
  } catch {
    runtime.serviceStatus = { checking: false, ready: false, cluster: null };
  }
  renderServiceStatus();
}

function renderConversations() {
  const prof = getActiveProfile();
  el.conversationsList.innerHTML = '';

  if (!prof.conversations.length) {
    const empty = document.createElement('div');
    empty.className = 'thread-empty';
    empty.textContent = '还没有对话，直接发一句话开始聊天';
    el.conversationsList.appendChild(empty);
    return;
  }

  const sorted = [...prof.conversations].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  sorted.forEach((conv) => {
    const card = document.createElement('div');
    card.className = `thread-card${conv.id === prof.activeConvId ? ' is-active' : ''}`;
    card.dataset.convId = conv.id;

    const lastMsg = conv.messages?.[conv.messages.length - 1];
    const preview = lastMsg
      ? lastMsg.rawContent
          .replace(/```[a-zA-Z0-9_+#.-]*\s*/g, '')
          .replace(/```/g, '')
          .replace(/\s+/g, ' ')
          .trim() || '代码卡片'
      : '点击直接开始聊天';
    const timeStr = formatClockTime(conv.updatedAt);

    card.innerHTML = `
      <div class="thread-main">
        <div class="thread-top">
          <span class="thread-title"></span>
          <span class="thread-time tabular">${timeStr}</span>
        </div>
        <div class="thread-preview"></div>
      </div>
      <div class="thread-actions">
        <button class="thread-action-btn" data-action="rename" type="button" title="重命名" aria-label="重命名对话">
          <svg class="icon-14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9" />
            <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
          </svg>
        </button>
        <button class="thread-action-btn is-danger" data-action="delete" type="button" title="删除" aria-label="删除对话">
          <svg class="icon-14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6h18" />
            <path d="M8 6V4h8v2" />
            <path d="M19 6l-1 14H6L5 6" />
          </svg>
        </button>
      </div>
    `;

    card.querySelector('.thread-title').textContent = conv.title;
    card.querySelector('.thread-preview').textContent = preview;

    card.addEventListener('click', (e) => {
      const actionBtn = e.target.closest('[data-action]');
      if (actionBtn) {
        e.stopPropagation();
        const act = actionBtn.dataset.action;
        if (act === 'rename') openRenameModal(conv);
        if (act === 'delete') deleteConversation(conv.id);
        return;
      }
      selectConversation(conv.id);
      toggleMobileDrawer(false);
    });

    el.conversationsList.appendChild(card);
  });
}

function renderPeerBurst(bubbles, options = {}) {
  const { animate = false, moodLabel = '', createdAt = null, nodeId = '', isStreaming = false } = options;
  const turn = document.createElement('div');
  turn.className = 'msg-turn is-peer';

  const row = document.createElement('div');
  row.className = 'peer-turn-row';

  const avatar = document.createElement('div');
  avatar.className = 'turn-avatar';
  avatar.textContent = '鱼';

  const stack = document.createElement('div');
  stack.className = 'bubble-stack';

  bubbles.forEach((line, idx) => {
    const b = document.createElement('div');
    let posClass = '';
    if (bubbles.length > 1) {
      if (idx === 0) posClass = ' is-first-of-burst';
      else if (idx === bubbles.length - 1 && !isStreaming) posClass = ' is-last-of-burst';
      else posClass = ' is-middle-of-burst';
    }
    b.className = `chat-bubble is-peer${posClass}${animate ? ' is-entering' : ''}`;
    if (animate) {
      b.style.animationDelay = `${idx * 160}ms`;
    }
    populateBubbleContent(b, line);
    stack.appendChild(b);
  });

  if (!isStreaming) {
    const footer = document.createElement('div');
    footer.className = 'bubble-meta-footer';
    if (createdAt) {
      const timeSpan = document.createElement('span');
      timeSpan.className = 'tabular';
      timeSpan.textContent = formatClockTime(createdAt);
      footer.appendChild(timeSpan);
    }
    if (moodLabel) {
      const tag = document.createElement('span');
      tag.className = 'bubble-mode-tag';
      tag.textContent = moodLabel;
      footer.appendChild(tag);
    }
    if (nodeId) {
      const nodeTag = document.createElement('span');
      nodeTag.className = 'bubble-node-tag tabular';
      nodeTag.textContent = nodeId;
      footer.appendChild(nodeTag);
    }
    stack.appendChild(footer);
  }

  row.appendChild(avatar);
  row.appendChild(stack);
  turn.appendChild(row);
  return turn;
}

function renderMessages() {
  const conv = getActiveConversation();
  el.chatTimeline.innerHTML = '';

  el.convSubtitle.textContent = conv?.isSample
    ? '示例回放 · 可直接点击左下角新对话开始聊天'
    : '拖动左侧性格球，随时切换说话语气';

  const messages = conv?.messages || [];
  if (!messages.length && !runtime.isSending && !runtime.pendingError) {
    const empty = document.createElement('div');
    empty.className = 'empty-stage';
    empty.innerHTML = `
      <h3 class="empty-stage-title">和小肥鱼打个招呼</h3>
      <p class="empty-stage-desc">左侧球盘可自由混调性格倾向，发一句话马上开聊。</p>
      <div class="starter-chips"></div>
    `;
    const chipsWrap = empty.querySelector('.starter-chips');
    STARTER_PROMPTS.forEach((promptText) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'starter-chip-btn';
      btn.textContent = promptText;
      btn.addEventListener('click', () => {
        sendMessage(promptText);
      });
      chipsWrap.appendChild(btn);
    });
    el.chatTimeline.appendChild(empty);
    return;
  }

  let lastDay = '';
  messages.forEach((msg) => {
    const dayLabel = formatDayLabel(msg.createdAt);
    if (dayLabel !== lastDay) {
      lastDay = dayLabel;
      const div = document.createElement('div');
      div.className = 'time-divider';
      div.textContent = dayLabel;
      el.chatTimeline.appendChild(div);
    }

    const bubbles = Array.isArray(msg.bubbles) && msg.bubbles.length
      ? msg.bubbles
      : splitBubbles(msg.rawContent);

    if (msg.role === 'user') {
      const turn = document.createElement('div');
      turn.className = 'msg-turn is-user';
      const stack = document.createElement('div');
      stack.className = 'bubble-stack';
      bubbles.forEach((line) => {
        const b = document.createElement('div');
        b.className = 'chat-bubble is-user';
        populateBubbleContent(b, line);
        stack.appendChild(b);
      });
      const footer = document.createElement('div');
      footer.className = 'bubble-meta-footer tabular';
      footer.textContent = formatClockTime(msg.createdAt);
      stack.appendChild(footer);
      turn.appendChild(stack);
      el.chatTimeline.appendChild(turn);
    } else {
      const shouldAnimate = runtime.animateMsgId === msg.id;
      el.chatTimeline.appendChild(
        renderPeerBurst(bubbles, {
          animate: shouldAnimate,
          moodLabel: msg.moodLabel || '',
          createdAt: msg.createdAt,
          nodeId: msg.nodeId || '',
        })
      );
    }
  });

  if (runtime.isSending) {
    const activeBubbles = runtime.streamingBubbles.length > 0 ? [...runtime.streamingBubbles] : [];
    const partialLines = runtime.streamingPartial ? splitBubbles(runtime.streamingPartial) : [];
    const displayBubbles = activeBubbles.length > 0 ? activeBubbles : partialLines;

    if (displayBubbles.length > 0) {
      el.chatTimeline.appendChild(renderPeerBurst(displayBubbles, { animate: false, isStreaming: true }));
    } else {
      const turn = document.createElement('div');
      turn.className = 'msg-turn is-peer';
      const row = document.createElement('div');
      row.className = 'peer-turn-row';

      const avatar = document.createElement('div');
      avatar.className = 'turn-avatar';
      avatar.textContent = '鱼';

      const wrap = document.createElement('div');
      wrap.className = 'typing-row-wrap';
      const typing = document.createElement('div');
      typing.className = 'typing-bubble';
      typing.setAttribute('aria-label', '对方正在输入');
      typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
      const label = document.createElement('span');
      label.className = 'typing-label';
      label.textContent = '对方正在输入...';
      wrap.appendChild(typing);
      wrap.appendChild(label);

      row.appendChild(avatar);
      row.appendChild(wrap);
      turn.appendChild(row);
      el.chatTimeline.appendChild(turn);
    }
  }

  if (runtime.pendingError) {
    const errCard = document.createElement('div');
    errCard.className = 'error-inline-card';
    const msgSpan = document.createElement('span');
    msgSpan.textContent = runtime.pendingError;
    errCard.appendChild(msgSpan);

    if (runtime.lastFailedPrompt) {
      const retryBtn = document.createElement('button');
      retryBtn.type = 'button';
      retryBtn.className = 'secondary-btn';
      retryBtn.textContent = '再试一次';
      retryBtn.addEventListener('click', () => {
        const promptToRetry = runtime.lastFailedPrompt;
        runtime.pendingError = '';
        runtime.lastFailedPrompt = '';
        sendMessage(promptToRetry, { skipAppendUser: true });
      });
      errCard.appendChild(retryBtn);
    }
    el.chatTimeline.appendChild(errCard);
  }

  el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
}

function selectConversation(convId) {
  const prof = getActiveProfile();
  const conv = prof.conversations.find((c) => c.id === convId);
  if (!conv) return;
  prof.activeConvId = conv.id;
  if (conv.orb) prof.orb = { ...conv.orb };
  runtime.pendingError = '';
  runtime.lastFailedPrompt = '';
  runtime.animateMsgId = null;
  saveStore(store);
  renderMoodCompass();
  renderConversations();
  renderMessages();
}

function createNewConversation() {
  const prof = getActiveProfile();
  const existingEmpty = prof.conversations.find((c) => !c.isSample && c.messages.length === 0);
  if (existingEmpty) {
    selectConversation(existingEmpty.id);
    toggleMobileDrawer(false);
    el.composerInput.focus();
    return;
  }

  const conv = {
    id: makeId('conv'),
    title: '新对话',
    isSample: false,
    orb: { ...prof.orb },
    updatedAt: Date.now(),
    messages: [],
  };
  prof.conversations.unshift(conv);
  prof.activeConvId = conv.id;
  runtime.pendingError = '';
  runtime.lastFailedPrompt = '';
  saveStore(store);
  renderMoodCompass();
  renderConversations();
  renderMessages();
  toggleMobileDrawer(false);
  el.composerInput.focus();
}

function clearAllConversations() {
  if (runtime.isSending) return;
  const prof = getActiveProfile();
  const fresh = {
    id: makeId('conv'),
    title: '新对话',
    isSample: false,
    orb: { ...prof.orb },
    updatedAt: Date.now(),
    messages: [],
  };
  prof.conversations = [fresh];
  prof.activeConvId = fresh.id;
  runtime.pendingError = '';
  runtime.lastFailedPrompt = '';
  saveStore(store);
  renderMoodCompass();
  renderConversations();
  renderMessages();
}

function deleteConversation(convId) {
  const prof = getActiveProfile();
  prof.conversations = prof.conversations.filter((c) => c.id !== convId);
  if (prof.conversations.length === 0) {
    const fresh = {
      id: makeId('conv'),
      title: '新对话',
      isSample: false,
      orb: { ...prof.orb },
      updatedAt: Date.now(),
      messages: [],
    };
    prof.conversations.push(fresh);
    prof.activeConvId = fresh.id;
  } else if (prof.activeConvId === convId) {
    prof.activeConvId = prof.conversations[0].id;
  }
  saveStore(store);
  renderMoodCompass();
  renderConversations();
  renderMessages();
}

function openRenameModal(conv) {
  runtime.renamingConvId = conv.id;
  el.renameInput.value = conv.title;
  el.renameModal.hidden = false;
  el.renameInput.focus();
}

function closeRenameModal() {
  el.renameModal.hidden = true;
  runtime.renamingConvId = null;
}

function renderProfileList() {
  const active = getActiveProfile();
  el.profileList.innerHTML = '';

  store.profiles.forEach((prof) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `profile-item-btn${prof.profileId === active.profileId ? ' is-current' : ''}`;
    const nameSpan = document.createElement('span');
    nameSpan.textContent = prof.nickname;
    const metaSpan = document.createElement('span');
    metaSpan.className = 'profile-item-meta';
    const convCount = prof.conversations.filter((c) => c.messages.length > 0).length;
    metaSpan.textContent = prof.profileId === active.profileId ? '使用中' : `${convCount} 段对话`;

    btn.appendChild(nameSpan);
    btn.appendChild(metaSpan);
    btn.addEventListener('click', () => {
      store.activeProfileId = prof.profileId;
      runtime.pendingError = '';
      runtime.lastFailedPrompt = '';
      saveStore(store);
      closeProfileModal();
      renderProfileBadge();
      renderMoodCompass();
      renderConversations();
      renderMessages();
    });
    el.profileList.appendChild(btn);
  });
}

function openProfileModal() {
  const prof = getActiveProfile();
  el.profileNicknameInput.value = prof.nickname;
  renderProfileList();
  el.profileModal.hidden = false;
  setTimeout(() => el.profileNicknameInput.focus(), 30);
}

function closeProfileModal() {
  el.profileModal.hidden = true;
}

async function sendMessage(rawText, options = {}) {
  const clean = String(rawText || '').trim();
  if (!clean || runtime.isSending) return;

  const prof = getActiveProfile();
  let conv = getActiveConversation();
  if (!conv) {
    createNewConversation();
    conv = getActiveConversation();
  }

  const orb = conv.orb || prof.orb || { u: 0, v: 0 };
  const mood = orbToMoodState(orb.u, orb.v);

  if (!options.skipAppendUser) {
    if (conv.title === '新对话' && conv.messages.length === 0) {
      conv.title = clean.replace(/\s+/g, ' ').slice(0, 18) || '新对话';
    }
    conv.messages.push({
      id: makeId('msg'),
      role: 'user',
      rawContent: clean,
      bubbles: [clean],
      moodLabel: mood.label,
      createdAt: Date.now(),
    });
    conv.updatedAt = Date.now();
    saveStore(store);
  }

  runtime.isSending = true;
  runtime.streamingBubbles = [];
  runtime.streamingPartial = '';
  runtime.pendingError = '';
  runtime.lastFailedPrompt = '';
  runtime.animateMsgId = null;
  el.sendBtn.disabled = true;

  renderConversations();
  renderMessages();
  refreshServiceStatus();

  const payloadMessages = conv.messages
    .filter((m) => m.role === 'user' || m.role === 'assistant')
    .slice(-20)
    .map((m) => ({ role: m.role, content: m.rawContent }));

  try {
    const payloadBody = {
      visitorId: store.visitorId,
      profileId: prof.profileId,
      visitorNickname: prof.nickname,
      conversationId: conv.id,
      messages: payloadMessages,
      weights: mood.weights,
    };

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payloadBody),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData?.error?.message || '刚才没连上，请稍后再试一次。');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = 'message';
    let doneReceived = false;
    let hitNodeId = String(res.headers.get('x-xfy-cluster-node') || '').trim();
    if (hitNodeId === 'waking') hitNodeId = '';
    if (hitNodeId) {
      runtime.lastHitNodeId = hitNodeId;
      renderClusterPopover();
    }

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('event:')) {
          currentEvent = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          const dataStr = line.slice(5).trim();
          if (!dataStr) continue;
          let payload = {};
          try {
            payload = JSON.parse(dataStr);
          } catch {
            continue;
          }

          if (currentEvent === 'meta') {
            if (typeof payload.node_id === 'string' && payload.node_id.trim()) {
              hitNodeId = payload.node_id.trim();
              runtime.lastHitNodeId = hitNodeId;
              renderClusterPopover();
            }
          } else if (currentEvent === 'bubble') {
            if (typeof payload.text === 'string' && payload.text.length > 0) {
              runtime.streamingBubbles.push(payload.text);
              renderMessages();
            }
          } else if (currentEvent === 'delta') {
            if (typeof payload.text === 'string') {
              runtime.streamingPartial = payload.text;
              if (runtime.streamingBubbles.length === 0) {
                renderMessages();
              }
            }
          } else if (currentEvent === 'done') {
            doneReceived = true;
            if (typeof payload.node_id === 'string' && payload.node_id.trim()) {
              hitNodeId = payload.node_id.trim();
              runtime.lastHitNodeId = hitNodeId;
            }
            const finalText = String(payload.text || runtime.streamingPartial || '').trim();
            const finalBubbles = Array.isArray(payload.bubbles) && payload.bubbles.length > 0
              ? payload.bubbles
              : splitBubbles(finalText);
            const msgId = makeId('msg');
            conv.messages.push({
              id: msgId,
              role: 'assistant',
              rawContent: finalText || '信号刚晃了一下，你再跟我说一遍嘛',
              bubbles: finalBubbles.length > 0 ? finalBubbles : [finalText || '信号刚晃了一下，你再跟我说一遍嘛'],
              moodLabel: mood.label,
              ...(hitNodeId ? { nodeId: hitNodeId } : {}),
              createdAt: Date.now(),
            });
            conv.updatedAt = Date.now();
            runtime.animateMsgId = runtime.streamingBubbles.length === 0 ? msgId : null;
            saveStore(store);
          } else if (currentEvent === 'error') {
            throw new Error(payload.message || '消息发送遇到小波动，请再试一次。');
          }
        }
      }
    }

    if (!doneReceived && runtime.streamingPartial.trim()) {
      const fallbackText = runtime.streamingPartial.trim();
      const msgId = makeId('msg');
      conv.messages.push({
        id: msgId,
        role: 'assistant',
        rawContent: fallbackText,
        bubbles: splitBubbles(fallbackText),
        moodLabel: mood.label,
        ...(hitNodeId ? { nodeId: hitNodeId } : {}),
        createdAt: Date.now(),
      });
      conv.updatedAt = Date.now();
      saveStore(store);
    }
  } catch (err) {
    runtime.pendingError = err.message || '网络有点抖动，请稍后再试。';
    runtime.lastFailedPrompt = clean;
  } finally {
    runtime.isSending = false;
    runtime.streamingBubbles = [];
    runtime.streamingPartial = '';
    el.sendBtn.disabled = false;
    renderConversations();
    renderMessages();
    refreshServiceStatus();
  }
}

function updateOrbFromPointer(clientX, clientY) {
  const rect = el.orbPad.getBoundingClientRect();
  const relX = (clientX - rect.left) / rect.width;
  const relY = (clientY - rect.top) / rect.height;
  const u = clampUnit((relX - 0.5) / 0.32);
  const v = clampUnit((0.5 - relY) / 0.32);
  applyOrbCoords(u, v);
}

let isOrbDragging = false;
el.orbPad.addEventListener('pointerdown', (e) => {
  isOrbDragging = true;
  el.orbPad.classList.add('is-dragging');
  el.orbPad.setPointerCapture(e.pointerId);
  updateOrbFromPointer(e.clientX, e.clientY);
});

el.orbPad.addEventListener('pointermove', (e) => {
  if (!isOrbDragging) return;
  updateOrbFromPointer(e.clientX, e.clientY);
});

function stopOrbDrag(e) {
  if (!isOrbDragging) return;
  isOrbDragging = false;
  el.orbPad.classList.remove('is-dragging');
  if (el.orbPad.hasPointerCapture(e.pointerId)) {
    el.orbPad.releasePointerCapture(e.pointerId);
  }
}

el.orbPad.addEventListener('pointerup', stopOrbDrag);
el.orbPad.addEventListener('pointercancel', stopOrbDrag);

el.orbPad.addEventListener('keydown', (e) => {
  const step = 0.14;
  const prof = getActiveProfile();
  const conv = getActiveConversation();
  const cur = (conv && conv.orb) ? conv.orb : prof.orb;
  if (e.key === 'ArrowLeft') { e.preventDefault(); applyOrbCoords(cur.u - step, cur.v); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); applyOrbCoords(cur.u + step, cur.v); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); applyOrbCoords(cur.u, cur.v + step); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); applyOrbCoords(cur.u, cur.v - step); }
  else if (e.key === 'Home' || e.key === '0') { e.preventDefault(); applyOrbCoords(0, 0); }
});

el.mobileMenuBtn.addEventListener('click', () => toggleMobileDrawer(true));
el.drawerBackdrop.addEventListener('click', () => toggleMobileDrawer(false));

el.resetMoodBtn.addEventListener('click', () => applyOrbCoords(0, 0));
el.newChatBtn.addEventListener('click', createNewConversation);
if (el.clearChatsBtn) {
  el.clearChatsBtn.addEventListener('click', clearAllConversations);
}

el.profileBadgeBtn.addEventListener('click', openProfileModal);
el.closeProfileModal.addEventListener('click', closeProfileModal);
el.profileModal.addEventListener('click', (e) => {
  if (e.target === el.profileModal) closeProfileModal();
});

el.welcomeForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const name = el.welcomeNicknameInput.value.trim().slice(0, 24);
  if (!name) {
    el.welcomeNicknameInput.focus();
    return;
  }
  const prof = getActiveProfile();
  prof.nickname = name;
  prof.nicknameConfirmed = true;
  saveStore(store);
  el.welcomeModal.hidden = true;
  renderProfileBadge();
  el.composerInput.focus();
});

el.profileRenameForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const name = el.profileNicknameInput.value.trim().slice(0, 24);
  if (!name) return;
  const prof = getActiveProfile();
  prof.nickname = name;
  prof.nicknameConfirmed = true;
  saveStore(store);
  renderProfileBadge();
  renderProfileList();
  closeProfileModal();
});

el.createProfileBtn.addEventListener('click', () => {
  const next = createSeedProfile('', true);
  store.profiles.unshift(next);
  store.activeProfileId = next.profileId;
  runtime.pendingError = '';
  runtime.lastFailedPrompt = '';
  saveStore(store);
  closeProfileModal();
  renderProfileBadge();
  renderMoodCompass();
  renderConversations();
  renderMessages();
});

el.closeRenameModal.addEventListener('click', closeRenameModal);
el.cancelRenameBtn.addEventListener('click', closeRenameModal);
el.renameModal.addEventListener('click', (e) => {
  if (e.target === el.renameModal) closeRenameModal();
});

el.renameForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const prof = getActiveProfile();
  const conv = prof.conversations.find((c) => c.id === runtime.renamingConvId);
  const nextTitle = el.renameInput.value.trim().slice(0, 60);
  if (conv && nextTitle) {
    conv.title = nextTitle;
    saveStore(store);
  }
  closeRenameModal();
  renderConversations();
  renderMessages();
});

el.composerInput.addEventListener('input', () => {
  el.composerInput.style.height = 'auto';
  el.composerInput.style.height = `${Math.min(el.composerInput.scrollHeight, 140)}px`;
});

el.composerInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    el.composerForm.requestSubmit();
  }
});

el.composerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = el.composerInput.value;
  if (!text.trim() || runtime.isSending) return;
  el.composerInput.value = '';
  el.composerInput.style.height = 'auto';
  await sendMessage(text);
});

el.serviceStatusPill.addEventListener('click', (e) => {
  e.stopPropagation();
  setClusterPopoverOpen(!runtime.clusterPopoverOpen);
});

document.addEventListener('click', (e) => {
  if (!runtime.clusterPopoverOpen) return;
  if (el.clusterPopover?.contains(e.target) || el.serviceStatusPill?.contains(e.target)) return;
  setClusterPopoverOpen(false);
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && runtime.clusterPopoverOpen) {
    setClusterPopoverOpen(false);
    el.serviceStatusPill.focus();
  }
});

renderProfileBadge();
renderMoodCompass();
renderConversations();
renderMessages();
renderServiceStatus();
refreshServiceStatus();
setInterval(refreshServiceStatus, 30000);

const activeBootProfile = getActiveProfile();
if (!activeBootProfile.nicknameConfirmed) {
  el.welcomeNicknameInput.value = '';
  el.welcomeModal.hidden = false;
  setTimeout(() => el.welcomeNicknameInput.focus(), 40);
}
