export const MAX_BODY_BYTES = 128 * 1024;
export const MAX_MESSAGES = 20;
export const MAX_MESSAGE_CHARS = 3000;
export const WEIGHT_MIN = 0.0;
export const WEIGHT_MAX = 1.5;
export const RATE_LIMIT_WINDOW_MS = 60_000;
export const RATE_LIMIT_MAX_REQUESTS = 10;

export const WEIGHT_KEYS = [
  'xiaofeiyu_v4',
  'neko_clingy',
  'older_tease',
  'brat_mesugaki',
  'fragile_porcelain',
];

export const DEFAULT_WEIGHTS_OBJ = {
  xiaofeiyu_v4: 1.0,
  neko_clingy: 0.0,
  older_tease: 0.0,
  brat_mesugaki: 0.0,
  fragile_porcelain: 0.0,
};

export const DEFAULT_UPSTREAM_CAPACITY = 4;
export const DEFAULT_EWMA_FIRST_BYTE_MS = 2500;
export const EWMA_ALPHA = 0.3;
export const UPSTREAM_COOLDOWN_MS = 25_000;

const ipWindows = new Map();
const clusterHealthMap = new Map();

function getClusterNodeKey(node) {
  return String(node?.url || node?.id || '').trim();
}

function defaultEwmaForNode(node) {
  const gpu = String(node?.gpu || '').toUpperCase();
  if (gpu === 'H200' || gpu === 'H100' || gpu === 'A100' || gpu === 'G4') return 900;
  return DEFAULT_EWMA_FIRST_BYTE_MS;
}

function gpuPriorityBonus(node) {
  const gpu = String(node?.gpu || '').toUpperCase();
  if (gpu === 'H200' || gpu === 'H100' || gpu === 'A100' || gpu === 'G4') return -0.28;
  return 0;
}

function ensureClusterNodeState(node) {
  const key = getClusterNodeKey(node);
  let state = clusterHealthMap.get(key);
  if (!state) {
    state = {
      inFlight: 0,
      ewmaFirstByteMs: defaultEwmaForNode(node),
      cooldownUntil: 0,
      lastActiveAt: null,
      upstreamActiveSlots: null,
    };
    clusterHealthMap.set(key, state);
  }
  return state;
}

export function getNodeHealth(node, now = Date.now()) {
  const key = getClusterNodeKey(node);
  const state = clusterHealthMap.get(key);
  if (!state) {
    return {
      inFlight: 0,
      ewmaFirstByteMs: defaultEwmaForNode(node),
      cooldownUntil: 0,
      lastActiveAt: null,
      upstreamActiveSlots: null,
      healthy: true,
    };
  }
  return {
    inFlight: state.inFlight,
    ewmaFirstByteMs: state.ewmaFirstByteMs,
    cooldownUntil: state.cooldownUntil,
    lastActiveAt: state.lastActiveAt,
    upstreamActiveSlots: state.upstreamActiveSlots,
    healthy: now >= state.cooldownUntil,
  };
}

export function acquireUpstream(node, now = Date.now()) {
  const state = ensureClusterNodeState(node);
  state.inFlight += 1;
  state.lastActiveAt = now;
  return state.inFlight;
}

export function recordUpstreamMeta(node, meta = {}, now = Date.now()) {
  const state = ensureClusterNodeState(node);
  state.lastActiveAt = now;
  if (meta && Number.isFinite(Number(meta.active_slots))) {
    state.upstreamActiveSlots = Math.max(0, Math.floor(Number(meta.active_slots)));
  }
}

export function shouldTripUpstreamCooldown(status, error) {
  if (error) return true;
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

export function releaseUpstream(node, outcome = {}) {
  const state = ensureClusterNodeState(node);
  state.inFlight = Math.max(0, state.inFlight - 1);
  const now = outcome.now ?? Date.now();
  state.lastActiveAt = now;

  if (outcome.ok === true) {
    state.cooldownUntil = 0;
    if (Number.isFinite(outcome.firstByteMs) && outcome.firstByteMs >= 0) {
      state.ewmaFirstByteMs =
        state.ewmaFirstByteMs * (1 - EWMA_ALPHA) + outcome.firstByteMs * EWMA_ALPHA;
    }
    return getNodeHealth(node, now);
  }

  if (outcome.ok === false && outcome.tripCooldown !== false) {
    if (shouldTripUpstreamCooldown(outcome.status, outcome.error) || outcome.tripCooldown === true) {
      state.cooldownUntil = now + UPSTREAM_COOLDOWN_MS;
    }
  }
  return getNodeHealth(node, now);
}

export function resetClusterHealthForTests() {
  clusterHealthMap.clear();
}

export function getEnv(name, envOverride) {
  if (envOverride && Object.prototype.hasOwnProperty.call(envOverride, name)) {
    return String(envOverride[name] ?? '');
  }
  const procEnv = typeof process !== 'undefined' && process?.env ? process.env : {};
  return String(procEnv[name] ?? '');
}

export function trimUrl(url) {
  return String(url || '').trim().replace(/\/+$/, '');
}

export function normalizeChatCompletionsUrl(rawUrl) {
  const clean = trimUrl(rawUrl);
  if (!clean) return '';
  if (clean.endsWith('/chat/completions')) return clean;
  if (clean.endsWith('/v1')) return `${clean}/chat/completions`;
  return `${clean}/v1/chat/completions`;
}

function sanitizeNodeId(raw, fallback) {
  const clean = String(raw || '')
    .trim()
    .replace(/[^a-zA-Z0-9_.-]/g, '')
    .slice(0, 64);
  return clean || fallback;
}

function parseNodeCapacity(raw, fallback = DEFAULT_UPSTREAM_CAPACITY) {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.max(1, Math.floor(n));
}

function normalizeGpuLabel(rawGpu, rawId = '') {
  const source = String(rawGpu || '').trim();
  const hint = (source || String(rawId || '')).toLowerCase();
  if (hint.includes('h200')) return 'H200';
  if (hint.includes('h100')) return 'H100';
  if (hint.includes('a100')) return 'A100';
  if (hint.includes('g4')) return 'G4';
  if (hint.includes('l4')) return 'L4';
  if (source) {
    const clean = source.replace(/[^a-zA-Z0-9_-]/g, '').toUpperCase().slice(0, 16);
    if (clean) return clean;
  }
  return 'L4';
}

export function resolveUpstreams(envOverride) {
  const rawUrls = getEnv('XFY_UPSTREAM_URLS', envOverride);
  const rawKeys = getEnv('XFY_SERVICE_KEY', envOverride);
  const rawCaps = getEnv('XFY_UPSTREAM_CAPS', envOverride);
  const rawIds = getEnv('XFY_UPSTREAM_IDS', envOverride);
  const rawGpus = getEnv('XFY_UPSTREAM_GPUS', envOverride);

  const keys = rawKeys
    .split(',')
    .map((k) => k.trim())
    .filter(Boolean);

  const caps = rawCaps.split(',').map((c) => c.trim());
  const ids = rawIds.split(',').map((id) => id.trim());
  const gpus = rawGpus.split(',').map((g) => g.trim());

  const urls = rawUrls
    .split(',')
    .map((u) => normalizeChatCompletionsUrl(u))
    .filter(Boolean);

  const gpuCounters = new Map();
  return urls.map((url, idx) => {
    const id = sanitizeNodeId(ids[idx], `node-${idx + 1}`);
    const gpu = normalizeGpuLabel(gpus[idx], id);
    const seq = (gpuCounters.get(gpu) || 0) + 1;
    gpuCounters.set(gpu, seq);
    return {
      id,
      publicId: `${gpu}-${seq}`,
      gpu,
      url,
      key: keys.length > 0 ? keys[idx % keys.length] : '',
      capacity: parseNodeCapacity(caps[idx], DEFAULT_UPSTREAM_CAPACITY),
    };
  });
}

const GPU_TIER_ORDER = ['H200', 'H100', 'A100', 'G4', 'L4'];

export function formatGpuSummary(upstreams) {
  if (!Array.isArray(upstreams) || upstreams.length === 0) return '';
  const counts = new Map();
  for (const node of upstreams) {
    const gpu = String(node?.gpu || 'L4').trim() || 'L4';
    counts.set(gpu, (counts.get(gpu) || 0) + 1);
  }
  const entries = [...counts.entries()].sort((a, b) => {
    const ia = GPU_TIER_ORDER.indexOf(a[0]);
    const ib = GPU_TIER_ORDER.indexOf(b[0]);
    const ra = ia === -1 ? 999 : ia;
    const rb = ib === -1 ? 999 : ib;
    if (ra !== rb) return ra - rb;
    return a[0].localeCompare(b[0]);
  });
  return entries.map(([gpu, count]) => `${count}\u00d7${gpu}`).join(' + ');
}

export function buildClusterSummary(upstreams, now = Date.now()) {
  const items = upstreams.map((node) => {
    const health = getNodeHealth(node, now);
    const capacity = Math.max(1, node.capacity || DEFAULT_UPSTREAM_CAPACITY);
    const inFlight = health.inFlight;
    return {
      id: node.publicId || `${node.gpu || 'L4'}-1`,
      gpu: node.gpu || 'L4',
      capacity,
      inFlight,
      available: Math.max(0, capacity - inFlight),
      healthy: health.healthy,
      ewmaFirstByteMs: Math.round(health.ewmaFirstByteMs),
    };
  });

  const totalCapacity = items.reduce((sum, item) => sum + item.capacity, 0);
  const totalInFlight = items.reduce((sum, item) => sum + item.inFlight, 0);
  const healthyNodes = items.filter((item) => item.healthy).length;

  return {
    nodes: items.length,
    healthyNodes,
    totalCapacity,
    totalInFlight,
    gpuSummary: formatGpuSummary(upstreams),
    items,
  };
}

export function hasServiceKey(envOverride) {
  return Boolean(getEnv('XFY_SERVICE_KEY', envOverride).trim());
}

export function pickOrderedUpstreams(upstreams, now = Date.now(), options = {}) {
  if (!upstreams.length) return [];
  const jitterFn = typeof options.jitterFn === 'function' ? options.jitterFn : () => Math.random() * 0.05;

  const healthy = [];
  const cooling = [];

  for (const node of upstreams) {
    const health = getNodeHealth(node, now);
    const cap = Math.max(1, node.capacity || DEFAULT_UPSTREAM_CAPACITY);
    const score =
      health.inFlight / cap + health.ewmaFirstByteMs / 20_000 + gpuPriorityBonus(node) + jitterFn(node);
    const entry = { node, health, score };
    if (health.healthy) {
      healthy.push(entry);
    } else {
      cooling.push(entry);
    }
  }

  healthy.sort((a, b) => a.score - b.score);
  cooling.sort((a, b) => a.health.cooldownUntil - b.health.cooldownUntil || a.score - b.score);

  return [...healthy, ...cooling].map((item) => item.node);
}

export function extractClientIp(request) {
  const xff = request.headers.get('x-forwarded-for');
  if (xff) {
    const first = xff.split(',')[0]?.trim();
    if (first) return first.slice(0, 128);
  }
  const realIp = request.headers.get('x-real-ip') || request.headers.get('cf-connecting-ip');
  if (realIp && realIp.trim()) return realIp.trim().slice(0, 128);
  return 'unknown';
}

export function checkRateLimit(ip, options = {}) {
  const maxRequests = options.maxRequests ?? RATE_LIMIT_MAX_REQUESTS;
  const windowMs = options.windowMs ?? RATE_LIMIT_WINDOW_MS;
  const now = options.now ?? Date.now();
  const cutoff = now - windowMs;

  if (ipWindows.size > 2000) {
    for (const [k, stamps] of ipWindows.entries()) {
      const kept = stamps.filter((t) => t > cutoff);
      if (kept.length === 0) ipWindows.delete(k);
      else ipWindows.set(k, kept);
    }
  }

  const prev = ipWindows.get(ip) || [];
  const active = prev.filter((t) => t > cutoff);

  if (active.length >= maxRequests) {
    const oldest = active[0] ?? now;
    const retryAfterSec = Math.max(1, Math.ceil((oldest + windowMs - now) / 1000));
    ipWindows.set(ip, active);
    return { allowed: false, remaining: 0, retryAfterSec };
  }

  active.push(now);
  ipWindows.set(ip, active);
  return {
    allowed: true,
    remaining: Math.max(0, maxRequests - active.length),
    retryAfterSec: 0,
  };
}

export function resetRateLimitForTests() {
  ipWindows.clear();
}

export function round3(val) {
  return Math.round(val * 1000) / 1000;
}

export function clampWeight(val, fallback = 0.0) {
  const num = Number(val);
  if (!Number.isFinite(num)) return fallback;
  return round3(Math.min(WEIGHT_MAX, Math.max(WEIGHT_MIN, num)));
}

export function sanitizeWeights(input) {
  if (Array.isArray(input) && input.length === 5) {
    const obj = {
      xiaofeiyu_v4: clampWeight(input[0], 1.0),
      neko_clingy: clampWeight(input[1], 0.0),
      older_tease: clampWeight(input[2], 0.0),
      brat_mesugaki: clampWeight(input[3], 0.0),
      fragile_porcelain: clampWeight(input[4], 0.0),
    };
    return {
      weights: obj,
      tuple: WEIGHT_KEYS.map((k) => obj[k]),
    };
  }

  const src = input && typeof input === 'object' ? input : {};
  const baseVal = src.xiaofeiyu_v4 !== undefined ? src.xiaofeiyu_v4 : src.xiaofeiyu_v2;
  const obj = {
    xiaofeiyu_v4: clampWeight(baseVal, 1.0),
    neko_clingy: clampWeight(src.neko_clingy, 0.0),
    older_tease: clampWeight(src.older_tease, 0.0),
    brat_mesugaki: clampWeight(src.brat_mesugaki, 0.0),
    fragile_porcelain: clampWeight(src.fragile_porcelain, 0.0),
  };
  return {
    weights: obj,
    tuple: WEIGHT_KEYS.map((k) => obj[k]),
  };
}

export function formatModelTag(tuple) {
  const base = tuple.map((v) => Number(v).toFixed(3)).join(',');
  return `xiaofeiyu-v4|${base}`;
}

export function sanitizeIdentifier(raw, fallback, maxLen = 64) {
  const s = String(raw || '').trim();
  const clean = s.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, maxLen);
  return clean || fallback;
}

export function sanitizeNickname(raw, fallback = '访客') {
  const s = String(raw || '')
    .replace(/[\u0000-\u001f\u007f]/g, '')
    .trim()
    .slice(0, 40);
  return s || fallback;
}

export function sanitizeChatPayload(body) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw Object.assign(new Error('请求格式须为 JSON 对象'), { status: 400, code: 'INVALID_BODY' });
  }

  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw Object.assign(new Error('messages 不能为空'), { status: 400, code: 'EMPTY_MESSAGES' });
  }

  const cleaned = [];
  for (const item of body.messages) {
    if (!item || typeof item !== 'object') continue;
    const role = item.role;
    if (role !== 'user' && role !== 'assistant') continue;
    if (typeof item.content !== 'string') continue;
    const trimmed = item.content.trim();
    if (!trimmed) continue;
    cleaned.push({ role, content: trimmed });
  }

  const sliced = cleaned.slice(-MAX_MESSAGES);
  if (sliced.length === 0) {
    throw Object.assign(new Error('请先输入有效对话内容'), { status: 400, code: 'EMPTY_MESSAGES' });
  }

  const lastMsg = sliced[sliced.length - 1];
  if (lastMsg.role !== 'user') {
    throw Object.assign(new Error('最后一条消息须为用户输入'), { status: 400, code: 'INVALID_TURN_ORDER' });
  }

  if (lastMsg.content.length > MAX_MESSAGE_CHARS) {
    throw Object.assign(
      new Error(`单条消息请控制在 ${MAX_MESSAGE_CHARS} 字符以内`),
      { status: 400, code: 'MESSAGE_TOO_LONG' }
    );
  }

  const boundedMessages = sliced.map((m) => ({
    role: m.role,
    content: m.content.slice(0, MAX_MESSAGE_CHARS),
  }));

  const visitorId = sanitizeIdentifier(body.visitorId, 'v_anon', 64);
  const profileId = sanitizeIdentifier(body.profileId || body.visitorId, 'p_default', 64);
  const visitorNickname = sanitizeNickname(body.visitorNickname, '访客');
  const conversationId = sanitizeIdentifier(body.conversationId, 'conv_default', 64);
  const { weights, tuple } = sanitizeWeights(body.weights);

  return {
    visitorId,
    profileId,
    visitorNickname,
    conversationId,
    messages: boundedMessages,
    weights,
    weightTuple: tuple,
    modelTag: formatModelTag(tuple),
    lastUserMessage: boundedMessages[boundedMessages.length - 1].content,
  };
}

export function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store',
      ...extraHeaders,
    },
  });
}

export function errorResponse(status, code, message, extraHeaders = {}) {
  return jsonResponse({ error: { code, message } }, status, extraHeaders);
}

export function formatSseEvent(eventName, data) {
  return `event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function splitReplyBubbles(rawContent) {
  if (typeof rawContent !== 'string') return [];
  const normalized = rawContent.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
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
    fenceLines.push('```');
    bubbles.push(fenceLines.join('\n'));
  }

  return bubbles;
}

export class IncrementalBubbleEmitter {
  constructor() {
    this.lineBuffer = '';
    this.inFence = false;
    this.fenceLines = [];
    this.emittedCount = 0;
  }

  push(chunk) {
    if (!chunk) return [];
    const normalized = String(chunk).replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const emitted = [];

    for (let i = 0; i < normalized.length; i += 1) {
      const ch = normalized[i];
      if (ch !== '\n') {
        this.lineBuffer += ch;
        continue;
      }
      const line = this.lineBuffer;
      this.lineBuffer = '';
      const completed = this._processLine(line);
      if (completed !== null) {
        emitted.push({ index: this.emittedCount++, text: completed });
      }
    }
    return emitted;
  }

  flush() {
    const emitted = [];
    if (this.lineBuffer.length > 0) {
      const line = this.lineBuffer;
      this.lineBuffer = '';
      const completed = this._processLine(line);
      if (completed !== null) {
        emitted.push({ index: this.emittedCount++, text: completed });
      }
    }
    if (this.inFence && this.fenceLines.length > 0) {
      this.fenceLines.push('```');
      emitted.push({ index: this.emittedCount++, text: this.fenceLines.join('\n') });
      this.inFence = false;
      this.fenceLines = [];
    }
    return emitted;
  }

  _processLine(line) {
    if (!this.inFence) {
      const openMatch = line.match(/^\s*```([^\r\n`]*)$/);
      if (openMatch) {
        this.inFence = true;
        this.fenceLines = [`\`\`\`${openMatch[1].trim()}`];
        return null;
      }
      const trimmed = line.trim();
      return trimmed.length > 0 ? trimmed : null;
    }

    if (/^\s*```\s*$/.test(line)) {
      this.fenceLines.push('```');
      const card = this.fenceLines.join('\n');
      this.inFence = false;
      this.fenceLines = [];
      return card;
    }

    this.fenceLines.push(line);
    return null;
  }
}

export async function hashIp(ip, salt = '') {
  const input = new TextEncoder().encode(`${salt}:${ip}`);
  const digest = await crypto.subtle.digest('SHA-256', input);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
