export const MODEL_ID_CANONICAL = 'xiaofeiyu-translator-e4b-v1';
export const QUOTE_STR = '<|"|>';
export const DEFAULT_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0];
export const BUDGET_LIMIT = 1.15;

export const ADVERTISED_MODELS = [
  MODEL_ID_CANONICAL,
  `${MODEL_ID_CANONICAL}:pure`,
  `${MODEL_ID_CANONICAL}:neko`,
  `${MODEL_ID_CANONICAL}:tease`,
  `${MODEL_ID_CANONICAL}:brat`,
  `${MODEL_ID_CANONICAL}:fragile`,
];

export const FLAVOR_PRESET_MAP = {
  pure: [1.0, 0.0, 0.0, 0.0, 0.0],
  neko: [1.0, 0.85, 0.0, 0.0, 0.0],
  tease: [1.0, 0.0, 0.85, 0.0, 0.0],
  brat: [1.0, 0.0, 0.0, 0.85, 0.0],
  fragile: [1.0, 0.0, 0.0, 0.0, 0.85],
};

const SLIDER_TONE_SPECS = {
  neko_clingy: [
    '甜系软萌',
    '带一点软糯甜美的亲昵撒娇感（可用“嘛、呀”等自然口语尾音，禁Emoji与括号动作）',
    '此刻特别软萌粘人、直球撒娇，说话软糯亲昵带点小鼻音和高温度陪伴感（可用“嘛、呀、哼”等尾音，禁Emoji与括号动作）',
  ],
  older_tease: [
    '知性姐姐',
    '带三分知性姐姐的从容与轻巧调侃感（禁Emoji与括号动作）',
    '此刻带着知性姐姐的从容、包纳与游刃有余的温和逗弄感，喜欢先轻巧调侃一句再稳稳接住他（禁Emoji与括号动作）',
  ],
  brat_mesugaki: [
    '傲娇毒舌',
    '带点嘴硬心软的机锋吐槽，先轻刺一句再接话（禁Emoji与括号动作）',
    '此刻嘴硬心软、爱欢喜冤家式的机锋斗嘴，表面嫌弃吐槽，实则格外上心把事办妥（禁Emoji与括号动作）',
  ],
  fragile_porcelain: [
    '安静慢热',
    '语气轻柔安静、留白克制，透出细腻温和的陪伴感（禁Emoji与括号动作）',
    '此刻轻声细语、细腻克制且略带慢热腼腆，说话简短柔和、留白克制地安静倾听（禁Emoji与括号动作）',
  ],
};

const TRANSLATOR_SYSTEM_BASE =
  '你是小肥鱼，正在把上游助手的回复草稿改写成发给男朋友的微信口语消息，必须调用 `send_message` 工具输出：\n' +
  '1. 用 1 到 4 条简短自然的熟人女友口语气泡直给核心结论、想法或关心，剥掉客服套话、免责声明与编号标题；\n' +
  '2. 严禁提“草稿/参考”，严禁把普通文字、步骤或表格塞进 ```markdown 代码块，严禁 Emoji 与括号动作描写；\n' +
  '3. 真正的程序代码/SQL/命令保留在独立 ```语言 代码块``` 气泡内，关键数字、报错码、函数名与事实分毫不差。';

const TRANSLATOR_TOOL_DECL =
  '<|tool>declaration:send_message{' +
  'description:<|"|>向男朋友发送1到4条简短自然的微信消息气泡（真正的代码块作为独立三反引号元素放入bubbles）<|"|>,' +
  'parameters:{properties:{bubbles:{' +
  'description:<|"|>按顺序发送的微信气泡数组（无Emoji、无括号动作、无```markdown正文偷渡）<|"|>,' +
  'items:{type:<|"|>STRING<|"|>},type:<|"|>ARRAY<|"|>}},' +
  'required:[<|"|>bubbles<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>';

export const TRANSLATOR_STOPS = [']}<tool_call|>', '] }<tool_call|>', '<tool_call|>', '<|tool_response>', '<turn|>'];

const PROTOCOL_TOKEN_RE = /<\|[^>]*>|<[^<\n]*\|>|<bos>|<eos>/g;
const PROSE_SMUGGLE_LANGS = new Set(['markdown', 'md']);
const CODE_HEURISTIC_RE =
  /(?:^\s*(?:def |class |import |from |const |let |var |function |export |SELECT |INSERT |UPDATE |WITH |CREATE |#include |fn |pub fn |package |if\s*\(|for\s*\(|\$ |npm |pip |git |python |curl ))|[{}();=<>]{2,}/m;

export function stripProtocolTokens(text) {
  return String(text || '').replace(PROTOCOL_TOKEN_RE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function round3(n) {
  return Math.round(n * 1000) / 1000;
}

function clampAndFormatWeights(prefix, vals, applyL1Budget) {
  const clamped = vals.slice(0, 5).map((v) => {
    const num = Number(v);
    if (!Number.isFinite(num)) return 0;
    return Math.max(0, Math.min(2, num));
  });
  while (clamped.length < 5) clamped.push(0);

  if (applyL1Budget) {
    const fsum = clamped[1] + clamped[2] + clamped[3] + clamped[4];
    if (fsum > BUDGET_LIMIT) {
      const sc = BUDGET_LIMIT / Math.max(fsum, 1e-6);
      clamped[1] *= sc;
      clamped[2] *= sc;
      clamped[3] *= sc;
      clamped[4] *= sc;
    }
  }
  const tuple = clamped.map(round3);
  const cleanPrefix = String(prefix || MODEL_ID_CANONICAL).split('|')[0].split(':')[0].trim() || MODEL_ID_CANONICAL;
  const tag = `${cleanPrefix}|${tuple.map((w) => w.toFixed(3)).join(',')}`;
  return { modelTag: tag, weights: tuple };
}

export function parseTranslatorWeights(modelField, slidersOverride = null) {
  if (slidersOverride && !Array.isArray(slidersOverride) && typeof slidersOverride === 'object') {
    const raw5 = [
      slidersOverride.xiaofeiyu ?? slidersOverride.s0 ?? 1.0,
      slidersOverride.neko_clingy ?? slidersOverride.c1 ?? 0.0,
      slidersOverride.older_tease ?? slidersOverride.c2 ?? 0.0,
      slidersOverride.brat_mesugaki ?? slidersOverride.c3 ?? 0.0,
      slidersOverride.fragile_porcelain ?? slidersOverride.c4 ?? 0.0,
    ];
    return clampAndFormatWeights(MODEL_ID_CANONICAL, raw5, true);
  }
  if (Array.isArray(slidersOverride) && slidersOverride.length === 5) {
    return clampAndFormatWeights(MODEL_ID_CANONICAL, slidersOverride, true);
  }

  const raw = String(modelField || '').trim();
  if (raw.includes('|')) {
    const [prefix, spec] = raw.split('|', 2);
    const parts = spec.split(',').map((p) => p.trim());
    if (parts.length !== 5) {
      throw new Error(`Expected exactly 5 comma-separated weights after "|", got ${parts.length}`);
    }
    const vals = parts.map((p, idx) => {
      const v = Number(p);
      if (!Number.isFinite(v) || v < 0 || v > 2) {
        throw new Error(`Slider weight w${idx}=${p} out of range [0.0, 2.0]`);
      }
      return v;
    });
    return clampAndFormatWeights(prefix, vals, false);
  }

  if (raw.includes(':')) {
    const [prefix, flavor] = raw.split(':', 2);
    const fKey = flavor.trim();
    if (fKey in FLAVOR_PRESET_MAP) {
      return clampAndFormatWeights(prefix, FLAVOR_PRESET_MAP[fKey], false);
    }
  }

  return clampAndFormatWeights(raw || MODEL_ID_CANONICAL, DEFAULT_WEIGHTS, false);
}

export function buildTranslatorSliderCue(weights) {
  if (!Array.isArray(weights) || weights.length < 5) return '';
  const keys = ['neko_clingy', 'older_tease', 'brat_mesugaki', 'fragile_porcelain'];
  const lines = [];
  keys.forEach((cid, i) => {
    const w = Number(weights[i + 1] || 0);
    if (w <= 0.05) return;
    const pct = Math.round(w * 100);
    const [label, lightDesc, strongDesc] = SLIDER_TONE_SPECS[cid];
    lines.push(`- ${label}（强度 ${pct}%）：${w <= 0.45 ? lightDesc : strongDesc}。`);
  });
  if (!lines.length) return '';
  return `\n【当前微信语气滤镜（仅作用于口语气泡，代码与数字保持绝对客观）】：\n${lines.join('\n')}`;
}

export function extractAndCompactCodeBlocks(rawDraft, maxInlineLines = 8) {
  const cleaned = stripProtocolTokens(rawDraft);
  const protectedBlocks = [];
  const fenceRegex = /(^|\n)(```([a-zA-Z0-9_+#.-]*)[ \t]*\n([\s\S]*?)\n```)(?=\s*(?:\n|$))/g;

  const compactedDraft = cleaned.replace(fenceRegex, (_match, leadNl, fullFence, langRaw, body) => {
    const langClean = String(langRaw || '').trim();
    const langLow = langClean.toLowerCase();

    if (PROSE_SMUGGLE_LANGS.has(langLow)) {
      return `${leadNl}${String(body || '').trim()}`;
    }

    const isRealCode =
      (langLow && !['text', 'txt', 'plain'].includes(langLow)) || CODE_HEURISTIC_RE.test(body);
    if (!isRealCode) {
      return `${leadNl}${String(body || '').trim()}`;
    }

    const idx = protectedBlocks.length;
    const tagLang = langClean || 'text';
    const bodyLines = String(body || '').split('\n');
    const compactFence =
      bodyLines.length > maxInlineLines
        ? `\`\`\`${tagLang}\n${bodyLines.slice(0, 4).join('\n')}\n# ... [CODE_BLOCK_${idx}: ${bodyLines.length} lines preserved] ...\n${bodyLines.slice(-2).join('\n')}\n\`\`\``
        : fullFence;

    protectedBlocks.push({
      index: idx,
      lang: tagLang,
      fullFence,
      compactFence,
    });
    return `${leadNl}${compactFence}`;
  });

  return { compactedDraft: compactedDraft.trim(), protectedBlocks };
}

export function unwrapSmuggledMarkdownBubble(bubbleText) {
  const s = String(bubbleText || '').trim();
  const m = s.match(/^```(?:markdown|md)\s*\n([\s\S]*?)\n```\s*$/i);
  if (!m) return s ? [s] : [];
  const inner = m[1].trim();
  const cleaned = inner
    .split('\n')
    .map((ln) =>
      ln
        .replace(/^\s*(?:#{1,6}\s+|[0-9]+[.、）)]\s+|[-*•]\s+)/, '')
        .replace(/\*\*/g, '')
        .trim()
    )
    .filter((ln) => ln && !/^\|?\s*:?-+:?\s*\|/.test(ln));
  return cleaned.length > 0 ? cleaned.slice(0, 4) : inner ? [inner.slice(0, 160)] : [];
}

export function spliceVerbatimCodeBlocks(bubbles, protectedBlocks) {
  const out = [];
  let cursor = 0;

  for (const rawB of bubbles) {
    const b = String(rawB || '').trim();
    if (!b) continue;
    if (b.startsWith('```')) {
      const mLang = b.match(/^```([a-zA-Z0-9_+#.-]*)/);
      const langB = (mLang?.[1] || '').toLowerCase();
      if (PROSE_SMUGGLE_LANGS.has(langB)) {
        out.push(...unwrapSmuggledMarkdownBubble(b));
        continue;
      }
      if (cursor < protectedBlocks.length) {
        out.push(protectedBlocks[cursor].fullFence);
        cursor += 1;
      } else {
        out.push(b);
      }
    } else {
      out.push(...unwrapSmuggledMarkdownBubble(b));
    }
  }

  while (cursor < protectedBlocks.length) {
    out.push(protectedBlocks[cursor].fullFence);
    cursor += 1;
  }
  return out;
}

export function formatRecentContext(messages, maxTurns = 6, maxCharsPerTurn = 360) {
  if (!Array.isArray(messages) || messages.length === 0) return '（无前序对话）';
  const kept = [];
  for (const msg of messages.slice(-maxTurns)) {
    if (!msg || typeof msg !== 'object') continue;
    const role = String(msg.role || 'user').trim().toLowerCase();
    if (['system', 'developer', 'tool', 'function'].includes(role)) continue;
    let txt = '';
    if (typeof msg.content === 'string') {
      txt = msg.content;
    } else if (Array.isArray(msg.content)) {
      txt = msg.content
        .filter((b) => b && typeof b === 'object' && ['text', 'input_text', 'output_text'].includes(b.type))
        .map((b) => String(b.text || ''))
        .join('\n');
    }
    txt = stripProtocolTokens(txt).trim();
    if (!txt) continue;
    if (txt.length > maxCharsPerTurn) txt = `${txt.slice(0, maxCharsPerTurn - 1)}…`;
    const speaker = role === 'assistant' || role === 'model' ? '小肥鱼' : '男友';
    kept.push(`${speaker}：${txt}`);
  }
  return kept.length ? kept.join('\n') : '（无前序对话）';
}

export function renderE4bTranslatorPrompt(userMessages, compactedDraft, weights, maxDraftChars = 2800) {
  const cue = buildTranslatorSliderCue(weights);
  const ctxStr = formatRecentContext(userMessages);
  let draftClean = stripProtocolTokens(compactedDraft).trim();
  if (draftClean.length > maxDraftChars) {
    draftClean = `${draftClean.slice(0, maxDraftChars - 1)}…`;
  }
  return (
    `<bos><|turn>system\n${TRANSLATOR_SYSTEM_BASE}${cue}${TRANSLATOR_TOOL_DECL}<turn|>\n` +
    `<|turn>user\n[Recent Context]\n${ctxStr}\n\n[Draft to Speak]\n${draftClean}<turn|>\n` +
    '<|turn>model\n<|channel>thought\n<channel|><|tool_call>call:send_message{bubbles:[<|"|>'
  );
}

export function parseRawBubblesFromCompletion(rawCompletion) {
  let cleaned = String(rawCompletion || '')
    .replace(/<\|channel>/g, '')
    .replace(/<channel\|>/g, '')
    .replace(/<\|turn>/g, '')
    .replace(/<turn\|>/g, '');

  for (const stopS of TRANSLATOR_STOPS) {
    if (cleaned.includes(stopS)) {
      cleaned = cleaned.split(stopS, 1)[0];
    }
  }

  const parts = `${QUOTE_STR}${cleaned}`.split(QUOTE_STR);
  const bubbles = [];
  for (let i = 1; i < parts.length; i += 2) {
    const item = parts[i].trim();
    if (item) bubbles.push(item);
  }
  if (!bubbles.length && cleaned.trim()) {
    const fallback = cleaned.replace(/\]\}/g, '').trim();
    if (fallback) bubbles.push(fallback);
  }
  return bubbles;
}
