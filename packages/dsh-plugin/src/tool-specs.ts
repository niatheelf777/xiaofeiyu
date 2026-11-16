import type { OpenAIMessage } from './tool-remap.ts';

export const XIAOFEIYU_BUBBLES_TOOL_NAME = 'xiaofeiyu_send_bubbles';
export const XIAOFEIYU_CODE_TOOL_NAME = 'xiaofeiyu_send_code';

export const NATIVE_DELIVERY_TOOL_NAMES: ReadonlySet<string> = new Set([
  XIAOFEIYU_BUBBLES_TOOL_NAME,
  'send_bubbles',
  XIAOFEIYU_CODE_TOOL_NAME,
]);

const MIDTASK_PROMISE_RE = /(?:这就去|先跑|先看|我先|先帮你|我看看咯|等我先|先去查)/;
const TASK_DONE_RE = /(?:全绿|通过|修好|改好|搞定|完成|加好|写好|弄好)/;

export interface SendBubblesArgs {
  readonly bubbles: readonly string[];
  readonly flavor?: string;
  readonly weights?: readonly number[];
}

export interface SendBubblesOutput {
  readonly delivered: boolean;
  readonly bubbleCount: number;
  readonly bubbles: string[];
  readonly flavor?: string;
  readonly weights?: number[];
}

export interface SendCodeArgs {
  readonly language: string;
  readonly content: string;
  readonly title?: string;
}

export interface SendCodeOutput {
  readonly delivered: boolean;
  readonly language: string;
  readonly lineCount: number;
  readonly content: string;
  readonly title?: string;
}

export const XIAOFEIYU_SEND_BUBBLES_SPEC = {
  name: XIAOFEIYU_BUBBLES_TOOL_NAME,
  description:
    'Deliver Xiaofeiyu spoken IM chat bubbles to the conversation timeline. Each item in `bubbles` renders as a distinct short chat bubble without stage brackets or emoji clutter.',
  parameters: {
    bubbles: {
      type: 'array' as const,
      required: true,
      description: 'Ordered spoken IM bubbles (typically 1-4 concise conversational lines).',
      items: {
        type: 'string' as const,
      },
    },
    flavor: {
      type: 'string' as const,
      description: 'Optional active persona flavor tag (pure, neko, tease, brat, fragile, mix).',
    },
    weights: {
      type: 'array' as const,
      description: 'Optional 5-slider effective weight tuple [s0, c1, c2, c3, c4].',
      items: {
        type: 'number' as const,
      },
    },
  },
  output: {
    schema: {
      type: 'object' as const,
      additionalProperties: false,
      properties: {
        delivered: { type: 'boolean' as const, required: true },
        bubbleCount: { type: 'integer' as const, required: true },
        bubbles: {
          type: 'array' as const,
          required: true,
          items: { type: 'string' as const },
        },
        flavor: { type: 'string' as const },
        weights: {
          type: 'array' as const,
          items: { type: 'number' as const },
        },
      },
    },
    render: (_args: SendBubblesArgs, value: SendBubblesOutput) => [
      {
        type: 'text' as const,
        text: `Delivered ${value.bubbleCount} chat bubble(s).`,
      },
    ],
  },
  isConcurrencySafe: () => true,
  async execute(args: SendBubblesArgs): Promise<SendBubblesOutput> {
    const cleaned = (Array.isArray(args.bubbles) ? args.bubbles : [])
      .map((item) => String(item ?? '').trim())
      .filter((item) => item.length > 0);
    if (cleaned.length === 0) {
      throw new Error('xiaofeiyu_send_bubbles requires at least one non-empty bubble string');
    }
    const weights = Array.isArray(args.weights)
      ? args.weights.filter((n) => typeof n === 'number' && Number.isFinite(n)).slice(0, 5)
      : undefined;
    const flavor = typeof args.flavor === 'string' && args.flavor.trim() ? args.flavor.trim() : undefined;
    return {
      delivered: true,
      bubbleCount: cleaned.length,
      bubbles: cleaned,
      ...(flavor !== undefined ? { flavor } : {}),
      ...(weights && weights.length === 5 ? { weights } : {}),
    };
  },
};

export const XIAOFEIYU_SEND_CODE_SPEC = {
  name: XIAOFEIYU_CODE_TOOL_NAME,
  description:
    'Deliver a clean, copyable source code, command, or diff card into the conversation timeline.',
  parameters: {
    language: {
      type: 'string' as const,
      required: true,
      description: 'Language or format tag (for example python, typescript, js, bash, sql, diff).',
    },
    content: {
      type: 'string' as const,
      required: true,
      description: 'Raw code or document content without outer markdown fences.',
    },
    title: {
      type: 'string' as const,
      description: 'Optional short filename or card caption.',
    },
  },
  output: {
    schema: {
      type: 'object' as const,
      additionalProperties: false,
      properties: {
        delivered: { type: 'boolean' as const, required: true },
        language: { type: 'string' as const, required: true },
        lineCount: { type: 'integer' as const, required: true },
        content: { type: 'string' as const, required: true },
        title: { type: 'string' as const },
      },
    },
    render: (_args: SendCodeArgs, value: SendCodeOutput) => [
      {
        type: 'text' as const,
        text: `Delivered ${value.language} code card (${value.lineCount} line(s)).`,
      },
    ],
  },
  isConcurrencySafe: () => true,
  async execute(args: SendCodeArgs): Promise<SendCodeOutput> {
    const language = String(args.language || 'text').trim() || 'text';
    const content = String(args.content ?? '').replace(/\r\n/g, '\n');
    const lineCount = content.length === 0 ? 0 : content.split('\n').length;
    const title = typeof args.title === 'string' && args.title.trim() ? args.title.trim() : undefined;
    return {
      delivered: true,
      language,
      lineCount,
      content,
      ...(title !== undefined ? { title } : {}),
    };
  },
};

export function hasNativeBubbleToolInTools(rawTools: readonly unknown[] | undefined): boolean {
  if (!Array.isArray(rawTools) || rawTools.length === 0) return false;
  for (const item of rawTools) {
    if (typeof item !== 'object' || item === null) continue;
    const rec = item as Record<string, unknown>;
    const fn =
      typeof rec.function === 'object' && rec.function !== null
        ? (rec.function as Record<string, unknown>)
        : rec;
    const name = typeof fn.name === 'string' ? fn.name.trim() : '';
    if (name === XIAOFEIYU_BUBBLES_TOOL_NAME || name === 'send_bubbles') {
      return true;
    }
  }
  return false;
}

function countNonNativeExternalTools(rawTools: readonly unknown[] | undefined): number {
  if (!Array.isArray(rawTools)) return 0;
  let count = 0;
  for (const item of rawTools) {
    if (typeof item !== 'object' || item === null) continue;
    const rec = item as Record<string, unknown>;
    const fn =
      typeof rec.function === 'object' && rec.function !== null
        ? (rec.function as Record<string, unknown>)
        : rec;
    const name = typeof fn.name === 'string' ? fn.name.trim() : '';
    if (name && !NATIVE_DELIVERY_TOOL_NAMES.has(name)) {
      count += 1;
    }
  }
  return count;
}

function extractMessageText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (typeof part === 'object' && part !== null && 'text' in part) {
          return String((part as { text?: unknown }).text ?? '');
        }
        return '';
      })
      .join('\n');
  }
  return '';
}

/**
 * Detect whether the latest DSH agent-loop step ONLY executed `xiaofeiyu_send_bubbles` / `xiaofeiyu_send_code`
 * and settled without error, allowing the loopback bridge to close the turn in <1ms with zero GPU calls.
 */
export function isSettledNativeDeliveryTurn(
  messages: readonly OpenAIMessage[],
  rawTools: readonly unknown[] | undefined,
): boolean {
  if (!hasNativeBubbleToolInTools(rawTools) || messages.length === 0) return false;

  let idx = messages.length - 1;
  const toolMessages: OpenAIMessage[] = [];
  while (idx >= 0 && messages[idx]?.role === 'tool') {
    toolMessages.push(messages[idx]!);
    idx -= 1;
  }
  if (toolMessages.length === 0 || idx < 0) return false;

  const parent = messages[idx];
  if (!parent || (parent.role !== 'assistant' && parent.role !== 'model')) return false;
  if (!Array.isArray(parent.tool_calls) || parent.tool_calls.length === 0) return false;

  const expectedIds = new Set<string>();
  const spokenBubbles: string[] = [];

  for (const tc of parent.tool_calls) {
    const fnName = tc.function?.name?.trim() ?? '';
    if (!NATIVE_DELIVERY_TOOL_NAMES.has(fnName)) return false;
    if (tc.id && tc.id.trim()) expectedIds.add(tc.id.trim());
    if (fnName === XIAOFEIYU_BUBBLES_TOOL_NAME || fnName === 'send_bubbles') {
      try {
        const parsed = JSON.parse(tc.function?.arguments ?? '{}') as { bubbles?: unknown };
        if (Array.isArray(parsed.bubbles)) {
          for (const b of parsed.bubbles) {
            if (typeof b === 'string' && b.trim()) spokenBubbles.push(b.trim());
          }
        }
      } catch {
        // ignore malformed args
      }
    }
  }

  const settledIds = new Set<string>();
  for (const tm of toolMessages) {
    if (typeof tm.tool_call_id === 'string' && tm.tool_call_id.trim()) {
      settledIds.add(tm.tool_call_id.trim());
    }
    const text = extractMessageText(tm.content).trimStart();
    if (text.startsWith('Error:') || text.startsWith('{"error"')) {
      return false;
    }
  }

  for (const reqId of expectedIds) {
    if (!settledIds.has(reqId)) return false;
  }

  if (countNonNativeExternalTools(rawTools) > 0 && spokenBubbles.length > 0) {
    const joined = spokenBubbles.join('');
    if (MIDTASK_PROMISE_RE.test(joined) && !TASK_DONE_RE.test(joined)) {
      return false;
    }
  }

  return true;
}
