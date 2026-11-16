export interface OpenAIFunctionTool {
  readonly type: 'function';
  readonly function: {
    readonly name: string;
    readonly description?: string;
    readonly parameters?: Record<string, unknown>;
  };
}

export interface OpenAIToolCall {
  readonly id?: string;
  readonly index?: number;
  readonly type?: string;
  readonly function?: {
    readonly name?: string;
    readonly arguments?: string;
  };
}

export interface OpenAIMessage {
  readonly role: string;
  readonly content?: unknown;
  readonly name?: string;
  readonly tool_call_id?: string;
  readonly tool_calls?: readonly OpenAIToolCall[];
  readonly reasoning_content?: string;
}

const RESERVED_PC2_ALIASES: Readonly<Record<string, string>> = {
  send_message: 'agent_send_message',
  send_code_or_doc: 'ext_send_code_or_doc',
  calc_python: 'ext_calc_python',
  run_python: 'ext_run_python',
};

const DEFAULT_REVERSE_ALIASES: Readonly<Record<string, string>> = {
  agent_send_message: 'send_message',
  ext_send_code_or_doc: 'send_code_or_doc',
  ext_calc_python: 'calc_python',
  ext_run_python: 'run_python',
};

export interface ToolRemapTable {
  readonly toUpstream: ReadonlyMap<string, string>;
  readonly fromUpstream: ReadonlyMap<string, string>;
  readonly tools: readonly OpenAIFunctionTool[];
}

function allocateUpstreamName(original: string, taken: ReadonlySet<string>): string {
  const preferred = RESERVED_PC2_ALIASES[original];
  if (!preferred) return original;
  let candidate = preferred;
  let suffix = 2;
  while (taken.has(candidate) || candidate in RESERVED_PC2_ALIASES) {
    candidate = `${preferred}_${suffix}`;
    suffix += 1;
  }
  return candidate;
}

export function buildToolRemapTable(rawTools: readonly unknown[] | undefined): ToolRemapTable {
  const toUpstream = new Map<string, string>();
  const fromUpstream = new Map<string, string>();

  if (!rawTools || rawTools.length === 0) {
    for (const [orig, alias] of Object.entries(RESERVED_PC2_ALIASES)) {
      toUpstream.set(orig, alias);
      fromUpstream.set(alias, orig);
    }
    return { toUpstream, fromUpstream, tools: [] };
  }

  const existingNames = new Set<string>();
  for (const item of rawTools) {
    if (typeof item !== 'object' || item === null) continue;
    const rec = item as Record<string, unknown>;
    const fn =
      typeof rec.function === 'object' && rec.function !== null ? (rec.function as Record<string, unknown>) : rec;
    if (typeof fn.name === 'string' && fn.name.trim()) {
      existingNames.add(fn.name.trim());
    }
  }

  const taken = new Set<string>(existingNames);
  const rewritten: OpenAIFunctionTool[] = [];

  for (const item of rawTools) {
    if (typeof item !== 'object' || item === null) continue;
    const rec = item as Record<string, unknown>;
    const fn =
      typeof rec.function === 'object' && rec.function !== null ? (rec.function as Record<string, unknown>) : rec;
    const origName = typeof fn.name === 'string' ? fn.name.trim() : '';
    if (!origName) continue;

    let upstreamName = origName;
    if (origName in RESERVED_PC2_ALIASES) {
      upstreamName = allocateUpstreamName(origName, taken);
      taken.add(upstreamName);
      toUpstream.set(origName, upstreamName);
      fromUpstream.set(upstreamName, origName);
    }

    rewritten.push({
      type: 'function',
      function: {
        name: upstreamName,
        ...(typeof fn.description === 'string' ? { description: fn.description } : {}),
        parameters:
          typeof fn.parameters === 'object' && fn.parameters !== null
            ? (fn.parameters as Record<string, unknown>)
            : { type: 'object', properties: {} },
      },
    });
  }

  return { toUpstream, fromUpstream, tools: rewritten };
}

export function mapToolNameToUpstream(name: string, table: ToolRemapTable): string {
  return table.toUpstream.get(name) ?? RESERVED_PC2_ALIASES[name] ?? name;
}

export function restoreToolNameFromUpstream(name: string, table?: ToolRemapTable): string {
  if (table?.fromUpstream.has(name)) {
    return table.fromUpstream.get(name)!;
  }
  return DEFAULT_REVERSE_ALIASES[name] ?? name;
}

export function remapMessagesToUpstream(
  messages: readonly OpenAIMessage[],
  table: ToolRemapTable,
): OpenAIMessage[] {
  return messages.map((msg) => {
    let next: OpenAIMessage = msg;

    if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      let changed = false;
      const mappedCalls = msg.tool_calls.map((tc) => {
        const orig = tc.function?.name;
        if (!orig) return tc;
        const mapped = mapToolNameToUpstream(orig, table);
        if (mapped === orig) return tc;
        changed = true;
        return { ...tc, function: { ...tc.function, name: mapped } };
      });
      if (changed) next = { ...next, tool_calls: mappedCalls };
    }

    if (msg.role === 'tool' && typeof msg.name === 'string') {
      const mapped = mapToolNameToUpstream(msg.name, table);
      if (mapped !== msg.name) next = { ...next, name: mapped };
    }

    return next;
  });
}

export function restoreOpenAIToolCalls(
  toolCalls: readonly OpenAIToolCall[] | undefined,
  table?: ToolRemapTable,
): OpenAIToolCall[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined;
  return toolCalls.map((tc) => {
    const rawName = tc.function?.name;
    if (!rawName) return tc;
    const restored = restoreToolNameFromUpstream(rawName, table);
    if (restored === rawName) return tc;
    return { ...tc, function: { ...tc.function, name: restored } };
  });
}
