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

const RESTRICTED_PROXY_PARAM_ALIASES: Readonly<Record<string, string>> = {
  file: 'target_path',
  files: 'target_paths',
  attachment: 'attachment_ref',
  attachments: 'attachment_refs',
  file_data: 'text_payload',
  file_id: 'target_ref_id',
  file_url: 'target_uri',
  image: 'media_ref',
  image_url: 'media_uri',
  input_image: 'input_media_ref',
  input_file: 'input_path_ref',
  document: 'doc_text',
  audio: 'audio_ref',
  input_audio: 'input_audio_ref',
  video: 'video_ref',
  b64_json: 'encoded_text',
  base64: 'encoded_payload',
  bytes: 'byte_count',
  blob: 'raw_chunk',
  binary: 'is_binary_flag',
  data_url: 'inline_uri',
  download_url: 'source_uri',
  upload_url: 'dest_uri',
};

export interface ParamKeyRemap {
  readonly toUpstream: ReadonlyMap<string, string>;
  readonly fromUpstream: ReadonlyMap<string, string>;
}

export interface ToolRemapTable {
  readonly toUpstream: ReadonlyMap<string, string>;
  readonly fromUpstream: ReadonlyMap<string, string>;
  readonly paramRemapsByUpstreamTool: ReadonlyMap<string, ParamKeyRemap>;
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

function normalizeParamKeyLookup(key: string): string {
  return key.toLowerCase().replace(/-/g, '_');
}

function allocateSafeParamKey(originalKey: string, siblingKeys: ReadonlySet<string>): string {
  const norm = normalizeParamKeyLookup(originalKey);
  const base = RESTRICTED_PROXY_PARAM_ALIASES[norm] ?? `arg_${norm}`;
  let candidate = base;
  let suffix = 2;
  while (
    siblingKeys.has(candidate) ||
    normalizeParamKeyLookup(candidate) in RESTRICTED_PROXY_PARAM_ALIASES
  ) {
    candidate = `${base}_${suffix}`;
    suffix += 1;
  }
  return candidate;
}

function rewriteSchemaObject(
  schemaNode: Record<string, unknown>,
  toUpstream: Map<string, string>,
  fromUpstream: Map<string, string>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(schemaNode)) {
    if (k === 'properties' && typeof v === 'object' && v !== null && !Array.isArray(v)) {
      const props = v as Record<string, unknown>;
      const siblingSet = new Set<string>(Object.keys(props));
      const nextProps: Record<string, unknown> = {};
      for (const [propName, propVal] of Object.entries(props)) {
        let mappedProp = toUpstream.get(propName) ?? propName;
        if (normalizeParamKeyLookup(propName) in RESTRICTED_PROXY_PARAM_ALIASES && !toUpstream.has(propName)) {
          mappedProp = allocateSafeParamKey(propName, siblingSet);
          siblingSet.add(mappedProp);
          toUpstream.set(propName, mappedProp);
          fromUpstream.set(mappedProp, propName);
        }
        nextProps[mappedProp] =
          typeof propVal === 'object' && propVal !== null && !Array.isArray(propVal)
            ? rewriteSchemaObject(propVal as Record<string, unknown>, toUpstream, fromUpstream)
            : propVal;
      }
      out.properties = nextProps;
    } else if (k === 'required' && Array.isArray(v)) {
      out.required = v.map((r) => (typeof r === 'string' ? (toUpstream.get(r) ?? r) : r));
    } else if (k === 'items' && typeof v === 'object' && v !== null && !Array.isArray(v)) {
      out.items = rewriteSchemaObject(v as Record<string, unknown>, toUpstream, fromUpstream);
    } else if ((k === 'anyOf' || k === 'oneOf' || k === 'allOf') && Array.isArray(v)) {
      out[k] = v.map((branch) =>
        typeof branch === 'object' && branch !== null && !Array.isArray(branch)
          ? rewriteSchemaObject(branch as Record<string, unknown>, toUpstream, fromUpstream)
          : branch,
      );
    } else if (normalizeParamKeyLookup(k) in RESTRICTED_PROXY_PARAM_ALIASES) {
      const safeMetaKey = RESTRICTED_PROXY_PARAM_ALIASES[normalizeParamKeyLookup(k)] ?? `meta_${k}`;
      out[safeMetaKey] = v;
    } else {
      out[k] = v;
    }
  }
  return out;
}

function rewriteArgsJsonKeys(rawArgs: string | undefined, keyMap: ReadonlyMap<string, string>): string | undefined {
  if (!rawArgs || keyMap.size === 0) return rawArgs;
  try {
    const parsed: unknown = JSON.parse(rawArgs);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return rawArgs;
    const transformObj = (obj: Record<string, unknown>): Record<string, unknown> => {
      const next: Record<string, unknown> = {};
      for (const [k, val] of Object.entries(obj)) {
        const mappedKey = keyMap.get(k) ?? k;
        if (Array.isArray(val)) {
          next[mappedKey] = val.map((item) =>
            typeof item === 'object' && item !== null && !Array.isArray(item)
              ? transformObj(item as Record<string, unknown>)
              : item,
          );
        } else if (typeof val === 'object' && val !== null) {
          next[mappedKey] = transformObj(val as Record<string, unknown>);
        } else {
          next[mappedKey] = val;
        }
      }
      return next;
    };
    return JSON.stringify(transformObj(parsed as Record<string, unknown>));
  } catch {
    return rawArgs;
  }
}

export function buildToolRemapTable(rawTools: readonly unknown[] | undefined): ToolRemapTable {
  const toUpstream = new Map<string, string>();
  const fromUpstream = new Map<string, string>();
  const paramRemapsByUpstreamTool = new Map<string, ParamKeyRemap>();

  if (!rawTools || rawTools.length === 0) {
    for (const [orig, alias] of Object.entries(RESERVED_PC2_ALIASES)) {
      toUpstream.set(orig, alias);
      fromUpstream.set(alias, orig);
    }
    return { toUpstream, fromUpstream, paramRemapsByUpstreamTool, tools: [] };
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

    const rawParams =
      typeof fn.parameters === 'object' && fn.parameters !== null && !Array.isArray(fn.parameters)
        ? (fn.parameters as Record<string, unknown>)
        : { type: 'object', properties: {} };

    const pToUpstream = new Map<string, string>();
    const pFromUpstream = new Map<string, string>();
    const sanitizedParams = rewriteSchemaObject(rawParams, pToUpstream, pFromUpstream);
    if (pToUpstream.size > 0) {
      paramRemapsByUpstreamTool.set(upstreamName, {
        toUpstream: pToUpstream,
        fromUpstream: pFromUpstream,
      });
    }

    rewritten.push({
      type: 'function',
      function: {
        name: upstreamName,
        ...(typeof fn.description === 'string' ? { description: fn.description } : {}),
        parameters: sanitizedParams,
      },
    });
  }

  return { toUpstream, fromUpstream, paramRemapsByUpstreamTool, tools: rewritten };
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
        const paramMap = table.paramRemapsByUpstreamTool.get(mapped);
        const mappedArgs = paramMap ? rewriteArgsJsonKeys(tc.function?.arguments, paramMap.toUpstream) : tc.function?.arguments;
        if (mapped === orig && mappedArgs === tc.function?.arguments) return tc;
        changed = true;
        return {
          ...tc,
          function: {
            ...tc.function,
            name: mapped,
            ...(mappedArgs !== undefined ? { arguments: mappedArgs } : {}),
          },
        };
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

function sanitizeRestoredDshArgs(rawArgs: string | undefined): string | undefined {
  if (!rawArgs) return rawArgs;
  try {
    const parsed: unknown = JSON.parse(rawArgs);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return rawArgs;
    const rec = { ...(parsed as Record<string, unknown>) };
    let mutated = false;
    const sp = rec.sandbox_permissions;
    const hasRealEscalation = sp === 'danger-full-access' || sp === 'require_escalated';
    if (!hasRealEscalation) {
      if ('justification' in rec) {
        delete rec.justification;
        mutated = true;
      }
      if (sp === 'workspace-write' || sp === 'read-only' || sp === '') {
        delete rec.sandbox_permissions;
        mutated = true;
      }
    }
    return mutated ? JSON.stringify(rec) : rawArgs;
  } catch {
    return rawArgs;
  }
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
    const paramMap = table?.paramRemapsByUpstreamTool.get(rawName);
    const keyRestoredArgs = paramMap ? rewriteArgsJsonKeys(tc.function?.arguments, paramMap.fromUpstream) : tc.function?.arguments;
    const restoredArgs = sanitizeRestoredDshArgs(keyRestoredArgs);
    if (restored === rawName && restoredArgs === tc.function?.arguments) return tc;
    return {
      ...tc,
      function: {
        ...tc.function,
        name: restored,
        ...(restoredArgs !== undefined ? { arguments: restoredArgs } : {}),
      },
    };
  });
}
