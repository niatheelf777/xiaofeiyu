export type XiaofeiyuRuntimeMode = 'unified' | 'translator';

export interface ResolvedUpstreamConnection {
  readonly mode: XiaofeiyuRuntimeMode;
  readonly upstreamBaseUrl: string;
  readonly upstreamApiKey: string;
  readonly translatorProxyUrl: string;
}

export type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>;

function trimUrl(url: string | undefined): string {
  return String(url ?? '').trim().replace(/\/+$/, '');
}

export function resolveCompletionsUrl(baseUrl: string): string {
  const clean = trimUrl(baseUrl);
  if (!clean) return '';
  if (clean.endsWith('/chat/completions')) return clean;
  if (clean.endsWith('/v1')) return `${clean}/chat/completions`;
  return `${clean}/v1/chat/completions`;
}

export function resolveActiveEndpointUrl(cfg: ResolvedUpstreamConnection): string {
  if (cfg.mode === 'translator' && cfg.translatorProxyUrl) {
    return resolveCompletionsUrl(cfg.translatorProxyUrl);
  }
  return resolveCompletionsUrl(cfg.upstreamBaseUrl);
}

export async function queryUpstreamHealth(
  cfg: ResolvedUpstreamConnection,
  fetchImpl: FetchLike = fetch,
): Promise<{ readonly configured: boolean; readonly ok: boolean; readonly mode: XiaofeiyuRuntimeMode; readonly status: number }> {
  const base = cfg.mode === 'translator' && cfg.translatorProxyUrl ? cfg.translatorProxyUrl : cfg.upstreamBaseUrl;
  const clean = trimUrl(base);
  if (!clean) {
    return { configured: false, ok: false, mode: cfg.mode, status: 0 };
  }
  const modelsUrl = clean.endsWith('/v1') ? `${clean}/models` : `${clean}/v1/models`;
  try {
    const headers: Record<string, string> = {};
    if (cfg.upstreamApiKey) headers.authorization = `Bearer ${cfg.upstreamApiKey}`;
    const res = await fetchImpl(modelsUrl, { method: 'GET', headers });
    return { configured: true, ok: res.ok, mode: cfg.mode, status: res.status };
  } catch {
    return { configured: true, ok: false, mode: cfg.mode, status: 0 };
  }
}

export async function dispatchUpstreamChatCompletions(options: {
  readonly cfg: ResolvedUpstreamConnection;
  readonly payload: Record<string, unknown>;
  readonly fetchImpl?: FetchLike;
  readonly signal?: AbortSignal;
}): Promise<Response> {
  const { cfg, payload, fetchImpl = fetch, signal } = options;
  const endpointUrl = resolveActiveEndpointUrl(cfg);
  if (!endpointUrl) {
    const err = new Error(
      'Xiaofeiyu upstream endpoint is unconfigured: set upstreamBaseUrl (or translatorProxyUrl in translator mode).',
    );
    (err as Error & { code?: string }).code = 'UPSTREAM_UNCONFIGURED';
    throw err;
  }

  const headers: Record<string, string> = {
    'content-type': 'application/json',
    accept: payload.stream ? 'text/event-stream, application/json' : 'application/json',
  };
  if (cfg.upstreamApiKey) headers.authorization = `Bearer ${cfg.upstreamApiKey}`;

  return fetchImpl(endpointUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
    signal,
  });
}
