import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import {
  ADVERTISED_MODELS,
  BubbleStreamFormatter,
  formatCompleteBubbleText,
  resolveUpstreamModelTag,
  type BubbleJoinMode,
  type SliderInputConfig,
} from './weights.ts';
import {
  buildToolRemapTable,
  remapMessagesToUpstream,
  restoreOpenAIToolCalls,
  type OpenAIMessage,
  type OpenAIToolCall,
} from './tool-remap.ts';
import {
  dispatchUpstreamChatCompletions,
  queryUpstreamHealth,
  type FetchLike,
  type ResolvedUpstreamConnection,
} from './upstream.ts';

export interface BridgeRuntimeStateProvider {
  readonly getSliderConfig: () => SliderInputConfig;
  readonly getUpstreamConnection: () => Promise<ResolvedUpstreamConnection>;
  readonly getBubbleJoinMode: () => BubbleJoinMode;
  readonly fetchImpl?: FetchLike;
}

export interface RunningBridgeServer {
  readonly server: Server;
  readonly host: string;
  readonly port: number;
  readonly baseUrl: string;
  readonly close: () => Promise<void>;
}

function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = Buffer.from(JSON.stringify(payload), 'utf8');
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': String(body.byteLength),
  });
  res.end(body);
}

async function readJsonBody(req: IncomingMessage, maxBytes = 8 * 1024 * 1024): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.byteLength;
    if (total > maxBytes) {
      const err = new Error('Request body exceeds 8 MiB limit');
      (err as Error & { status?: number }).status = 413;
      throw err;
    }
    chunks.push(buf);
  }
  const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('Request JSON body must be an object');
  }
  return parsed as Record<string, unknown>;
}

export async function startLocalOpenAiBridge(options: {
  readonly host?: string;
  readonly port: number;
  readonly state: BridgeRuntimeStateProvider;
}): Promise<RunningBridgeServer> {
  const host = options.host ?? '127.0.0.1';
  const { state } = options;
  const sockets = new Set<Socket>();

  const server = createServer(async (req, res) => {
    const urlPath = (req.url ?? '/').split('?', 1)[0] ?? '/';

    if (req.method === 'GET' && (urlPath === '/health' || urlPath === '/status')) {
      const sliderCfg = state.getSliderConfig();
      const conn = await state.getUpstreamConnection();
      const { modelTag, weights } = resolveUpstreamModelTag(undefined, sliderCfg, conn.mode);
      const health = await queryUpstreamHealth(conn, state.fetchImpl);
      sendJson(res, 200, { status: 'ok', activeModelTag: modelTag, weights, upstream: health });
      return;
    }

    if (req.method === 'GET' && urlPath === '/v1/models') {
      sendJson(res, 200, {
        object: 'list',
        data: ADVERTISED_MODELS.map((m) => ({
          id: m.id,
          object: 'model',
          created: 1763100000,
          owned_by: 'xiaofeiyu',
        })),
      });
      return;
    }

    if (req.method !== 'POST' || urlPath !== '/v1/chat/completions') {
      sendJson(res, 404, { error: { message: `Unsupported endpoint: ${urlPath}`, code: 'not_found' } });
      return;
    }

    let reqBody: Record<string, unknown>;
    try {
      reqBody = await readJsonBody(req);
    } catch (err) {
      sendJson(res, (err as { status?: number })?.status ?? 400, {
        error: { message: err instanceof Error ? err.message : String(err), code: 'bad_request' },
      });
      return;
    }

    const reqAbort = new AbortController();
    const onClose = () => {
      if (!res.writableEnded) reqAbort.abort(new Error('client_disconnected'));
    };
    res.on('close', onClose);

    try {
      const sliderCfg = state.getSliderConfig();
      const bubbleMode = state.getBubbleJoinMode();
      const conn = await state.getUpstreamConnection();
      const requestedModel = typeof reqBody.model === 'string' ? reqBody.model : undefined;
      const { modelTag } = resolveUpstreamModelTag(requestedModel, sliderCfg, conn.mode);

      const rawTools = Array.isArray(reqBody.tools) ? reqBody.tools : undefined;
      const remapTable = buildToolRemapTable(rawTools);
      const rawMessages = Array.isArray(reqBody.messages) ? (reqBody.messages as OpenAIMessage[]) : [];
      const mappedMessages = remapMessagesToUpstream(rawMessages, remapTable);

      const stream = Boolean(reqBody.stream);
      const upstreamPayload: Record<string, unknown> = {
        ...reqBody,
        model: modelTag,
        messages: mappedMessages,
        stream,
      };
      if (rawTools !== undefined) {
        upstreamPayload.tools = remapTable.tools;
      }

      const upstreamRes = await dispatchUpstreamChatCompletions({
        cfg: conn,
        payload: upstreamPayload,
        fetchImpl: state.fetchImpl,
        signal: reqAbort.signal,
      });

      if (!upstreamRes.ok) {
        const errText = await upstreamRes.text().catch(() => '');
        sendJson(res, upstreamRes.status, {
          error: { message: `Upstream returned HTTP ${upstreamRes.status}: ${errText.slice(0, 260)}` },
        });
        return;
      }

      const contentType = String(upstreamRes.headers.get('content-type') ?? '').toLowerCase();
      if (!stream || !contentType.includes('text/event-stream') || !upstreamRes.body) {
        const json = (await upstreamRes.json()) as Record<string, unknown>;
        const choices = Array.isArray(json.choices) ? json.choices : [];
        const rewrittenChoices = choices.map((c) => {
          if (typeof c !== 'object' || c === null) return c;
          const choice = c as Record<string, unknown>;
          const message =
            typeof choice.message === 'object' && choice.message !== null
              ? { ...(choice.message as Record<string, unknown>) }
              : undefined;
          if (message) {
            if (typeof message.content === 'string') {
              message.content = formatCompleteBubbleText(message.content, bubbleMode);
            }
            if (Array.isArray(message.tool_calls)) {
              message.tool_calls = restoreOpenAIToolCalls(message.tool_calls as OpenAIToolCall[], remapTable);
            }
          }
          return { ...choice, ...(message ? { message } : {}) };
        });
        sendJson(res, 200, { ...json, choices: rewrittenChoices });
        return;
      }

      res.writeHead(200, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });

      const reader = upstreamRes.body.getReader();
      const decoder = new TextDecoder();
      const bubbleFormatter = new BubbleStreamFormatter(bubbleMode);
      let sseBuffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split(/\r?\n/);
        sseBuffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          if (trimmed.startsWith(':')) {
            res.write(`${trimmed}\n\n`);
            continue;
          }
          if (!trimmed.startsWith('data:')) continue;
          const dataStr = trimmed.slice(5).trim();
          if (dataStr === '[DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }
          try {
            const chunkObj = JSON.parse(dataStr) as Record<string, unknown>;
            if (Array.isArray(chunkObj.choices)) {
              chunkObj.choices = chunkObj.choices.map((c) => {
                if (typeof c !== 'object' || c === null) return c;
                const choice = { ...(c as Record<string, unknown>) };
                if (typeof choice.delta === 'object' && choice.delta !== null) {
                  const delta = { ...(choice.delta as Record<string, unknown>) };
                  if (typeof delta.content === 'string' && delta.content.length > 0) {
                    delta.content = bubbleFormatter.push(delta.content);
                  }
                  if (Array.isArray(delta.tool_calls)) {
                    delta.tool_calls = restoreOpenAIToolCalls(delta.tool_calls as OpenAIToolCall[], remapTable);
                  }
                  choice.delta = delta;
                }
                return choice;
              });
            }
            res.write(`data: ${JSON.stringify(chunkObj)}\n\n`);
          } catch {
            res.write(`data: ${dataStr}\n\n`);
          }
        }
      }
      res.end();
    } catch (err) {
      if (!res.headersSent) {
        sendJson(res, 502, { error: { message: err instanceof Error ? err.message : String(err) } });
      } else if (!res.writableEnded) {
        res.end();
      }
    } finally {
      res.off('close', onClose);
    }
  });

  server.on('connection', (socket) => {
    sockets.add(socket);
    socket.once('close', () => sockets.delete(socket));
  });

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(options.port, host, () => {
      server.off('error', reject);
      resolve();
    });
  });

  const addr = server.address();
  const actualPort = typeof addr === 'object' && addr !== null ? addr.port : options.port;
  let closed = false;
  return {
    server,
    host,
    port: actualPort,
    baseUrl: `http://${host}:${actualPort}/v1`,
    close: async () => {
      if (closed) return;
      closed = true;
      for (const s of sockets) s.destroy();
      sockets.clear();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
