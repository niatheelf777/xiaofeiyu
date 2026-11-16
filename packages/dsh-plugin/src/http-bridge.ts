import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { Socket } from 'node:net';
import {
  ADVERTISED_MODELS,
  BubbleStreamFormatter,
  SLIDER_PRESETS,
  clampSlider,
  computeEffectiveWeights,
  formatCompleteBubbleText,
  resolveRawSlidersFromConfig,
  resolveUpstreamModelTag,
  type BubbleJoinMode,
  type SliderInputConfig,
  type SliderPresetId,
} from './weights.ts';
import { isSettledNativeDeliveryTurn } from './tool-specs.ts';
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
  readonly setSliderConfig?: (next: SliderInputConfig, overrideModelFlavor: boolean) => void;
  readonly getOverrideModelFlavor?: () => boolean;
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

function sendJson(res: ServerResponse, status: number, payload: unknown, extraHeaders: Record<string, string> = {}): void {
  const body = Buffer.from(JSON.stringify(payload), 'utf8');
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET, POST, OPTIONS',
    'access-control-allow-headers': 'Content-Type, Authorization',
    'content-length': String(body.byteLength),
    ...extraHeaders,
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
  let lastRequestedModel: string | undefined;

  const server = createServer(async (req, res) => {
    const urlPath = (req.url ?? '/').split('?', 1)[0] ?? '/';

    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET, POST, OPTIONS',
        'access-control-allow-headers': 'Content-Type, Authorization',
        'content-length': '0',
      });
      res.end();
      return;
    }

    if (urlPath === '/sliders' && (req.method === 'GET' || req.method === 'POST')) {
      if (req.method === 'POST') {
        let patchBody: Record<string, unknown>;
        try {
          patchBody = await readJsonBody(req, 64 * 1024);
        } catch (err) {
          sendJson(res, 400, { error: { message: err instanceof Error ? err.message : String(err) } });
          return;
        }
        const cur = state.getSliderConfig();
        const rawPreset = typeof patchBody.preset === 'string' ? patchBody.preset.trim() : cur.preset ?? 'custom';
        const validPreset: SliderPresetId =
          rawPreset === 'custom' || rawPreset in SLIDER_PRESETS
            ? (rawPreset as SliderPresetId)
            : 'custom';
        const baseVals =
          validPreset !== 'custom' ? SLIDER_PRESETS[validPreset].values : resolveRawSlidersFromConfig(cur);
        const nextCfg: SliderInputConfig = {
          preset: validPreset,
          xiaofeiyu: patchBody.xiaofeiyu !== undefined ? clampSlider(patchBody.xiaofeiyu, 1.0) : baseVals[0],
          neko_clingy: patchBody.neko_clingy !== undefined ? clampSlider(patchBody.neko_clingy, 0.0) : baseVals[1],
          older_tease: patchBody.older_tease !== undefined ? clampSlider(patchBody.older_tease, 0.0) : baseVals[2],
          brat_mesugaki: patchBody.brat_mesugaki !== undefined ? clampSlider(patchBody.brat_mesugaki, 0.0) : baseVals[3],
          fragile_porcelain:
            patchBody.fragile_porcelain !== undefined ? clampSlider(patchBody.fragile_porcelain, 0.0) : baseVals[4],
        };
        const overrideFlag = patchBody.overrideModelFlavor !== undefined ? Boolean(patchBody.overrideModelFlavor) : true;
        state.setSliderConfig?.(nextCfg, overrideFlag);
      }

      const sliderCfg = state.getSliderConfig();
      const conn = await state.getUpstreamConnection();
      const overrideModelFlavor = state.getOverrideModelFlavor?.() ?? false;
      const resolved = resolveUpstreamModelTag(lastRequestedModel, sliderCfg, conn.mode, { overrideModelFlavor });
      const effective = computeEffectiveWeights(resolved.weights);
      sendJson(res, 200, {
        ok: true,
        sliderConfig: sliderCfg,
        overrideModelFlavor,
        effective,
        activeModelTag: resolved.modelTag,
      });
      return;
    }

    if (req.method === 'GET' && (urlPath === '/health' || urlPath === '/status')) {
      const sliderCfg = state.getSliderConfig();
      const conn = await state.getUpstreamConnection();
      const overrideModelFlavor = state.getOverrideModelFlavor?.() ?? false;
      const { modelTag, weights } = resolveUpstreamModelTag(undefined, sliderCfg, conn.mode, { overrideModelFlavor });
      const health = await queryUpstreamHealth(conn, state.fetchImpl);
      sendJson(res, 200, { status: 'ok', activeModelTag: modelTag, weights, overrideModelFlavor, upstream: health });
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
      if (requestedModel) {
        lastRequestedModel = requestedModel;
      }
      const overrideModelFlavor = state.getOverrideModelFlavor?.() ?? false;
      const { modelTag } = resolveUpstreamModelTag(requestedModel, sliderCfg, conn.mode, { overrideModelFlavor });

      const rawTools = Array.isArray(reqBody.tools) ? reqBody.tools : undefined;
      const rawMessages = Array.isArray(reqBody.messages) ? (reqBody.messages as OpenAIMessage[]) : [];
      const stream = Boolean(reqBody.stream);

      if (isSettledNativeDeliveryTurn(rawMessages, rawTools)) {
        const settledId = `chatcmpl_settled_${Date.now().toString(36)}`;
        const created = Math.floor(Date.now() / 1000);
        const usage = { prompt_tokens: 0, completion_tokens: 1, total_tokens: 1 };
        const settlementNote = '已通过微信气泡送达回复。';
        if (!stream) {
          sendJson(res, 200, {
            id: settledId,
            object: 'chat.completion',
            created,
            model: modelTag,
            choices: [
              {
                index: 0,
                message: { role: 'assistant', content: null, reasoning_content: settlementNote },
                finish_reason: 'stop',
              },
            ],
            usage,
          });
          return;
        }
        res.writeHead(200, {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
        });
        const openChunk = {
          id: settledId,
          object: 'chat.completion.chunk',
          created,
          model: modelTag,
          choices: [
            {
              index: 0,
              delta: { role: 'assistant', reasoning_content: settlementNote },
              finish_reason: null,
            },
          ],
        };
        const stopChunk = {
          id: settledId,
          object: 'chat.completion.chunk',
          created,
          model: modelTag,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          usage,
        };
        res.write(`data: ${JSON.stringify(openChunk)}\n\n`);
        res.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }

      const remapTable = buildToolRemapTable(rawTools);
      const mappedMessages = remapMessagesToUpstream(rawMessages, remapTable);
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
