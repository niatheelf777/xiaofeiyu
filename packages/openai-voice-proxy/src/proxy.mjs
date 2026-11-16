import { createServer } from 'node:http';
import { randomBytes } from 'node:crypto';
import {
  ADVERTISED_MODELS,
  TRANSLATOR_STOPS,
  extractAndCompactCodeBlocks,
  parseRawBubblesFromCompletion,
  parseTranslatorWeights,
  renderE4bTranslatorPrompt,
  spliceVerbatimCodeBlocks,
} from './translator-core.mjs';

function sendJson(res, status, payload) {
  const buf = Buffer.from(JSON.stringify(payload), 'utf8');
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': String(buf.byteLength),
  });
  res.end(buf);
}

async function readJsonBody(req, maxBytes = 8 * 1024 * 1024) {
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.byteLength;
    if (total > maxBytes) {
      throw Object.assign(new Error('Request body exceeds 8 MiB limit'), { status: 413 });
    }
    chunks.push(buf);
  }
  const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw Object.assign(new Error('JSON body must be an object'), { status: 400 });
  }
  return parsed;
}

function normalizeCompletionsUrl(baseUrl) {
  const clean = String(baseUrl || '').trim().replace(/\/+$/, '');
  if (!clean) return '';
  if (clean.endsWith('/chat/completions')) return clean;
  if (clean.endsWith('/v1')) return `${clean}/chat/completions`;
  return `${clean}/v1/chat/completions`;
}

async function callUpstreamLlm({ upstreamUrl, upstreamApiKey, upstreamModel, reqBody, fetchImpl, signal }) {
  const headers = { 'content-type': 'application/json', accept: 'application/json' };
  if (upstreamApiKey) headers.authorization = `Bearer ${upstreamApiKey}`;

  const upstreamPayload = {
    ...reqBody,
    model: upstreamModel || reqBody.upstream_model || 'deepseek-chat',
    stream: false,
  };
  delete upstreamPayload.sliders;
  delete upstreamPayload.bubble_join;

  const res = await fetchImpl(upstreamUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamPayload),
    signal,
  });

  const text = await res.text();
  if (!res.ok) {
    throw Object.assign(new Error(`Upstream LLM HTTP ${res.status}: ${text.slice(0, 300)}`), {
      status: res.status >= 500 ? 502 : res.status,
    });
  }
  return JSON.parse(text);
}

async function translateDraftWithE4B({
  translatorUrl,
  translatorMode,
  userMessages,
  rawDraft,
  modelTag,
  weights,
  joinSep,
  fetchImpl,
  signal,
}) {
  const { compactedDraft, protectedBlocks } = extractAndCompactCodeBlocks(rawDraft);

  if (translatorMode === 'passthrough') {
    const fallbackBubbles = spliceVerbatimCodeBlocks([compactedDraft], protectedBlocks);
    return {
      bubbles: fallbackBubbles,
      content: fallbackBubbles.join(joinSep),
      telemetry: { mode: 'passthrough', protected_code_blocks: protectedBlocks.length },
    };
  }

  if (translatorMode === 'e4b-http') {
    const targetUrl = String(translatorUrl || '').replace(/\/+$/, '') + '/v1/translate';
    const res = await fetchImpl(targetUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        user_messages: userMessages,
        raw_draft: rawDraft,
        model: modelTag,
        sliders: weights,
        bubble_join: joinSep === '\n' ? 'newline' : 'paragraph',
      }),
      signal,
    });
    if (!res.ok) {
      throw new Error(`E4B translator HTTP ${res.status}: ${(await res.text()).slice(0, 240)}`);
    }
    const data = await res.json();
    const bubbles = spliceVerbatimCodeBlocks(Array.isArray(data.bubbles) ? data.bubbles : [], protectedBlocks);
    return {
      bubbles,
      content: bubbles.join(joinSep),
      telemetry: data.telemetry || { mode: 'e4b-http' },
    };
  }

  // Default: direct `llama-server /completion` call with zero-thought prefill & 5-slider LoRA
  const completionUrl = String(translatorUrl || '').replace(/\/+$/, '') + '/completion';
  const prompt = renderE4bTranslatorPrompt(userMessages, compactedDraft, weights);
  const lora = weights.map((w, idx) => ({ id: idx, scale: Math.abs(w) > 1e-6 ? Number(w.toFixed(4)) : 0.0 }));

  const res = await fetchImpl(completionUrl, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      prompt,
      n_predict: 256,
      temperature: 0.45,
      top_p: 0.9,
      cache_prompt: true,
      stop: TRANSLATOR_STOPS,
      lora,
      stream: false,
    }),
    signal,
  });
  if (!res.ok) {
    throw new Error(`llama-server /completion HTTP ${res.status}: ${(await res.text()).slice(0, 240)}`);
  }
  const data = await res.json();
  const rawBubbles = parseRawBubblesFromCompletion(data.content || '');
  const finalBubbles = spliceVerbatimCodeBlocks(rawBubbles, protectedBlocks);
  return {
    bubbles: finalBubbles,
    content: finalBubbles.join(joinSep),
    telemetry: {
      mode: 'llama-completion',
      bubble_count: finalBubbles.length,
      protected_code_blocks: protectedBlocks.length,
    },
  };
}

export async function startVoiceProxyServer(options = {}) {
  const host = options.host || '127.0.0.1';
  const port = Number(options.port ?? 18088);
  const upstreamUrl = normalizeCompletionsUrl(options.upstreamBaseUrl || '');
  const upstreamApiKey = String(options.upstreamApiKey || '');
  const upstreamModel = String(options.upstreamModel || 'deepseek-chat');
  const translatorUrl = String(options.translatorUrl || 'http://127.0.0.1:18085').replace(/\/+$/, '');
  const translatorMode = options.translatorMode || 'llama-completion';
  const defaultJoinSep = options.bubbleJoin === 'newline' ? '\n' : '\n\n';
  const fetchImpl = options.fetchImpl || fetch;

  if (!upstreamUrl) {
    throw new Error('upstreamBaseUrl is required (e.g. https://api.deepseek.com/v1)');
  }

  const sockets = new Set();

  const server = createServer(async (req, res) => {
    const urlPath = (req.url || '/').split('?', 1)[0];

    if (req.method === 'GET' && (urlPath === '/health' || urlPath === '/status')) {
      sendJson(res, 200, {
        status: 'ok',
        service: 'xiaofeiyu-openai-voice-proxy',
        translatorMode,
        upstreamConfigured: Boolean(upstreamUrl),
      });
      return;
    }

    if (req.method === 'GET' && urlPath === '/v1/models') {
      sendJson(res, 200, {
        object: 'list',
        data: ADVERTISED_MODELS.map((id) => ({
          id,
          object: 'model',
          created: 1763100000,
          owned_by: 'xiaofeiyu',
        })),
      });
      return;
    }

    if (req.method !== 'POST' || urlPath !== '/v1/chat/completions') {
      sendJson(res, 404, { error: { message: `Not found: ${urlPath}`, type: 'invalid_request_error' } });
      return;
    }

    let reqBody;
    try {
      reqBody = await readJsonBody(req);
    } catch (err) {
      sendJson(res, err.status || 400, { error: { message: err.message, type: 'invalid_request_error' } });
      return;
    }

    const reqAbort = new AbortController();
    const onClose = () => {
      if (!res.writableEnded) reqAbort.abort(new Error('client_disconnected'));
    };
    res.on('close', onClose);

    try {
      const slidersOverride = reqBody.sliders || reqBody.metadata?.sliders || null;
      const { modelTag, weights } = parseTranslatorWeights(reqBody.model, slidersOverride);
      const joinSep = reqBody.bubble_join === 'newline' ? '\n' : reqBody.bubble_join === 'paragraph' ? '\n\n' : defaultJoinSep;
      const stream = Boolean(reqBody.stream);
      const messages = Array.isArray(reqBody.messages) ? reqBody.messages : [];

      const upstreamJson = await callUpstreamLlm({
        upstreamUrl,
        upstreamApiKey,
        upstreamModel,
        reqBody,
        fetchImpl,
        signal: reqAbort.signal,
      });

      const choice0 = Array.isArray(upstreamJson.choices) ? upstreamJson.choices[0] : null;
      const msg0 = choice0?.message || {};
      const toolCalls = Array.isArray(msg0.tool_calls) ? msg0.tool_calls : [];

      // Pass upstream tool_calls straight through so Cursor / DSH / AstrBot tool loops work unaltered
      if (toolCalls.length > 0) {
        if (!stream) {
          sendJson(res, 200, { ...upstreamJson, model: modelTag });
          return;
        }
        const cmplId = upstreamJson.id || `chatcmpl_vp_${randomBytes(6).toString('hex')}`;
        const created = upstreamJson.created || Math.floor(Date.now() / 1000);
        res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' });
        res.write(
          `data: ${JSON.stringify({
            id: cmplId,
            object: 'chat.completion.chunk',
            created,
            model: modelTag,
            choices: [
              {
                index: 0,
                delta: {
                  role: 'assistant',
                  ...(msg0.content ? { content: msg0.content } : {}),
                  tool_calls: toolCalls.map((tc, idx) => ({ index: idx, ...tc })),
                },
                finish_reason: null,
              },
            ],
          })}\n\n`
        );
        res.write(
          `data: ${JSON.stringify({
            id: cmplId,
            object: 'chat.completion.chunk',
            created,
            model: modelTag,
            choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
          })}\n\n`
        );
        res.write('data: [DONE]\n\n');
        res.end();
        return;
      }

      const rawDraft = String(msg0.content || '').trim();
      const translated = await translateDraftWithE4B({
        translatorUrl,
        translatorMode,
        userMessages: messages,
        rawDraft,
        modelTag,
        weights,
        joinSep,
        fetchImpl,
        signal: reqAbort.signal,
      });

      const cmplId = `chatcmpl_vp_${randomBytes(6).toString('hex')}`;
      const created = Math.floor(Date.now() / 1000);

      if (!stream) {
        sendJson(res, 200, {
          id: cmplId,
          object: 'chat.completion',
          created,
          model: modelTag,
          choices: [
            {
              index: 0,
              message: { role: 'assistant', content: translated.content },
              finish_reason: choice0?.finish_reason || 'stop',
            },
          ],
          usage: upstreamJson.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          xiaofeiyu_telemetry: translated.telemetry,
        });
        return;
      }

      res.writeHead(200, {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });

      res.write(
        `data: ${JSON.stringify({
          id: cmplId,
          object: 'chat.completion.chunk',
          created,
          model: modelTag,
          choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
        })}\n\n`
      );

      for (let i = 0; i < translated.bubbles.length; i += 1) {
        const piece = (i > 0 ? joinSep : '') + translated.bubbles[i];
        res.write(
          `data: ${JSON.stringify({
            id: cmplId,
            object: 'chat.completion.chunk',
            created,
            model: modelTag,
            choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
          })}\n\n`
        );
      }

      res.write(
        `data: ${JSON.stringify({
          id: cmplId,
          object: 'chat.completion.chunk',
          created,
          model: modelTag,
          choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
          usage: upstreamJson.usage,
          xiaofeiyu_telemetry: translated.telemetry,
        })}\n\n`
      );
      res.write('data: [DONE]\n\n');
      res.end();
    } catch (err) {
      if (!res.headersSent) {
        sendJson(res, err.status || 502, {
          error: { message: err.message || 'Voice proxy error', type: 'proxy_error' },
        });
      } else if (!res.writableEnded) {
        res.end();
      }
    } finally {
      res.off('close', onClose);
    }
  });

  server.on('connection', (sock) => {
    sockets.add(sock);
    sock.once('close', () => sockets.delete(sock));
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      server.off('error', reject);
      resolve();
    });
  });

  const addr = server.address();
  const actualPort = typeof addr === 'object' && addr ? addr.port : port;
  return {
    server,
    host,
    port: actualPort,
    baseUrl: `http://${host}:${actualPort}/v1`,
    close: async () => {
      for (const s of sockets) s.destroy();
      sockets.clear();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}
