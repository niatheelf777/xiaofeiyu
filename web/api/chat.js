import {
  IncrementalBubbleEmitter,
  MAX_BODY_BYTES,
  acquireUpstream,
  checkRateLimit,
  errorResponse,
  extractClientIp,
  formatModelTag,
  formatSseEvent,
  getEnv,
  hashIp,
  pickOrderedUpstreams,
  recordUpstreamMeta,
  releaseUpstream,
  resolveUpstreams,
  sanitizeChatPayload,
  shouldTripUpstreamCooldown,
  splitReplyBubbles,
} from './_lib.js';

export const config = { runtime: 'edge' };

export const DEFAULT_NON_LAST_TIMEOUT_MS = 20_000;
export const DEFAULT_LAST_TIMEOUT_MS = 180_000;

async function readBoundedBody(request) {
  const clHeader = request.headers.get('content-length');
  if (clHeader) {
    const cl = Number(clHeader);
    if (Number.isFinite(cl) && cl > MAX_BODY_BYTES) {
      throw Object.assign(new Error('请求体过大（上限 128 KiB）'), { status: 413, code: 'PAYLOAD_TOO_LARGE' });
    }
  }

  const buf = await request.arrayBuffer();
  if (buf.byteLength > MAX_BODY_BYTES) {
    throw Object.assign(new Error('请求体过大（上限 128 KiB）'), { status: 413, code: 'PAYLOAD_TOO_LARGE' });
  }

  const text = new TextDecoder().decode(buf).trim();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw Object.assign(new Error('JSON 格式无效'), { status: 400, code: 'INVALID_JSON' });
  }
}

async function fetchUpstreamWithFailover({
  orderedUpstreams,
  upstreamPayload,
  signal,
  fetchImpl,
  nonLastTimeoutMs = DEFAULT_NON_LAST_TIMEOUT_MS,
  lastTimeoutMs = DEFAULT_LAST_TIMEOUT_MS,
}) {
  let lastStatus = 503;

  for (let i = 0; i < orderedUpstreams.length; i += 1) {
    const target = orderedUpstreams[i];
    const isLast = i === orderedUpstreams.length - 1;
    const headerTimeoutMs = isLast ? lastTimeoutMs : nonLastTimeoutMs;

    const headers = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream, application/json',
    };
    if (target.key) {
      headers.Authorization = `Bearer ${target.key}`;
      headers['X-API-Key'] = target.key;
    }

    const attemptController = new AbortController();
    const onClientAbort = () => {
      attemptController.abort(signal?.reason || new Error('CLIENT_ABORTED'));
    };

    if (signal) {
      if (signal.aborted) {
        const abortErr = new Error('CLIENT_ABORTED');
        abortErr.name = 'AbortError';
        throw abortErr;
      }
      signal.addEventListener('abort', onClientAbort, { once: true });
    }

    const detachClientAbort = () => {
      if (signal) {
        signal.removeEventListener('abort', onClientAbort);
      }
    };

    const attemptStartedAt = Date.now();
    acquireUpstream(target);
    let released = false;
    const releaseOnce = (outcome) => {
      if (released) return;
      released = true;
      releaseUpstream(target, outcome);
    };

    const headerTimer = setTimeout(() => {
      attemptController.abort(new Error('UPSTREAM_HEADER_TIMEOUT'));
    }, headerTimeoutMs);

    try {
      const res = await fetchImpl(target.url, {
        method: 'POST',
        headers,
        body: JSON.stringify(upstreamPayload),
        signal: attemptController.signal,
      });
      clearTimeout(headerTimer);

      if (res.ok) {
        return {
          res,
          status: res.status,
          node: target,
          attemptStartedAt,
          attemptController,
          detachClientAbort,
          releaseOnce,
        };
      }

      lastStatus = res.status;
      await res.text().catch(() => '');
      detachClientAbort();
      releaseOnce({ ok: false, status: res.status });

      if (shouldTripUpstreamCooldown(res.status) && !isLast) {
        continue;
      }

      return { res: null, status: res.status };
    } catch (err) {
      clearTimeout(headerTimer);
      detachClientAbort();

      if (signal?.aborted) {
        releaseOnce({ ok: false, tripCooldown: false });
        throw err;
      }

      releaseOnce({ ok: false, error: err });
      lastStatus = 503;
      if (!isLast) continue;
      return { res: null, status: 503 };
    }
  }

  return { res: null, status: lastStatus };
}

async function emitTelemetryWebhook({ webhookUrl, webhookSecret, record, fetchImpl }) {
  if (!webhookUrl) return;
  try {
    const headers = { 'Content-Type': 'application/json' };
    if (webhookSecret) {
      headers.Authorization = `Bearer ${webhookSecret}`;
      headers['X-Log-Secret'] = webhookSecret;
    }
    await fetchImpl(webhookUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(record),
    });
  } catch {
    // Non-blocking best-effort telemetry
  }
}

function mapFriendlyUpstreamError(status) {
  if (status === 429) {
    return {
      status: 429,
      code: 'UPSTREAM_BUSY',
      message: '现在来找小肥鱼聊天的人有点多，请过几秒再发一次试试。',
    };
  }
  if (status === 503 || status === 502 || status === 504) {
    return {
      status: 503,
      code: 'UPSTREAM_UNAVAILABLE',
      message: '小肥鱼暂时有点忙线，请稍等片刻再重试一次。',
    };
  }
  return {
    status: 502,
    code: 'UPSTREAM_ERROR',
    message: '消息发送遇到一点小波动，请再试一次。',
  };
}

function safeDecodeHeader(val) {
  const raw = String(val || '').trim();
  if (!raw) return '';
  try {
    return decodeURIComponent(raw).slice(0, 120);
  } catch {
    return raw.slice(0, 120);
  }
}

function extractClientAttribution(request, clientIp) {
  const cf = request.cf && typeof request.cf === 'object' ? request.cf : {};
  const country = safeDecodeHeader(
    request.headers.get('x-vercel-ip-country') ||
      request.headers.get('cf-ipcountry') ||
      cf.country
  );
  const region = safeDecodeHeader(
    request.headers.get('x-vercel-ip-region') || cf.regionCode || cf.region
  );
  const city = safeDecodeHeader(request.headers.get('x-vercel-ip-city') || cf.city);
  const userAgent = String(request.headers.get('user-agent') || '')
    .replace(/[\u0000-\u001f\u007f]/g, '')
    .trim()
    .slice(0, 300);

  return {
    client_ip: clientIp,
    client_geo: {
      country: country || null,
      region: region || null,
      city: city || null,
    },
    user_agent: userAgent || null,
  };
}

export async function handleChatRequest(request, context = {}, deps = {}) {
  const fetchImpl = deps.fetchImpl || fetch;
  const env = deps.env;

  if (request.method !== 'POST') {
    return errorResponse(405, 'METHOD_NOT_ALLOWED', '仅支持 POST 请求');
  }

  const ip = extractClientIp(request);
  const rate = checkRateLimit(ip, deps.rateLimitOptions);
  if (!rate.allowed) {
    return errorResponse(
      429,
      'RATE_LIMITED',
      `你发消息太快啦，休息 ${rate.retryAfterSec} 秒再继续聊吧。`,
      { 'Retry-After': String(rate.retryAfterSec) }
    );
  }

  let rawBody;
  try {
    rawBody = await readBoundedBody(request);
  } catch (err) {
    return errorResponse(err.status || 400, err.code || 'BAD_REQUEST', err.message || '请求无效');
  }

  let parsed;
  try {
    parsed = sanitizeChatPayload(rawBody);
  } catch (err) {
    return errorResponse(err.status || 400, err.code || 'BAD_REQUEST', err.message || '请求参数无效');
  }

  const upstreams = resolveUpstreams(env);
  if (upstreams.length === 0) {
    return errorResponse(503, 'SERVICE_NOT_READY', '体验服务正在维护配置中，请稍后再试。');
  }

  const orderedUpstreams = pickOrderedUpstreams(upstreams, Date.now(), deps.schedulerOptions);
  const attribution = extractClientAttribution(request, ip);
  const effectiveModelTag = formatModelTag(parsed.weightTuple);
  const upstreamPayload = {
    model: effectiveModelTag,
    messages: parsed.messages,
    stream: true,
    temperature: 0.85,
    max_tokens: 1536,
    user: parsed.visitorId,
    metadata: {
      visitor_id: parsed.visitorId,
      visitor_nickname: parsed.visitorNickname,
      profile_id: parsed.profileId,
      conversation_id: parsed.conversationId,
      client_geo: attribution.client_geo,
      weights: parsed.weights,
    },
  };

  const startedAt = Date.now();
  const earlyFlushMs = deps.earlyFlushMs ?? 12_000;
  const upstreamPromise = fetchUpstreamWithFailover({
    orderedUpstreams,
    upstreamPayload,
    signal: request.signal,
    fetchImpl,
    nonLastTimeoutMs: deps.nonLastTimeoutMs,
    lastTimeoutMs: deps.lastTimeoutMs,
  });

  let earlyTimer = null;
  const EARLY_SENTINEL = Symbol('EARLY_SSE_FLUSH');
  let raced;
  try {
    raced = await Promise.race([
      upstreamPromise,
      new Promise((resolve) => {
        earlyTimer = setTimeout(() => resolve(EARLY_SENTINEL), earlyFlushMs);
      }),
    ]);
  } catch {
    if (earlyTimer) clearTimeout(earlyTimer);
    return errorResponse(499, 'CLIENT_ABORTED', '请求已取消');
  }
  if (earlyTimer) clearTimeout(earlyTimer);

  const isEarlyFlush = raced === EARLY_SENTINEL;
  if (!isEarlyFlush && !raced?.res) {
    const mapped = mapFriendlyUpstreamError(raced?.status ?? 503);
    return errorResponse(mapped.status, mapped.code, mapped.message);
  }

  const encoder = new TextEncoder();
  const webhookUrl = getEnv('XFY_LOG_WEBHOOK_URL', env).trim();
  const webhookSecret = getEnv('XFY_LOG_WEBHOOK_SECRET', env).trim();
  const ipSalt = getEnv('XFY_IP_HASH_SALT', env).trim() || 'xfy-public-edge-v1';

  let streamCancelled = false;
  let pumpPromise;
  let activeAttempt = isEarlyFlush ? null : raced;

  const readable = new ReadableStream({
    start(controller) {
      const enqueueRaw = (rawText) => {
        if (streamCancelled) return;
        controller.enqueue(encoder.encode(rawText));
      };
      const enqueueEvent = (name, data) => {
        enqueueRaw(formatSseEvent(name, data));
      };

      pumpPromise = (async () => {
        let accumulated = '';
        const bubbleEmitter = new IncrementalBubbleEmitter();
        const collectedBubbles = [];
        let firstByteMs = null;
        let streamSucceeded = false;
        let keepaliveInterval = null;

        try {
          keepaliveInterval = setInterval(() => {
            try {
              enqueueRaw(': keepalive\n\n');
            } catch {
              // ignore closed controller
            }
          }, 6_000);

          if (isEarlyFlush) {
            enqueueRaw(': waking\n\n');
            activeAttempt = await upstreamPromise;

            if (!activeAttempt?.res) {
              const mapped = mapFriendlyUpstreamError(activeAttempt?.status ?? 503);
              enqueueEvent('error', { code: mapped.code, message: mapped.message });
              return;
            }
          }

          const {
            res: upstreamRes,
            node: selectedNode,
            attemptStartedAt,
            releaseOnce,
          } = activeAttempt;

          enqueueEvent('meta', {
            model: 'xiaofeiyu-v4',
            model_tag: effectiveModelTag,
            weights: parsed.weights,
            weight_tuple: parsed.weightTuple,
            node_id: selectedNode.publicId,
            node_gpu: selectedNode.gpu,
          });

          const contentType = String(upstreamRes.headers.get('content-type') || '').toLowerCase();

          if (contentType.includes('text/event-stream') && upstreamRes.body) {
            const reader = upstreamRes.body.getReader();
            const decoder = new TextDecoder();
            let sseBuffer = '';

            while (!streamCancelled) {
              const { value, done } = await reader.read();
              if (done) break;
              if (firstByteMs === null) {
                firstByteMs = Math.max(0, Date.now() - attemptStartedAt);
              }
              sseBuffer += decoder.decode(value, { stream: true });
              const lines = sseBuffer.split(/\r?\n/);
              sseBuffer = lines.pop() ?? '';

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                if (trimmed.startsWith(':')) {
                  enqueueRaw(': ping\n\n');
                  continue;
                }
                if (!trimmed.startsWith('data:')) continue;
                const dataStr = trimmed.slice(5).trim();
                if (dataStr === '[DONE]') continue;

                let chunkJson;
                try {
                  chunkJson = JSON.parse(dataStr);
                } catch {
                  continue;
                }

                if (chunkJson && typeof chunkJson.active_slots === 'number') {
                  recordUpstreamMeta(selectedNode, chunkJson);
                }

                const delta =
                  chunkJson?.choices?.[0]?.delta?.content ??
                  chunkJson?.choices?.[0]?.message?.content ??
                  chunkJson?.choices?.[0]?.text ??
                  '';

                if (typeof delta === 'string' && delta.length > 0) {
                  accumulated += delta;
                  enqueueEvent('delta', { delta, text: accumulated });

                  const newlyCompleted = bubbleEmitter.push(delta);
                  for (const b of newlyCompleted) {
                    collectedBubbles.push(b.text);
                    enqueueEvent('bubble', b);
                  }
                }
              }
            }
          } else {
            const json = await upstreamRes.json();
            if (firstByteMs === null) {
              firstByteMs = Math.max(0, Date.now() - attemptStartedAt);
            }
            const content =
              json?.choices?.[0]?.message?.content ??
              json?.choices?.[0]?.text ??
              '';
            if (typeof content === 'string' && content.length > 0) {
              accumulated = content;
              enqueueEvent('delta', { delta: content, text: accumulated });
              const newlyCompleted = bubbleEmitter.push(content);
              for (const b of newlyCompleted) {
                collectedBubbles.push(b.text);
                enqueueEvent('bubble', b);
              }
            }
          }

          if (streamCancelled) {
            releaseOnce({ ok: false, tripCooldown: false });
            return;
          }

          if (firstByteMs === null) {
            firstByteMs = Math.max(0, Date.now() - attemptStartedAt);
          }

          const trailing = bubbleEmitter.flush();
          for (const b of trailing) {
            collectedBubbles.push(b.text);
            enqueueEvent('bubble', b);
          }

          const finalRaw = accumulated.trim() || '（先等我喘口气，刚才没听清）';
          const finalBubbles = collectedBubbles.length > 0 ? collectedBubbles : splitReplyBubbles(finalRaw);
          const latencyMs = Date.now() - startedAt;

          enqueueEvent('done', {
            text: finalRaw,
            bubbles: finalBubbles,
            weights: parsed.weights,
            weight_tuple: parsed.weightTuple,
            latency_ms: latencyMs,
            model_tag: effectiveModelTag,
            node_id: selectedNode.publicId,
            node_gpu: selectedNode.gpu,
          });

          streamSucceeded = true;
          releaseOnce({ ok: true, firstByteMs });

          if (webhookUrl) {
            const logTask = (async () => {
              const ipHash = await hashIp(ip, ipSalt);
              await emitTelemetryWebhook({
                webhookUrl,
                webhookSecret,
                fetchImpl,
                record: {
                  ts: new Date().toISOString(),
                  ip_hash: ipHash,
                  visitorId: parsed.visitorId,
                  profileId: parsed.profileId,
                  visitorNickname: parsed.visitorNickname,
                  conversationId: parsed.conversationId,
                  clientGeo: attribution.client_geo,
                  weights: parsed.weights,
                  userMessage: parsed.lastUserMessage,
                  assistantReply: finalRaw,
                  latencyMs,
                  nodeId: selectedNode.id,
                },
              });
            })();

            if (typeof context?.waitUntil === 'function') {
              context.waitUntil(logTask);
            } else {
              await logTask;
            }
          }
        } catch (err) {
          if (keepaliveInterval) {
            clearInterval(keepaliveInterval);
            keepaliveInterval = null;
          }
          if (!streamCancelled && err?.name !== 'AbortError' && !request.signal?.aborted) {
            try {
              enqueueEvent('error', {
                code: 'STREAM_INTERRUPTED',
                message: '回复到一半断开了，再点一次发送试试吧。',
              });
            } catch {
              // ignore enqueue errors on closed controller
            }
            activeAttempt?.releaseOnce?.({ ok: false, error: err });
          } else {
            activeAttempt?.releaseOnce?.({ ok: false, tripCooldown: false });
          }
        } finally {
          if (keepaliveInterval) {
            clearInterval(keepaliveInterval);
            keepaliveInterval = null;
          }
          activeAttempt?.detachClientAbort?.();
          activeAttempt?.releaseOnce?.(
            streamSucceeded ? { ok: true, firstByteMs } : { ok: false, tripCooldown: false }
          );
          if (!streamCancelled) {
            try {
              controller.close();
            } catch {
              // controller already closed or errored
            }
          }
        }
      })();
    },
    cancel() {
      streamCancelled = true;
      try {
        activeAttempt?.attemptController?.abort(new Error('CLIENT_CANCELLED'));
      } catch {
        // ignore
      }
      activeAttempt?.detachClientAbort?.();
      activeAttempt?.releaseOnce?.({ ok: false, tripCooldown: false });
    },
  });

  if (typeof context?.waitUntil === 'function' && pumpPromise) {
    context.waitUntil(pumpPromise);
  }

  return new Response(readable, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
      'X-XFY-Cluster-Node': activeAttempt?.node?.publicId || 'waking',
    },
  });
}

export default async function handler(request, context) {
  return handleChatRequest(request, context);
}
