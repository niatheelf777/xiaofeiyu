#!/usr/bin/env node
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import statusHandler from '../api/status.js';
import { handleChatRequest } from '../api/chat.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(APP_ROOT, 'public');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
};

function startMockUpstream() {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', 'http://127.0.0.1');

    if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      let body = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      } catch {
        body = {};
      }

      const lastUser = [...(body.messages || [])].reverse().find((m) => m.role === 'user')?.content || '';
      const weights = body?.metadata?.weights || {};

      await new Promise((r) => setTimeout(r, 120));

      let replyChunks;
      if (/代码|python|限流|sql|写个/i.test(lastUser)) {
        replyChunks = [
          '给你写好啦，拿去直接用\n',
          '```python\nimport asyncio\n\nclass AsyncRateLimiter:\n    def __init__(self, rate: int) -> None:\n        self.rate = rate\n```\n',
          '跑通了记得回来跟我说一声',
        ];
      } else if ((weights.brat_mesugaki || 0) >= 0.4) {
        replyChunks = ['就这点小事还特地跑来找我啊\n', '行吧看在你这么诚实的份上，勉强听你说两句'];
      } else if ((weights.older_tease || 0) >= 0.4) {
        replyChunks = ['今天怎么突然这么乖\n', '过来坐近点，跟姐姐说说刚才发生什么了'];
      } else if ((weights.neko_clingy || 0) >= 0.4) {
        replyChunks = ['终于来找我说话啦\n', '先把手里的事情放一放嘛'];
      } else if ((weights.fragile_porcelain || 0) >= 0.4) {
        replyChunks = ['你刚才一直没回消息\n', '现在方便多陪我聊一会儿吗'];
      } else {
        replyChunks = ['刚刚风是不是特别大\n', '先找个暖和点的地方站着，别冻着了\n', '到家跟我报个平安呀'];
      }

      res.writeHead(200, {
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache',
      });

      for (const piece of replyChunks) {
        res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: piece } }] })}\n\n`);
        await new Promise((r) => setTimeout(r, 60));
      }
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  return server;
}

function toWebRequest(req, bodyBuffer) {
  const host = req.headers.host || '127.0.0.1';
  const fullUrl = new URL(req.url || '/', `http://${host}`);
  const headers = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (v === undefined) continue;
    if (Array.isArray(v)) v.forEach((item) => headers.append(k, item));
    else headers.set(k, v);
  }
  const method = (req.method || 'GET').toUpperCase();
  const init = { method, headers };
  if (method !== 'GET' && method !== 'HEAD') init.body = bodyBuffer;
  return new Request(fullUrl.toString(), init);
}

async function sendWebResponse(nodeRes, webRes) {
  const outHeaders = {};
  webRes.headers.forEach((val, key) => {
    outHeaders[key] = val;
  });
  nodeRes.writeHead(webRes.status, outHeaders);

  if (!webRes.body) {
    nodeRes.end();
    return;
  }
  const reader = webRes.body.getReader();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    nodeRes.write(Buffer.from(value));
  }
  nodeRes.end();
}

export async function startLocalServer(options = {}) {
  const useMock = Boolean(options.mock);
  const port = Number(options.port ?? process.env.PORT ?? 3788);
  const host = options.host || process.env.HOST || '127.0.0.1';

  let mockServer = null;
  let envOverride = options.envOverride ? { ...options.envOverride } : undefined;
  const apiUrl = String(options.apiUrl || '').trim();

  if (apiUrl && !useMock) {
    envOverride = {
      XFY_UPSTREAM_URLS: apiUrl,
      ...(options.apiKey ? { XFY_SERVICE_KEY: String(options.apiKey) } : {}),
      ...envOverride,
    };
  }

  if (useMock) {
    mockServer = startMockUpstream();
    await new Promise((resolve) => mockServer.listen(0, '127.0.0.1', resolve));
    const mockPort = mockServer.address().port;
    envOverride = {
      XFY_UPSTREAM_URLS: `http://127.0.0.1:${mockPort}/v1/chat/completions`,
      XFY_SERVICE_KEY: 'local-mock-key',
      ...envOverride,
    };
    Object.assign(process.env, envOverride);
  }

  const localRateLimitOptions =
    options.rateLimitOptions ??
    (process.env.XFY_ENABLE_RATE_LIMIT === '1' ? undefined : { maxRequests: 600, windowMs: 60_000 });

  const pendingTasks = new Set();
  const context = {
    waitUntil: (promise) => {
      const p = Promise.resolve(promise).catch(() => {});
      pendingTasks.add(p);
      p.finally(() => pendingTasks.delete(p));
    },
  };

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || '/', `http://${req.headers.host || '127.0.0.1'}`);
    const pathname = url.pathname;
    const method = (req.method || 'GET').toUpperCase();

    if (pathname.startsWith('/api/')) {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const bodyBuf = Buffer.concat(chunks);
      const webReq = toWebRequest(req, bodyBuf);

      let webRes;
      if (pathname === '/api/status' || pathname === '/api/cluster') {
        webRes = await statusHandler(webReq, context, { env: envOverride });
      } else if (pathname === '/api/chat') {
        webRes = await handleChatRequest(webReq, context, {
          env: envOverride,
          rateLimitOptions: localRateLimitOptions,
        });
      } else {
        webRes = new Response(JSON.stringify({ error: { code: 'NOT_FOUND', message: 'Not found' } }), {
          status: 404,
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
        });
      }
      await sendWebResponse(res, webRes);
      return;
    }

    if (method === 'GET' || method === 'HEAD') {
      const cleanPath = pathname === '/' ? '/index.html' : pathname;
      const resolved = path.resolve(PUBLIC_DIR, `.${cleanPath}`);
      if (!resolved.startsWith(path.resolve(PUBLIC_DIR))) {
        res.writeHead(403);
        res.end('Forbidden');
        return;
      }
      if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) {
        const ext = path.extname(resolved).toLowerCase();
        const mime = MIME_TYPES[ext] || 'application/octet-stream';
        const buf = fs.readFileSync(resolved);
        res.writeHead(200, {
          'Content-Type': mime,
          'Content-Length': buf.length,
          'Cache-Control': 'no-cache',
        });
        if (method === 'HEAD') return res.end();
        res.end(buf);
        return;
      }
    }

    res.writeHead(404);
    res.end('Not found');
  });

  await new Promise((resolve) => server.listen(port, host, resolve));
  const addr = server.address();

  return {
    server,
    mockServer,
    url: `http://${host}:${addr.port}`,
    close: async () => {
      await Promise.allSettled([...pendingTasks]);
      await new Promise((resolve) => server.close(resolve));
      if (mockServer) {
        await new Promise((resolve) => mockServer.close(resolve));
      }
    },
  };
}

const isDirectMain =
  process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isDirectMain) {
  const mock = process.argv.includes('--mock');
  startLocalServer({ mock }).then((app) => {
    console.log(`xiaofeiyu-web local harness (${mock ? 'mock' : 'live'}) running at ${app.url}`);
  });
}
