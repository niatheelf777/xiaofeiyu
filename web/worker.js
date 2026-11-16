import statusHandler from './api/status.js';
import { errorResponse } from './api/_lib.js';
import { handleChatRequest } from './api/chat.js';

function applySecurityHeaders(response) {
  const headers = new Headers(response.headers);
  if (!headers.has('X-Content-Type-Options')) {
    headers.set('X-Content-Type-Options', 'nosniff');
  }
  if (!headers.has('X-Frame-Options')) {
    headers.set('X-Frame-Options', 'DENY');
  }
  if (!headers.has('Referrer-Policy')) {
    headers.set('Referrer-Policy', 'no-referrer');
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    if (pathname === '/api/status') {
      return applySecurityHeaders(await statusHandler(request, ctx, { env }));
    }
    if (pathname === '/api/chat') {
      return applySecurityHeaders(await handleChatRequest(request, ctx, { env }));
    }
    if (pathname.startsWith('/api/')) {
      return applySecurityHeaders(errorResponse(404, 'NOT_FOUND', 'Not found'));
    }

    if (env?.ASSETS && typeof env.ASSETS.fetch === 'function') {
      const assetRes = await env.ASSETS.fetch(request);
      return applySecurityHeaders(assetRes);
    }

    return applySecurityHeaders(new Response('Not found', { status: 404 }));
  },
};
