import {
  buildClusterSummary,
  getEnv,
  jsonResponse,
  resolveUpstreams,
} from './_lib.js';

export const config = { runtime: 'edge' };

export default async function handler(request, _context, deps = {}) {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return jsonResponse({ error: { code: 'METHOD_NOT_ALLOWED', message: 'Method not allowed' } }, 405);
  }

  const env = deps?.env;
  const upstreams = resolveUpstreams(env);
  const ready = upstreams.length > 0;
  const cluster = buildClusterSummary(upstreams, Date.now());

  return jsonResponse({
    ok: true,
    mode: 'public-edge',
    model: getEnv('XFY_PUBLIC_MODEL_NAME', env) || 'xiaofeiyu-v4',
    ready,
    cluster,
  });
}
