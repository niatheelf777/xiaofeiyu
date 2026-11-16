#!/usr/bin/env node
/**
 * Zero-dependency local Web UI server for Xiaofeiyu (`web/public/` + `/api/{status,cluster,chat}`).
 *
 * Usage:
 *   node web/serve_local.mjs --api http://127.0.0.1:8081 [--port 3788] [--host 127.0.0.1] [--open]
 *   node web/serve_local.mjs --mock
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startLocalServer } from './scripts/dev-local.mjs';

export function parseServeArgs(argv) {
  let apiUrl = process.env.XFY_UPSTREAM_URLS || 'http://127.0.0.1:8081';
  let apiKey = process.env.XFY_SERVICE_KEY || '';
  let host = process.env.HOST || '127.0.0.1';
  let port = Number(process.env.PORT || 3788);
  let mock = false;
  let openBrowser = false;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--mock') {
      mock = true;
    } else if (arg === '--open') {
      openBrowser = true;
    } else if ((arg === '--api' || arg === '--upstream') && argv[i + 1]) {
      apiUrl = argv[++i];
    } else if (arg.startsWith('--api=')) {
      apiUrl = arg.slice('--api='.length);
    } else if (arg === '--api-key' && argv[i + 1]) {
      apiKey = argv[++i];
    } else if (arg === '--host' && argv[i + 1]) {
      host = argv[++i];
    } else if (arg === '--port' && argv[i + 1]) {
      port = Number(argv[++i]);
    }
  }

  return {
    apiUrl: String(apiUrl).trim(),
    apiKey: String(apiKey).trim(),
    host: String(host).trim() || '127.0.0.1',
    port: Number.isFinite(port) && port >= 0 ? port : 3788,
    mock,
    openBrowser,
  };
}

function tryOpenBrowser(targetUrl) {
  try {
    if (process.platform === 'win32') {
      const child = spawn('cmd', ['/c', 'start', '', targetUrl], { stdio: 'ignore', detached: true });
      child.unref();
    } else if (process.platform === 'darwin') {
      const child = spawn('open', [targetUrl], { stdio: 'ignore', detached: true });
      child.unref();
    } else {
      const child = spawn('xdg-open', [targetUrl], { stdio: 'ignore', detached: true });
      child.unref();
    }
  } catch {
    // Non-fatal in headless environments
  }
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMain) {
  const opts = parseServeArgs(process.argv.slice(2));
  startLocalServer({
    host: opts.host,
    port: opts.port,
    mock: opts.mock,
    apiUrl: opts.apiUrl,
    apiKey: opts.apiKey,
  }).then((app) => {
    console.log(`[xiaofeiyu-web] 2D Persona Orb Web UI: ${app.url}`);
    console.log(`[xiaofeiyu-web] Upstream backend:      ${opts.mock ? 'built-in mock' : opts.apiUrl}`);
    if (opts.openBrowser) tryOpenBrowser(app.url);
  });
}
