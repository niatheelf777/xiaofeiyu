#!/usr/bin/env node
import { startVoiceProxyServer } from '../src/proxy.mjs';

function parseArgs(argv) {
  const opts = {
    host: process.env.XFY_PROXY_HOST || '127.0.0.1',
    port: Number(process.env.XFY_PROXY_PORT || 18088),
    upstreamBaseUrl: process.env.UPSTREAM_BASE_URL || 'https://api.deepseek.com/v1',
    upstreamApiKey: process.env.UPSTREAM_API_KEY || '',
    upstreamModel: process.env.UPSTREAM_MODEL || 'deepseek-chat',
    translatorUrl: process.env.TRANSLATOR_URL || 'http://127.0.0.1:18085',
    translatorMode: process.env.TRANSLATOR_MODE || 'llama-completion',
    bubbleJoin: process.env.BUBBLE_JOIN || 'paragraph',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '--host' && argv[i + 1]) opts.host = argv[++i];
    else if (a === '--port' && argv[i + 1]) opts.port = Number(argv[++i]);
    else if (a === '--upstream-url' && argv[i + 1]) opts.upstreamBaseUrl = argv[++i];
    else if (a === '--upstream-key' && argv[i + 1]) opts.upstreamApiKey = argv[++i];
    else if (a === '--upstream-model' && argv[i + 1]) opts.upstreamModel = argv[++i];
    else if (a === '--translator-url' && argv[i + 1]) opts.translatorUrl = argv[++i];
    else if (a === '--translator-mode' && argv[i + 1]) opts.translatorMode = argv[++i];
    else if (a === '--bubble-join' && argv[i + 1]) opts.bubbleJoin = argv[++i];
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
startVoiceProxyServer(opts).then((srv) => {
  console.log(
    `[xiaofeiyu-voice-proxy] Listening at ${srv.baseUrl} -> upstream=${opts.upstreamBaseUrl} (${opts.upstreamModel}), translator=${opts.translatorUrl} (${opts.translatorMode})`
  );
});
