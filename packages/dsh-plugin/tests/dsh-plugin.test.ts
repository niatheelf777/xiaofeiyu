import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  BUDGET_LIMIT,
  BubbleStreamFormatter,
  computeEffectiveWeights,
  formatCompleteBubbleText,
  resolveUpstreamModelTag,
} from '../src/weights.ts';
import {
  buildToolRemapTable,
  remapMessagesToUpstream,
  restoreOpenAIToolCalls,
  restoreToolNameFromUpstream,
} from '../src/tool-remap.ts';
import { startLocalOpenAiBridge } from '../src/http-bridge.ts';
import { runInstaller } from '../scripts/install-dsh.mjs';

test('5-slider LoRA weight normalization & unified/translator model tag resolution', () => {
  const projected = computeEffectiveWeights([1.0, 0.8, 0.8, 0.4, 0.3]);
  assert.equal(projected.isClamped, true);
  assert.equal(projected.weights[0], 1);
  const l1 = projected.weights[1] + projected.weights[2] + projected.weights[3] + projected.weights[4];
  assert.ok(l1 <= BUDGET_LIMIT + 1e-6);
  assert.ok(l1 > 1.13);

  const neko = resolveUpstreamModelTag('xiaofeiyu-v4:neko', {});
  assert.equal(neko.modelTag, 'xiaofeiyu-v4|1.000,0.850,0.000,0.000,0.000');

  const tr = resolveUpstreamModelTag(
    undefined,
    { preset: 'custom', xiaofeiyu: 1.0, older_tease: 0.5 },
    'translator'
  );
  assert.equal(tr.modelTag, 'xiaofeiyu-translator-e4b-v1|1.000,0.000,0.500,0.000,0.000');
});

test('BubbleStreamFormatter expands spoken newlines and keeps code fences intact', () => {
  const raw = [
    '刚看完那个B站视频啦',
    '1分45秒笑死我了',
    '```python',
    'def add(a, b):',
    '    return a + b',
    '```',
    '顺手给你点上赞了',
  ].join('\n');

  const fmt = new BubbleStreamFormatter('paragraph');
  let streamed = '';
  for (let i = 0; i < raw.length; i += 4) {
    streamed += fmt.push(raw.slice(i, i + 4));
  }
  assert.ok(streamed.includes('刚看完那个B站视频啦\n\n1分45秒笑死我了'));
  assert.ok(streamed.includes('```python\ndef add(a, b):\n    return a + b\n```'));
  assert.equal(formatCompleteBubbleText(raw, 'paragraph'), streamed.trim());
});

test('send_message <-> agent_send_message disambiguation leaves Bilibili & web tools untouched', () => {
  const table = buildToolRemapTable([
    { type: 'function', function: { name: 'send_message', description: 'Subagent msg', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'bili_look', description: 'Watch Bilibili', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'web_fetch', description: 'Fetch URL', parameters: { type: 'object' } } },
  ]);
  assert.deepEqual(
    table.tools.map((t) => t.function.name),
    ['agent_send_message', 'bili_look', 'web_fetch']
  );

  const remapped = remapMessagesToUpstream(
    [
      {
        role: 'assistant',
        content: null,
        tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'send_message', arguments: '{"to":"w1"}' } }],
      },
    ],
    table
  );
  assert.equal(remapped[0]?.tool_calls?.[0]?.function?.name, 'agent_send_message');
  assert.equal(restoreToolNameFromUpstream('agent_send_message', table), 'send_message');

  const restored = restoreOpenAIToolCalls(
    [{ index: 0, id: 'tc2', type: 'function', function: { name: 'agent_send_message', arguments: '{}' } }],
    table
  );
  assert.equal(restored?.[0]?.function?.name, 'send_message');
});

test('local OpenAI bridge supports both unified and translator modes and shuts down cleanly', async () => {
  const capturedBodies = [];
  const mockFetch = async (_url, init) => {
    capturedBodies.push(JSON.parse(String(init?.body || '{}')));
    const sse = [
      'data: ' +
        JSON.stringify({
          id: 'c1',
          choices: [
            {
              index: 0,
              delta: {
                content: '第一句\n第二句',
                tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'agent_send_message', arguments: '{}' } }],
              },
              finish_reason: null,
            },
          ],
        }),
      '',
      'data: [DONE]',
      '',
    ].join('\n');
    return new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  };

  let mode = 'unified';
  const bridge = await startLocalOpenAiBridge({
    host: '127.0.0.1',
    port: 0,
    state: {
      getSliderConfig: () => ({ preset: 'custom', xiaofeiyu: 1.0, neko_clingy: 0.4 }),
      getBubbleJoinMode: () => 'paragraph',
      fetchImpl: mockFetch,
      getUpstreamConnection: async () => ({
        mode,
        upstreamBaseUrl: 'http://127.0.0.1:8081/v1',
        translatorProxyUrl: 'http://127.0.0.1:18088/v1',
        upstreamApiKey: '',
      }),
    },
  });

  try {
    const res1 = await fetch(`${bridge.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        stream: true,
        messages: [{ role: 'user', content: '你好' }],
        tools: [{ type: 'function', function: { name: 'send_message', parameters: { type: 'object' } } }],
      }),
    });
    const sse1 = await res1.text();
    assert.equal(capturedBodies[0].model, 'xiaofeiyu-v4|1.000,0.400,0.000,0.000,0.000');
    assert.equal(capturedBodies[0].tools[0].function.name, 'agent_send_message');
    assert.ok(sse1.includes('"name":"send_message"'));
    assert.ok(sse1.includes('第一句\\n\\n第二句'));

    mode = 'translator';
    const res2 = await fetch(`${bridge.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ stream: true, messages: [{ role: 'user', content: '切换翻译模式' }] }),
    });
    await res2.text();
    assert.equal(capturedBodies[1].model, 'xiaofeiyu-translator-e4b-v1|1.000,0.400,0.000,0.000,0.000');
  } finally {
    await bridge.close();
  }
});

test('install-dsh.mjs supports reversible --dry-run / --apply / --uninstall without touching credentials', () => {
  const fakeHome = fs.mkdtempSync(path.join(os.tmpdir(), 'xfy-dsh-os-'));
  try {
    const credPath = path.join(fakeHome, '.credentials.yaml');
    fs.writeFileSync(credPath, 'DUMMY_KEY: keep_untouched\n', 'utf8');

    const webDir = path.join(fakeHome, 'profiles', 'web');
    fs.mkdirSync(webDir, { recursive: true });
    const origPkg = JSON.stringify({ name: 'dsh-profile-web', dependencies: { 'dsh-bili-eyes': '^0.8.3' }, dsh: { profile: { bundles: ['@deepseek-ai/dsh-base', 'dsh-bili-eyes'] } } }, null, 2) + '\n';
    const origPatch = '[]\n';
    fs.writeFileSync(path.join(webDir, 'package.json'), origPkg, 'utf8');
    fs.writeFileSync(path.join(webDir, 'cordis.patch.yml'), origPatch, 'utf8');

    runInstaller({ mode: 'dry-run', dshHome: fakeHome, profile: 'web' });
    assert.equal(fs.readFileSync(path.join(webDir, 'package.json'), 'utf8'), origPkg);

    runInstaller({ mode: 'apply', dshHome: fakeHome, profile: 'web', runtimeMode: 'translator', bridgePort: 18971 });
    const installedPkg = JSON.parse(fs.readFileSync(path.join(webDir, 'package.json'), 'utf8'));
    assert.ok(installedPkg.dsh.profile.bundles.includes('@xiaofeiyu/dsh-plugin'));
    assert.ok(fs.existsSync(path.join(fakeHome, '.agent-presets', 'xiaofeiyu', 'agent.cordis.yml')));

    runInstaller({ mode: 'uninstall', dshHome: fakeHome, profile: 'web' });
    assert.equal(fs.readFileSync(path.join(webDir, 'package.json'), 'utf8'), origPkg);
    assert.equal(fs.readFileSync(path.join(webDir, 'cordis.patch.yml'), 'utf8'), origPatch);
    assert.equal(fs.readFileSync(credPath, 'utf8'), 'DUMMY_KEY: keep_untouched\n');
  } finally {
    fs.rmSync(fakeHome, { recursive: true, force: true });
  }
});
