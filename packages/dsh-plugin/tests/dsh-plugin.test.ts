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
import {
  XIAOFEIYU_SEND_BUBBLES_SPEC,
  XIAOFEIYU_SEND_CODE_SPEC,
  isSettledNativeDeliveryTurn,
} from '../src/tool-specs.ts';
import { runInstaller } from '../scripts/install-dsh.mjs';
import { buildClientBundleText } from '../scripts/build-client.mjs';
import vm from 'node:vm';

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

test('proxy-restricted parameter keys (file, image, bytes) are losslessly remapped and restored', () => {
  const table = buildToolRemapTable([
    {
      type: 'function',
      function: {
        name: 'custom_upload_check',
        description: 'Inspect local source text',
        parameters: {
          type: 'object',
          properties: {
            file: { type: 'string', description: 'Relative path' },
            bytes: { type: 'number' },
          },
          required: ['file'],
        },
      },
    },
  ]);

  const props = table.tools[0]?.function.parameters?.properties as Record<string, unknown>;
  assert.ok('target_path' in props);
  assert.ok('byte_count' in props);
  assert.ok(!('file' in props));
  assert.deepEqual(table.tools[0]?.function.parameters?.required, ['target_path']);

  const upstreamMsgs = remapMessagesToUpstream(
    [
      {
        role: 'assistant',
        tool_calls: [
          {
            id: 'tc_file',
            type: 'function',
            function: { name: 'custom_upload_check', arguments: '{"file":"src/app.js","bytes":128}' },
          },
        ],
      },
    ],
    table
  );
  assert.equal(upstreamMsgs[0]?.tool_calls?.[0]?.function?.arguments, '{"target_path":"src/app.js","byte_count":128}');

  const restored = restoreOpenAIToolCalls(
    [
      {
        index: 0,
        id: 'tc_file_2',
        type: 'function',
        function: {
          name: 'custom_upload_check',
          arguments: '{"target_path":"src/board.js","byte_count":64,"justification":"routine edit","sandbox_permissions":"workspace-write"}',
        },
      },
    ],
    table
  );
  assert.equal(restored?.[0]?.function?.arguments, '{"file":"src/board.js","bytes":64}');
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

test('preset files only reference registered DSH persona variables (provider, model)', () => {
  const presetFiles = [
    path.resolve(import.meta.dirname, '..', 'presets', 'xiaofeiyu.patch.yml'),
    path.resolve(import.meta.dirname, '..', 'presets', 'xiaofeiyu', 'agent.cordis.yml'),
  ];
  const allowedVars = new Set(['provider', 'model']);
  for (const file of presetFiles) {
    const raw = fs.readFileSync(file, 'utf8');
    const matches = [...raw.matchAll(/\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}/g)];
    for (const m of matches) {
      assert.ok(allowedVars.has(m[1]), `Unregistered prompt variable {{${m[1]}}} in ${path.basename(file)}`);
    }
  }
});

test('Mode 2 native tool specs, <1ms bridge settlement, and live /sliders mixer work end-to-end', async () => {
  const bOut = await XIAOFEIYU_SEND_BUBBLES_SPEC.execute({
    bubbles: [' 哎呀怎么淋雨啦 ', '', '快去擦干头发'],
    flavor: 'neko',
    weights: [1, 0.85, 0, 0, 0],
  });
  assert.equal(bOut.bubbleCount, 2);
  assert.deepEqual(bOut.bubbles, ['哎呀怎么淋雨啦', '快去擦干头发']);

  const cOut = await XIAOFEIYU_SEND_CODE_SPEC.execute({
    language: 'python',
    content: 'def f():\n    return 42',
    title: 'helper.py',
  });
  assert.equal(cOut.lineCount, 2);
  assert.equal(cOut.language, 'python');

  const mode2Tools = [
    { type: 'function', function: { name: 'xiaofeiyu_send_bubbles', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'xiaofeiyu_send_code', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'pwsh', parameters: { type: 'object' } } },
  ];

  // Terminal bubble/code step settles immediately
  assert.equal(
    isSettledNativeDeliveryTurn(
      [
        { role: 'user', content: '你好呀' },
        {
          role: 'assistant',
          tool_calls: [
            { id: 'c1', type: 'function', function: { name: 'xiaofeiyu_send_bubbles', arguments: '{"bubbles":["好久不见"]}' } },
          ],
        },
        { role: 'tool', tool_call_id: 'c1', content: 'Delivered 1 chat bubble(s).' },
      ],
      mode2Tools,
    ),
    true,
  );

  // Mid-task promise with external tools available MUST NOT settle prematurely
  assert.equal(
    isSettledNativeDeliveryTurn(
      [
        { role: 'user', content: '帮我跑下测试' },
        {
          role: 'assistant',
          tool_calls: [
            { id: 'c2', type: 'function', function: { name: 'xiaofeiyu_send_bubbles', arguments: '{"bubbles":["收到，我先跑下单测看看"]}' } },
          ],
        },
        { role: 'tool', tool_call_id: 'c2', content: 'Delivered 1 chat bubble(s).' },
      ],
      mode2Tools,
    ),
    false,
  );

  let upstreamFetchCount = 0;
  let sliderCfg = { preset: 'pure' as const, xiaofeiyu: 1, neko_clingy: 0, older_tease: 0, brat_mesugaki: 0, fragile_porcelain: 0 };
  let overrideFlag = false;

  const bridge = await startLocalOpenAiBridge({
    host: '127.0.0.1',
    port: 0,
    state: {
      getSliderConfig: () => sliderCfg,
      setSliderConfig: (next, ov) => {
        sliderCfg = next as typeof sliderCfg;
        overrideFlag = ov;
      },
      getOverrideModelFlavor: () => overrideFlag,
      getBubbleJoinMode: () => 'paragraph',
      fetchImpl: async () => {
        upstreamFetchCount += 1;
        return new Response('{}', { status: 200 });
      },
      getUpstreamConnection: async () => ({
        mode: 'unified',
        upstreamBaseUrl: 'http://127.0.0.1:8081/v1',
        translatorProxyUrl: 'http://127.0.0.1:18088/v1',
        upstreamApiKey: '',
      }),
    },
  });

  try {
    const baseRoot = bridge.baseUrl.replace(/\/v1$/, '');
    const postSliders = await fetch(`${baseRoot}/sliders`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ preset: 'brat_spice', overrideModelFlavor: true }),
    });
    assert.equal(postSliders.status, 200);
    const sliderJson = await postSliders.json();
    assert.equal(sliderJson.overrideModelFlavor, true);
    assert.deepEqual(sliderJson.effective.weights, [1, 0, 0, 0.85, 0]);

    // Settled Mode 2 turn returns stop SSE with 0 upstream fetches
    const settleRes = await fetch(`${bridge.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'xiaofeiyu-v4:pure',
        stream: true,
        messages: [
          { role: 'user', content: '吐槽我一句' },
          {
            role: 'assistant',
            tool_calls: [
              { id: 'call_b1', type: 'function', function: { name: 'xiaofeiyu_send_bubbles', arguments: '{"bubbles":["笨蛋"]}' } },
            ],
          },
          { role: 'tool', tool_call_id: 'call_b1', content: 'Delivered 1 chat bubble(s).' },
        ],
        tools: mode2Tools,
      }),
    });
    assert.equal(settleRes.status, 200);
    const settleSse = await settleRes.text();
    assert.ok(settleSse.includes('"finish_reason":"stop"'));
    assert.ok(settleSse.includes('xiaofeiyu-v4|1.000,0.000,0.000,0.850,0.000'));
    assert.equal(upstreamFetchCount, 0);
  } finally {
    await bridge.close();
  }
});

test('lib/client.js matches source and registers DSH WebUI slots cleanly via __ModuleLoader__', () => {
  const libClientPath = path.resolve(import.meta.dirname, '..', 'lib', 'client.js');
  const built = buildClientBundleText();
  assert.equal(fs.readFileSync(libClientPath, 'utf8'), built);

  let registeredModule: { id: string; factory: (req: (id: string) => unknown) => { apply: (ctx: unknown) => void; inject: string[] } } | null = null;
  const sandbox = {
    window: {
      __ModuleLoader__: {
        load(mod: typeof registeredModule) {
          registeredModule = mod;
        },
      },
    },
  };
  vm.runInNewContext(built, sandbox);
  assert.ok(registeredModule !== null);
  assert.equal(registeredModule!.id, '@xiaofeiyu/dsh-plugin');

  const fakeReact = {
    useState: (init: unknown) => [init, () => {}],
    useEffect: () => {},
    useCallback: (fn: unknown) => fn,
    useRef: (init: unknown) => ({ current: init }),
  };
  const fakeJsx = {
    jsx: (type: unknown, props: unknown) => ({ type, props }),
    jsxs: (type: unknown, props: unknown) => ({ type, props }),
  };
  const exportsObj = registeredModule!.factory((id: string) => {
    if (id === 'react') return fakeReact;
    if (id === 'react/jsx-runtime') return fakeJsx;
    throw new Error(`Unexpected client require: ${id}`);
  });

  assert.deepEqual(Array.from(exportsObj.inject), ['slots', 'locale', 'uiConversation']);
  const slotRegistrations: Array<{ meta: Record<string, unknown> }> = [];
  let registeredEventKind = '';
  exportsObj.apply({
    effect: (fn: () => void) => fn(),
    locale: { register: () => () => {} },
    uiConversation: {
      events: {
        register(def: { kind: string }) {
          registeredEventKind = def.kind;
        },
      },
    },
    slots: {
      inject(_name: string, fn: () => unknown) {
        return fn();
      },
      register(meta: Record<string, unknown>) {
        slotRegistrations.push({ meta });
        return () => {};
      },
    },
  });

  assert.equal(registeredEventKind, 'xiaofeiyu');
  const slotNames = slotRegistrations.map((s) => `${s.meta.name}:${s.meta.key ?? s.meta.id}`);
  assert.deepEqual(slotNames, [
    'conversation.chat.turnTail:@xiaofeiyu/dsh-plugin',
    'tool.call.toolview:xiaofeiyu_send_bubbles',
    'tool.call.toolview:xiaofeiyu_send_code',
    'conversation.session.header.utilities:xiaofeiyu-five-slider-mixer',
  ]);
});
