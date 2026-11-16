import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import {
  extractAndCompactCodeBlocks,
  parseRawBubblesFromCompletion,
  parseTranslatorWeights,
  spliceVerbatimCodeBlocks,
} from '../src/translator-core.mjs';
import { startVoiceProxyServer } from '../src/proxy.mjs';

test('parseTranslatorWeights clamps L1 <= 1.15 and rejects >5 weights', () => {
  const { modelTag, weights } = parseTranslatorWeights('xiaofeiyu-translator-e4b-v1', {
    xiaofeiyu: 1.0,
    neko_clingy: 0.8,
    older_tease: 0.8,
    brat_mesugaki: 0.0,
    fragile_porcelain: 0.0,
  });
  assert.ok(modelTag.startsWith('xiaofeiyu-translator-e4b-v1|1.000,'));
  assert.ok(weights[1] + weights[2] + weights[3] + weights[4] <= 1.151);

  assert.throws(() => parseTranslatorWeights('xiaofeiyu-translator-e4b-v1|1,0,0,0,0,1'));
});

test('code block protector preserves real Python code verbatim and unwraps smuggled markdown prose', () => {
  const draft = [
    '这是一个示例：',
    '```markdown',
    '### 总结',
    '1. 先检查端口',
    '2. 再重启服务',
    '```',
    '```python',
    'def add(a, b):',
    '    # line 2',
    '    # line 3',
    '    # line 4',
    '    # line 5',
    '    # line 6',
    '    # line 7',
    '    # line 8',
    '    # line 9',
    '    return a + b',
    '```',
  ].join('\n');

  const { compactedDraft, protectedBlocks } = extractAndCompactCodeBlocks(draft, 8);
  assert.equal(protectedBlocks.length, 1);
  assert.equal(protectedBlocks[0].lang, 'python');
  assert.ok(compactedDraft.includes('[CODE_BLOCK_0: 10 lines preserved]'));
  assert.ok(!compactedDraft.includes('```markdown'));

  const translatedRaw = parseRawBubblesFromCompletion(
    '先检查下端口再重启服务嘛<|"|>,<|"|>```markdown\n1. 备用提示\n```<|"|>,<|"|>```python\nplaceholder\n```<|"|>]}>'
  );
  const restored = spliceVerbatimCodeBlocks(translatedRaw, protectedBlocks);
  assert.equal(restored[0], '先检查下端口再重启服务嘛');
  assert.equal(restored[1], '备用提示');
  assert.ok(restored[2].includes('# line 9'));
  assert.ok(restored[2].includes('return a + b'));
});

test('proxy routes upstream draft -> E4B llama-server translator -> OpenAI SSE bubbles & passes tool_calls through', async () => {
  let capturedLora = null;
  const mockUpstream = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    const lastUser = body.messages.at(-1)?.content || '';

    if (lastUser.includes('CALL_TOOL')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          id: 'up_tc_1',
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  { id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"README.md"}' } },
                ],
              },
              finish_reason: 'tool_calls',
            },
          ],
        })
      );
      return;
    }

    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        id: 'up_txt_1',
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: '给你写好了：\n```python\nprint(42)\n```\n请注意查收。',
            },
            finish_reason: 'stop',
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 20, total_tokens: 32 },
      })
    );
  });

  const mockLlama = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    capturedLora = body.lora;
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        content: '写好啦拿去直接跑<|"|>,<|"|>```python\nplaceholder\n```<|"|>,<|"|>有报错随时喊我嘛<|"|>]}>',
      })
    );
  });

  await new Promise((r) => mockUpstream.listen(0, '127.0.0.1', r));
  await new Promise((r) => mockLlama.listen(0, '127.0.0.1', r));

  const proxy = await startVoiceProxyServer({
    host: '127.0.0.1',
    port: 0,
    upstreamBaseUrl: `http://127.0.0.1:${mockUpstream.address().port}/v1`,
    upstreamApiKey: 'mock_key',
    upstreamModel: 'deepseek-chat',
    translatorUrl: `http://127.0.0.1:${mockLlama.address().port}`,
    translatorMode: 'llama-completion',
  });

  try {
    // 1. Streaming text -> E4B 5-slider bubbles + verbatim code
    const sseRes = await fetch(`${proxy.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'xiaofeiyu-translator-e4b-v1:neko',
        stream: true,
        messages: [{ role: 'user', content: '帮我写个打印42' }],
      }),
    });
    assert.equal(sseRes.status, 200);
    const sseText = await sseRes.text();
    assert.ok(sseText.includes('写好啦拿去直接跑'));
    assert.ok(sseText.includes('print(42)'));
    assert.ok(sseText.includes('有报错随时喊我嘛'));
    assert.deepEqual(capturedLora, [
      { id: 0, scale: 1.0 },
      { id: 1, scale: 0.85 },
      { id: 2, scale: 0.0 },
      { id: 3, scale: 0.0 },
      { id: 4, scale: 0.0 },
    ]);

    // 2. Tool calls passthrough
    const tcRes = await fetch(`${proxy.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: 'xiaofeiyu-translator-e4b-v1:pure',
        stream: false,
        messages: [{ role: 'user', content: 'CALL_TOOL' }],
      }),
    });
    assert.equal(tcRes.status, 200);
    const tcJson = await tcRes.json();
    assert.equal(tcJson.choices[0].finish_reason, 'tool_calls');
    assert.equal(tcJson.choices[0].message.tool_calls[0].function.name, 'read_file');
  } finally {
    await proxy.close();
    await new Promise((r) => mockUpstream.close(r));
    await new Promise((r) => mockLlama.close(r));
  }
});
