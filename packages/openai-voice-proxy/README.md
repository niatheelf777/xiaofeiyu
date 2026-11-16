# `@xiaofeiyu/openai-voice-proxy`

Zero-dependency OpenAI-compatible sidecar proxy pairing any upstream reasoning/coding LLM (DeepSeek, OpenAI, Claude, vLLM, Ollama) with the **5-slider Gemma 4 E4B Voice Translator** (`<0.8s` zero-thought voice layer).

## What It Does

1. **Upstream Brain**: Sends your conversation to any OpenAI-compatible `UPSTREAM_BASE_URL` (`deepseek-chat`, `gpt-*`, local models). Any `tool_calls` from the upstream model pass straight through untouched so coding/agent loops (Cursor, DSH, OpenWebUI) keep working.
2. **Code-Block Protection & Anti-Smuggle**: Extracts real fenced code blocks (`python`, `ts`, `sql`, `bash`, etc.), compacts long blocks for E4B speed, unwraps smuggled ```` ```markdown ```` prose blocks, and splices original code back byte-for-byte.
3. **5-Slider Voice Mixing**: Routes `xiaofeiyu-translator-e4b-v1|w0,w1,w2,w3,w4` or virtual flavors (`:pure`, `:neko`, `:tease`, `:brat`, `:fragile`) directly into `llama-server` LoRA scales.

## Quickstart

```bash
# Terminal 1: Start E4B llama-server with the 5 GGUF cartridges
python ../../server/launch_local.py --preset e4b --llama-port 18085

# Terminal 2: Start the OpenAI Voice Proxy on :18088
UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
UPSTREAM_API_KEY=your_api_key \
UPSTREAM_MODEL=deepseek-chat \
TRANSLATOR_URL=http://127.0.0.1:18085 \
node bin/xiaofeiyu-voice-proxy.mjs --port 18088
```

Point OpenWebUI / AstrBot / Cursor / DSH to `http://127.0.0.1:18088/v1` and select `xiaofeiyu-translator-e4b-v1:neko` (or any 5-slider tag).

## Run Tests

```bash
npm test
```
