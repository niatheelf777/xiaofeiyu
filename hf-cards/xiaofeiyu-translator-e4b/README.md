---
license: apache-2.0
base_model: google/gemma-4-E4B-it
language:
- zh
- en
library_name: peft
tags:
- gemma-4
- lora
- voice-translator
- openai-proxy
- xiaofeiyu
---

# `niatheelf777/xiaofeiyu-translator-e4b`

Lightweight **4B Universal Voice Translator Sidecar** (`google/gemma-4-E4B-it`, upper 20 layers `22..41` non-QKV, `80` modules, `0` QKV, zero-thought direct bubble translation). Companion repository: [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu).

## What It Does

Pair **any** frontier reasoning model (DeepSeek, Claude, GPT, Qwen, local Llama) with `@xiaofeiyu/openai-voice-proxy` or `@xiaofeiyu/dsh-plugin`:
1. The frontier model generates accurate answers, tool calls, and code blocks.
2. `xiaofeiyu-translator-e4b` translates ONLY the conversational prose into natural, multi-bubble Chinese IM speech using the 5-slider weights (`s0` + `c1..c4`) — while passing code fences, math/identifiers, and tool calls through verbatim.

## Verified Benchmarks (`v1`)

- **Validation Loss & Preference**: Eval NLL `2.598 -> 1.368`; held-out DPO preference accuracy **`96.05%` (`73/76`)**.
- **Latency & Invariance**: Median 5-slider translation latency **`302.1 ms`** (`129–140 tok/s` `Q8_0`), **`0 / 36` (`0%`)** code-fence smuggling, and **`7 / 7` (`100%`)** exact video ID + timestamp preservation.

## Turnkey Quickstart

Clone [`niatheelf777/xiaofeiyu`](https://github.com/niatheelf777/xiaofeiyu), place the 5 `.gguf` adapters under `loras/e4b/`, and put `gemma-4-E4B-it-Q8_0.gguf` under `models/`:

```bash
# 1. Start local E4B llama-server on 127.0.0.1:18085
python server/launch_local.py --preset e4b --llama-only --auto-download-llama

# 2. Start zero-dependency OpenAI voice proxy on 127.0.0.1:18088/v1
UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
UPSTREAM_API_KEY=your_key \
TRANSLATOR_URL=http://127.0.0.1:18085 \
node packages/openai-voice-proxy/bin/xiaofeiyu-voice-proxy.mjs
```

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
