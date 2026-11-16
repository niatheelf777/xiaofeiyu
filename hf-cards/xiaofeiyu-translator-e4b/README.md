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

Lightweight **4B Universal Voice Translator Sidecar** (`google/gemma-4-E4B-it`, upper 20 layers `22..41` non-QKV, `80` modules, `0` QKV, zero-thought direct bubble translation).

## What It Does

Pair **any** frontier reasoning model (DeepSeek, Claude, GPT, Qwen, local Llama) with `@xiaofeiyu/openai-voice-proxy` or `@xiaofeiyu/dsh-plugin`:
1. The frontier model generates accurate answers, tool calls, and code blocks.
2. `xiaofeiyu-translator-e4b` translates ONLY the conversational prose into natural, multi-bubble Chinese IM speech using the 5-slider weights (`s0` + `c1..c4`) — while passing code fences, math/identifiers, and tool calls through verbatim.

## Verified Benchmarks (`v1`)

- **Validation Loss & Preference**: Eval NLL `2.598 -> 1.368`; held-out DPO preference accuracy **`96.05%` (`73/76`)**.
- **Latency & Invariance**: Median 5-slider translation latency **`302.1 ms`** (`129–140 tok/s` `Q8_0`), **`0 / 36` (`0%`)** code-fence smuggling, and **`7 / 7` (`100%`)** exact video ID + timestamp preservation.
- Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
