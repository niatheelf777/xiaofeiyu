---
license: gemma
base_model: google/gemma-4-E4B-it
language:
  - zh
  - en
pipeline_tag: text-generation
library_name: gguf
tags:
  - gemma-4
  - roleplay
  - conversational
  - chinese
  - im-chat
  - persona
  - multi-lora
  - split-phase
  - llama-cpp
  - gguf
  - companion
  - 8gb-vram
  - openai-proxy
  - xiaofeiyu
---

# Xiaofeiyu Translator E4B (小肥鱼 E4B 302ms 通用声带翻译核 — Universal Voice Sidecar for DeepSeek / Claude / GPT)

> **6–8 GB 小显存显卡 / 云端大脑外挂首选**：给任意前沿大模型（DeepSeek / Claude / GPT / Qwen）外挂「真人短气泡五滑条声带」，代码块与工具调用 `0%` 改动。基于 `google/gemma-4-E4B-it`（仅占用 `~5.3 GiB` 显存，`RTX 3060 8GB` / `RTX 4060 8GB` / 轻薄本均可常驻），配合 `@xiaofeiyu/openai-voice-proxy` 流式 AST 切分器，将上游大模型的说明散文在 **`302 ms` 中位延迟**内实时转译为小肥鱼五滑条短气泡口语，而把 Markdown 代码块、数学公式、URL/视频时间戳与 `tool_calls` 原封不动直通透传。

> **English Summary**: `niatheelf777/xiaofeiyu-translator-e4b` is a lightweight 4B (`google/gemma-4-E4B-it`, upper 20 non-QKV layers `L22..41`) 5-slider spoken-persona sidecar built for `6–8 GB` GPUs or users who want to keep DeepSeek, Claude, GPT, or Qwen as their upstream brain via `@xiaofeiyu/openai-voice-proxy`. Prose turns into natural multi-bubble Mandarin IM chat in `~302 ms` median latency while fenced code blocks, identifiers, and tool calls pass through byte-for-byte untouched (`0 / 36` code-fence smuggling).

## Xiaofeiyu (小肥鱼) Family Navigation / 家族导航

| 组件 / 仓库 | 定位与显存门槛 | 链接 |
| :--- | :--- | :--- |
| **12B 主力打磨版（端到端首选）** | `≥ 16 GB` 显卡 / `24 GB+` Mac 首选 `Q8_0` 原版挂 LoRA（`~12.9 GiB`）；`12 GB` 显卡选特制 `UD-Q4_K_XL-xfy`（`~8.5–8.9 GiB`） | [`xiaofeiyu-12b-v4`（12B 主力打磨版）](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) |
| **E4B 通用声带翻译核** | `6–8 GB` 显卡适用（`~5.3 GiB`），外挂 DeepSeek / Claude / GPT 零改代码以 `302 ms` 改写短气泡 | **`xiaofeiyu-translator-e4b`（当前仓 · E4B 通用声带翻译核）** |
| **在线二维性格球试玩** | 免下载直接拖拽性格球体验五滑条实时混音与代码/算账分相 | [https://xfy.tospeak.top](https://xfy.tospeak.top) |
| **GitHub 一键本地启动仓** | 内置 `:8081` 纸墨风性格球网页 + OpenAI `/v1` 网关 + `DSH` 双模插件 | [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu) |
| **HuggingFace 官方合集** | 小肥鱼主力底座与五槽口语卡带总览 | [Xiaofeiyu Spoken Persona Collection](https://huggingface.co/collections/niatheelf777/xiaofeiyu-split-phase-spoken-persona-family-6afb60d6cecd8097f5fb3a13) |

## Verified Benchmarks (`Xiaofeiyu-Translator-E4B`)

| Metric | Result | Details |
| :--- | :--- | :--- |
| **Median 5-Slider Translation Latency** | **`302.1 ms`** | `129–140 tok/s` on `Q8_0` (`~5.3 GiB` total VRAM w/ 5 `.gguf` LoRAs) |
| **Held-Out DPO Preference Accuracy** | **`96.05%` (`73/76`)** | Zero-thought direct multi-bubble rewrite (`Eval NLL 2.598 -> 1.368`) |
| **Code-Fence & Tool-Call Alteration** | **`0 / 36` (`0%`)** | Fenced code blocks, CLI flags, and JSON `tool_calls` bypass rewrite verbatim |
| **Exact ID & Timestamp Preservation** | **`7 / 7` (`100%`)** | Video IDs, URLs, numbers, and timestamps preserved across all 5 sliders |

## 3-Minute Turnkey Quickstart (`@xiaofeiyu/openai-voice-proxy` + `DSH`)

```bash
git clone https://github.com/niatheelf777/xiaofeiyu.git && cd xiaofeiyu
pip install tokenizers huggingface_hub

# 1. Auto-download llama-server + gemma-4-E4B-it-Q8_0.gguf + 5 voice cartridges on 127.0.0.1:18085
python server/launch_local.py --preset e4b --llama-only --auto-download-llama

# 2. Launch zero-dependency OpenAI Voice Proxy on http://127.0.0.1:18088/v1
UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
UPSTREAM_API_KEY=your_key \
TRANSLATOR_URL=http://127.0.0.1:18085 \
node packages/openai-voice-proxy/bin/xiaofeiyu-voice-proxy.mjs
```

Point any OpenAI client, Cursor, or **DeepSeek Harness (`DSH`)** at `http://127.0.0.1:18088/v1` (pass `X-Xiaofeiyu-Sliders: 1.0,0.85,0,0,0` or install `@xiaofeiyu/dsh-plugin` for native multi-bubble rendering and the live 2D Persona Orb mixer):

```bash
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:18088/v1
```

## 5-Slot Voice Cartridges in This Repo

- `s0_xiaofeiyu_backbone.gguf` — Slot 0 Everyday Warmth (`元气日常`)
- `c1_neko_clingy.gguf` — Slot 1 Sweet & Clingy (`甜系软萌`)
- `c2_older_tease.gguf` — Slot 2 Poised Older Sister (`知性姐姐`)
- `c3_brat_mesugaki.gguf` — Slot 3 Tsundere Wit (`傲娇毒舌`)
- `c4_fragile_porcelain.gguf` — Slot 4 Quiet & Gentle (`安静慢热`)

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
