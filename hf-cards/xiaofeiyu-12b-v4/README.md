---
license: gemma
base_model: google/gemma-4-12B-it
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
  - 12gb-vram
  - 16gb-vram
  - xiaofeiyu
---

# Xiaofeiyu 12B v4 (小肥鱼 12B 主力打磨版 — `Q8_0` High-Precision Primary + `12 GB` Custom Dynamic GGUF)

> **公网试玩站 [`xfy.tospeak.top`](https://xfy.tospeak.top) 同款主力打磨版（`google/gemma-4-12B-it`）**：采用 **Split-Phase 脑声分相架构**——模型在 `<|channel>thought` 思考、`calc_python` 算账与 `send_code_or_doc` 写代码时全程保持 **`LoRA = 0.0` 原厂满血智商（`GEN1..5 = 5/5`）**；仅在 `<|"|>` 口语短气泡区间动态挂载 5 枚轻量 `.gguf` 声带卡带（`s0` 原版日常 + `c1` 粘人猫娘 + `c2` 年上御姐 + `c3` 恶劣雌小鬼 + `c4` 易碎依赖）。
>
> **精度推荐原则**：`≥ 16 GB` 显存显卡与 `24 GB+` Mac **强烈优先选用未经过低比特压缩的高精度底模 `gemma-4-12B-it-Q8_0.gguf`（或 `BF16` 原版）挂载 5 枚 LoRA**（`--preset 12b` / `12b-q8`，总显存约 `12.8–13.0 GiB`），以保留最细腻的微表情与语气推拉；`RTX 3060 / 4070 12GB` 等显存受限显卡则选用本仓专为 12GB 环境打造的 BF16 `imatrix` 特制动态量化版 **`gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf`**（`--preset 12b-q4`，总显存约 `8.5–8.9 GiB`）。

> **English Summary**: `niatheelf777/xiaofeiyu-12b-v4` is our battle-tested online production model powering [xfy.tospeak.top](https://xfy.tospeak.top). For `≥ 16 GB` GPUs and `24 GB+` Apple Silicon Macs, we **strongly recommend pairing the 5 `.gguf` LoRA cartridges (`L36..47`, `24.3 MiB` each) with high-precision `gemma-4-12B-it-Q8_0.gguf` (or `BF16`)** (`--preset 12b` / `--preset 12b-q8`, `~12.8–13.0 GiB` total VRAM w/ 8k KV) to preserve full emotional subtext and bubble rhythm without 4-bit degradation. For `12 GB` VRAM-constrained cards (`RTX 3060 12GB`, `RTX 4070 12GB`), use our custom BF16-derived `llama-imatrix` **`gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf`** (`--preset 12b-q4`, `~8.5–8.9 GiB` total VRAM).

## Xiaofeiyu (小肥鱼) Family Navigation / 家族导航

| 组件 / 仓库 | 定位与显存门槛 | 链接 |
| :--- | :--- | :--- |
| **12B 主力打磨版** | `≥ 16 GB` 首选 `Q8_0` 原版挂 LoRA（`~12.9 GiB`）；`12 GB` 显卡选特制 `UD-Q4_K_XL-xfy`（`~8.5–8.9 GiB`） | **`xiaofeiyu-12b-v4`（当前仓 · 线上同款主力版）** |
| **E4B 通用声带翻译核** | `6–8 GB` 小显存适用（`~5.3 GiB`），外挂 DeepSeek / Claude / GPT 以 `302 ms` 改写短气泡（代码 `0%` 改写） | [`xiaofeiyu-translator-e4b`（E4B 通用声带翻译核）](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) |
| **在线二维性格球试玩** | 免下载直接拖拽性格球体验五滑条实时混音与代码/算账分相 | [https://xfy.tospeak.top](https://xfy.tospeak.top) |
| **GitHub 一键本地启动仓** | 内置 `:8081` 纸墨风性格球网页 + OpenAI `/v1` 网关 + `DSH` 双模插件 | [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu) |
| **HuggingFace 官方合集** | 小肥鱼主力底座与五槽口语卡带总览 | [Xiaofeiyu Spoken Persona Collection](https://huggingface.co/collections/niatheelf777/xiaofeiyu-split-phase-spoken-persona-family-6afb60d6cecd8097f5fb3a13) |

## Three-Tier Hardware & Precision Hierarchy

1. **#1 Primary Recommendation — `Q8_0` (or `BF16`) Base + 5 LoRAs (`≥ 16 GB` VRAM / `24 GB+` Mac, `--preset 12b` / `12b-q8`)**
   - Uses `ggml-org/gemma-4-12B-it-GGUF` (`gemma-4-12B-it-Q8_0.gguf`, `12.22 GiB`) + 5 `.gguf` LoRA cartridges from this repo (`~12.8–13.0 GiB` total VRAM w/ 8k KV).
   - **Exact precision running on `xfy.tospeak.top`**: zero 4-bit loss on micro-expressions, subtext granularity, and short-bubble rhythm (`20/25` held-out `eval_25`).
2. **Dedicated `12 GB` VRAM Sweet Spot — Custom `UD-Q4_K_XL-xfy` (`12 GB` GPUs / heavy-WDDM `16 GB` Windows / `16 GB` Mac, `--preset 12b-q4`)**
   - Because `12B Q8_0` (`~13 GiB`) cannot fit inside `12 GB` VRAM, we quantized `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` (`7.72 GiB` file, `~8.5–8.9 GiB` total VRAM w/ 5 LoRAs + 8k KV) directly from `BF16` using `llama-imatrix --parse-special` on Mandarin IM + code + native Gemma 4 tool/thought turns (protecting `token_embd`, `attn_v`, `ffn_down`, and edge bubble-boundary layers `blk.{0,1,46,47}` in `Q6_K`, eliminating `Q4_0` comma-spam collapse).
3. **`6–8 GB` Small-VRAM Sidecar — Cloud Brain + Local `4B` Voice Translator (`--preset e4b`)**
   - Pair [`niatheelf777/xiaofeiyu-translator-e4b`](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) (`gemma-4-E4B-it-Q8_0.gguf` + 5 LoRAs, `~5.3 GiB` VRAM) with DeepSeek / Claude / GPT via `@xiaofeiyu/openai-voice-proxy`.

All VRAM measurements below were captured on `llama-server` (`-ngl 99 -fa on --cache-ram 0 -ctk q8_0 -ctv q8_0`) with all 5 `.gguf` LoRA cartridges preloaded simultaneously:

| Tier | Preset | Base GGUF | File Size | Total VRAM (5 LoRAs + 8k KV) | Held-Out `eval_25` | `GEN1..5` | Target Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 首选推荐 (`Q8_0`)** | `12b` / `12b-q8` | `ggml-org/gemma-4-12B-it-GGUF` (`Q8_0`) | `12.22 GiB` | `~12.8–13.0 GiB` | **`20/25`** | `4/5`–`5/5` | **`RTX 4060 Ti 16GB`**, `4070 Ti SUPER`, `4080 / 4090 / 3090`, `Mac 24GB+` |
| **2. `12 GB` 甜点定制 (`UD-Q4_K_XL-xfy`)** | `12b-q4` | `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` *(in this repo)* | `7.72 GiB` | `~8.5–8.9 GiB` | `19/25` (`0` comma collapse) | `5/5` | **`RTX 3060 12GB`**, **`RTX 4070 / 4070 SUPER 12GB`**, heavy-WDDM `16GB` Windows, `Mac 16GB` |
| **3. `6–8 GB` 外挂代理 (`E4B Q8_0`)** | `e4b` | `ggml-org/gemma-4-E4B-it-GGUF` (`Q8_0`) | `4.65 GiB` | `~5.3 GiB` | — | Upstream Brain | `RTX 3060 8GB`, `RTX 4060 8GB`, ultrabooks + Cloud API |

## 3-Minute Turnkey Quickstart (Built-in `:8081` 2D Persona Orb Web UI + Dual-Mode `DSH`)

### Option A — 1-Command Local Web UI & OpenAI `/v1` API (Zero Node Dependencies)

```bash
git clone https://github.com/niatheelf777/xiaofeiyu.git && cd xiaofeiyu
pip install tokenizers huggingface_hub

# 1. #1 Primary Recommendation (>= 16 GB VRAM / 24 GB+ Mac): Q8_0 base + 5 LoRAs (~12.9 GiB VRAM)
python server/launch_local.py --preset 12b --auto-download-llama --open

# 2. Dedicated 12 GB VRAM Sweet Spot (RTX 3060 / 4070 12GB): custom UD-Q4_K_XL-xfy (~8.5-8.9 GiB VRAM)
python server/launch_local.py --preset 12b-q4 --auto-download-llama --open
```

### Option B — Connect DeepSeek Harness (`DSH`) or Any OpenAI Client

- **Mode 1 (Zero-Plugin OpenAI Direct)**: Point `DSH`, Cursor, or OpenWebUI to `http://127.0.0.1:8081/v1` and set `model` to `xiaofeiyu-v4|1.0,0.85,0,0,0` (`s0,c1,c2,c3,c4`).
- **Mode 2 (`@xiaofeiyu/dsh-plugin` Native Multi-Bubble + Live Header Mixer)**:

```bash
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1
# Reversible clean uninstall:
node packages/dsh-plugin/scripts/install-dsh.mjs --uninstall --profile web
```

### Option C — Manual `llama-server` (`b8827+` / `v1.0.0+`) Launch

```bash
# 1. Primary (16 GB+ VRAM): download Q8_0 base + 5 LoRA cartridges
hf download ggml-org/gemma-4-12B-it-GGUF gemma-4-12B-it-Q8_0.gguf --local-dir ./xiaofeiyu-12b
hf download niatheelf777/xiaofeiyu-12b-v4 \
  s0_xiaofeiyu_backbone.gguf c1_neko_clingy.gguf c2_older_tease.gguf c3_brat_mesugaki.gguf c4_fragile_porcelain.gguf \
  --local-dir ./xiaofeiyu-12b

llama-server \
  -m ./xiaofeiyu-12b/gemma-4-12B-it-Q8_0.gguf \
  --lora ./xiaofeiyu-12b/s0_xiaofeiyu_backbone.gguf,./xiaofeiyu-12b/c1_neko_clingy.gguf,./xiaofeiyu-12b/c2_older_tease.gguf,./xiaofeiyu-12b/c3_brat_mesugaki.gguf,./xiaofeiyu-12b/c4_fragile_porcelain.gguf \
  --lora-init-without-apply -ngl 99 -fa on --cache-ram 0 --special \
  -ctk q8_0 -ctv q8_0 -c 16384 -np 1 --host 127.0.0.1 --port 8082

# 2. For 12 GB VRAM cards, replace `-m` with `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` from this repo and set `-c 8192`.
```

## Verified Blind Gate Benchmarks (`Xiaofeiyu-12B-v4`)

| Evaluation Suite | Result | Notes |
| :--- | :--- | :--- |
| **Engineering & Algorithm Probes (`GEN1..5`)** | **`5/5 (100%)`** | Full base-model coding & SQL/algorithm fidelity (`LoRA = 0.0` in Phase A) |
| **Multi-Hop Tool Syntax & Memory (`TC1 + TC2`)** | **`7/7 (100%)`** | Exact `calc_python` execution (`278.625`) & multi-turn entity recall |
| **42-Turn Daily + Push-Pull + Math/Poem Suite** | **`12/12 (100%)`** | Natural 9–10 char median Mandarin IM short bubbles |
| **Consumer `UD-Q4_K_XL` Held-Out `eval_25`** | **`19/25`** | `0` comma-collapse turns, `0` stage-bracket / emoji leaks |

## Repository Files & SHA256 Checksums

| File | Size (Bytes) | SHA256 Prefix | Role |
| :--- | :--- | :--- | :--- |
| `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` | `8,285,657,920` | `5402dd418057` | Dedicated `12 GB` GPU custom `imatrix` base GGUF |
| `s0_xiaofeiyu_backbone.gguf` | `25,434,912` | `fe2a5cc4ad2e` | Slot 0 Voice Core LoRA (`L36..47` non-QKV, DPO-polished) |
| `s0_xiaofeiyu_v4_1_dpo.gguf` | `25,434,912` | `fe2a5cc4ad2e` | Versioned alias of Slot 0 |
| `c1_neko_clingy.gguf` | `25,434,912` | `5f78a32dff72` | Slot 1 (`粘人猫娘` — Clingy Catgirl) |
| `c2_older_tease.gguf` | `25,434,912` | `57325470303a` | Slot 2 (`年上御姐` — Teasing Older Sister) |
| `c3_brat_mesugaki.gguf` | `25,434,912` | `fa0e68a8d151` | Slot 3 (`恶劣雌小鬼` — Cheeky Tsundere) |
| `c4_fragile_porcelain.gguf` | `25,434,912` | `ac6a18e51c03` | Slot 4 (`易碎依赖` — Delicate Attachment) |
| `lora_gguf_manifest.json` | `1,150` | — | 5-slot LoRA manifest |

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
