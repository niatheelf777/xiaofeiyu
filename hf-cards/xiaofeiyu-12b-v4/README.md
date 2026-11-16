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
  - xiaofeiyu
---

# Xiaofeiyu 12B v4 (小肥鱼 12B 主力全能版 — 12GB Consumer GGUF + 5-Slider Split-Phase Voice Suite)

> **微调不降智的中文微信短气泡伴侣模型（`google/gemma-4-12B-it`）**：采用 **Split-Phase 脑声分相架构**——模型在 `<|channel>thought` 思考、`calc_python` 算账与 `send_code_or_doc` 写代码时全程保持 **`LoRA = 0.0` 原厂满血智商（`GEN1..5 = 5/5`）**；仅在 `<|"|>` 口语短气泡区间动态挂载 5 枚轻量 `.gguf` 声带卡带（`s0` 原版日常 + `c1` 粘人猫娘 + `c2` 年上御姐 + `c3` 恶劣雌小鬼 + `c4` 易碎依赖），在一张 **`RTX 3060 12GB` / `RTX 4070 12GB`** 消费卡上即可跑满自然中文短句聊天与高质量编程辅助。

> **English Summary**: `niatheelf777/xiaofeiyu-12b-v4` pairs five thin upper-layer non-QKV `.gguf` LoRA voice cartridges (`L36..47`, `24.3 MiB` each) with a BF16-derived dynamic `llama-imatrix` `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` (`7.72 GiB` file, `~8.9 GiB` total VRAM with all 5 LoRAs + Q8 KV cache). Reasoning and code generation run at `LoRA = 0.0` (`GEN1..5 = 5/5`), while spoken Mandarin IM bubbles mix five persona sliders in real time on a single `llama-server` instance (`0` stage brackets, `0` comma-spam collapse).

## Xiaofeiyu (小肥鱼) Family Navigation / 家族矩阵导航

| 组件 / 仓库 | 定位与显存门槛 | 链接 |
| :--- | :--- | :--- |
| **12B 主力全能版** | `RTX 3060 / 4070 12GB` 首选单模（`UD-Q4_K_XL` `~8.9 GiB` 显存，代码推理 `5/5`） | **`xiaofeiyu-12b-v4`（当前仓 · 12B 主力全能版）** |
| **26B-A4B 极速 MoE 版** | `4B` 激活参数高吞吐 MoE（16GB `UD-Q4_K_XL` / `IQ4_XS`；12GB `UD-IQ3_M`） | [`xiaofeiyu-26b-a4b`（26B-A4B 极速 MoE 版）](https://huggingface.co/niatheelf777/xiaofeiyu-26b-a4b) |
| **E4B 通用声带翻译核** | `302 ms` 零思考口语翻译旁路（`~5.3 GiB`），外挂 DeepSeek / Claude / GPT 零改代码 | [`xiaofeiyu-translator-e4b`（E4B 302ms 通用声带翻译核）](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) |
| **在线二维性格球试玩** | 免下载直接拖拽性格球体验五滑条实时混音与代码/算账分相 | [https://xfy.tospeak.top](https://xfy.tospeak.top) |
| **GitHub 一键本地启动仓** | 内置 `:8081` 纸墨风性格球网页 + OpenAI `/v1` 网关 + `DSH` 双模插件 | [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu) |
| **HuggingFace 官方合集** | 小肥鱼全系 GGUF 底座与五槽口语卡带总览 | [Xiaofeiyu Spoken Persona Collection](https://huggingface.co/collections/niatheelf777/xiaofeiyu-split-phase-spoken-persona-family-6afb60d6cecd8097f5fb3a13) |

## Hardware & Consumer GGUF Fit (`RTX 3060 / 4070 12GB` Ready)

Quantizing directly from the exact `BF16` `google/gemma-4-12B-it` checkpoint with `llama-imatrix --parse-special` (protecting `token_embd`, `attn_v`, `ffn_down`, and edge layers `blk.{0,1,46,47}` in `Q6_K`) eliminates the comma-spam collapse seen when pairing BF16-trained LoRAs with drifted `QAT Q4_0` weights. All VRAM numbers below are measured on `llama-server` (`-ngl 99 -fa on --cache-ram 0 -ctk q8_0 -ctv q8_0`) with all 5 `.gguf` LoRA cartridges preloaded simultaneously:

| Edition | File | File Size | `ctx=4096` VRAM (`L4`) | `ctx=8192` VRAM (`L4`) | L4 Decode (`p50`) | `GEN1..5` | Held-Out `eval_25` | Target GPUs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`12b-q4`** *(Recommended)* | `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` *(in this repo)* | `7.72 GiB` | `8.47 GiB` | `8.51 GiB` | `25.6 tok/s` | `5/5` | `19/25` (`0` comma collapse) | **`RTX 3060 12GB`**, **`RTX 4070 12GB`**, `RTX 4060 Ti 16GB` |
| **`12b`** *(Reference `Q8_0`)* | `ggml-org/gemma-4-12B-it-GGUF` (`Q8_0`) | `12.22 GiB` | — | `12.84 GiB` (`-np 2`) | `17.9 tok/s` | `4/5` | `20/25` | `RTX 4060 Ti 16GB`, `NVIDIA L4 24GB`, Apple Silicon `24GB+` |

## 3-Minute Turnkey Quickstart (Built-in `:8081` 2D Persona Orb Web UI + Dual-Mode `DSH`)

### Option A — 1-Command Local Web UI & OpenAI `/v1` API (Zero Node Dependencies)

```bash
git clone https://github.com/niatheelf777/xiaofeiyu.git && cd xiaofeiyu
pip install tokenizers huggingface_hub

# Auto-downloads llama-server + gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf + 5 voice cartridges
# and opens the interactive 2D Persona Orb Web UI at http://127.0.0.1:8081/
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
hf download niatheelf777/xiaofeiyu-12b-v4 \
  gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf \
  s0_xiaofeiyu_backbone.gguf c1_neko_clingy.gguf c2_older_tease.gguf c3_brat_mesugaki.gguf c4_fragile_porcelain.gguf \
  --local-dir ./xiaofeiyu-12b

llama-server \
  -m ./xiaofeiyu-12b/gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf \
  --lora ./xiaofeiyu-12b/s0_xiaofeiyu_backbone.gguf,./xiaofeiyu-12b/c1_neko_clingy.gguf,./xiaofeiyu-12b/c2_older_tease.gguf,./xiaofeiyu-12b/c3_brat_mesugaki.gguf,./xiaofeiyu-12b/c4_fragile_porcelain.gguf \
  --lora-init-without-apply -ngl 99 -fa on --cache-ram 0 --special \
  -ctk q8_0 -ctv q8_0 -c 8192 -np 1 --host 127.0.0.1 --port 8082
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
| `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` | `8,285,657,920` | `5402dd418057` | 12 GB consumer dynamic `imatrix` base GGUF |
| `s0_xiaofeiyu_backbone.gguf` | `25,434,912` | `fe2a5cc4ad2e` | Slot 0 Voice Core LoRA (`L36..47` non-QKV, DPO-polished) |
| `s0_xiaofeiyu_v4_1_dpo.gguf` | `25,434,912` | `fe2a5cc4ad2e` | Versioned alias of Slot 0 |
| `c1_neko_clingy.gguf` | `25,434,912` | `5f78a32dff72` | Slot 1 (`粘人猫娘` — Clingy Catgirl) |
| `c2_older_tease.gguf` | `25,434,912` | `57325470303a` | Slot 2 (`年上御姐` — Teasing Older Sister) |
| `c3_brat_mesugaki.gguf` | `25,434,912` | `fa0e68a8d151` | Slot 3 (`恶劣雌小鬼` — Cheeky Tsundere) |
| `c4_fragile_porcelain.gguf` | `25,434,912` | `ac6a18e51c03` | Slot 4 (`易碎依赖` — Delicate Attachment) |
| `lora_gguf_manifest.json` | `1,150` | — | 5-slot LoRA manifest |

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
