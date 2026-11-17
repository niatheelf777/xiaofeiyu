# Xiaofeiyu · Split-Phase 5-Slider Persona Mixing & Universal Voice Proxy

[中文 README](README.md) · [Live Demo](https://xfy.tospeak.top) · [Hugging Face Collection](https://huggingface.co/collections/niatheelf777/xiaofeiyu-split-phase-spoken-persona-family-6afb60d6cecd8097f5fb3a13) · [Pages Overview](https://niatheelf777.github.io/xiaofeiyu/) · [llms.txt](llms.txt)

Xiaofeiyu is a low-invasion spoken IM voice engine for Gemma 4: reasoning, code generation, and tool calls execute at **zero LoRA (`LoRA = 0.0`)**, while lightweight 5-slider persona adapters activate exclusively inside `<|"|>` speech-bubble delimiters—delivering authentic multi-bubble human chat without sacrificing intelligence.

Live Demo: [https://xfy.tospeak.top](https://xfy.tospeak.top)

---

## Hardware & Quantization Precision Hierarchy

[`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) is our battle-tested production flagship powering [`xfy.tospeak.top`](https://xfy.tospeak.top). Choose the precision tier that fits your VRAM:

| Tier | Target Hardware | Launcher Preset | Base Model & Precision | Total VRAM (5 LoRAs + 8k KV) | Why This Tier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. #1 Primary Recommendation · High-Precision Base + LoRA** | `≥ 16 GB` VRAM (`RTX 4060 Ti 16GB`, `4070 Ti SUPER`, `4080` / `4090` / `3090`), Apple Silicon `24 GB+` | `--preset 12b` (or `--preset 12b-q8`) | `gemma-4-12B-it-Q8_0.gguf` (or `BF16`) + 5 `.gguf` LoRAs | `~12.8–13.0 GiB` | **Exact precision running on `xfy.tospeak.top`**: avoids low-bit quantization loss on subtle emotional push-pull, subtext granularity, and short-bubble rhythm |
| **2. Dedicated `12 GB` VRAM Sweet Spot · Custom `imatrix` Quant** | `RTX 3060 12GB`, `RTX 4070 / 4070 SUPER 12GB`, `16 GB` Windows WDDM w/ heavy desktop load, `16 GB` Mac | `--preset 12b-q4` | `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` + 5 `.gguf` LoRAs | `~8.5–8.9 GiB` | **Purpose-built for `12 GB` VRAM limits**: calibrated from `BF16` on Mandarin IM + tool turns protecting `token_embd`, `attn_v`, and bubble-boundary layers against `Q4_0` comma collapse |
| **3. `6–8 GB` Small-VRAM Sidecar · Cloud Brain + Local `4B` Voice** | `RTX 3060 8GB`, `RTX 4060 8GB`, ultrabooks | `--preset e4b` | `gemma-4-E4B-it-Q8_0.gguf` + 5 `.gguf` LoRAs | `~5.3 GiB` | **Keep your cloud brain**: DeepSeek / Claude / GPT handle reasoning and code while local `4B` rewrites prose into spoken bubbles in `302 ms` median latency |

![Consumer GPU & precision selection guide: Q8_0 + 5 LoRAs primary on 16GB+, custom UD-Q4_K_XL-xfy for 12GB, E4B sidecar for 6-8GB](assets/showcase/consumer-gpu-guide.png)

---

## 3-Minute Quickstart: Three Turnkey Ways to Run

Install the minimal Python prerequisites first (`pip install tokenizers huggingface_hub`). `launch_local.py` automatically downloads the base GGUF, five LoRA cartridges from [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4), and an official prebuilt `llama-server` binary.

### 1. One-Command Local 2D Persona Orb Web UI (Zero Node Setup)

`pc2_split_phase_server.py` serves the complete paper-and-ink 2D Persona Orb Web UI directly on `http://127.0.0.1:8081/`:

```bash
# 1. #1 Primary Recommendation (>= 16 GB VRAM / 24 GB+ Mac): High-precision Q8_0 + 5 LoRAs (~12.9 GiB VRAM, matches xfy.tospeak.top)
python server/launch_local.py --preset 12b --auto-download-llama --open

# 2. Dedicated 12 GB VRAM Sweet Spot (RTX 3060 / 4070 12GB): Custom BF16-derived UD-Q4_K_XL-xfy (~8.5-8.9 GiB VRAM)
python server/launch_local.py --preset 12b-q4 --auto-download-llama --open
```

Your browser opens `http://127.0.0.1:8081/` with live 2D orb mixing, multi-bubble streaming, and copyable code cards (`http://127.0.0.1:8081/v1` is exposed simultaneously for OpenAI clients). To run the frontend standalone against any remote `:8081` gateway, use `node web/serve_local.mjs --api http://127.0.0.1:8081`.

![Turnkey local 2D Persona Orb Web UI with multi-bubble streaming and copyable code card](assets/showcase/local-webui-turnkey.png)

### 2. DeepSeek Harness (`DSH`) Coding Agent (Dual-Mode Support)

The same local `:8081` gateway supports both zero-plugin OpenAI clients and native DSH bubble/mixer UI:

| Mode | When to Use | Setup | Conversation & Tool Experience |
| :--- | :--- | :--- | :--- |
| **Mode 1: Standard Zero-Plugin API** | Stock DSH, Cursor, OpenWebUI, or any OpenAI SDK | Point Base URL to `http://127.0.0.1:8081/v1` with `model="xiaofeiyu-v4\|1.0,0.8,0,0,0"` | Unwraps multi-bubble speech and fenced code cleanly into `assistant.content` with full multi-hop tool support |
| **Mode 2: `@xiaofeiyu/dsh-plugin`** | Native tactile IM bubbles, code cards, and live 5-slider header mixer inside DSH WebUI | Run the one-line installer below (byte-exact `--uninstall` supported) | Registers collision-free `xiaofeiyu_send_bubbles` & `xiaofeiyu_send_code` tools, tactile chat bubbles + copyable code cards in the main chat flow, and a live 5-slider header popover |

```bash
# Install Xiaofeiyu preset, native bubble/code tools, and 5-slider WebUI mixer into DSH
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1

# Revert byte-for-byte at any time
node packages/dsh-plugin/scripts/install-dsh.mjs --uninstall --profile web
```

![DSH Mode 2 live session: file read tool call, tactile IM chat bubbles, copyable JS card, and live 5-slider mixer popover](assets/showcase/dsh-mode2-mixer.png)

### 3. OpenAI-Compatible API & `E4B` Universal Voice Proxy (`6–8 GB` GPUs)

On `6–8 GB` GPUs—or when you want to keep DeepSeek, Claude, GPT, or Qwen as your upstream reasoning brain—run the `4B` translator sidecar locally (`gemma-4-E4B-it-Q8_0.gguf` + 5 LoRAs, `~5.3 GiB` VRAM) to rewrite conversational prose into multi-bubble persona speech (`302 ms` median latency, `0%` code-block or tool-call alteration):

```bash
# 1. Start local E4B llama-server (port 18085)
python server/launch_local.py --preset e4b --llama-only --auto-download-llama

# 2. Start zero-dependency OpenAI voice proxy (listens on 127.0.0.1:18088/v1)
UPSTREAM_BASE_URL=https://api.deepseek.com/v1 \
UPSTREAM_API_KEY=your_key \
TRANSLATOR_URL=http://127.0.0.1:18085 \
node packages/openai-voice-proxy/bin/xiaofeiyu-voice-proxy.mjs
```

---

## Side-by-Side Chat Showcase

Prompt-engineered role-play models fall back on verbose paragraphs, theatrical stage brackets (`（气鼓鼓）`, `*sighs*`), and emoji clutter; conversely, naive full-sequence fine-tuning degrades coding and multi-turn recall. Xiaofeiyu eliminates surface slop while keeping every reasoning channel intact.

![Side-by-side chat comparison: Xiaofeiyu vs typical RLHF role-play slop](assets/showcase/chat-vs-rlhf-slop.png)

---

## 5-Slider Real-Time Persona Mixing

Built atop a shared IM voice backbone (`s0`) plus four modular personality cartridges (`c1..c4`), weights can be mixed smoothly per request via OpenAI `model="xiaofeiyu-v4|w0,w1,w2,w3,w4"` with zero model reloads:

| Slot | Branch ID | Role | Mixing Budget |
| :---: | :--- | :--- | :--- |
| `0` | `s0_xiaofeiyu_backbone` | Core natural spoken IM backbone | `w0 = 1.0` |
| `1` | `c1_neko_clingy` | Affectionate, clingy warmth | `w1 + w2 + w3 + w4 <= 1.15` |
| `2` | `c2_older_tease` | Calm, teasing older-sister banter | same |
| `3` | `c3_brat_mesugaki` | Sharp-tongued playful tsundere | same |
| `4` | `c4_fragile_porcelain` | Soft-spoken fragile warmth | same |

![Five distinct personas from one prompt via 5-slider linear mixing](assets/showcase/five-persona-grid.png)

---

## Core Design: Split-Phase Brain–Voice Decoupling

```text
User Turn ─▶ Phase A: Thinking & Tool/Code (LoRA = 0.0)
              ├─ <channel>thought ... <channel|>       ──▶ Base reasoning intact
              ├─ calc_python / send_code_or_doc        ──▶ Exact code & math invariance
              └─ call:send_message{bubbles:[
                    <|"|>  ──▶ Phase B: Activate 5-slider LoRA (w0..w4)
                    bubble text
                    <|"|>  ──▶ Closing delimiter resets LoRA = 0.0
                 ]}
```

![Code generation and tool accuracy stay at 100% across all persona sliders](assets/showcase/code-without-lobotomy.png)

| Evaluation Gate | Prompt-Only RP | Full-Network Always-On (`v3`) | **`Xiaofeiyu-12B-v4` (Split-Phase)** |
| :--- | :---: | :---: | :---: |
| **42-Turn Daily + Push-Pull + Math/Poem IQ** | `12 / 12` (`100%`) | `7 / 12` (`58.3%`) | **`12 / 12 (100%)`** |
| **Engineering & Algorithm Probes (`GEN1..5`)** | `5 / 5` (`100%`) | `2 / 5` (`40.0%`) | **`5 / 5 (100%)`** |
| **Multi-Turn Memory & Tool Syntax (`TC1 + TC2`)** | `7 / 7` (`100%`) | `1 / 4` (`25.0%`) | **`7 / 7 (100%)`** |
| **Stage Brackets / Emoji / Markdown Slop** | `88.1%` | `9.5%` | **`0.0%` (`0 / 42` gate)** |

---

## Repository Layout

- [`server/`](server/) — Split-phase `llama-server` gateway, built-in zero-Node 2D Persona Orb Web UI, and turnkey launcher (`python server/test_server.py -v`)
- [`packages/dsh-plugin/`](packages/dsh-plugin/) — DeepSeek Harness (`DSH`) native bubble/code tools, 5-slider React WebUI mixer, and reversible installer (`npm --prefix packages/dsh-plugin test`)
- [`packages/openai-voice-proxy/`](packages/openai-voice-proxy/) — Zero-dep Node OpenAI voice translator proxy (`npm --prefix packages/openai-voice-proxy test`)
- [`web/`](web/) — Paper-ink 2D Persona Orb Web UI, standalone local launcher, and Cloudflare/Vercel Edge proxy (`npm --prefix web test`)
- [`train/`](train/) — Split-phase SFT, residual cartridges, anti-smuggle DPO, Online DPO miner, 5-slot GGUF exporter (`pytest train/test_train.py -v`)
- [`hf-cards/`](hf-cards/) — HuggingFace model cards (`12b-v4` flagship and `translator-e4b` sidecar)
- [`scripts/`](scripts/) — Clean-room leak auditor (`audit_zero_leak.py`) and HuggingFace release verifier (`publish_hf_release.py`)

## License

Code is licensed under **Apache-2.0** ([`LICENSE`](LICENSE)). Gemma 4 base weights and derivative adapters are subject to the Google [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
