---
license: apache-2.0
base_model: google/gemma-4-12B-it
language:
- zh
- en
library_name: peft
tags:
- gemma-4
- lora
- split-phase
- xiaofeiyu
- conversational-ai
---

# `niatheelf777/xiaofeiyu-12b-v4`

Verified **48-module thin upper-layer non-QKV** 5-slider LoRA suite on `google/gemma-4-12B-it` (`s0_xiaofeiyu_backbone` + `c1_neko_clingy`, `c2_older_tease`, `c3_brat_mesugaki`, `c4_fragile_porcelain`), powering [https://xfy.tospeak.top](https://xfy.tospeak.top) and [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu).

## Adapter Topology

- **Target Modules**: Strictly `layers 36..47` × `{o_proj, gate_proj, up_proj, down_proj}` (`48` linear modules, `0` QKV modules, `12,713,984` trainable parameters per slot at `rank=16, alpha=32.0`).
- **Exact `<|"|>` (`Token ID 52`) Training & Inference Gate**: Forward-pass `LoRA` activation matches Split-Phase bubble decoding (`0.0` FP32 parity drift).
- **Orthogonal 5-Slider Subspace**: Mean pairwise weight cosine $|\cos(\Delta W_{c_i}, \Delta W_{c_j})| = 0.0746$ (`5.4×` lower interference than full-layer LoRAs), obeying Pythagorean norm addition within `1.1%`.
- **Consumer `12 GB` GPU Support**: Pair with `Q4_K_M` / `UD-Q4_K_XL` (`--preset 12b-q4`) to fit base weights + KV cache + all 5 LoRAs in **`9.12 GiB` VRAM** (`KLD = 0.0361`).

## Verified `v4` Blind Gate Benchmarks

| Gate | Target | `Xiaofeiyu-12B-v4` (`PC2`) | Full-48 Always-On `v3` |
| :--- | :--- | :--- | :--- |
| **Human IM Bubble $W_1$ Distance** | `< 4.0` chars | **`2.52 – 3.87` chars** (`median = 9–10`) | — |
| **42-Turn Daily + Push-Pull + Math/Poem IQ** | `>= 11/12` | **`12/12 (100%)`** | `7/12 (58.3%)` |
| **Engineering & Algorithm Probes (`GEN1..5`)** | `5/5` | **`5/5 (100%)`** | `2/5 (40.0%)` |
| **Multi-Hop Tool Syntax & Memory (`TC1 + TC2`)** | `7/7` | **`7/7 (100%)`** | `1/4 (25.0%)` |
| **Hygiene Leaks (Emoji / Brackets / Cantonese / Profanity)** | `0` | **`0` across 78 turns** | — |
| **5-Profile Syntax & `calc_python` Invariance (`278.625`)** | `100%` | **`29/29` syntax, `5/5` exact `278.625`** | — |

## Turnkey Quickstart (Built-in 2D Orb Web UI + OpenAI API + DSH)

Clone [`niatheelf777/xiaofeiyu`](https://github.com/niatheelf777/xiaofeiyu), place the 5 `.gguf` adapters under `loras/12b/`, and put `gemma-4-12B-it-Q4_K_M.gguf` (or `Q8_0.gguf`) under `models/`:

```bash
# One command launches llama-server + Split-Phase gateway + 2D Persona Orb Web UI on http://127.0.0.1:8081/
python server/launch_local.py --preset 12b-q4 --auto-download-llama --open

# Optional: install native IM bubble/code tools & live 5-slider mixer into DeepSeek Harness (DSH)
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1
```

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
