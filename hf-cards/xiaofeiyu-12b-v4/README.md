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

Verified **48-module thin upper-layer non-QKV** 5-slider LoRA suite on `google/gemma-4-12B-it` (`s0` `xiaofeiyu_v4` + `c1_neko_clingy`, `c2_older_tease`, `c3_brat_mesugaki`, `c4_fragile_porcelain`), powering [https://xfy.tospeak.top](https://xfy.tospeak.top).

## Adapter Topology

- **Target Modules**: Strictly `layers 36..47` × `{o_proj, gate_proj, up_proj, down_proj}` (`48` linear modules, `0` QKV modules, `12,713,984` trainable parameters per slot at `rank=16, alpha=32.0`).
- **Exact `<|"|>` (`Token ID 52`) Training & Inference Gate**: Forward-pass `LoRA` activation matches Split-Phase bubble decoding (`0.0` FP32 parity drift).
- **Orthogonal 5-Slider Subspace**: Mean pairwise weight cosine $|\cos(\Delta W_{c_i}, \Delta W_{c_j})| = 0.0746$ (`5.4×` lower interference than full-layer LoRAs), obeying Pythagorean norm addition within `1.1%`.
- **Consumer `12 GB` GPU Support**: Pair with non-QAT `UD-Q4_K_XL` (`imatrix --parse-special`) to fit base weights + `8K` KV cache + all 5 LoRAs in **`9.12 GiB` VRAM** (`KLD = 0.0361`).

## Verified `v4` Blind Gate Benchmarks

| Gate | Target | `Xiaofeiyu-12B-v4` (`PC2`) | Full-48 Always-On `v3` |
| :--- | :--- | :--- | :--- |
| **Human IM Bubble $W_1$ Distance** | `< 4.0` chars | **`2.52 – 3.87` chars** (`median = 9–10`) | — |
| **42-Turn Daily + Push-Pull + Math/Poem IQ** | `>= 11/12` | **`12/12 (100%)`** | `7/12 (58.3%)` |
| **Engineering & Algorithm Probes (`GEN1..5`)** | `5/5` | **`5/5 (100%)`** | `2/5 (40.0%)` |
| **Multi-Hop Tool Syntax & Memory (`TC1 + TC2`)** | `7/7` | **`7/7 (100%)`** | `1/4 (25.0%)` |
| **Hygiene Leaks (Emoji / Brackets / Cantonese / Profanity)** | `0` | **`0` across 78 turns** | — |
| **5-Profile Syntax & `calc_python` Invariance (`278.625`)** | `100%` | **`29/29` syntax, `5/5` exact `278.625`** | — |

## Quickstart (`llama-server` + Split-Phase Server)

```bash
python server/launch_local.py --preset 12b
python server/pc2_split_phase_server.py --llama-base http://127.0.0.1:18082
```

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
