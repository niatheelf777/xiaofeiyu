---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
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

# `niatheelf777/xiaofeiyu-26b-a4b`

High-throughput **5-slider Split-Phase** MoE LoRA suite on `google/gemma-4-26B-A4B-it` (`s0_xiaofeiyu_backbone` + `c1_neko_clingy`, `c2_older_tease`, `c3_brat_mesugaki`, `c4_fragile_porcelain`). Companion repository: [github.com/niatheelf777/xiaofeiyu](https://github.com/niatheelf777/xiaofeiyu).

## Architecture Highlights

- **Split-Phase Token-Gated Inference (`LoRA = 0.0` during reasoning & code, `LoRA = w` inside `<|"|>` speech bubbles)**: Preserves `GEN1..5` code generation (`5/5`) and `calc_python` tool accuracy while accelerating single-stream decode to `192–233 tok/s` (`3.8B` active parameters per token).
- **Upper-Layer Non-QKV + MoE Expert Injection**: Adapts upper non-QKV output/MLP modules (`0` Q/K/V modules), preserving hybrid local/global KV-cache coherence.
- **5-Slider Linear Mixing**: Compose base Xiaofeiyu + any $L_1 \le 1.15$ combination of 4 personality cartridges at runtime via `model="xiaofeiyu-v4|w0,w1,w2,w3,w4"`.

## Turnkey Quickstart

Clone [`niatheelf777/xiaofeiyu`](https://github.com/niatheelf777/xiaofeiyu), place the 5 `.gguf` adapters under `loras/26b-a4b/`, and put `gemma-4-26B-A4B-it-Q8_0.gguf` under `models/`:

```bash
# Launches llama-server + Split-Phase OpenAI gateway + 2D Persona Orb Web UI on http://127.0.0.1:8081/
python server/launch_local.py --preset 26b-a4b --auto-download-llama --open
```

Base model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
