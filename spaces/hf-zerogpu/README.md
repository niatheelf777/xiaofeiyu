---
title: Xiaofeiyu 12B v4 · Split-Phase Companion
emoji: "○"
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.23.0
python_version: "3.11"
app_file: app.py
suggested_hardware: zero-a10g
models:
  - google/gemma-4-12B-it
  - niatheelf777/xiaofeiyu-12b-v4
pinned: false
license: apache-2.0
short_description: Split-Phase 5-slider Mandarin IM companion on Gemma 4 12B BF16 (ZeroGPU)
---

# Xiaofeiyu-12B-v4 · HuggingFace ZeroGPU (PyTorch BF16)

In-process **PyTorch BF16 Split-Phase** deployment of **Xiaofeiyu-12B-v4** (`google/gemma-4-12B-it` + 5 GGUF LoRA voice cartridges from `niatheelf777/xiaofeiyu-12b-v4`) built specifically for **HuggingFace ZeroGPU (`48 GB` Blackwell / `zero-a10g`)**.

## Why In-Process PyTorch on ZeroGPU?

On HuggingFace ZeroGPU, GPU devices are attached dynamically inside `@spaces.GPU(duration=20)` forked worker contexts. External C++ `llama-server` subprocesses cannot access the forked ZeroGPU CUDA handle reliably, whereas in-process PyTorch `bfloat16` inside `@spaces.GPU` receives the full `48 GB` VRAM slice.

- **Direct GGUF-to-PyTorch LoRA Loader**: `app.py` reads the exact `lora_a` / `lora_b` tensors directly from the five `24.3 MiB` `.gguf` cartridges hosted in `niatheelf777/xiaofeiyu-12b-v4` and mounts them as `FrozenFiveSliderLinear` hooks on layers `36..47` (`o_proj`, `gate_proj`, `up_proj`, `down_proj`, `alpha/r = 2.0`).
- **Split-Phase Execution**:
  - **Phase A (`g_voice = 0.0`)**: `<|channel>thought` reasoning, `calc_python` sandbox math, and `send_code_or_doc` fenced code output run at `100%` native Gemma 4 intelligence (`LoRA = 0.0`).
  - **Phase B (`g_voice = 1.0`)**: Only tokens inside `send_message{bubbles:[<|"|>...<|"|>]}` activate the 5-slider linear combination `[w0, w1, w2, w3, w4]`.

## Five Persona Slots

| Slot | Cartridge | Label | English Name |
| :---: | :--- | :--- | :--- |
| `s0` | `s0_xiaofeiyu_backbone.gguf` | **元气日常** | Everyday Warmth |
| `c1` | `c1_neko_clingy.gguf` | **甜系软萌** | Sweet & Clingy |
| `c2` | `c2_older_tease.gguf` | **知性姐姐** | Poised Older Sister |
| `c3` | `c3_brat_mesugaki.gguf` | **傲娇毒舌** | Tsundere Wit |
| `c4` | `c4_fragile_porcelain.gguf` | **安静慢热** | Quiet & Gentle |

## Space Secrets / Environment Variables

- `HF_TOKEN` (optional if `google/gemma-4-12B-it` is gated for your account; accept Gemma license terms on HF first)
- `BASE_MODEL_ID` (default: `google/gemma-4-12B-it`)
- `ADAPTERS_REPO_ID` (default: `niatheelf777/xiaofeiyu-12b-v4`)
