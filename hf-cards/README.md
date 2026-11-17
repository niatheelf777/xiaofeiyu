# Xiaofeiyu HuggingFace Model Cards (`niatheelf777`)

This directory holds the HuggingFace model cards for the Xiaofeiyu open-source Split-Phase persona & consumer quantization family:

| Repository | Status & Role | Included Weights & Base Precision | Hardware Fit |
| :--- | :--- | :--- | :--- |
| [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) | **主力打磨版（线上同款首选）** | 5 `L36..47` `.gguf` voice cartridges (pair primarily with `Q8_0` / `BF16`) + custom BF16 `imatrix` `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` (`7.72 GiB`) | `≥ 16 GB` VRAM / `24 GB+` Mac 首选 `Q8_0` (`~12.9 GiB`)；`12 GB` 显卡选特制 `UD-Q4_K_XL-xfy` (`~8.5–8.9 GiB`) |
| [`niatheelf777/xiaofeiyu-translator-e4b`](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) | **通用声带翻译代理（小显存 / 云端外挂）** | 5 `L22..41` `.gguf` voice translator cartridges (`302 ms` median sidecar rewrite) | `6–8 GB` VRAM (`~5.3 GiB` w/ `E4B Q8_0`) paired with DeepSeek / Claude / GPT |
| [`niatheelf777/xiaofeiyu-26b-a4b`](https://huggingface.co/niatheelf777/xiaofeiyu-26b-a4b) | **实验存档（不推荐日常使用）** | `26B-A4B` MoE migration archive (未经过充分日常聊天手感打磨，请直接选用 [`xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4)) | Experimental archive only |

> Note: `xiaofeiyu-26b-a4b` is kept solely as an experimental MoE architecture archive and is **not recommended for daily chat**; please use the battle-tested flagship [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4).

Use `scripts/publish_hf_release.py` to validate adapters, run `audit_zero_leak.py`, and publish to HuggingFace using `HF_TOKEN`.
