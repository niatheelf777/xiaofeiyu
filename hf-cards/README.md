# Xiaofeiyu HuggingFace Model Cards (`niatheelf777`)

This directory holds the HuggingFace model cards for the Xiaofeiyu open-source Split-Phase persona & consumer quantization family:

| Repository | Base Model | Included GGUFs & 5-Slot Cartridges | Hardware Fit |
| :--- | :--- | :--- | :--- |
| [`niatheelf777/xiaofeiyu-12b-v4`](https://huggingface.co/niatheelf777/xiaofeiyu-12b-v4) | `google/gemma-4-12B-it` | `gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf` (`7.72 GiB`) + 5 `L36..47` `.gguf` voice cartridges | `RTX 3060 / 4070 12GB` (`~8.9 GiB` VRAM) |
| [`niatheelf777/xiaofeiyu-26b-a4b`](https://huggingface.co/niatheelf777/xiaofeiyu-26b-a4b) | `google/gemma-4-26B-A4B-it` | `UD-Q4_K_XL-xfy.gguf` (`13.36 GiB`), `UD-IQ4_XS-xfy.gguf` (`13.09 GiB`), `UD-IQ3_M-xfy.gguf` (`9.03 GiB`) + 5 `v2` `.gguf` cartridges | `RTX 4060 Ti 16GB` (`~14.4–14.6 GiB`) & `RTX 3060 / 4070 12GB` (`~10.3 GiB`) |
| [`niatheelf777/xiaofeiyu-translator-e4b`](https://huggingface.co/niatheelf777/xiaofeiyu-translator-e4b) | `google/gemma-4-E4B-it` | 5 `L22..41` `.gguf` voice translator cartridges (`302 ms` median sidecar rewrite) | Any `6GB+` GPU (`~5.3 GiB` w/ `E4B Q8_0`) |

Use `scripts/publish_hf_release.py` to validate adapters, run `audit_zero_leak.py`, and publish to HuggingFace using `HF_TOKEN`.
