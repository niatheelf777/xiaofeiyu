# Xiaofeiyu HuggingFace Model Cards (`niatheelf777`)

This directory holds the clean, reproducible HuggingFace model cards for the Xiaofeiyu open-source adapter suite:

| Repository | Base Model | Scope | Status |
| :--- | :--- | :--- | :--- |
| `niatheelf777/xiaofeiyu-12b-v4` | `google/gemma-4-12B-it` | Flagship 48-module `L36..47` non-QKV 5-slider voice adapters (`v4` + `v4.1` Phase-B DPO) | Verified (`12/12` IQ, `5/5` GEN, $W_1=2.52$) |
| `niatheelf777/xiaofeiyu-translator-e4b` | `google/gemma-4-E4B-it` | Universal 5-slider voice translator sidecar (`L22..41` non-QKV, `80` modules) | Verified (`96.05%` DPO, `302 ms` P50, `0%` smuggle) |
| `niatheelf777/xiaofeiyu-26b-a4b` | `google/gemma-4-26B-A4B-it` | High-throughput MoE Split-Phase 5-slider voice adapters (`s0, c1..c4`) | Experimental MoE Preset |

Use `scripts/publish_hf_release.py` to validate adapters, run `audit_zero_leak.py`, and publish to HuggingFace using `HF_TOKEN`.
