# Xiaofeiyu · Split-Phase 5-Slider Persona Mixing & Universal Voice Proxy

[中文 README](README.md) · [Live Demo](https://xfy.tospeak.top)

Xiaofeiyu is a low-invasion spoken IM voice engine for Gemma 4: reasoning, code generation, and tool calls execute at **zero LoRA (`LoRA = 0.0`)**, while lightweight 5-slider persona adapters activate exclusively inside `<|"|>` speech-bubble delimiters—delivering authentic multi-bubble human chat without sacrificing intelligence.

Live Demo: [https://xfy.tospeak.top](https://xfy.tospeak.top)

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

## Consumer GPU Guide & Three Deployment Modes

![Consumer 12GB, 16GB, 24GB, and Apple Silicon deployment guide](assets/showcase/consumer-gpu-guide.png)

1. **`12B` Dense Flagship (`Xiaofeiyu-12B-v4`, `9.12 GiB` `UD-Q4_K_XL` fits `RTX 3060/4070 12GB`)**:
   ```bash
   python server/launch_local.py --preset 12b
   python server/pc2_split_phase_server.py --llama-base http://127.0.0.1:18082 --port 8081
   ```
2. **`E4B` Universal Voice Proxy (`302 ms` median latency, `0%` code alteration)**: Keep DeepSeek / Claude / GPT / local Llama as the upstream brain; run `packages/openai-voice-proxy` to translate conversational bubbles while passing code blocks and tool calls verbatim.
3. **`26B-A4B` MoE High-Throughput Preset (`192–233 tok/s`)**:
   ```bash
   python server/launch_local.py --preset 26b-a4b --dry-run
   python server/pc2_split_phase_server.py --llama-base http://127.0.0.1:18081 --port 8081
   ```

---

## Repository Layout

- [`server/`](server/) — Split-phase `llama-server` proxy, OpenAI tool bridge, launcher (`pytest server/test_server.py -v`)
- [`packages/openai-voice-proxy/`](packages/openai-voice-proxy/) — Zero-dep Node OpenAI voice translator proxy (`npm test`)
- [`packages/dsh-plugin/`](packages/dsh-plugin/) — DeepSeek Harness (`DSH`) plugin & reversible installer (`npm test`)
- [`web/`](web/) — Paper-ink 2D Personality Orb Web UI + Edge proxy (`npm test`)
- [`train/`](train/) — Split-phase SFT, residual cartridges, anti-smuggle DPO, Online DPO miner, 5-slot GGUF exporter (`pytest train/test_train.py -v`)
- [`hf-cards/`](hf-cards/) — HuggingFace model cards (`12b-v4`, `26b-a4b`, `translator-e4b`)
- [`scripts/`](scripts/) — HuggingFace 5-slot adapter & model card publisher (`publish_hf_release.py`)

## License

Code is licensed under **Apache-2.0** ([`LICENSE`](LICENSE)). Gemma 4 base weights and derivative adapters are subject to the Google [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
