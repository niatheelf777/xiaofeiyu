# `@xiaofeiyu/dsh-plugin`

DeepSeek Harness (`@deepseek-ai/dsh`) plugin, native bubble/code delivery tools, WebUI 5-slider mixer, and loopback OpenAI bridge (`127.0.0.1:18964/v1`) for Xiaofeiyu.

## Dual-Mode DSH Compatibility

1. **Mode 1 — Standard Zero-Plugin OpenAI API (`:8081/v1`)**: Any stock DSH profile (including `标准模式`) or OpenAI-compatible client can talk directly to `pc2_split_phase_server.py` without loading `tool-xiaofeiyu`. Phase B bubbles and Phase A code blocks unwrap cleanly into `assistant.content` while multi-hop filesystem, shell, and web tools execute normally.
2. **Mode 2 — Full `@xiaofeiyu/dsh-plugin` Experience (`小肥鱼 (Xiaofeiyu v4)` preset)**:
   - **Collision-Free Native Delivery Tools (`@xiaofeiyu/dsh-plugin/tools`)**: Registers `xiaofeiyu_send_bubbles` and `xiaofeiyu_send_code` via `@deepseek-ai/dsh-tools`, leaving DSH's subagent `send_message` (`agent_send_message`), Bilibili, web, and filesystem tools untouched.
   - **Tactile IM Bubbles & Code Cards (`@xiaofeiyu/dsh-plugin/client`)**: Renders multi-bubble chat bursts, active 5-slider weight badges, and copyable code/diff cards permanently in the main conversation flow (`conversation.chat.turnTail`) plus compact trajectory rows (`tool.call.toolview`).
   - **Zero-Extra-GPU Settlement**: Terminal bubble/code delivery turns settle locally in `<1 ms` with zero extra GPU requests and zero duplicate assistant prose.
   - **Live 5-Slider Header Mixer**: Adds a real-time 5-slider popover (`conversation.session.header.utilities`) synced to `http://127.0.0.1:18964/sliders` with $L_1 \le 1.15$ budget clamping across both Unified (`12B` / `26B-A4B`) and `E4B` Translator modes.

## Install & Verify

```bash
# Preview changes without modifying your DSH home
node packages/dsh-plugin/scripts/install-dsh.mjs --dry-run --profile web

# Install into DSH web profile (default upstream http://127.0.0.1:8081/v1)
node packages/dsh-plugin/scripts/install-dsh.mjs --apply --profile web --upstream-base http://127.0.0.1:8081/v1

# Restore your original DSH profile byte-for-byte
node packages/dsh-plugin/scripts/install-dsh.mjs --uninstall --profile web

# Rebuild WebUI client bundle & run unit tests
npm --prefix packages/dsh-plugin test
```
