# `@xiaofeiyu/dsh-plugin`

DSH (`@deepseek-ai/dsh`) plugin and loopback OpenAI bridge for Xiaofeiyu:

1. **Unified Mode & E4B Translator Mode**: Switch seamlessly between direct unified Split-Phase servers (`xiaofeiyu-26b-a4b` / `xiaofeiyu-v4`) and the `@xiaofeiyu/openai-voice-proxy` E4B translator sidecar (`xiaofeiyu-translator-e4b-v1`).
2. **Real-Time 5-Slider Mixing**: Projects `xiaofeiyu` + `neko_clingy`, `older_tease`, `brat_mesugaki`, `fragile_porcelain` with $L_1 \le 1.15$ budget clamping.
3. **`send_message` & Proxy Schema Remapping**: Remaps external subagent `send_message` <-> `agent_send_message` and proxy-restricted parameter keys (`file` <-> `target_path`, `bytes` <-> `byte_count`, etc.) losslessly, and strips orphaned non-escalating `justification` fields while leaving Bilibili (`dsh-bili-eyes`), web (`dsh-tool-web`), and filesystem tools intact.
4. **Zero-Config `llm-pi-ai` Route & Engineering Preset**: Automatically registers the `xiaofeiyu` OpenAI-completions provider (`http://127.0.0.1:18964/v1`) and the `xiaofeiyu` agent preset (`persona` + `tool-fs` + `tool-fs-search` + `tool-pwsh`/`tool-bash` + `tool-todo` + `tool-web` + `compaction`, without subagent `send_message` collisions).

## Usage

```bash
# Preview install into DSH web profile
node scripts/install-dsh.mjs --dry-run --profile web

# Apply install
node scripts/install-dsh.mjs --apply --profile web

# Restore original profile byte-for-byte
node scripts/install-dsh.mjs --uninstall --profile web

# Run unit tests
npm test
```
