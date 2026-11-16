# Xiaofeiyu 2D Persona Orb Web UI (`web/`)

Paper-and-ink IM web frontend and Edge API proxy featuring the **2D Persona Orb Compass** (`xiaofeiyu_v4` + `neko_clingy`, `older_tease`, `brat_mesugaki`, `fragile_porcelain`), incremental fence-aware bubble streaming, copyable code cards, and multi-upstream EWMA load-balancing.

## Run Locally

### Option 1: Built-in Zero-Node Mode (Recommended)

`server/pc2_split_phase_server.py` mounts `web/public/` alongside `/api/status`, `/api/cluster`, and `/api/chat` automatically—no Node.js process required:

```bash
python server/launch_local.py --preset 12b-q4 --auto-download-llama --open
# Opens http://127.0.0.1:8081/ directly
```

### Option 2: Standalone Node Frontend Server (`serve_local.mjs`)

Use `serve_local.mjs` when pointing the UI at a remote or separate `:8081` Split-Phase gateway, or when previewing UI interactions without a GPU:

```bash
# Connect to a local or remote Split-Phase server
node web/serve_local.mjs --api http://127.0.0.1:8081 --port 3000 --open

# Instant offline mock mode (no GPU needed)
node web/serve_local.mjs --mock --open

# Run unit tests
npm --prefix web test
```

## Edge Deployment

- **Cloudflare Workers**: Set `account_id` in `wrangler.jsonc`, configure `XFY_UPSTREAM_URLS` and `XFY_SERVICE_KEY` secrets, and run `npx wrangler deploy`.
- **Vercel Edge**: Deploy `web/` and configure `XFY_UPSTREAM_URLS` and `XFY_SERVICE_KEY` in Environment Variables.
