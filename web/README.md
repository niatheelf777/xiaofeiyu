# Xiaofeiyu Public Web UI (`web/`)

Warm paper-ink IM web frontend + Edge API proxy featuring the **2D Mood Orb Compass** (`xiaofeiyu_v4` + `neko_clingy`, `older_tease`, `brat_mesugaki`, `fragile_porcelain`), incremental fence-aware bubble streaming, and multi-upstream EWMA load-balancing & cooldown failover.

## Local Development

```bash
# Instant mock mode (no GPU needed)
npm run dev

# Pointing at a live local Split-Phase server (default port 8081)
XFY_UPSTREAM_URLS=http://127.0.0.1:8081/v1/chat/completions npm run dev:live

# Run tests
npm test
```

## Deploy

- **Cloudflare Workers**: Edit `wrangler.jsonc` (`account_id`), set secrets (`XFY_UPSTREAM_URLS`, `XFY_SERVICE_KEY`), and run `npx wrangler deploy`.
- **Vercel Edge**: Deploy `web/` and set `XFY_UPSTREAM_URLS` + `XFY_SERVICE_KEY` in Environment Variables.
