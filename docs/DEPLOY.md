# Deployment

Three places: the **API** (Railway: Node + Postgres), the **static web app** (Cloudflare Pages or GitHub Pages) and
the **brain worker** (Tair's laptop, the only machine with a GPU). The laptop never accepts inbound connections: the
worker dials out to the API. When the laptop is off the public site keeps working in *ghost mode* (recorded fly runs).

## Current state (2026-09-21)

| Piece | State | URL / note |
|---|---|---|
| Web app | **deployed** by GitHub Actions (`.github/workflows/pages.yml`) | <https://tairqaldy.github.io/dino-fly/> |
| Railway project `dino-fly` | created; **Postgres running**; `api` service created with `WORKER_TOKEN`, `RUN_TOKEN_SECRET`, `DATABASE_URL`, `PORT`, `NODE_ENV` set | <https://railway.com/project/84090b88-674e-4b22-8a63-fb34c0137d6e> |
| API deploy | **FAILED at "scheduling build" with no build log** (two attempts via `railway up`). The same image builds and runs locally against Postgres. Looks like an account/plan-level refusal rather than a code problem — see "Finish the API deploy" below | intended URL: <https://api-production-dad9.up.railway.app> |
| Cloudflare Pages / R2 / Turnstile / Tunnel | not set up (no Cloudflare credentials on this machine) | steps below |
| GitHub Pages build variables | `VITE_API_URL`, `VITE_FLY_WS` point at the Railway URL above | `gh variable list` |

The generated `WORKER_TOKEN` / `RUN_TOKEN_SECRET` are in the git-ignored `.env` at the repository root.

## 1. Local everything (works today)

```bash
docker compose -f infra/docker-compose.yml up -d                     # Postgres on :5433
export DATABASE_URL=postgres://dinofly:dinofly@localhost:5433/dinofly
pnpm install
pnpm --filter @dino-fly/api migrate && pnpm --filter @dino-fly/api dev   # API + WebSocket hub on :8787
cd brain && uv sync --extra gpu && uv run --no-sync flybrain download-data
uv run --no-sync flybrain worker --api ws://localhost:8787/worker --token dev-only-worker-token   # the fly, live on ws://localhost:8765
pnpm --filter @dino-fly/web dev                                       # http://localhost:5173  (lab page enabled in dev)
```

Without `DATABASE_URL` the API uses an in-memory store; without R2 variables it writes action logs to `./storage`.

## 2. API on Railway

```bash
railway login                       # once
railway link                        # project dino-fly, service api
railway up --service api            # builds infra/api.Dockerfile (see railway.toml); runs migrations, then the server
railway domain --service api
curl https://<domain>/health        # {"ok":true,"flyOnline":false}
```

Variables (`railway variables --service api`):

| Name | Value |
|---|---|
| `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` |
| `WORKER_TOKEN` | ≥ 16 random chars; the laptop worker must present it |
| `RUN_TOKEN_SECRET` | ≥ 16 random chars; signs human run tokens |
| `NODE_ENV` | `production` |
| `PORT` | `8787` |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` | optional; without them logs go to the container disk (ephemeral on Railway!) |

**Finish the API deploy.** Open the build-log link printed by `railway up` in the dashboard — the CLI shows nothing
after "scheduling build". If it is a plan limit, upgrade or free a service slot and run `railway up --service api`
again; no code change should be needed (`docker build -f infra/api.Dockerfile .` succeeds locally).

## 3. Web app

**GitHub Pages (live):** every push to `main` that touches `apps/web`, `packages` or `docs` rebuilds and deploys.
Repository variables `VITE_API_URL` and `VITE_FLY_WS` are baked in at build time.

**Cloudflare Pages (documented target):**

```bash
pnpm --filter @dino-fly/web build
npx wrangler login
npx wrangler pages project create dino-fly --production-branch main
VITE_API_URL=https://<api-domain> VITE_FLY_WS=wss://<api-domain>/ws pnpm --filter @dino-fly/web build
npx wrangler pages deploy apps/web/dist --project-name dino-fly
```

R2: `npx wrangler r2 bucket create dino-fly`, create an S3 API token in the Cloudflare dashboard and put the four
`R2_*` variables on the Railway service. Turnstile is not wired into the submit form yet.

## 4. The laptop worker

```bash
cd brain
uv run --no-sync flybrain worker --api wss://<api-domain>/worker --token "$WORKER_TOKEN"
```

Windows autostart: Task Scheduler → "At log on" → `powershell -Command "cd C:\path\to\dino-fly\brain; uv run --no-sync flybrain worker --api wss://<api-domain>/worker"` with `WORKER_TOKEN` as a user environment variable.
WSL2 / Linux (systemd user unit `~/.config/systemd/user/dino-fly-worker.service`):

```ini
[Service]
WorkingDirectory=%h/dino-fly/brain
EnvironmentFile=%h/dino-fly/.env
ExecStart=/usr/bin/env uv run --no-sync flybrain worker --api wss://<api-domain>/worker
Restart=always
[Install]
WantedBy=default.target
```

The public "live fly" exists only while this process runs; the site shows `brain offline` otherwise and serves ghosts.
Optional: expose the local dashboard feed through a Cloudflare Tunnel (`infra/cloudflared.example.yml`).

## 5. CI

`.github/workflows/ci.yml`: TypeScript lint + types + tests + golden-fixture regeneration check; Python ruff + pytest on
the synthetic connectome (no GPU, no data) + docs-in-sync checks. No secrets are used in CI. A firmware compile job is
not set up yet (PlatformIO is not installed on the dev machine, so the firmware has not been compiled at all).
