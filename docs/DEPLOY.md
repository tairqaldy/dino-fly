# Deployment

Three places: the **API** (Railway: Node + Postgres + WebSocket hub), the **static web app** (Vercel, mirrored on
GitHub Pages) and the **brain worker** (Tair's laptop, the only machine with a GPU). The laptop never accepts inbound
connections: the worker dials out to the API. When the laptop is off the public site keeps working in *ghost mode*
(recorded fly runs).

## Current state (2026-09-21, all live)

| Piece | State | URL / note |
|---|---|---|
| Web app (primary) | **deployed by Vercel** from GitHub on every push to `main`; Vercel Auth off so it is public | <https://flybrain-dino.vercel.app> |
| Web app (mirror) | **deployed** by GitHub Actions (`.github/workflows/pages.yml`) | <https://tairqaldy.github.io/dino-fly/> |
| API + Postgres | **deployed and healthy** on Railway (`/health` → 200) | <https://api-production-dad9.up.railway.app> |
| Railway project `dino-fly` | Postgres + `api` service, `WORKER_TOKEN`, `RUN_TOKEN_SECRET`, `DATABASE_URL`, `PORT`, `NODE_ENV` set | <https://railway.com/project/84090b88-674e-4b22-8a63-fb34c0137d6e> |
| Build vars | `VITE_API_URL`, `VITE_FLY_WS` on both Vercel and GitHub Pages point at the Railway URL | `gh variable list` |
| Cloudflare Pages / R2 / Turnstile | not set up and no longer needed (Vercel + Railway + Postgres cover it); R2 stays optional for action logs | — |

**Why the API deploy used to fail.** Railway ignored `railway.toml`: config-as-code is deprecated there and services
created after 2026-08-28 cannot opt into it, so `railway up` always fell back to the Railpack builder, which stopped
with *"No start command detected"* — the build log was only visible in the dashboard, not in the CLI. The fix is a
`start` script in the **root** `package.json`; `infra/api.Dockerfile` is kept for local/other hosts (D23).

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

**Redeploying the API:** `railway up --service api --ci` from the repository root. Railpack builds the pnpm
workspace and runs the root `start` script, which applies migrations and then starts the server. If a build ever
fails again, the *useful* log is in the Railway dashboard (the CLI stops at "scheduling build").

## 3. Web app

**Vercel (primary, live):** project `dino-fly` in team `tairqaldy-projects`, connected to the GitHub repo, root
directory `apps/web`, install `cd ../.. && pnpm install --frozen-lockfile`, build `pnpm run build`, output `dist`.
Every push to `main` deploys automatically; `VITE_API_URL` / `VITE_FLY_WS` are project environment variables.
Vercel Authentication is **off** (otherwise visitors would hit a login wall). A custom domain would have to be
bought first; `flybrain-dino.vercel.app` is the free one this project owns.

**GitHub Pages (mirror, live):** every push to `main` that touches `apps/web`, `packages` or `docs` rebuilds and
deploys. Repository variables `VITE_API_URL` and `VITE_FLY_WS` are baked in at build time.

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
