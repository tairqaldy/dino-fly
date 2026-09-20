/**
 * Process entry: HTTP (Hono) + WebSocket hub on one port.
 *   /ws                browsers: receive the live fly feed + leaderboard updates (read-only)
 *   /worker?token=…    the brain worker on the laptop connects OUTBOUND to here and pushes fly.* messages
 * When no worker is connected the site degrades to ghost mode (recorded fly runs) — nothing breaks.
 */
import { timingSafeEqual } from "node:crypto";
import type { IncomingMessage } from "node:http";
import { serve } from "@hono/node-server";
import { makeEnvelope } from "@dino-fly/protocol";
import pg from "pg";
import { WebSocket, WebSocketServer } from "ws";
import { createApp, type Deps, ingestWorkerMessage, leaderboard } from "./app.js";
import { type BlobStore, DiskBlobStore, MemoryRepo, PgRepo, R2BlobStore, type Repo } from "./repo.js";

const env = process.env;
const port = Number(env.PORT ?? 8787);
const workerToken = env.WORKER_TOKEN ?? "";
const runSecret = env.RUN_TOKEN_SECRET ?? "";
if (env.NODE_ENV === "production" && (workerToken.length < 16 || runSecret.length < 16)) {
  throw new Error("WORKER_TOKEN and RUN_TOKEN_SECRET (>= 16 chars) are required in production");
}

const repo: Repo = env.DATABASE_URL ? new PgRepo(new pg.Pool({ connectionString: env.DATABASE_URL, max: 5 })) : new MemoryRepo();
const blobs: BlobStore =
  env.R2_ACCOUNT_ID && env.R2_ACCESS_KEY_ID && env.R2_SECRET_ACCESS_KEY && env.R2_BUCKET
    ? new R2BlobStore({ accountId: env.R2_ACCOUNT_ID, accessKeyId: env.R2_ACCESS_KEY_ID, secretAccessKey: env.R2_SECRET_ACCESS_KEY, bucket: env.R2_BUCKET })
    : new DiskBlobStore(env.BLOB_DIR ?? "./storage");

const browsers = new Set<WebSocket>();
let worker: WebSocket | null = null;
let lastHello: string | null = null;

function broadcast(data: string): void {
  for (const ws of browsers) {
    if (ws.readyState === WebSocket.OPEN && ws.bufferedAmount < 1_000_000) ws.send(data);
  }
}

async function pushLeaderboard(): Promise<void> {
  broadcast(JSON.stringify(makeEnvelope("leaderboard.update", await leaderboard({ repo }))));
}

const deps: Deps = {
  repo,
  blobs,
  runSecret: runSecret || "dev-only-run-secret",
  flyOnline: () => worker !== null,
  onLeaderboardChanged: () => void pushLeaderboard(),
};

const server = serve({ fetch: createApp(deps).fetch, port });
const wss = new WebSocketServer({ noServer: true });

function tokenOk(given: string): boolean {
  const a = Buffer.from(given);
  const b = Buffer.from(workerToken || "dev-only-worker-token");
  return a.length === b.length && timingSafeEqual(a, b);
}

server.on("upgrade", (req: IncomingMessage, socket, head) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  if (url.pathname === "/ws") {
    wss.handleUpgrade(req, socket, head, (ws) => {
      browsers.add(ws);
      ws.send(JSON.stringify(makeEnvelope("fly.status", { online: worker !== null })));
      if (lastHello && worker) ws.send(lastHello);
      ws.on("close", () => browsers.delete(ws));
      ws.on("message", () => undefined); // browsers are read-only
    });
  } else if (url.pathname === "/worker" && tokenOk(url.searchParams.get("token") ?? "")) {
    wss.handleUpgrade(req, socket, head, (ws) => {
      worker?.close();
      worker = ws;
      broadcast(JSON.stringify(makeEnvelope("fly.status", { online: true })));
      ws.on("message", (data) => {
        const raw = String(data);
        void ingestWorkerMessage(deps, raw).then((r) => {
          if (r.broadcast) {
            if (raw.includes('"fly.hello"')) lastHello = raw;
            broadcast(r.broadcast);
          }
          if (r.leaderboardChanged) void pushLeaderboard();
        });
      });
      ws.on("close", () => {
        if (worker === ws) {
          worker = null;
          broadcast(JSON.stringify(makeEnvelope("fly.status", { online: false })));
        }
      });
    });
  } else {
    socket.destroy();
  }
});

console.log(`[api] listening on :${port} · db=${env.DATABASE_URL ? "postgres" : "memory"} · blobs=${blobs.constructor.name}`);
