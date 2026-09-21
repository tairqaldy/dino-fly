/**
 * HTTP API (Hono). The WebSocket hub lives in server.ts; this module is pure request handling so tests can call
 * `app.request(...)` without opening a port.
 */
import { parseEnvelope } from "@dino-fly/protocol";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { buildLeaderboard, cleanName, issueRunToken, pickHumanSeed, RateLimiter, validateRun, verifyRunToken } from "./core.js";
import type { BlobStore, Repo } from "./repo.js";

export interface Deps {
  repo: Repo;
  blobs: BlobStore;
  runSecret: string;
  flyOnline: () => boolean;
  onLeaderboardChanged?: () => void;
}

export function createApp(deps: Deps): Hono {
  const app = new Hono();
  const submitLimiter = new RateLimiter(30, 10 * 60 * 1000);
  const startLimiter = new RateLimiter(120, 10 * 60 * 1000);
  const clientKey = (c: { req: { header: (k: string) => string | undefined } }): string =>
    c.req.header("cf-connecting-ip") ?? c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ?? "local";

  app.use("*", cors({ origin: "*", allowMethods: ["GET", "POST", "OPTIONS"] }));

  app.get("/health", (c) => c.json({ ok: true, flyOnline: deps.flyOnline() }));
  // The corpus grows on its own; nothing trains on it automatically — a new generation is always rolled out by hand.
  app.get("/api/status", async (c) => c.json({ flyOnline: deps.flyOnline(), teaching: await deps.repo.countHumanRuns() }));

  app.post("/api/runs/start", (c) => {
    if (!startLimiter.allow(clientKey(c))) return c.json({ error: "rate limited" }, 429);
    const seed = pickHumanSeed();
    return c.json({ seed, runToken: issueRunToken(deps.runSecret, seed) });
  });

  app.post("/api/runs", async (c) => {
    if (!submitLimiter.allow(clientKey(c))) return c.json({ accepted: false, score: 0, reason: "rate limited" }, 429);
    let body: Record<string, unknown>;
    try {
      body = (await c.req.json()) as Record<string, unknown>;
    } catch {
      return c.json({ accepted: false, score: 0, reason: "bad json" }, 400);
    }
    const claims = typeof body.runToken === "string" ? verifyRunToken(deps.runSecret, body.runToken) : null;
    if (!claims || claims.seed !== body.seed) {
      return c.json({ accepted: false, score: 0, reason: "invalid or expired run token" }, 400);
    }
    const checked = validateRun(claims.seed, body.actions, body.frames, Date.now() - claims.issuedAt);
    if (!checked.ok) return c.json({ accepted: false, score: 0, reason: checked.reason }, 400);
    const run = checked.run;
    const key = `runs/human/${claims.seed}-${claims.nonce}.json`;
    await deps.blobs.put(key, JSON.stringify({ seed: run.seed, actions: body.actions, frames: run.frames }));
    const fresh = await deps.repo.insertRun({
      agentType: "human",
      generation: null,
      playerName: cleanName(body.name),
      seed: run.seed,
      score: run.score,
      frames: run.frames,
      jumps: run.jumps,
      ducks: run.ducks,
      cleared: run.cleared,
      deathType: run.deathType,
      actionLogKey: key,
      nonce: claims.nonce,
      validated: true,
    });
    if (!fresh) return c.json({ accepted: false, score: run.score, reason: "this run was already submitted" }, 409);
    deps.onLeaderboardChanged?.();
    const ghost = await deps.repo.bestFlyRun(run.seed);
    return c.json({
      accepted: true,
      score: run.score, // the replayed score, never the client's claim
      rank: await deps.repo.humanRank(run.score),
      ...(ghost ? { beatsFly: run.score > ghost.score } : {}),
    });
  });

  app.get("/api/leaderboard", async (c) => c.json(await leaderboard(deps)));

  // Export for crowd teaching (brain/experiments/crowd_teaching.py): validated human runs with their action logs.
  app.get("/api/runs/human", async (c) => {
    const limit = Math.min(Math.max(Number(c.req.query("limit") ?? 500) || 500, 1), 5000);
    const rows = await deps.repo.listHumanRuns(limit);
    const out = [];
    for (const r of rows) {
      const blob = await deps.blobs.get(r.actionLogKey);
      if (blob) out.push({ seed: r.seed, score: r.score, frames: r.frames, actions: (JSON.parse(blob) as { actions: unknown }).actions });
    }
    return c.json({ runs: out });
  });

  app.get("/api/ghost", async (c) => {
    const seedParam = c.req.query("seed");
    const seed = seedParam === undefined ? undefined : Number(seedParam);
    if (seed !== undefined && !Number.isSafeInteger(seed)) return c.json({ error: "bad seed" }, 400);
    const best = (seed !== undefined ? await deps.repo.bestFlyRun(seed) : null) ?? (seed === undefined ? await deps.repo.bestFlyRun() : null);
    if (!best) return c.json({ error: "no fly run recorded for this seed yet" }, 404);
    const blob = await deps.blobs.get(best.actionLogKey);
    if (!blob) return c.json({ error: "action log missing" }, 404);
    return c.json({ seed: best.seed, generation: best.generation, score: best.score, actions: (JSON.parse(blob) as { actions: unknown }).actions });
  });

  return app;
}

export async function leaderboard(deps: Pick<Deps, "repo">) {
  return buildLeaderboard(await deps.repo.humanScores(), await deps.repo.flyBestPerGeneration());
}

/**
 * Messages arriving from the authenticated brain worker. Fly runs are re-validated by replay as well, so a
 * compromised worker token cannot forge scores either.
 */
export async function ingestWorkerMessage(deps: Deps, raw: string): Promise<{ broadcast: string | null; leaderboardChanged: boolean }> {
  const m = parseEnvelope(raw);
  if (!m) return { broadcast: null, leaderboardChanged: false };
  if (m.type === "fly.run_end") {
    const p = m.payload;
    const checked = validateRun(p.seed, p.actions, Math.max(p.frames, 1));
    if (checked.ok && checked.run.score === p.score) {
      const key = `runs/fly/gen${p.generation}/${p.seed}-${m.ts}.json`;
      await deps.blobs.put(key, JSON.stringify({ seed: p.seed, actions: p.actions, frames: p.frames }));
      await deps.repo.insertRun({
        agentType: "fly",
        generation: p.generation,
        playerName: null,
        seed: p.seed,
        score: checked.run.score,
        frames: checked.run.frames,
        jumps: checked.run.jumps,
        ducks: checked.run.ducks,
        cleared: checked.run.cleared,
        deathType: checked.run.deathType,
        actionLogKey: key,
        nonce: null,
        validated: true,
      });
      return { broadcast: raw, leaderboardChanged: true };
    }
    return { broadcast: null, leaderboardChanged: false };
  }
  if (m.type === "dopamine.event") {
    await deps.repo.recordDopamine(m.payload);
  }
  const relay = m.type === "fly.hello" || m.type === "fly.frame" || m.type === "fly.stats" || m.type === "dopamine.event";
  return { broadcast: relay ? raw : null, leaderboardChanged: false };
}
