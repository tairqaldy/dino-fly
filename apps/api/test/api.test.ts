import { createInitialState, inputToBits, score, step } from "@dino-fly/dino-core";
import { makeEnvelope } from "@dino-fly/protocol";
import { describe, expect, it } from "vitest";
import { createApp, type Deps, ingestWorkerMessage } from "../src/app.js";
import { buildLeaderboard, cleanName, issueRunToken, RateLimiter, TOKEN_TTL_MS, validateRun, verifyRunToken } from "../src/core.js";
import { MemoryBlobStore, MemoryRepo } from "../src/repo.js";

/** A legit run: jump every 41 frames until the crash; returns what a browser would submit. */
function playRun(seed: number): { actions: [number, number][]; frames: number; score: number } {
  let s = createInitialState(seed);
  const actions: [number, number][] = [];
  let bits = 0;
  while (!s.crashed && s.frame < 5000) {
    const input = { jump: s.frame % 41 < 6, duck: false };
    const b = inputToBits(input);
    if (b !== bits) {
      actions.push([s.frame, b]);
      bits = b;
    }
    s = step(s, input);
  }
  return { actions, frames: s.frame, score: score(s) };
}

function makeDeps(): Deps & { repo: MemoryRepo; blobs: MemoryBlobStore; changed: number } {
  const d = { repo: new MemoryRepo(), blobs: new MemoryBlobStore(), runSecret: "test-secret-0123456789", flyOnline: () => false, changed: 0 } as Deps & {
    repo: MemoryRepo;
    blobs: MemoryBlobStore;
    changed: number;
  };
  d.onLeaderboardChanged = () => {
    d.changed += 1;
  };
  return d;
}

describe("run tokens", () => {
  it("round-trip, bind the seed, expire, and cannot be forged", () => {
    const t = issueRunToken("s3cret", 100000007, 1_000_000);
    expect(verifyRunToken("s3cret", t, 1_000_500)).toMatchObject({ seed: 100000007, issuedAt: 1_000_000 });
    expect(verifyRunToken("other", t, 1_000_500)).toBeNull();
    expect(verifyRunToken("s3cret", t, 1_000_000 + TOKEN_TTL_MS + 1)).toBeNull();
    expect(verifyRunToken("s3cret", t, 999_999)).toBeNull();
    expect(verifyRunToken("s3cret", t.replace(/^100000007/, "100000008"), 1_000_500)).toBeNull();
    expect(verifyRunToken("s3cret", "garbage", 1)).toBeNull();
    expect(verifyRunToken("s3cret", `1.5.x.y`, 1)).toBeNull();
  });
});

describe("validation by deterministic replay", () => {
  it("accepts a real run and returns the replayed score", () => {
    const run = playRun(100000003);
    const v = validateRun(100000003, run.actions, run.frames);
    expect(v).toMatchObject({ ok: true, run: { score: run.score, frames: run.frames, crashed: true } });
  });

  it("a forged log cannot claim more than the engine gives it", () => {
    const v = validateRun(100000003, [], 100_000); // claims to have survived 100k frames without pressing anything
    expect(v.ok && v.run.frames < 400 && v.run.crashed).toBe(true);
  });

  it("rejects malformed input and runs faster than real time", () => {
    expect(validateRun(1, "nope", 10).ok).toBe(false);
    expect(validateRun(1, [[5, 1], [5, 0]], 10).ok).toBe(false);
    expect(validateRun(1, [[0, 9]], 10).ok).toBe(false);
    expect(validateRun(1, [[0, 1, 2]], 10).ok).toBe(false);
    expect(validateRun(1, [], 0).ok).toBe(false);
    expect(validateRun(1, [], 1e9).ok).toBe(false);
    expect(validateRun(1, [], 6000, 10_000)).toEqual({ ok: false, reason: "run is faster than real time" });
    expect(validateRun(1, [], 600, 10_000).ok).toBe(true);
  });
});

describe("leaderboard", () => {
  it("keeps each player's best, sorts, and computes the fly's percentile among all human runs", () => {
    const lb = buildLeaderboard(
      [
        { name: "ann", score: 50 },
        { name: "ann", score: 300 },
        { name: "bob", score: 80 },
        { name: "cy", score: 120 },
      ],
      [
        { generation: 1, bestScore: 100 },
        { generation: 0, bestScore: 60 },
      ],
    );
    expect(lb.humans).toEqual([
      { name: "ann", score: 300 },
      { name: "cy", score: 120 },
      { name: "bob", score: 80 },
    ]);
    expect(lb.fly.map((f) => f.generation)).toEqual([0, 1]);
    expect(lb.flyPercentile).toBe(50); // generation 1's best (100) beats 2 of 4 human runs
    expect(buildLeaderboard([], []).flyPercentile).toBeUndefined();
  });

  it("sanitises names and rate-limits", () => {
    expect(cleanName("  <b>Tair</b> 🦖 ")).toBe("bTairb");
    expect(cleanName(42)).toBe("anonymous");
    expect(cleanName("x".repeat(99))).toHaveLength(24);
    const rl = new RateLimiter(2, 1000);
    expect([rl.allow("a", 0), rl.allow("a", 1), rl.allow("a", 2), rl.allow("b", 2), rl.allow("a", 1500)]).toEqual([true, true, false, true, true]);
  });
});

describe("HTTP API", () => {
  it("start → submit → leaderboard → duplicate rejected", async () => {
    const deps = makeDeps();
    const app = createApp(deps);
    const issued = (await (await app.request("/api/runs/start", { method: "POST" })).json()) as { seed: number; runToken: string };
    expect(verifyRunToken(deps.runSecret, issued.runToken)?.seed).toBe(issued.seed);
    // the player then needs real time to play: a token issued a minute ago stands in for that
    const started = { seed: issued.seed, runToken: issueRunToken(deps.runSecret, issued.seed, Date.now() - 60_000) };
    const run = playRun(started.seed);
    const tooFast = JSON.stringify({ seed: issued.seed, runToken: issued.runToken, name: "bot", actions: run.actions, frames: run.frames });
    expect((await app.request("/api/runs", { method: "POST", body: tooFast })).status).toBe(400);
    const body = JSON.stringify({ seed: started.seed, runToken: started.runToken, name: "tair", actions: run.actions, frames: run.frames, score: 999999 });
    const res = await app.request("/api/runs", { method: "POST", body });
    expect(res.status).toBe(200);
    expect(await res.json()).toMatchObject({ accepted: true, score: run.score, rank: 1 }); // the claimed 999999 is ignored
    expect(deps.changed).toBe(1);
    expect((await app.request("/api/runs", { method: "POST", body })).status).toBe(409);
    const lb = (await (await app.request("/api/leaderboard")).json()) as { humans: unknown[] };
    expect(lb.humans).toEqual([{ name: "tair", score: run.score }]);
    expect((await (await app.request("/health")).json()) as object).toEqual({ ok: true, flyOnline: false });
  });

  it("rejects bad tokens, seeds that do not match the token, and bad JSON", async () => {
    const app = createApp(makeDeps());
    const started = (await (await app.request("/api/runs/start", { method: "POST" })).json()) as { seed: number; runToken: string };
    const post = (b: unknown) => app.request("/api/runs", { method: "POST", body: typeof b === "string" ? b : JSON.stringify(b) });
    expect((await post({ seed: started.seed + 1, runToken: started.runToken, actions: [], frames: 10 })).status).toBe(400);
    expect((await post({ seed: started.seed, runToken: "x.y.z.w", actions: [], frames: 10 })).status).toBe(400);
    expect((await post("{not json")).status).toBe(400);
    expect((await post({ seed: started.seed, runToken: started.runToken, actions: [[3, 7]], frames: 10 })).status).toBe(400);
  });

  it("serves the fly's recorded run as a ghost once the worker has reported it (re-validated by replay)", async () => {
    const deps = makeDeps();
    const app = createApp(deps);
    expect((await app.request("/api/ghost")).status).toBe(404);
    const run = playRun(100000009);
    const msg = makeEnvelope("fly.run_end", { seed: 100000009, generation: 0, score: run.score, frames: run.frames, cleared: 0, deathType: 0, jumps: 0, actions: run.actions });
    expect(await ingestWorkerMessage(deps, JSON.stringify(msg))).toMatchObject({ leaderboardChanged: true });
    const forged = makeEnvelope("fly.run_end", { ...msg.payload, score: run.score + 500 });
    expect(await ingestWorkerMessage(deps, JSON.stringify(forged))).toEqual({ broadcast: null, leaderboardChanged: false });
    const ghost = (await (await app.request("/api/ghost?seed=100000009")).json()) as { score: number; actions: unknown };
    expect(ghost).toMatchObject({ seed: 100000009, generation: 0, score: run.score, actions: run.actions });
    expect((await app.request("/api/ghost?seed=5")).status).toBe(404);
    expect((await app.request("/api/ghost?seed=abc")).status).toBe(400);
    expect(await ingestWorkerMessage(deps, "junk")).toEqual({ broadcast: null, leaderboardChanged: false });
    const dopa = JSON.stringify(makeEnvelope("dopamine.event", { kind: "reward", magnitude: 1, source: "hardware" }));
    expect((await ingestWorkerMessage(deps, dopa)).broadcast).toBe(dopa);
    expect(deps.repo.dopamine).toHaveLength(1);
  });
});
