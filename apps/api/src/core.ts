/**
 * Pure logic of the API: run tokens, server-side validation by deterministic replay, leaderboard maths, rate limiting.
 * No I/O here, so everything is unit-tested without a database.
 */
import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { type ActionLog, replay, summarize, validateActionLog } from "@dino-fly/dino-core";
import type { LeaderboardUpdate } from "@dino-fly/protocol";

export const MAX_FRAMES = 200_000; // ≈ 55 minutes of play; far beyond any human run
export const MAX_ACTIONS = 40_000;
export const TOKEN_TTL_MS = 6 * 60 * 60 * 1000;

// ------------------------------------------------------------------------------------------------ run tokens
/** Stateless run token: seed.issuedAt.nonce.signature — the server picks the seed, the client cannot. */
export function issueRunToken(secret: string, seed: number, now: number = Date.now()): string {
  const body = `${seed}.${now}.${randomBytes(9).toString("base64url")}`;
  return `${body}.${sign(secret, body)}`;
}

function sign(secret: string, body: string): string {
  return createHmac("sha256", secret).update(body).digest("base64url");
}

export interface TokenClaims {
  seed: number;
  issuedAt: number;
  nonce: string;
}

export function verifyRunToken(secret: string, token: string, now: number = Date.now()): TokenClaims | null {
  const parts = token.split(".");
  if (parts.length !== 4) return null;
  const [seedS, issuedS, nonce, sig] = parts as [string, string, string, string];
  const expected = Buffer.from(sign(secret, `${seedS}.${issuedS}.${nonce}`));
  const given = Buffer.from(sig);
  if (expected.length !== given.length || !timingSafeEqual(expected, given)) return null;
  const seed = Number(seedS);
  const issuedAt = Number(issuedS);
  if (!Number.isSafeInteger(seed) || !Number.isSafeInteger(issuedAt)) return null;
  if (now < issuedAt || now - issuedAt > TOKEN_TTL_MS) return null;
  return { seed, issuedAt, nonce };
}

/** Human seeds come from a range the fly never trains or is evaluated on (see brain/flybrain/seeds.py). */
export const HUMAN_SEED_BASE = 100_000_000;
export function pickHumanSeed(buckets: number = 50): number {
  return HUMAN_SEED_BASE + Math.floor(Math.random() * buckets);
}

// ------------------------------------------------------------------------------------------------ validation
export interface ValidatedRun {
  seed: number;
  score: number;
  frames: number;
  jumps: number;
  ducks: number;
  cleared: number;
  deathType: number;
  crashed: boolean;
}

export type Validation = { ok: true; run: ValidatedRun } | { ok: false; reason: string };

/**
 * The client's claimed score is ignored: the action log is replayed in the same deterministic engine and the
 * replayed result is what gets stored. `claimedFrames` only bounds the replay (a run that has not crashed by then
 * is accepted as "still alive at that frame", e.g. the player closed the tab).
 */
export function validateRun(seed: number, actions: unknown, claimedFrames: unknown, elapsedMs?: number): Validation {
  if (!Number.isSafeInteger(claimedFrames) || (claimedFrames as number) < 1 || (claimedFrames as number) > MAX_FRAMES) {
    return { ok: false, reason: "bad frame count" };
  }
  if (!Array.isArray(actions) || actions.length > MAX_ACTIONS) {
    return { ok: false, reason: "bad action log" };
  }
  for (const a of actions) {
    if (!Array.isArray(a) || a.length !== 2 || !Number.isSafeInteger(a[0]) || !Number.isSafeInteger(a[1])) {
      return { ok: false, reason: "bad action log entry" };
    }
  }
  try {
    validateActionLog(actions as ActionLog);
  } catch {
    return { ok: false, reason: "action log is not strictly increasing" };
  }
  const frames = claimedFrames as number;
  // A 60 Hz game cannot produce more frames than wall-clock time allows (generous 25 % + 2 s slack for fast displays).
  if (elapsedMs !== undefined && frames > (elapsedMs / 1000) * 60 * 1.25 + 120) {
    return { ok: false, reason: "run is faster than real time" };
  }
  const final = replay(seed, actions as ActionLog, frames);
  const s = summarize(seed, final);
  return {
    ok: true,
    run: { seed, score: s.score, frames: s.frames, jumps: s.jumps, ducks: s.ducks, cleared: s.cleared, deathType: s.deathType, crashed: s.crashed },
  };
}

// ------------------------------------------------------------------------------------------------ leaderboard
export interface ScoreRow {
  name: string;
  score: number;
}

export function buildLeaderboard(
  humanRuns: ScoreRow[],
  flyBestPerGeneration: { generation: number; bestScore: number }[],
  top: number = 25,
): LeaderboardUpdate {
  const bestByName = new Map<string, number>();
  for (const r of humanRuns) {
    bestByName.set(r.name, Math.max(bestByName.get(r.name) ?? 0, r.score));
  }
  const humans = [...bestByName.entries()]
    .map(([name, score]) => ({ name, score }))
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, top);
  const fly = [...flyBestPerGeneration].sort((a, b) => a.generation - b.generation);
  const out: LeaderboardUpdate = { humans, fly };
  const latest = fly.at(-1);
  if (latest && humanRuns.length > 0) {
    out.flyPercentile = (100 * humanRuns.filter((r) => r.score < latest.bestScore).length) / humanRuns.length;
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ rate limiting
/** Sliding-window limiter keyed by client (IP). In-memory: one API instance is enough for this project. */
export class RateLimiter {
  private hits = new Map<string, number[]>();
  constructor(
    private readonly limit: number,
    private readonly windowMs: number,
  ) {}

  allow(key: string, now: number = Date.now()): boolean {
    const recent = (this.hits.get(key) ?? []).filter((t) => now - t < this.windowMs);
    if (recent.length >= this.limit) {
      this.hits.set(key, recent);
      return false;
    }
    recent.push(now);
    this.hits.set(key, recent);
    return true;
  }
}

export function cleanName(raw: unknown): string {
  const s = typeof raw === "string" ? raw : "";
  const cleaned = s.replace(/[^\p{L}\p{N} _.\-]/gu, "").trim().slice(0, 24);
  return cleaned.length > 0 ? cleaned : "anonymous";
}
