/**
 * Persistence. `Repo` is the interface the server talks to; `MemoryRepo` backs tests and offline dev,
 * `PgRepo` is Postgres via Drizzle (schema in ./schema.ts, DDL in ../migrations). Action logs go to a BlobStore.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { and, desc, eq, sql } from "drizzle-orm";
import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import type pg from "pg";
import { brains, dopamineEvents, generations, hardwareDevices, runs } from "./schema.js";

export interface StoredRun {
  agentType: "human" | "fly" | "ghost";
  generation: number | null;
  playerName: string | null;
  seed: number;
  score: number;
  frames: number;
  jumps: number;
  ducks: number;
  cleared: number;
  deathType: number;
  actionLogKey: string;
  nonce: string | null;
  validated: boolean;
}

export interface Repo {
  /** false when the nonce was already used (replayed submission). */
  insertRun(run: StoredRun): Promise<boolean>;
  humanScores(): Promise<{ name: string; score: number }[]>;
  flyBestPerGeneration(): Promise<{ generation: number; bestScore: number }[]>;
  bestFlyRun(seed?: number): Promise<{ seed: number; generation: number; score: number; actionLogKey: string } | null>;
  humanRank(score: number): Promise<number>;
  /** Validated human runs (newest first) for crowd teaching: metadata only, the action log is fetched by key. */
  listHumanRuns(limit: number): Promise<{ seed: number; score: number; frames: number; actionLogKey: string }[]>;
  /** How big the teaching corpus is. Collection is continuous; training from it is always a manual step. */
  countHumanRuns(): Promise<{ runs: number; seeds: number }>;
  recordDopamine(e: { kind: "reward" | "punish"; magnitude: number; source: "game" | "human_button" | "hardware"; seed?: number; frame?: number }): Promise<void>;
  touchDevice(deviceId: string, kind: string, firmware?: string): Promise<void>;
}

// ------------------------------------------------------------------------------------------------ blobs
export interface BlobStore {
  put(key: string, body: string): Promise<void>;
  get(key: string): Promise<string | null>;
}

export class DiskBlobStore implements BlobStore {
  constructor(private readonly root: string) {}
  async put(key: string, body: string): Promise<void> {
    const path = join(this.root, key);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, body, "utf8");
  }
  async get(key: string): Promise<string | null> {
    try {
      return await readFile(join(this.root, key), "utf8");
    } catch {
      return null;
    }
  }
}

/** Cloudflare R2 through its S3-compatible API. */
export class R2BlobStore implements BlobStore {
  private client: Promise<import("@aws-sdk/client-s3").S3Client>;
  constructor(
    private readonly cfg: { accountId: string; accessKeyId: string; secretAccessKey: string; bucket: string },
  ) {
    this.client = import("@aws-sdk/client-s3").then(
      ({ S3Client }) =>
        new S3Client({
          region: "auto",
          endpoint: `https://${cfg.accountId}.r2.cloudflarestorage.com`,
          credentials: { accessKeyId: cfg.accessKeyId, secretAccessKey: cfg.secretAccessKey },
        }),
    );
  }
  async put(key: string, body: string): Promise<void> {
    const { PutObjectCommand } = await import("@aws-sdk/client-s3");
    await (await this.client).send(new PutObjectCommand({ Bucket: this.cfg.bucket, Key: key, Body: body, ContentType: "application/json" }));
  }
  async get(key: string): Promise<string | null> {
    const { GetObjectCommand } = await import("@aws-sdk/client-s3");
    try {
      const res = await (await this.client).send(new GetObjectCommand({ Bucket: this.cfg.bucket, Key: key }));
      return (await res.Body?.transformToString()) ?? null;
    } catch {
      return null;
    }
  }
}

export class MemoryBlobStore implements BlobStore {
  readonly data = new Map<string, string>();
  async put(key: string, body: string): Promise<void> {
    this.data.set(key, body);
  }
  async get(key: string): Promise<string | null> {
    return this.data.get(key) ?? null;
  }
}

// ------------------------------------------------------------------------------------------------ memory
export class MemoryRepo implements Repo {
  readonly runs: StoredRun[] = [];
  readonly dopamine: unknown[] = [];
  readonly devices = new Map<string, { kind: string; firmware?: string }>();

  async insertRun(run: StoredRun): Promise<boolean> {
    if (run.nonce !== null && this.runs.some((r) => r.nonce === run.nonce)) return false;
    this.runs.push(run);
    return true;
  }
  async humanScores() {
    return this.runs.filter((r) => r.agentType === "human" && r.validated).map((r) => ({ name: r.playerName ?? "anonymous", score: r.score }));
  }
  async flyBestPerGeneration() {
    const best = new Map<number, number>();
    for (const r of this.runs.filter((x) => x.agentType === "fly")) {
      best.set(r.generation ?? 0, Math.max(best.get(r.generation ?? 0) ?? 0, r.score));
    }
    return [...best.entries()].map(([generation, bestScore]) => ({ generation, bestScore }));
  }
  async bestFlyRun(seed?: number) {
    const pool = this.runs.filter((r) => r.agentType === "fly" && (seed === undefined || r.seed === seed));
    const top = pool.sort((a, b) => b.score - a.score)[0];
    return top ? { seed: top.seed, generation: top.generation ?? 0, score: top.score, actionLogKey: top.actionLogKey } : null;
  }
  async humanRank(score: number) {
    return 1 + this.runs.filter((r) => r.agentType === "human" && r.validated && r.score > score).length;
  }
  async listHumanRuns(limit: number) {
    return this.runs
      .filter((r) => r.agentType === "human" && r.validated)
      .slice(-limit)
      .reverse()
      .map((r) => ({ seed: r.seed, score: r.score, frames: r.frames, actionLogKey: r.actionLogKey }));
  }
  async countHumanRuns() {
    const valid = this.runs.filter((r) => r.agentType === "human" && r.validated);
    return { runs: valid.length, seeds: new Set(valid.map((r) => r.seed)).size };
  }
  async recordDopamine(e: unknown) {
    this.dopamine.push(e);
  }
  async touchDevice(deviceId: string, kind: string, firmware?: string) {
    this.devices.set(deviceId, firmware === undefined ? { kind } : { kind, firmware });
  }
}

// ------------------------------------------------------------------------------------------------ postgres
export class PgRepo implements Repo {
  private db: NodePgDatabase;
  private generationIds = new Map<number, number>();

  constructor(pool: pg.Pool) {
    this.db = drizzle(pool);
  }

  private async generationId(genNumber: number): Promise<number> {
    const cached = this.generationIds.get(genNumber);
    if (cached !== undefined) return cached;
    await this.db.insert(brains).values({ name: "fly-1", connectomeVersion: "flywire783" }).onConflictDoNothing();
    const [brain] = await this.db.select().from(brains).where(eq(brains.name, "fly-1"));
    if (!brain) throw new Error("brain row missing");
    await this.db.insert(generations).values({ brainId: brain.id, genNumber }).onConflictDoNothing();
    const [gen] = await this.db.select().from(generations).where(and(eq(generations.brainId, brain.id), eq(generations.genNumber, genNumber)));
    if (!gen) throw new Error("generation row missing");
    this.generationIds.set(genNumber, gen.id);
    return gen.id;
  }

  async insertRun(run: StoredRun): Promise<boolean> {
    const generationId = run.generation === null ? null : await this.generationId(run.generation);
    const inserted = await this.db
      .insert(runs)
      .values({
        agentType: run.agentType,
        generationId,
        playerName: run.playerName,
        seed: run.seed,
        score: run.score,
        durationFrames: run.frames,
        jumps: run.jumps,
        ducks: run.ducks,
        cleared: run.cleared,
        deathCause: run.deathType,
        actionLogUrl: run.actionLogKey,
        runNonce: run.nonce,
        validated: run.validated,
      })
      .onConflictDoNothing({ target: runs.runNonce })
      .returning({ id: runs.id });
    return inserted.length > 0;
  }

  async humanScores() {
    const rows = await this.db
      .select({ name: runs.playerName, score: runs.score })
      .from(runs)
      .where(and(eq(runs.agentType, "human"), eq(runs.validated, true)));
    return rows.map((r) => ({ name: r.name ?? "anonymous", score: r.score }));
  }

  async flyBestPerGeneration() {
    const rows = await this.db
      .select({ generation: generations.genNumber, bestScore: sql<number>`max(${runs.score})::int` })
      .from(runs)
      .innerJoin(generations, eq(runs.generationId, generations.id))
      .where(eq(runs.agentType, "fly"))
      .groupBy(generations.genNumber);
    return rows;
  }

  async bestFlyRun(seed?: number) {
    const cond = seed === undefined ? eq(runs.agentType, "fly") : and(eq(runs.agentType, "fly"), eq(runs.seed, seed));
    const [row] = await this.db
      .select({ seed: runs.seed, score: runs.score, actionLogKey: runs.actionLogUrl, generation: generations.genNumber })
      .from(runs)
      .leftJoin(generations, eq(runs.generationId, generations.id))
      .where(cond)
      .orderBy(desc(runs.score))
      .limit(1);
    return row ? { seed: row.seed, generation: row.generation ?? 0, score: row.score, actionLogKey: row.actionLogKey } : null;
  }

  async humanRank(score: number) {
    const [row] = await this.db
      .select({ n: sql<number>`count(*)::int` })
      .from(runs)
      .where(and(eq(runs.agentType, "human"), eq(runs.validated, true), sql`${runs.score} > ${score}`));
    return 1 + (row?.n ?? 0);
  }

  async listHumanRuns(limit: number) {
    return this.db
      .select({ seed: runs.seed, score: runs.score, frames: runs.durationFrames, actionLogKey: runs.actionLogUrl })
      .from(runs)
      .where(and(eq(runs.agentType, "human"), eq(runs.validated, true)))
      .orderBy(desc(runs.id))
      .limit(limit);
  }

  async countHumanRuns() {
    const [row] = await this.db
      .select({ runs: sql<number>`count(*)::int`, seeds: sql<number>`count(distinct ${runs.seed})::int` })
      .from(runs)
      .where(and(eq(runs.agentType, "human"), eq(runs.validated, true)));
    return { runs: row?.runs ?? 0, seeds: row?.seeds ?? 0 };
  }

  async recordDopamine(e: { kind: "reward" | "punish"; magnitude: number; source: "game" | "human_button" | "hardware"; seed?: number; frame?: number }) {
    await this.db.insert(dopamineEvents).values({ kind: e.kind, magnitude: e.magnitude, source: e.source, seed: e.seed ?? null, frame: e.frame ?? null });
  }

  async touchDevice(deviceId: string, kind: string, firmware?: string) {
    await this.db
      .insert(hardwareDevices)
      .values({ deviceId, kind, firmwareVersion: firmware ?? null })
      .onConflictDoUpdate({ target: hardwareDevices.deviceId, set: { lastSeen: sql`now()`, firmwareVersion: firmware ?? null } });
  }
}
