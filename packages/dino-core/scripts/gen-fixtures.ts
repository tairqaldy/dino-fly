/**
 * Generates the committed golden fixtures that the Python port must reproduce byte-for-byte.
 * CI re-runs this script and fails if `git diff` is non-empty.
 *
 *   pnpm --filter @dino-fly/dino-core gen-fixtures
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { ENGINE_VERSION } from "../src/engine.js";
import { summarize } from "../src/replay.js";
import { nextU32, randInt, seedToState } from "../src/rng.js";
import { CHECKPOINT_EVERY, constantsSha256, HASH_CHARS, recordRun } from "./hash.js";
import { duckPattern, holdJump, never, oracle, type Policy, periodic, seededRandom } from "./policies.js";

const fixturesDir = new URL("../fixtures/", import.meta.url);
mkdirSync(fixturesDir, { recursive: true });

function writeJson(name: string, value: unknown, pretty: boolean): void {
  const text = pretty ? JSON.stringify(value, null, 2) : JSON.stringify(value);
  writeFileSync(new URL(name, fixturesDir), `${text}\n`, "utf8");
}

function rngVectors(): unknown {
  const seeds = [0, 1, 42, 2026, 123456789, 2147483647, 2147483648, 4294967295, -1];
  return {
    description:
      "mulberry32 cross-language vectors: 16 u32 draws, then 16 randInt(lo, hi) draws from the same stream",
    seeds: seeds.map((seed, i) => {
      let state = seedToState(seed);
      const u32: number[] = [];
      for (let k = 0; k < 16; k++) {
        const draw = nextU32(state);
        state = draw.state;
        u32.push(draw.value);
      }
      const lo = -3 * i;
      const hi = 1000 * (i + 1);
      const values: number[] = [];
      for (let k = 0; k < 16; k++) {
        const draw = randInt(state, lo, hi);
        state = draw.state;
        values.push(draw.value);
      }
      return { seed, u32, randInt: { lo, hi, values } };
    }),
  };
}

export const RUNS: { name: string; seed: number; policy: () => Policy; maxFrames: number }[] = [
  { name: "never-1", seed: 1, policy: () => never, maxFrames: 2000 },
  { name: "hold-jump-2", seed: 2, policy: () => holdJump, maxFrames: 3000 },
  { name: "periodic-37-3", seed: 3, policy: () => periodic(37, 6), maxFrames: 3000 },
  { name: "periodic-61-2026", seed: 2026, policy: () => periodic(61, 25), maxFrames: 3000 },
  { name: "duck-pattern-4", seed: 4, policy: () => duckPattern, maxFrames: 3000 },
  { name: "random-5", seed: 5, policy: () => seededRandom(55), maxFrames: 3000 },
  { name: "random-6", seed: 6, policy: () => seededRandom(66), maxFrames: 3000 },
  { name: "random-7", seed: 4294967295, policy: () => seededRandom(77), maxFrames: 3000 },
  { name: "oracle-11", seed: 11, policy: () => oracle, maxFrames: 12000 },
  { name: "oracle-12", seed: 12, policy: () => oracle, maxFrames: 12000 },
  { name: "oracle-13", seed: 13, policy: () => oracle, maxFrames: 12000 },
];

function gameFixtures(): unknown {
  return {
    description:
      "Golden runs of dino-core. `hashes` = per-frame sha256(canonical state string) truncated to `hashChars` hex " +
      "chars, concatenated; `checkpoints` = full canonical integer states every `checkpointEvery` frames and at the " +
      "crash. Replay `actions` ([stepIndex, bits], bits 1 = jump, 2 = duck) to reproduce them.",
    engineVersion: ENGINE_VERSION,
    constantsSha256: constantsSha256(),
    hashChars: HASH_CHARS,
    checkpointEvery: CHECKPOINT_EVERY,
    runs: RUNS.map((r) => {
      const rec = recordRun(r.seed, r.policy(), r.maxFrames);
      return {
        name: r.name,
        seed: r.seed,
        maxFrames: r.maxFrames,
        summary: summarize(r.seed, rec.final),
        actions: rec.actions,
        checkpoints: rec.checkpoints,
        hashes: rec.hashes,
      };
    }),
  };
}

writeJson("rng_vectors.json", rngVectors(), true);
const games = gameFixtures() as { runs: { name: string; summary: { frames: number; score: number; cleared: number } }[] };
writeJson("game_runs.json", games, false);
for (const r of games.runs) {
  console.log(`${r.name.padEnd(20)} frames=${r.summary.frames} score=${r.summary.score} cleared=${r.summary.cleared}`);
}
console.log("fixtures written to", fixturesDir.pathname);
