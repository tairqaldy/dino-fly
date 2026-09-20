/**
 * Generates the committed golden fixtures that the Python port must reproduce byte-for-byte.
 * CI re-runs this script and fails if `git diff` is non-empty.
 *
 *   pnpm --filter @dino-fly/dino-core gen-fixtures
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { nextU32, randInt, seedToState } from "../src/rng.js";

const fixturesDir = new URL("../fixtures/", import.meta.url);
mkdirSync(fixturesDir, { recursive: true });

function writeJson(name: string, value: unknown): void {
  writeFileSync(new URL(name, fixturesDir), `${JSON.stringify(value, null, 2)}\n`, "utf8");
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

writeJson("rng_vectors.json", rngVectors());
console.log("fixtures written to", fixturesDir.pathname);
