import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { nextU32, randInt, seedToState } from "../src/rng.js";

/** The widely published float version of mulberry32, used here only as an independent reference. */
function mulberry32Reference(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("mulberry32 (integer API)", () => {
  it("matches the reference float implementation bit-for-bit", () => {
    for (const seed of [0, 1, 42, 123456789, 0x7fffffff, 0x80000000, 0xffffffff]) {
      const ref = mulberry32Reference(seed);
      let state = seedToState(seed);
      for (let i = 0; i < 200; i++) {
        const draw = nextU32(state);
        state = draw.state;
        expect(draw.value / 4294967296).toBe(ref());
      }
    }
  });

  it("keeps state and values as canonical uint32", () => {
    let state = seedToState(-5);
    expect(state).toBe(4294967291);
    for (let i = 0; i < 1000; i++) {
      const draw = nextU32(state);
      state = draw.state;
      for (const x of [draw.state, draw.value]) {
        expect(Number.isInteger(x)).toBe(true);
        expect(x).toBeGreaterThanOrEqual(0);
        expect(x).toBeLessThanOrEqual(0xffffffff);
      }
    }
  });

  it("randInt stays within inclusive bounds and hits both ends", () => {
    let state = seedToState(7);
    const seen = new Set<number>();
    for (let i = 0; i < 2000; i++) {
      const draw = randInt(state, -2, 3);
      state = draw.state;
      expect(draw.value).toBeGreaterThanOrEqual(-2);
      expect(draw.value).toBeLessThanOrEqual(3);
      seen.add(draw.value);
    }
    expect([...seen].sort((x, y) => x - y)).toEqual([-2, -1, 0, 1, 2, 3]);
  });

  it("randInt with lo == hi consumes exactly one draw", () => {
    const state = seedToState(99);
    const draw = randInt(state, 5, 5);
    expect(draw.value).toBe(5);
    expect(draw.state).toBe(nextU32(state).state);
  });

  it("rejects bad arguments", () => {
    expect(() => randInt(1, 3, 2)).toThrow(RangeError);
    expect(() => randInt(1, 0.5, 2)).toThrow(RangeError);
    expect(() => randInt(1, 0, 4294967296)).toThrow(RangeError);
    expect(() => seedToState(1.5)).toThrow(RangeError);
  });

  it("matches the committed cross-language vectors", () => {
    const url = new URL("../fixtures/rng_vectors.json", import.meta.url);
    const vectors = JSON.parse(readFileSync(url, "utf8")) as {
      seeds: { seed: number; u32: number[]; randInt: { lo: number; hi: number; values: number[] } }[];
    };
    expect(vectors.seeds.length).toBeGreaterThan(0);
    for (const v of vectors.seeds) {
      let state = seedToState(v.seed);
      for (const expected of v.u32) {
        const draw = nextU32(state);
        state = draw.state;
        expect(draw.value).toBe(expected);
      }
      for (const expected of v.randInt.values) {
        const draw = randInt(state, v.randInt.lo, v.randInt.hi);
        state = draw.state;
        expect(draw.value).toBe(expected);
      }
    }
  });
});
