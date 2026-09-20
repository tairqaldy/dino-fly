/**
 * mulberry32 PRNG with an integer-only API.
 *
 * The state is a canonical uint32 and lives inside the game state, so `step(state, input)` stays a pure function.
 * Only integer draws are exposed (no floats), and the draw order is part of the engine spec.
 * Python port: `brain/flybrain/dino_core.py` (`rng_next_u32`, `rng_rand_int`).
 */
import { imod } from "./fixed.js";

export interface RngDraw {
  /** New PRNG state (canonical uint32). */
  readonly state: number;
  /** Drawn value. */
  readonly value: number;
}

/** Canonicalise any integer seed to a uint32 state. */
export function seedToState(seed: number): number {
  if (!Number.isSafeInteger(seed)) {
    throw new RangeError(`seed must be a safe integer, got ${seed}`);
  }
  return imod(seed, 4294967296);
}

/** One mulberry32 step: returns the new state and a uniformly distributed uint32. */
export function nextU32(state: number): RngDraw {
  const a = (state + 0x6d2b79f5) >>> 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return { state: a, value: (t ^ (t >>> 14)) >>> 0 };
}

/**
 * Uniform-ish integer in [lo, hi] (inclusive) via `lo + u32 mod span`.
 * The modulo bias is < span / 2^32 and is accepted and documented: spans in the engine are < 2^20.
 */
export function randInt(state: number, lo: number, hi: number): RngDraw {
  if (!Number.isSafeInteger(lo) || !Number.isSafeInteger(hi) || hi < lo) {
    throw new RangeError(`randInt needs safe integers with lo <= hi, got [${lo}, ${hi}]`);
  }
  const span = hi - lo + 1;
  if (span > 4294967296) {
    throw new RangeError(`randInt span too large: ${span}`);
  }
  const draw = nextU32(state);
  return { state: draw.state, value: lo + imod(draw.value, span) };
}
