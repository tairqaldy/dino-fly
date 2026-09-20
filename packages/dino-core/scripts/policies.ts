/**
 * Scripted (non-learned) policies. Used to generate golden fixtures and as labelled non-fly baselines.
 * The Python side never re-implements these: fixtures store the recorded per-frame actions.
 */
import { CONSTANTS } from "../src/engine.js";
import { nextU32 } from "../src/rng.js";
import type { GameState, Input } from "../src/types.js";
import { NO_INPUT } from "../src/types.js";

export type Policy = (state: GameState) => Input;

const FPX = CONSTANTS.fp;
const D = CONSTANTS.dino;

export const never: Policy = () => NO_INPUT;

export const holdJump: Policy = () => ({ jump: true, duck: false });

/** Jump (held for `hold` frames) every `period` frames. */
export function periodic(period: number, hold: number): Policy {
  return (s) => ({ jump: s.frame % period < hold, duck: false });
}

/** Alternates ducking, short taps and speed drops. */
export const duckPattern: Policy = (s) => {
  const phase = s.frame % 90;
  if (phase < 30) {
    return { jump: false, duck: true };
  }
  if (phase < 34) {
    return { jump: true, duck: false };
  }
  if (phase >= 40 && phase < 46) {
    return { jump: false, duck: true };
  }
  return NO_INPUT;
};

/** Sticky random key presses from an independent mulberry32 stream (never the game's PRNG). */
export function seededRandom(seed: number): Policy {
  let rng = seed >>> 0;
  let current: Input = NO_INPUT;
  return () => {
    const draw = nextU32(rng);
    rng = draw.state;
    if (draw.value % 8 === 0) {
      const bits = (draw.value >>> 8) % 4;
      current = { jump: bits === 1 || bits === 3, duck: bits >= 2 };
    }
    return current;
  };
}

/**
 * Hand-written look-ahead player: jumps cacti and low pterodactyls, ducks under mid ones, runs under high ones,
 * and speed-drops after clearing an obstacle. A scripted ceiling reference — not a fly, not learned.
 */
export const oracle: Policy = (s) => {
  const dinoBack = D.x * FPX;
  const dinoFront = (D.x + D.width) * FPX;
  const ahead = s.obstacles.filter((o) => o.x + o.widthPx * FPX >= dinoBack);
  const o = ahead[0];
  if (o === undefined) {
    return { jump: false, duck: s.jumping };
  }
  const rel = s.speed + o.speedOffset;
  const dist = o.x - dinoFront;
  const isPtero = o.type === 2;
  const runUnder = isPtero && o.yBottomPx >= 50;
  const duckUnder = isPtero && o.yBottomPx === 25;

  if (s.jumping) {
    // land quickly once nothing is below or about to be below us
    const clearBelow = dist > rel * 9;
    return { jump: !clearBelow, duck: clearBelow };
  }
  if (runUnder) {
    return NO_INPUT;
  }
  if (duckUnder) {
    return { jump: false, duck: dist < rel * 14 };
  }
  const lead = o.heightPx >= 50 ? 7 : 6;
  return { jump: dist <= rel * lead, duck: false };
};
