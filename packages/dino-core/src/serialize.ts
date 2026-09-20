/**
 * Canonical serialisation of a game state: a flat list of integers in a fixed, documented order.
 * The golden fixtures hash the comma-joined decimal string of this list; the Python port must produce the same bytes.
 *
 * Order: frame, rng, speed, distance, crashed, deathType, dinoY, dinoVy, jumping, ducking, reachedMinHeight,
 * speedDrop, prevJump, jumps, ducks, cleared, history0, history1, nObstacles, then per obstacle:
 * type, x, yBottomPx, size, widthPx, heightPx, gap, speedOffset, followingCreated, passed.
 */
import type { GameState, Input } from "./types.js";

export function stateToInts(s: GameState): number[] {
  const b = (x: boolean): number => (x ? 1 : 0);
  const out = [
    s.frame,
    s.rng,
    s.speed,
    s.distance,
    b(s.crashed),
    s.deathType,
    s.dinoY,
    s.dinoVy,
    b(s.jumping),
    b(s.ducking),
    b(s.reachedMinHeight),
    b(s.speedDrop),
    b(s.prevJump),
    s.jumps,
    s.ducks,
    s.cleared,
    s.history0,
    s.history1,
    s.obstacles.length,
  ];
  for (const o of s.obstacles) {
    out.push(
      o.type,
      o.x,
      o.yBottomPx,
      o.size,
      o.widthPx,
      o.heightPx,
      o.gap,
      o.speedOffset,
      b(o.followingCreated),
      b(o.passed),
    );
  }
  return out;
}

const HEAD = 19;
const PER_OBSTACLE = 10;

/** Inverse of `stateToInts` (used by clients that receive the fly's state over the wire). */
export function intsToState(ints: readonly number[]): GameState {
  const n = ints[HEAD - 1];
  if (n === undefined || ints.length !== HEAD + PER_OBSTACLE * n || !ints.every((v) => Number.isSafeInteger(v))) {
    throw new RangeError("malformed canonical state");
  }
  const at = (i: number): number => ints[i] as number;
  const obstacles = [];
  for (let k = 0; k < n; k++) {
    const o = HEAD + PER_OBSTACLE * k;
    obstacles.push({
      type: at(o),
      x: at(o + 1),
      yBottomPx: at(o + 2),
      size: at(o + 3),
      widthPx: at(o + 4),
      heightPx: at(o + 5),
      gap: at(o + 6),
      speedOffset: at(o + 7),
      followingCreated: at(o + 8) === 1,
      passed: at(o + 9) === 1,
    });
  }
  return {
    frame: at(0),
    rng: at(1),
    speed: at(2),
    distance: at(3),
    crashed: at(4) === 1,
    deathType: at(5),
    dinoY: at(6),
    dinoVy: at(7),
    jumping: at(8) === 1,
    ducking: at(9) === 1,
    reachedMinHeight: at(10) === 1,
    speedDrop: at(11) === 1,
    prevJump: at(12) === 1,
    jumps: at(13),
    ducks: at(14),
    cleared: at(15),
    history0: at(16),
    history1: at(17),
    obstacles,
  };
}

export function canonicalString(s: GameState): string {
  return stateToInts(s).join(",");
}

/** Input bits: 1 = jump, 2 = duck. */
export function inputToBits(input: Input): number {
  return (input.jump ? 1 : 0) + (input.duck ? 2 : 0);
}

export function bitsToInput(bits: number): Input {
  if (!Number.isInteger(bits) || bits < 0 || bits > 3) {
    throw new RangeError(`input bits must be 0..3, got ${bits}`);
  }
  return { jump: bits === 1 || bits === 3, duck: bits >= 2 };
}
