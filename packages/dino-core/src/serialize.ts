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
