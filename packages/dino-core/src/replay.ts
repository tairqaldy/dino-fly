/**
 * Deterministic replay of a run from (seed, action log). The API uses this to validate submitted human runs:
 * the server accepts the replayed score, never the client's claim.
 */
import { createInitialState, ENGINE_VERSION, score, step } from "./engine.js";
import { bitsToInput } from "./serialize.js";
import type { ActionLog, GameState } from "./types.js";

export interface RunSummary {
  readonly engineVersion: number;
  readonly seed: number;
  readonly frames: number;
  readonly score: number;
  readonly crashed: boolean;
  readonly deathType: number;
  readonly jumps: number;
  readonly ducks: number;
  readonly cleared: number;
}

export function summarize(seed: number, state: GameState): RunSummary {
  return {
    engineVersion: ENGINE_VERSION,
    seed,
    frames: state.frame,
    score: score(state),
    crashed: state.crashed,
    deathType: state.deathType,
    jumps: state.jumps,
    ducks: state.ducks,
    cleared: state.cleared,
  };
}

/** Throws if the log is malformed (frames must start at >= 0 and strictly increase; bits in 0..3). */
export function validateActionLog(log: ActionLog): void {
  let prev = -1;
  for (const entry of log) {
    const [frame, bits] = entry;
    if (!Number.isSafeInteger(frame) || frame <= prev) {
      throw new RangeError(`action log frames must be strictly increasing safe integers (got ${frame} after ${prev})`);
    }
    bitsToInput(bits);
    prev = frame;
  }
}

/**
 * Entry [f, bits] means: from step index f on (the step that produces frame f + 1), hold these bits.
 * The run ends at the crash or after `maxFrames` steps.
 */
export function replay(
  seed: number,
  log: ActionLog,
  maxFrames: number,
  onFrame?: (state: GameState) => void,
): GameState {
  validateActionLog(log);
  let state = createInitialState(seed);
  let bits = 0;
  let cursor = 0;
  for (let f = 0; f < maxFrames && !state.crashed; f++) {
    const entry = log[cursor];
    if (entry !== undefined && entry[0] === f) {
      bits = entry[1];
      cursor += 1;
    }
    state = step(state, bitsToInput(bits));
    onFrame?.(state);
  }
  return state;
}
