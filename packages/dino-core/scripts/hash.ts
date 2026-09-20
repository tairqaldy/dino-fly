/** Node-only helpers for golden fixtures (kept out of `src/` so the engine stays browser-safe). */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createInitialState, step } from "../src/engine.js";
import { canonicalString, inputToBits, stateToInts } from "../src/serialize.js";
import type { GameState, Input } from "../src/types.js";
import type { Policy } from "./policies.js";

export const HASH_CHARS = 8;
export const CHECKPOINT_EVERY = 250;

export function stateHash(state: GameState): string {
  return createHash("sha256").update(canonicalString(state), "utf8").digest("hex").slice(0, HASH_CHARS);
}

export function constantsSha256(): string {
  const raw = readFileSync(new URL("../constants.json", import.meta.url));
  // hash the LF-normalised text so the value does not depend on the platform's checkout line endings
  return createHash("sha256").update(raw.toString("utf8").replace(/\r\n/g, "\n"), "utf8").digest("hex");
}

export interface RecordedRun {
  readonly actions: [number, number][];
  readonly hashes: string;
  readonly checkpoints: number[][];
  readonly final: GameState;
}

/** Play `policy` on `seed`, recording input changes, a per-frame state hash and sparse full-state checkpoints. */
export function recordRun(seed: number, policy: Policy, maxFrames: number): RecordedRun {
  let state = createInitialState(seed);
  const actions: [number, number][] = [];
  const checkpoints: number[][] = [];
  let hashes = "";
  let bits = 0;
  for (let f = 0; f < maxFrames && !state.crashed; f++) {
    const input: Input = policy(state);
    const b = inputToBits(input);
    if (b !== bits) {
      actions.push([f, b]);
      bits = b;
    }
    state = step(state, input);
    hashes += stateHash(state);
    if (state.frame % CHECKPOINT_EVERY === 0 || state.crashed) {
      checkpoints.push(stateToInts(state));
    }
  }
  return { actions, hashes, checkpoints, final: state };
}
