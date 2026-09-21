import { bitsToInput, createInitialState, type GameState, step } from "@dino-fly/dino-core";
import type { Ghost } from "./api.js";

/**
 * Plays recorded fly runs back at 60 Hz, one after another, so the fly's screen is never empty.
 * These are real runs the connectome played — the same action logs the leaderboard validates — replayed by the
 * deterministic engine, not a canned animation. When the live brain is connected the page shows that instead.
 */
export class FlyReplay {
  private runs: Ghost[] = [];
  private index = 0;
  private cursor = 0;
  private bits = 0;
  private idle = 0;
  state: GameState;

  constructor(runs: Ghost[] = []) {
    this.setRuns(runs);
    this.state = createInitialState(this.current?.seed ?? 0);
  }

  get current(): Ghost | undefined {
    return this.runs[this.index];
  }

  get hasRuns(): boolean {
    return this.runs.length > 0;
  }

  setRuns(runs: Ghost[]): void {
    const usable = runs.filter((r) => r.actions.length > 0);
    if (usable.length === 0) return;
    this.runs = usable;
    this.index = 0;
    this.restart();
  }

  private restart(): void {
    this.cursor = 0;
    this.bits = 0;
    this.idle = 0;
    this.state = createInitialState(this.current?.seed ?? 0);
  }

  /** Advance one frame; after a crash hold the wreck briefly, then start the next recorded run. */
  tick(): GameState {
    const run = this.current;
    if (!run) return this.state;
    if (this.state.crashed) {
      this.idle += 1;
      if (this.idle > 45) {
        this.index = (this.index + 1) % this.runs.length;
        this.restart();
      }
      return this.state;
    }
    const next = run.actions[this.cursor];
    if (next !== undefined && next[0] === this.state.frame) {
      this.bits = next[1] as number;
      this.cursor += 1;
    }
    this.state = step(this.state, bitsToInput(this.bits));
    return this.state;
  }
}
