/**
 * One human run on a seed, optionally raced against a recorded fly run ("ghost") on the same seed.
 * Pure logic (no DOM): the component calls `tick(input)` at a fixed 60 Hz and renders `human` / `ghost`.
 */
import {
  type ActionLog,
  bitsToInput,
  createInitialState,
  type GameState,
  type Input,
  inputToBits,
  score,
  step,
} from "@dino-fly/dino-core";

export class RaceSession {
  human: GameState;
  ghost: GameState | null;
  readonly actions: [number, number][] = [];
  private bits = 0;
  private ghostBits = 0;
  private ghostCursor = 0;

  constructor(
    readonly seed: number,
    private readonly ghostActions: ActionLog | null,
  ) {
    this.human = createInitialState(seed);
    this.ghost = ghostActions ? createInitialState(seed) : null;
  }

  get over(): boolean {
    return this.human.crashed;
  }

  /** Advance both dinos by one frame. The human's input changes are recorded for server-side replay validation. */
  tick(input: Input): void {
    if (!this.human.crashed) {
      const b = inputToBits(input);
      if (b !== this.bits) {
        this.actions.push([this.human.frame, b]);
        this.bits = b;
      }
      this.human = step(this.human, input);
    }
    if (this.ghost && this.ghostActions && !this.ghost.crashed) {
      const next = this.ghostActions[this.ghostCursor];
      if (next !== undefined && next[0] === this.ghost.frame) {
        this.ghostBits = next[1];
        this.ghostCursor += 1;
      }
      this.ghost = step(this.ghost, bitsToInput(this.ghostBits));
    }
  }

  result(): { seed: number; frames: number; score: number; actions: [number, number][]; flyScore: number | null } {
    return {
      seed: this.seed,
      frames: this.human.frame,
      score: score(this.human),
      actions: this.actions,
      flyScore: this.ghost ? score(this.ghost) : null,
    };
  }
}

/** Fixed-timestep accumulator: converts variable requestAnimationFrame deltas into whole 60 Hz ticks. */
export class FixedStep {
  private acc = 0;
  constructor(
    readonly stepMs: number = 1000 / 60,
    readonly maxCatchUp: number = 5,
  ) {}

  ticks(deltaMs: number): number {
    this.acc += Math.min(deltaMs, this.stepMs * this.maxCatchUp);
    let n = 0;
    while (this.acc >= this.stepMs) {
      this.acc -= this.stepMs;
      n += 1;
    }
    return n;
  }
}
