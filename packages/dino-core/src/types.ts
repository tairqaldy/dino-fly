/** Obstacle kinds, indexing `constants.obstacles`. */
export const ObstacleType = { CACTUS_SMALL: 0, CACTUS_LARGE: 1, PTERODACTYL: 2 } as const;
export type ObstacleTypeId = (typeof ObstacleType)[keyof typeof ObstacleType];

/** Held state of the two keys during one frame. */
export interface Input {
  readonly jump: boolean;
  readonly duck: boolean;
}

export const NO_INPUT: Input = { jump: false, duck: false };

/** All numeric fields are safe integers. Positions/speeds are fixed point (1 px = 1000) unless the name says `Px`. */
export interface Obstacle {
  readonly type: number;
  /** Left edge, fixed point. */
  readonly x: number;
  /** Height of the bottom edge above the ground, px. */
  readonly yBottomPx: number;
  /** Group size 1–3 (cacti); always 1 for pterodactyls. */
  readonly size: number;
  readonly widthPx: number;
  readonly heightPx: number;
  /** Distance to the next obstacle, fixed point. */
  readonly gap: number;
  /** Extra speed relative to the ground, fixed point per frame (pterodactyls only, signed). */
  readonly speedOffset: number;
  readonly followingCreated: boolean;
  /** True once the obstacle's right edge is behind the dino's left edge (counted in `cleared`). */
  readonly passed: boolean;
}

export interface GameState {
  readonly frame: number;
  /** mulberry32 state (canonical uint32). */
  readonly rng: number;
  /** Ground speed, fixed point per frame. */
  readonly speed: number;
  /** Distance run, fixed point. */
  readonly distance: number;
  readonly crashed: boolean;
  /** Obstacle type that killed the dino, -1 while alive. */
  readonly deathType: number;
  /** Height of the dino's feet above the ground, fixed point (>= 0). */
  readonly dinoY: number;
  /** Vertical velocity, fixed point per frame, positive = up. */
  readonly dinoVy: number;
  readonly jumping: boolean;
  readonly ducking: boolean;
  readonly reachedMinHeight: boolean;
  readonly speedDrop: boolean;
  readonly prevJump: boolean;
  /** Number of jumps started / duck phases started / obstacles cleared so far. */
  readonly jumps: number;
  readonly ducks: number;
  readonly cleared: number;
  /** Types of the last two spawned obstacles (-1 = none), most recent first. */
  readonly history0: number;
  readonly history1: number;
  readonly obstacles: readonly Obstacle[];
}

/** Action log entry: the input bits (1 = jump, 2 = duck) that hold from `frame` on. */
export type ActionLog = readonly (readonly [frame: number, bits: number])[];
