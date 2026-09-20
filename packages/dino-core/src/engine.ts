/**
 * dino-core engine v1 — a deterministic, all-integer Chrome-Dino-style runner.
 *
 * `step(state, input)` is a pure function at a fixed 60 Hz tick. Every field of the state is a safe integer or a
 * boolean; positions and speeds are fixed point (1 px = 1000 units). The PRNG state is part of the game state and
 * the order of random draws is part of the spec, so the obstacle sequence is fully determined by the seed.
 *
 * Physics constants and update order follow the Chromium offline game for "feel" (BSD, The Chromium Authors;
 * see THIRD_PARTY_NOTICES.md) — this is our own implementation. Deliberate differences: no per-frame pixel
 * rounding (obstacles move at exact sub-pixel speed), no deltaTime scaling, score is floor() not round(),
 * every obstacle is collision-checked (not only the first), own collision boxes.
 *
 * Coordinates: x grows to the right, y is the height above the ground. The Python port
 * (`brain/flybrain/dino_core.py`) mirrors this file statement by statement.
 */
import constants from "../constants.json";
import { idiv } from "./fixed.js";
import { randInt, seedToState } from "./rng.js";
import type { GameState, Input, Obstacle } from "./types.js";

export const CONSTANTS = constants;
export const ENGINE_VERSION: number = constants.engineVersion;

const C = constants;
const FPX = C.fp;
const D = C.dino;

export function createInitialState(seed: number): GameState {
  return {
    frame: 0,
    rng: seedToState(seed),
    speed: C.speed.initial,
    distance: 0,
    crashed: false,
    deathType: -1,
    dinoY: 0,
    dinoVy: 0,
    jumping: false,
    ducking: false,
    reachedMinHeight: false,
    speedDrop: false,
    prevJump: false,
    jumps: 0,
    ducks: 0,
    cleared: 0,
    history0: -1,
    history1: -1,
    obstacles: [],
  };
}

export function score(state: GameState): number {
  return idiv(state.distance, C.distancePerPoint);
}

/** Collision boxes of an obstacle in its local frame [x, yBottom, w, h] (px); cactus groups stretch the middle box. */
export function obstacleBoxes(type: number, size: number): number[][] {
  const t = C.obstacles[type];
  if (t === undefined) {
    throw new RangeError(`unknown obstacle type ${type}`);
  }
  const [left, middle, right] = t.boxes as [number[], number[], number[]];
  if (size === 1) {
    return [left, middle, right];
  }
  const width = t.width * size;
  const rightW = right[2] as number;
  const leftW = left[2] as number;
  return [
    left,
    [middle[0] as number, middle[1] as number, width - leftW - rightW, middle[3] as number],
    [width - rightW, right[1] as number, rightW, right[3] as number],
  ];
}

function overlaps(
  aLeft: number,
  aRight: number,
  aBottom: number,
  aTop: number,
  bLeft: number,
  bRight: number,
  bBottom: number,
  bTop: number,
): boolean {
  return aLeft < bRight && aRight > bLeft && aBottom < bTop && aTop > bBottom;
}

/** Broad phase on 1-px-inset bounding boxes, then narrow phase on the per-sprite collision boxes. */
export function collides(dinoY: number, ducking: boolean, o: Obstacle): boolean {
  const w = ducking ? D.duckWidth : D.width;
  const h = ducking ? D.duckHeight : D.height;
  const dinoLeft = D.x * FPX;
  const obsBottom = o.yBottomPx * FPX;
  if (
    !overlaps(
      dinoLeft + FPX,
      dinoLeft + (w - 1) * FPX,
      dinoY + FPX,
      dinoY + (h - 1) * FPX,
      o.x + FPX,
      o.x + (o.widthPx - 1) * FPX,
      obsBottom + FPX,
      obsBottom + (o.heightPx - 1) * FPX,
    )
  ) {
    return false;
  }
  const dinoBoxes = ducking ? D.duckBoxes : D.boxes;
  const obsBoxes = obstacleBoxes(o.type, o.size);
  for (const a of dinoBoxes) {
    const [ax, ay, aw, ah] = a as [number, number, number, number];
    for (const b of obsBoxes) {
      const [bx, by, bw, bh] = b as [number, number, number, number];
      if (
        overlaps(
          dinoLeft + ax * FPX,
          dinoLeft + (ax + aw) * FPX,
          dinoY + ay * FPX,
          dinoY + (ay + ah) * FPX,
          o.x + bx * FPX,
          o.x + (bx + bw) * FPX,
          obsBottom + by * FPX,
          obsBottom + (by + bh) * FPX,
        )
      ) {
        return true;
      }
    }
  }
  return false;
}

interface Spawn {
  readonly rng: number;
  readonly obstacle: Obstacle;
}

/** Draw order (part of the spec): type (retry while rejected), size, altitude index, speed-offset sign, gap. */
function spawnObstacle(rngIn: number, speed: number, history0: number, history1: number): Spawn {
  let rng = rngIn;
  let type = 0;
  for (;;) {
    const draw = randInt(rng, 0, C.obstacles.length - 1);
    rng = draw.state;
    type = draw.value;
    const duplicated = history0 === type && history1 === type;
    const t = C.obstacles[type] as (typeof C.obstacles)[number];
    if (!duplicated && speed >= t.minSpeed) {
      break;
    }
  }
  const t = C.obstacles[type] as (typeof C.obstacles)[number];
  const sizeDraw = randInt(rng, 1, C.maxObstacleLength);
  const size = sizeDraw.value > 1 && t.multipleSpeed > speed ? 1 : sizeDraw.value;
  const altDraw = randInt(sizeDraw.state, 0, t.yBottoms.length - 1);
  const signDraw = randInt(altDraw.state, 0, 1);
  const speedOffset = t.speedOffset === 0 ? 0 : signDraw.value === 1 ? t.speedOffset : -t.speedOffset;
  const widthPx = t.width * size;
  const minGap = widthPx * speed + t.minGap * C.gap.coefficientMilli;
  const maxGap = idiv(minGap * C.gap.maxNumerator, C.gap.maxDenominator);
  const gapDraw = randInt(signDraw.state, minGap, maxGap);
  return {
    rng: gapDraw.state,
    obstacle: {
      type,
      x: (C.world.width + t.width) * FPX,
      yBottomPx: t.yBottoms[altDraw.value] as number,
      size,
      widthPx,
      heightPx: t.height,
      gap: gapDraw.value,
      speedOffset,
      followingCreated: false,
      passed: false,
    },
  };
}

export function step(state: GameState, input: Input): GameState {
  if (state.crashed) {
    return state;
  }
  let { dinoY, dinoVy, jumping, ducking, reachedMinHeight, speedDrop, jumps, ducks } = state;

  // 1) input → dino
  if (jumping) {
    if (input.duck && !speedDrop) {
      speedDrop = true;
      dinoVy = D.speedDropVelocity;
    } else if (state.prevJump && !input.jump && reachedMinHeight && dinoVy > D.dropVelocity) {
      dinoVy = D.dropVelocity; // key released: cut the jump short
    }
  } else if (input.duck) {
    if (!ducking) {
      ducks += 1;
    }
    ducking = true;
  } else {
    ducking = false;
    if (input.jump) {
      jumping = true;
      dinoVy = D.jumpVelocityBase + idiv(state.speed, D.jumpVelocitySpeedDivisor);
      reachedMinHeight = false;
      speedDrop = false;
      jumps += 1;
    }
  }

  // 2) jump physics (same order as the original: move, gravity, min height, max-height clamp, landing)
  if (jumping) {
    dinoY += speedDrop ? dinoVy * D.speedDropCoefficient : dinoVy;
    dinoVy -= D.gravity;
    if (dinoY > D.minJumpHeight || speedDrop) {
      reachedMinHeight = true;
    }
    if ((dinoY > D.maxJumpHeight || speedDrop) && reachedMinHeight && dinoVy > D.dropVelocity) {
      dinoVy = D.dropVelocity;
    }
    if (dinoY < 0) {
      dinoY = 0;
      dinoVy = 0;
      jumping = false;
      speedDrop = false;
      reachedMinHeight = false;
    }
  }

  // 3) world: move obstacles, count cleared ones, drop off-screen ones, spawn
  const frame = state.frame + 1;
  let rng = state.rng;
  let cleared = state.cleared;
  let history0 = state.history0;
  let history1 = state.history1;
  let obstacles: Obstacle[] = [];
  if (frame > C.clearFrames) {
    for (const o of state.obstacles) {
      const x = o.x - (state.speed + o.speedOffset);
      let passed = o.passed;
      if (!passed && x + o.widthPx * FPX < D.x * FPX) {
        passed = true;
        cleared += 1;
      }
      if (x + o.widthPx * FPX > 0) {
        obstacles.push({ ...o, x, passed });
      }
    }
    const last = obstacles[obstacles.length - 1];
    let spawn: Spawn | undefined;
    if (last === undefined) {
      spawn = spawnObstacle(rng, state.speed, history0, history1);
    } else if (!last.followingCreated && last.x + last.widthPx * FPX + last.gap < C.world.width * FPX) {
      obstacles = [...obstacles.slice(0, -1), { ...last, followingCreated: true }];
      spawn = spawnObstacle(rng, state.speed, history0, history1);
    }
    if (spawn !== undefined) {
      rng = spawn.rng;
      obstacles.push(spawn.obstacle);
      history1 = history0;
      history0 = spawn.obstacle.type;
    }
  }

  // 4) collision
  let crashed = false;
  let deathType = -1;
  for (const o of obstacles) {
    if (collides(dinoY, ducking, o)) {
      crashed = true;
      deathType = o.type;
      break;
    }
  }

  // 5) progress
  let speed = state.speed;
  let distance = state.distance;
  if (!crashed) {
    distance += speed;
    if (speed < C.speed.max) {
      speed += C.speed.accelPerFrame;
    }
  }

  return {
    frame,
    rng,
    speed,
    distance,
    crashed,
    deathType,
    dinoY,
    dinoVy,
    jumping,
    ducking,
    reachedMinHeight,
    speedDrop,
    prevJump: input.jump,
    jumps,
    ducks,
    cleared,
    history0,
    history1,
    obstacles,
  };
}
