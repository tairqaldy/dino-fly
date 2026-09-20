import { readdirSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { recordRun, stateHash } from "../scripts/hash.js";
import { holdJump, never, oracle, seededRandom } from "../scripts/policies.js";
import { CONSTANTS, collides, createInitialState, ENGINE_VERSION, obstacleBoxes, score, step } from "../src/engine.js";
import { replay, summarize, validateActionLog } from "../src/replay.js";
import { bitsToInput, canonicalString, inputToBits, intsToState, stateToInts } from "../src/serialize.js";
import type { GameState, Input, Obstacle } from "../src/types.js";
import { NO_INPUT } from "../src/types.js";

const FPX = CONSTANTS.fp;
const JUMP: Input = { jump: true, duck: false };
const DUCK: Input = { jump: false, duck: true };

function deepFreeze<T>(x: T): T {
  if (typeof x === "object" && x !== null) {
    for (const v of Object.values(x)) {
      deepFreeze(v);
    }
    Object.freeze(x);
  }
  return x;
}

function run(seed: number, frames: number, policy: (s: GameState) => Input): GameState[] {
  const states = [createInitialState(seed)];
  for (let f = 0; f < frames; f++) {
    const s = states[states.length - 1] as GameState;
    states.push(step(s, policy(s)));
  }
  return states;
}

function obstacle(partial: Partial<Obstacle>): Obstacle {
  return {
    type: 0,
    x: 60 * FPX,
    yBottomPx: 0,
    size: 1,
    widthPx: 17,
    heightPx: 35,
    gap: 100000,
    speedOffset: 0,
    followingCreated: true,
    passed: false,
    ...partial,
  };
}

describe("engine basics", () => {
  it("exposes its version and starts from a clean state", () => {
    expect(ENGINE_VERSION).toBe(1);
    const s = createInitialState(7);
    expect(s).toMatchObject({ frame: 0, speed: 6000, distance: 0, crashed: false, dinoY: 0, obstacles: [] });
    expect(score(s)).toBe(0);
  });

  it("is deterministic and pure (never mutates its input)", () => {
    const a = run(99, 600, oracle);
    const b = run(99, 600, oracle);
    expect(a.map(canonicalString)).toEqual(b.map(canonicalString));
    let s = deepFreeze(createInitialState(5));
    for (let f = 0; f < 400; f++) {
      s = deepFreeze(step(s, oracle(s)));
    }
    expect(s.frame).toBe(400);
  });

  it("different seeds give different obstacle sequences", () => {
    const a = run(1, 400, never).at(-1) as GameState;
    const b = run(2, 400, never).at(-1) as GameState;
    expect(canonicalString(a)).not.toBe(canonicalString(b));
  });

  it("keeps every state field a safe integer", () => {
    for (const s of run(3, 3000, oracle)) {
      for (const v of stateToInts(s)) {
        expect(Number.isSafeInteger(v)).toBe(true);
      }
    }
  });

  it("a crashed state is frozen", () => {
    const end = run(1, 400, never).at(-1) as GameState;
    expect(end.crashed).toBe(true);
    expect(end.deathType).toBeGreaterThanOrEqual(0);
    expect(step(end, JUMP)).toBe(end);
  });

  it("accelerates by 1/1000 px per frame up to the maximum and scores distance/40", () => {
    const states = run(11, 7600, oracle);
    expect((states[1000] as GameState).speed).toBe(7000);
    expect((states[7600] as GameState).speed).toBe(13000);
    const s = states[1000] as GameState;
    expect(score(s)).toBe(Math.floor(s.distance / 40000));
  });
});

describe("jump physics (hand-computed at speed 6)", () => {
  it("full jump follows the expected trajectory with the max-height clamp", () => {
    const states = run(1, 40, () => JUMP);
    const ys = states.slice(1, 18).map((s) => s.dinoY);
    expect(ys).toEqual([
      10600, 20600, 30000, 38800, 47000, 54600, 61600, 68000, 73000, 77400, 81200, 84400, 87000, 89000, 90400, 91200,
      91400,
    ]);
    expect((states[8] as GameState).dinoVy).toBe(5000); // clamped to the drop velocity above 63 px
    expect((states[3] as GameState).reachedMinHeight).toBe(false); // exactly 30 px is not "above"
    expect((states[4] as GameState).reachedMinHeight).toBe(true);
    expect((states[1] as GameState).jumps).toBe(1);
  });

  it("lands, and holding the key jumps again immediately", () => {
    const states = run(1, 80, () => JUMP);
    const landing = states.findIndex((s, i) => i > 1 && !s.jumping);
    expect(landing).toBeGreaterThan(30);
    expect(landing).toBeLessThan(40);
    expect((states[landing] as GameState).dinoY).toBe(0);
    expect((states[landing + 1] as GameState).jumping).toBe(true);
    expect((states[landing + 1] as GameState).jumps).toBe(2);
  });

  it("releasing the key between min and max height shortens the jump; an early tap does not", () => {
    const apex = (hold: number): number =>
      Math.max(...run(1, 30, (s) => (s.frame < hold ? JUMP : NO_INPUT)).map((s) => s.dinoY));
    expect(apex(30)).toBe(91400);
    expect(apex(2)).toBe(91400); // released before the minimum height: nothing happens
    expect(apex(5)).toBeLessThan(80000);
    expect(apex(5)).toBeGreaterThan(47000);
  });

  it("duck in mid-air is a speed drop; duck on the ground ducks; jump is ignored while ducking", () => {
    const states = run(1, 30, (s) => (s.frame < 6 ? JUMP : DUCK));
    const drop = states[7] as GameState;
    expect(drop.speedDrop).toBe(true);
    expect(drop.dinoVy).toBe(-1000 - 600);
    expect(drop.dinoY).toBe(54600 - 3000);
    const landed = states.find((s) => s.frame > 7 && !s.jumping) as GameState;
    expect(landed.speedDrop).toBe(false);
    const after = states[landed.frame + 1] as GameState;
    expect(after.ducking).toBe(true);
    expect(after.ducks).toBe(1);
    const both = run(1, 5, () => ({ jump: true, duck: true })).at(-1) as GameState;
    expect(both.jumping).toBe(false);
    expect(both.ducking).toBe(true);
    expect(both.ducks).toBe(1);
  });
});

describe("obstacles", () => {
  it("none before frame 181, first one spawns at x = 600 + type width", () => {
    const states = run(1, 181, never);
    expect((states[180] as GameState).obstacles).toHaveLength(0);
    const first = (states[181] as GameState).obstacles[0] as Obstacle;
    const t = CONSTANTS.obstacles[first.type] as (typeof CONSTANTS.obstacles)[number];
    expect(first.x).toBe((600 + t.width) * FPX);
  });

  it("respects gap bounds, pterodactyl speed gating, group gating and the duplication rule", () => {
    for (const seed of [11, 12, 13, 14]) {
      const spawned: { o: Obstacle; speed: number }[] = [];
      let prev = createInitialState(seed);
      for (let f = 0; f < 9000; f++) {
        const next = step(prev, oracle(prev));
        const last = next.obstacles.at(-1);
        if (last !== undefined && last.x === (600 + (CONSTANTS.obstacles[last.type]?.width ?? 0)) * FPX) {
          spawned.push({ o: last, speed: prev.speed });
        }
        prev = next;
      }
      expect(spawned.length).toBeGreaterThan(100);
      const types = spawned.map((x) => x.o.type);
      expect(new Set(types).size).toBe(3);
      for (let i = 2; i < types.length; i++) {
        expect(types[i] === types[i - 1] && types[i] === types[i - 2]).toBe(false);
      }
      for (const { o, speed } of spawned) {
        const t = CONSTANTS.obstacles[o.type] as (typeof CONSTANTS.obstacles)[number];
        const minGap = o.widthPx * speed + t.minGap * 600;
        expect(o.gap).toBeGreaterThanOrEqual(minGap);
        expect(o.gap).toBeLessThanOrEqual(Math.floor((minGap * 3) / 2));
        expect(speed).toBeGreaterThanOrEqual(t.minSpeed);
        expect(o.widthPx).toBe(t.width * o.size);
        if (o.size > 1) {
          expect(speed).toBeGreaterThanOrEqual(t.multipleSpeed);
        }
        if (o.type === 2) {
          expect([0, 25, 50]).toContain(o.yBottomPx);
          expect(Math.abs(o.speedOffset)).toBe(800);
        } else {
          expect(o.speedOffset).toBe(0);
          expect(o.yBottomPx).toBe(0);
        }
      }
      expect(spawned.some((x) => x.o.size === 3)).toBe(true);
      expect(new Set(spawned.filter((x) => x.o.type === 2).map((x) => x.o.yBottomPx)).size).toBe(3);
    }
  });

  it("counts cleared obstacles exactly once and removes them off-screen", () => {
    const end = run(12, 3000, oracle).at(-1) as GameState;
    expect(end.cleared).toBeGreaterThan(30);
    expect(end.obstacles.every((o) => o.x + o.widthPx * FPX > 0)).toBe(true);
  });

  it("stretches the middle collision box of cactus groups", () => {
    expect(obstacleBoxes(0, 1)).toEqual(CONSTANTS.obstacles[0]?.boxes);
    expect(obstacleBoxes(0, 3)).toEqual([
      [0, 12, 5, 16],
      [5, 0, 41, 35],
      [46, 14, 5, 14],
    ]);
    expect(() => obstacleBoxes(9, 1)).toThrow(RangeError);
  });
});

describe("collisions", () => {
  it("standing dino hits a cactus it overlaps, clears it from above", () => {
    expect(collides(0, false, obstacle({ x: 70 * FPX }))).toBe(true);
    expect(collides(40 * FPX, false, obstacle({ x: 70 * FPX }))).toBe(false);
    expect(collides(0, false, obstacle({ x: 300 * FPX }))).toBe(false);
  });

  it("bounding boxes may overlap without the sprites touching", () => {
    // cactus arm region vs. the empty corner under the dino's head
    expect(collides(30 * FPX, false, obstacle({ x: 90 * FPX }))).toBe(false);
  });

  it("pterodactyl altitudes: low needs a jump, mid needs a duck, high can be run under", () => {
    const ptero = (yBottomPx: number): Obstacle =>
      obstacle({ type: 2, x: 60 * FPX, yBottomPx, widthPx: 46, heightPx: 40 });
    expect(collides(0, false, ptero(0))).toBe(true);
    expect(collides(0, false, ptero(25))).toBe(true);
    expect(collides(0, true, ptero(25))).toBe(false);
    expect(collides(0, false, ptero(50))).toBe(false);
    expect(collides(60 * FPX, false, ptero(50))).toBe(true);
  });
});

describe("replay and serialisation", () => {
  it("replaying a recorded action log reproduces the run", () => {
    const rec = recordRun(6, seededRandom(66), 3000);
    const again = replay(6, rec.actions, 3000);
    expect(canonicalString(again)).toBe(canonicalString(rec.final));
    expect(summarize(6, again)).toMatchObject({ seed: 6, engineVersion: 1, frames: rec.final.frame });
    let calls = 0;
    replay(6, rec.actions, 50, () => {
      calls += 1;
    });
    expect(calls).toBe(50);
  });

  it("rejects malformed action logs and input bits", () => {
    expect(() => validateActionLog([[5, 1], [5, 0]])).toThrow(RangeError);
    expect(() => validateActionLog([[-1, 1]])).toThrow(RangeError);
    expect(() => validateActionLog([[0.5, 1]])).toThrow(RangeError);
    expect(() => validateActionLog([[0, 4]])).toThrow(RangeError);
    expect(() => bitsToInput(-1)).toThrow(RangeError);
    for (const bits of [0, 1, 2, 3]) {
      expect(inputToBits(bitsToInput(bits))).toBe(bits);
    }
  });

  it("serialises to 19 + 10·n integers and back", () => {
    const s = run(11, 400, oracle).at(-1) as GameState;
    expect(stateToInts(s)).toHaveLength(19 + 10 * s.obstacles.length);
    expect(intsToState(stateToInts(s))).toEqual(s);
    expect(() => intsToState([1, 2, 3])).toThrow(RangeError);
    expect(() => intsToState([...stateToInts(s), 0.5])).toThrow(RangeError);
    expect(stateHash(s)).toMatch(/^[0-9a-f]{8}$/);
  });
});

describe("golden fixtures", () => {
  it("are reproduced by the current engine (regenerate with `pnpm gen-fixtures` after an ENGINE_VERSION bump)", () => {
    const file = JSON.parse(readFileSync(new URL("../fixtures/game_runs.json", import.meta.url), "utf8")) as {
      engineVersion: number;
      hashChars: number;
      runs: { name: string; seed: number; maxFrames: number; actions: [number, number][]; hashes: string }[];
    };
    expect(file.engineVersion).toBe(ENGINE_VERSION);
    expect(file.runs.length).toBeGreaterThanOrEqual(10);
    for (const r of file.runs) {
      let hashes = "";
      replay(r.seed, r.actions, r.maxFrames, (s) => {
        hashes += stateHash(s);
      });
      expect(hashes === r.hashes, `fixture ${r.name} diverged`).toBe(true);
    }
  });

  it("hold-jump dies on the first obstacle it lands on", () => {
    const end = run(2, 3000, holdJump).at(-1) as GameState;
    expect(end.crashed).toBe(true);
  });
});

describe("physics files use integer helpers only", () => {
  it("no raw division, modulo, rounding or int32 coercion in engine / serialize / replay", () => {
    const dir = new URL("../src/", import.meta.url);
    const banned = [/[^/*]\/[^/*]/, /%/, /Math\.(round|trunc|floor|ceil)/, /\|\s*0\b/, />>>?|<</];
    for (const name of readdirSync(dir)) {
      if (!["engine.ts", "serialize.ts", "replay.ts"].includes(name)) {
        continue;
      }
      const code = readFileSync(new URL(name, dir), "utf8")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/.*$/gm, "")
        .replace(/(["'`])(?:\\.|(?!\1).)*\1/g, '""');
      for (const re of banned) {
        expect(re.test(code), `${name} must not match ${re}`).toBe(false);
      }
    }
  });
});
