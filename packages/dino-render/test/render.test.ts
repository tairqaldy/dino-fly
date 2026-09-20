import { createInitialState, type GameState, step } from "@dino-fly/dino-core";
import { describe, expect, it } from "vitest";
import {
  ACCENT,
  bitmapSize,
  type Ctx,
  DINO_DEAD,
  DINO_DUCK_A,
  DINO_DUCK_B,
  DINO_RUN_A,
  DINO_RUN_B,
  isNight,
  PTERO_DOWN,
  PTERO_UP,
  render,
} from "../src/index.js";

function fakeCtx(): Ctx & { rects: { style: string; alpha: number }[]; texts: string[] } {
  const rects: { style: string; alpha: number }[] = [];
  const texts: string[] = [];
  const ctx = {
    fillStyle: "" as string,
    strokeStyle: "" as string,
    globalAlpha: 1,
    font: "",
    rects,
    texts,
    fillRect() {
      rects.push({ style: String(ctx.fillStyle), alpha: ctx.globalAlpha });
    },
    strokeRect() {
      rects.push({ style: String(ctx.strokeStyle), alpha: ctx.globalAlpha });
    },
    fillText(t: string) {
      texts.push(t);
    },
    save() {},
    restore() {},
  };
  return ctx;
}

function play(frames: number, policy: (s: GameState) => { jump: boolean; duck: boolean }): GameState {
  let s = createInitialState(3);
  for (let f = 0; f < frames; f++) {
    s = step(s, policy(s));
  }
  return s;
}

describe("sprites", () => {
  it("are rectangular and match the engine's sprite boxes", () => {
    for (const bmp of [DINO_RUN_A, DINO_RUN_B, DINO_DEAD, DINO_DUCK_A, DINO_DUCK_B, PTERO_UP, PTERO_DOWN]) {
      expect(new Set(bmp.map((r) => r.length)).size).toBe(1);
      expect(bmp.join("")).toMatch(/^[#.o]+$/);
    }
    expect(bitmapSize(DINO_RUN_A)).toEqual({ width: 44, height: 48 });
    expect(bitmapSize(DINO_RUN_B)).toEqual(bitmapSize(DINO_RUN_A));
    expect(bitmapSize(DINO_DUCK_A)).toEqual({ width: 60, height: 26 });
    expect(bitmapSize(PTERO_UP)).toEqual({ width: 46, height: 40 });
    expect(bitmapSize([])).toEqual({ width: 0, height: 0 });
  });
});

describe("render", () => {
  it("draws every game situation without throwing", () => {
    const situations = [
      play(10, () => ({ jump: false, duck: false })),
      play(12, () => ({ jump: true, duck: false })),
      play(12, () => ({ jump: false, duck: true })),
      play(17, () => ({ jump: false, duck: true })),
      play(400, () => ({ jump: false, duck: false })), // crashed into the first obstacle
    ];
    for (const s of situations) {
      const ctx = fakeCtx();
      render(ctx, s, { showBoxes: true });
      expect(ctx.rects.length).toBeGreaterThan(50);
      expect(ctx.texts[0]).toMatch(/^SCORE \d{5}$/);
    }
    expect(situations[4]?.crashed).toBe(true);
  });

  it("draws the fly as a translucent accent-coloured ghost with its own score", () => {
    const human = play(30, () => ({ jump: false, duck: false }));
    const fly = play(30, () => ({ jump: true, duck: false }));
    const ctx = fakeCtx();
    render(ctx, human, { ghost: fly, ghostLabel: "FLY", label: "YOU" });
    expect(ctx.rects.some((r) => r.style === ACCENT && r.alpha < 1)).toBe(true);
    expect(ctx.texts).toEqual([expect.stringMatching(/^YOU /), expect.stringMatching(/^FLY /)]);
    const solo = fakeCtx();
    render(solo, fly, { accentDino: true });
    expect(solo.rects.some((r) => r.style === ACCENT && r.alpha === 1)).toBe(true);
  });

  it("flips to night every 700 points", () => {
    const base = createInitialState(1);
    expect(isNight(base)).toBe(false);
    expect(isNight({ ...base, distance: 700 * 40000 })).toBe(true);
    expect(isNight({ ...base, distance: 1400 * 40000 })).toBe(false);
    const ctx = fakeCtx();
    render(ctx, { ...base, distance: 700 * 40000 });
    expect(ctx.rects[0]?.style).toBe("#16161a");
  });
});
