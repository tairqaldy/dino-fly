/**
 * Canvas renderer for dino-core states: classic monochrome pixel look, day/night flip, one accent colour for the
 * fly's ghost. Pure function of (state, options) — all cosmetics (clouds, ground bumps, night) derive from the state.
 */
import { CONSTANTS, type GameState, obstacleBoxes, score } from "@dino-fly/dino-core";
import {
  CELL_PX,
  DINO_DEAD,
  DINO_DUCK_A,
  DINO_DUCK_B,
  DINO_RUN_A,
  DINO_RUN_B,
  PTERO_DOWN,
  PTERO_UP,
} from "./sprites.js";

export * from "./sprites.js";

export const WORLD = { width: CONSTANTS.world.width, height: CONSTANTS.world.height, groundY: 140 };
export const NIGHT_EVERY_POINTS = 700;
export const ACCENT = "#ff5a1f";

export interface Palette {
  paper: string;
  ink: string;
  faint: string;
}

export const DAY: Palette = { paper: "#f7f7f7", ink: "#535353", faint: "#d8d8d8" };
export const NIGHT: Palette = { paper: "#16161a", ink: "#e8e8e8", faint: "#3a3a42" };

export interface RenderOptions {
  /** A second dino (the fly) drawn translucent in the accent colour on the same track. */
  ghost?: GameState | null;
  /** Draw the dino of `state` in the accent colour (used when the main view IS the fly). */
  accentDino?: boolean;
  showBoxes?: boolean;
  label?: string;
  ghostLabel?: string;
}

/** Minimal subset of CanvasRenderingContext2D we use (lets tests pass a recording fake). */
export type Ctx = Pick<CanvasRenderingContext2D, "fillRect" | "fillText" | "save" | "restore" | "strokeRect"> & {
  fillStyle: string | CanvasGradient | CanvasPattern;
  strokeStyle: string | CanvasGradient | CanvasPattern;
  globalAlpha: number;
  font: string;
};

export function isNight(state: GameState): boolean {
  return Math.floor(score(state) / NIGHT_EVERY_POINTS) % 2 === 1;
}

function drawBitmap(ctx: Ctx, bitmap: readonly string[], x: number, yTop: number, ink: string, paper: string): void {
  for (let r = 0; r < bitmap.length; r++) {
    const row = bitmap[r] as string;
    for (let c = 0; c < row.length; c++) {
      const ch = row[c];
      if (ch === ".") {
        continue;
      }
      ctx.fillStyle = ch === "o" ? paper : ink;
      ctx.fillRect(x + c * CELL_PX, yTop + r * CELL_PX, CELL_PX, CELL_PX);
    }
  }
}

function drawDino(ctx: Ctx, s: GameState, ink: string, paper: string): void {
  const x = CONSTANTS.dino.x;
  const feetY = WORLD.groundY - s.dinoY / CONSTANTS.fp;
  const stride = Math.floor(s.frame / 5) % 2 === 0;
  if (s.crashed) {
    drawBitmap(ctx, DINO_DEAD, x, feetY - DINO_DEAD.length * CELL_PX + 1, ink, paper);
  } else if (s.ducking) {
    const bmp = stride ? DINO_DUCK_A : DINO_DUCK_B;
    drawBitmap(ctx, bmp, x, feetY - bmp.length * CELL_PX + 1, ink, paper);
  } else {
    const bmp = s.jumping || stride ? DINO_RUN_A : DINO_RUN_B;
    drawBitmap(ctx, bmp, x, feetY - bmp.length * CELL_PX + 1, ink, paper);
  }
}

function drawObstacles(ctx: Ctx, s: GameState, p: Palette, showBoxes: boolean): void {
  for (const o of s.obstacles) {
    const x = Math.round(o.x / CONSTANTS.fp);
    const bottom = WORLD.groundY - o.yBottomPx;
    if (o.type === 2) {
      const bmp = Math.floor(s.frame / 10) % 2 === 0 ? PTERO_UP : PTERO_DOWN;
      drawBitmap(ctx, bmp, x, bottom - o.heightPx, p.ink, p.paper);
    } else {
      // cacti are drawn straight from their collision boxes: a stem per unit with two arms
      ctx.fillStyle = p.ink;
      const unit = o.widthPx / o.size;
      for (let k = 0; k < o.size; k++) {
        const [l, m, r] = obstacleBoxes(o.type, 1) as [number[], number[], number[]];
        for (const [bx, by, bw, bh] of [l, m, r] as [number, number, number, number][]) {
          ctx.fillRect(x + k * unit + bx, bottom - by - bh, bw, bh);
        }
        ctx.fillStyle = p.paper;
        ctx.fillRect(x + k * unit + (m[0] as number) + 2, bottom - o.heightPx + 3, 1, o.heightPx - 8);
        ctx.fillStyle = p.ink;
      }
    }
    if (showBoxes) {
      ctx.strokeStyle = ACCENT;
      for (const [bx, by, bw, bh] of obstacleBoxes(o.type, o.size) as [number, number, number, number][]) {
        ctx.strokeRect(x + bx + 0.5, bottom - by - bh + 0.5, bw, bh);
      }
    }
  }
}

function drawScenery(ctx: Ctx, s: GameState, p: Palette): void {
  ctx.fillStyle = p.paper;
  ctx.fillRect(0, 0, WORLD.width, WORLD.height);
  const dist = Math.floor(s.distance / CONSTANTS.fp);
  ctx.fillStyle = p.faint;
  for (let i = 0; i < 4; i++) {
    const cx = WORLD.width - ((Math.floor(dist / 5) + i * 190) % (WORLD.width + 80)) + 40;
    const cy = 24 + ((i * 37) % 40);
    ctx.fillRect(cx, cy, 34, 4);
    ctx.fillRect(cx + 6, cy - 4, 20, 4);
  }
  ctx.fillStyle = p.ink;
  ctx.fillRect(0, WORLD.groundY - 3, WORLD.width, 1);
  for (let i = 0; i < 24; i++) {
    const gx = WORLD.width - ((dist + i * 53) % (WORLD.width + 20));
    ctx.fillRect(gx, WORLD.groundY + ((i * 7) % 5), (i % 3) + 2, 1);
  }
}

export function render(ctx: Ctx, state: GameState, opts: RenderOptions = {}): void {
  const p = isNight(state) ? NIGHT : DAY;
  ctx.save();
  ctx.globalAlpha = 1;
  drawScenery(ctx, state, p);
  drawObstacles(ctx, state, p, opts.showBoxes === true);
  if (opts.ghost) {
    ctx.globalAlpha = 0.55;
    drawDino(ctx, opts.ghost, ACCENT, p.paper);
    ctx.globalAlpha = 1;
  }
  drawDino(ctx, state, opts.accentDino ? ACCENT : p.ink, p.paper);
  ctx.font = "10px monospace";
  ctx.fillStyle = p.ink;
  ctx.fillText(`${opts.label ?? "SCORE"} ${String(score(state)).padStart(5, "0")}`, WORLD.width - 110, 14);
  if (opts.ghost) {
    ctx.fillStyle = ACCENT;
    ctx.fillText(`${opts.ghostLabel ?? "FLY"} ${String(score(opts.ghost)).padStart(5, "0")}`, WORLD.width - 110, 27);
  }
  ctx.restore();
}
