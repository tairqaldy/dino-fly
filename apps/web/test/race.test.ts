import { replay, score } from "@dino-fly/dino-core";
import { describe, expect, it } from "vitest";
import { FixedStep, RaceSession } from "../src/lib/race.js";

describe("RaceSession", () => {
  it("records an action log that the server-side replay reproduces exactly", () => {
    const s = new RaceSession(7, null);
    while (!s.over && s.human.frame < 3000) {
      s.tick({ jump: s.human.frame % 41 < 6, duck: false });
    }
    const r = s.result();
    const replayed = replay(7, r.actions, 3000);
    expect(score(replayed)).toBe(r.score);
    expect(replayed.frame).toBe(r.frames);
    expect(r.flyScore).toBeNull();
  });

  it("steps the ghost from its recorded actions on the same seed, independently of the human", () => {
    const flyActions: [number, number][] = [[250, 1], [260, 0]];
    const s = new RaceSession(3, flyActions);
    for (let f = 0; f < 600; f++) s.tick({ jump: false, duck: false });
    const ghostAlone = replay(3, flyActions, 600);
    expect(s.ghost?.frame).toBe(ghostAlone.frame);
    expect(s.result().flyScore).toBe(score(ghostAlone));
    expect(s.over).toBe(true);
  });
});

describe("FixedStep", () => {
  it("turns variable frame deltas into whole 60 Hz ticks and caps catch-up", () => {
    const clock = new FixedStep();
    let ticks = 0;
    for (let i = 0; i < 60; i++) ticks += clock.ticks(1000 / 60);
    expect(ticks).toBeGreaterThanOrEqual(59);
    expect(ticks).toBeLessThanOrEqual(60);
    const caught = new FixedStep().ticks(10_000); // a 10 s stall must not fast-forward the game
    expect(caught).toBeGreaterThanOrEqual(4);
    expect(caught).toBeLessThanOrEqual(5);
    expect(new FixedStep().ticks(5)).toBe(0);
  });
});
