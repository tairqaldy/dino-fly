import { replay, score } from "@dino-fly/dino-core";
import { describe, expect, it } from "vitest";
import type { Ghost } from "../src/lib/api.js";
import { FlyReplay } from "../src/lib/replay.js";

const ghost = (seed: number): Ghost => ({ seed, generation: 0, score: 0, actions: [[250, 1], [260, 0]] });

describe("FlyReplay", () => {
  it("reproduces the recorded run frame for frame", () => {
    const player = new FlyReplay([ghost(3)]);
    let ticks = 0;
    while (!player.state.crashed && ticks < 5000) {
      player.tick();
      ticks += 1;
    }
    const reference = replay(3, ghost(3).actions, 5000);
    expect(player.state.crashed).toBe(true);
    expect(player.state.frame).toBe(reference.frame);
    expect(score(player.state)).toBe(score(reference));
  });

  it("moves on to the next recorded run after the crash", () => {
    const player = new FlyReplay([ghost(3), ghost(11)]);
    while (!player.state.crashed) player.tick();
    const crashedSeed = player.current?.seed;
    for (let i = 0; i < 50; i++) player.tick();
    expect(player.current?.seed).not.toBe(crashedSeed);
    expect(player.state.crashed).toBe(false);
    expect(player.state.frame).toBeLessThan(10);
  });

  it("is inert without runs and adopts them later", () => {
    const player = new FlyReplay([]);
    expect(player.hasRuns).toBe(false);
    expect(player.tick().frame).toBe(0);
    player.setRuns([ghost(7)]);
    expect(player.hasRuns).toBe(true);
    expect(player.tick().frame).toBe(1);
  });
});
