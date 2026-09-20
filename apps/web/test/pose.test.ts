import { describe, expect, it } from "vitest";
import { applyPoseAction, parsePoseMessage } from "../src/lib/pose.js";

const msg = (payload: unknown, type = "human.input") => JSON.stringify({ type, v: 1, ts: 0, payload });

describe("pose input", () => {
  it("accepts only pose human.input messages with a known action", () => {
    expect(parsePoseMessage(msg({ source: "pose", action: "jump" }))).toBe("jump");
    expect(parsePoseMessage(msg({ source: "pose", action: "duck" }))).toBe("duck");
    expect(parsePoseMessage(msg({ source: "pose", action: "release" }))).toBe("release");
    expect(parsePoseMessage(msg({ source: "keyboard", action: "jump" }))).toBeNull();
    expect(parsePoseMessage(msg({ source: "pose", action: "fly" }))).toBeNull();
    expect(parsePoseMessage(msg({ source: "pose", action: "jump" }, "fly.frame"))).toBeNull();
    expect(parsePoseMessage("not json")).toBeNull();
    expect(parsePoseMessage(new ArrayBuffer(4))).toBeNull();
  });

  it("holds the body's action until release", () => {
    expect(applyPoseAction("jump")).toEqual({ jump: true, duck: false });
    expect(applyPoseAction("duck")).toEqual({ jump: false, duck: true });
    expect(applyPoseAction("release")).toEqual({ jump: false, duck: false });
  });
});
