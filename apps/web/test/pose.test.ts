import { describe, expect, it } from "vitest";
import { ActionDetector, applyPoseAction, POSE_DEFAULTS, type PoseSample } from "../src/lib/pose.js";

/** Same fixture as brain/tests/test_pose.py: standing still, then a jump, then a crouch. */
const still = (t: number): PoseSample => ({ t, hipY: 0.8, noseY: 0.2 });

function calibrate(d: ActionDetector): void {
  for (let t = 0; t <= POSE_DEFAULTS.calibrationS + 0.1; t += 0.1) expect(d.update(still(t))).toBeNull();
  expect(d.baseline).not.toBeNull();
}

describe("ActionDetector (port of brain/pose/pose_input.py)", () => {
  it("calibrates on the first two seconds and reports progress", () => {
    const d = new ActionDetector();
    expect(d.calibrationProgress).toBe(0);
    d.update(still(0));
    d.update(still(1));
    expect(d.calibrationProgress).toBeCloseTo(0.5, 5);
    calibrate(d);
    expect(d.calibrationProgress).toBe(1);
    expect(d.baseline?.hipY).toBeCloseTo(0.8, 6);
  });

  it("fires jump when the hips rise by more than 12 % of a body unit, and only on change", () => {
    const d = new ActionDetector();
    calibrate(d);
    const body = 0.6; // 0.8 hip − 0.2 nose
    expect(d.update({ t: 3, hipY: 0.8 - 0.11 * body, noseY: 0.2 })).toBeNull(); // below threshold
    expect(d.update({ t: 3.1, hipY: 0.8 - 0.2 * body, noseY: 0.1 })).toBe("jump");
    expect(d.update({ t: 3.2, hipY: 0.8 - 0.25 * body, noseY: 0.1 })).toBeNull(); // still jumping, no repeat
    expect(d.update({ t: 3.5, hipY: 0.8, noseY: 0.2 })).toBe("release");
  });

  it("fires duck when the nose drops by more than 30 % of a body unit", () => {
    const d = new ActionDetector();
    calibrate(d);
    const body = 0.6;
    expect(d.update({ t: 3, hipY: 0.8, noseY: 0.2 + 0.2 * body })).toBeNull();
    expect(d.update({ t: 3.1, hipY: 0.82, noseY: 0.2 + 0.4 * body })).toBe("duck");
    expect(d.update({ t: 3.6, hipY: 0.8, noseY: 0.2 })).toBe("release");
  });

  it("recalibrates on demand", () => {
    const d = new ActionDetector();
    calibrate(d);
    d.reset();
    expect(d.baseline).toBeNull();
    expect(d.state).toBe("release");
    expect(d.update(still(10))).toBeNull();
  });

  it("holds the body's action until release", () => {
    expect(applyPoseAction("jump")).toEqual({ jump: true, duck: false });
    expect(applyPoseAction("duck")).toEqual({ jump: false, duck: true });
    expect(applyPoseAction("release")).toEqual({ jump: false, duck: false });
  });
});
