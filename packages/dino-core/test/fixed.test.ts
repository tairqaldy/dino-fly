import { describe, expect, it } from "vitest";
import { FP, idiv, imod } from "../src/fixed.js";

describe("fixed-point helpers", () => {
  it("uses 1000 units per pixel", () => {
    expect(FP).toBe(1000);
  });

  it("idiv is floor division (Python //) for positive divisors", () => {
    expect(idiv(7, 2)).toBe(3);
    expect(idiv(-7, 2)).toBe(-4);
    expect(idiv(0, 5)).toBe(0);
    expect(idiv(-1, 1000)).toBe(-1);
    expect(idiv(999, 1000)).toBe(0);
    expect(idiv(-1000, 1000)).toBe(-1);
  });

  it("imod is the non-negative remainder (Python %)", () => {
    expect(imod(7, 2)).toBe(1);
    expect(imod(-7, 2)).toBe(1);
    expect(imod(-1, 1000)).toBe(999);
    expect(imod(4294967295, 4294967296)).toBe(4294967295);
  });

  it("is exact near 2^53", () => {
    const a = Number.MAX_SAFE_INTEGER; // 2^53 - 1
    for (const b of [1, 2, 3, 7, 1000, 4294967296, 94906267]) {
      const q = idiv(a, b);
      const r = imod(a, b);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThan(b);
      expect(BigInt(q) * BigInt(b) + BigInt(r)).toBe(BigInt(a));
    }
  });

  it("agrees with BigInt floor division on adversarial large operands", () => {
    // a = m*b - r just below a multiple of b is where a rounded double quotient could cross an integer.
    let x = 0x9e3779b9;
    const next = (): number => {
      x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
      return x;
    };
    for (let i = 0; i < 20000; i++) {
      const b = (next() % 134217728) + 1; // up to 2^27
      const m = Math.floor((Number.MAX_SAFE_INTEGER - 1) / b) - (next() % 5);
      const r = next() % Math.min(b, 4);
      const sign = next() % 2 === 0 ? 1 : -1;
      const a = sign * (m * b - r);
      if (!Number.isSafeInteger(a)) continue;
      const qBig = BigInt(a) / BigInt(b);
      const floorBig = BigInt(a) < 0n && qBig * BigInt(b) !== BigInt(a) ? qBig - 1n : qBig;
      expect(BigInt(idiv(a, b))).toBe(floorBig);
    }
  });

  it("satisfies a = q*b + r on a sweep incl. negatives", () => {
    for (let a = -50; a <= 50; a++) {
      for (let b = 1; b <= 9; b++) {
        expect(idiv(a, b) * b + imod(a, b)).toBe(a);
      }
    }
  });

  it("rejects floats, unsafe integers and non-positive divisors", () => {
    expect(() => idiv(1.5, 2)).toThrow(RangeError);
    expect(() => idiv(1, 0)).toThrow(RangeError);
    expect(() => idiv(1, -3)).toThrow(RangeError);
    expect(() => idiv(2 ** 53, 3)).toThrow(RangeError);
    expect(() => idiv(3, 2.5)).toThrow(RangeError);
  });
});
