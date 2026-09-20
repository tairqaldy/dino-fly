/**
 * Integer helpers. The engine state is all-integer so that the TypeScript engine and the Python port
 * (`brain/flybrain/dino_core.py`) produce byte-identical state sequences.
 *
 * Rules for physics code: no floats, no raw `/`, `%`, `Math.round`, `Math.trunc`, `|0` — only `idiv` / `imod`.
 * (JS `%` follows the dividend's sign, Python's follows the divisor's; `|0` wraps at 2^31.)
 */

/** Fixed-point scale: 1 px = 1000 units. */
export const FP = 1000;

function assertSafe(name: string, x: number): void {
  if (!Number.isSafeInteger(x)) {
    throw new RangeError(`${name} must be a safe integer, got ${x}`);
  }
}

/** Floor division for a positive divisor. Exact for all safe integers. Python equivalent: `a // b`. */
export function idiv(a: number, b: number): number {
  assertSafe("idiv a", a);
  assertSafe("idiv b", b);
  if (b <= 0) {
    throw new RangeError(`idiv divisor must be positive, got ${b}`);
  }
  // Exact for safe integers: a correctly rounded a/b can only round across an integer boundary when
  // |a| > 2^53 (checked against BigInt in test/fixed.test.ts).
  return Math.floor(a / b);
}

/** Non-negative remainder for a positive divisor. Python equivalent: `a % b`. */
export function imod(a: number, b: number): number {
  return a - b * idiv(a, b);
}
