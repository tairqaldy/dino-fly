import { describe, expect, it } from "vitest";
import { nextRetryMs, RETRY_MAX_MS, RETRY_MIN_MS } from "../src/lib/feed.js";

describe("feed reconnect backoff", () => {
  it("doubles from 3 s and never exceeds one minute", () => {
    const delays: number[] = [];
    let d = RETRY_MIN_MS;
    for (let i = 0; i < 8; i++) {
      delays.push(d);
      d = nextRetryMs(d);
    }
    expect(delays.slice(0, 5)).toEqual([3000, 6000, 12_000, 24_000, 48_000]);
    expect(Math.max(...delays)).toBe(RETRY_MAX_MS);
    expect(nextRetryMs(0)).toBe(RETRY_MIN_MS);
  });
});
