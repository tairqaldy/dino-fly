import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";
import { MESSAGE_SCHEMA, MESSAGE_TYPES, makeEnvelope, parseEnvelope } from "../src/index.js";

const examples = JSON.parse(readFileSync(new URL("../examples/messages.json", import.meta.url), "utf8")) as unknown[];
const validate = new Ajv2020({ allErrors: true, strict: false }).compile(MESSAGE_SCHEMA);

describe("protocol v1", () => {
  it("has one valid example per message type", () => {
    const seen = new Set<string>();
    for (const ex of examples) {
      expect(validate(ex), JSON.stringify(validate.errors)).toBe(true);
      seen.add((ex as { type: string }).type);
      expect(parseEnvelope(JSON.stringify(ex))).not.toBeNull();
    }
    expect([...seen].sort()).toEqual([...MESSAGE_TYPES].sort());
  });

  it("rejects malformed messages", () => {
    const bad = [
      { type: "fly.frame", v: 1, ts: 1, payload: { seed: 1 } },
      { type: "nope", v: 1, ts: 1, payload: {} },
      { type: "fly.status", v: 2, ts: 1, payload: { online: true } },
      { type: "dopamine.event", v: 1, ts: 1, payload: { kind: "bribe", magnitude: 1, source: "game" } },
      { type: "fly.status", v: 1, ts: 1, payload: { online: true }, extra: 1 },
    ];
    for (const m of bad) {
      expect(validate(m)).toBe(false);
    }
    expect(parseEnvelope("not json")).toBeNull();
    expect(parseEnvelope("42")).toBeNull();
    expect(parseEnvelope(JSON.stringify({ type: "nope", v: 1, ts: 1, payload: {} }))).toBeNull();
    expect(parseEnvelope(JSON.stringify({ type: "fly.status", v: 2, ts: 1, payload: {} }))).toBeNull();
  });

  it("makeEnvelope produces schema-valid messages", () => {
    const m = makeEnvelope("fly.status", { online: false }, 123);
    expect(m).toEqual({ type: "fly.status", v: 1, ts: 123, payload: { online: false } });
    expect(validate(m)).toBe(true);
    expect(typeof makeEnvelope("human.input", { source: "keyboard", action: "jump" }).ts).toBe("number");
  });
});
