/**
 * dino-fly realtime protocol v1. The JSON Schema in `schemas/messages.schema.json` is the single source of truth;
 * these types and the pydantic models in `brain/flybrain/protocol.py` mirror it, and both sides validate the shared
 * examples in `examples/messages.json` against the schema in their test suites.
 */
import schema from "../schemas/messages.schema.json";

export const PROTOCOL_VERSION = 1 as const;
export const MESSAGE_SCHEMA = schema;

export interface FlyHello {
  workerId: string;
  connectome: string;
  neurons: number;
  engineVersion: number;
  transducerVersion: number;
  generation: number;
  bioMsPerFrame: number;
  gpu?: string;
  groups: string[];
}

export interface FlyFrame {
  seed: number;
  /** Canonical dino-core state (`stateToInts`). */
  state: number[];
  gf: number;
  jump: boolean;
  theta: number;
  thetaDot: number;
  /** [LC4, LPLC2] commanded rates in Hz. */
  rates: [number, number];
  activity: number[];
  dopamine?: number[];
}

export interface FlyRunEnd {
  seed: number;
  generation: number;
  score: number;
  frames: number;
  cleared: number;
  deathType: number;
  jumps: number;
  actions: [number, number][];
}

export interface FlyStats {
  generation: number;
  gamesPlayed: number;
  bestScore: number;
  meanScoreRecent: number;
  totalJumps: number;
  totalDeaths?: number;
  realtimeFactor?: number;
  learningCurve: { generation: number; heldoutMean: number }[];
}

export interface FlyStatus {
  online: boolean;
  generation?: number;
}

export interface DopamineEvent {
  kind: "reward" | "punish";
  magnitude: number;
  source: "game" | "human_button" | "hardware";
  seed?: number;
  frame?: number;
}

export interface HumanInput {
  source: "keyboard" | "touch" | "pose";
  action: "jump" | "duck" | "release";
}

export interface LabCommand {
  command: "start" | "stop" | "set_seed" | "set_generation" | "reward" | "punish";
  value?: number;
}

export interface LeaderboardUpdate {
  humans: { name: string; score: number }[];
  fly: { generation: number; bestScore: number }[];
  flyPercentile?: number;
}

export interface PayloadByType {
  "fly.hello": FlyHello;
  "fly.frame": FlyFrame;
  "fly.run_end": FlyRunEnd;
  "fly.stats": FlyStats;
  "fly.status": FlyStatus;
  "dopamine.event": DopamineEvent;
  "human.input": HumanInput;
  "lab.command": LabCommand;
  "leaderboard.update": LeaderboardUpdate;
}

export type MessageType = keyof PayloadByType;

export interface Envelope<T extends MessageType = MessageType> {
  type: T;
  v: typeof PROTOCOL_VERSION;
  ts: number;
  payload: PayloadByType[T];
}

export type AnyMessage = { [T in MessageType]: Envelope<T> }[MessageType];

export const MESSAGE_TYPES: readonly MessageType[] = [
  "fly.hello",
  "fly.frame",
  "fly.run_end",
  "fly.stats",
  "fly.status",
  "dopamine.event",
  "human.input",
  "lab.command",
  "leaderboard.update",
];

export function makeEnvelope<T extends MessageType>(type: T, payload: PayloadByType[T], ts: number = Date.now()): Envelope<T> {
  return { type, v: PROTOCOL_VERSION, ts, payload };
}

/** Cheap structural parse for untrusted input (full validation = the JSON Schema). Returns null when malformed. */
export function parseEnvelope(raw: string): AnyMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null) {
    return null;
  }
  const m = data as Record<string, unknown>;
  if (m.v !== PROTOCOL_VERSION || typeof m.ts !== "number" || typeof m.payload !== "object" || m.payload === null) {
    return null;
  }
  if (typeof m.type !== "string" || !(MESSAGE_TYPES as readonly string[]).includes(m.type)) {
    return null;
  }
  return data as AnyMessage;
}
