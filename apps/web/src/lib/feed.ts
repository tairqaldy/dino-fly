/** Live fly feed: a reconnecting WebSocket that keeps the latest of each message type in a small store. */
import type { GameState } from "@dino-fly/dino-core";
import { intsToState } from "@dino-fly/dino-core";
import {
  type DopamineEvent,
  type FlyFrame,
  type FlyHello,
  type FlyRunEnd,
  type FlyStats,
  type LabCommand,
  type LeaderboardUpdate,
  makeEnvelope,
  parseEnvelope,
} from "@dino-fly/protocol";
import { useEffect, useSyncExternalStore } from "react";

export interface FeedSnapshot {
  online: boolean;
  hello: FlyHello | null;
  frame: FlyFrame | null;
  state: GameState | null;
  stats: FlyStats | null;
  lastRun: FlyRunEnd | null;
  leaderboard: LeaderboardUpdate | null;
  /** Rolling history (≈ last 6 s) of per-group activity and GF spikes, newest last. */
  history: { activity: number[]; gf: number; rates: [number, number]; dopamine?: number[] }[];
  dopamine: DopamineEvent[];
  framesPerSecond: number;
}

const HISTORY = 360;
export const RETRY_MIN_MS = 3000;
export const RETRY_MAX_MS = 60_000;

/** Exponential backoff for reconnects: 3 s, 6 s, 12 s, … capped at one minute. */
export function nextRetryMs(current: number): number {
  return Math.min(RETRY_MAX_MS, Math.max(RETRY_MIN_MS, current * 2));
}

export class FlyFeed {
  private ws: WebSocket | null = null;
  private listeners = new Set<() => void>();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private retryMs = RETRY_MIN_MS;
  private frameTimes: number[] = [];
  private closed = false;
  snapshot: FeedSnapshot = {
    online: false,
    hello: null,
    frame: null,
    state: null,
    stats: null,
    lastRun: null,
    leaderboard: null,
    history: [],
    dopamine: [],
    framesPerSecond: 0,
  };

  constructor(private readonly url: string) {}

  start(): void {
    this.closed = false;
    this.connect();
  }

  stop(): void {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
  }

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getSnapshot = (): FeedSnapshot => this.snapshot;

  send(command: LabCommand): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(makeEnvelope("lab.command", command)));
    }
  }

  private update(patch: Partial<FeedSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const fn of this.listeners) fn();
  }

  private connect(): void {
    try {
      this.ws = new WebSocket(this.url);
    } catch {
      this.retry();
      return;
    }
    this.ws.onopen = () => {
      this.retryMs = RETRY_MIN_MS;
      this.update({ online: true });
    };
    this.ws.onclose = () => {
      this.update({ online: false });
      this.retry();
    };
    this.ws.onerror = () => this.ws?.close();
    this.ws.onmessage = (ev) => this.onMessage(String(ev.data));
  }

  /** Reconnect with exponential backoff: the brain is often offline for hours, no need to knock every 3 s. */
  private retry(): void {
    if (this.closed) return;
    this.timer = setTimeout(() => this.connect(), this.retryMs);
    this.retryMs = nextRetryMs(this.retryMs);
  }

  private onMessage(raw: string): void {
    const m = parseEnvelope(raw);
    if (!m) return;
    switch (m.type) {
      case "fly.hello":
        this.update({ hello: m.payload });
        break;
      case "fly.frame": {
        const now = performance.now();
        this.frameTimes.push(now);
        while (this.frameTimes.length > 0 && now - (this.frameTimes[0] as number) > 2000) this.frameTimes.shift();
        let state: GameState | null = null;
        try {
          state = intsToState(m.payload.state);
        } catch {
          state = null;
        }
        const entry = {
          activity: m.payload.activity,
          gf: m.payload.gf,
          rates: m.payload.rates,
          ...(m.payload.dopamine ? { dopamine: m.payload.dopamine } : {}),
        };
        this.update({
          frame: m.payload,
          state,
          history: [...this.snapshot.history.slice(-(HISTORY - 1)), entry],
          framesPerSecond: this.frameTimes.length / 2,
        });
        break;
      }
      case "fly.stats":
        this.update({ stats: m.payload });
        break;
      case "fly.run_end":
        this.update({ lastRun: m.payload });
        break;
      case "leaderboard.update":
        this.update({ leaderboard: m.payload });
        break;
      case "dopamine.event":
        this.update({ dopamine: [...this.snapshot.dopamine.slice(-19), m.payload] });
        break;
      default:
        break;
    }
  }
}

let shared: FlyFeed | null = null;

export function useFlyFeed(url: string): [FeedSnapshot, FlyFeed] {
  if (!shared) {
    shared = new FlyFeed(url);
  }
  const feed = shared; // a process-wide singleton, so the effect below runs once
  useEffect(() => {
    feed.start();
    return () => feed.stop();
  }, []);
  return [useSyncExternalStore(feed.subscribe, feed.getSnapshot), feed];
}
