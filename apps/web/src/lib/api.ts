/** REST client for apps/api. Every call degrades gracefully: the site must work when the API or the fly is offline. */
import type { ActionLog } from "@dino-fly/dino-core";
import type { LeaderboardUpdate } from "@dino-fly/protocol";
import { CONFIG } from "./config.js";

export interface Ghost {
  seed: number;
  generation: number;
  score: number;
  actions: [number, number][];
}

export interface StartedRun {
  seed: number;
  runToken: string;
}

export interface SubmitResult {
  accepted: boolean;
  score: number;
  rank?: number;
  beatsFly?: boolean;
  reason?: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T | null> {
  if (!CONFIG.apiUrl) return null;
  try {
    const res = await fetch(`${CONFIG.apiUrl}${path}`, { ...init, headers: { "content-type": "application/json" } });
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

/** Ghost runs bundled with the static site (recorded fly runs on DEV seeds), used when the API is unreachable. */
export async function bundledGhosts(): Promise<Ghost[]> {
  try {
    const res = await fetch("./ghosts.json");
    return res.ok ? ((await res.json()) as { ghosts: Ghost[] }).ghosts : [];
  } catch {
    return [];
  }
}

export const api = {
  startRun: () => call<StartedRun>("/api/runs/start", { method: "POST", body: "{}" }),
  submitRun: (run: { seed: number; runToken: string; name: string; actions: ActionLog; frames: number }) =>
    call<SubmitResult>("/api/runs", { method: "POST", body: JSON.stringify(run) }),
  leaderboard: () => call<LeaderboardUpdate>("/api/leaderboard"),
  ghost: (seed?: number) => call<Ghost>(`/api/ghost${seed === undefined ? "" : `?seed=${seed}`}`),
  /** How many validated human runs are in the teaching corpus so far (training from it is always manual). */
  status: () => call<{ flyOnline: boolean; teaching?: { runs: number; seeds: number } }>("/api/status"),
};
