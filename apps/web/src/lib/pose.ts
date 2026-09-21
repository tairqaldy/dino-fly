import type { Input } from "@dino-fly/dino-core";

/** Body control: `brain/pose/pose_input.py` watches a camera and emits `human.input` messages on a local WebSocket. */
export type PoseAction = "jump" | "duck" | "release";

/** Parse one WebSocket frame; anything that is not a pose `human.input` message is ignored. */
export function parsePoseMessage(data: unknown): PoseAction | null {
  if (typeof data !== "string") return null;
  try {
    const msg = JSON.parse(data) as { type?: unknown; payload?: { source?: unknown; action?: unknown } };
    const action = msg.payload?.action;
    if (msg.type !== "human.input" || msg.payload?.source !== "pose") return null;
    return action === "jump" || action === "duck" || action === "release" ? action : null;
  } catch {
    return null;
  }
}

/** A physical jump / crouch is held until the body returns to neutral ("release"). */
export function applyPoseAction(action: PoseAction): Input {
  return { jump: action === "jump", duck: action === "duck" };
}

/** Connects (and keeps reconnecting) to the pose WebSocket. Returns a disposer. */
export function connectPose(url: string, onAction: (a: PoseAction) => void, onStatus: (connected: boolean) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let retryMs = 2000;
  const open = () => {
    if (closed) return;
    ws = new WebSocket(url);
    ws.onopen = () => {
      retryMs = 2000;
      onStatus(true);
    };
    ws.onmessage = (e) => {
      const action = parsePoseMessage(e.data);
      if (action) onAction(action);
    };
    ws.onclose = () => {
      onStatus(false);
      if (!closed) {
        timer = setTimeout(open, retryMs);
        retryMs = Math.min(30_000, retryMs * 2); // the tracker is usually just not running: back off
      }
    };
    ws.onerror = () => ws?.close();
  };
  open();
  return () => {
    closed = true;
    clearTimeout(timer);
    ws?.close();
  };
}
