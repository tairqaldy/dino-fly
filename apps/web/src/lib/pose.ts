import type { Input } from "@dino-fly/dino-core";

/**
 * Body control. The laptop camera runs MediaPipe Pose in the browser; the detector below is a direct port of
 * `brain/pose/pose_input.py` (same thresholds, same state machine) so both paths behave identically.
 * Thresholds are in body-height units, so they do not depend on how far you sit from the camera.
 */
export type PoseAction = "jump" | "duck" | "release";

export interface PoseSample {
  /** seconds */
  t: number;
  /** normalised image coordinates, 0 = top, 1 = bottom */
  hipY: number;
  noseY: number;
}

export const POSE_DEFAULTS = { calibrationS: 2.0, jumpRise: 0.12, duckDrop: 0.3 } as const;

export class ActionDetector {
  readonly calibrationS: number;
  readonly jumpRise: number;
  readonly duckDrop: number;
  private calib: PoseSample[] = [];
  baseline: PoseSample | null = null;
  state: PoseAction = "release";

  constructor(opts: Partial<typeof POSE_DEFAULTS> = {}) {
    const { calibrationS, jumpRise, duckDrop } = { ...POSE_DEFAULTS, ...opts };
    this.calibrationS = calibrationS;
    this.jumpRise = jumpRise;
    this.duckDrop = duckDrop;
  }

  /** Fraction of the calibration that is done (0…1); the UI shows this as a countdown. */
  get calibrationProgress(): number {
    if (this.baseline) return 1;
    const first = this.calib[0];
    const last = this.calib[this.calib.length - 1];
    if (!first || !last) return 0;
    return Math.min(1, (last.t - first.t) / this.calibrationS);
  }

  /** Returns an action only when it *changes*, exactly like the Python detector. */
  update(s: PoseSample): PoseAction | null {
    if (!this.baseline) {
      this.calib.push(s);
      const first = this.calib[0] as PoseSample;
      if (s.t - first.t >= this.calibrationS) {
        const n = this.calib.length;
        this.baseline = {
          t: s.t,
          hipY: this.calib.reduce((a, x) => a + x.hipY, 0) / n,
          noseY: this.calib.reduce((a, x) => a + x.noseY, 0) / n,
        };
      }
      return null;
    }
    const body = Math.max(this.baseline.hipY - this.baseline.noseY, 1e-3); // nose-to-hip distance = one "body unit"
    const rise = (this.baseline.hipY - s.hipY) / body;
    const drop = (s.noseY - this.baseline.noseY) / body;
    const next: PoseAction = rise > this.jumpRise ? "jump" : drop > this.duckDrop ? "duck" : "release";
    if (next === this.state) return null;
    this.state = next;
    return next;
  }

  /** Forget the baseline and calibrate again (the "recalibrate" button). */
  reset(): void {
    this.calib = [];
    this.baseline = null;
    this.state = "release";
  }
}

/** A physical jump / crouch is held until the body returns to neutral ("release"). */
export function applyPoseAction(action: PoseAction): Input {
  return { jump: action === "jump", duck: action === "duck" };
}

// MediaPipe Pose landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
const NOSE = 0;
const LEFT_HIP = 23;
const RIGHT_HIP = 24;
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

export interface PoseTrackerHandle {
  stop(): void;
  recalibrate(): void;
}

export interface PoseTrackerCallbacks {
  onAction(action: PoseAction): void;
  onStatus(status: { phase: "loading" | "calibrating" | "tracking" | "error"; progress?: number; message?: string }): void;
  /** Receives the camera stream so the page can show a small self-view. */
  onStream?(stream: MediaStream): void;
}

/**
 * Opens the laptop camera and tracks the body. Everything stays in the browser: no frame ever leaves the machine,
 * only the resulting jump / duck actions are used. Returns a handle to stop it again.
 */
export async function startPoseTracker(cb: PoseTrackerCallbacks): Promise<PoseTrackerHandle> {
  const detector = new ActionDetector();
  let stopped = false;
  let raf = 0;
  let stream: MediaStream | null = null;

  cb.onStatus({ phase: "loading" });
  const { FilesetResolver, PoseLandmarker } = await import("@mediapipe/tasks-vision");
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
  const landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  cb.onStream?.(stream);
  const video = document.createElement("video");
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();

  cb.onStatus({ phase: "calibrating", progress: 0, message: "stand still" });
  let lastVideoTime = -1;
  const tick = () => {
    if (stopped) return;
    raf = requestAnimationFrame(tick);
    if (video.currentTime === lastVideoTime || video.readyState < 2) return;
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    const lm = result.landmarks?.[0];
    if (!lm) return;
    const nose = lm[NOSE];
    const hipL = lm[LEFT_HIP];
    const hipR = lm[RIGHT_HIP];
    if (!nose || !hipL || !hipR) return;
    const sample: PoseSample = { t: performance.now() / 1000, hipY: (hipL.y + hipR.y) / 2, noseY: nose.y };
    const action = detector.update(sample);
    if (detector.baseline) {
      if (action) cb.onAction(action);
      cb.onStatus({ phase: "tracking", message: detector.state });
    } else {
      cb.onStatus({ phase: "calibrating", progress: detector.calibrationProgress, message: "stand still" });
    }
  };
  raf = requestAnimationFrame(tick);

  return {
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      for (const track of stream?.getTracks() ?? []) track.stop();
      landmarker.close();
    },
    recalibrate() {
      detector.reset();
      cb.onStatus({ phase: "calibrating", progress: 0, message: "stand still" });
    },
  };
}
