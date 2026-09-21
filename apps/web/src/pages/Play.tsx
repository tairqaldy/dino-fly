import { createInitialState, type Input } from "@dino-fly/dino-core";
import { useCallback, useEffect, useRef, useState } from "react";
import { Fly3D, type FlyAction, type FlyMood } from "../components/Fly3D.js";
import { GameCanvas } from "../components/GameCanvas.js";
import { Onboarding, OnboardingReplayButton } from "../components/Onboarding.js";
import { api, bundledGhosts, type Ghost, type SubmitResult } from "../lib/api.js";
import type { FeedSnapshot } from "../lib/feed.js";
import { applyPoseAction, type PoseTrackerHandle, startPoseTracker } from "../lib/pose.js";
import { FixedStep, RaceSession } from "../lib/race.js";

type Phase = "idle" | "running" | "over";
type PoseStatus = { phase: "off" | "loading" | "calibrating" | "tracking" | "error"; progress?: number; message?: string };
const IDLE = createInitialState(0);

export function Play({ feed }: { feed: FeedSnapshot }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("dino-fly-name") ?? "";
    } catch {
      return "";
    }
  });
  const [outcome, setOutcome] = useState<{ score: number; flyScore: number | null; submit: SubmitResult | null } | null>(null);
  const [flyAction, setFlyAction] = useState<FlyAction>("idle");
  const [flyMood, setFlyMood] = useState<FlyMood>("neutral");
  const [pose, setPose] = useState<PoseStatus>({ phase: "off" });
  const session = useRef<RaceSession | null>(null);
  const token = useRef<string | null>(null);
  const keys = useRef<Input>({ jump: false, duck: false });
  const tracker = useRef<PoseTrackerHandle | null>(null);
  const selfView = useRef<HTMLVideoElement | null>(null);
  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;

  const start = useCallback(async () => {
    const started = await api.startRun();
    let ghost: Ghost | null = null;
    let seed: number;
    if (started) {
      seed = started.seed;
      token.current = started.runToken;
      ghost = await api.ghost(seed);
    } else {
      const ghosts = await bundledGhosts();
      ghost = ghosts.length > 0 ? (ghosts[Math.floor(Math.random() * ghosts.length)] as Ghost) : null;
      seed = ghost?.seed ?? Math.floor(Math.random() * 1_000_000);
      token.current = null;
    }
    session.current = new RaceSession(seed, ghost && ghost.seed === seed ? ghost.actions : null);
    setOutcome(null);
    setFlyMood("neutral");
    setPhase("running");
  }, []);

  const finish = useCallback(async () => {
    const s = session.current;
    if (!s) return;
    const r = s.result();
    setPhase("over");
    setFlyAction("idle");
    // the fly celebrates when it beat you on the same obstacles, and cries when it did not
    setFlyMood(r.flyScore === null ? "neutral" : r.flyScore > r.score ? "win" : "lose");
    try {
      localStorage.setItem("dino-fly-name", name);
    } catch {
      /* ignore */
    }
    const submit = token.current
      ? await api.submitRun({ seed: r.seed, runToken: token.current, name: name || "anonymous", actions: r.actions, frames: r.frames })
      : null;
    setOutcome({ score: r.score, flyScore: r.flyScore, submit });
  }, [name]);

  // 60 Hz fixed-step loop
  useEffect(() => {
    if (phase !== "running") return;
    const clock = new FixedStep();
    let last = performance.now();
    let raf = 0;
    const loop = (now: number) => {
      const s = session.current;
      if (!s) return;
      for (let n = clock.ticks(now - last); n > 0 && !s.over; n--) s.tick(keys.current);
      last = now;
      // drive the 3D fly from the ghost: exactly the key presses the fly's own motor transducer made
      const g = s.ghost;
      setFlyAction(g ? (g.jumping ? "jump" : g.ducking ? "duck" : "idle") : "idle");
      if (s.over) {
        void finish();
        return;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [phase, finish]);

  // when nobody is racing, the little fly mirrors the live brain: a Giant Fiber spike presses ↑
  useEffect(() => {
    if (phase === "running") return;
    if (feed.frame && feed.frame.gf > 0) {
      setFlyAction("jump");
      const id = setTimeout(() => setFlyAction("idle"), 200);
      return () => clearTimeout(id);
    }
  }, [feed.frame, phase]);

  // keyboard + touch
  useEffect(() => {
    const set = (code: string, down: boolean): boolean => {
      if (code === "Space" || code === "ArrowUp" || code === "KeyW") keys.current = { ...keys.current, jump: down };
      else if (code === "ArrowDown" || code === "KeyS") keys.current = { ...keys.current, duck: down };
      else return false;
      return true;
    };
    const down = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (set(e.code, true)) {
        e.preventDefault();
        if (phase !== "running" && (e.code === "Space" || e.code === "ArrowUp")) void start();
      }
    };
    const up = (e: KeyboardEvent) => void set(e.code, false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [phase, start]);

  useEffect(() => () => tracker.current?.stop(), []);

  const toggleCamera = useCallback(async () => {
    if (tracker.current) {
      tracker.current.stop();
      tracker.current = null;
      keys.current = { jump: false, duck: false };
      setPose({ phase: "off" });
      return;
    }
    try {
      tracker.current = await startPoseTracker({
        onAction: (action) => {
          keys.current = applyPoseAction(action);
          if (action === "jump" && phaseRef.current !== "running") void start();
        },
        onStatus: (s) => setPose(s),
        onStream: (stream) => {
          const v = selfView.current;
          if (v) {
            v.srcObject = stream;
            void v.play();
          }
        },
      });
    } catch (err) {
      setPose({ phase: "error", message: err instanceof Error ? err.message : "camera unavailable" });
      tracker.current = null;
    }
  }, [start]);

  const touch = (down: boolean) => (e: React.TouchEvent | React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const x = "touches" in e ? (e.touches[0]?.clientX ?? e.changedTouches[0]?.clientX ?? 0) : e.clientX;
    const duckSide = x - rect.left < rect.width * 0.3;
    keys.current = duckSide ? { jump: false, duck: down } : { jump: down, duck: false };
    if (down && phase !== "running") void start();
  };

  const getState = useCallback(() => {
    const s = session.current;
    return s ? { state: s.human, options: { ghost: s.ghost, label: "YOU", ghostLabel: "FLY" } } : { state: IDLE, options: { label: "YOU" } };
  }, []);

  const poseLabel =
    pose.phase === "loading"
      ? "loading the pose model…"
      : pose.phase === "calibrating"
        ? `stand still… ${Math.round((pose.progress ?? 0) * 100)} %`
        : pose.phase === "tracking"
          ? `tracking — jump to jump, crouch to duck (${pose.message})`
          : pose.phase === "error"
            ? `camera off: ${pose.message}`
            : "";

  return (
    <section>
      <Onboarding />
      <h1>You vs. a fruit-fly brain</h1>
      <p className="lede">
        The orange dino is driven by the complete <em>Drosophila</em> connectome — 138,639 spiking neurons. It sees each cactus as a
        looming object and jumps when its Giant Fiber escape neuron fires. No neural network was trained on top of it.
      </p>

      <div className="play-grid">
        <div
          className="stage"
          onTouchStart={touch(true)}
          onTouchEnd={touch(false)}
          onMouseDown={touch(true)}
          onMouseUp={touch(false)}
          role="application"
          aria-label="Dino game. Space or tap to jump, arrow down or tap the left edge to duck."
        >
          <GameCanvas state={null} getState={getState} />
          {phase !== "running" && (
            <div className="overlay">
              {phase === "idle" ? <p>press SPACE or tap to run</p> : null}
              {phase === "over" && outcome ? (
                <p>
                  you {outcome.score}
                  {outcome.flyScore !== null ? ` — fly ${outcome.flyScore} on the same track` : ""}
                  {outcome.submit?.accepted ? ` · rank #${outcome.submit.rank ?? "?"}` : ""}
                  <br />
                  <small>SPACE / tap to try again{outcome.submit && !outcome.submit.accepted ? ` · not recorded: ${outcome.submit.reason}` : ""}</small>
                </p>
              ) : null}
            </div>
          )}
        </div>

        <aside className="fly-desk">
          <Fly3D action={flyAction} mood={flyMood} />
          <p className="hint">
            {flyMood === "win" && phase === "over"
              ? "the fly beat you and is very pleased with itself"
              : flyMood === "lose" && phase === "over"
                ? "the fly lost this one"
                : "the fly at its keyboard: a leg presses ↑ the moment the Giant Fiber fires"}
          </p>
        </aside>
      </div>

      <div className="row">
        <label>
          name on the leaderboard{" "}
          <input value={name} maxLength={24} onChange={(e) => setName(e.target.value)} placeholder="anonymous" />
        </label>
        <span className="hint">jump: SPACE / ↑ / tap · duck: ↓ / tap left edge</span>
        <OnboardingReplayButton />
      </div>

      <div className="row">
        <button type="button" onClick={() => void toggleCamera()}>
          {tracker.current || pose.phase !== "off" ? "stop camera" : "play with your body (camera)"}
        </button>
        <video ref={selfView} className={pose.phase === "off" ? "selfview hidden" : "selfview"} muted playsInline />
        {poseLabel ? <span className="hint">{poseLabel}</span> : <span className="hint">everything stays on your machine — no video is uploaded</span>}
      </div>

      <h2>
        The fly, live <span className={feed.online ? "dot on" : "dot"} /> <small>{feed.online ? "brain online" : "brain offline — showing recorded runs only"}</small>
      </h2>
      {feed.state ? (
        <>
          <GameCanvas state={feed.state} options={{ accentDino: true, label: `GEN ${feed.stats?.generation ?? 0}` }} />
          <p className="hint">
            Giant Fiber spikes this frame: <b className={feed.frame && feed.frame.gf > 0 ? "accent" : ""}>{feed.frame?.gf ?? 0}</b> · looming size θ ={" "}
            {feed.frame?.theta.toFixed(1)}° · LPLC2 drive {feed.frame?.rates[1].toFixed(2)} Hz · {feed.framesPerSecond.toFixed(0)} fps
          </p>
        </>
      ) : (
        <p className="hint">
          The live fly runs on Tair&apos;s laptop GPU. When it is off, you still race its best recorded runs (the translucent ghost).
        </p>
      )}
    </section>
  );
}
