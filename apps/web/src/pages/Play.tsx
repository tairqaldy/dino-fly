import { createInitialState, type GameState, type Input, score as scoreOf } from "@dino-fly/dino-core";
import { useCallback, useEffect, useRef, useState } from "react";
import { Fly3D, type FlyAction, type FlyMood } from "../components/Fly3D.js";
import { GameCanvas } from "../components/GameCanvas.js";
import { NameGate, Onboarding, OnboardingReplayButton } from "../components/Onboarding.js";
import { api, bundledGhosts, type Ghost, type SubmitResult } from "../lib/api.js";
import type { FeedSnapshot } from "../lib/feed.js";
import { applyPoseAction, type PoseTrackerHandle, startPoseTracker } from "../lib/pose.js";
import { FixedStep, RaceSession } from "../lib/race.js";
import { FlyReplay } from "../lib/replay.js";

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
  const [flyMood, setFlyMood] = useState<FlyMood>("neutral");
  const [pose, setPose] = useState<PoseStatus>({ phase: "off" });
  const [flyLabel, setFlyLabel] = useState<{ live: boolean; score: number }>({ live: false, score: 0 });
  const session = useRef<RaceSession | null>(null);
  const token = useRef<string | null>(null);
  const keys = useRef<Input>({ jump: false, duck: false });
  const tracker = useRef<PoseTrackerHandle | null>(null);
  const selfView = useRef<HTMLVideoElement | null>(null);
  const replayer = useRef(new FlyReplay());
  const flyNow = useRef<GameState>(IDLE);
  const flyAction = useRef<FlyAction>("idle");
  const [flyActionState, setFlyActionState] = useState<FlyAction>("idle");
  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;

  // recorded fly runs keep the fly's screen alive whenever the live brain is not connected
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const bundled = await bundledGhosts();
      const fresh = await api.recentFlyRuns(12);
      const runs = [...(fresh ?? []), ...bundled];
      if (!cancelled && runs.length > 0) replayer.current.setRuns(runs as Ghost[]);
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

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

  // one 60 Hz clock drives both screens: the human's game and the fly's (live feed or recorded run)
  useEffect(() => {
    const clock = new FixedStep();
    let last = performance.now();
    let raf = 0;
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const ticks = clock.ticks(now - last);
      last = now;
      if (ticks === 0) return;
      const s = session.current;
      for (let n = ticks; n > 0; n--) {
        if (s && phaseRef.current === "running" && !s.over) s.tick(keys.current);
        if (!feed.state) replayer.current.tick();
      }
      const fly = feed.state ?? (replayer.current.hasRuns ? replayer.current.state : null);
      flyNow.current = fly ?? IDLE;
      const action: FlyAction = fly ? (fly.jumping ? "jump" : fly.ducking ? "duck" : "idle") : "idle";
      if (action !== flyAction.current) {
        flyAction.current = action;
        setFlyActionState(action);
      }
      setFlyLabel((prev) => {
        const live = Boolean(feed.state);
        const sc = fly ? scoreOf(fly) : 0;
        return prev.live === live && prev.score === sc ? prev : { live, score: sc };
      });
      if (s && phaseRef.current === "running" && s.over) void finish();
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [feed.state, finish]);

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

  const humanState = useCallback(
    () => ({ state: session.current?.human ?? IDLE, options: { label: (name || "YOU").toUpperCase() } }),
    [name],
  );
  const flyField = useCallback(() => ({ state: flyNow.current, options: { accentDino: true, label: `FLY${feed.stats ? ` · GEN ${feed.stats.generation}` : ""}` } }), [feed.stats]);

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
      <Onboarding name={name} onName={setName} />
      <NameGate name={name} onName={setName} />
      <h1>You vs. a fruit-fly brain</h1>
      <p className="lede">
        Two screens, the same game. On top the <em>Drosophila</em> connectome plays — 138,639 spiking neurons, jumping when its
        Giant Fiber escape neuron fires. Below, you.
      </p>

      <div className="screens">
        <div className="screen">
          <div className="screen-head">
            <span className="accent">FLY</span>
            <span className={flyLabel.live ? "dot on" : "dot"} />
            <small>{flyLabel.live ? "live brain — playing right now" : replayer.current.hasRuns ? "recorded run — the brain is offline" : "no runs yet"}</small>
            <b className="accent">{flyLabel.score}</b>
          </div>
          <GameCanvas state={null} getState={flyField} />
          {feed.frame ? (
            <p className="hint">
              Giant Fiber spikes this frame: <b className={feed.frame.gf > 0 ? "accent" : ""}>{feed.frame.gf}</b> · looming size θ ={" "}
              {feed.frame.theta.toFixed(1)}° · LPLC2 drive {feed.frame.rates[1].toFixed(2)} Hz · {feed.framesPerSecond.toFixed(0)} fps
            </p>
          ) : (
            <p className="hint">Recorded runs are replayed by the same deterministic engine that validates the leaderboard.</p>
          )}
        </div>

        <div className="screen">
          <div className="screen-head">
            <span>{(name || "YOU").toUpperCase()}</span>
            <small>{phase === "running" ? "your run" : "press SPACE or tap"}</small>
            <b>{session.current ? scoreOf(session.current.human) : 0}</b>
          </div>
          <div
            className="stage"
            onTouchStart={touch(true)}
            onTouchEnd={touch(false)}
            onMouseDown={touch(true)}
            onMouseUp={touch(false)}
            role="application"
            aria-label="Dino game. Space or tap to jump, arrow down or tap the left edge to duck."
          >
            <GameCanvas state={null} getState={humanState} />
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
          <p className="hint">jump: SPACE / ↑ / tap · duck: ↓ / tap left edge · your inputs are saved so the fly can be taught from them</p>
        </div>
      </div>

      <div className="play-grid">
        <div>
          <div className="row">
            <label>
              name{" "}
              <input value={name} maxLength={24} onChange={(e) => setName(e.target.value)} placeholder="anonymous" />
            </label>
            <OnboardingReplayButton />
          </div>
          <div className="row">
            <button type="button" onClick={() => void toggleCamera()}>
              {pose.phase !== "off" ? "stop camera" : "play with your body (camera)"}
            </button>
            <video ref={selfView} className={pose.phase === "off" ? "selfview hidden" : "selfview"} muted playsInline />
            <span className="hint">{poseLabel || "everything stays on your machine — no video is uploaded"}</span>
          </div>
        </div>
        <aside className="fly-desk">
          <Fly3D action={flyActionState} mood={flyMood} />
          <p className="hint">
            {flyMood === "win" && phase === "over"
              ? "the fly beat you and is very pleased with itself"
              : flyMood === "lose" && phase === "over"
                ? "the fly lost this one"
                : "a leg presses ↑ the moment the Giant Fiber fires"}
          </p>
        </aside>
      </div>
    </section>
  );
}
