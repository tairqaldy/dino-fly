import type { LeaderboardUpdate } from "@dino-fly/protocol";
import { marked } from "marked";
import { useEffect, useState } from "react";
import { api } from "../lib/api.js";
import type { FeedSnapshot, FlyFeed } from "../lib/feed.js";

export function Leaderboard({ feed }: { feed: FeedSnapshot }) {
  const [board, setBoard] = useState<LeaderboardUpdate | null>(null);
  const [teaching, setTeaching] = useState<{ runs: number; seeds: number } | null>(null);
  useEffect(() => {
    void api.leaderboard().then(setBoard);
    void api.status().then((s) => setTeaching(s?.teaching ?? null));
  }, []);
  const data = feed.leaderboard ?? board;
  if (!data) {
    return (
      <section>
        <h1>Humans vs. fly</h1>
        <p className="lede">The leaderboard API is not reachable right now.</p>
      </section>
    );
  }
  const flyBest = Math.max(0, ...data.fly.map((f) => f.bestScore));
  return (
    <section>
      <h1>Humans vs. fly</h1>
      {data.flyPercentile !== undefined && (
        <p className="headline">
          the fly beats <b className="accent">{data.flyPercentile.toFixed(1)}%</b> of human runs
        </p>
      )}
      {teaching && (
        <p className="hint">
          Teaching corpus: <b>{teaching.runs}</b> validated human runs on <b>{teaching.seeds}</b> obstacle courses. Every run you
          play is kept. Nothing trains on them automatically — a new generation of the fly is trained by hand and rolled out
          with its own pre-registered evaluation.
        </p>
      )}
      <div className="grid2">
        <div>
          <h2>Humans</h2>
          <ol className="board">
            {data.humans.map((h, i) => (
              // biome-ignore lint/suspicious/noArrayIndexKey: rank IS the identity of a leaderboard row
              <li key={`${h.name}-${i}`} className={h.score < flyBest ? "below-fly" : ""}>
                <span>{h.name}</span>
                <b>{h.score}</b>
              </li>
            ))}
          </ol>
          {data.humans.length === 0 && <p className="hint">No validated human runs yet — be the first.</p>}
        </div>
        <div>
          <h2>The fly, per generation</h2>
          <ol className="board">
            {data.fly.map((f) => (
              <li key={f.generation}>
                <span>generation {f.generation}</span>
                <b className="accent">{f.bestScore}</b>
              </li>
            ))}
          </ol>
          {data.fly.length === 0 && <p className="hint">No recorded fly runs yet.</p>}
        </div>
      </div>
      <p className="hint">Every human score is re-computed on the server by replaying the submitted key presses in the same deterministic engine.</p>
    </section>
  );
}

export function Research() {
  const [html, setHtml] = useState<string>("");
  useEffect(() => {
    void fetch("./research/RESEARCH.md")
      .then((r) => (r.ok ? r.text() : "# Research\n\nnot bundled in this build"))
      .then((md) => setHtml(marked.parse(md.replaceAll("](figures/", "](./research/figures/"), { async: false })));
  }, []);
  // biome-ignore lint/security/noDangerouslySetInnerHtml: the markdown is our own repository file, bundled at build time
  return <section className="prose" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Lab({ feed, control }: { feed: FeedSnapshot; control: FlyFeed }) {
  const [seed, setSeed] = useState(1_000_000);
  return (
    <section>
      <h1>
        Lab <span className={feed.online ? "dot on" : "dot"} />
      </h1>
      <p className="lede">Local controls for the brain worker. Commands travel over the same WebSocket as the feed.</p>
      <div className="row">
        <button type="button" onClick={() => control.send({ command: "start" })}>
          ▶ run
        </button>
        <button type="button" onClick={() => control.send({ command: "stop" })}>
          ❚❚ pause
        </button>
        <label>
          next seed <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </label>
        <button type="button" onClick={() => control.send({ command: "set_seed", value: seed })}>
          set
        </button>
      </div>
      <div className="row">
        <button type="button" className="reward" onClick={() => control.send({ command: "reward", value: 1 })}>
          + reward (PAM)
        </button>
        <button type="button" className="punish" onClick={() => control.send({ command: "punish", value: 1 })}>
          − punish (PPL1)
        </button>
      </div>
      <h2>Dopamine events</h2>
      <ul className="events">
        {feed.dopamine
          .slice()
          .reverse()
          .map((d, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: append-only event log
            <li key={`${d.frame}-${i}`}>
              {d.kind} × {d.magnitude} · {d.source} · seed {d.seed ?? "–"} frame {d.frame ?? "–"}
            </li>
          ))}
      </ul>
      {feed.dopamine.length === 0 && <p className="hint">none yet</p>}
      <h2>Last run</h2>
      <pre>{feed.lastRun ? JSON.stringify({ ...feed.lastRun, actions: `${feed.lastRun.actions.length} changes` }, null, 2) : "–"}</pre>
    </section>
  );
}
