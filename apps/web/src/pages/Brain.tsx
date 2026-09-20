import { BrainMap } from "../components/BrainMap.js";
import { GameCanvas } from "../components/GameCanvas.js";
import { Sparkline } from "../components/Sparkline.js";
import type { FeedSnapshot } from "../lib/feed.js";

export function Brain({ feed }: { feed: FeedSnapshot }) {
  const gf = feed.history.map((h) => h.gf);
  const lplc2 = feed.history.map((h) => h.rates[1]);
  const lc4 = feed.history.map((h) => h.rates[0]);
  const total = feed.history.map((h) => h.activity.reduce((a, b) => a + b, 0));
  const curve = feed.stats?.learningCurve.map((p) => p.heldoutMean) ?? [];
  return (
    <section>
      <h1>
        Inside the fly <span className={feed.online ? "dot on" : "dot"} />
      </h1>
      {!feed.online && <p className="lede">The brain is offline right now — this page shows live activity when the laptop worker is running.</p>}
      {feed.hello && (
        <p className="hint">
          {feed.hello.connectome} · {feed.hello.neurons.toLocaleString()} neurons · {feed.hello.gpu} · engine v{feed.hello.engineVersion} · transducer v
          {feed.hello.transducerVersion} · {feed.hello.bioMsPerFrame} ms of fly time per game frame
        </p>
      )}
      <div className="grid2">
        <div>
          <BrainMap feed={feed} />
        </div>
        <div>
          {feed.state && <GameCanvas state={feed.state} options={{ accentDino: true, label: `GEN ${feed.stats?.generation ?? 0}` }} />}
          <dl className="stats">
            <dt>generation</dt>
            <dd>{feed.stats?.generation ?? "–"}</dd>
            <dt>games played</dt>
            <dd>{feed.stats?.gamesPlayed ?? "–"}</dd>
            <dt>best score</dt>
            <dd>{feed.stats?.bestScore ?? "–"}</dd>
            <dt>mean of last 50</dt>
            <dd>{feed.stats ? feed.stats.meanScoreRecent.toFixed(1) : "–"}</dd>
            <dt>brain speed</dt>
            <dd>{feed.stats?.realtimeFactor ? `${feed.stats.realtimeFactor.toFixed(2)}× fly real time` : "–"}</dd>
          </dl>
        </div>
      </div>
      <h2>Last six seconds</h2>
      <div className="traces">
        <figure>
          <figcaption>looming drive to LPLC2 (Hz, commanded by the fixed transducer)</figcaption>
          <Sparkline values={lplc2} width={560} min={0} />
        </figure>
        <figure>
          <figcaption>looming drive to LC4 (Hz)</figcaption>
          <Sparkline values={lc4} width={560} min={0} />
        </figure>
        <figure>
          <figcaption>Giant Fiber spikes per frame — every spike on the ground is a jump</figcaption>
          <Sparkline values={gf} width={560} min={0} accent />
        </figure>
        <figure>
          <figcaption>whole-brain spikes per frame</figcaption>
          <Sparkline values={total} width={560} min={0} />
        </figure>
      </div>
      <h2>Learning curve</h2>
      <Sparkline values={curve} width={560} accent label="no generations evaluated yet — the naive fly is generation 0" />
    </section>
  );
}
