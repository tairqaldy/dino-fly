import type { FeedSnapshot } from "../lib/feed.js";

/**
 * Brain map v1: a schematic of the pathway the game actually uses, heated by live spike counts per cell group.
 * Left/right optic lobes (LC4, LPLC2) → Giant Fiber → JUMP, with the mushroom body (KC, MBON, PAM, PPL1) and the pool
 * of all other descending neurons shown for context. Positions are hand-placed; the numbers are real.
 */
const NODES: { group: string; label: string; x: number; y: number; r: number }[] = [
  { group: "LPLC2", label: "LPLC2 · size", x: 90, y: 70, r: 26 },
  { group: "LC4", label: "LC4 · velocity", x: 90, y: 170, r: 22 },
  { group: "KC", label: "Kenyon cells", x: 300, y: 50, r: 22 },
  { group: "MBON", label: "MBON", x: 390, y: 50, r: 14 },
  { group: "PAM", label: "PAM (reward)", x: 300, y: 118, r: 12 },
  { group: "PPL1", label: "PPL1 (punish)", x: 390, y: 118, r: 12 },
  { group: "GF", label: "Giant Fiber", x: 300, y: 200, r: 20 },
  { group: "DN", label: "other descending", x: 430, y: 200, r: 16 },
  { group: "other", label: "rest of the brain", x: 520, y: 100, r: 30 },
];
const EDGES: [string, string][] = [
  ["LPLC2", "GF"],
  ["LC4", "GF"],
  ["LPLC2", "other"],
  ["KC", "MBON"],
  ["PAM", "KC"],
  ["PPL1", "KC"],
  ["other", "GF"],
];

export function BrainMap({ feed }: { feed: FeedSnapshot }) {
  const groups = feed.hello?.groups ?? [];
  const window = feed.history.slice(-30); // ≈ 0.5 s
  const rate = (g: string): number => {
    const i = groups.indexOf(g);
    if (i < 0 || window.length === 0) return 0;
    return window.reduce((acc, h) => acc + (h.activity[i] ?? 0), 0) / window.length;
  };
  const pos = Object.fromEntries(NODES.map((n) => [n.group, n]));
  const gfNow = (feed.frame?.gf ?? 0) > 0;
  return (
    <svg className="brainmap" viewBox="0 0 600 260" role="img" aria-label="Live activity of the cell groups used by the game">
      {EDGES.map(([a, b]) => {
        const p = pos[a];
        const q = pos[b];
        if (!p || !q) return null;
        const hot = rate(a) > 0 && (b !== "GF" || rate("GF") > 0 || a === "LPLC2" || a === "LC4");
        return <line key={`${a}-${b}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} className={hot ? "edge hot" : "edge"} />;
      })}
      {NODES.map((n) => {
        const v = rate(n.group);
        const heat = Math.min(1, Math.log1p(v) / Math.log1p(n.group === "other" ? 60 : 6));
        return (
          <g key={n.group}>
            <circle cx={n.x} cy={n.y} r={n.r} className="node" style={{ fillOpacity: 0.08 + 0.92 * heat }} />
            <text x={n.x} y={n.y + n.r + 12} textAnchor="middle" className="node-label">
              {n.label}
            </text>
            <text x={n.x} y={n.y + 4} textAnchor="middle" className="node-value">
              {v.toFixed(v < 10 ? 1 : 0)}
            </text>
          </g>
        );
      })}
      <g transform="translate(300 238)">
        <rect x={-44} y={-12} width={88} height={20} className={gfNow || feed.frame?.jump ? "jump on" : "jump"} />
        <text x={0} y={3} textAnchor="middle" className="jump-label">
          JUMP
        </text>
      </g>
      <line x1={300} y1={220} x2={300} y2={226} className={gfNow ? "edge hot" : "edge"} />
      <text x={8} y={252} className="node-label">
        numbers = spikes per game frame (10 ms of fly time), averaged over the last ½ s
      </text>
    </svg>
  );
}
