interface Props {
  values: number[];
  width?: number;
  height?: number;
  accent?: boolean;
  min?: number;
  label?: string;
}

/** Tiny dependency-free line chart (SVG). */
export function Sparkline({ values, width = 240, height = 48, accent = false, min, label }: Props) {
  if (values.length < 2) {
    return <div className="sparkline-empty">{label ?? "no data yet"}</div>;
  }
  const lo = min ?? Math.min(...values);
  const hi = Math.max(...values, lo + 1e-9);
  const pts = values
    .map((v, i) => `${((i / (values.length - 1)) * width).toFixed(1)},${(height - 2 - ((v - lo) / (hi - lo)) * (height - 4)).toFixed(1)}`)
    .join(" ");
  return (
    <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={label}>
      <polyline points={pts} fill="none" stroke={accent ? "var(--accent)" : "currentColor"} strokeWidth="1.5" />
    </svg>
  );
}
