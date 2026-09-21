"""Render the result blocks of docs/RESEARCH.md from the committed results JSONs.

Every number in RESEARCH.md lives between `<!-- BEGIN:<name> -->` / `<!-- END:<name> -->` markers and is produced
here from `brain/experiments/results/<name>.json` (volatile "meta" fields are never rendered). CI runs `--check`.

    uv run --no-sync python -m experiments.report           # rewrite the blocks
    uv run --no-sync python -m experiments.report --check   # fail if RESEARCH.md is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable

from flybrain.config import repo_root, results_dir

NOT_MEASURED = "not yet measured"


def _load(name: str) -> dict | None:
    path = results_dir() / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------------------------------------ sugar → MN9
def render_sugar_mn9() -> str:
    r = _load("sugar_mn9")
    if r is None:
        return NOT_MEASURED
    lines: list[str] = []
    cross = _load("brian2_crosscheck_flywire630")
    native = _load("brian2_native_rerun_flywire630")
    c = r["criterion"]
    v630, v783 = r["v630"], r["v783"]
    rates = v630["grn_rate_hz"]

    lines.append("**Result 1 — spike-exact equivalence with Brian2.**")
    if cross:
        t = cross["trials"]
        spikes = ", ".join(f"{x['brian2_spikes']:,}" for x in t)
        verdict = "identical spike trains, 0 mismatches" if cross["all_identical"] else "MISMATCH — see JSON"
        lines.append(
            f"On the real v630 connectome (127,400 neurons, 14.7 M connections), real Brian2 and our engine were fed the "
            f"same kick schedule ({_f(cross['grn_rate_hz'], 0)} Hz on the 21 sugar GRNs, {cross['n_steps']:,} steps, float64): "
            f"**{verdict}** in {len(t)} trials ({spikes} spikes). Brian2 (numpy target) needed "
            f"{_f(sum(x['brian2_wall_s'] for x in t) / len(t))} s per trial, our engine {_f(sum(x['engine_wall_s'] for x in t) / len(t))} s. "
            "The same check on a synthetic 1k-neuron network with autapses, duplicate edges and strong inhibition is a "
            "unit test in CI (`tests/test_lif_brian2_golden.py`). "
            "Source: `results/brian2_crosscheck_flywire630.json`."
        )
    else:
        lines.append(NOT_MEASURED)

    lines.append("")
    lines.append("**Result 2 — the published sugar → MN9 curve (connectome v630, 30 trials × 1 s).**")
    lines.append("")
    lines.append("| GRN rate (Hz) | MN9 published | MN9 ours | diff | MN9 partner published | MN9 partner ours |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    ref = json.loads((repo_root() / "brain/flybrain/reference/shiu2024_fig1d.json").read_text(encoding="utf-8"))
    for i, rate in enumerate(rates):
        if rate % 20 and rate != 50 and rate != 150:
            continue
        pub, par = ref["neurons"]["mn9_published"]["rate_hz"][i], ref["neurons"]["mn9_partner"]["rate_hz"][i]
        lines.append(
            f"| {rate} | {_f(pub)} | {_f(v630['mn9_published']['rate_hz'][i])} | "
            f"{v630['mn9_published']['rate_hz'][i] - pub:+.1f} | {_f(par)} | {_f(v630['mn9_partner']['rate_hz'][i])} |"
        )
    n_in = sum(c["neurons"]["mn9_published"]["within"])
    lines.append("")
    lines.append(
        f"RMSE over the 20-point curve: {_f(c['neurons']['mn9_published']['rmse_hz'], 2)} Hz (MN9), "
        f"{_f(c['neurons']['mn9_partner']['rmse_hz'], 2)} Hz (partner); largest deviation "
        f"{_f(c['neurons']['mn9_published']['max_abs_diff_hz'], 2)} Hz; {n_in}/20 points within the pre-registered band."
    )
    verdict = "PASS" if c["pass_100hz"] else "FAIL"
    lines.append(
        f"Pre-registered criterion at 100 Hz (|ours − 65.7| ≤ {_f(c['tolerance_hz'], 0)} Hz): ours "
        f"{_f(c['at_100hz']['ours_hz'], 2)} Hz, diff {c['at_100hz']['diff_hz']:+.2f} Hz → **{verdict}**."
    )
    pr = v630.get("precise_100hz")
    ex = r["v630_vs_published_spikes"]["100"]["published_mn9_hz_this_file"]
    ex_mn9 = ex[v630["mn9_ids"][0]]
    ctx = [
        f"the paper's Fig. 1d run gives 65.7 Hz and the authors' own second published 100 Hz run "
        f"(`results/example/sugarR_100Hz.parquet`) gives {_f(ex_mn9, 2)} Hz"
    ]
    if native:
        ctx.append(
            f"re-running the published stochastic protocol in real Brian2 on this machine ({native['n_trials']} trials) gives "
            f"{_f(native['mn9_published_mean_hz'], 2)} ± {_f(native['mn9_published_se_hz'], 2)} Hz (mean ± SE)"
        )
    if pr:
        ctx.append(
            f"our engine with {pr['n_trials']} trials gives {_f(pr['mn9_published_mean_hz'], 2)} ± "
            f"{_f(pr['mn9_published_se_hz'], 2)} Hz"
        )
    lines.append("Context for that number: " + "; ".join(ctx) + ". Given Result 1, remaining differences are sampling noise of 30-trial estimates, not model differences.")

    lines.append("")
    lines.append("**Result 3 — every neuron, not just MN9 (v630 vs. the published example spike files).**")
    lines.append("")
    lines.append("| GRN rate | active neurons (published / ours) | Jaccard | Pearson r of rates | spikes per trial (published / ours) |")
    lines.append("|---:|---:|---:|---:|---:|")
    for rate in ("100", "200"):
        s = r["v630_vs_published_spikes"][rate]
        lines.append(
            f"| {rate} Hz | {s['published_active']} / {s['ours_active']} | {_f(s['jaccard_active'], 3)} | "
            f"{_f(s['pearson_r'], 4)} | {s['published_total_spikes_per_trial']:,.0f} / {s['ours_total_spikes_per_trial']:,.0f} |"
        )

    lines.append("")
    lines.append("**Result 4 — connectome v783, the one dino-fly uses (no published reference exists).**")
    i100, i200 = rates.index(100), rates.index(200)
    lines.append(
        f"MN9 at 100 Hz sugar drive: **{_f(v783['mn9_published']['rate_hz'][i100])} Hz** "
        f"(partner {_f(v783['mn9_partner']['rate_hz'][i100])} Hz); at 200 Hz: {_f(v783['mn9_published']['rate_hz'][i200])} Hz. "
        f"Network-wide {v783['total_spikes_per_trial'][i200]:,.0f} spikes per simulated second at 200 Hz across "
        f"{v783['active_neurons'][i200]} active neurons of {v783['connectome_stats']['neurons']:,} "
        f"(a third-party Brian2-CPU benchmark of the same condition reports ≈ "
        f"{r['v783_reference']['flybrain_brian2_cpu_spikes_per_s_at_200hz']:,}). "
    )
    lines.append("")
    lines.append("![sugar → MN9](figures/sugar_mn9.png)")
    lines.append("")
    lines.append("Sources: `results/sugar_mn9.json`, `flybrain/reference/shiu2024_fig1d.json` (script-extracted from doi:10.17617/3.CZODIW).")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ benchmark
def render_benchmark() -> str:
    r = _load("benchmark")
    if r is None:
        return NOT_MEASURED
    lines = [
        f"GPU: {r['gpu']}; torch {r['torch']}; connectome {r['connectome']} "
        f"({r['neurons']:,} neurons, {r['edges']:,} connections); dt = 0.1 ms; float32.",
        "",
        "| engine path | brains (B) | drive | ms / step | biological s per wall s (per brain) | … (all brains) | peak VRAM (GB) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['path']} | {row['batch']} | {row['drive']} | {_f(row['ms_per_step'], 3)} | "
            f"{_f(row['realtime_per_brain'], 3)} | {_f(row['realtime_aggregate'], 2)} | {_f(row['peak_vram_gb'], 2)} |"
        )
    lines.append("")
    lines.append("Source: `results/benchmark.json`.")
    return "\n".join(lines)


def _later(name: str, module_name: str | None = None, fn_name: str = "render_report") -> Callable[[], str]:
    def render() -> str:
        try:
            module = __import__(f"experiments.{module_name or name}", fromlist=[fn_name])
        except ImportError:
            return NOT_MEASURED
        fn = getattr(module, fn_name, None)
        result = _load(name)
        return fn(result) if fn and result else NOT_MEASURED

    return render


RENDERERS: dict[str, Callable[[], str]] = {
    "sugar_mn9": render_sugar_mn9,
    "benchmark": render_benchmark,
    "looming_gf": _later("looming_gf"),
    "naive_play": _later("naive_play"),
    "mbon_influence": _later("mbon_influence"),
    "mb_drive": _later("mb_drive"),
    "mb_drive_kc": _later("mb_drive_kc", "mb_drive"),
    "mb_wiring": _later("mb_wiring"),
    "kc_loops": _later("mb_wiring", fn_name="render_loops"),
    "da_burst": _later("da_burst"),
    "kc_volley": _later("kc_volley"),
    "learning": _later("learning"),
    "eval_invariance": _later("eval_invariance"),
    "gain_sensitivity": _later("gain_sensitivity"),
    "learning_h2": _later("learning_h2", "learning"),
    "crowd_teaching": _later("crowd_teaching"),
}


def render_document(text: str) -> str:
    for name, renderer in RENDERERS.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n)(.*?)(\n<!-- END:{name} -->)", re.DOTALL)
        if pattern.search(text):
            body = renderer()
            text = pattern.sub(lambda m, body=body: m.group(1) + body + m.group(3), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = repo_root() / "docs" / "RESEARCH.md"
    old = path.read_text(encoding="utf-8")
    new = render_document(old)
    if args.check:
        if old != new:
            print("docs/RESEARCH.md is stale — run `python -m experiments.report`", file=sys.stderr)
            return 1
        print("docs/RESEARCH.md is in sync with the results JSONs")
        return 0
    path.write_text(new, encoding="utf-8")
    print(f"rendered {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
