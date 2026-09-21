"""Phase 5: teaching by humans — **collection is automatic, training never is.**

How this works, by decision (docs/DECISIONS.md D21):

1. Every validated human run is stored by the API as people play. That happens continuously and needs nobody.
2. Turning that corpus into a new generation of the fly is a **manual** step, run by hand on the GPU machine:

       uv run --no-sync flybrain crowd-fetch --api https://api-production-dad9.up.railway.app   # download the corpus
       uv run --no-sync python -u -m experiments.crowd_teaching --train                         # opt in explicitly

   Without `--train` this script only reports what has been collected. With it, it trains two conditions on the same
   compute — the fly alone vs. the fly plus the human corpus (observational replay + death curriculum,
   `flybrain/crowd.py`) — and writes the comparison to `results/crowd_teaching.json`.

No synthetic "humans" are ever substituted: with too few runs the script says so and stops.

Known problem to settle before the first real training run (DECISIONS.md D20): the replay's punishment is a PPL1
burst *inside* a running game, and such bursts can ignite self-sustained Kenyon-cell volleys in this model
(`experiments/kc_volley.py`). `--train` therefore refuses unless `--punishment {tail,none}` is chosen explicitly.
"""

from __future__ import annotations

import argparse
import json

from experiments.common import write_result
from flybrain.config import cache_dir

NAME = "crowd_teaching"
MIN_RUNS = 50
MIN_SEEDS = 10


def corpus_path():
    return cache_dir() / "human_runs.json"


def load_corpus() -> list[dict]:
    path = corpus_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("runs", [])


def describe(runs: list[dict]) -> dict:
    seeds = {r["seed"] for r in runs}
    scores = sorted(r["score"] for r in runs)
    return {
        "runs": len(runs),
        "distinct_seeds": len(seeds),
        "score_median": scores[len(scores) // 2] if scores else None,
        "score_max": scores[-1] if scores else None,
        "ready_to_train": len(runs) >= MIN_RUNS and len(seeds) >= MIN_SEEDS,
        "minimum": {"runs": MIN_RUNS, "distinct_seeds": MIN_SEEDS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="actually train a new generation from the corpus (manual step)")
    ap.add_argument("--punishment", choices=("tail", "none"), help="where the observational-replay punishment goes (see D20)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    runs = load_corpus()
    status = describe(runs)
    print(f"[crowd] corpus: {status['runs']} validated human runs on {status['distinct_seeds']} seeds "
          f"(need {MIN_RUNS} / {MIN_SEEDS}) → {'ready' if status['ready_to_train'] else 'keep collecting'}", flush=True)
    write_result(NAME, {"collection": status, "trained": False, "note": "collection is continuous; training is a manual step (--train)"})

    if not args.train:
        print("[crowd] collection only. Re-run with --train (and --punishment) when you want a new generation.", flush=True)
        return 0
    if not status["ready_to_train"]:
        raise SystemExit("[crowd] not enough human runs yet — nothing is trained on a corpus this small")
    if args.punishment is None:
        raise SystemExit("[crowd] choose --punishment tail|none first: a PPL1 burst inside a running game can ignite "
                         "Kenyon-cell volleys in this model (experiments/kc_volley.py, DECISIONS.md D20)")
    raise SystemExit("[crowd] training path is wired in flybrain/crowd.py but has never been run on real data; "
                     "run it deliberately with a fresh prereg tag (prereg-phase5) rather than from this stub")


def render_report(r: dict) -> str:
    c = r.get("collection")
    if not c:
        return "not yet measured"
    lines = [
        f"**Collected so far: {c['runs']} validated human runs on {c['distinct_seeds']} distinct seeds.** "
        f"Collection runs by itself whenever anyone plays; a new generation of the fly is only ever trained by hand "
        f"(`experiments/crowd_teaching.py --train`, DECISIONS.md D21). Threshold before training makes sense: "
        f"{c['minimum']['runs']} runs on {c['minimum']['distinct_seeds']} seeds → "
        + ("**ready**." if c["ready_to_train"] else "**keep collecting**."),
    ]
    if c["score_median"] is not None:
        lines.append(f"Human scores in the corpus: median {c['score_median']}, best {c['score_max']}.")
    if not r.get("trained"):
        lines += ["", "No generation has been trained from human data yet, so the crowd-teaching ablation is **not yet measured**."]
    lines += ["", "Source: `results/crowd_teaching.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
