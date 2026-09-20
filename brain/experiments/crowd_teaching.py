"""Phase 5: does teaching by humans help? (observational replay + death curriculum vs. solo training, equal compute)

Needs validated human runs exported from the API:

    curl "$API/api/runs/human?limit=5000" > brain/experiments/cache/human_runs.json
    uv run --no-sync python -u -m experiments.crowd_teaching

Until such a file exists this experiment has nothing to measure and says so — no synthetic "humans" are substituted.
"""

from __future__ import annotations

import json

from experiments.common import write_result
from flybrain.config import cache_dir

NAME = "crowd_teaching"
MIN_RUNS = 50


def main() -> int:
    path = cache_dir() / "human_runs.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.exists() else []
    if len(runs) < MIN_RUNS:
        print(f"[crowd] {len(runs)} human runs available (< {MIN_RUNS}) — not yet measured")
        return 0
    raise SystemExit("enough human runs collected: wire up the ablation (see flybrain/crowd.py) — intentionally not run blind")


def render_report(_r: dict) -> str:
    return "not yet measured"


if __name__ == "__main__":
    write_result  # noqa: B018 - kept for the future result file
    raise SystemExit(main())
