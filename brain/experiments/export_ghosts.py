"""Bundle recorded fly runs into the static web app, so the site has ghosts to race when the laptop brain is offline.

Source: the cached per-game records of the naive fly on DEV seeds (`experiments.naive_play --dev`). Every exported run
is re-validated by replaying its action log in the game engine; a run whose replay does not reproduce the recorded
score is dropped. Held-out seeds are never exported.

    uv run --no-sync python -m experiments.export_ghosts
"""

from __future__ import annotations

import json

from flybrain import dino_core as dc
from flybrain import seeds as seedsets
from flybrain.config import cache_dir, repo_root

OUT = repo_root() / "apps" / "web" / "public" / "ghosts.json"


def main() -> int:
    path = cache_dir() / "naive_play_dev" / "intact.jsonl"
    if not path.exists():
        raise SystemExit("no cached DEV runs — run `python -m experiments.naive_play --dev --only intact` first")
    ghosts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["seed"] in seedsets.HELDOUT_200:
            continue
        actions = [tuple(a) for a in r["actions"]]
        final = dc.replay(r["seed"], actions, r["frames"])
        if dc.score(final) != r["score"] or final.frame != r["frames"]:
            print(f"[ghosts] seed {r['seed']}: replay mismatch — dropped")
            continue
        ghosts.append({"seed": r["seed"], "generation": 0, "score": r["score"], "actions": [list(a) for a in actions]})
    ghosts.sort(key=lambda g: g["seed"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"engineVersion": dc.ENGINE_VERSION, "ghosts": ghosts}) + "\n", encoding="utf-8", newline="\n")
    print(f"[ghosts] wrote {len(ghosts)} replay-verified fly runs to {OUT} (scores {min(g['score'] for g in ghosts)}–{max(g['score'] for g in ghosts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
