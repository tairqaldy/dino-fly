"""`flybrain` command line interface."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="dino-fly brain tools")


@app.command("download-data")
def download_data(
    only: list[str] = typer.Option(None, "--only", help="Manifest keys to fetch (default: all)."),  # noqa: B008
    record: bool = typer.Option(
        False,
        "--record",
        help="Record sha256 of files that have none in the manifest yet (maintainers only).",
    ),
) -> None:
    """Download the pinned connectome / annotation files and verify their checksums."""
    from flybrain.data_manifest import ensure_all

    paths = ensure_all(list(only) if only else None, record=record, log=typer.echo)
    typer.echo(f"{len(paths)} file(s) present and verified.")


@app.command("info")
def info() -> None:
    """Print environment and data status."""
    from flybrain.config import data_dir, run_metadata
    from flybrain.data_manifest import load_manifest

    for key, value in run_metadata().items():
        typer.echo(f"{key:>14}: {value}")
    typer.echo(f"{'data_dir':>14}: {data_dir()}")
    for key, entry in load_manifest().items():
        state = "present" if entry.local_path().exists() else "missing"
        typer.echo(f"{key:>28}: {state}")


@app.command("worker")
def worker(
    host: str = typer.Option("127.0.0.1", help="Bind address of the local WebSocket feed."),
    port: int = typer.Option(8765),
    api: str = typer.Option(None, help="Public hub to push to, e.g. wss://api.example.com/worker (outbound only)."),
    token: str = typer.Option(None, envvar="WORKER_TOKEN", help="Shared secret for the hub."),
    device: str = typer.Option("cuda"),
    connectome: str = typer.Option("flywire783"),
) -> None:
    """Let the fly play live and stream it (local dashboard feed; optionally the public hub)."""
    from flybrain.worker import main as worker_main

    worker_main(host=host, port=port, api=api, token=token, device=device, connectome=connectome)


@app.command("crowd-fetch")
def crowd_fetch(
    api: str = typer.Option("https://api-production-dad9.up.railway.app", help="Public API base URL."),
    limit: int = typer.Option(5000, help="How many of the newest validated human runs to fetch."),
) -> None:
    """Download the human teaching corpus (metadata + action logs) into the local cache.

    Collection happens by itself while people play; this only copies it to the GPU machine. Training a new
    generation from it is a separate, explicitly manual step (`python -m experiments.crowd_teaching --train`).
    """
    import json
    import urllib.request

    from flybrain.config import cache_dir

    url = f"{api.rstrip('/')}/api/runs/human?limit={int(limit)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    runs = payload.get("runs", [])
    out = cache_dir() / "human_runs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": url, "runs": runs}, indent=1) + "\n", encoding="utf-8")
    seeds = len({r["seed"] for r in runs})
    print(f"[crowd] {len(runs)} validated human runs on {seeds} seeds → {out}")
    print("[crowd] nothing is trained automatically: run `python -m experiments.crowd_teaching` to see the corpus,")
    print("        and `--train --punishment tail|none` when you decide to roll out a new generation.")


@app.command("neurons-doc")
def neurons_doc(
    check: bool = typer.Option(False, "--check", help="Fail if docs/NEURONS.md is out of date."),
    refresh_lock: bool = typer.Option(
        False, "--refresh-lock", help="Re-resolve all neuron sets against the annotation table (needs data)."
    ),
) -> None:
    """Generate docs/NEURONS.md from flybrain/neurons.py."""
    from flybrain.neurons_doc import main as neurons_doc_main

    raise typer.Exit(neurons_doc_main(check=check, refresh_lock=refresh_lock))


if __name__ == "__main__":  # pragma: no cover
    app()
