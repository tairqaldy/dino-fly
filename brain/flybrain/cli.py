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
