#!/usr/bin/env bash
# Download the pinned FlyWire connectome / annotation files and verify their SHA-256 checksums.
# Sources, sizes and checksums live in brain/flybrain/data_manifest.json (single source of truth).
#   scripts/download-data.sh                 # everything
#   scripts/download-data.sh --only flywire783_connectivity
set -euo pipefail
cd "$(dirname "$0")/../brain"
exec uv run --no-sync flybrain download-data "$@"
