# The brain as a hosted service: the same LIF engine, on CPU.
#
# Measured on 8 threads with the active-set engine: ~31 game frames per second on the full FlyWire 783 connectome,
# i.e. the fly plays at roughly half of real-time speed. That is enough to stream a live game to the site, so the
# public page no longer depends on Tair's laptop being on. The GPU worker (`flybrain worker --device cuda`) is still
# the one used for experiments; this container only plays.
#
# Build context = the `brain/` directory:  cd brain && railway up --service brain
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    DINOFLY_DATA_DIR=/data \
    OMP_NUM_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /usr/local/bin/uv

WORKDIR /app
# dependencies first, so a code change does not re-download torch
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --extra cpu --locked --no-install-project
COPY flybrain flybrain
COPY experiments experiments
RUN uv sync --extra cpu --locked

ENV PATH="/opt/venv/bin:$PATH"
# the connectome (~104 MB) is fetched on first boot and verified against the pinned SHA-256s; mount a volume
# at /data to keep it across deploys
CMD ["sh", "-c", "flybrain download-data --only flywire783_completeness --only flywire783_connectivity && exec flybrain worker --host 0.0.0.0 --device cpu --api \"$HUB_URL\" --token \"$WORKER_TOKEN\""]
