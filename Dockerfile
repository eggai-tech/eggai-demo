# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Build
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# copy=link mode matters: the venv is copied to another stage, so hardlinks
# into the uv cache would dangle.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Dependency manifests only — keeps this layer cached across source changes.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser agents    /app/agents
COPY --chown=appuser:appuser libraries /app/libraries

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

# Generic across agents — the chart supplies the module as args,
# e.g. args: ["agents.triage.main"]
ENTRYPOINT ["/app/.venv/bin/python", "-m"]