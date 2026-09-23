# =============================================================================
# Inference-audit demo — self-contained reproducibility image
#
# Reproduces the challenge-response inference audit on SYNTHETIC data, with no
# proprietary dataset and no network. CPU-only, depends on nothing but numpy
# and torch (the detector primitives are inlined in the demo).
#
# Build context: this directory.
#
# Build:
#   docker build -t inference-audit-demo .
# Run the demo (no network, no data):
#   docker run --rm --network none inference-audit-demo
# Run the smoke test:
#   docker run --rm --network none inference-audit-demo pytest -q
# =============================================================================

FROM ghcr.io/astral-sh/uv:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r audit && useradd -r -g audit -d /home/audit -m audit
WORKDIR /app
ENV UV_PYTHON_INSTALL_DIR=/app/.python

# The uv base image ships no system Python, so install a managed interpreter
# explicitly (into UV_PYTHON_INSTALL_DIR, under /app) and build the venv on it,
# so /app/.venv/bin/python resolves at runtime.
# CPU-only torch from the PyTorch CPU index (small, multi-arch), the rest from PyPI.
RUN uv python install 3.12 \
 && uv venv --python 3.12 /app/.venv \
 && uv pip install --python /app/.venv/bin/python \
      --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && uv pip install --python /app/.venv/bin/python "numpy>=1.26" "pytest>=8.0" \
 && uv cache clean

COPY . /app
RUN chmod +x /app/docker-entrypoint.sh \
 && chown -R audit:audit /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/tmp

USER audit

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "inference_audit_demo.py"]
