# Multi-stage build for nas-engine.
#
# Stage 1 builds a wheel. Stage 2 installs only that wheel plus its runtime dependencies,
# so the final image carries no compiler toolchain, no test suite, and no build cache.
#
# The default is CPU-only: the CPU wheel of PyTorch is roughly 200 MB against several
# gigabytes for the CUDA build, and a NAS smoke run does not need a GPU. GPU instructions
# are at the bottom of this file.
#
#   docker build -t nas-engine:latest .
#   docker run --rm nas-engine:latest smoke
#   docker run --rm -v "$PWD/artifacts:/data/artifacts" nas-engine:latest \
#       search --config /app/configs/smoke_test.yaml --set project.output_dir=/data/artifacts

# --------------------------------------------------------------------------- builder --
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m pip install --upgrade pip build

# Copy only what the build backend needs, so a source change does not invalidate the
# dependency layer above it.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m build --wheel --outdir /build/dist

# --------------------------------------------------------------------------- runtime --
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="nas-engine" \
      org.opencontainers.image.description="Neural Architecture Search from scratch" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/example/neural-architecture-search"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Reduce thread thrash: the container is usually CPU-limited, and oversubscribing
    # BLAS threads makes latency measurements noisier without making anything faster.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # Deterministic cuBLAS workspaces must be set before the first CUDA context exists.
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# `libgomp1` is required by PyTorch's CPU kernels; the slim base image omits it.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# A non-root user. Running as root inside a container is a needless privilege: a container
# escape then starts from uid 0, and any file the container writes to a mounted volume is
# owned by root on the host.
RUN groupadd --gid 1000 nas \
 && useradd --uid 1000 --gid nas --create-home --shell /bin/bash nas

WORKDIR /app

# The CPU wheel index keeps the image small. Remove `--index-url` for the default CUDA build.
COPY --from=builder /build/dist/*.whl /tmp/
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install /tmp/*.whl \
 && rm -rf /tmp/*.whl

# Configuration files and the smoke script travel with the image so the container is
# useful without a mounted source tree.
COPY --chown=nas:nas configs ./configs
COPY --chown=nas:nas scripts/run_smoke_search.sh ./scripts/run_smoke_search.sh
COPY --chown=nas:nas docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh ./scripts/run_smoke_search.sh

# Mount points for the things that must outlive the container. Created before the USER
# switch so their ownership is right.
RUN mkdir -p /data/artifacts /data/db /data/reports /data/datasets \
 && chown -R nas:nas /data /app

USER nas

VOLUME ["/data"]

# A cheap liveness probe: if the package imports and the CLI answers, the image is sound.
# It runs in about a second and needs no database or dataset.
HEALTHCHECK --interval=60s --timeout=20s --start-period=10s --retries=3 \
  CMD ["nas-engine", "doctor", "--set", "project.output_dir=/data/artifacts", "--json"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]

# ---------------------------------------------------------------------------------------
# GPU execution (optional)
# ---------------------------------------------------------------------------------------
# The default image is CPU-only. For CUDA:
#
#   1. Base both stages on `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` and install
#      Python into it, or start from an official PyTorch CUDA image.
#   2. Drop the `--index-url https://download.pytorch.org/whl/cpu` flag so pip installs
#      the CUDA build of torch.
#   3. Install the NVIDIA Container Toolkit on the host and run with `--gpus all`:
#
#        docker run --rm --gpus all -v "$PWD/artifacts:/data/artifacts" \
#          nas-engine:cuda search --config /app/configs/random_search.yaml \
#          --set hardware.device=cuda --set project.output_dir=/data/artifacts
#
#   4. Verify with `docker run --rm --gpus all nas-engine:cuda doctor`, which reports the
#      detected devices and the CUDA version.
#
# GPU support is deliberately not the default: it multiplies the image size by roughly ten
# and every capability in this project works on CPU.
