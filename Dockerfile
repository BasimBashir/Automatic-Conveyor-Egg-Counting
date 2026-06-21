# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/app/.torch \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HOME=/app

# Ubuntu 24.04 ships Python 3.12 — no PPA needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
# Cache wheels across builds so a requirements change doesn't re-download the
# multi-GB CUDA/torch/TensorRT stack. --break-system-packages: the base image's
# Python is PEP-668 externally-managed.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt

COPY app/ app/
COPY best.pt .

# engine_cache holds the TensorRT engine built on first boot (see
# docker-entrypoint.sh). It is also a docker-compose named-volume mount point so
# the engine survives container recreation.
RUN mkdir -p app/uploads app/outputs app/data .torch engine_cache

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd  --uid 1001 --gid 1001 --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5590

# First boot on a new GPU builds the TRT engine (2-6 min on a 3090, longer on
# smaller cards), so the health start-period must tolerate the cold-build case.
# Cached boots are fast. Disable the build with TRT_AUTO_BUILD=0.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -f http://localhost:5590/health || exit 1

CMD ["/docker-entrypoint.sh"]
