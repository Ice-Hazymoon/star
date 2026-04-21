FROM oven/bun:debian AS python-deps

USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY= \
    no_proxy=

RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Install CPU-only torch first. Upstream pulls the CUDA build by default on
# linux, which would add ~1.5GB of GPU libraries we don't use. Install
# torch/torchvision from the official CPU wheel index before the rest of the
# requirements so transformers picks it up as already-satisfied.
RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && pip install --proxy "" --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch torchvision

COPY requirements.txt ./
RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && pip install --proxy "" --no-cache-dir -r requirements.txt \
    && find /app/.venv -name '*.pyc' -delete \
    && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf '{}' +

# Bake the sky-mask model into the image so the container doesn't pull it
# from HuggingFace on first request, and export it to int8-quantized ONNX
# (annotate_sky_mask.py loads this via onnxruntime at runtime, which is 2-3×
# faster than the PyTorch path). Fails the build if either step breaks —
# better than silently degrading at runtime.
COPY python/annotate_sky_mask.py /tmp/annotate_sky_mask.py
COPY python/export_sky_mask_onnx.py /tmp/export_sky_mask_onnx.py
RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && python3 /tmp/export_sky_mask_onnx.py /app/hf_cache \
    && python3 -c "import sys; sys.path.insert(0, '/tmp'); import annotate_sky_mask as m; assert m.preload(), 'sky-mask model failed to load during build'" \
    && rm /tmp/annotate_sky_mask.py /tmp/export_sky_mask_onnx.py \
    && find /app/hf_cache -name '*.pyc' -delete 2>/dev/null || true

FROM oven/bun:debian AS data-bootstrap

USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY= \
    no_proxy=

RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY data/catalog ./data/catalog
COPY data/reference ./data/reference
COPY samples ./samples

RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && mkdir -p /app/data/astrometry \
    && for index in 4107 4108 4109 4110 4111 4112 4113 4114 4115 4116 4117 4118 4119; do \
        echo "download index-${index}.fits"; \
        curl -fsSL --retry 3 --retry-delay 2 \
          "http://data.astrometry.net/4100/index-${index}.fits" \
          --output "/app/data/astrometry/index-${index}.fits"; \
      done

FROM oven/bun:debian

USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HTTP_PROXY= \
    HTTPS_PROXY= \
    ALL_PROXY= \
    http_proxy= \
    https_proxy= \
    all_proxy= \
    NO_PROXY= \
    no_proxy= \
    PORT=3000

RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update \
    && apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y --no-install-recommends \
    astrometry.net \
    ca-certificates \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=python-deps /app/.venv /app/.venv
COPY --from=python-deps --chown=bun:bun /app/hf_cache /app/hf_cache
ENV PATH="/app/.venv/bin:${PATH}"

COPY --from=data-bootstrap --chown=bun:bun /app/data /app/data
COPY --from=data-bootstrap --chown=bun:bun /app/samples /app/samples

COPY --chown=bun:bun package.json bun.lock docker-entrypoint.sh ./
COPY --chown=bun:bun public ./public
COPY --chown=bun:bun python ./python
COPY --chown=bun:bun src ./src

RUN chmod +x /app/docker-entrypoint.sh

USER bun

EXPOSE 3000

ENTRYPOINT ["/app/docker-entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["env", "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "http_proxy", "-u", "https_proxy", "-u", "ALL_PROXY", "-u", "all_proxy", "-u", "NO_PROXY", "-u", "no_proxy", "python3", "-c", "import os, sys, urllib.request; port = os.environ.get('PORT', '3000'); response = urllib.request.urlopen(f'http://127.0.0.1:{port}/readyz', timeout=4); sys.exit(0 if 200 <= response.status < 400 else 1)"]

CMD ["bun", "run", "start"]
