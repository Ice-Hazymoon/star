FROM oven/bun:debian

USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    PIP_CONFIG_FILE=/dev/null \
    PYTHONUNBUFFERED=1 \
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
    curl \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

COPY requirements.txt ./
RUN export http_proxy= HTTPS_PROXY= HTTP_PROXY= https_proxy= ALL_PROXY= all_proxy= NO_PROXY= no_proxy= \
    && pip install --proxy "" --no-cache-dir -r requirements.txt

COPY --chown=bun:bun . .

RUN chmod +x /app/docker-entrypoint.sh \
    && chown -R bun:bun /app

USER bun

EXPOSE 3000

ENTRYPOINT ["/app/docker-entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/readyz" || exit 1

CMD ["bun", "run", "start"]
