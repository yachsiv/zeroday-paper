# zeroday-paper image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/New_York

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 1000 zp \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash zp

WORKDIR /app

COPY pyproject.toml README.md ./
COPY zeroday_paper ./zeroday_paper
COPY config ./config
COPY entrypoint.sh /app/entrypoint.sh

RUN pip install --upgrade pip \
 && pip install . \
 && chmod +x /app/entrypoint.sh \
 && mkdir -p /data /app/reports /app/logs \
 && chown -R zp:zp /data /app

USER zp

ENV ZP_DUCKDB_PATH=/data/paper.duckdb \
    ZP_REPORT_DIR=/data/reports

ENTRYPOINT ["/app/entrypoint.sh"]
