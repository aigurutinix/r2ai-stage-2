# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KINGPRO_HOST=0.0.0.0 \
    KINGPRO_PORT=8080 \
    HOME=/tmp

WORKDIR /app

RUN groupadd --gid 10001 kingpro \
    && useradd --uid 10001 --gid kingpro --no-create-home --shell /usr/sbin/nologin kingpro

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY src ./src
COPY scripts/serve_product.py scripts/_env.py ./scripts/
COPY docker/backend_entrypoint.py ./docker/backend_entrypoint.py

RUN mkdir -p /app/outputs /tmp/kingpro \
    && chown -R kingpro:kingpro /app/outputs /tmp/kingpro

USER kingpro

EXPOSE 8080

ENTRYPOINT ["python", "docker/backend_entrypoint.py"]
CMD ["python", "scripts/serve_product.py", "--host", "0.0.0.0", "--port", "8080"]

