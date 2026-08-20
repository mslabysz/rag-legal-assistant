FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY README.md ./
COPY src ./src
COPY data ./data

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "rag_legal_assistant.api.server:app", "--host", "0.0.0.0", "--port", "8000"]