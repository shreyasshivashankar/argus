FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs

# ---------------------------------------------------------------------------
# Test stage — includes test deps, runs pytest
# ---------------------------------------------------------------------------
FROM base AS test

RUN pip install --no-cache-dir \
    pytest>=8.0.0 \
    pytest-asyncio>=0.23.0 \
    aioresponses>=0.7.0 \
    freezegun>=1.0.0

CMD ["pytest", "tests/", "-v"]

# ---------------------------------------------------------------------------
# Production stage — slim, no test deps
# ---------------------------------------------------------------------------
FROM base AS production

ENTRYPOINT ["python", "main.py"]
