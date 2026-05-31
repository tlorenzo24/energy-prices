# Base image for both the ingestion/forecast worker and the Streamlit dashboard.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps for lightgbm/statsmodels wheels are usually prebuilt; keep slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with Postgres support (Timescale uses the psycopg2 driver).
RUN pip install --upgrade pip && pip install ".[postgres]"

EXPOSE 8501

# Default command runs the dashboard; the ingest service overrides it in compose.
CMD ["streamlit", "run", "src/energy_prices/dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
