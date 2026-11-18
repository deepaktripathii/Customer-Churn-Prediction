# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    CHURN_DATA_DIR=/app/data \
    CHURN_ARTIFACT_DIR=/app/artifacts

WORKDIR /app

# Dependencies first so code changes do not invalidate the pip layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-deps -e . && chmod +x /usr/local/bin/docker-entrypoint.sh

# The entrypoint downloads the dataset and trains an initial model when the
# mounted volume is empty, which keeps `docker run` a one-liner.
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "churn_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000
