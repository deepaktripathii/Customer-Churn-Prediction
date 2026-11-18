#!/bin/sh
# Bootstrap the container: make sure the dataset and at least one model exist.
set -eu

: "${CHURN_DATA_DIR:=/app/data}"
: "${CHURN_ARTIFACT_DIR:=/app/artifacts}"
export CHURN_DATA_DIR CHURN_ARTIFACT_DIR

mkdir -p "$CHURN_DATA_DIR/raw" "$CHURN_ARTIFACT_DIR/models"

if [ ! -f "$CHURN_DATA_DIR/raw/telco_churn.csv" ]; then
    echo "[entrypoint] dataset missing, downloading Telco churn CSV"
    python scripts/download_data.py
fi

if [ -z "$(ls -A "$CHURN_ARTIFACT_DIR/models" 2>/dev/null || true)" ]; then
    echo "[entrypoint] no trained model found, training initial models"
    python scripts/train.py
fi

exec "$@"
