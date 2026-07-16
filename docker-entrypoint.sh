#!/bin/sh
# Bootstrap the container: make sure the dataset and at least one model exist.
set -eu

: "${RETAINIQ_DATA_DIR:=/app/data}"
: "${RETAINIQ_ARTIFACT_DIR:=/app/artifacts}"
export RETAINIQ_DATA_DIR RETAINIQ_ARTIFACT_DIR

mkdir -p "$RETAINIQ_DATA_DIR/raw" "$RETAINIQ_ARTIFACT_DIR/models"

if [ ! -f "$RETAINIQ_DATA_DIR/raw/telco_churn.csv" ]; then
    echo "[entrypoint] dataset missing, downloading Telco churn CSV"
    python scripts/download_data.py
fi

if [ -z "$(ls -A "$RETAINIQ_ARTIFACT_DIR/models" 2>/dev/null || true)" ]; then
    echo "[entrypoint] no trained model found, training initial models"
    python scripts/train.py
fi

exec "$@"
