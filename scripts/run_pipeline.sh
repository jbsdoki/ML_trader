#!/usr/bin/env bash
# End-to-end pipeline: ingest -> sentiment -> daily features CSV -> train XGBoost -> predict.
# From the repo root:  bash scripts/run_pipeline.sh
#
# Optional environment (defaults align with runbook.md / config.yaml):
#   PYTHON              Python executable (default: python)
#   INGEST_CONFIG       YAML path for ingest (default: config.yaml). If this file is missing, SYMBOLS/START/END are used.
#   SYMBOLS             Comma-separated tickers (default: AAPL,MSFT)
#   START, END          Ingest dates YYYY-MM-DD when not using INGEST_CONFIG (default: 2025-01-01, 2025-03-18)
#   INTERVAL            Bar interval for CLI ingest (default: 1d)
#   SENTIMENT_LIMIT     Max articles per sentiment pass (default: 500)
#   MODEL_ID            Sentiment / training id (default: finbert)
#   BAR_SOURCE          bars.source_api filter; must match data you ingested (default: alpaca)
#   TRAIN_FRAC          Training fraction (default: 0.8)
#   MODEL_OUT           Saved XGBoost JSON path (default: data_store/xgb_daily.json)
#   FEATURES_OUT        Daily features CSV path (default: data_store/daily_features.csv)
#   PREDICT_MODE        latest_bar or session (default: latest_bar)
#   PIPELINE_VERBOSE    Set to 1 to pass -v to every Python step
#   SKIP_BUILD_FEATURES Set to 1 to skip the features CSV step
#   SKIP_TRAIN          Set to 1 to skip training (predict needs an existing MODEL_OUT unless SKIP_PREDICT=1)
#   SKIP_PREDICT        Set to 1 to skip prediction
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

SYMBOLS="${SYMBOLS:-AAPL,MSFT}"
START="${START:-2025-01-01}"
END="${END:-2025-03-18}"
INTERVAL="${INTERVAL:-1d}"
INGEST_CONFIG="${INGEST_CONFIG:-config.yaml}"

SENTIMENT_LIMIT="${SENTIMENT_LIMIT:-500}"
MODEL_ID="${MODEL_ID:-finbert}"
BAR_SOURCE="${BAR_SOURCE:-alpaca}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
MODEL_OUT="${MODEL_OUT:-data_store/xgb_daily.json}"
FEATURES_OUT="${FEATURES_OUT:-data_store/daily_features.csv}"
PREDICT_MODE="${PREDICT_MODE:-latest_bar}"

SKIP_BUILD_FEATURES="${SKIP_BUILD_FEATURES:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"

VERBOSE_FLAG=()
if [[ "${PIPELINE_VERBOSE:-0}" == "1" ]]; then
  VERBOSE_FLAG=(-v)
fi

run_ingest() {
  if [[ -f "$INGEST_CONFIG" ]]; then
    "$PYTHON" scripts/run_ingest.py --config "$INGEST_CONFIG" "${VERBOSE_FLAG[@]}"
  else
    "$PYTHON" scripts/run_ingest.py \
      --symbols "$SYMBOLS" --start "$START" --end "$END" --interval "$INTERVAL" "${VERBOSE_FLAG[@]}"
  fi
}

echo "[1/5] Ingest"
run_ingest

echo "[2/5] Sentiment"
"$PYTHON" scripts/run_sentiment.py \
  --symbols "$SYMBOLS" --limit "$SENTIMENT_LIMIT" --model-id "$MODEL_ID" "${VERBOSE_FLAG[@]}"

if [[ "$SKIP_BUILD_FEATURES" == "1" ]]; then
  echo "[3/5] Daily features CSV (skipped: SKIP_BUILD_FEATURES=1)"
else
  echo "[3/5] Daily features CSV"
  mkdir -p "$(dirname "$FEATURES_OUT")"
  "$PYTHON" scripts/build_daily_features.py \
    --symbols "$SYMBOLS" --model-id "$MODEL_ID" --bar-source "$BAR_SOURCE" \
    --out "$FEATURES_OUT" "${VERBOSE_FLAG[@]}"
fi

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "[4/5] Train XGBoost (skipped: SKIP_TRAIN=1)"
else
  echo "[4/5] Train XGBoost"
  mkdir -p "$(dirname "$MODEL_OUT")"
  "$PYTHON" scripts/train_daily_xgb.py \
    --symbols "$SYMBOLS" --model-id "$MODEL_ID" --bar-source "$BAR_SOURCE" \
    --train-frac "$TRAIN_FRAC" --save-model "$MODEL_OUT" "${VERBOSE_FLAG[@]}"
fi

if [[ "$SKIP_PREDICT" == "1" ]]; then
  echo "[5/5] Predict (skipped: SKIP_PREDICT=1)"
else
  if [[ "$SKIP_TRAIN" == "1" ]] && [[ ! -f "$MODEL_OUT" ]]; then
    echo "error: SKIP_TRAIN=1 but model file missing: $MODEL_OUT" >&2
    exit 1
  fi
  echo "[5/5] Predict"
  "$PYTHON" scripts/predict_daily.py \
    --model "$MODEL_OUT" --mode "$PREDICT_MODE" --symbols "$SYMBOLS" \
    --bar-source "$BAR_SOURCE" --model-id "$MODEL_ID" "${VERBOSE_FLAG[@]}"
fi

echo "run_pipeline.sh finished."
