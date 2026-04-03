# Tests

Run from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest testing
```

Focused runs:

```bash
python -m pytest testing/test_storage_articles.py -v
python -m pytest testing/test_pipeline_ingest.py -v
python -m pytest testing/test_data_retrieval_yfinance.py -v
```

| Module prefix | What it covers |
|---------------|----------------|
| `test_storage_*` | Schema, `articles_repo`, `bars_repo`, `sentiment_repo` |
| `test_features_*` | NYSE session bounds, training labels, training frame integration |
| `test_data_retrieval_yfinance.py` | Mocked `Ticker.history` + column normalizer |
| `test_pipeline_ingest.py` | `normalize_symbols`, `parse_sources_csv`, `run_ingest_pipeline` with mocks |
| `test_inference_features.py` | `latest_bar_inference_frame`, `build_sentiment_features_for_target_sessions`, `inference_feature_matrix` |

Inference CLI: ``python scripts/predict_daily.py --model PATH --mode latest_bar --symbols AAPL`` (after ``train_daily_xgb.py --save-model``).

`conftest.py` registers a **stub `yfinance` module** when the package is not installed so imports succeed; tests still **monkeypatch** `Ticker` for deterministic OHLCV. With the real `yfinance` installed, behavior is the same.
