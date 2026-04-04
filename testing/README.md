# Tests

Run from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest testing
```

Focused runs (by area):

```bash
python -m pytest testing/data_retrieval -v
python -m pytest testing/features -v
python -m pytest testing/pipelines -v
python -m pytest testing/storage -v
python -m pytest testing/storage/test_storage_articles.py -v
```

| Directory / prefix | What it covers |
|--------------------|----------------|
| `testing/data_retrieval/` | yfinance (mocked), NewsAPI helpers, Finnhub record DF, Alpaca bar normalization |
| `testing/features/` | NYSE session, training labels, training frame integration, daily sentiment join, inference helpers |
| `testing/pipelines/` | `normalize_symbols`, `parse_sources_csv`, `run_ingest_pipeline` mocks, YAML `load_ingest_config_yaml` |
| `testing/storage/` | Schema, `articles_repo`, `bars_repo`, `sentiment_repo` |

Inference CLI: `python scripts/predict_daily.py --model PATH --mode latest_bar --symbols AAPL` (after `train_daily_xgb.py --save-model`).

`conftest.py` registers a **stub `yfinance` module** when the package is not installed so imports succeed; tests still **monkeypatch** `Ticker` for deterministic OHLCV. With the real `yfinance` installed, behavior is the same.
