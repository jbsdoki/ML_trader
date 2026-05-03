# Tests

Run from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest testing
```

``pytest.ini`` sets ``pythonpath = src`` so imports like ``from storage...`` resolve without installing the repo as a package.

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
| `testing/data_retrieval/` | NewsAPI helpers, Finnhub record DF, Alpaca bar normalization |
| `testing/features/` | NYSE session, training labels, training frame integration, daily sentiment join, inference helpers |
| `testing/pipelines/` | `normalize_symbols`, `parse_sources_csv`, `run_ingest_pipeline` mocks, YAML `load_ingest_config_yaml` |
| `testing/storage/` | Schema, `articles_repo`, `bars_repo`, `sentiment_repo` |

Inference CLI: `python scripts/predict_daily.py --model PATH --mode latest_bar --symbols AAPL` (after `train_daily_xgb.py --save-model`).

