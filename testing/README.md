# Tests

Run from the repository root:

```bash
pip install -r requirements-dev.txt
python -m pytest testing
```

``pytest.ini`` sets ``pythonpath = src`` so imports like ``from storage...`` resolve without installing the repo as a package.

Focused runs (by area, mirroring ``src/``):

```bash
python -m pytest testing/src/data_retrieval -v
python -m pytest testing/src/features -v
python -m pytest testing/src/pipelines -v
python -m pytest testing/src/storage -v
python -m pytest testing/data_store/test_data_format_reference_export.py -v
```

Human-readable schema / ingest column reference → ``logs/testing/data_formats/DATA_FORMATS.md`` (gitignored).

```bash
|--------------------|----------------|
| `testing/src/data_retrieval/` | NewsAPI helpers, Finnhub record DF, Alpaca bar normalization |
| `testing/src/features/` | NYSE session, training labels, training frame integration, daily sentiment join, inference helpers |
| `testing/src/pipelines/` | `normalize_symbols`, `parse_sources_csv`, `run_ingest_pipeline` mocks, YAML `load_ingest_config_yaml` |
| `testing/src/storage/` | Schema, `articles_repo`, `bars_repo`, `sentiment_repo` |
| `testing/integration/` | Cross-module SQLite flows |
| `testing/data_store/` | SQLite inspection helpers; ``test_data_format_reference_export.py`` writes ``logs/testing/data_formats/DATA_FORMATS.md`` for reading |

Inference CLI: `python scripts/predict_daily.py --model PATH --mode latest_bar --symbols AAPL` (after `train_daily_xgb.py --save-model`).
