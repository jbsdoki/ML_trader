# ML_Trader runbook

Step-by-step commands from the **repository root** with `conda activate ml_trader` (or your env). Adjust dates and symbols to match your run.

## Prerequisites

1. **Environment**
   - Recommended: `conda env create -f environment.yml` then `conda activate ml_trader`
   - Or: `pip install -r requirements.txt` (SQLite **CLI** is optional; use `conda install sqlite` if you want the `sqlite3` command)
2. **Secrets:** copy `.env.example` to `.env` and set API keys (Finnhub, NewsAPI, Alpaca, etc.).
3. **Database path:** by default the DB is `./data_store/ml_trader.db` (or set `ML_TRADER_DATA_DIR` to an absolute folder; the file is `ml_trader.db` inside it).

## 1. Ingest news and bars

Using CLI flags:

```bash
python scripts/run_ingest.py --symbols AAPL,MSFT --start 2025-01-01 --end 2025-03-18 --interval 1d
```

Or YAML:

```bash
python scripts/run_ingest.py --config config.yaml
```

- **Sources:** `--sources finnhub,newsapi,yfinance,alpaca` (omit any you do not need). If Yahoo rate-limits you, ingest may still write **Alpaca** bars; training defaults expect **`bars.source_api`** to match what you have (see step 4).
- Exit code **2** means at least one source/symbol logged an error (partial success is possible).

## 2. Score sentiment (FinBERT)

```bash
python scripts/run_sentiment.py --symbols AAPL,MSFT --limit 500 -v
```

Uses `article_sentiment.model_id` (default `finbert`); align with training/predict `--model-id`.

## 3. (Optional) Export daily features CSV

Joins bars + NYSE-session sentiment (default **open-cutoff** window: prior close through before session open):

```bash
python scripts/build_daily_features.py --symbols AAPL,MSFT --model-id finbert --bar-source alpaca --out data_store/daily_features.csv
```

Set `--bar-source` to the vendor you actually have in `bars` (check with SQL in `data_store/readme.md`).

## 4. Train XGBoost (open-to-close return)

```bash
python scripts/train_daily_xgb.py --symbols AAPL --model-id finbert --bar-source alpaca --train-frac 0.8 --save-model data_store/xgb_daily.json
```

- **`--bar-source`** must match rows in `bars` for that symbol (e.g. **alpaca** if you have no yfinance rows).
- Requires enough overlapping **bars + scored sentiment**; otherwise you get “No training rows”.

## 5. Predict

After a saved model exists:

```bash
python scripts/predict_daily.py --model data_store/xgb_daily.json --mode latest_bar --symbols AAPL --bar-source alpaca
```

- **`latest_bar`:** one row per symbol using the **latest complete** daily bar + sentiment (needs bars in SQLite).
- **`session`:** sentiment only for a target session (use `--session-date YYYY-MM-DD`) before that day’s bar exists.

Match **`--sentiment-mode`** and **`--model-id`** to training when you rely on the same feature logic.

## 6. Tests

```bash
pip install -r requirements-dev.txt
python -m pytest testing
```

## Troubleshooting

| Symptom | Things to check |
|---------|-------------------|
| No training rows | `SELECT source_api, COUNT(*) FROM bars WHERE symbol='AAPL' GROUP BY source_api;` — pass `--bar-source` to match. |
| Model file not found | Training must finish and `--save-model` must run (training failed early if no rows). |
| yfinance errors | Rate limits; use Alpaca bars or retry later. |

## Full pipeline (bash)

From the repository root (Git Bash, WSL, or Linux):

```bash
bash scripts/run_pipeline.sh
```

Uses `config.yaml` for ingest when present; otherwise set `SYMBOLS`, `START`, and `END` (see comments at the top of `scripts/run_pipeline.sh`). Important variables include `BAR_SOURCE` (default `alpaca`, must match `bars.source_api` in SQLite), `MODEL_OUT`, and optional `SKIP_BUILD_FEATURES`, `SKIP_TRAIN`, `SKIP_PREDICT` for partial runs.

## Related docs

- `README.md` — project overview and ingest details
- `data_store/readme.md` — SQLite file location and ad hoc queries
- `config.yaml` — ingest-only parameters (not the conda env; see `environment.yml`)
