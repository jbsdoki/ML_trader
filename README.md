# ML_Trader

Experimentation and automation for stock research: ingest prices and news, score sentiment, build features, train models (e.g. XGBoost), run daily predictions, and (optionally) schedule inference. A broader backtest harness and Alpaca execution are listed under **Next steps**.

---

## Setup

```bash
conda create -n ml_trader python=3.11 -y
conda activate ml_trader
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add keys for the APIs you use. Never commit `.env`.

Alternatively create the conda env (Python + SQLite CLI + pip deps) with `conda env create -f environment.yml`.

**Operational run order** (ingest through prediction): see **`runbook.md`** in the repo root. SQLite layout and ad hoc queries: **`data_store/readme.md`**.

**One-shot full pipeline (bash):** from the repo root, `bash scripts/run_pipeline.sh` (Git Bash / WSL / Linux). Uses `config.yaml` for ingest when that file exists; otherwise set `SYMBOLS`, `START`, and `END`. Override paths and steps via env vars documented in the script header.

---

## How the `src/data_retrieval/` ingestion modules are used

Each file talks to **one provider** and returns **pandas-friendly** tables (or a small dict for Alpaca clock). You import them when you need that source—there is no single mega “fetch everything” function on purpose.

| Module | Typical use |
|--------|----------------|
| **`src/data_retrieval/alpaca_ingest.py`** | OHLCV from Alpaca’s market data API (same account as trading). Ingest writes **`bars`** with `source_api = alpaca`. **`get_market_clock()`** answers “is the market open?” before your scheduled job places orders. |
| **`src/data_retrieval/finnhub_ingest.py`** | Company-scoped news by ticker and date range. Strong fit for “all headlines for `AAPL` this week.” |
| **`src/data_retrieval/newsapi_ingest.py`** | Keyword / search-driven news (`everything`) or broad **top headlines**. Use to complement Finnhub or for macro / multi-company queries. |

**Research (notebooks / one-off scripts):** import `fetch_stock_bars`, `fetch_company_news`, etc., explore DataFrames, then save Parquet/SQLite or push to cloud storage.

**Production (VPS / cron):** run **`scripts/run_ingest.py`** (see below) on a schedule: pull bars + news → SQLite upserts (deduped). Then **`scripts/run_sentiment.py`**, optional **`scripts/build_daily_features.py`**, **`scripts/train_daily_xgb.py`** / **`scripts/predict_daily.py`** — see **`runbook.md`** for the full order and flags. **`get_market_clock()`** can gate trading steps during market hours.

### Ingest CLI (`scripts/run_ingest.py`)

From the repo root (with `.env` loaded for API keys and optional `ML_TRADER_DATA_DIR`):

```bash
python scripts/run_ingest.py --symbols AAPL,MSFT --start 2025-01-01 --end 2025-03-18 --interval 1d
```

YAML config (`config.yaml` example in repo):

```bash
python scripts/run_ingest.py --config config.yaml
```

- **`--sources`** — comma list: `finnhub`, `newsapi`, `alpaca` (omit individual APIs you do not use). Bars are fetched only from **Alpaca** when `alpaca` is enabled.
- **Dates** — `start` / `end` are **inclusive** calendar days for news and the bar window.
- Exit code **2** if any symbol/source logged an error (partial success still possible); **0** if clean.

Core logic lives in **`src/pipelines/ingest_pipeline.py`** (`IngestConfig`, `run_ingest_pipeline`, `IngestSummary`).

**`src/data_retrieval/__init__.py`** re-exports the main symbols so you can write `from data_retrieval import fetch_stock_bars, fetch_company_news` after putting **`src/`** on ``PYTHONPATH`` (``export PYTHONPATH=src``) or by using the repo ``scripts/`` entrypoints, which add ``src`` automatically.

---

## Pipeline overview (`_|` outline)

Ingestion modules stay **separate**; orchestration is **per-stage scripts** or **`scripts/run_pipeline.sh`** (details and CLI flags: **`runbook.md`**).

```
                              _|_
                               |
                        (scheduler / cron)
                               |
                               v
+----------------------------------------------------------+
|  scripts/run_ingest.py  ->  src/pipelines/ingest_pipeline    |
|  optional: get_market_clock() before live trading steps   |
+----------------------------------------------------------+
                               |
       ________________________|__________________________
       |                        |                          |
       v                        v                          v
  prices OHLCV              news articles            session / clock
       |                        |                          |
  alpaca_ingest.py        finnhub_ingest.py          alpaca_ingest.py
  fetch_stock_bars(...)   fetch_company_news(...)    get_market_clock()
       |                  newsapi_ingest.py                |
       |                  fetch_everything(...)             |
       |                  fetch_for_symbol(...)            |
       |                        |                          |
       +------------+-----------+                          |
                    |                                      |
                    v                                      |
            SQLite: bars, articles (src/storage/)            |
                    |                                      |
                    v                                      |
            scripts/run_sentiment.py (FinBERT)              |
                    |                                      |
                    v                                      |
     optional: scripts/build_daily_features.py              |
                    |                                      |
         +----------+----------+                           |
         |                     |                           |
         v                     v                           |
 scripts/train_daily_xgb.py  scripts/predict_daily.py       |
 (joblib/json model)         bar-source + model-id match   |
         |                     |                           |
         |                     v                           |
         |              Alpaca paper / live (future) <-----+
         |                     |
         v                     v
   evaluation / CSV         logs / alerts
```

**Legend**

- **Vertical `|`** — control or data flows downward.
- **`_|_`** — something external (scheduler) kicks the pipeline.
- **Branches** — multiple ingestion sources feed the same SQLite store.
- **Training vs predict** — use the same **`--bar-source`** and **`--model-id`** as in the runbook so features line up.

---

## Repo map (current)

| Path | Role |
|------|------|
| `src/` | Library code: `storage`, `data_retrieval`, `pipelines`, `features`, `models`, `sentiment`. |
| `src/data_retrieval/` | Ingestion only (per-API fetchers). |
| `src/pipelines/` | `ingest_pipeline` (`IngestConfig`, `run_ingest_pipeline`). |
| `scripts/` | `run_ingest.py`, `run_sentiment.py`, `build_daily_features.py`, `train_daily_xgb.py`, `predict_daily.py`, `run_pipeline.sh` (full pipeline). |
| `src/features/` | Daily features, NYSE session windows, inference helpers. |
| `src/storage/` | SQLite path, schema, upserts, sentiment reads. |
| `data_store/` | Default DB directory; see `data_store/readme.md`. |
| `src/models/` | Model wrappers and factory (e.g. XGBoost). |
| `testing/` | `pytest` suite (`requirements-dev.txt`); subdirs `data_retrieval/`, `features/`, `pipelines/`, `storage/`. |
| `requirements.txt` | Pip dependencies. |
| `environment.yml` | Conda env (Python + SQLite CLI + pip deps). |
| `.env.example` | Environment variable template. |
| `runbook.md` | End-to-end command order (ingest through predict). |

---

## Next steps (not implemented here)

- Backtest / walk-forward evaluation harness beyond train/validate split.
- Alpaca paper or live execution wired to `predict_daily` outputs.
