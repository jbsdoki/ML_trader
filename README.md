# ML_Trader

Experimentation and automation for stock research: ingest prices and news, score sentiment, build features, train models (e.g. XGBoost), backtest, and (optionally) run scheduled inference / paper trading.

---

## Setup

```bash
conda create -n ml_trader python=3.11 -y
conda activate ml_trader
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add keys for the APIs you use. Never commit `.env`.

---

## How the `data/` ingestion files are used

Each file talks to **one provider** and returns **pandas-friendly** tables (or a small dict for Alpaca clock). You import them when you need that source—there is no single mega “fetch everything” function on purpose.

| Module | Typical use |
|--------|----------------|
| **`data/yfinance_ingest.py`** | Free OHLCV history for backtests and features. Good default when you don’t need a broker-aligned feed. |
| **`data/alpaca_ingest.py`** | OHLCV from Alpaca’s market data API (same account as trading). Use when you want bars consistent with what you’ll trade on, or when yfinance is flaky. **`get_market_clock()`** answers “is the market open?” before your scheduled job places orders. |
| **`data/finnhub_ingest.py`** | Company-scoped news by ticker and date range. Strong fit for “all headlines for `AAPL` this week.” |
| **`data/newsapi_ingest.py`** | Keyword / search-driven news (`everything`) or broad **top headlines**. Use to complement Finnhub or for macro / multi-company queries. |

**Research (notebooks / one-off scripts):** import `fetch_ohlcv`, `fetch_company_news`, etc., explore DataFrames, then save Parquet/SQLite or push to cloud storage.

**Production (VPS / cron):** run **`scripts/run_ingest.py`** (see below) on a schedule: pull bars + news → SQLite upserts (deduped). Later: sentiment → features → model → (optional) Alpaca orders. **`get_market_clock()`** can gate trading steps during market hours.

### Ingest CLI (`scripts/run_ingest.py`)

From the repo root (with `.env` loaded for API keys and optional `ML_TRADER_DATA_DIR`):

```bash
python scripts/run_ingest.py --symbols AAPL,MSFT --start 2025-01-01 --end 2025-03-18 --interval 1d
```

YAML config (`config.yaml` example in repo):

```bash
python scripts/run_ingest.py --config config.yaml
```

- **`--sources`** — comma list: `finnhub`, `newsapi`, `yfinance`, `alpaca` (omit individual APIs you do not use).
- **Dates** — `start` / `end` are **inclusive** calendar days for news; yfinance’s exclusive `end` is adjusted inside `pipelines/ingest_pipeline.py`.
- Exit code **2** if any symbol/source logged an error (partial success still possible); **0** if clean.

Core logic lives in **`pipelines/ingest_pipeline.py`** (`IngestConfig`, `run_ingest_pipeline`, `IngestSummary`).

**`data/__init__.py`** re-exports the main symbols so you can write `from data import fetch_ohlcv, fetch_company_news` from the repo root (with the project directory on `PYTHONPATH` or run from repo root).

---

## Pipeline overview (`_|` outline)

Planned flow: ingestion modules stay **separate**; later stages **combine** their outputs. Boxes are conceptual—some scripts don’t exist yet.

```
                              _|_
                               |
                        (scheduler / cron)
                               |
                               v
+----------------------------------------------------------+
|                     pipeline.py (future)                  |
|  optional: get_market_clock() -> skip if market closed   |
+----------------------------------------------------------+
                               |
       ________________________|__________________________
       |                        |                          |
       v                        v                          v
  prices OHLCV              news articles            session / clock
       |                        |                          |
       |                        |                          |
  yfinance_ingest.py      finnhub_ingest.py          alpaca_ingest.py
  fetch_ohlcv(...)      fetch_company_news(...)     get_market_clock()
       |                  newsapi_ingest.py                |
       |                  fetch_everything(...)             |
       |                  fetch_for_symbol(...)            |
       |                        |                          |
       +------------+-----------+                          |
                    |                                      |
                    v                                      |
            raw store / dedupe (SQLite, Parquet, GCS)       |
                    |                                      |
                    v                                      |
            sentiment (FinBERT / model)                     |
                    |                                      |
                    v                                      |
            feature engineering (TA + sentiment)            |
                    |                                      |
         +----------+----------+                           |
         |                     |                           |
         v                     v                           |
   train (offline)       predict (live)                    |
   models/ + joblib      same feature builder              |
         |                     |                           |
         |                     v                           |
         |              Alpaca trading (paper first) <------+
         |                     |
         v                     v
      backtest            logs / alerts
```

**Legend**

- **Vertical `|`** — control or data flows downward.
- **`_|_`** — top: something external (scheduler) kicks the pipeline.
- **Branches** — multiple ingestion sources feed the same downstream stages.
- **Dashed connection** — clock informs whether trading steps run.

---

## Repo map (current)

| Path | Role |
|------|------|
| `data/` | Ingestion only (per-API fetchers). |
| `pipelines/` | Orchestration (`ingest_pipeline`); more pipelines later. |
| `scripts/` | CLI entrypoints (`run_ingest.py`). |
| `storage/` | SQLite path, schema, `upsert_articles` / `upsert_bars`. |
| `models/` | Model wrappers and factory (e.g. XGBoost). |
| `main.py` | Scratch / experiments (replace with real entrypoints over time). |
| `requirements.txt` | Pip dependencies. |
| `.env.example` | Environment variable template. |

---

## Next steps (not implemented here)

- Sentiment scoring + aggregation (read from `articles` table).
- Single feature-builder module that joins OHLCV + sentiment.
- `inference_pipeline` + Alpaca paper execution + cron on VPS.
