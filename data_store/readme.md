# Data store (SQLite)

The primary database file is **`ml_trader.db`** in this directory when `ML_TRADER_DATA_DIR` is unset (default: `./data_store` under the process working directory). Override with env var **`ML_TRADER_DATA_DIR`** for cron or other machines.

**End-to-end commands** (ingest through predict) are documented in the repo root **`runbook.md`**.

## Tables

| Table | Role |
|-------|------|
| **`articles`** | Ingested news: `dedupe_key`, `symbol`, `published_at`, headline, summary, URL, etc. |
| **`bars`** | OHLCV: composite primary key on `source_api`, `symbol`, `bar_interval`, `bar_ts`. Vendor is stored in **`source_api`** (e.g. `alpaca`, `yfinance`). |
| **`article_sentiment`** | FinBERT (or other) scores per article and `model_id`; FK to `articles.dedupe_key`. |

Training and prediction scripts filter bars with **`--bar-source`**; it must match the **`source_api`** values you actually ingested (many setups have **Alpaca** only if Yahoo was rate-limited).

## Optional: SQLite CLI

Install the **`sqlite3`** shell in your conda env (not the Python `sqlite3` module):

```bash
conda install sqlite
```

From the **repository root**:

```bash
sqlite3 .\data_store\ml_trader.db ".tables"
```

## Useful queries

**List user tables:**

```sql
SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name;
```

**Bars per symbol and vendor:**

```sql
SELECT source_api, COUNT(*) AS n FROM bars WHERE symbol = 'AAPL' GROUP BY source_api;
```

**Column layout (example: `bars`):**

```sql
PRAGMA table_info(bars);
```

**Sample rows with headers (SQLite 3.30+):**

```bash
sqlite3 .\data_store\ml_trader.db -header -column "SELECT * FROM bars WHERE symbol = 'AAPL' ORDER BY bar_ts DESC LIMIT 10;"
```

**Without the `sqlite3` executable** — use Python:

```bash
python -c "import sqlite3; c=sqlite3.connect('data_store/ml_trader.db'); print(c.execute('PRAGMA table_info(bars)').fetchall())"
```

## Example `PRAGMA` output (reference)

<details>
<summary><code>PRAGMA table_info(articles)</code> (expand)</summary>

Same columns as defined in `storage/schema.py`: `dedupe_key` (PK), `source_api`, `symbol`, `published_at`, `headline`, `summary`, `source_name`, `url`, `author`, `category`, `related`, `image_url`, `content_snippet`, `ingested_at`.

</details>

<details>
<summary><code>PRAGMA table_info(article_sentiment)</code> (expand)</summary>

`dedupe_key` + `model_id` (composite PK), `score`, `prob_pos`, `prob_neg`, `prob_neutral`, `scored_at`, `text_hash`, `error`.

</details>

<details>
<summary><code>PRAGMA table_info(bars)</code> (expand)</summary>

`source_api`, `symbol`, `bar_interval`, `bar_ts` (composite PK), OHLCV, `vwap`, `trade_count`, `dividends`, `stock_splits`, `ingested_at`.

</details>

## Other artifacts

- **`daily_features.csv`** (or `.parquet`) — optional export from `scripts/build_daily_features.py`
- **`xgb_daily.json`** — optional saved model from `scripts/train_daily_xgb.py --save-model`
