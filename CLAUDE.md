# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repo contains `MyQTM/` as a git submodule — a quantitative trading framework for high-growth tech stocks. All active code lives under `MyQTM/`. Work from within `MyQTM/` or reference paths relative to it.

## Environment Setup

Requires Python 3.11 and a virtual environment:

```bash
cd MyQTM
./1_setup.sh   # creates venv, installs deps, pulls DVC data
source venv/bin/activate
```

Required credentials in `MyQTM/.env`:
- `FMP_APIKEY` — Financial Modeling Prep API key
- `HUGGINEFACE_KEY` — HuggingFace API key
- `GOOGLE_APPLICATION_CREDENTIALS` + `PROJECT_ID` — GCP service account for Gemini LLM
- `TWS_USERID` / `TWS_PASSWORD` — Interactive Brokers credentials

## Common Commands

All commands run from `MyQTM/` with venv activated:

```bash
./2_data.sh              # Download & transform FMP data (runs src/data.py -> data_pipeline.py)
./3_train.sh             # Train XGBoost model (src/train.py)
./4_search_hyperparams.sh # CMA-ES hyperparameter search (src/search_params.py)
./5_benchmark.sh         # Run backtest (src/benchmark.py)
./8_robot.sh             # Start live trading robot (src/robot.py)
```

Run tests:
```bash
pytest                   # all tests
pytest tests/test_fmp.py # single test file
```

Lint: flake8 with E501 ignored, max line length 120 (see `.flake8`).

## Architecture

The system has four main phases:

### 1. Data Pipeline (`src/data_pipeline.py`)
Orchestrates in sequence:
1. `data_download_fmp.py` — fetches raw OHLCV, fundamentals, news, economic indicators from FMP API into `data/fmp_data/`
2. `data_transform_price_trends_indicators_time_series.py` — price/technical indicators
3. `data_transform_key_metrics_time_series.py` — fundamental key metrics
4. `data_transform_economic_indicators_time_series.py` — macro indicators
5. `data_transform_analyst_stock_recommendations_time_series.py`
6. `data_transform_ratings_time_series.py`
7. `data_transform_stock_news.py` — filters news to `RELIABLE_NEWS_SITES`
8. `data_transform_stock_news_to_sentiment_scores.py` — uses Google Gemini (`src/llm.py`) to score news sentiment; results are cached in `llm_cache.pkl`
9. `data_tranform_clean.py` — cleans/aligns all data
10. `data_transform_sentiments_time_series.py`
11. `data_transform_split_intervals.py` — creates interlaced train/test windows

### 2. Model Training (`src/train.py`)
- XGBoost multi-class classifier (long / short / hold) per stock
- Custom `EvalF1Callback` for early stopping on macro F1
- Feature selection via `mean / std^power` importance ranking (configurable `mean_std_power` in `PARAM_GRID`)
- Output saved to `outputs/last_train/`

### 3. Hyperparameter Search (`src/search_params.py`)
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) via `scikit-optimize`
- Optimizes 9 trading thresholds (open/close probabilities for long/short, position sizing)
- Parallel evaluation with stock dropout for robustness
- Output saved to `outputs/last_cma/`

### 4. Live Trading (`src/robot.py`)
- Runs on cron: data fetch at 1:30 PM, trading at 3:30 PM (weekdays, Paris time)
- Connects to Interactive Brokers via `src/ib.py` (uses `ib_insync`)
- Position management: `src/trade.py` — `select_positions_to_open/close`, `open/close_positions`

## Key Configuration (`src/config.py`)

- `BENCHMARK_END_DATE` — update to today before running new data pipeline
- `TRADE_STOCKS` — union of `TRADE_GROWTH_STOCKS + TRADE_VALUE_STOCKS + NEW_CANDIDATE_STOCKS`
- `TS_SIZE = 6` — time series window size for features
- `MAX_POSITIONS = 12` — CMA-ES typically selects 3-4 in practice
- `PARAM_GRID` — XGBoost hyperparameters including `mean_std_power` for feature ranking
- `INIT_SPACE` — CMA-ES search bounds (all in [0.01, 0.999])

## Data Versioning (DVC)

Raw data, model outputs, and LLM cache are versioned via DVC backed by Google Cloud Storage:
```bash
dvc pull -r gcs outputs.dvc
dvc pull -r gcs data/fmp_data.dvc
dvc pull -r gcs llm_cache.pkl.dvc
./6_dvc_push.sh   # push after updating data/models
```

## IB Gateway (Docker)

Interactive Brokers gateway runs via Docker Compose:
```bash
docker-compose up   # starts IB Gateway (see docker-compose.yml)
```
Default mode is `TRADING_MODE=paper` — change in `.env` for live trading.
