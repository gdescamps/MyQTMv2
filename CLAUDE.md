# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **crisis-avoidance ("risk-off") trading system**. A walk-forward XGBoost model predicts when to be invested vs. in cash, producing a continuous allocation (0 → 100%, x1). The live system runs the strategy nightly on QQQ and executes the resulting allocation each morning on a **Boursorama PEA** account via the unofficial `bourso-cli`, trading the Amundi PEA Nasdaq-100 ETF (**PUST**, x1 only).

All Python code lives directly under `src/` (the old `MyQTM/` submodule is gone). The only git submodule is `external/bourso-api` — the pinned Rust source of `bourso-cli` (see "Bourso CLI" in `BOURSO.md`). Run everything from the repo root with the venv active.

> `README.md` and `BOURSO.md` both describe the current risk-off system: `README.md` is the strategy/architecture overview, `BOURSO.md` is the operational runbook for the PEA/cron/bourso-cli pipeline. Keep all three in sync. (The old 26-ETF momentum design — `etf.py` / `train.py` / `robot.py` — is gone; ignore any lingering references to it in `memory/`.)

## Environment Setup

Python 3.12 + venv:

```bash
./1_setup_interpreter.sh   # rm venv, python3.12 -m venv, pip install -r requirements.txt
source venv/bin/activate
```

`.env` keys (loaded via python-dotenv):
- `FRED` — FRED API key (macro data: VIX, BAA spread)
- `FMPAPI` / `FINHUB` — Financial Modeling Prep / Finnhub (PE data for exploration)
- `BOURSO_ID` / `BOURSO_CODE` — BoursoBank credentials for PEA execution
- `GMAIL_APP_PASSWORD` — 16-char Gmail app password for email notifications
- `GOOGLE_APPLICATION_CREDENTIALS` / `PROJECT_ID` — GCP (legacy LLM, not on the live path)

`bourso-cli` (Rust, v0.5.3) is installed at `~/.local/bin/bourso-cli` — see `BOURSO.md` for build/config. It is **not** in the default cron PATH (see cron pitfalls below).

## Architecture

### 1. Data (`src/download_*.py` → `data/*.parquet`)
- `download_ohlcv.py` — yfinance OHLCV. `RISK_OFF_TICKERS` (QQQ, SPY, ACWI + VIX, TLT) vs. `--extra` exploration tickers.
- `download_macro_data.py` — FRED macro series (BAA spread, etc.) → `data/fred_*.parquet`.
- `download_pe_qqq_top5.py` — cap-weighted PE of top-5 NASDAQ-100 names (FMP), for the webapp/exploration only.

Data is DVC-backed (`data.dvc`, `outputs.dvc`, `.env.dvc` → GCS). `*_revised.parquet` files are the latest-revised series; the unsuffixed files are point-in-time snapshots used for the PIT-vs-revised comparison.

### 2. Risk-off strategy (`src/risk_off_strategy/`)
- `data.py` — `load_data` (price + VIX + BAA spread + TLT, spread shifted J+3 for FRED publication lag), `build_features`, and `build_realtime_target`: a **drawdown state machine** (exit at −10%, re-enter at −5%) producing the binary in/out label.
- `backtest.py` — `walk_forward` (rolling train, 21-day step, interlaced-block feature selection by stable importance `mean/std^1.7`, embargo ≥ lookahead; **no cache** — recomputed every run), `simulate` (**QQQ close-to-close, x1 only, no fees**: `equity = cumprod(1 + ret·alloc)`), plotting (`plot_results`, `plot_recent`, `plot_comparison`). **Execution alignment `exec_lag=1`**: the prob at index `i` is computed ~5 min before the US close[i] and the order goes out then, so the position only earns from close[i] → `ret[i+1]` (no look-ahead). `exec_lag=0` (prob[i]×ret[i]) is look-ahead by one day — diagnostic only.
- `run.py` — entry point. Downloads fresh data (with retry/backoff), runs walk-forward, writes charts and `outputs/<ticker>_strategy/signal.json`. Key config: `MIN_TRAIN=504`, `STEP=21`, `LOOKAHEAD=6`, `DD_EXIT=-0.10`, `DD_REENTER=-0.05`, `PROB_CASH=0.70`, `PROB_FULL=0.75`, `TEMPERATURE=3.0`. x1 only (no leverage, no PUST/projection backtest).
- `test_lookahead.py` — compares lookahead=0 vs 6 across tickers (diagnostic script, x1).

`signal.json` schema and the `allocation = clip((prob - PROB_CASH)/(PROB_FULL - PROB_CASH), 0, 1)` formula are documented in `BOURSO.md`. `status` is `"running"` during the backtest and `"ok"` on success — the morning script refuses to act on a non-`ok` or stale signal (`MAX_SIGNAL_AGE_HOURS=90`, wide enough to tolerate weekend/holiday gaps so Monday and post-long-weekend mornings still execute; older = the evening backtest stopped).

### 3. PEA execution (`src/real_bourso.py` + `src/bourso/`)
- `real_bourso.py` — morning entry point. Reads `signal.json`, checks the PEA via `bourso-cli`, and buys/sells PUST. Sells only when `delta_alloc ≥ SELL_THRESHOLD=0.20` (0.5% sell fee; buys are free). `logs/emergency_off.json` with `{"active": true}` forces allocation to 0%. Two-phase retry (exponential backoff → hourly until deadline). `--execute` runs live (the morning cron uses it); without the flag the script is a dry-run.
- `src/bourso/` — `prepare.py` (dry-run state/capacity), `execute.py` (manual interactive order), `list_accounts.py`, `quote.py`, `notify.py` (Gmail SMTP recap/trade emails, inline-image HTML, `MAILING_LIST` currently just the owner).

### 4. Webapp (`src/webapp.py`)
NiceGUI dashboard (backtests, PE chart, allocations, trade history). Runs in Docker on port 8081 (`webapp_build.sh` / `webapp_run.sh` / `webapp_kill.sh`).

## Cron (the live system)

Two weekday jobs + one daily check (see `crontab -l`); the crontab **must** define `PATH` and `DIR` at the top:

```cron
PATH=/home/greg/.local/bin:/usr/local/bin:/usr/bin:/bin
DIR=/home/greg/data_local/code/MyQTMv2

# 22:30 — evening backtest + PIT comparison + email recap (GPU if free, else CPU)
30 22 * * 1-5 cd $DIR && XGBOOST_DEVICE=auto ./venv/bin/python -m src.risk_off_strategy.run QQQ ...; ... src.bourso.notify --recap ...

# 09:05 — PEA PUST execution at Euronext Paris open (live: --execute)
5 9 * * 1-5 cd $DIR && ./venv/bin/python -m src.real_bourso --execute >> logs/cron_pea.log 2>&1

# 20:00 daily — bourso-cli dry-run tests + upstream-commit check + email report
0 20 * * * cd $DIR && ./venv/bin/python -m src.bourso.check_cli >> logs/cron_bourso_check.log 2>&1
```

**Cron pitfalls (already hit — keep them in mind):**
1. `PATH` line is mandatory — `bourso-cli` lives in `~/.local/bin`, not the default cron PATH.
2. `cd $DIR &&` is mandatory — cron runs from `$HOME`; relative paths (`logs/`, `outputs/`) and module imports break otherwise.
3. `XGBOOST_DEVICE=auto` lets `_detect_device` (in `backtest.py`) use the GPU when it's free and fall back to CPU when another workload is using it (checked via `nvidia-smi`). Force a device with `XGBOOST_DEVICE=cpu` or `=cuda`.

The `cron` service is `enabled` (survives reboots) — the crontab is persisted on disk, not in memory.

Logs: `logs/cron_backtest.log`, `logs/cron_pea.log`, `logs/cron_bourso_check.log`, `logs/trades.jsonl` (order history, read by the webapp).

`bourso-cli` source is a git submodule at `external/bourso-api` pointing at **our fork** `gdescamps/bourso-api`, branch **`myqtm`** = upstream tag **v0.5.3** + 1 patch commit adding a `trade summary` CLI subcommand (azerpas exposes the `get_trading_summary` lib fn but never wires it to the CLI; `src/bourso/prepare.py` needs it to read PEA cash/positions). Submodule has two remotes: `origin`=fork (branch `myqtm`, push), `upstream`=azerpas (tags/main, read). Build/install with `./2_bourso_cli_update.sh` (`--pull` rebases `myqtm` onto the latest azerpas tag, then push the fork + commit the gitlink). `src/bourso/prepare.py` parses `bourso-cli trade summary --account <id>` JSON (falls back to `src/bourso/quote.py` HTTP scrape for the price when a symbol isn't held). `tests/` catches a broken/unpatched build immediately (incl. `test_trade_summary_patch_present`); `test_prepare_live` (marker `live`, gated by `BOURSO_LIVE_TESTS=1`) verifies the real account via `trade summary`. `src.bourso.check_cli` (20:00 cron) runs the build tests + a real `trade summary` and emails an alert: `BUILD CASSE`, `CONNEXION COMPTE KO`, or `nouveau tag vX.Y.Z` (with changelog). See `BOURSO.md`.

## Common Commands

From repo root, venv active:

```bash
# Strategy
python -m src.risk_off_strategy.run QQQ          # backtest + signal.json (QQQ | SPY | ACWI | ALL)

# Data
python -m src.download_ohlcv                     # risk-off tickers (--all / --extra for more)
python -m src.download_macro_data                # FRED macro series

# PEA
python -m src.real_bourso                        # dry-run (what would be done)
python -m src.real_bourso --execute              # live execution
python -m src.bourso.execute pea PUST buy 4      # manual interactive order
python -m src.bourso.notify --recap              # send evening recap email

# Tests / lint
pytest                                           # see pytest.ini
flake8                                           # E501 ignored, max line 120
```

DVC: `dvc pull -r gcs data.dvc` / `dvc pull -r gcs outputs.dvc`; push with `./6_dvc_push.sh` if present.

## Conventions

- Write analysis/scratch files to `myfiles/`.
- Prefer structured commits with clear messages; the user works on the `dev` branch (PRs target `master`).
- Don't run long training/backtests unless asked — the user prefers to kill-and-restart over waiting.
- `BOURSO.md` is the operational runbook for the PEA/cron pipeline; keep it in sync when changing the live path.
