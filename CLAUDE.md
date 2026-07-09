# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **crisis-avoidance ("risk-off") trading system**. A **deterministic hand-made rule** (trend-following + vol-managed + macro circuit-breakers — no ML) produces a continuous allocation (0 → 100%, x1). The live system runs the strategy nightly on QQQ and executes the resulting allocation each morning on a **Boursorama PEA** account via the unofficial `bourso-cli`, trading the Amundi PEA Nasdaq-100 ETF (**PUST**, x1 only). (The strategy was migrated from a walk-forward XGBoost model to this hand-made formula in 2026-07 — the ML/oracle code is gone.)

All Python code lives directly under `src/` (the old `MyQTM/` submodule is gone). The only git submodule is `external/bourso-api` — the pinned Rust source of `bourso-cli` (see "Bourso CLI" in `BOURSO.md`). Run everything from the repo root with the venv active.

> `README.md` and `BOURSO.md` both describe the current risk-off system: `README.md` is the strategy/architecture overview, `BOURSO.md` is the operational runbook for the PEA/cron/bourso-cli pipeline. Keep all three in sync. (The old 26-ETF momentum design — `etf.py` / `train.py` / `robot.py` — is gone; ignore any lingering references to it in `memory/`.)

## Environment Setup

Python 3.12 + venv:

```bash
./1_setup_interpreter.sh   # rm venv, python3.12 -m venv, pip install -r requirements.txt
source venv/bin/activate
```

`.env` keys (loaded via python-dotenv):
- `FRED` — FRED API key (macro data: BAA spread, NFCI, CPI YoY — the last two feed the strategy's macro guardrails)
- `FMPAPI` / `FINHUB` — Financial Modeling Prep / Finnhub (PE data for exploration)
- `BOURSO_ID` / `BOURSO_CODE` — BoursoBank credentials for PEA execution
- `GMAIL_APP_PASSWORD` — 16-char Gmail app password for email notifications
- `GOOGLE_APPLICATION_CREDENTIALS` / `PROJECT_ID` — GCP (legacy LLM, not on the live path)

`bourso-cli` (Rust, v0.5.3) is installed at `~/.local/bin/bourso-cli` — see `BOURSO.md` for build/config. It is **not** in the default cron PATH (see cron pitfalls below).

## Architecture

### 1. Data (`src/download_*.py` → `data/*.parquet`)
- `download_ohlcv.py` — yfinance OHLCV. `RISK_OFF_TICKERS` (QQQ, SPY, ACWI + VIX, TLT) vs. `--extra` exploration tickers.
- `download_macro_data.py` — FRED macro series (BAA spread, **NFCI**, **CPI YoY** via `units=pc1`) → `data/fred_*.parquet`. NFCI + CPI feed the strategy's macro circuit-breakers.
- `download_pe_qqq_top5.py` — cap-weighted PE of top-5 NASDAQ-100 names (FMP); kept for the webapp NDX Top-5/20 market-cap/concentration tabs only (no longer the valuation-context series).
- `download_shiller_cape.py` — S&P 500 **CAPE** (Shiller P/E10) + **Excess CAPE Yield (ECY)**, full history **1871 → today, daily-updated** → `data/shiller/cape_ecy.parquet` (monthly) + `outputs/shiller/cape_ecy.png`. CAPE primary source = **multpl.com** table scrape (correct up-to-date E10 denominator + live current-month point; the naive "freeze Shiller's E10 and scale by price" is ~+6% wrong over 2y as earnings grow); **fallback + cross-check = Shiller's `ie_data.xls`** (Yale, stale ~2024-09). ECY = Shiller's ECY column for history + tail reconstructed as `1/CAPE − DFII10` (FRED 10y TIPS real rate, recalibrated to Shiller's level — TIPS only exist since 2003). This is the **valuation-context** variable (rate-adjusted, era-comparable — replaced the top-5 PE in the backtest chart + webapp). Context only, **not** in the strategy. Refreshed daily to stay current: on weekdays `risk_off_strategy.run` calls it best-effort just before the backtest (`refresh_valuation_context`, so the chart always reads the same-day CAPE — no implicit dependency on a separate cron); a weekend-only cron (22:20 Sat/Sun) keeps the webapp current when no backtest runs. Needs `xlrd` + `lxml`.
- `download_ndx_excess_yield.py` — QQQ-specific valuation lens: **excess earnings yield** of the NDX **top-5 / top-10** (cap-weighted 1/PE, **trailing + forward**) minus the real 10y rate (FRED DFII10) → `data/pe/ndx_excess_yield.json`. The right analog of ECY for concentrated growth leaders — a true CAPE (10y-smoothed earnings) is meaningless on compounders (10y-avg earnings ≪ current → absurd CAPE), so we keep the "yield − real rate" structure without cyclical smoothing. Trailing→forward gap = the growth ("AI") credit made explicit. Source = **yfinance** `trailingPE`/`forwardPE`/`marketCap` (FMP v3 is dead — legacy endpoints 403 since 2025-08-31; several tickers premium-gated on the current FMP plan). Shown in the webapp Valorisation tab beside the S&P ECY. Context only; refreshed daily alongside the CAPE (weekday backtest `refresh_valuation_context` + weekend cron 22:20).

Data is DVC-backed (`data.dvc`, `outputs.dvc`, `.env.dvc` → GCS). `*_revised.parquet` files are the latest-revised series; the unsuffixed files are point-in-time snapshots used for the PIT-vs-revised comparison.

### 2. Risk-off strategy (`src/risk_off_strategy/`)
Deterministic hand-made allocation (no ML), backtested from 2000. x1 only, close-to-close, no fees, `exec_lag=1` (the allocation computed at the evening close is executed the next morning — no look-ahead).
- `strategy.py` — `compute_allocation(price, nfci, cpi)`:
  ```
  s = SMA(price, 250);  rvol20 = std(ret, 20)·√252
  decay = clip(1 + (price/s − 1)/0.15, 0, 1)          # cuts exposure when far below the MA
  alloc = where(price > s, 1, 0.8·clip(0.08/rvol20, 0, 1)·decay)
  then cash (0) if NFCI > 0.5 or CPI YoY > 7%          # macro circuit-breakers (anti-crisis guardrails)
  ```
  Params: `SMA_LONG=250`, `VOL_TARGET=0.08`, `BELOW_SCALE=0.8`, `GAP_CUTOFF=0.15`, `NFCI_OFF=0.5`, `CPI_OFF=7.0`. Also `simulate(price, alloc)` → equity/metrics (gross). Backtest QQQ 2000-2026: CAGR ~11.9%, maxDD −36%, Sharpe 0.81. **Honest caveat**: `SMA250/V/L2` are best-in-sample (a 2-fold held-out does **not** beat the `trend150+VM` baseline — only the longer-MA drawdown gain is robust); the macro circuit-breakers are fit on n≈4 events (NFCI>0.5 ≈ 2008, CPI>7% ≈ 2022) and are kept as anti-crisis **guardrails**, not alpha.
  - `simulate_net(price, alloc, leverage, funding)` — **net-of-fees** backtest with discretized Bourso execution: instruments modelled in QQQ-index space (PUST = ret − TER 0.30%; LQQ = 2·ret − (funding + swap-spread 0.40% + TER 0.60%)/252; cash = 0). **Drag-minimal realization** of target exposure `E = leverage·alloc`: `E≤1` → PUST=E + cash, `E>1` → PUST=2−E + LQQ=E−1 (LQQ only carries the leverage above 100% → financing drag paid at the minimum). **Asymmetric no-trade band** (`·leverage`): rebalance up only when `E_target−E_eff ≥ BUY_THR_ALLOC` (0.25, buys free) and down only when `E_eff−E_target ≥ SELL_THR_ALLOC` (0.50, 0.5% sell fee → de-lever only in big steps). Returns `(equity, fees_yr%, revis_yr)`. The `(0.25/0.50)` couple **maximises net Sharpe AND Calmar** and dominates the old symmetric `0.20` band on every axis (swept + half-split-validated in `myfiles/asym_band_optimize.py`). Two lessons: (1) widening both bands hard kills the vol-managed arm's whipsaw → fewer losing sells → *more* return AND *less* drawdown; (2) the useful asymmetry is moderate (sell = 2×buy), not extreme — tracking the up-move too finely over-invests and pays for it on the pullback; beyond `sell ≥ exposure 1.0` the band cancels the risk-off (x2 maxDD → −52%). This is what the backtest chart + `run.py` log now report. Net x1 QQQ 2000-2026 ≈ CAGR 9.9%, maxDD −17.6%, Sharpe 0.85, Calmar 0.57, ~0.32%/yr fees, ~4 revis/yr; net x2 (same thresholds ·leverage) ≈ 17.9%/−33%/Calmar 0.54 — leverage still trades drawdown for return. (Deeper x2 optimization — vol-scaled exposure, `e_max` cap, PUST-base — in `myfiles/x2_lqq_optimize.py`.)
- `data.py` — `load_price(ticker, start, end)` and `load_macro(dates)` → NFCI (lag 5) + CPI YoY (lag 15), aligned & publication-lagged; returns `None` for an arm if its FRED file is absent (that macro guardrail is silently disabled — graceful degradation). `load_funding(dates)` → USD short rate (FRED **DFF**, added to `download_macro_data.py`) as an annual fraction, for the LQQ financing in the net backtest (`None` if absent → the chart falls back to gross).
- `backtest.py` — `plot_backtest(price, alloc, nfci, cpi, cape, ecy, funding=None, last_days=None)`: 6-panel chart since 2000 (equity+SMA250, vol, allocation, NFCI, inflation, CAPE/ECY S&P on a twin axis; the CAPE/ECY panel also carries `ndx_ey` — today's NDX top-5 excess-earnings-yield trailing/forward as horizontal reference lines, snapshot from `load_ndx_excess_snapshot`, since no forward history exists). The results table + equity curves are **net of fees** (x1/x2.0 solid = net, dotted = gross to show the fee cost) with `frais/an` + `revis/an` columns; `last_days=252/21` re-uses the same layout windowed for the 1y/1m views.
- `run.py` — entry point. Refreshes data (retry/backoff), computes the allocation for **QQQ only** (pass another ticker explicitly for exploration), writes `backtest.png`/`_1y`/`_1m` and `outputs/qqq_strategy/signal.json`.

`signal.json` keeps the schema read by `real_bourso.py` (`status`, `date`, `probability`, `allocation`, `timestamp`, + a `macro_off` bool); `probability` now equals `allocation` (kept for backward compat). `status` is `"running"` during the run and `"ok"` on success — the morning script refuses to act on a non-`ok` or stale signal (`MAX_SIGNAL_AGE_HOURS=90`, wide enough to tolerate weekend/holiday gaps so Monday and post-long-weekend mornings still execute; older = the evening run stopped).

### 3. PEA execution (`src/real_bourso.py` + `src/bourso/`)
- `real_bourso.py` — morning entry point. Reads `signal.json`, checks the PEA via `bourso-cli`, and buys/sells PUST. Same **asymmetric no-trade band as the backtest** (imports `BUY_THR_ALLOC`/`SELL_THR_ALLOC` from `strategy.py` — single source of truth): buys up only when `delta_alloc ≥ 0.25` (free), de-levers only when `delta_alloc ≤ −0.50` (0.5% sell fee). Both thresholds live in portfolio-weight space, so they hold for PUST **and** LQQ (LQQ exposure = 2×weight, but the backtest's exposure threshold = `thr·leverage`, so per-weight it's the same `thr`). Forced full rebalance to/from cash (like `simulate_net`'s `force`): a macro-off/emergency `alloc=0` **always** liquidates even below the sell band, and a first entry from full cash executes even below the buy band. **Split/anomaly guard** (`detect_split`): if the instrument's price jumps by ≥`SPLIT_DETECT_FACTOR`=1.5× (or ÷1.5) vs the previous session (stored per-instrument in `logs/last_price.json`, updated on each LIVE run) — the signature of a split (e.g. LQQ /200) or a broker display glitch, not a market move (a 2× ETF can't do ±50% in a session) — it **takes no position that day** (avoids trading on a distorted price the day of the split) and emails a "SPLIT detecte" alert; trading resumes next session (reference updated to the post-split price). `logs/emergency_off.json` with `{"active": true}` forces allocation to 0%. Orders are **limit orders with a tolerance** (`LIMIT_TOLERANCE_PCT`=1.5%: limit = last ±1.5%, buy:+, sell:−) — buffers the opening gap so they fill reliably while still bounding the price (a bare at-the-price limit didn't fill on 07-07 when the price ticked away). Two-phase retry (exponential backoff → hourly until deadline). `--execute` runs live (the morning cron uses it); without the flag the script is a dry-run.
- `src/bourso/` — `prepare.py` (dry-run state/capacity), `execute.py` (manual interactive order), `list_accounts.py`, `quote.py`, `notify.py` (Gmail SMTP recap/trade emails, inline-image HTML, `MAILING_LIST` currently just the owner).

### 4. Webapp (`src/webapp.py`)
NiceGUI dashboard (backtests, PE chart, allocations, trade history). Runs in Docker on port 8081 (`webapp_build.sh` / `webapp_run.sh` / `webapp_kill.sh`).

## Cron (the live system)

Two weekday jobs + one weekend refresh + one daily check (see `crontab -l`); the crontab **must** define `PATH` and `DIR` at the top:

```cron
PATH=/home/greg/.local/bin:/usr/local/bin:/usr/bin:/bin
DIR=/home/greg/data_local/code/MyQTMv2

# 22:30 — evening strategy run (refresh CAPE/NDX inclus + signal.json + charts) + email recap
30 22 * * 1-5 cd $DIR && ./venv/bin/python -m src.risk_off_strategy.run QQQ ...; ... src.bourso.notify --recap ...

# 09:05 — PEA PUST execution at Euronext Paris open (live: --execute)
5 9 * * 1-5 cd $DIR && ./venv/bin/python -m src.real_bourso --execute >> logs/cron_pea.log 2>&1

# 22:20 weekend-only — refresh CAPE/ECY + NDX excess yield for the webapp (weekdays: the 22:30 backtest does it)
20 22 * * 6,0 cd $DIR && ./venv/bin/python -m src.download_shiller_cape >> logs/cron_cape.log 2>&1; ./venv/bin/python -m src.download_ndx_excess_yield >> logs/cron_cape.log 2>&1

# 20:00 daily — bourso-cli dry-run tests + upstream-commit check + email report
0 20 * * * cd $DIR && ./venv/bin/python -m src.bourso.check_cli >> logs/cron_bourso_check.log 2>&1
```

**Cron pitfalls (already hit — keep them in mind):**
1. `PATH` line is mandatory — `bourso-cli` lives in `~/.local/bin`, not the default cron PATH.
2. `cd $DIR &&` is mandatory — cron runs from `$HOME`; relative paths (`logs/`, `outputs/`) and module imports break otherwise.

(The strategy is a plain deterministic formula now — no GPU/XGBoost, so no `XGBOOST_DEVICE` env is needed anymore.)

The `cron` service is `enabled` (survives reboots) — the crontab is persisted on disk, not in memory.

Logs: `logs/cron_backtest.log`, `logs/cron_pea.log`, `logs/cron_bourso_check.log`, `logs/trades.jsonl` (order history, read by the webapp).

`bourso-cli` source is a git submodule at `external/bourso-api` pointing at **our fork** `gdescamps/bourso-api`, branch **`myqtm`** = upstream tag **v0.5.3** + 1 patch commit adding a `trade summary` CLI subcommand (azerpas exposes the `get_trading_summary` lib fn but never wires it to the CLI; `src/bourso/prepare.py` needs it to read PEA cash/positions). Submodule has two remotes: `origin`=fork (branch `myqtm`, push), `upstream`=azerpas (tags/main, read). Build/install with `./2_bourso_cli_update.sh` (`--pull` rebases `myqtm` onto the latest azerpas tag, then push the fork + commit the gitlink). `src/bourso/prepare.py` parses `bourso-cli trade summary --account <id>` JSON (falls back to `src/bourso/quote.py` HTTP scrape for the price when a symbol isn't held). `tests/` catches a broken/unpatched build immediately (incl. `test_trade_summary_patch_present`); `test_prepare_live` (marker `live`, gated by `BOURSO_LIVE_TESTS=1`) verifies the real account via `trade summary`. `src.bourso.check_cli` (20:00 cron) runs the build tests + a real `trade summary` and emails an alert: `BUILD CASSE`, `CONNEXION COMPTE KO`, or `nouveau tag vX.Y.Z` (with changelog). See `BOURSO.md`.

## Common Commands

From repo root, venv active:

```bash
# Strategy
python -m src.risk_off_strategy.run              # backtest + signal.json (QQQ only by default)

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
