# MyQTMv2 — Risk-Off Strategy

A **crisis-avoidance trading system** for QQQ (Nasdaq-100). A **deterministic hand-made
rule** (trend-following + volatility targeting + macro circuit-breakers — no machine
learning) turns price and two macro series into a **continuous allocation** (0% → 100%,
x1). The goal is not to beat the market on return alone, but to **keep most of the upside
while cutting drawdowns**.

The live system runs nightly, writes a signal, and the next morning executes that
allocation on a **Boursorama PEA** account (ETF **PUST**, Amundi PEA Nasdaq-100, x1).
See [`BOURSO.md`](BOURSO.md) for the operational runbook and [`CLAUDE.md`](CLAUDE.md) for the
code map.

> The strategy was migrated in 2026-07 from a walk-forward XGBoost model to this hand-made
> formula (the ML/oracle code has been removed). Extensive testing showed ML did not beat a
> simple trend + vol-managed rule out-of-sample.

## Backtest (QQQ 2000-2026)

**QQQ close-to-close, x1, without fees.** Execution alignment is honest (`exec_lag=1`): the
allocation computed at the evening close of day `i` is executed the next morning, so the
position only earns the **next** return (close[i]→close[i+1]) — **no 1-day look-ahead**.

| Strategy | CAGR | maxDD | Sharpe | Calmar |
|----------|-----:|------:|-------:|-------:|
| Buy & Hold | 8.7% | −83% | 0.45 | 0.10 |
| trend150 + VM (baseline) | 10.2% | −65% | 0.65 | 0.16 |
| **deployed rule** | **11.9%** | **−36%** | **0.81** | **0.33** |

> Regenerate with `python -m src.risk_off_strategy.run` — the chart
> (`outputs/qqq_strategy/backtest.png`) shows equity vs B&H since 2000 with the volatility,
> allocation, NFCI and inflation panels; `backtest_1y.png` / `backtest_1m.png` are the same
> layout windowed to the last year / month.

## How it works

```
┌──────────────┐   ┌───────────────────────────┐   ┌──────────────────┐
│ QQQ close    │──▶│ trend (SMA250) + vol-target│──▶│ allocation       │
│ NFCI (FRED)  │   │ + gap decay + macro cutoff │   │ 0→100% (x1)      │
│ CPI YoY(FRED)│   │  (deterministic formula)   │   │ PEA execution    │
└──────────────┘   └───────────────────────────┘   └──────────────────┘
```

### 1. Data (`src/risk_off_strategy/data.py`)

Three daily inputs, all free:

| Series | Source | Role |
|--------|--------|------|
| QQQ close | yfinance | the asset traded (trend + realized vol) |
| NFCI | FRED | financial-conditions stress (lag 5d for publication) |
| CPI YoY | FRED (`units=pc1`) | inflation regime (lag 15d) |

NFCI and CPI drive the macro guardrails only; if a FRED file is missing, `load_macro`
returns `None` for that arm and the corresponding guardrail is silently disabled
(graceful degradation). Point-in-time and `*_revised.parquet` series are both kept as a
record of data revisions.

### 2. The allocation formula (`src/risk_off_strategy/strategy.py`)

```
s      = SMA(price, 250)                         # long trend
rvol20 = std(daily returns, 20) · √252           # realized vol, annualized
decay  = clip(1 + (price/s − 1) / 0.15, 0, 1)     # → 0 when price is ≥15% below the MA

alloc  = where(price > s, 1.0,                    # in the trend: fully invested
               0.8 · clip(0.08/rvol20, 0, 1) · decay)   # below: throttle by vol, fade out

# macro circuit-breakers (cash total):
if NFCI    > 0.5  → alloc = 0                     # credit stress (caught 2008)
if CPI YoY > 7%   → alloc = 0                     # extreme inflation (caught 2022)
```

Two edges do the work: **trend persistence** (`price > SMA250`) keeps you in during
uptrends, and **volatility clustering** (`0.08/rvol20`) cuts exposure as vol rises below
the trend — because a volatility crisis is a self-feeding process and vol is
autocorrelated, reacting to vol with a 1-day lag captures most of the protection. The
**gap decay** forces exposure to zero on deep breaks, and the **macro cutoffs** are
anti-crisis guardrails for the two slow bears price/vol catch late.

### 3. Allocation & simulation (`simulate`)

x1, no leverage; `equity = cumprod(1 + ret · alloc_shifted)` with `exec_lag=1`. No
transaction fees and no ETF TER are modeled. The live signal is simply `alloc[-1]` (the
latest close's decision, executed next morning).

### 4. Honest caveats

- `SMA250`, `VOL_TARGET=0.08`, `BELOW_SCALE=0.8` are the **best in-sample** grid point; a
  2-fold held-out does **not** beat the `trend150+VM` baseline on Sharpe. The one robust,
  non-overfit gain is the **longer moving average** (better drawdown), not the fine tuning.
- The macro cutoffs are fit on **n≈4 events** (NFCI>0.5 ≈ 2008, CPI>7% ≈ 2022). They are
  kept as **guardrails** (asymmetric payoff: on 4 firings, 3 big declines avoided for 1
  missed rebound), **not** as an alpha claim.

## Repository layout

| Path | Role |
|------|------|
| `src/risk_off_strategy/strategy.py` | the allocation formula + `simulate` |
| `src/risk_off_strategy/data.py` | `load_price` + `load_macro` (NFCI, CPI YoY) |
| `src/risk_off_strategy/backtest.py` | 5-panel chart since 2000 (`plot_backtest`, `last_days` for 1y/1m) |
| `src/risk_off_strategy/run.py` | entry point → charts + `signal.json` (QQQ only) |
| `src/download_ohlcv.py`, `download_macro_data.py` | refresh `data/*.parquet` |
| `src/real_bourso.py`, `src/bourso/` | morning PEA execution + email notifications |
| `src/bourso/check_cli.py` | daily bourso-cli health check (tests + account + upstream) → email |
| `external/bourso-api` (submodule) | pinned fork [gdescamps/bourso-api](https://github.com/gdescamps/bourso-api), branch `myqtm` = upstream tag + `trade summary` patch |
| `2_bourso_cli_update.sh` | build/install `bourso-cli` from the submodule (`--pull` to bump) |
| `tests/` | pytest dry-run + live-account checks for `bourso-cli` |
| `src/webapp.py` | NiceGUI dashboard (backtests, allocations, trade history) |

## Running

```bash
./1_setup_interpreter.sh        # Python 3.12 venv + requirements
source venv/bin/activate

git submodule update --init external/bourso-api   # bourso-cli source (pinned fork)
./2_bourso_cli_update.sh                           # build + install bourso-cli (Rust/cargo)

python -m src.download_ohlcv            # refresh QQQ / VIX / TLT
python -m src.download_macro_data       # refresh FRED (BAA spread, NFCI, CPI YoY)

python -m src.risk_off_strategy.run     # QQQ → outputs/qqq_strategy/{charts,signal.json}
```

`run.py` runs **QQQ only** by default; pass another ticker explicitly (e.g. `SPY`) for
exploration. Outputs land in `outputs/qqq_strategy/`: full / 1-year / 1-month charts and
`signal.json` consumed by the morning PEA script.

### Live automation

Three cron jobs drive production (see [`BOURSO.md`](BOURSO.md)):

```
22:30  src.risk_off_strategy.run QQQ     → signal.json  (+ email recap)                 [weekdays]
09:05  src.real_bourso --execute         → executes PUST allocation on the PEA (live)    [weekdays]
20:00  src.bourso.check_cli              → bourso-cli health check + email alert         [daily]
```

`signal.json` carries `status` (`"running"` → `"ok"`), `allocation` (and `probability` =
`allocation`, kept for backward compat), plus a `macro_off` flag; the morning script
refuses to act on a non-`ok` or stale signal (max age 90h — wide enough to tolerate
weekend/holiday gaps so Monday mornings still execute), and `logs/emergency_off.json`
forces 0% as a kill switch.

PEA execution talks to Boursorama through **`bourso-cli`** (Rust). Upstream
[azerpas/bourso-api](https://github.com/azerpas/bourso-api) exposes the account-reading
function `get_trading_summary` in its library but never wires it to the CLI, so the source
is a **pinned git-submodule fork** (`external/bourso-api`, branch `myqtm`) adding a
read-only **`trade summary`** command (real-time cash/positions via `position=INSTANT`).
The 20:00 check validates the installed binary (pytest dry-run, no rebuild), does a real
`trade summary`, and watches azerpas for new tags — emailing `BUILD CASSE` /
`CONNEXION COMPTE KO` / `nouveau tag` so the fork can be rebased manually with
`./2_bourso_cli_update.sh --pull`.

## Design choices

1. **Trend + vol-managed, not return prediction** — the two robust edges of an equity
   index are trend persistence and volatility clustering; both are exploited directly,
   with no fragile forecast.
2. **React to volatility, don't predict the spark** — a volatility crisis is a self-feeding
   deleveraging loop; vol is autocorrelated, so a 1-day-lagged vol-target captures most of
   the protection.
3. **Longer MA for robustness** — SMA250 over SMA150 trades a little responsiveness for
   fewer whipsaws and a materially better drawdown.
4. **Macro guardrails, assumed overfit** — NFCI / inflation cutoffs are rare, extreme,
   anti-crisis switches, not an alpha source.
5. **Honest close-to-close alignment** (`exec_lag=1`) — the day-`i` signal earns the
   day-`i+1` return, so the backtest carries no 1-day look-ahead.
6. **Free data only** — yfinance + FRED; reproducible with no paid API.
