# MyQTMv2 — Risk-Off Strategy

A **crisis-avoidance trading system** for QQQ (Nasdaq-100). A walk-forward XGBoost model
learns when to stay invested versus move to cash, turning its probability into a
**continuous allocation** (0% → 100%, x1). The goal is not to beat the market on return
alone, but to **keep most of the upside while cutting drawdowns**.

The live system runs nightly, writes a signal, and the next morning executes that
allocation on a **Boursorama PEA** account (ETF **PUST**, Amundi PEA Nasdaq-100, x1).
See [`BOURSO.md`](BOURSO.md) for the operational runbook and [`CLAUDE.md`](CLAUDE.md) for the
code map.

## Backtest (walk-forward OOS)

The backtest is **QQQ close-to-close, x1, without fees**. Execution alignment is honest
(`exec_lag=1`): the probability at day `i` is computed ~5 min before the US close[i] and
the order is sent then, so the position only earns the **next** return (close[i]→close[i+1]).
There is **no 1-day look-ahead** (an earlier version paired prob[i]×ret[i], which inflated
the CAGR several-fold).

> Regenerate the figures with `python -m src.risk_off_strategy.run QQQ` — the chart
> (`outputs/qqq_strategy/backtest.png`) prints QQQ Buy & Hold, the perfect-label Oracle,
> and the XGB strategy (CAGR / total / max drawdown). The previous headline table is
> removed because it embedded leverage, fees, and the look-ahead alignment.

## How it works

```
┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│ price + VIX  │──▶│ ~95 features +   │──▶│ walk-forward XGB  │──▶│ proba → alloc    │
│ + BAA spread │   │ drawdown-state   │   │ (per-step refit + │   │ 0→100% (x1)      │
│ + TLT        │   │ machine label    │   │ feature selection)│   │ PEA execution    │
└──────────────┘   └──────────────────┘   └───────────────────┘   └──────────────────┘
```

### 1. Data (`src/risk_off_strategy/data.py`)

Four daily series, all free:

| Series | Source | Role |
|--------|--------|------|
| QQQ close | yfinance | the asset traded |
| VIX | yfinance | implied volatility / fear gauge |
| BAA credit spread | FRED | credit stress (shifted **J+3** for publication lag) |
| TLT | yfinance | long Treasuries — flight-to-safety signal |

Point-in-time snapshots (`data/<x>.parquet`) and latest-revised series
(`data/<x>_revised.parquet`) are both kept as a record of data revisions.

### 2. Label — drawdown state machine (`build_realtime_target`)

The target is **not** a future return. It is a hysteresis state machine over price:

- **Exit to cash** when drawdown from the running peak crosses **−10%** (`DD_EXIT`).
- **Re-enter** when drawdown recovers above **−5%** (`DD_REENTER`).

This produces a clean in/out label that marks the dangerous regimes. The label at day *t*
is the machine's state at *t + LOOKAHEAD* (`LOOKAHEAD=6`), so the model learns to act
*before* the drawdown fully develops. The walk-forward embargo (21 days) is larger than
the lookahead, so there is **no leakage** — verified by `test_lookahead.py`.

### 3. Features (`build_features`)

~95 features per day, technical + macro + cross-asset:

- **Price**: SMA / returns / vol over {5,10,20,50,100,200}, RSI, Bollinger position,
  drawdown depth/speed/duration, rolling max-DD, consecutive down days, mean-reversion.
- **Non-linear risk**: asymmetric (down vs up) volatility, skew/kurtosis, acceleration,
  volatility-of-volatility, vol regime.
- **Macro**: VIX and BAA spread levels, SMAs, term-structure proxies, spikes, acceleration.
- **Cross-asset**: price/TLT ratio + momentum (risk-on vs risk-off), and interaction terms
  (`vix × spread`, `vix × drawdown`, `tlt × vix`, …) that fire only in joint stress.

### 4. Walk-forward XGBoost (`walk_forward`)

Expanding-window, retrained every 21 trading days:

| Parameter | Value |
|-----------|-------|
| Min train | 504 days (~2 years) |
| Step / test window | 21 days |
| Embargo (train→test gap) | 21 days (> label lookahead) |
| Per-step feature selection | stable importance `mean / std^1.7`, train-only |
| Model | XGBoost, `max_depth=4`, `lr=0.03`, subsample/colsample 0.7, strong L1/L2 |

Each step selects features on the **training slice only** (interlaced blocks, embargoed),
refits, and predicts the next 21 days out-of-sample. Raw logits are passed through a
**temperature-scaled sigmoid** (`T=3.0`) so the probability is smooth rather than
saturating at 0/1 — this makes the downstream allocation gradual. The walk-forward is
recomputed from scratch every run (no cache).

### 5. Allocation (`simulate`)

The probability maps linearly to an allocation, clamped to [0, 1] (x1, no leverage):

```
allocation = clip( (proba − PROB_CASH) / (PROB_FULL − PROB_CASH), 0, 1 )
           = clip( (proba − 0.70) / (0.75 − 0.70), 0, 1 )
```

So below proba 0.70 → fully cash; above 0.75 → fully invested; linear in between.

The simulator is **close-to-close and fee-free**: `equity = cumprod(1 + ret · alloc)`,
with `exec_lag=1` (the day-`i` signal earns the day-`i+1` return — no look-ahead). No
transaction fees and no ETF TER are modeled.

## Repository layout

| Path | Role |
|------|------|
| `src/risk_off_strategy/data.py` | data loading, features, drawdown-state label |
| `src/risk_off_strategy/backtest.py` | walk-forward, QQQ close-to-close simulation (x1, no fees), plots |
| `src/risk_off_strategy/run.py` | entry point → backtest + charts + `signal.json` |
| `src/risk_off_strategy/test_lookahead.py` | lookahead=0 vs 6 diagnostic |
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
python -m src.download_macro_data       # refresh FRED BAA spread

python -m src.risk_off_strategy.run QQQ          # backtest → outputs/qqq_strategy/{charts,signal.json}
```

`run.py` also accepts `SPY`, `ACWI`, or `ALL`. Outputs land in `outputs/<ticker>_strategy/`:
full / 1-year / 1-month equity charts and `signal.json` consumed by the morning PEA script.

### Live automation

Three cron jobs drive production (see [`BOURSO.md`](BOURSO.md)):

```
22:30  src.risk_off_strategy.run QQQ     → signal.json  (+ email recap)                 [weekdays]
09:05  src.real_bourso --execute         → executes PUST allocation on the PEA (live)    [weekdays]
20:00  src.bourso.check_cli              → bourso-cli health check + email alert         [daily]
```

`signal.json` carries `status` (`"running"` → `"ok"`), the probability, and the target
`allocation`; the morning script refuses to act on a non-`ok` or stale signal (max age
90h — wide enough to tolerate weekend/holiday gaps so Monday mornings still execute), and
`logs/emergency_off.json` forces 0% as a kill switch.

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

1. **Drawdown-state label, not return prediction** — the model learns to recognize
   dangerous regimes, a far easier and more stable target than forecasting returns.
2. **Lookahead label + embargo** — act before the crash, with a provable no-leakage gap.
3. **Temperature-scaled probability → continuous allocation** — gradual de-risking instead
   of binary on/off whipsaws.
4. **Honest close-to-close alignment** (`exec_lag=1`) — the day-`i` signal earns the
   day-`i+1` return, so the backtest carries no 1-day look-ahead.
5. **Free data only** — yfinance + FRED; reproducible with no paid API.
