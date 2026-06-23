# MyQTMv2 — Risk-Off Strategy

A **crisis-avoidance trading system** for QQQ (Nasdaq-100). A walk-forward XGBoost model
learns when to stay invested versus move to cash, turning its probability into a
**continuous allocation** (0% → up to 2x leverage). The goal is not to beat the market on
return alone, but to **capture most of the upside while cutting drawdowns by ~4x**.

The live system runs nightly, writes a signal, and the next morning executes that
allocation on a **Boursorama PEA** account (ETF **PUST**, Amundi PEA Nasdaq-100, x1).
See [`BOURSO.md`](BOURSO.md) for the operational runbook and [`CLAUDE.md`](CLAUDE.md) for the
code map.

## Performance (walk-forward OOS, 2005 → 2026, 21 years)

All figures are **net of Boursorama PEA fees** (0% buy, 0.5% sell) and ETF TER.

| Strategy | CAGR | Total | Max DD |
|----------|-----:|------:|------:|
| QQQ Buy & Hold | 16.1% | 23x | **−53.4%** |
| Oracle (perfect label) | 20.9% | 54x | −9.9% |
| **XGB x1.0 (live config)** | **21.9%** | **63x** | **−7.6%** |
| XGB x1.5 | 33.9% | 462x | −11.4% |
| XGB x2.0 | 46.8% | 3182x | −15.0% |

The headline is the **x1.0 line**: it beats Buy & Hold on return *and* shrinks the worst
drawdown from −53% to −8% — close to the theoretical oracle. Leverage multiplies returns
but is only used in backtest; the live PEA trades x1 (no leveraged PEA-eligible ETF cheap
enough per share). The drawdown stays bounded because the model exits to cash before
crashes deepen, not because it predicts tops.

## How it works

```
┌──────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────┐
│ price + VIX  │──▶│ ~95 features +   │──▶│ walk-forward XGB  │──▶│ proba → alloc    │
│ + BAA spread │   │ drawdown-state   │   │ (per-step refit + │   │ 0→2x, fee-aware  │
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
(`data/<x>_revised.parquet`) are both kept so look-ahead bias from data revisions can be
measured (`compare_pit.py`).

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
saturating at 0/1 — this makes the downstream allocation gradual. Results are cached
incrementally, so a daily run only computes the newest steps.

### 5. Allocation & fees (`simulate_with_fees`)

The probability maps linearly to an allocation, then is clamped and scaled by leverage:

```
allocation = clip( (proba − PROB_CASH) / (PROB_FULL − PROB_CASH), 0, 1 ) × leverage
           = clip( (proba − 0.70) / (0.75 − 0.70), 0, 1 ) × leverage
```

So below proba 0.70 → fully cash; above 0.75 → fully invested; linear in between.
The simulator applies real **Boursorama PEA** economics:

- **Buys are free**; **sells cost 0.5%** → only sell when the allocation drops by
  ≥ 20% (or all the way to cash), avoiding churn.
- ETF TER charged daily (PUST 0.23%/yr; LQQ 0.60%/yr for the levered portion in backtest).

## Repository layout

| Path | Role |
|------|------|
| `src/risk_off_strategy/data.py` | data loading, features, drawdown-state label |
| `src/risk_off_strategy/backtest.py` | walk-forward, fee-aware simulation, plots |
| `src/risk_off_strategy/run.py` | entry point → backtest + charts + `signal.json` |
| `src/risk_off_strategy/compare_pit.py` | point-in-time vs revised equity curves |
| `src/risk_off_strategy/test_lookahead.py` | leakage guard |
| `src/download_ohlcv.py`, `download_macro_data.py` | refresh `data/*.parquet` |
| `src/real_bourso.py`, `src/bourso/` | morning PEA execution + email notifications |
| `src/webapp.py` | NiceGUI dashboard (backtests, allocations, trade history) |

## Running

```bash
./1_setup_interpreter.sh        # Python 3.12 venv + requirements
source venv/bin/activate

python -m src.download_ohlcv            # refresh QQQ / VIX / TLT
python -m src.download_macro_data       # refresh FRED BAA spread

python -m src.risk_off_strategy.run QQQ          # backtest → outputs/qqq_strategy/{charts,signal.json}
python -m src.risk_off_strategy.compare_pit QQQ  # PIT vs revised sanity check
```

`run.py` also accepts `SPY`, `ACWI`, or `ALL`. Outputs land in `outputs/<ticker>_strategy/`:
full / 1-year / 1-month equity charts, the PIT comparison, a forward projection, and
`signal.json` consumed by the morning PEA script.

### Live automation

Two weekday cron jobs drive production (see [`BOURSO.md`](BOURSO.md)):

```
22:30  src.risk_off_strategy.run QQQ     → signal.json  (+ PIT compare + email recap)
09:05  src.real_bourso --execute         → executes PUST allocation on the PEA (live)
```

`signal.json` carries `status` (`"running"` → `"ok"`), the probability, and the target
`allocation`; the morning script refuses to act on a non-`ok` or stale signal (max age
90h — wide enough to tolerate weekend/holiday gaps so Monday mornings still execute), and
`logs/emergency_off.json` forces 0% as a kill switch.

## Design choices

1. **Drawdown-state label, not return prediction** — the model learns to recognize
   dangerous regimes, a far easier and more stable target than forecasting returns.
2. **Lookahead label + embargo** — act before the crash, with a provable no-leakage gap.
3. **Temperature-scaled probability → continuous allocation** — gradual de-risking instead
   of binary on/off whipsaws.
4. **Fee-aware sizing** — the 0.5% PEA sell fee is modeled, so the strategy only trims when
   the move is worth it.
5. **Free data only** — yfinance + FRED; reproducible with no paid API.
