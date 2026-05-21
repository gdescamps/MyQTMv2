# MyQTMv2 — Quantitative ETF Momentum Strategy

Systematic ETF allocation strategy combining XGBoost cross-sectional IC maximization with a rule-based regime overlay (calm-market top-3 Sharpe + VIX spike cash-out).

## Performance (backtest OOS 2019-09 → 2026-05, 6.83 years)

| Metric | Value |
|--------|-------|
| Total return (brut) | +1399.9% (×14.87) |
| Ann. return (brut) | **+48.66%** |
| Sharpe | **2.15** |
| Sortino | 2.99 |
| Max Drawdown | **-12.9%** |
| Calmar | 3.76 |
| Robustness (median, 50 runs, drop 5-10 ETF) | Sharpe **2.27**, DD -15.8% |

Per-year (brut): 2020 +62.4%, 2021 +11.6%, 2022 +109.5%, 2023 +50.9%, 2024 +18.2%, 2025 +50.3%.

## Algorithm Overview

```
┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ ~250 Features │───▶│ K-fold WF Feature │───▶│ XGBoost ensemble │───▶│  Per-day allocator: │
│  Engineering  │    │ selection (×10)   │    │ (×20, embargo Δ) │    │ calm Sharpe / model │
└───────────────┘    └──────────────────┘    └─────────────────┘    │ + 3d cash on spike  │
                                                                     └─────────────────────┘
```

## 1. Universes

Two parallel universes, both with iShares shares-outstanding (smart-money) coverage. They share the same pipeline, training methodology, and hyperparameters — only the constituent ETFs differ.

| Universe | ETFs | Earliest data | First test step | Mode-specific dirs |
|----------|------|---------------|------------------|---------------------|
| **Short** (default) | 27 | ~2007 | 2019-09 | `data/short/`, `outputs/short/` |
| **Long** (`--long`) | 17 | ~2000 | 2011-01 | `data/long/`, `outputs/long/` |

The long universe is a max-history subset of ETFs whose OHLCV proxies and iShares smart-money series both start on or before ~2006 — 17 of the short-universe ETFs qualify (the 18th, GLD/SPDR Gold, has no iShares coverage so it is omitted; gold exposure is retained via RING, iShares Gold Miners).

Raw OHLCV / iShares / FRED data lives at `data/` root (shared). Mode-specific products — `features.parquet`, `oos_predictions.parquet`, `backtest_results.parquet`, and all `outputs/` artefacts — live under their respective `data/{short,long}/` and `outputs/{short,long}/` subdirectories.

## 2. Data Sources

| Source | Data |
|--------|------|
| yfinance | OHLCV daily prices (via US/LSE proxy tickers for max history) |
| FRED | VIX, HY spread, yield curve, DXY |
| iShares XLS | Shares outstanding (smart-money flows) |

No FMP, no LLM, no paid API.

## 3. Feature Engineering (`feature_engineering.py`)

~250 candidate features computed per ETF per day. Categories: technical (momentum, vol, RSI, MA, ATR, Bollinger, drawdown, skew/kurtosis), macro (VIX/HY/yield curve/DXY plus z-scores/velocity/interactions), smart-money (shares outstanding z-scores + crossovers + momentum), cross-sectional (z-scores vs universe and within section, ranks, momentum × volume/SO/VIX, dispersion, breadth, rank persistence), and expanding stats since inception.

**Label**: `ret_10d_fwd` (forward 10-day return, absolute) — z-scored cross-sectionally per date so that MSE minimization = cross-sectional IC maximization (each day weighs the same).

## 4. K-fold Walk-Forward Training (`train.py`)

Replaces the prior dual-model A/B scheme. At each WF step we run two K-fold passes — one for feature selection, one for model training — both with **varying final embargo** between train data and the 21-day test window.

### Feature selection — 10 folds

| Parameter | Value |
|-----------|-------|
| Number of folds | 10 |
| Final embargo schedule | 30 → 12 days (step 2) |
| Stability metric | mean(importance) / std(importance)^1.7 |
| Cap | **top 110** features kept per step |

Each fold trains XGBoost (depth=7) on **even blocks** of the rolling 5y window up to `test_start − final_embargo`, validates on odd blocks. Features are ranked by stability across the 10 folds.

### Model ensemble — 20 models

| Parameter | Value |
|-----------|-------|
| Number of models | 20 |
| Final embargo schedule | 30 → 11 days (step 1) |
| Seeds | 0..19 (different per model) |
| Aggregation | Mean of 20 predictions on test window |

Each of the 20 models uses a different final embargo *and* a different seed. The mix of embargo durations covers different cuts of "how recent can train data be" — averaging absorbs both the noise of XGB row/col subsampling and the noise of where exactly the train/test boundary sits.

### Common config

| Parameter | Value |
|-----------|-------|
| Rolling train window | 1250 days (~5 years) |
| Test window | 21 days (~1 month) |
| Step | 21 days |
| Block size | 21 days (interlaced train/val) |
| Embargo | 10 days (>= label horizon) |
| Objective | reg:squarederror (IC maximization via z-scored labels) |
| max_depth | 6 |
| min_child_weight | 40 |
| learning_rate | 0.04 |

Hyperparameters are identical for short and long. The only mode-conditional setting is `MIN_TRAIN_ROWS` (2772 for long vs 5031 for short), which simply tracks how much history each universe makes available before the first walk-forward step.

### Honest val IC reporting

`val_ic` is computed only on **odd blocks AND non-embargoed positions** of the training window. Including even blocks would mix in-sample predictions (used to fit the trees) and inflate val_ic by ~+0.04 IC vs the true 21-day OOS test IC.

Smart-money model gap (val − test): from +0.07 (reported) → +0.02 (honest) — the apparent overfit was largely a K-fold reporting artefact.

## 5. Allocation Layer (`backtest.py`)

A per-day rule-based overlay decides whether each day uses the XGB model or the calm-mode fallback.

### Per-day regime selection (no step-boundary lag)

For each day in the 21-day test window:

```
if VIX EMA100 < CALM_THRESHOLD (19):
    use calm mode  → top-3 by rolling 252d Sharpe, weighted by Sharpe via softmax
else:
    use model     → softmax of XGB scores, top-3 by score
if (VIX 5-day change) > +4 points:
    cash out for 3 trading days
```

Previously the calm/model decision was evaluated once per WF step at `val_scores.index[-1]` (≈ test_start − 30d) and held for the whole step — that introduced up to 50 days of lag. The current per-day evaluation matches the chart's regime coloring exactly.

### Score → weights pipeline

1. Z-score test predictions cross-sectionally per day
2. Filter to positive scores (score > 0)
3. Top-N filter: keep top 3 by score (TOP_N_ALLOC=3)
4. Softmax with T=1.0 → weights
5. **No Sharpe weighting** (SHARPE_POWER=0). Earlier versions multiplied weights by historical Sharpe^p; this step is now disabled — empirically neutral and adds a parameter to maintain.
6. Normalize to sum = 1; 100% of capital deployed unless cash-out is active.

### Hysteresis rebalancing

Rebalance only when a currently allocated ETF leaves the monitored top-N (TOP_N_MONITOR=3). `prev_topn_set` is carried between WF steps — drops the number of rebals per year from ~180 to ~70-80 while improving net returns.

### Calm mode allocation

In calm mode, the model's XGB scores are **ignored**. Replaced by the top-3 ETFs by rolling 252d Sharpe at the step boundary. Weights are computed by softmax of those Sharpe values, so the higher-Sharpe ETF gets a larger weight (this is the only place a Sharpe-based weighting still applies).

### VIX spike cash-out

When the VIX 5-day change exceeds +4 points on day D, allocations are forced to NaN on days D, D+1, D+2 (3 trading days flat). Replaces the prior "progressive cap" logic — discrete, easier to reason about, and effective at trimming the worst single-week drawdowns.

## 6. Robustness (`backtest.py --robustness`)

50 Monte Carlo runs, randomly dropping 5-10 ETFs from the universe at each step (applied in both calm and model modes — so calm mode picks its top-3 Sharpe from a degraded universe too):

| Stat | Sharpe | DD |
|------|--------|-----|
| Original | 2.41 | -12.9% |
| Median | **2.27** | -15.8% |
| Min | 1.93 | — |
| Max | 2.67 | — |

## 7. Files

| File | Description |
|------|-------------|
| `etf.py` | UNIVERSE_SHORT (27 ETFs) + UNIVERSE_LONG (17 ETFs since ~2000), both smart-money |
| `feature_engineering.py` | ~250 features + ISHARES_MAP, identical pipeline for both modes |
| `train.py` | K-fold WF feature selection (×10) + XGBoost ensemble training (×20) |
| `backtest.py` | Per-day allocator (calm/model/spike-cash) + Monte Carlo robustness |
| `knowledge/backtest_long_calmonly.py` | Ablation: long mode with model disabled (calm only) |
| `knowledge/FAILED_XP.md` | Log of experiments tested and rejected (avoids redoing them) |

## 8. Running

```bash
source venv/bin/activate

# Short universe (27 ETFs since ~2007) — default
python feature_engineering.py            # data/short/features.parquet
python train.py                          # data/short/oos_predictions.parquet (~5 min GPU)
python backtest.py                       # outputs/short/backtest_equity.jpg + per-year + pies
python backtest.py --robustness          # outputs/short/backtest_robustness.jpg (~10 min)

# Long universe (17 ETFs since ~2000) — identical pipeline, max-history subset
python feature_engineering.py --long     # data/long/features.parquet
python train.py --long                   # data/long/oos_predictions.parquet
python backtest.py --long                # outputs/long/...
python knowledge/backtest_long_calmonly.py   # outputs/long_calmonly/... (no XGB, calm only)
```

## 9. Key Design Decisions

1. **K-fold ensembling over varying embargo** — averaging models trained with different train/test cut points cancels both subsampling noise and boundary-sensitivity of the rolling window.
2. **Honest val IC** — restricting val to odd & not_embargoed blocks reveals that the smart-money model overfits much less than naive val_ic suggests (gap +0.02 instead of +0.07).
3. **Per-day calm detection** — matches the chart, removes the ~50-day strategy lag, materially boosts 2023 (+50.9% vs mediocre).
4. **No Sharpe weighting at allocation** — empirically neutral, one fewer parameter to tune. Softmax on z-scored scores does the heavy lifting.
5. **3-day cash-out on VIX spike** — discrete and effective; halved max DD vs the prior progressive-cap heuristic.
6. **Calm-mode is a real fallback, not a tweak** — when VIX EMA100 is low, the strategy actively *ignores* the XGB and picks the rolling-Sharpe leaders.

## 10. Historical ablation — Smart-money is load-bearing

This section documents the original ablation that motivated the current setup: on an earlier `--long` configuration where smart-money was **deliberately disabled** (18 ETFs, no shares-outstanding features), the model lost all alpha.

| Variant (historical) | Test IC | CAGR brut | Sharpe | Max DD |
|----------------------|---------|-----------|--------|--------|
| Short (smart-money, 27 ETFs) | **+0.049** | +48.7%/an | **2.15** | -12.9% |
| Long *without* smart-money (18 ETFs) | -0.021 | +15.3%/an | 0.85 | -49.9% |
| Long calm-only (no XGB at all)       | n/a     | +21.7%/an | 1.23 | -24.4% |

Without smart-money the XGB only sees auto-correlated technical features and fails to generalize: mean test IC was **-0.02** (anti-predictive on average) and strictly worse than disabling the model entirely.

**Current state:** smart-money is now enabled on the long universe too (17 ETFs that all have iShares coverage — see §1). Re-running `train.py --long && backtest.py --long` produces fresh long-history numbers; see `outputs/long/` for the new artefacts.

## 11. Rejected experiments

See [`knowledge/FAILED_XP.md`](knowledge/FAILED_XP.md) for the log of experiments tested and abandoned (LightGBM ensemble, three institutional-flow features, residual-label learning). The current architecture appears saturated — adding capacity to the 20-XGB × 110-feature pool no longer moves test IC.
