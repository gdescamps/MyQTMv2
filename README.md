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

## 1. Universe (27 ETFs)

Tradeable on Boursorama (compte titre), all with iShares shares outstanding (smart-money) history since ≤2017. Sections: Geo equity (20), Thematic (4), Commodity (3).

A second universe (`--long`, 18 ETFs since ~2005, no smart-money) exists for stress-testing — see §10.

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
| max_depth | 6 (short) / 4 (long) |
| min_child_weight | 40 |
| learning_rate | 0.04 |

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
| `etf.py` | UNIVERSE_SHORT (27 ETFs, smart-money) + UNIVERSE_LONG (18 ETFs since 2005) |
| `feature_engineering.py` | ~250 features + ISHARES_MAP |
| `train.py` | K-fold WF feature selection (×10) + XGBoost ensemble training (×20) |
| `backtest.py` | Per-day allocator (calm/model/spike-cash) + Monte Carlo robustness |
| `myfiles/backtest_long_calmonly.py` | Ablation: long mode but with model disabled (calm only) |

## 8. Running

```bash
source venv/bin/activate

# Default pipeline (27-ETF smart-money universe)
python feature_engineering.py   # data/features.parquet
python train.py                 # data/oos_predictions.parquet (~5 min on GPU)
python backtest.py              # outputs/backtest_equity.jpg + per-year + pies
python backtest.py --robustness # outputs/backtest_robustness.jpg (~10 min)

# Long-history universe (no smart-money — see §10)
python feature_engineering.py --long
python train.py --long          # data/oos_predictions_long.parquet
python backtest.py --long       # outputs/long/...
python myfiles/backtest_long_calmonly.py   # outputs/long_calmonly/... (no XGB, calm only)
```

## 9. Key Design Decisions

1. **K-fold ensembling over varying embargo** — averaging models trained with different train/test cut points cancels both subsampling noise and boundary-sensitivity of the rolling window.
2. **Honest val IC** — restricting val to odd & not_embargoed blocks reveals that the smart-money model overfits much less than naive val_ic suggests (gap +0.02 instead of +0.07).
3. **Per-day calm detection** — matches the chart, removes the ~50-day strategy lag, materially boosts 2023 (+50.9% vs mediocre).
4. **No Sharpe weighting at allocation** — empirically neutral, one fewer parameter to tune. Softmax on z-scored scores does the heavy lifting.
5. **3-day cash-out on VIX spike** — discrete and effective; halved max DD vs the prior progressive-cap heuristic.
6. **Calm-mode is a real fallback, not a tweak** — when VIX EMA100 is low, the strategy actively *ignores* the XGB and picks the rolling-Sharpe leaders.

## 10. Long-history ablation — Smart-money is load-bearing

The `--long` universe (18 ETFs with data ≥ 2005, no smart-money features) was built to stress-test the model on a 15-year span (2011-2026). Three configurations:

| Variant | Test IC | CAGR brut | Sharpe | Max DD |
|---------|---------|-----------|--------|--------|
| Short (smart-money, 27 ETFs) | **+0.049** | +48.7%/an | **2.15** | -12.9% |
| Long with XGB (no smart-money, 18 ETFs) | -0.021 | +15.3%/an | 0.85 | -49.9% |
| Long calm-only (no XGB at all) | n/a | +21.7%/an | 1.23 | -24.4% |

**The model has no alpha without smart-money.** On the long universe:
- Mean test IC = **-0.02** (anti-predictive on average)
- The val→test gap is +0.10 even after honest reporting → genuine temporal drift, not a measurement artefact
- Strictly worse than disabling the model entirely (calm-only beats it by Sharpe +0.38, CAGR +6.4%/an)

Conclusion: the load-bearing signal is **iShares shares-outstanding flows** (smart-money). Without them the XGB only sees auto-correlated technical features and fails to generalize beyond the immediate training period. Any future work extending the universe should prioritize ETFs with iShares XLS coverage — that's the difference between Sharpe 2.15 and Sharpe 0.85.

## 11. Rejected experiments (saturation evidence)

These were tried and discarded — left here so they don't get re-attempted.

### LightGBM ensemble alongside XGBoost
Adding 20 LGB models with the same hyper-params, same K-fold embargo/seed pairing as the existing 20 XGB models:
| | XGB only | XGB + LGB | Δ |
|---|---|---|---|
| Test IC | +0.0500 | +0.0507 | +0.0007 |
| Val − Test gap | +0.0291 | +0.0291 | 0 |
| CAGR brut | +48.7%/an | +46.2%/an | **−2.4%** |
| Sharpe | 2.15 | 2.08 | −0.07 |
| Training time | 5m02s | 5m54s | +17% |

The +0.078 IC gain reported in the earlier 2-model era no longer applies — with 20 XGB models already averaging out subsampling noise, the marginal diversification value of LGB is exhausted. The slight test-IC bump (+0.0007) doesn't translate to backtest equity. Would only be worth revisiting if LGB hyper-params were deliberately differentiated (different depth/lr/leaves) to add orthogonality.

### Three new institutional-flow features (`inst_share_z*`, `rotation_idx_universe`, `flow_price_div_*d`)

Designed to inject orthogonal smart-money signal beyond the existing `shares_outstanding_z*` block. Baseline test IC = +0.0490.

| Variant | Test IC | Δ baseline | Best feature SHAP / kept in N steps |
|---|---|---|---|
| +F1 `inst_share_z20/z60` | +0.0496 | +0.0006 | 0.305 / 65 of 82 |
| +F1+F2 `rotation_idx_universe` | +0.0442 | **−0.0048** | F2 raw: 0.013 / 6 of 82 (filtered) |
| +F1+F2+F3 `flow_price_div_60d` | +0.0453 | −0.0037 | F3: **0.927** / 65 of 82 |
| +F1+F3 (F2 dropped) | +0.0483 | −0.0007 | F3: 0.881 / 66 of 82 |

The third feature `flow_price_div_60d` has one of the highest SHAP values in the whole model (0.927, kept in 79% of steps) **yet contributes nothing to test IC**. The current 20-XGB × 110-feature ensemble is at its Pareto frontier — adding a new informative feature *displaces* an equally informative one from the top-110 K-fold filter, netting to zero or worse. Same dynamic that killed the LGB experiment above.

F2 specifically backfired (−0.0048) because `rotation_idx_universe` is **cross-sectionally constant** (same value for all ETFs on a given date). The XGB optimises cross-sectional IC per date, so a constant cannot rank ETFs; it only contributes via interactions, costs tree depth, and occupies a slot in the top-110 that would have gone to a real ranking feature.

**Take-away**: the current architecture is signal-saturated. Adding features (or models) to the top-N pool no longer improves OOS performance. Future work that aims to genuinely move test IC must either (a) raise `FEAT_SEL_CAP` above 110, (b) prune a redundant block to free slots for new signal, or (c) change the architecture (different label, stacking, regime-conditional models). Otherwise it's wasted compute.
