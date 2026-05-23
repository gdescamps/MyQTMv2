# MyQTMv2 — Quantitative ETF Momentum Strategy

Systematic ETF allocation strategy combining XGBoost cross-sectional IC maximization with a VIX-based regime overlay. Three allocation modes — heuristic, Follow Leads model, Smart Money model — are activated based on market conditions and model validation.

## Performance (OOS backtest 2006-2026)

| Metric | Heuristic + Cash only | Full strategy (SM + FL) |
|--------|----------------------|------------------------|
| **Sharpe** | 1.278 | 1.380 |
| **Ann. return** | +22.1% | +24.7% |
| **Max DD** | -29.5% | -29.5% |
| **Final value** | 9.3M€ | 14.4M€ |

The heuristic baseline is already solid (Sharpe 1.28). The SM and FL models add +2.6%/year and +0.10 Sharpe without degrading drawdown.

### Robustness (50 Monte Carlo runs, 5-10 random ETFs dropped)

| Stat | Sharpe | Ann. return | Max DD |
|------|--------|-------------|--------|
| Original | 1.38 | +24.7% | -29.5% |
| **Median** | **1.31** | **+22.5%** | **-27.4%** |
| Min | 1.19 | +19.9% | — |
| Max | 1.44 | +25.3% | — |

## Strategy: VIX Regime Allocation

The system uses VIX EMA100 to determine market regime and selects the appropriate allocation model:

```
┌──────────────────────┬────────────────────┬────────────────────┬──────────────────────────────────┐
│ VIX EMA100           │ SM Model Validated │ FL Model Validated │ Allocation                       │
├──────────────────────┼────────────────────┼────────────────────┼──────────────────────────────────┤
│ < 19                 │ —                  │ no                 │ Heuristic top-5 Sharpe-wtd 2y    │
│ < 19                 │ —                  │ yes                │ Follow Leads Model if validated  │
│ ≥ 19 (turbulent)     │ yes                │ —                  │ Smart Money Model if validated   │
│ ≥ 20 + rising slope  │ no                 │ —                  │ Stay in cash                     │
│ Spike (Δ5d > 6)      │ —                  │ —                  │ Stay in cash                     │
└──────────────────────┴────────────────────┴────────────────────┴──────────────────────────────────┘
```

- **Heuristic**: top-5 ETFs by rolling 2-year Sharpe ratio, softmax-weighted by Sharpe
- **Follow Leads Model**: XGBoost predictions from the Follow Leads pipeline, activated when causal EMA of past test-IC > 0.015
- **Smart Money Model**: XGBoost predictions using institutional flow features, activated when causal EMA of past test-IC > 0.030
- **Stay in cash**: 0% invested when VIX is high + rising and no validated model, or during VIX spikes

### Model validation (IC gate)

Both models use a causal IC gate: the EMA24 of past test-IC values (from previous walk-forward steps) must exceed a threshold before the model can deploy. This prevents the model from trading when it has no demonstrated alpha. The gate requires at least 84 past steps (~7 years) before it can open, naturally keeping models dormant during early years when the cross-section is too sparse.

| Gate | Threshold | Min steps | Typical activation |
|------|-----------|-----------|-------------------|
| Smart Money | test-IC EMA24 > 0.030 | 84 | ~2018 onward |
| Follow Leads | test-IC EMA24 > 0.015 | 84 | ~2018 onward |

## Algorithm Overview

```
┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ ~250 Features │───▶│ K-fold WF Feature │───▶│ XGBoost ensemble │───▶│  VIX regime overlay: │
│  Engineering  │    │ selection (×10)   │    │ (×20, embargo Δ) │    │ heuristic / FL / SM │
└───────────────┘    └──────────────────┘    └─────────────────┘    │ + cash on spike/VIX │
                                                                     └─────────────────────┘
```

## 1. Universe (27 ETFs, growing over time)

Single universe of 27 ETFs, all with iShares smart-money coverage. Each ETF enters the cross-section as soon as its `shares_outstanding_z20` becomes valid. The walk-forward starts with whichever subset is already active; the cross-section grows as new ETFs come online.

Sections: Geo equity (20), Thematic (4), Commodity (3).

## 2. Data Sources

| Source | Data |
|--------|------|
| yfinance | OHLCV daily prices (via US/LSE proxy tickers for max history) |
| FRED | VIX, HY spread, yield curve, DXY |
| iShares XLS | Shares outstanding (smart-money flows) |

No FMP, no LLM, no paid API.

## 3. Feature Engineering (`feature_engineering.py`)

~250 candidate features computed per ETF per day. Categories: technical (momentum, vol, RSI, MA, ATR, Bollinger, drawdown, skew/kurtosis), macro (VIX/HY/yield curve/DXY plus z-scores/velocity/interactions), smart-money (shares outstanding z-scores + crossovers + momentum), cross-sectional (z-scores vs universe and within section, ranks, momentum × volume/SO/VIX, dispersion, breadth, rank persistence), and expanding stats since inception.

**Label**: `ret_10d_fwd` (forward 10-day return, absolute) — z-scored cross-sectionally per date so that MSE minimization = cross-sectional IC maximization.

## 4. K-fold Walk-Forward Training (`train.py`)

At each WF step we run two K-fold passes — one for feature selection, one for model training — both with **varying final embargo** between train data and the 21-day test window.

### Feature selection — 10 folds

| Parameter | Value |
|-----------|-------|
| Number of folds | 10 |
| Final embargo schedule | 30 → 12 days (step 2) |
| Stability metric | mean(importance) / std(importance)^1.7 |
| Cap | **top 110** features kept per step |

### Model ensemble — 20 models

| Parameter | Value |
|-----------|-------|
| Number of models | 20 |
| Final embargo schedule | 30 → 11 days (step 1) |
| Seeds | 0..19 (different per model) |
| Aggregation | Mean of 20 predictions on test window |

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

## 5. Allocation Layer (`backtest.py`)

### Score → weights pipeline

1. Z-score test predictions cross-sectionally per day
2. Filter to positive scores (score > 0)
3. Top-N filter: keep top 3 by score (TOP_N_ALLOC=3)
4. Softmax with T=1.0 → weights
5. Normalize to sum = 1; 100% of capital deployed unless cash-out is active

### Hysteresis rebalancing

Rebalance only when a currently allocated ETF leaves the monitored top-N (TOP_N_MONITOR=3). `prev_topn_set` is carried between WF steps.

### Transaction costs

- Broker fee: Interactive Brokers fixed (0.05% of trade value, min 3€)
- Spread cost: 0.01% per trade (liquid ETFs)
- Minimum rebalance threshold: 3% weight change

## 6. Robustness (`backtest.py --robustness`)

50 Monte Carlo runs, randomly dropping 5-10 ETFs from the universe at each step (applied in all modes — heuristic, FL, SM).

## 7. Files

| File | Description |
|------|-------------|
| `etf.py` | UNIVERSE (27 ETFs), all with iShares smart-money coverage |
| `feature_engineering.py` | ~250 features + ISHARES_MAP + per-ETF smart-money activation |
| `train.py` | K-fold WF feature selection (×10) + XGBoost ensemble training (×20) |
| `backtest.py` | VIX regime allocation + IC gates + Monte Carlo robustness |

## 8. Running

```bash
source venv/bin/activate

python feature_engineering.py            # data/features.parquet (~1 min)
python train.py                          # data/oos_predictions.parquet (~10 min GPU)
python backtest.py                       # outputs/backtest_equity.jpg
python backtest.py --robustness          # outputs/backtest_robustness.jpg (~15 min)
```

## 9. Key Design Decisions

1. **Single growing universe** — one pipeline, 27 ETFs activated as their smart-money series come online.
2. **Three-tier VIX regime** — heuristic in calm markets, ML models in turbulent markets (when validated), cash during extreme stress.
3. **IC gate (causal EMA24)** — models only deploy when they have demonstrated alpha on past OOS steps. Keeps allocation safe during early years and regime shifts.
4. **K-fold ensembling over varying embargo** — averaging models trained with different train/test cut points cancels both subsampling noise and boundary-sensitivity.
5. **Per-day regime detection** — VIX EMA100 checked daily, not per-step, avoiding ~50-day strategy lag.
6. **VIX slope condition for cash** — cash only when VIX ≥ 20 AND rising, not just high. Allows recovery participation when VIX is elevated but falling.
7. **Heuristic as a real fallback** — when models are not validated, the strategy actively ignores XGB and picks the rolling-Sharpe leaders. This alone delivers Sharpe 1.28.
