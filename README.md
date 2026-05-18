# MyQTMv2 — Quantitative ETF Momentum Strategy

Systematic ETF allocation strategy based on XGBoost cross-sectional IC maximization with VIX-adaptive risk management.

## Performance (backtest OOS 2019-09 → 2026-05)

| Metric | Value |
|--------|-------|
| Total return | +771% |
| Ann. return | 37.3% |
| Sharpe | 1.56 |
| Max Drawdown | -23.2% |
| Ann. volatility | 21.8% |
| Robustness (median, 50 runs drop 5-10 ETFs) | Sharpe 1.42 |

## Algorithm Overview

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐
│  250 Features │───▶│  Walk-Forward     │───▶│  XGBoost IC     │───▶│  VIX-Adaptive │
│  Engineering  │    │  Feature Select   │    │  Prediction     │    │  Allocation   │
└─────────────┘    └──────────────────┘    └────────────────┘    └──────────────┘
```

## 1. Universe (27 ETFs)

Tradeable on Boursorama (compte titre), all with iShares shares outstanding (smart money) history since ≤2017.

Sections: Geo equity (20), Thematic (4), Commodity (3).

## 2. Data Sources

| Source | Data |
|--------|------|
| yfinance | OHLCV daily prices (via US/LSE proxy tickers for max history) |
| FRED | VIX, HY spread, yield curve, DXY |
| iShares XLS | Shares outstanding (smart money flows) |

No FMP, no LLM, no paid API.

## 3. Feature Engineering (`feature_engineering.py`)

**250 candidate features** computed per ETF per day:

### Per-ETF technical (computed individually)
- **Momentum**: ret 1/5/10/20/40/60/120/250d, momentum ratios (5v20, 20v60, 60v120...)
- **Volatility**: vol 5/10/20/60/120d, vol ratios
- **RSI**: 5/7/14/21/60 periods, RSI crossovers
- **Moving averages**: price vs MA10/20/50/100/200, MA slopes, MA crossovers
- **ATR**: 7/14/21d, ATR ratios
- **Volume**: dollar volume z-scores 5/10/20/60/120d, volume crossovers
- **Bollinger**: bb_position 20/60/120d
- **Drawdown**: from rolling max 20/60/120/250d
- **High/Low position**: 20/60d range
- **Skewness & Kurtosis**: 20/60d
- **Smart money**: shares_outstanding z-scores 5/10/20/60/120d, SO crossovers, SO momentum
- **Expanding stats**: ann return since start, max DD, vol, Sharpe, current DD

### Macro features (same for all ETFs)
- **VIX**: level, velocity 5/20d, z-scores, crossovers, acceleration, squared
- **HY spread**: level, z-scores, velocity, crossovers
- **Yield curve**: level, z-scores, velocity, crossovers
- **DXY**: returns, z-scores, crossovers
- **SPX proxy**: returns, vol, drawdown
- **Interactions**: VIX x HY, yield_curve x VIX

### Cross-sectional features (computed across all ETFs per date)
- **Z-scores vs universe**: ret, vol, RSI, ATR, BB, drawdown
- **Z-scores within section**: ret 5/20/60d
- **Ranks**: ret, vol, SO (percentile)
- **Momentum acceleration**: short minus long, z-scored
- **Squared/cubic z-scores**: non-linear rank extremes
- **Interactions**: momentum x volume, momentum x SO, momentum x drawdown, RSI x momentum
- **Regime-conditional**: momentum x VIX, momentum x yield_curve, momentum x DXY
- **Dispersion**: universe std, breadth (% positive), ETF vs universe mean
- **Sector rotation**: section mean vs universe, ETF vs section
- **Smart money divergence**: SO direction vs price direction
- **Volatility-adjusted momentum**: Sharpe 20/60/120d, cross-sectional z-scores
- **Momentum acceleration (2nd derivative)**: diff of ret over time
- **Rank persistence**: lagged rank, rank change

### Label
- `ret_10d_fwd`: forward 10-day return (absolute, no demeaning)

## 4. Walk-Forward Feature Selection (`train.py`)

At each training step, features are re-selected using only data available at that point (**no lookahead**).

| Parameter | Value |
|-----------|-------|
| Method | 3-period interlaced stability |
| Metric | mean_importance / std^1.7 |
| Cap | 150 features per step |
| Recompute | every 5 steps |

Process:
1. Split train window into 3 interlaced groups (blocks of 21 days)
2. Train XGBoost (depth=7) on 2 groups, validate on 3rd — repeat x3
3. For each feature: stability = mean(importance) / std(importance)^1.7
4. Keep top 150 features by stability

## 5. XGBoost Training (`train.py`)

Walk-forward expanding window with rolling cap.

| Parameter | Value |
|-----------|-------|
| Rolling window | 1250 days (~5 years) |
| Test window | 21 days (~1 month) |
| Step | 21 days |
| Block size | 21 days (interlaced train/val) |
| Embargo | 10 days (>= label horizon) |
| Objective | reg:squarederror (IC maximization via z-scored labels) |
| max_depth | 6 |
| min_child_weight | 40 |
| learning_rate | 0.04 |
| subsample | 0.90 |
| colsample_bytree | 0.678 |
| early_stopping | 30 rounds (RMSE on val) |

At each step:
1. Walk-forward feature selection on train data (every 5 steps)
2. Split train into even/odd blocks (21d each, 10d embargo)
3. Train XGBoost on even blocks, early-stop on odd blocks
4. Predict on next 21 test days
5. Output: cross-sectional score per ETF per day

**IC maximization**: labels are z-scored cross-sectionally per date before training. Minimizing MSE(y_hat, zscore(y)) = maximizing equal-weighted cross-sectional IC.

## 6. Allocation (`backtest.py`)

### Score to Weights
1. Z-score test predictions cross-sectionally per day
2. Filter: keep only score > 0 (positive expected return)
3. Top-N: keep only best N scores (VIX-adaptive, see below)
4. Softmax: weights = softmax(score / T=1.0)
5. Sharpe-weight: weights x historical_sharpe^0.8
6. Normalize to sum = 1 (x max_alloc if spike)

### VIX-Adaptive Regime (inverse logic)

| VIX Level | Top-N ETFs | Rebal frequency |
|-----------|-----------|-----------------|
| < 15 (calm) | 7 (diversify) | every 5 days |
| 15-25 | interpolated | interpolated |
| > 25 (crisis) | 3 (concentrate) | every 2 days |

**Rationale**: In calm markets, signal is weaker so diversify. In crisis, few ETFs survive so concentrate on best-ranked.

### VIX Spike Detection (progressive capital cap)

When VIX slope (5-day change) rises steeply, reduce capital exposure progressively:

| VIX 5d change | Allocation cap | Rebal |
|---------------|----------------|-------|
| < +4 pts | 100% (fully invested) | normal |
| +4 pts | 80% invested, 20% cash | daily |
| +7 pts | 60% invested, 40% cash | daily |
| +10 pts | 40% invested, 60% cash | daily |

Linear interpolation between +4 and +10. Protects against sudden crashes while staying invested during gradual moves.

## 7. Robustness

Tested with 50 Monte Carlo runs, randomly dropping 5-10 ETFs (18-37% of universe) at each rebalance:

| Stat | Sharpe | Max DD |
|------|--------|--------|
| Original | 1.56 | -23.2% |
| Median | 1.42 | -23.3% |
| Min | 1.21 | — |
| Max | 1.60 | — |

## 8. Files

| File | Description |
|------|-------------|
| `etf.py` | Universe definition (27 ETFs) |
| `feature_engineering.py` | 250 features + ISHARES_MAP |
| `train.py` | Walk-forward XGBoost + WF feature selection |
| `select_features.py` | Standalone feature selection (3-period stability) |
| `backtest.py` | Portfolio simulation + VIX-adaptive allocation |
| `backtest_robustness.py` | Monte Carlo robustness test |
| `backtest_all.py` | Runs `backtest.py` + `backtest_robustness.py` in parallel |
| `search_features_xgb.py` | Grid search (power x cap x depth) |
| `download_ishares_xls.py` | iShares XLS downloader (smart money) |

## 9. Running

```bash
# Setup
source venv/bin/activate

# Full pipeline
python feature_engineering.py   # Build 250 features -> data/features.parquet
python train.py                 # Walk-forward training -> data/oos_predictions.parquet
python backtest_all.py          # Equity + robustness backtests in parallel
                                #   -> outputs/backtest_equity.jpg
                                #   -> outputs/backtest_robustness.jpg

# Hyperparameter search
python search_features_xgb.py  # Grid search power x cap x depth

# Individual backtests (also runnable on their own)
python backtest.py              # Portfolio backtest -> outputs/backtest_equity.jpg
python backtest_robustness.py   # 50 Monte Carlo runs -> outputs/backtest_robustness.jpg

# Feature selection (standalone)
python select_features.py       # 3-period stability -> outputs/selected_features.json
```

## 10. Key Design Decisions

1. **Walk-forward feature selection**: eliminates lookahead bias in feature choice. Features are re-selected every 5 steps using only past data.
2. **IC maximization** (not classification): z-scored labels make every date equally important regardless of return variance.
3. **Inverse VIX logic**: concentrate in crisis (few survivors), diversify in calm (weak signal).
4. **Progressive spike cap**: not binary — smooth reduction of exposure as VIX accelerates.
5. **No data leakage**: embargo (10d) >= label horizon (10d), feature selection uses only train data, no future information.
6. **Regularization**: depth=6, mcw=40, early stopping — prevents overfitting on 27-ETF universe.
