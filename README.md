# MyQTMv2 — Quantitative ETF Momentum Strategy

Systematic ETF allocation strategy combining XGBoost cross-sectional IC maximization with a VIX-based regime overlay. Three allocation modes — heuristic, Follow Leads model, Smart Money model — are activated based on market conditions and model validation.

## Performance (OOS backtest 2006-2026)

| Metric | Heuristic + Cash only | Full strategy (SM + FL) |
|--------|----------------------|------------------------|
| **Sharpe** | 1.278 | 1.385 |
| **Ann. return** | +22.1% | +24.0% |
| **Max DD** | -29.5% | -29.6% |
| **Final value** | 9.3M€ | 12.7M€ |

The heuristic baseline is already solid (Sharpe 1.28). The SM and FL models add +2%/year and +0.10 Sharpe without degrading drawdown.

### Robustness (50 Monte Carlo runs, 5-10 random ETFs dropped)

| Stat | Sharpe | Ann. return | Max DD |
|------|--------|-------------|--------|
| Original | 1.38 | +24.0% | -29.6% |
| **Median** | **1.32** | **+22.0%** | **-27.6%** |
| Min | 1.16 | +17.9% | — |
| Max | 1.45 | +24.5% | — |

## Strategy: VIX Regime Allocation

The system uses VIX EMA100 to determine market regime and selects the appropriate allocation model:

```
┌──────────────────────┬────────────────────┬────────────────────┬──────────────────────────────────┐
│ VIX EMA100           │ SM Model Validated │ FL Model Validated │ Allocation                       │
├──────────────────────┼────────────────────┼────────────────────┼──────────────────────────────────┤
│ —                    │ —                  │ no                 │ Heuristic top-5 Sharpe-wtd 2y    │
│ < 19                 │ —                  │ yes                │ Follow Leads Model               │
│ >= 19 (turbulent)    │ yes                │ —                  │ Smart Money Model                │
│ >= 20 + rising slope │ no                 │ —                  │ Cash                             │
│ Spike (d5d > 6)      │ —                  │ —                  │ Cash                             │
└──────────────────────┴────────────────────┴────────────────────┴──────────────────────────────────┘
```

- **Heuristic**: default allocator — top-5 ETFs by rolling 2-year Sharpe ratio, softmax-weighted. Active in calm markets and as fallback when VIX >= 19 but not in cash conditions.
- **Follow Leads Model**: XGBoost predictions from the Follow Leads pipeline, activated when causal EMA of past test-IC > 0.015
- **Smart Money Model**: XGBoost predictions using institutional flow features, activated when causal EMA of past test-IC > 0.030
- **Cash**: 0% invested when VIX is high + rising and no validated model, or during VIX spikes

### Model validation (IC gate)

Both models use a causal IC gate: the EMA24 of past test-IC values (from previous walk-forward steps) must exceed a threshold before the model can deploy. This prevents the model from trading when it has no demonstrated alpha. The gate requires at least 84 past steps (~7 years) before it can open, naturally keeping models dormant during early years when the cross-section is too sparse.

| Gate | Threshold | Min steps | Typical activation |
|------|-----------|-----------|-------------------|
| Smart Money | test-IC EMA24 > 0.030 | 84 | ~2018 onward |
| Follow Leads | test-IC EMA24 > 0.015 | 84 | ~2018 onward |

## Algorithm Overview

```
┌───────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────────┐
│ ~250 Features │───>│ K-fold WF Feature │───>│ XGBoost ensemble │───>│  VIX regime overlay: │
│  Engineering  │    │ selection (x10)   │    │ (x20, embargo d) │    │ heuristic / FL / SM │
└───────────────┘    └──────────────────┘    └─────────────────┘    │ + cash on spike/VIX │
                                                                     └─────────────────────┘
```

## 1. Universe (26 ETFs)

### Backtest universe (US proxy tickers)

US-listed proxy tickers are used for backtesting and training — they provide deeper historical data (shares outstanding since 2000+). The backtest uses these tickers for OHLCV and smart-money features.

| # | Proxy | Name | Section | Smart money since |
|---|-------|------|---------|-------------------|
| 1 | IVV | S&P 500 | geo | 2000 |
| 2 | QQQ | Nasdaq 100 | geo | 1999 |
| 3 | ACWI | MSCI World | geo | 2008 |
| 4 | EEM | Emerging Markets | geo | 2003 |
| 5 | IEMG | Core EM IMI | geo | 2012 |
| 6 | EMXC | EM ex-China | geo | 2017 |
| 7 | ILF | Latin America 40 | geo | 2001 |
| 8 | EWY | South Korea | geo | 2000 |
| 9 | EWT | Taiwan | geo | 2000 |
| 10 | EWZ | Brazil | geo | 2000 |
| 11 | EWW | Mexico | geo | 2000 |
| 12 | EWC | Canada | geo | 2000 |
| 13 | EWJ | Japan | geo | 1996 |
| 14 | TUR | Turkey | geo | 2008 |
| 15 | FXI | China Large-Cap | geo | 2004 |
| 16 | ISF.L | FTSE 100 | geo | ~2000 |
| 17 | IEUR | Core Europe | geo | 2014 |
| 18 | EZU | Eurozone | geo | 2000 |
| 19 | SUSA | USA SRI | geo | 2005 |
| 20 | SOXX | Semiconductors | thematic | 2001 |
| 21 | ROBO | Automation & Robotics | thematic | 2013 |
| 22 | ICLN | Global Clean Energy | thematic | 2008 |
| 23 | EXX1.DE | EURO STOXX Banks | thematic | ~2001 |
| 24 | RING | Gold Miners | commodity | 2012 |
| 25 | IEO | Oil & Gas E&P | commodity | 2006 |
| 26 | SXRS.DE | Diversified Commodity | commodity | 2006 |

### Live trading universe (UCITS EUR on Interactive Brokers)

For live trading, each proxy maps to a European UCITS ETF denominated in EUR, tradable on Interactive Brokers. All are iShares except 2 (Amundi and Xtrackers) which track the **exact same index** as their iShares proxy — no performance drift.

| # | Proxy | UCITS ticker | Exchange | Issuer | IB fee model |
|---|-------|-------------|----------|--------|-------------|
| 1 | IVV | SXR8 | XETRA | iShares | 0.10%, min 4€ |
| 2 | QQQ | SXRV | XETRA | iShares | 0.10%, min 4€ |
| 3 | ACWI | IUSQ | Amsterdam | iShares | 0.05%, min 4€ |
| 4 | EEM | IEMA | Amsterdam | iShares | 0.05%, min 4€ |
| 5 | IEMG | IEMA | Amsterdam | iShares | 0.05%, min 4€ |
| 6 | EMXC | EMXC | Euronext Paris | **Amundi** | 0.05%, min 3€ |
| 7 | ILF | LTAM | Amsterdam | iShares | 0.05%, min 4€ |
| 8 | EWY | IKRA | Amsterdam | iShares | 0.05%, min 4€ |
| 9 | EWT | ITWN | Amsterdam | iShares | 0.05%, min 4€ |
| 10 | EWZ | IBZL | Amsterdam | iShares | 0.05%, min 4€ |
| 11 | EWW | D5BI | XETRA | **Xtrackers** | 0.10%, min 4€ |
| 12 | EWC | SXR2 | XETRA | iShares | 0.10%, min 4€ |
| 13 | EWJ | SJPE | Amsterdam | iShares | 0.05%, min 4€ |
| 14 | TUR | ITKY | Amsterdam | iShares | 0.05%, min 4€ |
| 15 | FXI | FXC | Amsterdam | iShares | 0.05%, min 4€ |
| 16 | ISF.L | ISF | LSE | iShares | 6 GBP flat |
| 17 | IEUR | IMEU | Amsterdam | iShares | 0.05%, min 4€ |
| 18 | EZU | IMEU | Amsterdam | iShares | 0.05%, min 4€ |
| 19 | SUSA | 36B6 | XETRA | iShares | 0.10%, min 4€ |
| 20 | SOXX | ISQ5 | GETTEX | iShares | 0.10%, min 4€ |
| 21 | ROBO | RBOT | Amsterdam | iShares | 0.05%, min 4€ |
| 22 | ICLN | INRG | Milan | iShares | 0.05%, min 4€ |
| 23 | EXX1.DE | EXX1 | XETRA | iShares | 0.10%, min 4€ |
| 24 | RING | IS0E | XETRA | iShares | 0.10%, min 4€ |
| 25 | IEO | IS0D | XETRA | iShares | 0.10%, min 4€ |
| 26 | SXRS.DE | SXRS | XETRA | iShares | 0.10%, min 4€ |

**Non-iShares ETFs**: EMXC (Amundi) and EWW (Xtrackers) track the same MSCI indices as their iShares US proxies. No difference in price behavior — they replicate the exact same basket of stocks.

**Note**: IEMG and EEM both map to IEMA; IEUR and EZU both map to IMEU. These are distinct proxy tickers with different smart-money signals but trade the same UCITS product.

## 2. Data Sources

| Source | Data |
|--------|------|
| yfinance | OHLCV daily prices (via US/LSE proxy tickers for max history) |
| FRED | VIX, HY spread, yield curve, DXY |
| iShares XLS | Shares outstanding (smart-money flows) |

No FMP, no LLM, no paid API.

## 3. Feature Engineering (`feature_engineering.py`)

~250 candidate features computed per ETF per day. Categories: technical (momentum, vol, RSI, MA, ATR, Bollinger, drawdown, skew/kurtosis), macro (VIX/HY/yield curve/DXY plus z-scores/velocity/interactions), smart-money (shares outstanding z-scores + crossovers + momentum), cross-sectional (z-scores vs universe and within section, ranks, momentum x volume/SO/VIX, dispersion, breadth, rank persistence), and expanding stats since inception.

**Label**: `ret_10d_fwd` (forward 10-day return, absolute) — z-scored cross-sectionally per date so that MSE minimization = cross-sectional IC maximization.

## 4. K-fold Walk-Forward Training (`train.py`)

At each WF step we run two K-fold passes — one for feature selection, one for model training — both with **varying final embargo** between train data and the 21-day test window.

### Feature selection — 10 folds

| Parameter | Value |
|-----------|-------|
| Number of folds | 10 |
| Final embargo schedule | 30 -> 12 days (step 2) |
| Stability metric | mean(importance) / std(importance)^1.7 |
| Cap | **top 110** features kept per step |

### Model ensemble — 20 models

| Parameter | Value |
|-----------|-------|
| Number of models | 20 |
| Final embargo schedule | 30 -> 11 days (step 1) |
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

### Score -> weights pipeline

1. Z-score test predictions cross-sectionally per day
2. Filter to positive scores (score > 0)
3. Top-N filter: keep top 3 by score (TOP_N_ALLOC=3)
4. Softmax with T=1.0 -> weights
5. Normalize to sum = 1; 100% of capital deployed unless cash-out is active

### Hysteresis rebalancing

Rebalance only when a currently allocated ETF leaves the monitored top-N (TOP_N_MONITOR=3). `prev_topn_set` is carried between WF steps.

### Transaction costs (IB Fixed pricing)

Per-ETF fees based on the actual IB exchange used for live trading:

| Exchange | Fee | Min | ETFs |
|----------|-----|-----|------|
| XETRA (IBIS/IBIS2/GETTEX) | 0.10% | 4€ | SXR8, SXRV, SXR2, D5BI, 36B6, ISQ5, EXX1, IS0E, IS0D, SXRS |
| Amsterdam (AEB) | 0.05% | 4€ | IUSQ, IEMA, LTAM, IKRA, ITWN, IBZL, SJPE, ITKY, FXC, IMEU, RBOT |
| Euronext Paris (SBF) | 0.05% | 3€ | EMXC (Amundi) |
| Milan (BVME.ETF) | 0.05% | 4€ | INRG |
| LSE (LSEETF) | 6 GBP flat | — | ISF |
| Spread cost | 0.01% per trade | — | all |
| Rebalance threshold | 3% weight change | — | all |

## 6. Robustness (`backtest.py --robustness`)

50 Monte Carlo runs, randomly dropping 5-10 ETFs from the universe at each step (applied in all modes — heuristic, FL, SM).

## 7. Live Trading (`robot.py`)

Heuristic-only strategy executed via Interactive Brokers Gateway (Docker). Trades the UCITS EUR equivalents.

```bash
python robot.py --allocation          # show target allocation
python robot.py --data                # refresh OHLCV + VIX
python robot.py --dry-run             # connect to IB, show trade plan
python robot.py --trade               # execute trades
python robot.py --stop                # stop IB Gateway docker
```

## 8. Files

| File | Description |
|------|-------------|
| `etf.py` | UNIVERSE (26 ETFs) + TRADING_MAP (UCITS EUR) + IB fee schedule |
| `feature_engineering.py` | ~250 features + ISHARES_MAP + per-ETF smart-money activation |
| `train.py` | K-fold WF feature selection (x10) + XGBoost ensemble training (x20) |
| `backtest.py` | VIX regime allocation + IC gates + Monte Carlo robustness |
| `robot.py` | Live trading robot (heuristic strategy via IB Gateway) |
| `docker-compose.yml` | IB Gateway container config |

## 9. Running

```bash
source venv/bin/activate

python feature_engineering.py            # data/features.parquet (~1 min)
python train.py                          # data/oos_predictions.parquet (~10 min GPU)
python backtest.py                       # outputs/backtest_equity.jpg
python backtest.py --robustness          # outputs/backtest_robustness.jpg (~15 min)
```

## 10. Key Design Decisions

1. **Single growing universe** — one pipeline, 26 ETFs activated as their smart-money series come online.
2. **Three-tier VIX regime** — heuristic in calm markets, ML models in turbulent markets (when validated), cash during extreme stress.
3. **IC gate (causal EMA24)** — models only deploy when they have demonstrated alpha on past OOS steps. Keeps allocation safe during early years and regime shifts.
4. **K-fold ensembling over varying embargo** — averaging models trained with different train/test cut points cancels both subsampling noise and boundary-sensitivity.
5. **Per-day regime detection** — VIX EMA100 checked daily, not per-step, avoiding ~50-day strategy lag.
6. **VIX slope condition for cash** — cash only when VIX >= 20 AND rising, not just high. Allows recovery participation when VIX is elevated but falling.
7. **Heuristic as a real fallback** — when models are not validated, the strategy actively ignores XGB and picks the rolling-Sharpe leaders. This alone delivers Sharpe 1.28.
8. **Realistic IB fees** — per-exchange commission schedule matching Interactive Brokers Fixed pricing for UCITS EUR ETFs.
