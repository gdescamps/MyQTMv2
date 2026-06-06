# QQQ Crisis-Avoidance Strategy

## Objectif

Stratégie de timing sur QQQ (Nasdaq-100 ETF) qui détecte les périodes de crise pour réduire l'exposition, et utilise du levier modéré (jusqu'à 200%) en période favorable. Le modèle agit comme un **stop-loss intelligent** piloté par machine learning.

## Performance (net de frais Boursorama PEA)

Période : 2005-2026 (20.9 ans)

| Levier | CAGR | Total | MaxDD | Sells/an |
|---|---|---|---|---|
| QQQ Buy & Hold | 16.1% | 22.8x | -53.4% | — |
| **XGB x1.0** (PUST) | 14.9% | 18.4x | -9.0% | ~9 |
| **XGB x1.5** (PUST+LQQ) | 22.9% | 75.2x | -12.8% | ~11 |
| **XGB x2.0** (LQQ) | 30.9% | 279.2x | -16.8% | ~13 |

Dernières 10 ans :

| Levier | CAGR 10y | MaxDD 10y |
|---|---|---|
| QQQ Buy & Hold | 21.8% | -35.1% |
| **XGB x1.0** | 21.7% | -8.4% |
| **XGB x1.5** | 33.6% | -12.0% |
| **XGB x2.0** | 46.3% | -15.0% |

Frais modélisés : 0% achat (BoursoMarkets), 0.50% vente, seuil 20% pour désallocation, sortie crise immédiate. TER PUST 0.23%/an, LQQ 0.60%/an.

## Projection 5 ans (basée sur perf 10 dernières années)

PEA 150k€ + CTO 100k€ = 250k€. Fiscalité : PEA 17.2% PS à la sortie, CTO 30% flat tax/an.

| Levier | PEA net | CTO net | Total net | Impôts |
|---|---|---|---|---|
| **x1.0** (21.7%/an) | 357k€ | 203k€ | **560k€** | 87k€ |
| **x1.5** (33.6%/an) | 555k€ | 288k€ | **843k€** | 165k€ |
| **x2.0** (46.3%/an) | 857k€ | 407k€ | **1 264k€** | 278k€ |

## Architecture

### Workflow opérationnel

Clôture US → Signal → Exécution Europe le lendemain matin :

```
22h00 Paris  — Clôture US, données disponibles
22h30        — Calcul du signal (~2 min)
09h00        — Ouverture Euronext, passage d'ordre PUST/LQQ si nécessaire
```

PUST/LQQ sont des ETF synthétiques (swap-based) : à l'ouverture Europe, ils intègrent déjà la clôture US.

### Target : Realtime Drawdown State Machine

Label observable en temps réel (pas de lookahead) :

- **Exit** : drawdown atteint **-10%** → cash (target = 0)
- **Re-enter** : drawdown remonte au-dessus de **-5%** → investi (target = 1)

### Walk-Forward Expanding Window

```
MIN_TRAIN = 504 jours (~2 ans)
STEP = 21 jours (retrain mensuel)
EMBARGO = 21 jours (entre train et test)
```

Fenêtre d'entraînement **expansive** : chaque retrain utilise tout l'historique depuis le jour 0.

### Walk-Forward Strict Feature Selection

À chaque step du walk-forward, **avant l'entraînement** :

1. Découpe les données de train en **3 périodes entrelacées** (blocs de 21j, `block_idx % 3`)
2. Entraîne 3 modèles (chacun sur 2 périodes, importance mesurée sur la 3ème)
3. Classe les features par **stabilité = mean / std^1.7**
4. Garde uniquement les features stables (importance consistante entre les 3 périodes)

Résultat : 14 features au début (peu de données) → 65 features à la fin. **Zéro look-ahead bias** sur la sélection.

### XGBoost Configuration

```python
n_estimators=300, max_depth=4, learning_rate=0.03,
subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
reg_alpha=1.0, reg_lambda=5.0, gamma=1.0
```

### Temperature Scaling

```
P(invested) = sigmoid(logit / T)    avec T = 3.0
```

### Allocation Continue 0-150% (ou 0-200% en x2)

```
allocation = clip((P - 0.50) / (0.85 - 0.50), 0, 1) × max_leverage
```

### Implémentation PEA avec 2 ETF

| Allocation cible | PUST (1x) | LQQ (2x) | Cash |
|---|---|---|---|
| 0% | 0% | 0% | 100% |
| 100% | 100% | 0% | 0% |
| 150% | 50% | 50% | 0% |
| 200% | 0% | 100% | 0% |

Formule : pour allocation A > 100% → PUST = (2-A), LQQ = (A-1), ex: 150% = 50% PUST + 50% LQQ.

## Features (99 candidates, ~65 retenues après filtrage)

### Données sources

| Source | Fichier | Historique | Délai live | Téléchargement |
|---|---|---|---|---|
| QQQ (Nasdaq-100) | `QQQ.parquet` | 2000+ | Immédiat | yfinance |
| VIX (volatilité implicite) | `vix_ohlc.parquet` | 2000+ | Immédiat | yfinance (^VIX) |
| BAA Credit Spread | `fred_baa_spread.parquet` | 2000+ | **J+2** (décalé dans le backtest) | FRED API |
| TLT (20+ Year Treasury Bond) | `TLT.parquet` | 2002-07+ | Immédiat | yfinance |

Le spread BAA est décalé de 2 jours (`spread_lag=2`) dans le backtest pour refléter le délai de publication FRED. Impact neutre sur la performance.

### Catégories de features

**1. QQQ Price & Momentum (30 features)** — SMAs, returns, volatilité, drawdown, RSI
**2. VIX (10 features)** — SMAs, returns, VIX vs moyennes mobiles
**3. BAA Credit Spread (10 features)** — SMAs, returns, spread vs moyennes mobiles
**4. Asymmetric Volatility (6 features)** — downvol, upvol, ratio
**5. Skewness & Kurtosis (4 features)**
**6. Acceleration (3 features)** — diff des returns
**7. Volatility Regime (3 features)** — vol of vol, vol accel, vol20/vol100
**8. Drawdown Dynamics (3 features)** — vitesse, durée, profondeur × durée
**9. Cross-Asset Interactions (5 features)** — VIX × spread, VIX × DD, etc.
**10. Term Structure Proxies (2 features)** — VIX et spread short/long
**11. Bollinger Bands (2 features)**
**12. Mean Reversion (2 features)**
**13. Consecutive Down Days (1 feature)**
**14. Rolling Max Drawdown (2 features)**
**15. VIX Spike (2 features)**
**16. Spread Acceleration (1 feature)**
**17. TLT Risk-On/Risk-Off (16 features)** — QQQ/TLT ratio, momentum, interactions

### Top 20 Features (dernier modèle, après filtrage WF strict)

```
vix_x_dd                         0.2102    ← VIX × drawdown (#1 dominante)
qqq_dd                           0.1710    ← drawdown courant
dd_depth_x_duration              0.0471    ← profondeur × durée du drawdown
qqq_bband20                      0.0358    ← position Bollinger
qqq_vs_sma20                     0.0356    ← prix vs SMA20
qqq_maxdd20                      0.0307    ← max drawdown 20j
qqq_downvol20                    0.0301    ← volatilité baissière
qqq_vs_sma200                    0.0289    ← prix vs SMA200
qqq_vol50                        0.0280    ← volatilité 50j
qqq_vol20                        0.0268    ← volatilité 20j
dd_speed                         0.0243    ← vitesse du drawdown
qqq_ret10_vs_ret100              0.0241    ← mean reversion
qqq_ret5                         0.0215    ← return 5j
qqq_ret10                        0.0146    ← return 10j
qqq_accel5                       0.0121    ← accélération 5j
qqq_ret5_vs_ret50                0.0120    ← mean reversion
dd_duration                      0.0117    ← durée du drawdown
qqq_tlt_ratio_vs_sma20           0.0107    ← QQQ/TLT vs moyenne
qqq_upvol50                      0.0106    ← volatilité haussière
qqq_ret50                        0.0103    ← return 50j
```

## Tentatives d'ajout de features

### Feature retenue

| Feature | Impact | Détail |
|---|---|---|
| **TLT (risk-on/risk-off)** | **CAGR 13.4% → 24.8%** (+11.4%) | Le ratio QQQ/TLT capture les flux flight-to-safety. Mesure relative qui normalise les niveaux de prix absolus à travers tous les régimes. |

### Features testées et rejetées

| # | Feature | Features | Impact CAGR | Impact DD | Verdict |
|---|---------|---|---|---|---|
| 1 | **Yield curve (10Y-2Y)** | ~10 | Neutre | Neutre | Signal trop lent pour le timing de crise. C'est le **spread** 10Y-2Y, pas les taux directeurs Fed. |
| 3 | **HY spread (HYG/IEF)** | — | Non testé | — | Données `fred_hy_spread` trop courtes (depuis 2023). Infrastructure préparée mais jamais finalisé. |
| 4 | **Rotation sectorielle (XLY/XLP, XLK/XLU)** | 16 | +0.2% | -0.2% | Redondant avec VIX + spread + drawdown |
| 5 | **Dollar (UUP/DXY)** | 13 | N/A | N/A | Perte de 5 ans d'historique (UUP depuis 2007), neutre sur 10y |
| 6 | **SOXX vs QQQ (semiconducteurs)** | 15 | -0.1% | +0.1% | QQQ est déjà ~60% tech, signal redondant |
| 7 | **GLD (Gold)** | 16 | N/A | +1.0% | Perte de 2 ans d'historique, aucune feature dans top 20 |
| 8 | **Pétrole (USO)** | 11 | N/A | N/A | Perte de 4 ans d'historique, aucune feature utilisée |
| 9 | **Inflation (breakeven 5Y) + Fed Funds Rate** | 16 | -0.4% | -0.3% | Aucune feature dans top 20. VIX + spread réagissent déjà aux hausses de taux. FFR trop lent (8 changes/an). |

### Toutes les features candidates ont été testées

### Pourquoi les features additionnelles n'apportent rien (en général)

1. **VIX + BAA spread captent déjà le stress macro** : la rotation sectorielle, le dollar, l'or et le pétrole sont des conséquences du même stress
2. **Le drawdown QQQ est le signal dominant** : `vix_x_dd` représente 21% de l'importance — le modèle détecte les crises par le drawdown lui-même
3. **TLT est complémentaire car il mesure un axe différent** : le flight-to-safety est un signal distinct du VIX et du spread

### Note sur les taux directeurs Fed et l'inflation

Testés en dernier (feature #9) : breakeven inflation 5Y (T5YIE, quotidien FRED) + Fed Funds Rate (FEDFUNDS, mensuel forward-filled). 16 features ajoutées (niveaux, deltas 3/6/12 mois, interactions VIX/DD). Résultat : légèrement négatif (-0.4% CAGR), aucune feature retenue par le filtrage WF strict.

La crise 2022 a été causée par les hausses de taux, mais le VIX et le BAA spread **réagissent déjà** à ces hausses. Le drawdown QQQ capture directement l'effet quelle que soit la cause (taux, guerre, pandémie). Le FFR est trop discret (8 changes/an) pour apporter un signal au-delà de ce que le marché price déjà via VIX et spread.

## Fichiers

```
src/
├── download_ohlcv.py        # Télécharge QQQ, TLT, VIX via yfinance
├── download_macro_data.py   # Télécharge BAA spread via FRED
└── risk_off_strategy/
    ├── run.py               # Point d'entrée, configuration
    ├── data.py              # Chargement données, features, target
    └── backtest.py          # Walk-forward, feature selection, equity, graphiques

outputs/risk_off_strategy/
└── backtest.png             # 3 leviers + frais Bourso + projection fiscale 5 ans

data/
├── QQQ.parquet              # yfinance
├── TLT.parquet              # yfinance
├── vix_ohlc.parquet         # yfinance (^VIX)
└── fred_baa_spread.parquet  # FRED API (décalé J+2)
```

## Lancer le backtest

```bash
python3 src/risk_off_strategy/run.py
```

Ou via VSCode : configuration "QQQ backtest" dans `.vscode/launch.json`.
