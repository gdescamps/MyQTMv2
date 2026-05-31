# QQQ Crisis-Avoidance Strategy

## Objectif

Stratégie de timing sur QQQ (Nasdaq-100 ETF) qui détecte les périodes de crise pour réduire l'exposition, et utilise du levier modéré (jusqu'à 150%) en période favorable. Le modèle agit comme un **stop-loss intelligent** piloté par machine learning.

## Performance

| Métrique | QQQ Buy & Hold | XGBoost WF 0-150% |
|---|---|---|
| **Période** | 2005-2026 (20.9 ans) | 2005-2026 (20.9 ans) |
| **CAGR** | 16.0% | 24.8% |
| **Total** | 22.3x | 101.5x |
| **Max Drawdown** | -53.4% | -11.7% |

| Dernières 10 ans | QQQ Buy & Hold | XGBoost WF 0-150% |
|---|---|---|
| **CAGR** | 21.9% | 36.2% |
| **Total** | 7.2x | 21.9x |
| **Max Drawdown** | -35.1% | -10.8% |

Allocation moyenne : 89%, médiane : 128%. Le modèle est majoritairement investi avec levier, et passe à 0% pendant les crises.

## Architecture

### Convention close-to-close

On calcule la prédiction à T-30 minutes avant la clôture et on exécute sur QQQ (très liquide). Le backtest utilise les rendements close-to-close, ce qui est réaliste pour ce type d'exécution.

### Target : Realtime Drawdown State Machine

Le label d'entraînement est construit avec une **machine à états observable en temps réel** (pas de lookahead) :

- **Exit** : quand le drawdown depuis le plus haut atteint **-10%** → passe en cash (target = 0)
- **Re-enter** : quand le drawdown remonte au-dessus de **-5%** → repasse investi (target = 1)

```
DD_EXIT = -0.10
DD_REENTER = -0.05
```

Ce target est 100% réaliste car il n'utilise que le running max et le prix courant, tous deux connus au moment t.

### Walk-Forward Expanding Window

```
MIN_TRAIN = 504 jours (~2 ans)
STEP = 21 jours (retrain mensuel)
EMBARGO = 21 jours (entre train et test)
```

- Fenêtre d'entraînement **expansive** : chaque retrain utilise tout l'historique disponible
- Embargo de 21 jours entre la fin du train et le début du test pour éviter toute fuite
- Chaque fenêtre de test (21j) est concaténée pour former le backtest complet

### XGBoost Configuration

```python
n_estimators=300, max_depth=4, learning_rate=0.03,
subsample=0.7, colsample_bytree=0.7, min_child_weight=20,
reg_alpha=1.0, reg_lambda=5.0, gamma=1.0
```

Modèle fortement régularisé pour éviter l'overfitting sur les rares événements de crise.

### Temperature Scaling

Les logits bruts de XGBoost sont lissés via :

```
P(invested) = sigmoid(logit / T)    avec T = 3.0
```

La température T=3.0 compresse les probabilités vers 0.5, évitant les décisions binaires brutales et permettant une allocation continue.

### Allocation Continue 0-150%

```
allocation = clip((P - 0.50) / (0.85 - 0.50), 0, 1) × 1.5
```

| P(invested) | Allocation |
|---|---|
| < 0.50 | 0% (cash) |
| 0.50 | 0% |
| 0.675 | 75% |
| 0.85 | 150% (max levier) |
| > 0.85 | 150% |

## Features (99 total)

### Données sources

| Source | Fichier | Historique |
|---|---|---|
| QQQ (Nasdaq-100) | `QQQ.parquet` | 2000+ |
| VIX (volatilité implicite) | `vix_ohlc.parquet` | 2000+ |
| BAA Credit Spread | `fred_baa_spread.parquet` | 2000+ |
| TLT (20+ Year Treasury Bond) | `TLT.parquet` | 2002-07+ |

### Catégories de features

**1. QQQ Price & Momentum (30 features)**
- SMAs : 5, 10, 20, 50, 100, 200 jours
- Returns : 5, 10, 20, 50, 100, 200 jours
- Volatilité réalisée : 5, 10, 20, 50, 100, 200 jours
- Prix vs SMAs : 20, 50, 100, 200 jours
- Drawdown courant, RSI 14

**2. VIX (10 features)**
- SMAs : 5, 10, 20, 50 jours
- Returns : 5, 10, 20, 50 jours
- VIX vs SMA20, VIX vs SMA50

**3. BAA Credit Spread (10 features)**
- SMAs : 5, 10, 20, 50 jours
- Returns : 5, 10, 20, 50 jours
- Spread vs SMA20, Spread vs SMA50

**4. Asymmetric Volatility (6 features)**
- Downside vol, upside vol, ratio down/up pour fenêtres 20 et 50j

**5. Skewness & Kurtosis (4 features)**
- Rolling skew et kurtosis sur 20 et 50j

**6. Acceleration (3 features)**
- Diff du return sur 5, 10, 20j

**7. Volatility Regime (3 features)**
- Vol of vol (20j), vol acceleration (5j), vol regime (vol20 / vol100)

**8. Drawdown Dynamics (3 features)**
- DD speed (5j), DD duration, DD depth x duration

**9. Cross-Asset Interactions (5 features)**
- VIX x spread, VIX x drawdown, VIX x vol20, spread x vol20, VIX x ret20

**10. Term Structure Proxies (2 features)**
- VIX SMA5/SMA50, spread SMA5/SMA50

**11. Bollinger Bands (2 features)**
- Position dans la bande pour 20 et 50j

**12. Mean Reversion (2 features)**
- ret5 - ret50, ret10 - ret100

**13. Consecutive Down Days (1 feature)**

**14. Rolling Max Drawdown (2 features)**
- Max drawdown sur 20 et 50j

**15. VIX Spike (2 features)**
- Spike 1 jour, spike 5 jours

**16. Spread Acceleration (1 feature)**

**17. TLT Risk-On/Risk-Off (16 features)**
- QQQ/TLT ratio + momentum (5, 10, 20, 50j)
- Ratio vs SMA20, SMA50
- TLT momentum (5, 10, 20, 50j)
- TLT vs SMA20, SMA50
- Interactions : TLT ret5 x VIX, TLT ret5 x drawdown

### Top 20 Features (importance du dernier modèle)

```
vix_x_dd                         0.1803    ← VIX × drawdown (feature #1 dominante)
qqq_dd                           0.1222    ← drawdown courant
dd_depth_x_duration              0.0616    ← profondeur × durée du drawdown
qqq_downvol20                    0.0388    ← volatilité baissière
qqq_bband20                      0.0312    ← position Bollinger
qqq_maxdd20                      0.0285    ← max drawdown 20j
qqq_vs_sma20                     0.0285    ← prix vs SMA20
qqq_ret5                         0.0265    ← return 5j
qqq_vol50                        0.0264    ← volatilité 50j
qqq_vs_sma200                    0.0247    ← prix vs SMA200
qqq_ret10                        0.0233    ← return 10j
qqq_vol20                        0.0216    ← volatilité 20j
dd_duration                      0.0184    ← durée du drawdown
qqq_vol_asym50                   0.0182    ← asymétrie vol 50j
vix                              0.0166    ← VIX brut
qqq_ret5_vs_ret50                0.0145    ← mean reversion
dd_speed                         0.0137    ← vitesse du drawdown
qqq_ret100                       0.0121    ← return 100j
vix_x_spread                     0.0112    ← VIX × credit spread
qqq_tlt_ratio_ret5               0.0109    ← QQQ/TLT momentum 5j
```

Le modèle repose principalement sur : **drawdown × VIX**, **drawdown courant**, **dynamique du drawdown** et **volatilité asymétrique**.

## Tentatives d'ajout de features

### Feature retenue

| Feature | Impact | Détail |
|---|---|---|
| **TLT (risk-on/risk-off)** | **CAGR 13.4% → 24.8%** (+11.4%) | Le ratio QQQ/TLT est une mesure relative qui fonctionne à travers tous les régimes de prix. Capture les flux flight-to-safety. 16 features ajoutées. |

L'ajout de TLT a été le seul changement majeur. Avant TLT, le modèle était trop conservateur car le dot-com crash (2000-2002) dominait l'entraînement. Le ratio QQQ/TLT normalise les niveaux de prix absolus.

### Features testées et rejetées

| # | Feature | Features ajoutées | Impact CAGR | Impact DD | Verdict |
|---|---------|---|---|---|---|
| 1 | **Yield curve (10Y-2Y)** | ~10 | Neutre | Neutre | Signal trop lent pour le timing de crise |
| 4 | **Rotation sectorielle (XLY/XLP, XLK/XLU)** | 16 | +0.2% | -0.2% | Redondant avec VIX + spread + drawdown |
| 5 | **Dollar (UUP/DXY)** | 13 | N/A | N/A | Perte de 5 ans d'historique (UUP depuis 2007), neutre sur 10y |
| 6 | **SOXX vs QQQ (semiconducteurs)** | 15 | -0.1% | +0.1% | QQQ est déjà ~60% tech, signal redondant |
| 7 | **GLD (Gold)** | 16 | N/A | +1.0% | Perte de 2 ans d'historique (GLD depuis 2004), aucune feature dans top 20 |
| 8 | **Pétrole (USO)** | 11 | N/A | N/A | Perte de 4 ans d'historique (USO depuis 2006), aucune feature utilisée |

### Pourquoi les features additionnelles n'apportent rien

1. **VIX + BAA spread captent déjà le stress macro** : la rotation sectorielle, le dollar, l'or et le pétrole sont des conséquences du même stress que VIX et spread mesurent directement
2. **Le drawdown QQQ est le signal dominant** : `vix_x_dd` (VIX × drawdown) représente 18% de l'importance totale — le modèle détecte les crises principalement par le drawdown lui-même, amplifié par le VIX
3. **TLT est complémentaire car il mesure un axe différent** : le flight-to-safety (actions → obligations) est un signal distinct du VIX et du spread, c'est pourquoi il améliore le modèle

**Conclusion** : le modèle est au **Pareto frontier** avec 4 sources de données (QQQ, VIX, BAA spread, TLT). Les 99 features actuelles sont suffisantes.

## Fichiers

```
src/qqq_strategy/
├── run.py          # Point d'entrée, configuration
├── data.py         # Chargement données, features, target
└── backtest.py     # Walk-forward XGBoost, equity, graphiques

outputs/qqq_strategy/
└── backtest.png    # Graphique equity + allocation

data/
├── QQQ.parquet
├── vix_ohlc.parquet
├── fred_baa_spread.parquet
└── TLT.parquet
```

## Lancer le backtest

```bash
python src/qqq_strategy/run.py
```

Ou via VSCode : configuration "QQQ backtest" dans `.vscode/launch.json`.
