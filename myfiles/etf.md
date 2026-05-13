# Stratégie ETF sur Boursorama — Modèle Quantitatif

Synthèse de la conversation du 19 avril – 13 mai 2025.

---

## 1. Contexte et objectif

Construire un modèle de trading quantitatif sur ETF UCITS accessibles depuis Boursobank,
avec un signal daily (mise à jour chaque soir) et une exécution manuelle (~15 ordres/mois).

**Contraintes :**
- Pas d'API d'exécution officielle Boursorama → exécution manuelle
- Budget données : ~$50 one-shot + $0/mois en production
- Enveloppes fiscales : PEA (equity) + CTO (bonds, or, commodités)

---

## 2. Univers ETF — Équivalents UCITS sur Euronext

Les ETF US (SPY, QQQ…) sont bloqués pour les particuliers européens (règlement PRIIPs/MiFID II).
Il faut utiliser les équivalents UCITS cotés en EUR sur Euronext.

### Equity géographique
| Ticker | Description              | Enveloppe | Boursomarkets |
|--------|--------------------------|-----------|---------------|
| CSP1   | S&P 500 (iShares)        | PEA ✅    | ✅ (≥500€)    |
| CNX1   | Nasdaq 100 (iShares)     | PEA ✅    | ✅            |
| IWDA   | MSCI World (iShares)     | PEA ✅    | ✅            |
| AEEM   | Emergents (Amundi)       | CTO       | à vérifier    |
| FXC    | Chine Large Cap (iShares)| CTO       | à vérifier    |

### Secteurs US (iShares S&P 500 sectoriels)
| Ticker | Secteur               | Enveloppe |
|--------|-----------------------|-----------|
| IUFS   | Finance               | CTO       |
| IUES   | Énergie               | CTO       |
| IUIT   | Tech                  | CTO       |
| IUHC   | Santé                 | CTO       |
| IUII   | Industriels           | CTO       |
| IUCS   | Conso défensive       | CTO       |
| IUCD   | Conso cyclique        | CTO       |

### Obligations / Taux — CTO uniquement
| Ticker  | Description                | Signal           |
|---------|----------------------------|------------------|
| DTLA    | T-bonds 20 ans (iShares)   | Risk-off refuge  |
| IBTA    | T-bonds 7-10 ans (iShares) | Taux moyen terme |
| IHYU    | High Yield (iShares)       | Appétit au risque|
| TIP/ITPS| TIPS inflation             | Hedge inflation  |

### Commodités & Alternatif — CTO uniquement
| Ticker | Description              | Signal               |
|--------|--------------------------|----------------------|
| IGLN   | Or physique (iShares)    | Risk-off / inflation |
| OIL    | Pétrole (Lyxor)          | Macro commodités     |
| LVO    | VIX futures (Lyxor)      | produit complexe     |

### ETN Crypto (disponibles depuis mars 2025 sur Boursobank)
- iShares Bitcoin ETP, CoinShares Physical BTC, ETH, SOL, ADA, XRP
- CTO uniquement, ETN (ETF) : adossé à du crypto physique en custody

**Note Boursomarkets :** achat >= 500EUR -> 0EUR de courtage. Vente -> ~0.22% (tarif standard).
Symbole Boursorama = préfixe `1rT` + ticker (ex : CSP1 -> `1rTCSP1`).

---

## 3. Architecture du modèle

```
DONNEES HISTORIQUES (1 mois FMP ~$50)
         |
+---------------------------------------------+
|  Bulk download 5-7 ans                      |
|  Prix OHLCV + News + Sentiment              |
|  Holdings ETF + Analyst notes composants    |
|  Economic calendar + Earnings transcripts   |
|  -> stocké en Parquet local                 |
+---------------------------------------------+
                  |
+---------------------------------------------+
|  COUCHE NLP                                 |
|  spaCy NER -> entités (Fed, Trump, COVID…)  |
|  FinBERT -> score sentiment par entité      |
+---------------------------------------------+
                  |
+---------------------------------------------+
|  TEMPORAL CAUSAL GRAPH (TCG)                |
|  Arêtes causales : Trump->SPX, Fed->TLT…    |
|  Decay conditionné par régime de marché :   |
|    Bull       -> half-life 60j              |
|    Transition -> half-life 180j             |
|    Bear       -> half-life 365j             |
|    Crisis     -> half-life 730j             |
|  Reinforcement si événement similaire       |
|  Analogie structurelle (H5N1 ~ COVID x0.71) |
+---------------------------------------------+
                  |
+---------------------------------------------+
|  DETECTION DE REGIME                        |
|  Phase 1 MVP : règles expert                |
|    VIX > 50  -> crisis                      |
|    VIX > 30 + slope < 0 -> bear            |
|    VIX < 15 + slope > 0 -> bull            |
|    sinon -> transition                      |
|  Phase 2 prod : HMM (hmmlearn)              |
|    -> probabilités par régime               |
|    -> half-life pondéré continu             |
+---------------------------------------------+
                  |
+---------------------------------------------+
|  FEATURE ENGINEERING                        |
|  Sentiment :                                |
|    sentiment_weighted = score x influence   |
|                         x decay(t)          |
|    sentiment_zscore, sentiment_change       |
|    graph_resonance, novelty_score           |
|  Technique : momentum_20d, vol_20d, RSI     |
|  Fondamental : analyst_consensus pondéré    |
|    (notes analystes top 20 composants ETF)  |
|  Flux : rotation bonds->equity, HY spread  |
|  Régime : encoded, duration, confidence     |
|  VIX : deviation, velocity, reversion       |
+---------------------------------------------+
                  |
+---------------------------------------------+
|  XGBOOST PAR REGIME                         |
|  1 modèle bull / 1 transition / 1 bear-crisis|
|  Router : régime détecté -> bon modèle      |
|  Anti-overfit : max_depth=3, min_child=50   |
|  SHAP values -> validation alpha sentiment  |
+---------------------------------------------+
                  |
         Signal [-1, +1] par ETF
                  |
    Seuil de changement -> ordre ou HOLD
```

---

## 4. Signal de transition de phase (multi-confirmation)

Le modèle détecte les transitions entre phases via la **convergence** de 3 sources :

### Séquence chronologique typique
```
J-5 : News macro sémantique (NLP)
      "Fed pivote", "unlimited QE", "whatever it takes"
      -> signal le plus précoce

J-2 : Flux institutionnels bougent
      -> rotation TLT->CSP1 observable dans les volumes
      -> HY spread commence à se resserrer

J0  : VIX peak puis déclin confirmé
      -> triple confirmation

J+1 : Signal XGBoost déclenché
J+90: Rally historique ~72-89% de probabilité
```

### Score de récupération post-crise
```python
recovery_score = moyenne([
    vix < vix_5d_max * 0.90,      # VIX declining
    put_call < put_call_5d_avg,    # Fear declining
    hy_spread < hy_spread_5d_max,  # Credit healing
    pct_above_200ma > 25,          # Breadth recovering
])
# >= 0.75 -> BUY signal (fiabilité ~85-90%)
```

### Propriété mean-reverting du VIX
- VIX suit un processus d'Ornstein-Uhlenbeck -> obligé de retomber
- Plafond historique absolu : ~85 (COVID mars 2020)
- Demi-vie moyenne après VIX > 40 : retour sous 30 en 30-90 jours
- **Règle :** ne pas acheter pendant la montée, attendre le déclin confirmé

### ETF à surpondérer après un pic VIX
- J0-J30   : CSP1, CNX1 (rebond large marché et tech)
- J30-J90  : IUCD (cyclique), IUFS (financials), AEEM (émergents)
- Alléger  : DTLA (T-bonds), IGLN (or)

---

## 5. Features analyst notes synthétiques sur ETF

Les ETF n'ont pas de notes analystes directes. On construit un score composite :

```python
# Pour chaque ETF, top 20 composants pondérés
analyst_consensus_etf = sum(
    poids_i * (buy_i - sell_i) / total_analysts_i
    for i in top20_holdings
)

# Autres features dérivées
upgrade_momentum_30d   = sum(poids_i * (upgrades_i - downgrades_i))
price_target_upside    = sum(poids_i * (target_i - price_i) / price_i)
```

**Données nécessaires sur FMP :**
- `etf-holdings` -> poids des composants (1 call/mois par ETF)
- `analyst-stock-recommendations` -> Buy/Hold/Sell par action
- `upgrades-downgrades` -> momentum des révisions (1 call/semaine)

---

## 6. Stack technique

| Composant           | Outil                        | Coût    |
|---------------------|------------------------------|---------|
| Données bulk        | FMP (1 mois payant)          | ~$50    |
| Prix EOD daily      | Finnhub free                 | $0      |
| News daily          | Finnhub free                 | $0      |
| Macro officiel      | FRED API                     | $0      |
| News géopolitiques  | GDELT Project                | $0      |
| Holdings ETF        | FMP free (<250 calls/jour)   | $0      |
| Analyst notes       | FMP free (1x/semaine)        | $0      |
| NER                 | spaCy                        | $0      |
| Sentiment NLP       | FinBERT (HuggingFace)        | $0      |
| Graphe causal       | NetworkX + DoWhy             | $0      |
| Régime HMM          | hmmlearn                     | $0      |
| Modèle décision     | XGBoost + SHAP               | $0      |
| Backtest            | VectorBT                     | $0      |
| Scheduler           | APScheduler                  | $0      |
| Stockage            | Parquet local                | $0      |

**Coût total : $50 one-shot, $0/mois en production.**

---

## 7. Sources de données gratuites fiables (daily prod)

| Donnée                  | Source              | Fiabilité     |
|-------------------------|---------------------|---------------|
| Taux, inflation, GDP    | FRED API (officiel) | Excellente    |
| VIX historique          | FRED code VIXCLS    | Excellente    |
| HY spread               | FRED BAMLH0A0HYM2   | Excellente    |
| Money market flows      | FRED WRMFSL         | Excellente    |
| News géopolitiques      | GDELT (15min delay) | Très bonne    |
| Prix EOD                | Finnhub free        | Bonne         |
| News + sentiment stock  | Finnhub free        | Bonne         |
| Holdings ETF            | FMP free            | Bonne         |

---

## 8. Budget API daily (Finnhub + FMP free)

```
Finnhub free : 60 calls/minute
  Prix EOD 20 ETF      : 20 calls
  News 20 ETF          : 20 calls
  Macro (1 call)       :  1 call
  Total/jour           : ~41 calls -> 1 minute de batch OK

FMP free : 250 calls/jour
  Holdings (mensuel)   : 20/30 = 0.7 calls/jour
  Analyst notes (hebdo): 20x3/7 = 8.6 calls/jour
  Palmarès ETF         : 2 calls/jour
  Total/jour           : ~12 calls -> 5% du quota OK
```

---

## 9. Pipeline daily (21h30 chaque soir)

```
1.  Fetch FRED : VIX, HY spread, taux, inflation
2.  Fetch Finnhub free : prix EOD + news 20 ETF (~1 min)
3.  Fetch FMP free : palmarès + holdings (si jour J)
4.  NLP (GPU) : NER + FinBERT sur news du jour
5.  TCG update : decay recalculé au régime courant
6.  Régime détecté : VIX + slope 200MA
7.  Features générées : sentiment + technique + flux + régime
8.  XGBoost[régime].predict()
9.  Détection de changement de signal
10. Notification si ordre à passer -> exécution manuelle le matin
```

**Durée totale :** ~10-15 minutes.

---

## 10. Règle d'exécution (anti-churning)

```python
def should_trade(signal_t, signal_t_prev, threshold=0.3):
    if signal_t_prev > 0 and signal_t < -threshold:
        return "SELL"
    if signal_t_prev < 0 and signal_t > threshold:
        return "BUY"
    if abs(signal_t) > 0.7 and abs(signal_t_prev) < 0.3:
        return "BUY" if signal_t > 0 else "SELL"
    return "HOLD"  # majorité des jours
```

**Volume estimé :** 0-2 trades/jour normal, 2-5 en rotation, 5-10 en crise.
**Moyenne réaliste :** ~15-20 trades/mois -> frais ~30-40EUR/mois (vente ~2EUR/ordre).

---

## 11. Enveloppes fiscales (Boursobank)

| Enveloppe | Contenu                         | Fiscalité après 5 ans        |
|-----------|---------------------------------|------------------------------|
| PEA       | ETF equity (CSP1, CNX1, IWDA)  | Exonéré d'impôt (17.2% PS)  |
| CTO       | Bonds, or, commodités, crypto   | PFU 30% sur les gains        |

**Au-delà de 100K€ :**
- Cash en compte -> garanti FGDR à 100K€ seulement -> risque au-delà
- ETF détenus -> titres ségrégués du bilan Boursobank -> pas de limite de garantie
- Fonds euros assurance-vie -> exposition dette française + Loi Sapin 2 (blocage possible) -> moins sûr qu'ETF au-delà de 100K€

---

## 12. Backtest walk-forward (sans fuite temporelle)

```python
for t in trading_days:
    # JAMAIS de données futures
    train_data = all_data.loc[:t]               # expanding window
    graph      = build_causal_graph(train_data)  # graphe sur passé only
    half_lives = calibrate_halflife(train_data)  # calibré sur passé only

    features   = compute_features(graph, half_lives, t)
    signal     = xgboost[regime_t].predict(features)
    realized   = returns.loc[t + 1]             # observé après

# Métriques cibles :
# Sharpe OOS > 1.0
# IS Sharpe / OOS Sharpe < 2 (sinon overfit)
# Max drawdown < 15%
```

---

## 13. Validation de l'univers sur Boursorama

```python
# Symbole Boursorama = préfixe 1rT + ticker Euronext
def ticker_to_boursorama(ticker):
    return f"1rT{ticker}"

# Vérification disponibilité
url = f"https://www.boursorama.com/bourse/trackers/cours/1rT{ticker}/"
available = requests.get(url).status_code == 200

# Date de cotation (évite look-ahead dans le backtest)
# Via FMP : first_price_date = date du premier cours historique
```

---

## 14. Roadmap d'implémentation

| Semaine | Tâche                                                    |
|---------|----------------------------------------------------------|
| 1       | Bulk download FMP (1 mois), stockage Parquet             |
| 1       | Validation univers ETF sur Euronext/Boursorama           |
| 2       | Pipeline NLP (spaCy + FinBERT) + TCG statique initial    |
| 2       | Détection régime (règles expert VIX + slope)             |
| 3       | Feature engineering complet + XGBoost MVP                |
| 3       | Walk-forward backtest + SHAP validation                  |
| 4       | Scheduler daily + notifications + monitoring             |
| 4       | HMM pour détection régime probabiliste (phase 2)         |
