# Plan — Modèle Quantitatif ETF (MyQTM-ETF)

Stratégie de répartition dynamique de portefeuille sur l'univers ETF Boursorama/UCITS.
Inspiré de MyQTM (actions US) mais adapté aux ETF et à la gestion de portefeuille.

---

## 1. Objectif et différences avec MyQTM

### MyQTM (actions US)
```
Signal par action : long / short / hold  [-1, 0, +1]
CMA-ES → seuils de probabilité → sélectionne 3-4 positions parmi 20-30 stocks
MAX_POSITIONS = 12, horizon signal = daily
Sortie : liste de positions binaires
```

### MyQTM-ETF (portefeuille ETF)
```
Sortie : vecteur de poids w[i] ≥ 0, Σw[i] ≤ 1  (reste en cash si < 1)
XGBoost → score par ETF → softmax conditionné au régime → poids
Répartition dynamique selon phase de marché (bull / transition / bear / crisis)
Horizon : daily, ~15 ordres/mois, exécution manuelle sur Boursobank
```

**Pourquoi changer le formalisme ?**
Les ETF représentent des classes d'actifs entières — le bon outil est la **répartition de portefeuille**,
pas le long/short par asset. En phase de crise, on veut 0% equity + 40% or + 40% bonds + 20% cash,
pas un signal binaire par ligne. Le formalisme "poids" est plus naturel et plus stable OOS.

---

## 2. Univers

Voir `myfiles/etf_univers.md` — 41 ETFs UCITS Boursorama répartis en 6 sections :

| Section       | N  | FMP direct | Zero-fees | PEA |
|---------------|----|:---------:|:---------:|:---:|
| Geo equity    | 16 | 8          | 15        | 3   |
| Secteurs US   | 7  | 0 (proxy XL*) | 0      | 0   |
| Thématiques   | 9  | 6          | 9         | 2   |
| Commodités    | 4  | 3          | 3         | 1   |
| Obligations   | 4  | 0 (proxy US) | 0       | 0   |
| Crypto        | 1  | 0 (IBIT)   | 0         | 0   |

**Univers de trading actif** (modèle allège l'univers selon régime) :
- Bull    : Geo equity + Secteurs US + Thématiques → ~25 ETFs actifs
- Bear    : Obligations + Or + Commodités → ~6 ETFs défensifs
- Crisis  : TLT/IEF + IGLN + Cash → 3-4 ETFs max
- Transition : mix pondéré des deux

---

## 3. Pipeline de données

### 3a. Historique initial (FMP payant, 1 mois ~$50)

```
download_etf_data.py    → OHLCV max history pour 41 tickers FMP proxy
                          ex. QQQ depuis 2000, SEMI.AS depuis 2021
                          → ./data/{TICKER}.parquet

download_macro_data.py  → FRED : VIX, HY spread, yield curve, WTI, gold
                          → ./data/fred_{series}.parquet
                          + flow_proxies.parquet (dollar volume z-scores)

download_etf_holdings.py  [À CRÉER]
  FMP /v3/etf-holder/{symbol}          → top holdings + poids (mensuel)
  FMP /v3/etf-info/{symbol}            → AUM, expense ratio, sector weights
  FMP /v3/analyst-stock-recommendations/{comp}  → notes analystes composants
  → ./data/holdings/{ETF_TICKER}.parquet

download_etf_news.py  [À CRÉER]
  FMP /v3/stock_news?tickers={top20_composants}  → news des composants
  → ./data/news/{DATE}.parquet
```

### 3b. Mise à jour journalière (FMP free, ~250 calls/jour)

```
Budget FMP free 250 calls/jour :
  Prix EOD 41 ETFs     :  41 calls  (via /quote bulk)
  Holdings (mensuel)   :  41/30 ≈   2 calls/jour
  Analyst notes (hebdo):  20×3/7 ≈  9 calls/jour
  News composants      :  20        calls/jour
  Macro FRED           :   0        (FRED API gratuit, sans quota)
  ─────────────────────────────────
  Total                : ~72 calls  → 29% du quota ✅

Budget FRED illimité :
  VIX, HY spread, yield curve, WTI → mise à jour quotidienne
```

---

## 4. Features

### 4a. Signaux techniques — repris de MyQTM

Calculés sur chaque ETF proxy FMP (OHLCV) et sur les indices macro :

```python
# Par ETF (depuis data_transform_price_trends_indicators_time_series.py MyQTM)
features_technique = {
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",      # momentum multi-horizon
    "vol_20d", "vol_60d",                            # volatilité réalisée
    "rsi_14",                                        # RSI
    "ma_20", "ma_50", "ma_200",                      # moyennes mobiles
    "price_vs_ma50",  "price_vs_ma200",              # position vs MA
    "ma_20_slope", "ma_50_slope",                    # direction tendance
    "atr_14",                                        # range normalisé
    "volume_z20",                                    # volume z-score 20j
}

# Sur indices macro (repris identiquement de MyQTM)
features_macro_technique = {
    "qqq_ret_20d", "qqq_ma50_slope",                 # Nasdaq
    "vix_level", "vix_ret_5d", "vix_vs_ma20",        # VIX
    "gld_ret_20d", "gld_vs_ma50",                    # Or
    "wti_ret_20d",                                   # Pétrole
    "iwda_ret_20d",                                  # MSCI World (CSPX proxy)
    "tlt_ret_20d",                                   # Bonds LT
    "hyg_ret_5d",                                    # HY (risque crédit)
}
```

### 4b. Régime de marché — repris et étendu de MyQTM

```python
# Phase 1 MVP : règles VIX (identique MyQTM)
def detect_regime(vix, vix_slope):
    if vix > 50:                          return "crisis"
    if vix > 30 and vix_slope < 0:        return "bear"
    if vix < 15 and vix_slope > 0:        return "bull"
    return "transition"

# Phase 2 prod : HMM (hmmlearn) sur [VIX, HY_spread, yield_curve, ret_spx]
# → probabilités continues par régime (bull_prob, bear_prob, crisis_prob)
# → half-life de decay des features conditionné au régime

features_regime = {
    "regime_encoded",          # 0=bull, 1=transition, 2=bear, 3=crisis
    "regime_duration_days",    # jours dans le régime actuel
    "bull_prob", "bear_prob", "crisis_prob",  # HMM phase 2
    "vix_deviation",           # vix - vix_ma60 (distance à la normale)
    "vix_velocity",            # variation 5j du VIX
    "vix_reversion_force",     # (vix_mean - vix) / vix_std  (Ornstein-Uhlenbeck)
    "recovery_score",          # 0-4 confirmateurs (cf. §7 etf_strategy.txt)
}
```

### 4c. Analyst consensus synthétique — spécifique ETF

Les ETF n'ont pas de notes analystes directes. On construit un score composite
à partir des composants (repris de MyQTM `data_transform_analyst_stock_recommendations_time_series.py`) :

```python
# FMP: /etf-holder/{symbol} → top 20 holdings + poids
# FMP: /analyst-stock-recommendations/{comp} → buy/hold/sell

analyst_consensus_etf = Σ( weight_i × (buy_i - sell_i) / total_i )
                          pour i in top_20_holdings

upgrade_momentum_30d  = Σ( weight_i × (upgrades_30d_i - downgrades_30d_i) )
price_target_upside   = Σ( weight_i × (target_price_i - price_i) / price_i )

# Fréquence de mise à jour : hebdomadaire (économise les appels FMP)
# → features stables 7j, mis à jour chaque lundi
```

### 4d. Rotation de composition — signal précoce

```python
# FMP: /etf-holder → snapshot mensuel des poids sectoriels
# Variation de composition = signal avant le prix

sector_rotation_30d = etf_sector_weight_t - etf_sector_weight_t_minus_30d
# Ex : si XLK perd du poids dans le S&P 500 → rotation défensive imminente

features_composition = {
    "top10_concentration",          # Σ poids top 10 (stabilité/risque)
    "tech_weight_etf",              # poids secteur tech dans l'ETF
    "tech_weight_delta_30d",        # variation poids tech sur 30j
    "geo_us_weight",                # exposition US (pour ETF monde/EM)
    "holdings_count",               # nombre de lignes (diversification)
}
```

### 4e. Sentiment — news composants via LLM

Adapté de MyQTM (`data_transform_stock_news_to_sentiment_scores.py`) :

```python
# MyQTM utilise Gemini (GCP) pour scorer les news
# Pour les ETF : on score les news des top 10 composants
#
# Pipeline :
# 1. FMP /stock_news?tickers={top10} → news des composants (daily)
# 2. Filtre RELIABLE_NEWS_SITES (repris de MyQTM config.py)
# 3. Score via LLM (Gemini Flash ≈ $0.0001/news) → cache pickle
# 4. Agrégation pondérée par poids dans l'ETF

sentiment_etf = Σ( weight_i × sentiment_score_i × decay(age_news) )
                  pour i in top_10_holdings

features_sentiment = {
    "sentiment_weighted",      # score agrégé pondéré
    "sentiment_zscore_20d",    # déviation vs 20j (choc vs baseline)
    "sentiment_velocity",      # variation 3j du sentiment
    "news_volume_z20",         # volume de news (attention du marché)
    "novelty_score",           # nouveauté sémantique (nouveau thème?)
}

# Coût estimé : ~20 composants × 5 news/j × $0.0001 = $0.01/jour ✅
# Cache : llm_cache.pkl (identique MyQTM)
```

**Sentiment géographique et sectoriel :**
```python
# Agrégation cross-ETF pour détecter des thèmes globaux
geo_sentiment_asia = average(sentiment_etf(CSKR), sentiment_etf(ITWN), sentiment_etf(IFFI))
sector_sentiment_tech = average(sentiment_etf(IUIT), sentiment_etf(SEMI), sentiment_etf(CNX1))
```

### 4f. Mouvements institutionnels (smart money)

```python
# Proxy 1 : dollar volume z-score (déjà dans flow_proxies.parquet)
dollar_vol_z20 = (close × volume - mean_20d) / std_20d
rotation_z60   = dvol_risk_on_avg - dvol_risk_off_avg  (z-scoré 60j)

# Proxy 2 : 13F institutional holdings (FMP, trimestriel)
# FMP: /v4/institutional-ownership/symbol-ownership?symbol=QQQ
# → variation de positions des grands fonds vs trimestre précédent
inst_position_change_qtrly = shares_held_t - shares_held_t_minus_1q
inst_concentration_change  = (nb_new_holders - nb_exiting_holders)

# Proxy 3 : Put/Call ratio (CBOE, gratuit)
# → peur institutionnelle mesurée via le marché des options
put_call_ratio = put_vol / call_vol
put_call_z20   = z_score(put_call_ratio, 20d)

# Proxy 4 : HY spread (FRED BAMLH0A0HYM2)
# → les institutionnels fuient le crédit risqué en bear
hy_spread_z60  = z_score(hy_spread, 60d)
hy_spread_velocity = hy_spread_t - hy_spread_t_minus_5d

features_smart_money = {
    "rotation_z60",             # rotation risk-on vs risk-off (OHLCV)
    "hy_spread_z60",            # stress crédit (FRED)
    "hy_spread_velocity",       # accélération du stress
    "put_call_z20",             # peur institutionnelle (CBOE)
    "inst_position_change",     # variation 13F (trimestriel, interpolé)
    "yield_curve",              # 10Y-2Y (FRED T10Y2Y)
    "yield_curve_velocity",     # inversion/désinversion
}
```

---

## 5. Modèle XGBoost — adaptation pour portefeuille ETF

### Approche retenue : score par ETF → allocation par régime

```
MyQTM     : XGBoost classify → long/short/hold → CMA-ES seuils → top N positions
MyQTM-ETF : XGBoost regress  → score continu   → softmax × budget_régime → poids
```

### Architecture — 1 seul modèle avec régime intégré

Le régime n'est **pas un router externe** (3 modèles séparés comme MyQTM) mais une **feature** parmi les autres. XGBoost apprend lui-même les interactions entre régime et signaux techniques.

```python
# 1 seul XGBRegressor — régime intégré comme features
model = XGBRegressor(max_depth=3, min_child_weight=50, ...)

# Features d'entrée = technique + sentiment + smart_money + REGIME
X = [
    # ... features technique, sentiment, smart_money ...

    # Régime Phase 1 (règles VIX)
    "regime_encoded",        # 0=bull 1=transition 2=bear 3=crisis
    "regime_duration_days",  # depuis combien de jours dans ce régime

    # Régime Phase 2 (HMM probabiliste — sortie continue)
    "bull_prob",             # ex. 0.72
    "transition_prob",       # ex. 0.20
    "bear_prob",             # ex. 0.06
    "crisis_prob",           # ex. 0.02

    # Dynamique du régime
    "vix_level",
    "vix_velocity",          # Δvix 5j
    "vix_reversion_force",   # (vix_mean - vix) / vix_std
    "hy_spread_z60",
    "yield_curve",           # 10Y - 2Y
]

score_per_etf = model.predict(X_today)
# XGBoost apprend : "en bull_prob élevé, momentum_20d pèse plus"
#                   "en crisis_prob élevé, hy_spread_velocity pèse plus"
```

**Avantages vs router externe :**
- 1 modèle au lieu de 3 → plus de données d'entraînement, moins d'overfitting
- Transitions continues via `bull_prob` / `bear_prob` → pas de saut brutal entre régimes
- Interactions régime × feature apprises automatiquement par XGBoost

### Sélection des features — identique MyQTM

```python
# Méthode MyQTM : importance = mean / std^power (débruite les features instables)
# Reprise à l'identique depuis train.py
feature_importance = mean_importance / (std_importance ** mean_std_power)
# Seuil : top K% features → sous-ensemble débruité pour le modèle final
```

### Intervalles d'entraînement — identique MyQTM

```python
# data_transform_split_intervals.py : fenêtres train/test interlacées
# Permet de valider OOS sans look-ahead bias
# Paramètre TS_SIZE = 6 (fenêtres glissantes de 6 mois)
```

### Supervision des labels

```python
# Label = performance relative future vs univers complet (continu, pas de seuil binaire)
# 1 seul label universel — le modèle apprend lui-même l'horizon selon le régime

y[etf_i] = ret_20d[etf_i] - mean(ret_20d[all_universe])

# Le régime est dans les features X → XGBoost apprend que :
#   en crisis_prob élevé → l'horizon effectif est plus court (les patterns 5j dominent)
#   en bull_prob élevé   → les patterns 20j momentum sont plus prédictifs

# Les poids ne sont jamais des labels — ils émergent du softmax sur les scores.
```

### Sortie du modèle : score → poids

```python
# XGBoost.predict() → score continu par ETF
#        ↓
# softmax(scores / temperature)  → probabilités
#        ↓
# × budget_régime  → poids finaux

def scores_to_weights(scores: dict, regime_probs: dict) -> dict:
    # 1. Clamp scores négatifs à 0 (pas de short sur ETF)
    scores_pos = {k: max(0, v) for k, v in scores.items()}

    # 2. Budget equity interpolé en continu depuis les probabilités HMM
    #    → pas de saut brutal entre régimes
    equity_budget = (
        0.90 * regime_probs["bull"] +
        0.60 * regime_probs["transition"] +
        0.30 * regime_probs["bear"] +
        0.10 * regime_probs["crisis"]
    )
    defensive_budget = 1.0 - equity_budget - cash_min

    # 3. Softmax à l'intérieur de chaque bloc
    equity_etfs    = [e for e in scores_pos if e.section in ("geo", "sector_us", "thematic")]
    defensive_etfs = [e for e in scores_pos if e.section in ("bond", "commodity")]
    w_equity    = softmax(scores_pos[equity_etfs],    temperature) * equity_budget
    w_defensive = softmax(scores_pos[defensive_etfs], temperature) * defensive_budget

    return {**w_equity, **w_defensive}
```

### Optimisation des paramètres — CMA-ES adapté

```python
# CMA-ES optimise (comme MyQTM search_params.py) :
#   - seuil de changement de poids (évite le churning)
#   - facteur de concentration (nombre effectif de positions)
#   - allocation equity_budget par régime (affinement)
#
# Paramètres à optimiser (9 comme MyQTM) :
INIT_SPACE = [
    Real(0.01, 0.50, name="min_weight_change"),    # seuil pour passer un ordre
    Real(0.05, 0.40, name="max_single_weight"),    # concentration max par ETF
    Real(0.50, 0.95, name="equity_budget_bull"),
    Real(0.20, 0.65, name="equity_budget_trans"),
    Real(0.05, 0.35, name="equity_budget_bear"),
    Real(0.00, 0.15, name="equity_budget_crisis"),
    Real(0.10, 0.50, name="defensive_min_bear"),
    Real(0.00, 0.20, name="cash_min"),
    Real(1.0,  3.0,  name="softmax_temperature"),  # concentration du softmax
]
```

---

## 6. Pipeline complet — comparaison MyQTM vs MyQTM-ETF

```
                MyQTM (actions)              MyQTM-ETF (ETF)
                ─────────────────────────    ────────────────────────────
DONNÉES        FMP bulk 300 actions          FMP bulk 41 ETF proxies
               news par action               news des top composants ETF
               fundamentals (P/E, bilan)     holdings + analyst consensus synthétique
               ~$50/mois                     ~$50 one-shot ✅

FEATURES       technique par action          technique + cross-ETF momentum
               sentiment par action          sentiment agrégé sur composants
               regime VIX                   regime VIX (identique)
               analyst notes directes        analyst notes synthétiques (Σ poids×note)
               —                            rotation de composition ETF
               —                            smart money (HY spread, 13F, put/call)

MODÈLE         XGBoost classify (-1/0/+1)   XGBoost regress (score continu)
               3 modèles par régime          1 seul modèle, régime = features
               feature select mean/std^p     feature select mean/std^p (identique)
               intervals OOS                 intervals OOS (identique)

OPTIMISATION   CMA-ES 9 seuils prob         CMA-ES 9 paramètres allocation
               sélection top N positions     allocation par budget de régime

SORTIE         3-4 positions long/short      vecteur de poids Σ ≤ 1
               signal binaire               répartition continue

PRODUCTION     FMP free 250 calls/jour      FMP free ~72 calls/jour ✅
               Finnhub free 60 calls/min    FRED gratuit + FMP free
               ~15 ordres/mois             ~15 rééquilibrages/mois
```

---

## 7. Workflows de production

### Signal daily (chaque soir)

```
18h00 : download prix EOD (FMP /quote bulk, 41 calls)
18h05 : download news composants (FMP, ~20 calls)
18h10 : scoring sentiment LLM (Gemini Flash, cache)
18h20 : download FRED (VIX, HY spread, yield curve) — gratuit, pas de quota
18h30 : mise à jour flow_proxies.parquet
18h35 : feature engineering (technique + régime + sentiment + smart money)
18h45 : prédiction XGBoost → scores par ETF
18h50 : allocation → poids cibles vs poids actuels
18h55 : génération liste des ordres (si |Δpoids| > seuil_min)
19h00 : notification (email / push)

Matin : passage manuel des ordres sur Boursobank (~10 min)
```

### Mise à jour hebdomadaire (lundi matin)

```
Analyst notes composants (FMP, ~20×3 = 60 calls)
Rotation de composition ETF (FMP /etf-holder, 41 calls)
→ features analyst_consensus + sector_rotation mis à jour
```

### Mise à jour mensuelle

```
Holdings complets ETF (FMP /etf-holder detail, 41 calls)
→ recalcul poids composants pour la pondération du sentiment
```

---

## 8. Coûts récapitulatifs

```
Initialisation (1 fois) :
  FMP bulk historique      ~$50    (1 mois payant)
  Infrastructure           $0      (local Python)

Production mensuelle :
  FMP free                 $0      (~72 calls/jour, quota 250)
  FRED API                 $0      (illimité, gratuit)
  CBOE put/call            $0      (public)
  LLM scoring news         ~$3/mois (Gemini Flash, ~100 news/jour × $0.0001)
  ─────────────────────────────────
  Total récurrent          ~$3/mois ✅

Frais trading (Boursorama) :
  ~15 ventes × 1000€ × 0.22%  = 33€/mois
  TER ETF moyen                = 0.20%/an
```

---

## 9. Phases de développement

### Phase 1 — MVP (données + régime + technique)
- `download_etf_data.py` ✅ fait
- `download_macro_data.py` ✅ fait (FRED + flow proxies)
- Feature engineering technique + régime VIX
- XGBoost simple sur scores techniques + régime
- Backtest VectorBT sur données disponibles

### Phase 2 — Holdings + analyst consensus
- `download_etf_holdings.py` : FMP /etf-holder mensuel
- Calcul analyst_consensus_etf, upgrade_momentum, price_target_upside
- Ajout features composition (sector_rotation_30d, top10_concentration)

### Phase 3 — Sentiment LLM
- Reprendre `data_transform_stock_news_to_sentiment_scores.py` de MyQTM
- Adapter pour news des composants ETF (Gemini Flash)
- Agrégation pondérée par poids dans l'ETF

### Phase 4 — Smart money + CMA-ES
- Put/Call ratio CBOE
- 13F institutional ownership FMP (trimestriel)
- CMA-ES sur paramètres d'allocation
- Optimisation anti-overfit (dropout régimes comme MyQTM)

### Phase 5 — Production
- Scheduler daily (APScheduler)
- Notification signal → ordres manuels Boursobank
- Monitoring : drift de régime, performance OOS, SHAP stability
