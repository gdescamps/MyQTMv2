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

### Architecture — 1 seul XGBoost, pas de HMM

Le régime n'est pas modélisé explicitement. XGBoost reçoit les **variables macro brutes**
et apprend lui-même les patterns de régime via les splits d'arbres. Pas de leakage,
pas de modèle intermédiaire, interprétabilité via SHAP.

```python
# 1 seul XGBRegressor — pas de HMM, pas de router
model = XGBRegressor(max_depth=3, min_child_weight=50, ...)

# Features d'entrée = technique + sentiment + smart_money + MACRO BRUT
X = [
    # --- Technique par ETF (§4a) ---
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "rsi_14", "ma_20_slope", "ma_50_slope",
    "price_vs_ma50", "price_vs_ma200", "volume_z20",

    # --- Macro brut (capte le régime implicitement) ---
    "vix_level",             # niveau absolu de la peur
    "vix_velocity",          # Δvix 5j  (montée = danger, descente = signal achat)
    "vix_reversion_force",   # (vix_mean - vix) / vix_std  (Ornstein-Uhlenbeck)
    "hy_spread",             # stress crédit absolu
    "hy_spread_z60",         # déviation vs 60j
    "yield_curve",           # 10Y - 2Y  (inversion = récession)
    "yield_curve_velocity",  # Δyield_curve 20j
    "ret_spx_20d",           # momentum macro (QQQ proxy)

    # --- Sentiment agrégé (§4e) ---
    "sentiment_weighted", "sentiment_zscore_20d", "news_volume_z20",

    # --- Smart money (§4f) ---
    "rotation_z60", "put_call_z20", "hy_spread_velocity",

    # --- Composition ETF (§4d) ---
    "analyst_consensus_etf", "sector_rotation_30d", "top10_concentration",
]

score_per_etf = model.predict(X_today)
# XGBoost découvre seul :
#   vix > 50 + hy_spread_z60 > 2  → scores défensifs élevés
#   vix_velocity < 0 + vix > 40   → signal retournement, acheter equity
#   yield_curve < 0                → surpondérer bonds
# SHAP permet de visualiser ces régimes implicites a posteriori
```

**Avantages :**
- Aucun leakage (variables macro observables à t, pas de modèle intermédiaire)
- Moins de complexité (pas de HMM à entraîner en expanding window)
- SHAP révèle les régimes capturés — interprétabilité équivalente
- Phase 2 optionnelle : ajouter HMM si performances OOS insuffisantes

### Débruitage des features — identique MyQTM (intra-zone train)

Sur chaque zone train du Walk-Forward, 3 sous-fenêtres entrelacées :

```
ZONE TRAIN [2010 → 2022]
  ├── sf1 : train [2010-2014] → eval [2014-2016]  → importance_sf1
  ├── sf2 : train [2012-2016] → eval [2016-2018]  → importance_sf2
  └── sf3 : train [2014-2018] → eval [2018-2020]  → importance_sf3

# Méthode MyQTM (reprise à l'identique depuis train.py)
feature_score = mean(importance_sf1, importance_sf2, importance_sf3) \
              / std(importance_sf1,  importance_sf2, importance_sf3) ** mean_std_power

# Feature stable sur les 3 sous-fenêtres → vrai signal
# Feature importante sur 1 seule        → bruit ou artefact de régime → exclue
selected_features = top_k_percent(feature_score)
```

### Entraînement final avec early stopping

Une fois les features sélectionnées, le modèle final s'entraîne sur la zone train
avec une **zone de validation réservée** pour le early stopping :

```
ZONE TRAIN [2010 → 2022]
  ├── train early stop  : [2010 → 2021]   ← features débruitées seulement
  └── val   early stop  : [2021 → 2022]   ← jamais vu pour la sélection features
        → stoppe quand val_loss stagne (EvalF1Callback repris de MyQTM)

ZONE TEST [2022 → 2023]  → backtest OOS ✅ (jamais touché)
```

**3 zones, 3 rôles distincts — pas de leakage :**
```
Sous-fenêtres sf1/sf2/sf3  [2010-2020]  → débruitage features
Early stop val              [2021-2022]  → régularisation XGBoost
Zone test Walk-Forward      [2022-2023]  → backtest OOS réel
```

### Backtest Walk-Forward Expanding

```
Paramètres :
  MIN_TRAIN_YEARS = 3    # historique minimum avant premier step
  TEST_WINDOW     = 1    # 1 an de test par step
  STEP            = 6    # refit tous les 6 mois

Step 1 : train [2010-2013] early_stop [2013]  | test [2013-2014]
Step 2 : train [2010-2014] early_stop [2014]  | test [2014-2015]
Step 3 : train [2010-2015] early_stop [2015]  | test [2015-2016]
...
Step N : train [2010-2025] early_stop [2025]  | test [2025-2026]

Backtest OOS continu = concaténation des zones test
→ courbe d'équité lisse sur 13 ans ✅
→ chaque step refait la sélection de features (features peuvent changer)
→ cohérent avec la production (expanding window = tout l'historique disponible)

Note : ETFs récents (SEMI.AS depuis 2021, AINF.PA depuis 2024)
→ entrent dans l'univers progressivement quand leur historique est suffisant
→ pas de look-ahead sur leur existence
```

### Supervision des labels

```python
# Label = performance relative future vs univers complet (continu, pas de seuil binaire)
# 1 seul label universel — le modèle apprend lui-même l'horizon selon le régime

y[etf_i] = ret_20d[etf_i] - mean(ret_20d[all_universe])

# Le régime est capté implicitement via vix_level, hy_spread, yield_curve → XGBoost apprend :
#   vix élevé + hy_spread élevé → patterns défensifs courts (5j) dominent
#   vix bas + momentum positif  → patterns momentum longs (20j) dominent

# Les poids ne sont jamais des labels — ils émergent du softmax sur les scores.
```

### Deux fonctions indépendantes : rotation + exposition

Le modèle final combine deux calculs séparés qui répondent à deux questions distinctes :

```
XGBoost(X)          →  score par ETF        "quelle ROTATION à l'intérieur d'un bloc ?"
sigmoid(vix, hy)    →  budget_régime        "COMBIEN investir en equity vs défensif ?"
```

```python
# ── Fonction 1 : XGBoost ──────────────────────────────────────────────────
# Répond : parmi les ETF equity, lequel surperforme ?
#          parmi les défensifs, lequel ?
scores = xgboost.predict(X)   # score continu par ETF, toutes classes d'actifs

# ── Fonction 2 : budget_régime (sigmoid) ─────────────────────────────────
# Répond : quelle fraction du capital on risque aujourd'hui ?
# Indépendant de XGBoost — protection du capital avant tout
danger        = sigmoid(p1 * vix + p2 * hy_spread_z60 + p3)  # [0, 1]
equity_budget = equity_max * (1 - danger)   # ex. vix=60 → danger≈0.9 → 9% equity

# ── Combinaison finale ────────────────────────────────────────────────────
equity_etfs    = [e for e in scores if e.section in ("geo", "sector_us", "thematic")]
defensive_etfs = [e for e in scores if e.section in ("bond", "commodity")]

w_equity    = softmax(scores[equity_etfs],    temperature) * equity_budget
w_defensive = softmax(scores[defensive_etfs], temperature) * (1 - equity_budget - cash_min)
# cash implicite = 1 - Σw
```

**Pourquoi séparer ?**
Si XGBoost gérait les deux, il pourrait allouer 90% equity même en crise
(il optimise la rotation relative, pas l'exposition globale).
Le sigmoid est un **disjoncteur macro** : VIX explose → exposition réduite
indépendamment de ce que XGBoost pense de la rotation.

### Optimisation des paramètres — CMA-ES adapté

CMA-ES optimise conjointement les deux fonctions (comme MyQTM `search_params.py`) :

```python
INIT_SPACE = [
    # budget_régime (sigmoid)
    Real(-0.10, 0.00, name="p1_vix"),           # sensibilité au VIX
    Real(-0.50, 0.00, name="p2_hy_spread"),     # sensibilité au HY spread
    Real(-2.0,  2.0,  name="p3_bias"),          # biais du sigmoid
    Real(0.60,  0.95, name="equity_max"),        # exposition max en bull
    Real(0.00,  0.20, name="cash_min"),          # cash minimum garanti

    # softmax XGBoost
    Real(0.5,   3.0,  name="temperature"),       # concentration des poids
    Real(0.05,  0.40, name="max_single_weight"), # cap par ETF

    # gestion des ordres
    Real(0.01,  0.10, name="min_weight_change"), # seuil pour passer un ordre
    Real(0.10,  0.50, name="defensive_min"),     # défensif minimum si danger > 0.5
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
               3 modèles par régime          1 seul modèle, pas de HMM
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

### Démarrage à froid (capital 100% cash)

Le sigmoid gère le démarrage automatiquement — pas de logique spéciale nécessaire.

```python
# Jour 0 : portefeuille = 100% cash, poids actuels = {}

# Le modèle tourne normalement chaque soir :
danger        = sigmoid(p1 * vix + p2 * hy_spread_z60 + p3)
equity_budget = equity_max * (1 - danger)

# Cas 1 — conditions favorables (vix=18, hy_spread normal)
#   danger ≈ 0.10  →  equity_budget ≈ 85%
#   → allocation complète dès le premier soir ✅

# Cas 2 — conditions dégradées (vix=55, hy_spread élevé)
#   danger ≈ 0.90  →  equity_budget ≈ 6%
#   → modèle dit "reste quasi-cash" jusqu'à normalisation ✅

# Dans les deux cas : si |poids_cible - poids_actuel| > min_weight_change
#   → ordre généré  (filtre anti-churning CMA-ES)
# Sinon → HOLD, on attend le lendemain
```

Le démarrage est donc **conditionnel aux conditions macro du jour J** :
- Bon timing (VIX bas) → allocation complète en 1 jour, 1 seule session d'ordres
- Mauvais timing (VIX haut) → exposition progressive au fur et à mesure que le VIX baisse

### Rééquilibrage dynamique (fonctionnement normal)

En fonctionnement normal, deux sources génèrent des ordres chaque soir :

```python
# Source 1 : dérive naturelle des prix (profit-taking automatique)
#   ETF monte → poids actuel dépasse le poids cible → VENDRE le surplus
#   ETF baisse → poids actuel sous le poids cible   → ACHETER le manque

poids_actuel[etf] = valeur_position[etf] / valeur_portefeuille_total

# Source 2 : changement de signal XGBoost
#   Nouveau score → nouveau poids cible → delta à exécuter

# Filtre anti-churning : seuil calibré par CMA-ES (~3-5%)
for etf in universe:
    delta = poids_cible[etf] - poids_actuel[etf]
    if abs(delta) > min_weight_change:
        ordre(etf, delta)   # + acheter, - vendre
    # sinon : HOLD — la dérive est trop faible vs frais Boursorama (0.22%)
```

**Exemple concret :**
```
Jour J   : CSP1 poids cible = 25%, poids actuel = 25% → rien
Jour J+10: CSP1 monte +8%  → poids actuel = 27%
           poids cible inchangé = 25%
           delta = -2%  → si > min_weight_change → VENDRE 2% de CSP1
           → profit-taking automatique ✅

Jour J+10: sigmoid(vix↑) → equity_budget réduit de 85% à 60%
           tous les poids equity cibles baissent proportionnellement
           → ordres de vente générés sur tout le bloc equity ✅
```

**Le `min_weight_change` est le seul frein aux ordres** — CMA-ES l'optimise
pour que le gain espéré du rééquilibrage dépasse toujours les frais de vente (~0.22%).

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
