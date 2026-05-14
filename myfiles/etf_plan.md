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

**Architecture simplifiée — zéro FMP, zéro LLM, 100% gratuit.**

### 3a. Sources de données

```
SOURCE 1 — Yahoo Finance (yfinance, gratuit, sans clé API)
  download_ohlcv.py  [À CRÉER]
  → OHLCV journalier pour les 41 tickers (proxies US + UCITS européens)
  → yf.download(ticker, start="2000-01-01")
  → ./data/{TICKER}.parquet
  Couverture : tous tickers US + la majorité des UCITS (.AS, .PA, .DE)

SOURCE 2 — FRED (gratuit, clé API free à fred.stlouisfed.org)
  download_macro_data.py  ✅ déjà fait
  → VIX (VIXCLS), HY spread (BAMLH0A0HYM2), IG spread (BAMLC0A0CM)
  → Yield curve 10Y-2Y (T10Y2Y), DGS10, DGS2
  → WTI crude (DCOILWTICO), Gold LBMA (GOLDAMGBD228NLBM)
  → Dollar index (DTWEXBGS — broad USD index vs panier devises)
  → ./data/fred_{series}.parquet
  → ./data/flow_proxies.parquet  (dollar volume z-scores)

SOURCE 3 — iShares Data Download (gratuit, téléchargement manuel navigateur)
  parse_ishares_xls.py  [À CRÉER]
  Source : iShares.com → page produit → onglet "Data Download" → XLS
  Format : XML spreadsheet, sheet "Historical" :
    As Of | NAV per Share | Shares Outstanding | Non-FV NAV
  Couverture : 14 proxies iShares US, depuis inception :
    EWY  (2000-05-09)  EWT  (2000-06-23)  EWZ  (2000-07-14)
    EWC  (2000-01-03)  EWW  (2000-01-03)  ILF  (2001-10-26)
    TLT  (2002-07-26)  IEF  (2002-07-26)  TIP  (2003-12-05)
    HYG  (2007-04-11)  IEO  (2006-05-05)  RING (2012-02-02)
    TUR  (2008-03-28)  IBIT (2024-01-11)
  → ./data/ishares/{TICKER}_historical.parquet
  Note : ETF UCITS non couverts → dollar volume z-score en fallback
```

### 3b. Mise à jour journalière (100% gratuit)

```
18h00 : yfinance → prix EOD 41 ETFs          (gratuit, sans quota)
18h10 : FRED     → VIX, HY spread, yield curve (gratuit, illimité)
18h20 : calcul   → flow_proxies (dollar volume z-scores)
18h30 : feature engineering → prédiction → ordres

Mise à jour mensuelle :
  Téléchargement manuel XLS iShares (14 fichiers, ~5 min)
  → mise à jour shares outstanding
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

### 4b. Régime de marché — variables FRED brutes

XGBoost reçoit les variables macro brutes — pas de HMM, pas de régime encodé.
Il apprend lui-même les patterns de régime via les splits d'arbres.

```python
features_regime = {
    # VIX (peur / risk-off)
    "vix_level",             # niveau absolu
    "vix_velocity",          # Δvix 5j  (montée = danger, descente = signal achat)
    "vix_reversion_force",   # (vix_mean60 - vix) / vix_std60  (retour à la normale)

    # Crédit (stress systémique)
    "hy_spread",             # ICE BofA HY OAS (FRED BAMLH0A0HYM2)
    "hy_spread_z60",         # déviation vs 60j
    "hy_spread_velocity",    # Δspread 5j

    # Taux (cycle économique)
    "yield_curve",           # 10Y - 2Y (FRED T10Y2Y)
    "yield_curve_velocity",  # Δyield_curve 20j

    # Momentum macro global
    "ret_spx_20d",           # rendement S&P 500 sur 20j (via QQQ/CSPX proxy)

    # Dollar index — FRED DTWEXBGS (gratuit)
    # Impact direct sur ~20 ETFs : EM, or, obligations USD, matières premières
    "dxy_ret_20d",           # direction du dollar sur 20j
    "dxy_z60",               # déviation vs 60j (dollar fort/faible vs norme)
}
# → sigmoid(vix, hy_spread) → budget_régime (exposition equity vs défensif)
```

### 4c. Mouvements institutionnels (smart money)

```python
# Signal primaire : Shares Outstanding iShares (journalier depuis inception)
# Source : iShares.com "Data Download" → XLS → sheet "Historical"
# → ./data/ishares/{TICKER}_historical.parquet
#
shares_outstanding_z20 = z_score(Δshares_outstanding, 20d)
#   > 0 : création nette de parts  → inflow institutionnel  (bullish)
#   < 0 : rachat net de parts      → outflow institutionnel (bearish)
#
# Disponible : EWY, EWT, EWZ, EWC, EWW, ILF, TLT, IEF, HYG, TIP,
#              IEO, RING, TUR, IBIT  (14 proxies iShares US)
# Fallback    : dollar_volume_z20 pour les ETF UCITS non couverts

# Signal secondaire : dollar volume z-score (tous ETFs, yfinance)
dollar_vol_z20 = (close × volume - mean_20d) / std_20d
rotation_z60   = dvol_risk_on_avg - dvol_risk_off_avg   # z-scoré 60j

features_smart_money = {
    "shares_outstanding_z20",  # flux nets iShares (signal directionnel) ✅
    "dollar_vol_z20",           # intensité volume (signal d'activité)
    "rotation_z60",             # rotation risk-on vs risk-off
}
```

**Logique de décision :**
```
Macro apaisée (VIX bas + HY spread normal)
  + Rotation institutionnelle (shares_outstanding_z20 > 1.5 sur un ETF)
  + Technique favorable (RSI < 65 + MA slope positive)
  → XGBoost score élevé → poids fort dans le portefeuille
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
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",      # momentum multi-horizon
    "vol_20d", "vol_60d",                            # volatilité réalisée
    "rsi_14",                                        # RSI (surachat/survente)
    "ma_20_slope", "ma_50_slope",                    # direction tendance
    "price_vs_ma50", "price_vs_ma200",               # position vs moyennes mobiles
    "volume_z20",                                    # activité volume ETF

    # --- Régime macro brut — FRED (§4b) ---
    "vix_level",             # niveau absolu de la peur
    "vix_velocity",          # Δvix 5j
    "vix_reversion_force",   # retour à la moyenne (Ornstein-Uhlenbeck)
    "hy_spread",             # stress crédit absolu
    "hy_spread_z60",         # déviation vs 60j
    "hy_spread_velocity",    # accélération du stress
    "yield_curve",           # 10Y - 2Y  (inversion = récession)
    "yield_curve_velocity",  # Δyield_curve 20j
    "ret_spx_20d",           # momentum macro global
    "dxy_ret_20d",           # dollar index (FRED DTWEXBGS) — force USD
    "dxy_z60",               # déviation dollar vs 60j

    # --- Momentum cross-sectionnel (relatif au bloc) ---
    "ret_20d_z_within_block",  # z-score de ret_20d au sein du bloc (geo/sector/bond...)
    "ret_5d_z_within_block",   # idem court terme
    # → capte la ROTATION intra-bloc : "Corée surperforme les autres géo ?"
    # → XGBoost voit déjà tous les ETFs ensemble mais ce z-score rend explicite
    #   le signal de rotation que le modèle cherche à prédire

    # --- Mouvements institutionnels (§4c) ---
    "shares_outstanding_z20", # flux nets iShares (14 proxies US)
    "dollar_vol_z20",          # intensité volume (fallback UCITS)
    "rotation_z60",            # rotation risk-on vs risk-off
]

score_per_etf = model.predict(X_today)
# XGBoost apprend les interactions multi-actifs :
#   DXY fort + VIX monte          → EM (IEMA, CSKR, IBZL) score faible
#   Or monte + DXY baisse         → IGLN score élevé, IUIT score faible
#   Pétrole monte + yield_curve > 0 → IUES, IOGP, IBZL (Brésil) scores élevés
#   ret_20d_z_within_block élevé  → ETF en tête de rotation dans son bloc
# SHAP permet de visualiser ces régimes implicites a posteriori
```

**Avantages :**
- Aucun leakage (variables macro observables à t, pas de modèle intermédiaire)
- ~20 features pré-sélectionnées par la connaissance métier → pas besoin de sélection automatique
- SHAP révèle les régimes capturés — interprétabilité complète
- Régularisation assurée par early stopping + hyperparamètres XGBoost (max_depth=3, min_child_weight)

### Entraînement avec early stopping — split entrelacé 50/50

Sur chaque step du Walk-Forward, la **période d'entraînement** est divisée en
blocs alternés (~1 mois chacun, BLOCK_ROWS = 21 jours) :

```
PÉRIODE D'ENTRAÎNEMENT [2010 → 2022]  — blocs de 21 jours alternés :
  Blocs pairs  (jan, mar, mai …) → XGBoost train
  Blocs impairs (fév, avr, jun …) → Early stop + CMA-ES

PÉRIODE DE TEST [2022 → 2022.5]  → backtest OOS ✅ (jamais touché)
```

**Pourquoi entrelacé plutôt que premier/dernier 50% ?**
- Les blocs val couvrent **toutes les phases de marché** de la période d'entraînement
  (crise 2020, bull 2017, correction 2022…) — pas seulement la période récente
- Les scores XGBoost sur les blocs val sont **non biaisés** : le modèle n'a jamais
  vu ces blocs → les probabilités reflètent le comportement OOS réel du modèle
- CMA-ES optimisé sur ces scores non biaisés **généralise mieux** au test futur
- Si val = dernière année seulement, CMA-ES serait biaisé vers les conditions récentes

**3 zones, 3 rôles distincts :**
```
Blocs pairs  (50% de la période train)  → XGBoost training + early stopping
Blocs impairs (50% de la période train) → CMA-ES optimization (scores OOS du modèle)
Zone test walk-forward (future)         → backtest OOS réel avec params CMA-ES
```

### Backtest Walk-Forward Expanding

```
Paramètres :
  MIN_TRAIN_ROWS = 750   # ~3 ans de données avant premier step
  TEST_WINDOW    = 125   # ~6 mois de test par step (non-overlapping)
  STEP           = 125   # refit tous les 6 mois
  BLOCK_ROWS     = 21    # taille des blocs alternés (~1 mois)

Step 1 : train [2010-2013]  val_blocs entrelacés  | cmaes → params₁ | test [2013-2013.5]
Step 2 : train [2010-2013.5] val_blocs entrelacés | cmaes → params₂ | test [2013.5-2014]
Step 3 : train [2010-2014]  val_blocs entrelacés  | cmaes → params₃ | test [2014-2014.5]
...
Step N : train [2010-2025]  val_blocs entrelacés  | cmaes → paramsN | test [2025-2025.5]

Backtest OOS continu = concaténation des zones test (non-overlapping)
→ courbe d'équité sur ~12 ans ✅
→ chaque step a ses propres params CMA-ES → adaptation au régime courant

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

### Optimisation des paramètres — CMA-ES par step walk-forward

CMA-ES tourne **une fois par step walk-forward**, sur les blocs val (scores OOS non biaisés).
Les params trouvés sont ensuite appliqués uniquement à la zone test de ce step.
Pas de CMA-ES global sur toutes les données (ce serait du leakage).

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
  yfinance OHLCV           $0      (gratuit, sans clé API)
  FRED API key             $0      (gratuit à fred.stlouisfed.org)
  iShares XLS (14 fichiers)$0      (téléchargement manuel ~5 min)
  Infrastructure           $0      (local Python)
  ─────────────────────────────────
  Total initialisation     $0 ✅

Production mensuelle :
  yfinance prix EOD        $0      (gratuit, sans quota)
  FRED API                 $0      (illimité, gratuit)
  iShares XLS update       $0      (téléchargement manuel mensuel ~5 min)
  LLM / news               $0      (supprimé — ratio signal/bruit défavorable)
  ─────────────────────────────────
  Total récurrent          $0/mois ✅

Frais trading (Boursorama) :
  ~15 ventes × 1000€ × 0.22%  = 33€/mois
  TER ETF moyen                = 0.20%/an
```

---

## 9. Phases de développement

### Phase 1 — Données + feature engineering
- `download_ohlcv.py` : yfinance → OHLCV 41 tickers  [À CRÉER]
- `download_macro_data.py` ✅ fait (FRED + flow proxies)
- `parse_ishares_xls.py` : parser XLS iShares → shares outstanding parquet  [À CRÉER]
- Feature engineering : technique (RSI, MA, momentum, volume) + régime FRED

### Phase 2 — Modèle + backtest intégré CMA-ES
- XGBoost avec features 3 piliers (technique + régime FRED + smart money SO)
- Split entrelacé 50/50 (blocs de 21j) : blocs pairs=train, blocs impairs=val/CMA-ES
- Early stopping sur les blocs val (même split que CMA-ES)
- Walk-Forward Expanding : MIN_TRAIN=750j, TEST=125j, STEP=125j, BLOCK=21j
- CMA-ES par step sur les blocs val (scores OOS non biaisés, couvrant toutes les phases)
- Backtest OOS = concaténation des zones test avec params CMA-ES de chaque step
- Courbe d'équité vraiment OOS sur ~12 ans ✅

### Phase 4 — Production
- Scheduler daily (APScheduler)
- Notification signal → ordres manuels Boursobank
- Monitoring : drift de régime, performance OOS, SHAP stability
