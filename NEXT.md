# NEXT.md — Observations & prochaines étapes

## Observation critique : sensibilité au décalage temporel

Changer `MIN_TRAIN_ROWS` de 5031 à 4334 décale les frontières de steps de **4 jours**.
Ce décalage minime provoque :
- La feature selection WF voit des données marginalement différentes
- Les modèles divergent (effet papillon)
- Les résultats post-2019 changent significativement (ex: 2020 passe de +36% à -5%)

**Implication** : le modèle est sensible aux petites perturbations temporelles.
Le seed-averaging (5 seeds) réduit le bruit XGBoost mais ne corrige pas ce bruit structurel lié aux frontières de steps.

**Piste** : rendre les frontières de steps déterministes par rapport aux dates calendaires
(ex: toujours commencer un step le 1er du mois) plutôt que par comptage de lignes.

---

## Cycle de marché : phases 1-2-3

Le modèle performe selon un cycle en 3 phases :

### Phase 1 — Choc
- Événement exogène majeur (COVID, Ukraine, tarifs Trump)
- Corrélation inter-ETFs spike (tout baisse ensemble)
- Le modèle n'a pas de signal
- **Détection** : VIX spike (déjà géré par VIX-adaptive)

### Phase 2 — Rotation / Digestion
- Le marché différencie gagnants/perdants du choc
- L'argent se déplace entre secteurs/régions
- Corrélation inter-ETFs descend (dispersion croissante)
- **Le modèle performe le mieux** (2019, 2020 post-COVID, 2021, 2022, 2025)

### Phase 3 — Absorption / Habituation
- Le marché a intégré le nouveau paradigme
- Tout remonte ensemble ("the new normal", beta rally)
- Corrélation inter-ETFs remonte
- Le modèle patine (2017, 2018, 2023, 2024)
- **Transition inévitable** si pas de nouveau choc

### Transitions
| Transition | Prévisible ? | Indicateur |
|---|---|---|
| 3 → 1 | Non (choc exogène), mais **détectable en temps réel** | VIX spike + corrélation qui bondit |
| 1 → 2 | Oui | Corrélation commence à baisser (pente négative) |
| 2 → 3 | Oui | Corrélation remonte (pente redevient positive) |
| Rester en 3 | Inévitable sans nouveau choc | Corrélation haute et stable |

---

## Idée : oscillateur de régime (corrélation inter-ETFs)

### Construction
1. Calculer la **corrélation moyenne pairwise** des rendements rolling 60j des 27 ETFs → une série temporelle
2. Calculer la **pente** de cette série (régression linéaire 20j ou diff lissé)
3. Le signe de la pente détermine le régime :
   - **Pente négative** = phase 2 (dispersion croissante) → modèle actif
   - **Pente positive** = phase 3 (convergence) → réduire ou switcher

### Action selon la phase
- **Phase 2** (pente corrélation négative) : modèle actif, rotation ETFs
- **Phase 3** (pente corrélation positive/plate) : switch sur MSCI World (IWDA/URTH) pour capter le beta rally
- **Phase 1** (VIX spike) : cash/réduction (VIX-adaptive existant)

### Avantage du MSCI World en phase 3
En phase 3, le modèle de rotation ne gagne rien mais coûte des frais.
Un MSCI World capterait le beta (+15-20%/an) sans frais de rotation.
Ça transformerait les années 2023 (-10%) et 2024 (+11%) en ~+20%.

### Points de vigilance
- L'indicateur ne doit pas flip-flopper (lissage suffisant)
- Le timing 2→3 et 3→2 doit être assez propre pour ne pas rater les débuts de rotation
- L'oscillateur doit être validé visuellement contre l'equity curve avant implémentation
- Risque d'overfitting : seulement ~9 années de données

---

## Robustesse temporelle : test par décalage de MIN_TRAIN_ROWS

En plus du seed-averaging (bruit XGBoost) et du test de robustesse par ETF dropout,
tester la **sensibilité au décalage temporel** :

- Varier `MIN_TRAIN_ROWS` de ±100 autour de 4334 (ex: 4234, 4284, 4334, 4384, 4434)
- Chaque décalage produit des frontières de steps différentes (~4-5 jours de shift)
- Mesurer l'impact sur le rendement annuel et le drawdown
- Si les résultats sont stables → le modèle est robuste structurellement
- Si 2020 oscille entre -5% et +36% → le modèle est fragile sur cette période

L'objectif est de quantifier le **bruit structurel** (au-delà du seed noise)
et potentiellement moyenner les prédictions sur plusieurs décalages
(comme on moyenne sur les seeds).

---

## Optuna LGB (en cours)

Recherche des hyperparamètres LightGBM optimaux pour l'ensemble XGB+LGB.
60 trials, chaque trial = 1 walk-forward complet (~115 steps, 1 seed, model A only).
Script : `search_lgb_params.py`
Résultats : `myfiles/search_lgb_results.csv`

---

## Résultats par année (5 seeds, XGB+LGB, depuis 2017)

| Année | Retour brut | Phase |
|-------|------------|-------|
| 2017  | +18.7%     | 3     |
| 2018  | -11.5%     | 1     |
| 2019  | +42.0%     | 2     |
| 2020  | -4.7%      | 1→2 (*)  |
| 2021  | +37.7%     | 2     |
| 2022  | +99.1%     | 2     |
| 2023  | -10.0%     | 3     |
| 2024  | +11.1%     | 3     |
| 2025  | +43.7%     | 2     |
| 2026  | +40.6%     | 2 (partiel) |

(*) 2020 impacté par le décalage de 4 jours — résultat non fiable, voir observation ci-dessus.

Robustesse : médian Sharpe 0.94, DD -33.6%
Equity : Sharpe 1.02, net +14.5%/an, DD -33.8%
