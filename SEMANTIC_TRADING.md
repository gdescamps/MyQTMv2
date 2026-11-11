# SEMANTIC_TRADING.md — stratégie recherchée

> **Statut : recherche** (branche `dev_sementic_trading`). Rien de ce document n'est sur le chemin de production.
> La stratégie déployée reste la formule déterministe x1.7 PUST+LQQ décrite dans `README.md` / `CLAUDE.md`.
> Bibliographie et notes de lecture : `myfiles/papers/README.md` (section 4).

## 1. Idée

Faire piloter le **niveau d'allocation** du portefeuille par un LLM qui tient à jour, jour après jour, un
**contexte sémantique macro glissant** : ce qui anime le marché en ce moment, les thèmes en cours, les
échéances qui approchent et la confiance à accorder à ceux qui parlent.

Le LLM ne prédit pas un cours. Il maintient un **état du monde** lisible, et l'allocation se déduit de cet état.

Thèmes analysés :

- inflation, taux directeurs ;
- calendrier économique ;
- calendrier des résultats ;
- news ;
- indicateurs techniques.

### Pondération du Nasdaq : observer les fortes capitalisations

Le portefeuille est exposé au **Nasdaq-100** (PUST + LQQ), indice pondéré par la capitalisation et très
concentré : une poignée de très grandes capitalisations fait l'essentiel de son mouvement. La
surveillance doit donc suivre cette pondération :

- une information compte **en proportion du poids dans l'indice** du titre qu'elle touche ;
- les plus fortes capitalisations sont suivies **individuellement** (news, résultats, prévisions,
  valorisation, technique), le reste de l'indice de façon agrégée ;
- les poids changent : la liste suivie et ses pondérations sont recalculées régulièrement, pas figées.

Conséquence sur chaque étape : les thèmes sont jugés par leur effet sur ces leaders (étapes 2 et 3),
leurs dirigeants font partie des acteurs notés (étape 4), leurs résultats sont les échéances
d'entreprise prioritaires (étape 5), et leur valorisation pondérée entre dans l'allocation (étape 6).

## 2. Pipeline quotidien

```
 news du jour ─┐
 calendriers ──┼─► 1. contexte macro ─► 2. thèmes récurrents ─┐
 macro/prix ───┘        (glissant)      3. thèmes dynamiques ─┼─► 6. allocation (0 → 100 %)
                                        4. fiabilité acteurs ─┤
                                        5. échéances à venir ─┘
```

### Étape 1 — Mise à jour du contexte sémantique global

Un document court, réécrit chaque soir à partir du contexte de la veille et des informations du jour.
Il décrit le régime en cours : ce que le marché surveille, ce qu'il craint, ce qu'il a déjà intégré.

- Entrées : contexte J−1, news du jour, publications macro du jour, résultats du jour, état technique.
- Sortie : contexte J (texte structuré, taille bornée), plus la liste de ce qui a changé.
- Le contexte est **glissant** : une information ancienne s'efface si rien ne la ravive.

### Étape 2 — Surveillance des thèmes récurrents

Liste fixe de thèmes suivis en permanence, chacun avec un état (intensité, direction, tendance) :

- inflation ;
- consommation ;
- droits de douane ;
- transport ;
- guerre.

(Liste à compléter : emploi, crédit, énergie, banque centrale, immobilier…)

### Étape 3 — Surveillance des thèmes dynamiques

Thèmes ou **scénarios générés à partir du contexte** : le LLM propose lui-même ce qu'il faut surveiller
(« bulle IA », « crise bancaire régionale », « shutdown »…), les suit tant qu'ils sont actifs, puis les clôt.

- Chaque thème dynamique porte : hypothèse, signes qui la confirment, signes qui l'infirment, date d'ouverture.
- Un thème qui dure peut être promu en thème récurrent.

### Étape 4 — Fiabilité des acteurs

Un score de confiance par acteur (banque centrale, gouvernement, dirigeant, agence, média, analyste) :

- construit sur l'historique : ce que l'acteur a annoncé, comparé à ce qui s'est produit ;
- utilisé pour pondérer ses déclarations dans le contexte (une menace de droits de douane souvent
  retirée pèse moins qu'une décision de la Fed).

### Étape 5 — Échéances à venir à court terme

Agenda des prochains jours et semaines, avec pour chaque échéance : date, importance, attente du
consensus, et scénarios possibles.

- calendrier économique : CPI, emploi, réunions de banque centrale, PIB… ;
- calendrier des résultats : les fortes capitalisations du Nasdaq-100, chaque publication pesée par
  le poids du titre dans l'indice ;
- événements datés : élections, échéances de droits de douane, plafond de la dette.

### Étape 6 — Niveau d'allocation

L'allocation du portefeuille est déterminée en fonction de :

| Famille | Contenu |
|---|---|
| **Contexte économique** | inflation, douane, Fed, taux, spread de crédit |
| **Événements** | élections, pandémie, attentats, guerre, mouvements sociaux |
| **Valorisation** | investissement, revenu, prix, technique — mesurés sur les fortes capitalisations, pondérés par leur poids dans le Nasdaq |
| **Surprises d'échéance** | écart entre le publié et l'attendu sur les échéances passées |
| **Échéances à venir** | risque concentré sur les prochains jours |

## 3. État de la recherche (2026-11-10)

Aucun papier trouvé ne réalise ce pipeline de bout en bout. Détail dans `myfiles/papers/README.md`.

| Étape | Couverture dans la littérature | Références |
|---|---|---|
| 1. Contexte glissant | Partielle | Cheng & Chin (note macro/micro réécrite), FinMem, EdgeQuant |
| 2. Thèmes récurrents | Bonne | Trading Macro Narratives (65 narratifs, 2004–2025) |
| 3. Thèmes dynamiques | Faible (classification, pas découverte) | TrustMe (CLEF 2026) |
| 4. Fiabilité des acteurs | Faible (consensus entre agents, notation de médias) | TrustTrade, Loru et al. |
| 5. Échéances et surprises | **Aucune** | — |
| 6. Allocation | Bonne, mais sectorielle ou par ETF | Quek et al., He et al., CN-Buzz2Portfolio |

Les étapes 4 et 5 sont donc la partie originale de l'idée.

## 4. Contraintes à respecter

Ces points viennent de la littérature et des expériences déjà faites dans ce dépôt.

1. **Pas de backtest historique naïf.** Un LLM moderne connaît 2008, 2020 et 2022 : son backtest sur ces
   dates est contaminé (Look-Ahead-Bench, FinCAD). Deux voies seulement :
   - suivi en temps réel après le cutoff du modèle ;
   - modèle point-in-time (ChronoGPT, PIT-4B), de qualité bien moindre.
2. **Les news sont souvent déjà dans le cours** (TrustMe). L'information exploitable en avance se trouve
   plutôt dans les échéances connues à l'avance et dans l'écart au consensus.
3. **Les stratégies LLM sont trop prudentes en hausse et trop agressives en baisse** (FINSABER). Le socle
   de tendance et de volatilité doit rester déterministe.
4. **Peu d'événements.** Les garde-fous macro actuels reposent déjà sur n ≈ 4 crises ; les modèles appris
   testés en 2026-11 (`myfiles/failed_xp/pust15_*`) ne battent pas la formule. Une couche sémantique aura
   le même problème de validation.
5. **Exécution inchangée** : signal calculé le soir, exécuté le lendemain matin (`exec_lag=1`), bande
   asymétrique et frais de 0,5 % par sens. Une couche qui fait bouger l'allocation souvent sera mangée
   par les frais.
6. **Coût LLM** : à chiffrer avant tout lancement sur l'historique (tokens × tarif).

## 5. Proposition de mise en œuvre

*(Proposition de Claude, à valider.)*

**Principe : la couche sémantique tourne d'abord en observation seule.** Elle écrit chaque soir son
contexte et son allocation suggérée dans un journal, sans effet sur `signal.json`. On la compare à la
formule sur données réelles avant de lui donner le moindre poids.

| Phase | Contenu | Critère de sortie |
|---|---|---|
| **0. Données** | EODHD → `data/eodhd/` : `news/`, `econ/` (calendrier économique), `earnings/` (calendrier des résultats). Vérifier que consensus et valeur publiée sont horodatés point-in-time. | Couverture et qualité connues sur 2011 → aujourd'hui |
| **1. Échéances et surprises, sans LLM** | Série quotidienne « surprise » (publié − consensus, normalisé) pour CPI, emploi, Fed, résultats des leaders ; série « risque d'agenda » des jours à venir. Test contre le rendement du QQQ à J+1…J+5. | Effet mesurable hors échantillon, avec placebos de dates |
| **2. Contexte glissant en temps réel** | Tâche du soir : étapes 1 à 5, sortie JSON structurée + journal daté. Aucune action. | 3 à 6 mois de journal sans trou |
| **3. Allocation suggérée** | Étape 6 comme **modulateur borné** de la formule (par exemple plafond d'exposition abaissé, jamais relevé au-dessus de `E_MAX`). Toujours en observation. | Comparaison journal ↔ formule sur la période vécue |
| **4. Décision** | Intégration éventuelle comme garde-fou supplémentaire, au même titre que NFCI et CPI. | Décision explicite, documentée dans `README.md` / `BOURSO.md` |

La phase 1 est la plus rentable à faire en premier : elle ne demande pas de LLM, se backteste
honnêtement (les surprises sont des nombres datés) et porte sur la brique que personne n'a publiée.

## 6. Questions ouvertes

- Univers : rester sur QQQ seul, ou étendre l'allocation à d'autres actifs ?
- La couche sémantique peut-elle **augmenter** l'exposition, ou seulement la réduire ?
- Fréquence de mise à jour : quotidienne, ou seulement les jours d'échéance ?
- Source des news en production : EODHD seul, ou plusieurs flux pour recouper (fiabilité) ?
- Comment noter la fiabilité d'un acteur sans connaître la suite des événements (point-in-time) ?
- Modèle : Claude en temps réel uniquement, ou aussi un modèle point-in-time pour rejouer l'historique ?
