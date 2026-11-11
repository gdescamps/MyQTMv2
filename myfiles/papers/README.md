# Papiers — analyse de news / sentiment financier par LLM

Notes de lecture de la session du 2026-11-09.

**Question de départ** : existe-t-il des travaux qui génèrent un **corpus synthétique de scénarios catastrophes** pour entraîner un modèle à **filtrer les news importantes**, dans un cadre **point-in-time** (sans fuite d'information) ?

**Réponse courte** : non, aucun papier trouvé ne le fait de bout en bout. Les briques existent séparément :

- générer des scénarios de risque : FIN-FORCE, GRID ;
- fabriquer un corpus synthétique : Distillation, RAG-augmentation ;
- apprendre à partir de la réaction réelle du marché : FinSMART, Kirtac ;
- disposer de modèles sans fuite d'information : ChronoBERT/GPT, PIT-4B de Kelly.

L'idée « corpus de crises généré en point-in-time, puis validé sur le marché » reste ouverte.

---

## 1. PDF de ce répertoire

| Fichier | Papier | Verdict |
|---|---|---|
| `Financial-Language-Models-…-Market-Frictions.pdf` | Kirtac (MFAST), 2026 | ⭐⭐⭐ Protocole d'évaluation à copier ; résultats à prendre avec prudence |
| `FinSMART_-….pdf` | Iacovides, Zhou, Mandic, 2026 | ⭐⭐⭐ Récompense tirée du marché, le gabarit le plus proche de l'idée |
| `Efficient-Financial-Language-Understanding-…-Synthetic-Data.pdf` | Huang & Simpson, 2026 | ⭐⭐ Recette pratique pour fabriquer un corpus synthétique |
| `fdef97ce-45f4-4a34-908a-cef27975985c.pdf` | Bisharat & Hean, *Evaluating Financial Sentiment in the Age of AI*, 2026 | ⭐⭐ Douche froide : le sentiment ne prédit pas le rendement du lendemain |
| `Beyond-Correlation_-….pdf` | van der Heever, Ong, Satapathy, Cambria, 2026 | ⭐ Méthode utile (tests de réfutation), fond très limité |
| `QuantEval_-….pdf` | Kang et al., 2026 | Hors sujet (benchmark de code quant par LLM) |
| `Revolutionizing-Finance-with-LLMs_-….pdf` | Zhao et al., revue v5 2026 | Faible valeur, panorama général |
| `Hybrid-Quantum-Classical-Ensemble-…-S_&P-500-….pdf` | Weinberg, 2025 | ❌ À ignorer |
| `A-Hybrid-Quantum-Classical-Framework-…-Volatility-….pdf` | Chen, 2026 | ❌ À ignorer |

**Ordre de lecture conseillé** : Kirtac, puis FinSMART, Distillation, Bisharat & Hean, Beyond Correlation.

### Kirtac — *Financial Language Models … News-Based Trading under Market Frictions* (MFAST)
arXiv 2609.23703, UCL/Warwick, en révision (« revise and resubmit ») à *Eng. Appl. of AI*.

- **Données** : news Refinitiv + CRSP, actions **US**, 2010 → janv. 2026.
  - 3,1 M de news au départ, 973 000 retenues (une seule entreprise par news, filtre anti-doublons, filtre de liquidité), pour 3 452 entreprises.
  - Découpage : entraînement 2010–2023, zone tampon janv.–mai 2024, **test juin 2024 – janv. 2026, donc après le cutoff de Llama-3**.
- **Entraînement** : classification **supervisée**, pas de RL.
  - Étiquette = signe du rendement relatif au marché sur 3 jours après publication.
  - Tête de classification sur l'état caché du dernier token. Llama-3-8B et OPT-1.3B en LoRA ; RoBERTa, BERT, FinBERT en fine-tuning complet.
  - Probabilités recalibrées sur la validation.
- **Entrée = texte seul** (titre + chapeau + corps), sans prix, identifiant ni horodatage.
  - Ajouter prix, liquidité ou métadonnées change peu : précision 0,787 → 0,803.
- **Pas de moyenne mobile** : seulement une moyenne pondérée par la récence des news d'une même entreprise le même jour. Le classement utilise ensuite la « probabilité la plus récente ».
- **Portefeuille** : sélection d'actions **long-short en quintiles** (quintile du haut long, du bas short).
  - Pondération par capitalisation, 5 bp de frais, participation ≤ 10 % du volume quotidien.
  - Le nombre d'actions par jambe n'est pas donné. Estimation : quelques dizaines à quelques centaines.
- **Résultats (test)** :
  - Llama-3 : précision 0,787, Sharpe long-short **2,85** (IC 2,31–3,34), +180 %, maxDD −12 %.
  - FinBERT : Sharpe 1,75. Dictionnaire Loughran-McDonald : 0,68.
  - Le résultat tient après corrections White, Hansen SPA et Sharpe dégonflé (p = 0,014).
  - **Réplication publique** (GDELT + Yahoo) bien plus faible : précision 0,61, Sharpe 0,92.
- **Points forts, la liste de contrôle à copier** :
  - Tester après le cutoff du LLM, avec une zone tampon.
  - Règle d'observabilité : news avant 9h30 tradable le jour même, sinon le lendemain.
  - Anti-doublons à sens unique dans le temps (cosinus > 0,80 sur 5 jours).
  - Audit des doublons aux frontières entraînement/validation/test.
  - Rendements de sortie de cote (delisting) inclus.
  - Placebos : dates mélangées, entreprises mélangées, signes permutés.
  - Coût de calcul chiffré : Llama-3 = 54 Go et 9 articles/s, contre 4,8 Go et 106 articles/s pour RoBERTa.
- **⚠️ Réserves** :
  - Une précision de 78,7 % à partir du texte seul est anormalement élevée ; les valeurs habituelles sont de 52 à 58 %.
  - **Fuite probable (hypothèse)** : pour une news publiée avant l'ouverture, l'étiquette inclut le rendement CRSP du jour e (clôture de la veille → clôture). Celui-ci contient le saut d'ouverture que la dépêche décrit souvent déjà (« shares jumped 12% premarket »).
  - Incohérences internes :
    - Nasdaq : +0,12 %/jour affiché, mais seulement +35 % cumulé.
    - Llama-3 : +0,34 %/jour et 1,89 % de volatilité donneraient environ +250–300 %, pas +180 %.
    - La volatilité du long-short est supérieure à celle du Nasdaq.
  - Auteur seul, résultats « trop propres ».
- Non reproductible sur un PEA (vente à découvert, actions US). Ce qui se transpose, c'est **le protocole**, pas la stratégie.

### FinSMART — *Financial Sentiment Analysis … through Market-Aligned RL*
arXiv 2607.28127, Imperial College.

- **Idée** : le modèle apprend de la **réaction réelle du marché**, pas d'étiquettes humaines.
- **Méthode** :
  - Llama-3-8B-Instruct + GRPO (RL) + LoRA, sur des news Motley Fool et MarketWatch (2015 → juin 2021) concernant les actions du S&P 500.
  - Filtrage : reconnaissance d'entité (score > 98 %), puis filtre de sentiment. Une news est gardée seulement si le token le plus probable du modèle d'origine est Positive, Negative ou Neutral.
  - **Récompense à double filtre** : bon sens du mouvement **et** écart supérieur à 0,5 % avec le SPY. Elle est asymétrique (+2 si juste, −1 si neutre, −1,5 si faux).
  - Entraînement : **génération de 8 réponses** par article, renforcement des meilleures.
  - Utilisation : une seule passe, puis lecture des **logits** des tokens de sentiment (« logit-to-score » de FinDPO), ce qui donne un score continu entre −1 et +1. Pas d'embedding.
- **Résultats** (test 2019 → juin 2021, long-short quotidien 35 % / 35 %, ouverture à ouverture) :
  - Sharpe 1,97, contre 1,12 pour FinDPO et 1,09 pour le S&P 500 ; +265 % cumulé.
  - Avec réentraînement tous les 6 mois sur fenêtre croissante : +406 %.
- **⚠️ Réserves** :
  - **Fuite par le pré-entraînement** : Llama-3 a vu des textes jusqu'en 2023, donc toute la période de test.
  - Pas de frais de transaction.
  - Sélection d'actions, pas du timing de marché.
- **Pour l'idée** : c'est le meilleur gabarit. Récompense = réaction du QQQ ou du VIX à J+1/J+5 sur des news macro, partir d'un checkpoint point-in-time, réentraîner sur fenêtre croissante. Avec ≈4 crises, le risque de surapprentissage est fort : commencer plutôt par embedding + régression régularisée.

### Huang & Simpson — *Efficient Financial Language Understanding via Distillation with Synthetic Data*
arXiv 2606.18875, Bristol.

- Recette :
  1. Partir de 12 à 105 phrases réelles étiquetées.
  2. Choisir ces graines par **clustering** de leurs embeddings Sentence-BERT, pour couvrir toute leur variété.
  3. GPT-4o génère des variantes (zéro-, un- et few-shot).
  4. Entraîner DistilBERT ou ModernBERT sur le corpus obtenu.
- Le choix des graines par clustering bat le tirage au hasard. Sur des tweets bruités, **l'élève bat GPT-4o** sans exemples.
- **Pour l'idée** : regrouper les vraies news de crise (2000, 2008, 2020, 2022) pour choisir des graines, générer autour, puis entraîner un petit classifieur local (cohérent avec ChronoBERT). Ne jamais regénérer à partir de données synthétiques (cf. Zhao).

### Bisharat & Hean — *Evaluating Financial Sentiment in the Age of AI* (fichier `fdef97ce-…`)
Brouillon du 17 juil. 2026, Loyola Chicago / North Dakota State.

- 12 modèles comparés (dictionnaires, FinBERT et variantes, Llama 3.2, Gemma 3, Qwen 2.5, Mistral, + Claude Opus 4.5) sur deux critères :
  - **validité linguistique** : le modèle reproduit-il les étiquettes humaines (Financial PhraseBank) ?
  - **validité économique** : son sentiment est-il lié aux annonces de résultats (formulaires 8-K) ?
- **Résultats** :
  - Les LLM généralistes égalent les modèles spécialisés en finance (Claude : 93,6 % de précision).
  - Une meilleure précision **ne donne pas** un meilleur signal économique.
  - Le sentiment est lié aux **surprises de résultats**, mais **aucun modèle n'est lié au rendement du lendemain**.
  - Le signal est fort seulement pour les grosses surprises.
- Échantillon minuscule : 221 annonces, 15 entreprises, 2022–2026.
- **Pour l'idée** : le texte porte de l'information surtout sur les **événements extrêmes**, ce qui va dans le sens d'un filtre « catastrophes uniquement ».

### van der Heever et al. — *Beyond Correlation: Refutation-Validated ABSA for Energy Market Returns*
arXiv 2603.21473, NTU/NUS (groupe Cambria, comme FIN-FORCE).

- Sentiment par thème sur 𝕏, rapporté aux rendements de 6 actions énergie sur **un seul trimestre**. Régressions OLS avec erreurs Newey-West.
- **Intérêt = la batterie de tests de réfutation** : placebo, cause commune aléatoire, stabilité sur sous-échantillons, bootstrap. **La plupart des corrélations sentiment-rendement disparaissent.**
- **Pour l'idée** : avec n≈4 crises, ces tests (placebo de dates, mélange des news) sont indispensables pour valider un filtre.

### Hors sujet ou à ignorer
- **QuantEval** (arXiv 2601.08689) : 1 575 questions testant les LLM sur les connaissances en finance quantitative, le calcul, et l'écriture de stratégies **évaluées dans un backtester** de type CTA. Les LLM restent nettement en dessous des experts. Utile seulement pour faire coder des stratégies par un LLM.
- **Revolutionizing Finance with LLMs** (arXiv 2401.11641 v5) : revue générale (trading, crédit, ESG, fraude) + tests qualitatifs de GPT-4. Rien sur le point-in-time ni l'évaluation économique.
- **Weinberg, Hybrid Quantum S&P 500** (arXiv 2512.15738) :
  - 60,1 % de précision directionnelle sur seulement **286 jours de test**.
  - Le « sentiment quantique » **n'utilise aucune news** : ce sont des variables de prix passées dans un circuit simulé sur ordinateur classique.
  - 9 modèles sur 35 sont retenus avec un seuil de 52 %, d'où un risque de sélection sur le test.
- **Chen, QCBM volatility** (arXiv 2603.09789) :
  - Indices chinois en 5 min, le **seul point de comparaison est un LSTM** (ni GARCH ni HAR).
  - « 113 % d'amélioration du QLIKE », ce qui est incohérent.
  - Seule idée à retenir : le « Drop-Prior » (on entraîne avec un module d'aide, puis on le retire en production).

---

## 2. Papiers vus en ligne (pas de PDF local)

| Papier | Résumé | Intérêt |
|---|---|---|
| **Kelly, Malamud, Schwab, Xu — *Scaling Point-in-Time Language Models*** ([arXiv 2607.11889](https://arxiv.org/abs/2607.11889), NBER w35247) | PIT-1.5B et PIT-4B entraînés sur FineWeb filtré par date (1T tokens), checkpoints annuels 2013–2024. Benchmarks : 58,1 contre 48,9 pour ChronoGPT et 69,2 pour Gemma-3-4B. Embeddings de dépêches Dow Jones + ridge MSRR → Sharpe ≈ 1–1,5 hors échantillon, proche des modèles qui voient le futur. **Aucun test explicite de fuite.** | Checkpoints PIT réutilisables : [code](https://github.com/DjoFE2021/Point-In-Time-LLM), [HF Diamegs](https://huggingface.co/Diamegs). Données Dow Jones et CRSP non fournies. |
| **He, Lv, Manela, Wu — *Chronologically Consistent LLMs*** ([arXiv 2502.21206](https://arxiv.org/abs/2502.21206)) | ChronoBERT (149M) et ChronoGPT (1,5B), 26 versions annuelles 1999–2024. **Tests de fuite** : 0/76 événements futurs devinés. GLUE 85,5 (≈ Llama 8B). Long-short décile à J+1 sur l'univers CRSP : Sharpe ≈ 4,9, comparable à Llama 3.1 8B, donc biais d'anticipation « modeste » pour cette tâche. Probablement sans frais. | ChronoBERT est léger, idéal pour un classifieur PIT local : [HF manelalab](https://huggingface.co/manelalab). [Package de réplication](https://data.mendeley.com/datasets/s6yt64mrwb/1) avec données synthétiques au même format. |
| **Ong et al. — FIN-FORCE** ([arXiv 2505.19430](https://arxiv.org/pdf/2505.19430), EMNLP 2025) | Benchmark : 1 368 titres de news → un scénario de risque et un d'opportunité générés par LLM, notés par GPT-4o (plausibilité, directionnalité). Claude 3.5 Haiku en tête ; un Llama 8B entraîné sur ses propres sorties (DPO) bat les grands modèles. **Aucun lien avec le marché** : les news sont choisies par des annotateurs, pas par la réaction des prix. | Génération de scénarios de risque crédibles : faisable. Valeur prédictive : non testée. |
| **Soleimani — *LLM-Generated Counterfactual Stress Scenarios*** ([arXiv 2512.07867](https://arxiv.org/html/2512.07867)) | Scénarios macro G7 générés par LLM → VaR de portefeuille ≈ 4× la VaR historique. **La sévérité vient des prompts, pas du modèle** ; RAG et news n'ont quasiment aucun effet. Aucune confrontation à la réalité. | Le LLM écrit le récit, mais ne calibre pas l'ampleur. |
| **Serruya & Katz — GRID** ([arXiv 2610.38838](https://arxiv.org/html/2610.38838v1)) | GFlowNet qui génère des trajectoires de crise sur 12 mois (Nasdaq-100, VIX, taux long, spread HY…), avec une probabilité proportionnelle à la baisse prédite par un oracle entraîné avant la crise. Plus sévère et plus varié que REINFORCE, SAC ou SMCMC. Réalisme modeste (corrélation 0,25–0,42 avec les vraies crises). | Variables proches de la stratégie. Idée : stress-tester la règle x1.7 sur des milliers de crises plausibles ; tirer en proportion plutôt que maximiser. |
| **Kumar & Nalbaria — *Bridging the Data Gap in Financial Sentiment*** ([ACL SRW 2025](https://aclanthology.org/2025.acl-srw.98.pdf)) | Réécriture du Financial PhraseBank par Gemini Flash + RAG sur des titres Yahoo 2021–2022. Phrases plus « modernes » et sentiment conservé (98,8 %). **Il manque le test qui compte** (entraînement puis test sur de vraies données récentes), et le juge est circulaire. | Idées : ancrer la génération sur des news datées ; équilibrer les classes par la génération. |
| **Zhao — *Synthetic Data and Tail-Risk Pricing*** ([SSRN 7052037](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7052037), Oxford) | Résumé seul lu (SSRN bloque l'accès). Réentraîner génération après génération sur des données synthétiques **fait sous-estimer les événements rares** : spreads trop serrés, boucle en période calme, seule une crise réelle casse le cycle. | **La mise en garde clé** : ne jamais recycler du synthétique, forcer les queues, garder un ancrage réel. |
| **S. Zhao et al. — *Diachronic Sample Integration*** ([arXiv 2607.10810](https://arxiv.org/abs/2607.10810)) | Mélanger les échantillons de plusieurs checkpoints d'entraînement pour stabiliser l'estimation des queues. | Remède technique au point précédent. |
| Non lus : *Scenario Augmentation for Rare Financial Events via Multimodal Latent Diffusion* ([arXiv 2611.07282](https://arxiv.org/abs/2611.07282)), *TAIL-GAN* (Cont et al.), Shumway 1997 (biais de sortie de cote dans CRSP) | — | 2611.07282 : augmentation d'événements rares pour un XGBoost (AUC 0,95–0,96). À lire en premier si on poursuit. |

---

## 3. Leçons méthodologiques pour un filtre de news « crise » sur QQQ

1. **Modèle point-in-time obligatoire.** Un LLM moderne « connaît » 2008, 2020 et 2022. Il faut utiliser des checkpoints datés (ChronoBERT/GPT, PIT-4B) et tester après le cutoff, avec une zone tampon (cf. Kirtac).
2. **Étiquette = premier prix exécutable.** Rendement de l'ouverture J+1 à J+5, cohérent avec `exec_lag=1`. Ne jamais inclure un rendement que la news peut décrire.
3. **Masquer les mouvements de prix dans le texte** (« shares plunged 9% » devient `[MOVE]`), par regex ou réécriture LLM, ou exclure les comptes rendus de marché. Le modèle doit apprendre **les causes**, pas lire le prix.
4. **Test du détecteur de fuite** : entraîner TF-IDF + régression logistique à prédire, depuis le texte, le rendement **déjà réalisé** à la publication.
   - Une AUC d'environ 0,5 signifie pas de fuite ; au-dessus de 0,65, le masquage est insuffisant.
   - Les plus gros coefficients montrent les mots qui fuient encore.
   - Contre-test : une performance du vrai modèle qui s'effondre après masquage voulait dire que le signal venait de la fuite.
5. **Corpus synthétique** : graines réelles choisies par clustering, générateur point-in-time de préférence. Ne jamais regénérer à partir de données synthétiques (Zhao). Forcer les scénarios extrêmes et inédits, sinon le LLM produit des « 2008 bis ».
6. **Validation avec ≈4 crises** : placebos (dates mélangées), tests de réfutation, Sharpe dégonflé et SPA. Le risque de surapprentissage est élevé. Commencer par **embedding figé + régression régularisée** plutôt que par du RL.
7. **Attentes réalistes** : le sentiment « général » prédit mal le lendemain (Bisharat & Hean), et les réplications sur données publiques sont bien plus faibles (Kirtac : Sharpe 0,92 au lieu de 2,85). Le texte n'a de valeur probable que sur les événements extrêmes.

---

## 4. Session du 2026-11-10 — trading piloté par LLM avec contexte macro glissant

**Question** : existe-t-il des travaux où un LLM gère l'allocation en tenant à jour un **contexte macro glissant** (inflation, taux directeurs, calendrier économique, calendrier des résultats, news, technique), avec : thèmes de news récurrents, thèmes dynamiques générés par le contexte, fiabilité des acteurs, échéances à venir, puis niveau d'allocation ?

**Réponse courte** : non, pas de bout en bout. Chaque brique existe séparément ; **le calendrier des échéances (et leurs surprises) et la fiabilité des acteurs jugée sur leurs annonces passées ne sont couverts par aucun papier trouvé**.

> ⚠️ **Niveau de lecture** : PDF téléchargés, mais seuls les **résumés** (et la première page des deux papiers CLEF) ont été lus. Les notes ci-dessous ne valent pas les fiches détaillées de la section 1 ; les chiffres sont ceux annoncés par les auteurs, non vérifiés.

### 4.1 PDF ajoutés

| Fichier | Papier | Brique couverte | Intérêt a priori |
|---|---|---|---|
| `Empirical-Asset-Pricing-with-Large-Language-Model-Agents.pdf` | Cheng & Chin, [arXiv 2409.17266](https://arxiv.org/abs/2409.17266), 2024 | **Contexte glissant** : l'agent écrit des rapports d'analyse à partir des news, avec mémoire des rapports passés et une note macro/micro réécrite au fil de l'eau ; l'embedding des rapports alimente un modèle de pricing. Sharpe +10,6 % par rapport aux références ML. | ⭐⭐⭐ Le mécanisme de note mise à jour est le plus proche de l'étape 1. |
| `FinMem_-….pdf` | Yu et al., [arXiv 2311.13743](https://arxiv.org/abs/2311.13743), 2023 | **Mémoire en couches** (court / moyen / long terme) + profil de l'agent + décision. Actions individuelles. | ⭐⭐ Référence classique pour l'architecture de mémoire. |
| `EdgeQuant-Agent_-….pdf` | Kava et al., CLEF 2026 FinMMEval tâche 3 | Mémoire à 4 niveaux (news, 10-Q, 10-K, couche réflexive), ChromaDB. Décision quotidienne BUY/HOLD/SELL sur TSLA et BTC. 2ᵉ sur 18 en évaluation **live** sur TSLA, 8ᵉ sur BTC. | ⭐⭐ Mémoire hiérarchique testée en réel, mais fenêtre de 6 à 8 semaines. |
| `TrustMe-CLEF-2026-….pdf` | Kundu, Kaneshiro, Imai, CLEF 2026 FinMMEval tâche 3 | Agent LLM qui **suit la tendance par défaut** et ne s'écarte que sur preuve claire. Classe les news par narratif dominant et par horizon (catalyseur d'un jour ou thème de plusieurs jours). **Gros catalogue de résultats négatifs.** | ⭐⭐⭐ Constat clé : les news fournies décrivent des événements **déjà dans le cours de clôture**. Conclusion voisine de la nôtre (socle trend-following, LLM en garde-fou). |
| `Leveraging-LLMS-for-Top-Down-Sector-Allocation-….pdf` | Quek et al. (groupe Cambria), [arXiv 2503.09647](https://arxiv.org/abs/2503.09647), 2025 | **Allocation top-down** : documents de politique monétaire + indicateurs économiques + sentiment → poids sectoriels. Sharpe annoncé 2,51 contre −0,61 pour le momentum croisé. | ⭐⭐ Bon gabarit d'entrées macro ; rendement de 8,79 % seulement, période sans doute courte et contaminée. |
| `Hierarchical-AI-Multi-Agent-Fundamental-Investing_-….pdf` | He et al., [arXiv 2510.21147](https://arxiv.org/abs/2510.21147), 2025 | **Architecture complète** : agent macro (filtre les secteurs) → 4 agents par entreprise (fondamental, technique, rapports, news) → agent portefeuille (RL) → agent contrôle du risque. CSI 300. | ⭐⭐ Le découpage en agents le plus proche de la liste de thèmes. |
| `CN-Buzz2Portfolio_-….pdf` | Chen et al., [arXiv 2603.22305](https://arxiv.org/abs/2603.22305), 2026 | **News du jour → poids d'ETF** macro et sectoriels (pas d'actions). Chaîne en 3 étapes : compression, perception, allocation. 9 LLM comparés, 2024 → mi-2025. Données et code publiés. | ⭐⭐⭐ Le cadre le plus proche d'une allocation sur ETF à partir du flux de news. |
| `Enhancing-Regime-Shift-Detection-….pdf` | Yi et al., [arXiv 2605.30363](https://arxiv.org/abs/2605.30363), 2026 | **Changement de régime** : le LLM propose des dates à partir des minutes du FOMC, un test VAR les valide sur 14 séries taux/macro (et inversement). F1 = 0,82, détection le jour même. Pas d'allocation. | ⭐⭐⭐ Validation croisée texte ↔ données : bon modèle pour ne pas laisser le LLM décider seul. |
| `TrustTrade_-….pdf` | Li et al., [arXiv 2603.22567](https://arxiv.org/abs/2603.22567), 2026 | **Fiabilité de l'information** : plusieurs agents LLM indépendants, signaux pondérés par leur accord sémantique et numérique ; signaux temporels déterministes comme ancrage ; mémoire réflexive. | ⭐⭐ Fiabilité par consensus entre agents, pas par historique de l'acteur. |
| `The-simulation-of-judgment-in-LLMs.pdf` | Loru et al. (Sapienza), [arXiv 2502.04426](https://arxiv.org/abs/2502.04426), 2025 (v1 intitulée *Decoding AI Judgment*) | 6 LLM notent la fiabilité de sources de presse, comparés à NewsGuard et MBFC. Sorties alignées, mais critères différents : asymétrie politique, confusion entre forme du texte et fiabilité (« epistemia »). | ⭐ Mise en garde pour l'étape « fiabilité des acteurs ». |
| `Look-Ahead-Bench_-….pdf` | Benhenda, [arXiv 2601.13770](https://arxiv.org/abs/2601.13770), 2026 | **Biais d'anticipation** mesuré dans des flux de trading réalistes : l'alpha de Llama 3.1 et DeepSeek s'effondre après leur cutoff, pas celui des modèles point-in-time (Pitinf). [Code](https://github.com/benstaf/lookaheadbench). | ⭐⭐⭐ À lire avant tout backtest d'une couche LLM. |
| `Summoning-the-Oracle-to-Slay-It_-….pdf` | Li, Wang, Ma, [arXiv 2605.24564](https://arxiv.org/abs/2605.24564), 2026 | FinCAD : décodage qui atténue la mémoire des dates connues, sans réentraînement. Rendements in-sample corrigés jusqu'à −67 %, hors échantillon 2025 quasi inchangé. | ⭐⭐ Chiffre l'ampleur de la contamination. |
| `Can-LLM-based-Financial-Investing-Strategies-Outperform-….pdf` | Li, Kim, Cucuringu, Ma (FINSABER), [arXiv 2505.07078](https://arxiv.org/abs/2505.07078), 2025 | Backtests sur 20 ans et plus de 100 titres : l'avantage des stratégies LLM disparaît. Elles sont **trop prudentes en marché haussier et trop agressives en marché baissier**. Conclusion des auteurs : privilégier détection de tendance et contrôle du risque par régime. | ⭐⭐⭐ La douche froide ; conforte le socle déterministe actuel. |

**Ordre de lecture conseillé** : FINSABER, Look-Ahead-Bench, TrustMe (résultats négatifs), puis Cheng & Chin, CN-Buzz2Portfolio, Regime Shift Detection.

### 4.2 Vus en ligne, sans PDF local

| Papier | Résumé | Intérêt |
|---|---|---|
| **Taibi, Gross-Klussmann, Osterrieder — *Trading Macro Narratives*** (avril 2026, [lien d'origine](https://digital-ai-finance.github.io/macro_narratives/paper_signal_tilt_core_revised.pdf), **404 au 2026-11-10**) | 65 narratifs macro « evergreen » suivis sur 2004–2025, transformés en signaux pour des portefeuilles actions et facteurs. Rendements positifs annoncés ; rotation, frais et mise en œuvre laissés ouverts. | Le plus proche des **thèmes récurrents** (inflation, douane, guerre…). À retrouver (SSRN ou arXiv). |
| DatedGPT (Yan et al., 2026) | Modèles entraînés sur données filtrées par date. | Alternative à ChronoGPT / PIT-4B (section 2). |
| Bybee, Kelly, Manela, Xiu — *Business News and Business Cycles* (cité de mémoire, non revérifié) | Thèmes extraits du WSJ par modèle de sujets, reliés au cycle économique. | Ancêtre sans LLM du suivi de thèmes récurrents. |

### 4.3 Correspondance avec le pipeline envisagé

| Étape | Couverture | Sources |
|---|---|---|
| 1. Contexte macro glissant | Partielle | Cheng & Chin, FinMem, EdgeQuant |
| 2. Thèmes récurrents | Bonne | Trading Macro Narratives, Bybee et al. |
| 3. Thèmes dynamiques générés par le contexte | **Faible** : classification, pas découverte | TrustMe |
| 4. Fiabilité des acteurs | **Faible** : consensus entre agents ou notation de médias, rien sur l'historique d'un acteur | TrustTrade, Loru et al. |
| 5. Échéances à venir et surprises | **Aucune** | (Regime Shift Detection utilise les minutes du FOMC, sans calendrier) |
| 6. Niveau d'allocation | Bonne, mais sectorielle ou par ETF | Quek et al., He et al., CN-Buzz2Portfolio |

### 4.4 Conséquences pour ce dépôt

1. **Aucun backtest 2000–2026 d'une couche LLM n'est fiable** : le modèle connaît 2008, 2020 et 2022 (Look-Ahead-Bench, FinCAD). Seul le suivi en temps réel après le cutoff valide quelque chose, ou un modèle point-in-time (section 2).
2. **Garder la formule déterministe comme socle** et faire tourner la couche sémantique en **observation seule** : journal quotidien du contexte, des thèmes, des échéances et de l'allocation suggérée, sans effet sur les ordres. FINSABER et TrustMe arrivent tous deux à « tendance d'abord, LLM en garde-fou ».
3. **Les news sont souvent déjà dans le cours** (TrustMe) : ce qui peut porter de l'information en avance, ce sont les **échéances connues à l'avance** et l'écart au consensus, soit précisément la brique que personne n'a publiée.
4. **Ne pas laisser le LLM décider seul** : valider chaque signal textuel par les données (schéma texte ↔ VAR de Yi et al.), et ancrer sur des signaux déterministes (TrustTrade).

---

## 5. Données (pour reproduire les papiers sur actions individuelles)

- **CRSP** : payant, accessible via WRDS (abonnement institutionnel ou universitaire). Pas d'offre particulier raisonnable.
- **Gratuit, partiel** :
  - Kenneth French Data Library et facteurs JKP : portefeuilles agrégés seulement.
  - Quandl WIKI : figé en 2018.
  - QuantConnect : données sans biais du survivant, mais seulement dans leur cloud.
  - yfinance : biais du survivant.
- **Moins de 100 $** :
  - **MarketParquet**, 79 $ en paiement unique : environ 16 000 tickers dont environ 8 500 radiés, 2000 → aujourd'hui, Parquet. Ajusté des splits seulement, pas des dividendes ; source inconnue. Vérifier d'abord l'échantillon gratuit (Lehman, Enron…).
  - **EODHD**, environ 20 $ pour un mois : sociétés radiées complètes après 2018, prix quotidiens seulement avant. Vérifier la licence de conservation des données.
- **Au-dessus** : Sharadar (depuis 1998, + fondamentaux), Norgate Platinum ou Diamond (environ 630 à 790 $/an, avec la composition historique des indices).
- Pour la stratégie QQQ actuelle, aucune de ces sources n'est nécessaire.
