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

## 4. Données (pour reproduire les papiers sur actions individuelles)

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
