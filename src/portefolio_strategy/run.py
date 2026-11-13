"""Point d'entrée de portefolio_strategy : sélection longue dans l'univers Nasdaq dynamique (data/eodhd).

Étapes (chacune relit ce que la précédente a écrit dans data/portefolio_strategy) :
  universe   liste Nasdaq dynamique point-in-time
  news       extraction des news par (titre, jour) depuis 2020-01-01
  estimate   coût Haiku + temps GPU du traitement complet → outputs/portefolio_strategy/ESTIMATES.md
  features   features par (titre, jour), avec les traitements LLM déjà faits (llm/pit, llm/qwen25)
  train      modèle de classement walk-forward → scores hors échantillon (variante pit)
  benchmark  grille fréquence × nombre de titres, choix walk-forward, backtest → outputs/portefolio_strategy/pit
  compare    mêmes étapes pour chaque variante de features + tableau comparatif

Les traitements LLM se lancent à part (payants ou longs) : digest (Haiku), sentiment (PIT), qwen.

    python -m src.portefolio_strategy.run                       # universe news estimate features train benchmark
    python -m src.portefolio_strategy.run features compare
"""
import sys
import time

from . import benchmark, compare, estimate, features, news, train, universe

STAGES = {"universe": universe.main, "news": news.main, "estimate": estimate.main, "features": features.main,
          "train": train.main, "benchmark": benchmark.main, "compare": lambda: compare.main([])}
DEFAULT = ["universe", "news", "estimate", "features", "train", "benchmark"]


def main(argv=None):
    stages = (argv if argv is not None else sys.argv[1:]) or DEFAULT
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"étapes inconnues : {bad} (choix : {list(STAGES)} ; digest, sentiment et qwen se lancent à part)")
    for s in stages:
        t0 = time.time()
        print(f"=== {s} ===", flush=True)
        STAGES[s]()
        print(f"=== {s} : {time.time() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
