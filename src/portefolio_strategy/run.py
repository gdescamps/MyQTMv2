"""Point d'entrée du portage de ref/MyQTM sur data/eodhd.

Étapes (chacune relit ce que la précédente a écrit dans data/portefolio_strategy) :
  universe   liste Nasdaq dynamique point-in-time
  news       extraction des news par (titre, jour) depuis 2020-01-01
  estimate   coût Haiku + temps GPU du traitement complet → outputs/portefolio_strategy/ESTIMATES.md
  digest     condensés Haiku          (payant : jamais lancé par défaut, voir ci-dessous)
  sentiment  notes du LLM PIT (GPU)   (idem)
  features   features par (titre, jour), avec le sentiment déjà noté
  train      XGBoost walk-forward → prédictions hors échantillon
  benchmark  seuils walk-forward (CMA-ES) + backtest → outputs/portefolio_strategy

    python -m src.portefolio_strategy.run                       # universe news estimate features train benchmark
    python -m src.portefolio_strategy.run features train benchmark
    # traitement complet du corpus, après lecture de ESTIMATES.md :
    python -m src.portefolio_strategy.digest --full && python -m src.portefolio_strategy.sentiment
    python -m src.portefolio_strategy.run features train benchmark
"""
import sys
import time

from . import benchmark, estimate, features, news, train, universe

STAGES = {"universe": universe.main, "news": news.main, "estimate": estimate.main, "features": features.main,
          "train": train.main, "benchmark": benchmark.main}
DEFAULT = ["universe", "news", "estimate", "features", "train", "benchmark"]


def main(argv=None):
    stages = (argv if argv is not None else sys.argv[1:]) or DEFAULT
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"étapes inconnues : {bad} (choix : {list(STAGES)} ; digest et sentiment se lancent à part)")
    for s in stages:
        t0 = time.time()
        print(f"=== {s} ===", flush=True)
        STAGES[s]()
        print(f"=== {s} : {time.time() - t0:.0f} s ===", flush=True)


if __name__ == "__main__":
    main()
