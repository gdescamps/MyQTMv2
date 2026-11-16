#!/bin/bash
# Classifieur ts0 sur le S&P 500 reconstitué (réglages par défaut, arrêt F1), 3 graines, séquentiel.
cd "$(dirname "$0")/.."
for s in 42 7 123; do
  t0=$(date +%s)
  python -W ignore -m src.portefolio_strategy.myqtm.train all --sp --tail-sample=0 --sample-seed=$s --first=2024-01-01 2>&1 | cut -c1-260
  echo "graine $s ok : $(( $(date +%s) - t0 )) s"
done
echo TERMINÉ
