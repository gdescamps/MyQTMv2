#!/bin/bash
# Balayage de la taille / régularisation de XGBoost (ts0, graine 42), séquentiel, chronométré.
cd "$(dirname "$0")/.."
for x in 5,50,0.05,40 3,200,0.05,40 7,5,0.02,100 4,100,0.02,100; do
  t0=$(date +%s)
  python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --xgb=$x --first=2024-01-01 | grep -v "^  {"
  echo "variante $x : $(( $(date +%s) - t0 )) s"
done
echo TERMINÉ
