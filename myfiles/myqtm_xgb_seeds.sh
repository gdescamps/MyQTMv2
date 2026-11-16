#!/bin/bash
# Graines 7 et 123 de la variante XGBoost 4 / 100 / 0,02 / 100 (ts0).
cd "$(dirname "$0")/.."
for s in 7 123; do
  python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --xgb=4,100,0.02,100 --sample-seed=$s --first=2024-01-01 | grep -v "^  {"
  echo "graine $s ok"
done
echo TERMINÉ
