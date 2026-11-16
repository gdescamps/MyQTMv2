#!/bin/bash
# Classifieur ts0, réglages XGBoost par défaut, 3 graines, avec les lignes croisées A↔B (--save-val).
cd "$(dirname "$0")/.."
for s in 42 7 123; do
  python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --sample-seed=$s --first=2024-01-01 --save-val 2>&1 | grep -E "prédictions|Error|Traceback"
  echo "graine $s ok"
done
echo TERMINÉ
