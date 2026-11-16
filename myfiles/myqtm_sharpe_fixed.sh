#!/bin/bash
# Arrêt anticipé au Sharpe des seuils par défaut (sans CMA-ES), 3 graines, ts0, réglages XGBoost par défaut ;
# évaluation hors échantillon avec les seuils par défaut (graine 42 et moyenne des graines).
cd "$(dirname "$0")/.."
for s in 42 7 123; do
  python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --sample-seed=$s --first=2024-01-01 --stop=sharpe-fixed --save-val 2>&1 | grep -E "Sharpe|prédictions|Error|Traceback" | cut -c1-200
  echo "graine $s ok"
done
for a in "" "--avg"; do
  python -m src.portefolio_strategy.myqtm.cma_ab --pred=ts0 --sfx=_ssf --no-cma $a 2>&1 | grep -v -i "warn"
done
echo TERMINÉ
