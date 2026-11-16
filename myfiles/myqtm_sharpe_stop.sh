#!/bin/bash
# Arrêt anticipé au Sharpe (3 graines, ts0, réglages par défaut), puis seuils CMA-ES sur la ligne croisée de chaque
# coupe testés sur le trimestre suivant : arrêt F1 (référence) et arrêt Sharpe, graine 42 et moyenne des graines.
cd "$(dirname "$0")/.."
for s in 42 7 123; do
  python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --sample-seed=$s --first=2024-01-01 --stop=sharpe --save-val 2>&1 | grep -E "Sharpe|prédictions|Error|Traceback" | cut -c1-200
  echo "graine $s ok"
done
for sfx in "" "_ss"; do for a in "" "--avg"; do
  python -m src.portefolio_strategy.myqtm.cma_ab --pred=ts0 --sfx=$sfx $a 2>&1 | grep -v -i "warn"
done; done
echo TERMINÉ
