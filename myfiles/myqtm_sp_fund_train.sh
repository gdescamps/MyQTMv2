#!/bin/bash
# S&P 500, 200 features (fondamentaux + groupe GICS), 3 horizons de tendance × 3 graines, séquentiel.
# Ordre : graine 42 pour les 3 horizons d'abord (lecture rapide), puis graines 7 et 123.
cd "$(dirname "$0")/.."
for s in 42 7 123; do for t in "" "--trend=mid" "--trend=long"; do
  t0=$(date +%s)
  python -W ignore -m src.portefolio_strategy.myqtm.train all --sp --tail-sample=0 --sample-seed=$s --first=2024-01-01 $t > myfiles/.sp_fund_run.tmp 2>&1
  grep -E "prédictions|Error|Traceback" myfiles/.sp_fund_run.tmp | cut -c1-230
  echo "graine $s ${t:-court} ok : $(( $(date +%s) - t0 )) s"
done; done
rm -f myfiles/.sp_fund_run.tmp
echo TERMINÉ
