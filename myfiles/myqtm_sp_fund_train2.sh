#!/bin/bash
# Suite : horizon moyen (le seul prometteur sur la graine 42) pour les graines 7 et 123, puis classement à horizon fixe.
cd "$(dirname "$0")/.."
for s in 7 123; do
  t0=$(date +%s)
  python -W ignore -m src.portefolio_strategy.myqtm.train all --sp --tail-sample=0 --sample-seed=$s --first=2024-01-01 --trend=mid > myfiles/.sp_fund_run.tmp 2>&1
  grep -E "prédictions|Error|Traceback" myfiles/.sp_fund_run.tmp | cut -c1-230
  echo "graine $s mid ok : $(( $(date +%s) - t0 )) s"
done
rm -f myfiles/.sp_fund_run.tmp
python -W ignore -m src.portefolio_strategy.myqtm.fwd_rank 2>&1 | cut -c1-400
echo FINI
