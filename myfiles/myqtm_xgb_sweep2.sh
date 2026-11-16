#!/bin/bash
# 1) référence 4/100/0,02/100 sur 3 graines avec --save-val (scores croisés pour le second modèle)
# 2) balayage autour de la référence (graine 42), séquentiel, chronométré.
cd "$(dirname "$0")/.."
T="python -m src.portefolio_strategy.myqtm.train all --tail-sample=0 --first=2024-01-01 --save-val"
for s in 42 7 123; do
  t0=$(date +%s); $T --xgb=4,100,0.02,100 --sample-seed=$s | grep -v "^  {"; echo "référence graine $s : $(( $(date +%s) - t0 )) s"
done
echo RÉFÉRENCE OK
for x in 3,100,0.02,100 4,300,0.02,100 3,300,0.02,100 4,200,0.02,100 4,100,0.01,150; do
  t0=$(date +%s); $T --xgb=$x | grep -v "^  {"; echo "variante $x : $(( $(date +%s) - t0 )) s"
done
echo TERMINÉ
