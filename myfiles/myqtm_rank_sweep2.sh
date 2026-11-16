#!/bin/bash
# Point 2 seul : classement par jour sur l'étiquette de tendance.
cd "$(dirname "$0")/.."
M=src.portefolio_strategy.myqtm.train_rank
python -m $M --first=2024-01-01 --trend --tag=trend3
python -m $M --first=2024-01-01 --trend --metric=ndcg@10 --patience=100 --tag=trend10
echo TERMINÉ
