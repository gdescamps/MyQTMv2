#!/bin/bash
# Petits réglages du modèle de classement (≈ 1 min chacun), séquentiels.
cd "$(dirname "$0")/.."
M=src.portefolio_strategy.myqtm.train_rank
python -m $M --first=2024-01-01 --metric=ndcg@10 --patience=100 --tag=n10
python -m $M --first=2024-01-01 --metric=ndcg@10 --patience=100 --risk-adj --tag=n10ra
python -m $M --first=2024-01-01 --objective=rank:pairwise --metric=ndcg@10 --patience=100 --tag=pw10
python -m $M --first=2024-01-01 --objective=rank:pairwise --metric=ndcg@10 --patience=100 --risk-adj --tag=pw10ra
echo TERMINÉ
