#!/bin/bash
# Traitement complet des news structurées (events.py), détaché de la session Claude Code ; reprise automatique.
cd /home/greg/data_local/code/MyQTMv2_dev || exit 1
export PYTHONUNBUFFERED=1
[ -f data/portefolio_strategy/llm/llama31/texts.parquet ] || venv/bin/python -m src.portefolio_strategy.events prepare || exit 1
venv_vllm/bin/python src/portefolio_strategy/events_vllm.py || exit 1
venv/bin/python -m src.portefolio_strategy.events parse
echo "TERMINÉ $(date)"
