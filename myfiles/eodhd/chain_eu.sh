#!/bin/bash
# Enchaîne le téléchargement Europe après la fin du téléchargement US (PID passé en argument).
# Ne démarre que si le journal US se termine par « TERMINÉ » (sinon le US a planté : on ne fait rien).
cd /home/greg/data_local/code/MyQTMv2_dev || exit 1
US_PID=$1
LOG=myfiles/eodhd/download_eu.log
echo "$(date '+%F %T') attente de la fin du téléchargement US (PID $US_PID)" >> "$LOG"
tail --pid="$US_PID" -f /dev/null
if tail -1 myfiles/eodhd/download.log | grep -q 'TERMINÉ'; then
    echo "$(date '+%F %T') téléchargement US terminé : démarrage de la région eu" >> "$LOG"
    exec ./venv/bin/python -W ignore -m myfiles.eodhd.download_eodhd --region eu --wait >> "$LOG" 2>&1
else
    echo "$(date '+%F %T') le téléchargement US s'est arrêté sans « TERMINÉ » : région eu NON lancée" >> "$LOG"
fi
