#!/bin/bash
# Setup cron jobs for MyQTM-ETF robot
# Run: bash setup_cron.sh
#
# Jobs (Europe/Paris timezone):
#   16:30 lun-ven  --data    refresh OHLCV+VIX+iShares+features (~1h avant cloture XETRA)
#   17:15 lun-ven  --trade   execute trades (15 min avant cloture XETRA 17:30)
#   20:00 dimanche --train   retrain if end of 21-day block

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv/bin/activate"
LOG="$PROJECT_DIR/data/robot/cron.log"
PYTHON="$PROJECT_DIR/venv/bin/python"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/data/robot"

CRON_MARKER="# MyQTM-ETF Robot"

# Build cron entries
CRON_BLOCK="$CRON_MARKER — do not edit manually
CRON_TZ=Europe/Paris
30 16 * * 1-5 cd $PROJECT_DIR && source $VENV && $PYTHON robot.py --data >> $LOG 2>&1
15 17 * * 1-5 cd $PROJECT_DIR && source $VENV && $PYTHON robot.py --trade >> $LOG 2>&1
0 20 * * 0 cd $PROJECT_DIR && source $VENV && $PYTHON robot.py --train >> $LOG 2>&1
$CRON_MARKER — end"

echo "The following cron jobs will be installed:"
echo ""
echo "$CRON_BLOCK"
echo ""
echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON"
echo "Log:     $LOG"
echo ""

read -p "Install these cron jobs? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

# Remove old MyQTM entries if any, then append
(crontab -l 2>/dev/null | sed "/$CRON_MARKER/,/$CRON_MARKER/d"; echo "$CRON_BLOCK") | crontab -

echo "Cron jobs installed. Verify with: crontab -l"
