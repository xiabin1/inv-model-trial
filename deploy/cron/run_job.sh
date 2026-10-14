#!/bin/bash
# Run one scheduled job in the project environment and alert on failure (cron):
#   run_job.sh NAME LOG 'COMMAND'
# COMMAND runs with .env loaded and the virtualenv active; its output is appended to LOG. If it exits
# with an error, a Telegram alert with the last log lines is sent (at most one per job per hour, so a
# job that keeps failing does not flood the chat). The exit code is passed on.
NAME=$1
LOG=$2
shift 2
cd /home/ec2-user/project/portfolio-calc-data || exit 1
set -a
[[ -f .env ]] && source .env
set +a
source .venv/bin/activate
mkdir -p logs data/state/alerts
bash -c "$*" >> "$LOG" 2>&1
CODE=$?
if [ "$CODE" -ne 0 ]; then
    STAMP="data/state/alerts/${NAME// /_}"
    if [ ! -f "$STAMP" ] || [ $(( $(date +%s) - $(stat -c %Y "$STAMP") )) -ge 3600 ]; then
        touch "$STAMP"
        {
            echo "❌ Job failed: $NAME (exit $CODE), $(TZ=America/New_York date '+%a %m-%d %H:%M ET')"
            echo "log: $LOG — last lines:"
            grep -v -i 'warning' "$LOG" | tail -12
        } | python -m src.notify.telegram alert >> logs/alerts.log 2>&1
    fi
fi
exit "$CODE"
