#!/bin/bash
# Daily backup to S3 of what lives only on this server and is not in git (cron, 20:30 ET):
#   models/ logs/ data/paper/ data/state/   copied; older files are kept
#   crontab.txt                             the active schedule
# The code is versioned in GitHub (xiabin1/inv-model-trial), pushed by hand since 2026-10-13; the
# earlier code copies stay in S3 (code/latest/, code/snapshots/) but are no longer updated.
# Not copied (secrets): .env, ~/ibc/config.ini. Not copied (rebuildable): .venv/.
# Restore: docs/restore_from_backup.md
set -euo pipefail

PROJECT=/home/ec2-user/project/portfolio-calc-data
DEST=s3://portfolio-calc-platform-dev/financial/tp1/backup
cd "$PROJECT"

crontab -l > deploy/cron/crontab.active.txt 2>/dev/null || true

aws s3 sync models/ "$DEST/models/" --only-show-errors
aws s3 sync logs/ "$DEST/logs/" --only-show-errors
for folder in paper state; do
    [ -d "data/$folder" ] && aws s3 sync "data/$folder/" "$DEST/data/$folder/" --only-show-errors
done
aws s3 cp deploy/cron/crontab.active.txt "$DEST/crontab.txt" --only-show-errors

echo "$(date '+%F %T') backup done -> $DEST (models, logs, ledgers, state, crontab; code is in GitHub)"
