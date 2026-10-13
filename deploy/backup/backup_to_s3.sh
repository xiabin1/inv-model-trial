#!/bin/bash
# Daily backup of everything that lives only on this server to S3 (cron, 20:30 ET).
#   code/latest/          mirror of the code, configs, docs, tests, scripts, deploy files
#   code/snapshots/*.zip  one dated zip per day (history)
#   models/ logs/ data/paper/ data/state/   copied; older files are kept
#   crontab.txt           the active schedule
# Not copied (secrets): .env, ~/ibc/config.ini. Not copied (rebuildable): .venv/.
# Restore: docs/restore_from_backup.md
set -euo pipefail

PROJECT=/home/ec2-user/project/portfolio-calc-data
DEST=s3://portfolio-calc-platform-dev/financial/tp1/backup
DAY=$(TZ=America/New_York date +%Y-%m-%d)
cd "$PROJECT"

CODE_EXCLUDES=(--exclude ".venv/*" --exclude ".env" --exclude ".pytest_cache/*" --exclude "*__pycache__*"
               --exclude "logs/*" --exclude "models/*" --exclude "data/*" --exclude "src/src.zip")

crontab -l > deploy/cron/crontab.active.txt 2>/dev/null || true

aws s3 sync . "$DEST/code/latest/" "${CODE_EXCLUDES[@]}" --delete --only-show-errors
SNAPSHOT=$(mktemp --suffix=.zip)
rm -f "$SNAPSHOT"
zip -rq "$SNAPSHOT" . -x ".venv/*" ".env" ".pytest_cache/*" "*__pycache__*" "logs/*" "models/*" "data/*" "src/src.zip"
aws s3 cp "$SNAPSHOT" "$DEST/code/snapshots/$DAY.zip" --only-show-errors
rm -f "$SNAPSHOT"

aws s3 sync models/ "$DEST/models/" --only-show-errors
aws s3 sync logs/ "$DEST/logs/" --only-show-errors
for folder in paper state; do
    [ -d "data/$folder" ] && aws s3 sync "data/$folder/" "$DEST/data/$folder/" --only-show-errors
done
aws s3 cp deploy/cron/crontab.active.txt "$DEST/crontab.txt" --only-show-errors

echo "$(date '+%F %T') backup done -> $DEST (snapshot $DAY.zip)"
