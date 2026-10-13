# Restoring on a new server

Everything that exists only on the server is copied daily (20:30 ET) by
`deploy/backup/backup_to_s3.sh` to `s3://portfolio-calc-platform-dev/financial/tp1/backup/`.
The market data, features, labels and predictions already live in S3 (`financial/tp1/data/`).

| S3 path | Contents |
|---|---|
| `backup/code/latest/` | current code, configs, docs, tests, scripts, deploy files |
| `backup/code/snapshots/YYYY-MM-DD.zip` | the code as of each day |
| `backup/models/` | live models (`*.joblib` + `.json` training summary), `history/` (every live model version, see `src/models/model_registry.py`) and archives |
| `backup/logs/` | job logs |
| `backup/data/paper/`, `backup/data/state/` | paper ledgers, IBKR trade log, corporate-action workflow state |
| `backup/crontab.txt` | the schedule that was active |

**Not in the backup (secrets, keep them yourself):**
- `.env`: `DATABENTO_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SEC_USER_AGENT` (values in quotes if they contain spaces)
- `~/ibc/config.ini`: IBKR paper username and password (IbLoginId / IbPassword)

## Steps

```bash
# 1. code, models, logs, ledgers, state
mkdir -p ~/project/portfolio-calc-data && cd ~/project/portfolio-calc-data
B=s3://portfolio-calc-platform-dev/financial/tp1/backup
aws s3 sync $B/code/latest/ .
aws s3 sync $B/models/ models/
aws s3 sync $B/logs/ logs/
aws s3 sync $B/data/paper/ data/paper/
aws s3 sync $B/data/state/ data/state/

# 2. Python environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt ib_async

# 3. secrets: recreate .env (see above)

# 4. IB Gateway (optional, for IBKR): see deploy/ibkr/ (Xvfb, libXi, libXtst, IB Gateway, IBC,
#    ~/ibc/config.ini with the paper login), then deploy/ibkr/start_gateway.sh

# 5. schedule
crontab deploy/cron/crontab.txt

# 6. check
.venv/bin/python -m pytest -q
```

The server's IAM role (or credentials) must allow S3 access to the bucket.
