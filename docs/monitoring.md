# Monitoring and alerts (since 2026-10-13)

Problems that block the pipeline are sent to Telegram; they are fixed by hand.

## Job failures

Every scheduled job (deploy/cron/crontab.txt) runs through `deploy/cron/run_job.sh NAME LOG 'COMMAND'`:
the project environment is set up, the output is appended to LOG, and if the job exits with an error a
Telegram message is sent: `❌ Job failed: NAME (exit N), time` with the log path and its last lines. At most
one alert per job per hour (`data/state/alerts/`), so a job that keeps failing does not flood the chat.
Not wrapped: the every-minute Telegram tap job (`corporate_action_flow ack`) and the gateway watchdog.

Failures that are caught this way include: a nightly data job (download, features, labels), the 15:56
predictions (e.g. no daily features, the live stream too thin), **the traded model producing no picks**
(a shadow such as mini_l250 failing inside mini's run now stops the job with an error instead of only
logging "not trading"), IBKR sell / reconcile, paper settle (all ledgers still run; the job fails if any
did), corporate-action checks, retrains and the backup. In the nightly chain, mini_l250 only runs if
b1542_l250 (which builds its lists, daily features and labels) succeeded.

## IB Gateway

| Check | When (ET) | What |
|---|---|---|
| Watchdog `deploy/ibkr/start_gateway.sh` | every 5 min + boot | starts the gateway if its process is gone |
| Health check `deploy/ibkr/check_gateway.py` | trading days 06:00, 08:40, every 30 min 09:00-15:30, 15:45 | connects to the API (port 4002) and checks the strategy account is visible; if not, restarts the gateway and tests again |

Telegram: `⚠️ ... restarted automatically and connected again`, `❌ IBKR gateway DOWN (automatic restart did
not help)` (the next sells/buys will fail: log in by hand, check ~/ibc/logs and ~/ibc/config.ini),
`✅ reachable again`, `⏸ paused` (~/ibkr/PAUSED exists). Log: `logs/gateway_check.log`.

Why: on 2026-10-12 the gateway stayed logged out after IBKR's weekly re-login (Sunday 23:45 restart:
"full authentication will be required"); the process was running, so the watchdog did not notice, and
the 09:50 reconcile failed.

Manual alert: `echo "title\nbody" | python -m src.notify.telegram alert`.
